from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from psycopg import connect
from psycopg.rows import dict_row

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kabot.compat import UTC
from kabot.settings import load_settings


MAX_TTE_MINUTES = 14.0
DISTANCE_THRESHOLD_DOLLARS = 6.0
MIN_MARKET_VOLUME = 1000.0
SOFT_ENTRY_CAP_CENTS = 57
EXPENSIVE_ENTRY_EDGE_PREMIUM = 0.02
GBM_MIN_EDGE_MID = 0.06
GBM_MIN_EDGE_WIDE_YES = 0.03
FEE_RATE_BPS = 70.0


@dataclass(frozen=True)
class OpenPosition:
    market_ticker: str
    strategy_name: str
    side: str
    contracts: int
    entry_time: pd.Timestamp
    entry_price_cents: int
    expiry: pd.Timestamp
    gbm_edge: float
    gbm_probability: float


def norm_cdf(values: np.ndarray) -> np.ndarray:
    erf_vec = np.frompyfunc(lambda x: math.erf(float(x) / math.sqrt(2.0)), 1, 1)
    return 0.5 * (1.0 + np.asarray(erf_vec(values), dtype=float))


def load_live_snapshots(*, config_path: str | None, series: str) -> pd.DataFrame:
    settings = load_settings(config_path)
    query = """
        SELECT
            observed_at,
            series_ticker,
            market_ticker,
            contract_type,
            expiry,
            spot_price,
            threshold,
            direction,
            yes_bid,
            yes_ask,
            no_bid,
            no_ask,
            volume
        FROM kalshi_market_snapshots
        WHERE series_ticker = %s
          AND expiry <= now()
        ORDER BY observed_at, market_ticker
    """
    with connect(settings.storage["db_dsn"], row_factory=dict_row) as conn:
        frame = pd.DataFrame(conn.execute(query, [series]).fetchall())
    if frame.empty:
        return frame
    for column in ("observed_at", "expiry"):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    return frame.dropna(
        subset=[
            "observed_at",
            "expiry",
            "market_ticker",
            "spot_price",
            "threshold",
            "yes_bid",
            "yes_ask",
            "no_bid",
            "no_ask",
        ]
    ).sort_values(["observed_at", "market_ticker"]).reset_index(drop=True)


def load_settlement_cache(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {str(key): float(value) for key, value in raw.items() if value is not None}


def fetch_settlements(
    market_tickers: list[str],
    *,
    cache_path: Path,
    base_url: str,
    sleep_seconds: float,
    max_workers: int,
) -> dict[str, float]:
    cache = load_settlement_cache(cache_path)
    missing = [ticker for ticker in market_tickers if ticker not in cache]
    if not missing:
        return cache

    def _fetch_one(ticker: str) -> tuple[str, float | None]:
        session = requests.Session()
        url = f"{base_url.rstrip('/')}/markets/{ticker}"
        response = None
        for attempt in range(5):
            response = session.get(url, timeout=20)
            if response.status_code != 429:
                break
            time.sleep(1.5 * (attempt + 1))
        if response is None:
            return ticker, None
        response.raise_for_status()
        market = response.json().get("market", {})
        raw_value = market.get("expiration_value")
        return ticker, None if raw_value in (None, "") else float(raw_value)

    with ThreadPoolExecutor(max_workers=max(max_workers, 1)) as executor:
        futures = [executor.submit(_fetch_one, ticker) for ticker in missing]
        for index, future in enumerate(as_completed(futures), start=1):
            ticker, value = future.result()
            if value is not None:
                cache[ticker] = value
            if index % 50 == 0 or index == len(missing):
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")
    return cache


def attach_settlements(frame: pd.DataFrame, settlements: dict[str, float]) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["settlement_price"] = enriched["market_ticker"].map(settlements)
    enriched = enriched.dropna(subset=["settlement_price"]).copy()
    direction = enriched["direction"].fillna("").astype(str).str.lower()
    settles_yes = enriched["settlement_price"].astype(float) >= enriched["threshold"].astype(float)
    below_mask = direction.isin(["below", "down"])
    enriched["settles_yes"] = np.where(below_mask, ~settles_yes, settles_yes).astype(int)
    return enriched


def attach_rolling_volatility(frame: pd.DataFrame, *, floor: float = 0.05) -> pd.DataFrame:
    spot = (
        frame[["observed_at", "spot_price"]]
        .drop_duplicates("observed_at")
        .sort_values("observed_at")
        .set_index("observed_at")["spot_price"]
        .astype(float)
    )
    returns = np.log(spot.clip(lower=1e-12)).diff()
    rolling_std = returns.rolling("90min", min_periods=20).std()
    median_dt = spot.index.to_series().diff().dt.total_seconds().rolling("90min", min_periods=20).median()
    annualization = np.sqrt(31_536_000.0 / median_dt.clip(lower=0.001))
    vol = (rolling_std * annualization).clip(lower=floor).replace([np.inf, -np.inf], np.nan).fillna(floor)
    vol_frame = vol.rename("gbm_vol").reset_index()
    return pd.merge_asof(
        frame.sort_values("observed_at"),
        vol_frame.sort_values("observed_at"),
        on="observed_at",
        direction="backward",
    ).assign(gbm_vol=lambda df: df["gbm_vol"].fillna(floor))


def compute_probabilities(frame: pd.DataFrame) -> pd.DataFrame:
    spot = np.maximum(frame["spot_price"].astype(float).to_numpy(), 1e-12)
    threshold = np.maximum(frame["threshold"].astype(float).to_numpy(), 1e-12)
    tte_seconds = np.maximum((frame["expiry"] - frame["observed_at"]).dt.total_seconds().to_numpy(dtype=float), 0.0)
    tte_years = tte_seconds / 31_536_000.0
    volatility = np.maximum(frame["gbm_vol"].astype(float).to_numpy(), 0.05)
    denom = volatility * np.sqrt(np.maximum(tte_years, 1e-18))
    numerator = np.log(spot / threshold) - 0.5 * volatility * volatility * tte_years
    d = np.divide(numerator, denom, out=np.zeros_like(numerator), where=denom > 0)
    prob_above = norm_cdf(d)
    immediate = (spot >= threshold).astype(float)
    prob_yes = np.where(tte_years > 0, np.clip(prob_above, 0.0, 1.0), immediate)
    direction = frame["direction"].fillna("").astype(str).str.lower().to_numpy()
    below_mask = np.isin(direction, ["below", "down"])
    prob_yes = np.where(below_mask, 1.0 - prob_yes, prob_yes)

    enriched = frame.copy()
    enriched["tte_minutes"] = tte_seconds / 60.0
    enriched["prob_yes"] = prob_yes
    enriched["prob_no"] = 1.0 - prob_yes
    for side in ("yes", "no"):
        enriched[f"{side}_ask_c"] = np.rint(enriched[f"{side}_ask"].astype(float).to_numpy() * 100.0).astype(int)
        enriched[f"{side}_bid_c"] = np.rint(enriched[f"{side}_bid"].astype(float).to_numpy() * 100.0).astype(int)
        enriched[f"{side}_spread_c"] = enriched[f"{side}_ask_c"] - enriched[f"{side}_bid_c"]
        enriched[f"{side}_edge_ask"] = enriched[f"prob_{side}"] - enriched[f"{side}_ask"].astype(float)
    return enriched


def vectorized_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    base_mask = (
        (frame["contract_type"] == "threshold")
        & (frame["volume"].fillna(0.0).astype(float) >= MIN_MARKET_VOLUME)
        & (frame["tte_minutes"] > 0.0)
        & (frame["tte_minutes"] < MAX_TTE_MINUTES)
    )
    specs = (
        ("yes_continuation_mid", "yes", 0, 6, GBM_MIN_EDGE_MID),
        ("no_continuation_mid", "no", 1, 6, GBM_MIN_EDGE_MID),
        ("yes_continuation_wide", "yes", 2, 8, GBM_MIN_EDGE_WIDE_YES),
        ("no_continuation_wide", "no", 3, 8, GBM_MIN_EDGE_MID),
    )
    parts: list[pd.DataFrame] = []
    for strategy_name, side, priority, max_spread, base_edge in specs:
        ask_c = frame[f"{side}_ask_c"]
        spread_c = frame[f"{side}_spread_c"]
        edge = frame[f"{side}_edge_ask"]
        probability = frame[f"prob_{side}"]
        direction_mask = (
            frame["spot_price"].astype(float) >= (frame["threshold"].astype(float) + DISTANCE_THRESHOLD_DOLLARS)
            if side == "yes"
            else frame["spot_price"].astype(float) <= (frame["threshold"].astype(float) - DISTANCE_THRESHOLD_DOLLARS)
        )
        min_edge = np.where(ask_c > SOFT_ENTRY_CAP_CENTS, base_edge + EXPENSIVE_ENTRY_EDGE_PREMIUM, base_edge)
        mask = (
            base_mask
            & direction_mask
            & (ask_c >= 38)
            & (ask_c <= 62)
            & (spread_c < max_spread)
            & (edge >= min_edge)
        )
        if not bool(mask.any()):
            continue
        part = frame.loc[
            mask,
            [
                "observed_at",
                "market_ticker",
                "expiry",
                "settles_yes",
                "yes_ask_c",
                "no_ask_c",
                "yes_edge_ask",
                "no_edge_ask",
                "prob_yes",
                "prob_no",
            ],
        ].copy()
        part["strategy_name"] = strategy_name
        part["side"] = side
        part["strategy_priority"] = priority
        part["entry_price_cents"] = ask_c.loc[mask].to_numpy(dtype=int)
        part["gbm_edge"] = edge.loc[mask].to_numpy(dtype=float)
        part["gbm_probability"] = probability.loc[mask].to_numpy(dtype=float)
        parts.append(part)
    if not parts:
        return pd.DataFrame()
    candidates = pd.concat(parts, ignore_index=True)
    return (
        candidates.sort_values(["observed_at", "market_ticker", "strategy_priority"])
        .drop_duplicates(["observed_at", "market_ticker"], keep="first")
        .sort_values(["observed_at", "expiry", "market_ticker", "strategy_priority"])
        .reset_index(drop=True)
    )


def settle_position(position: OpenPosition, *, exit_time: pd.Timestamp, settles_yes: int) -> dict[str, Any]:
    winner = (position.side == "yes" and settles_yes == 1) or (position.side == "no" and settles_yes == 0)
    exit_price_cents = 100 if winner else 0
    gross_pnl = (exit_price_cents - position.entry_price_cents) * position.contracts
    fee_cents = (position.entry_price_cents * position.contracts) * (FEE_RATE_BPS / 10_000.0)
    return {
        "market_ticker": position.market_ticker,
        "strategy_name": position.strategy_name,
        "side": position.side,
        "contracts": position.contracts,
        "entry_time": position.entry_time,
        "exit_time": exit_time,
        "entry_price_cents": position.entry_price_cents,
        "exit_price_cents": exit_price_cents,
        "gross_pnl_cents": gross_pnl,
        "fee_cents": fee_cents,
        "realized_pnl_cents": gross_pnl - fee_cents,
        "gbm_edge": position.gbm_edge,
        "gbm_probability": position.gbm_probability,
        "exit_trigger": "settlement",
    }


def run_backtest(frame: pd.DataFrame, *, max_open_markets: int) -> pd.DataFrame:
    trades: list[dict[str, Any]] = []
    open_positions: dict[str, OpenPosition] = {}
    market_settles_yes = frame.groupby("market_ticker")["settles_yes"].last().astype(int).to_dict()
    market_expiry = frame.groupby("market_ticker")["expiry"].last().to_dict()
    candidates = vectorized_candidates(frame)
    if candidates.empty:
        return pd.DataFrame()

    for observed_at, tick in candidates.groupby("observed_at", sort=True):
        expired = [ticker for ticker, position in open_positions.items() if position.expiry <= observed_at]
        for ticker in expired:
            position = open_positions.pop(ticker)
            trades.append(
                settle_position(
                    position,
                    exit_time=pd.Timestamp(market_expiry.get(ticker, position.expiry)),
                    settles_yes=int(market_settles_yes[ticker]),
                )
            )
        if len(open_positions) >= max_open_markets:
            continue
        for _, row in tick.sort_values(["expiry", "market_ticker"]).iterrows():
            ticker = str(row["market_ticker"])
            if ticker in open_positions or len(open_positions) >= max_open_markets:
                continue
            open_positions[ticker] = OpenPosition(
                market_ticker=ticker,
                strategy_name=str(row["strategy_name"]),
                side=str(row["side"]),
                contracts=1,
                entry_time=pd.Timestamp(row["observed_at"]),
                entry_price_cents=int(row["entry_price_cents"]),
                expiry=pd.Timestamp(row["expiry"]),
                gbm_edge=float(row["gbm_edge"]),
                gbm_probability=float(row["gbm_probability"]),
            )

    for ticker, position in sorted(open_positions.items(), key=lambda item: item[1].expiry):
        if pd.Timestamp(position.expiry) <= pd.Timestamp.now(tz=UTC):
            trades.append(
                settle_position(
                    position,
                    exit_time=pd.Timestamp(market_expiry.get(ticker, position.expiry)),
                    settles_yes=int(market_settles_yes[ticker]),
                )
            )

    trades_frame = pd.DataFrame(trades)
    if not trades_frame.empty:
        trades_frame["entry_time"] = pd.to_datetime(trades_frame["entry_time"], utc=True)
        trades_frame["exit_time"] = pd.to_datetime(trades_frame["exit_time"], utc=True)
        trades_frame["hold_minutes"] = (trades_frame["exit_time"] - trades_frame["entry_time"]).dt.total_seconds() / 60.0
    return trades_frame


def summarize(trades: pd.DataFrame, snapshots: pd.DataFrame) -> dict[str, Any]:
    if trades.empty:
        return {
            "trade_count": 0,
            "win_rate": 0.0,
            "total_pnl_cents": 0.0,
            "total_pnl_dollars": 0.0,
            "max_drawdown_cents": 0.0,
        }
    pnl = trades["realized_pnl_cents"].astype(float)
    equity = pnl.cumsum()
    drawdown = equity - equity.cummax()
    return {
        "snapshot_start": str(snapshots["observed_at"].min()),
        "snapshot_end": str(snapshots["observed_at"].max()),
        "rows": int(len(snapshots)),
        "markets": int(snapshots["market_ticker"].nunique()),
        "trade_count": int(len(trades)),
        "win_rate": float((pnl > 0).mean()),
        "total_pnl_cents": float(pnl.sum()),
        "total_pnl_dollars": float(pnl.sum() / 100.0),
        "average_pnl_cents": float(pnl.mean()),
        "median_pnl_cents": float(pnl.median()),
        "max_drawdown_cents": float(drawdown.min()),
        "max_drawdown_dollars": float(drawdown.min() / 100.0),
        "roi": float(pnl.sum() / trades["entry_price_cents"].astype(float).sum()),
        "average_edge": float(trades["gbm_edge"].astype(float).mean()),
        "average_hold_minutes": float(trades["hold_minutes"].astype(float).mean()),
        "fee_rate_bps": FEE_RATE_BPS,
        "strategy_counts": trades["strategy_name"].value_counts().to_dict(),
        "side_counts": trades["side"].value_counts().to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay GOD/exp_fills2-style signals over live DB snapshots.")
    parser.add_argument("--config", help="Optional TOML config override path")
    parser.add_argument("--series", default="KXBTC15M")
    parser.add_argument("--output-dir", default=str(ROOT / "data" / "reports" / "god_live_db"))
    parser.add_argument("--max-open-markets", type=int, default=3)
    parser.add_argument("--kalshi-base-url", default="https://api.elections.kalshi.com/trade-api/v2")
    parser.add_argument("--settlement-sleep-seconds", type=float, default=0.0)
    parser.add_argument("--settlement-workers", type=int, default=3)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshots = load_live_snapshots(config_path=args.config, series=args.series)
    if snapshots.empty:
        raise SystemExit("No snapshots found.")
    expired_tickers = (
        snapshots.loc[snapshots["expiry"] <= pd.Timestamp.now(tz=UTC), "market_ticker"]
        .drop_duplicates()
        .astype(str)
        .tolist()
    )
    settlements = fetch_settlements(
        expired_tickers,
        cache_path=output_dir / "settlements_cache.json",
        base_url=args.kalshi_base_url,
        sleep_seconds=args.settlement_sleep_seconds,
        max_workers=args.settlement_workers,
    )
    settled = attach_settlements(snapshots, settlements)
    settled = settled.loc[settled["expiry"] <= pd.Timestamp.now(tz=UTC)].copy()
    enriched = compute_probabilities(attach_rolling_volatility(settled))
    trades = run_backtest(enriched, max_open_markets=args.max_open_markets)
    summary = summarize(trades, enriched)
    summary.update(
        {
            "mode": "GOD_live_db_replay",
            "candidate_rows": int(len(vectorized_candidates(enriched))),
            "assumptions": [
                "One contract per entry signal.",
                "Entry fill assumed at displayed ask; resting-order queue position is not replayed.",
                "GOD live profile is exp_fills2 plus Coinbase volatility bootstrap and no orderbook precheck.",
                "Settlement values are fetched from Kalshi market metadata and cached.",
            ],
        }
    )
    trades_path = output_dir / "god_live_db_trades.csv"
    summary_path = output_dir / "god_live_db_summary.json"
    trades.to_csv(trades_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    print(f"Saved trades to {trades_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()

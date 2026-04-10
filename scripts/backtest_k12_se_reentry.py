from __future__ import annotations

import sys
from dataclasses import dataclass
from math import erf, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kabot.models.gbm_threshold import GBMThresholdModel


DATA_PATH = ROOT / "data" / "kxbtc15m_30d_history_complement.csv"
VOL_PATH = ROOT / "data" / "reports" / "30d_kxbtc15m_report_pack" / "_vol_cache_15m.csv"
REPORT_DIR = ROOT / "data" / "reports" / "30d_kxbtc15m_report_pack"

DISTANCE_THRESHOLD_DOLLARS = 10.0
MIN_MARKET_VOLUME = 5000.0
SOFT_ENTRY_CAP_CENTS = 55
EXPENSIVE_ENTRY_EDGE_PREMIUM = 0.02
REENTRY_EDGE_PREMIUM = 0.02
REENTRY_MIN_PRICE_IMPROVEMENT_CENTS = 3
GBM_MIN_EDGE_MID = 0.02
GBM_MIN_EDGE_WIDE = 0.04
VOLATILITY_FILL = 0.5


@dataclass(frozen=True)
class ReentryState:
    side: str
    exit_price_cents: int
    successful_reentries: int = 0


def norm_cdf(values: np.ndarray) -> np.ndarray:
    erf_vec = np.frompyfunc(lambda x: erf(float(x) / sqrt(2.0)), 1, 1)
    return 0.5 * (1.0 + np.asarray(erf_vec(values), dtype=float))


def outcome_yes(frame: pd.DataFrame) -> np.ndarray:
    settlement = frame["settlement_price"].astype(float).to_numpy()
    threshold = frame["threshold"].astype(float).to_numpy()
    direction = frame["direction"].fillna("").astype(str).str.lower().to_numpy()
    result = settlement >= threshold
    below_mask = np.isin(direction, ["below", "down"])
    result = np.where(below_mask, settlement < threshold, result)
    return result.astype(int)


def attach_volatility(frame: pd.DataFrame, vol_frame: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge_asof(
        frame.sort_values("observed_at"),
        vol_frame.sort_values("observed_at"),
        on="observed_at",
        direction="backward",
    )
    merged["gbm_vol"] = merged["gbm_vol"].fillna(VOLATILITY_FILL)
    return merged


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
    enriched["yes_ask_c"] = np.rint(enriched["yes_ask"].astype(float).to_numpy() * 100.0).astype(int)
    enriched["no_ask_c"] = np.rint(enriched["no_ask"].astype(float).to_numpy() * 100.0).astype(int)
    enriched["yes_bid_c"] = np.rint(enriched["yes_bid"].astype(float).to_numpy() * 100.0).astype(int)
    enriched["no_bid_c"] = np.rint(enriched["no_bid"].astype(float).to_numpy() * 100.0).astype(int)
    enriched["yes_spread_c"] = enriched["yes_ask_c"] - enriched["yes_bid_c"]
    enriched["no_spread_c"] = enriched["no_ask_c"] - enriched["no_bid_c"]
    enriched["yes_edge_ask"] = enriched["prob_yes"] - enriched["yes_ask"].astype(float)
    enriched["no_edge_ask"] = enriched["prob_no"] - enriched["no_ask"].astype(float)
    enriched["yes_edge_bid"] = enriched["prob_yes"] - enriched["yes_bid"].astype(float)
    enriched["no_edge_bid"] = enriched["prob_no"] - enriched["no_bid"].astype(float)
    enriched["settles_yes"] = outcome_yes(enriched)
    return enriched


def required_gbm_edge(strategy_name: str, ask_cents: int) -> float:
    base = GBM_MIN_EDGE_WIDE if "wide" in strategy_name else GBM_MIN_EDGE_MID
    if ask_cents > SOFT_ENTRY_CAP_CENTS:
        return base + EXPENSIVE_ENTRY_EDGE_PREMIUM
    return base


def pick_candidate(row: pd.Series, *, max_tte_minutes: float) -> dict[str, object] | None:
    if row["contract_type"] != "threshold":
        return None
    if pd.isna(row["threshold"]):
        return None
    if float(row.get("volume", 0.0) or 0.0) < MIN_MARKET_VOLUME:
        return None
    tte = float(row["tte_minutes"])
    if not (0.0 < tte < max_tte_minutes):
        return None

    threshold = float(row["threshold"])
    spot = float(row["spot_price"])
    checks = (
        ("yes_continuation_mid", "yes", int(row["yes_ask_c"]), int(row["yes_spread_c"]), float(row["yes_edge_ask"]), 40, 60, 6),
        ("no_continuation_mid", "no", int(row["no_ask_c"]), int(row["no_spread_c"]), float(row["no_edge_ask"]), 40, 60, 6),
        ("yes_continuation_wide", "yes", int(row["yes_ask_c"]), int(row["yes_spread_c"]), float(row["yes_edge_ask"]), 35, 60, 8),
        ("no_continuation_wide", "no", int(row["no_ask_c"]), int(row["no_spread_c"]), float(row["no_edge_ask"]), 35, 60, 8),
    )
    for strategy_name, side, ask_cents, spread_cents, ask_edge, min_price, max_price, max_spread in checks:
        if side == "yes" and spot < (threshold + DISTANCE_THRESHOLD_DOLLARS):
            continue
        if side == "no" and spot > (threshold - DISTANCE_THRESHOLD_DOLLARS):
            continue
        if not (min_price <= ask_cents <= max_price):
            continue
        if spread_cents >= max_spread:
            continue
        if ask_edge < required_gbm_edge(strategy_name, ask_cents):
            continue
        confidence = "high" if (45 <= ask_cents <= 55 and spread_cents <= 2 and tte <= 3.0) else ("medium" if "mid" in strategy_name else "low")
        return {
            "strategy_name": strategy_name,
            "side": side,
            "price_cents": ask_cents,
            "edge": ask_edge,
            "confidence": confidence,
        }
    return None


def signal_break_reason(row: pd.Series, *, side: str) -> str | None:
    threshold = float(row["threshold"])
    spot = float(row["spot_price"])
    if side == "yes" and spot < (threshold + DISTANCE_THRESHOLD_DOLLARS):
        return "spot_below_yes_threshold"
    if side == "no" and spot > (threshold - DISTANCE_THRESHOLD_DOLLARS):
        return "spot_above_no_threshold"
    bid_edge = float(row["yes_edge_bid"] if side == "yes" else row["no_edge_bid"])
    if bid_edge < 0.0:
        return "gbm_edge_negative"
    return None


def compute_metrics(trades: pd.DataFrame) -> dict[str, float]:
    if trades.empty:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "pnl_cents": 0.0,
            "roi_pct": 0.0,
            "max_drawdown_cents": 0.0,
        }
    pnl = trades["realized_pnl_cents"].astype(float)
    wins = (pnl > 0).mean() * 100.0
    total_pnl = pnl.sum()
    total_entry = trades["entry_price_cents"].astype(float).sum()
    roi = (total_pnl / total_entry * 100.0) if total_entry > 0 else 0.0
    equity = pnl.cumsum()
    drawdown = equity - equity.cummax()
    return {
        "trades": float(len(trades)),
        "win_rate": float(wins),
        "pnl_cents": float(total_pnl),
        "roi_pct": float(roi),
        "max_drawdown_cents": float(drawdown.min()),
    }


def simulate_market(market: pd.DataFrame, *, max_tte_minutes: float) -> list[dict[str, object]]:
    trades: list[dict[str, object]] = []
    position: dict[str, object] | None = None
    reentry_state: ReentryState | None = None
    total_reentries = 0

    for _, row in market.iterrows():
        if position is not None:
            reason = signal_break_reason(row, side=str(position["side"]))
            if reason is not None:
                exit_price_cents = int(row["yes_bid_c"] if position["side"] == "yes" else row["no_bid_c"])
                trades.append(
                    {
                        "market_ticker": row["market_ticker"],
                        "strategy_name": f"{position['strategy_name']}:signal_break",
                        "trade_kind": position["trade_kind"],
                        "entry_time": position["entry_time"],
                        "exit_time": row["observed_at"],
                        "side": position["side"],
                        "entry_price_cents": position["entry_price_cents"],
                        "exit_price_cents": exit_price_cents,
                        "gbm_edge": position["gbm_edge"],
                        "realized_pnl_cents": exit_price_cents - int(position["entry_price_cents"]),
                        "exit_trigger": reason,
                    }
                )
                reentry_state = ReentryState(
                    side=str(position["side"]),
                    exit_price_cents=exit_price_cents,
                    successful_reentries=total_reentries,
                )
                position = None
                continue
            continue

        candidate = pick_candidate(row, max_tte_minutes=max_tte_minutes)
        if candidate is None:
            continue

        if reentry_state is not None:
            if reentry_state.successful_reentries >= 1:
                continue
            if str(candidate["side"]) != reentry_state.side:
                continue
            if float(candidate["edge"]) < (required_gbm_edge(str(candidate["strategy_name"]), int(candidate["price_cents"])) + REENTRY_EDGE_PREMIUM):
                continue
            if int(candidate["price_cents"]) > (reentry_state.exit_price_cents - REENTRY_MIN_PRICE_IMPROVEMENT_CENTS):
                continue
            trade_kind = "reentry"
            total_reentries += 1
            reentry_state = ReentryState(
                side=reentry_state.side,
                exit_price_cents=reentry_state.exit_price_cents,
                successful_reentries=total_reentries,
            )
        else:
            trade_kind = "initial"

        position = {
            "strategy_name": str(candidate["strategy_name"]),
            "entry_time": row["observed_at"],
            "side": str(candidate["side"]),
            "entry_price_cents": int(candidate["price_cents"]),
            "gbm_edge": float(candidate["edge"]),
            "trade_kind": trade_kind,
        }

    if position is not None:
        settles_yes = int(market["settles_yes"].iloc[-1])
        exit_price_cents = 100 if ((position["side"] == "yes" and settles_yes == 1) or (position["side"] == "no" and settles_yes == 0)) else 0
        trades.append(
            {
                "market_ticker": market["market_ticker"].iloc[-1],
                "strategy_name": str(position["strategy_name"]),
                "trade_kind": position["trade_kind"],
                "entry_time": position["entry_time"],
                "exit_time": market["expiry"].iloc[-1],
                "side": position["side"],
                "entry_price_cents": position["entry_price_cents"],
                "exit_price_cents": exit_price_cents,
                "gbm_edge": position["gbm_edge"],
                "realized_pnl_cents": exit_price_cents - int(position["entry_price_cents"]),
                "exit_trigger": "settlement",
            }
        )
    return trades


def run_backtest(frame: pd.DataFrame, *, max_tte_minutes: float) -> pd.DataFrame:
    trades: list[dict[str, object]] = []
    for _, market in frame.groupby("market_ticker", sort=False):
        trades.extend(simulate_market(market.sort_values("observed_at").reset_index(drop=True), max_tte_minutes=max_tte_minutes))
    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"], utc=True)
        trades_df["exit_time"] = pd.to_datetime(trades_df["exit_time"], utc=True)
        trades_df["hold_minutes"] = (trades_df["exit_time"] - trades_df["entry_time"]).dt.total_seconds() / 60.0
    return trades_df


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    snapshots = pd.read_csv(DATA_PATH)
    snapshots["observed_at"] = pd.to_datetime(snapshots["observed_at"], utc=True, errors="coerce")
    snapshots["expiry"] = pd.to_datetime(snapshots["expiry"], utc=True, errors="coerce")
    snapshots = snapshots.dropna(subset=["observed_at", "expiry", "spot_price", "threshold", "yes_bid", "yes_ask", "no_bid", "no_ask", "settlement_price"]).copy()

    vol_frame = pd.read_csv(VOL_PATH)
    vol_frame["observed_at"] = pd.to_datetime(vol_frame["observed_at"], utc=True, errors="coerce")
    vol_frame = vol_frame.dropna(subset=["observed_at"]).copy()

    enriched = attach_volatility(snapshots, vol_frame)
    enriched = compute_probabilities(enriched).sort_values(["market_ticker", "observed_at"]).reset_index(drop=True)

    all_summaries: list[dict[str, object]] = []
    last_trades_out: Path | None = None

    for max_tte_minutes in (8.0, 10.0, 12.0, 15.0):
        trades = run_backtest(enriched, max_tte_minutes=max_tte_minutes)
        label = int(max_tte_minutes)
        trades_out = REPORT_DIR / f"k{label}_se_reentry_30d_trades.csv"
        trades.to_csv(trades_out, index=False)
        last_trades_out = trades_out

        total_summary = compute_metrics(trades)
        initial_summary = compute_metrics(trades[trades["trade_kind"] == "initial"].copy())
        reentry_summary = compute_metrics(trades[trades["trade_kind"] == "reentry"].copy())

        all_summaries.extend(
            [
                {"max_tte_minutes": label, "variant": f"k{label}_se_reentry_total", **total_summary},
                {"max_tte_minutes": label, "variant": f"k{label}_se_reentry_initial_only", **initial_summary},
                {"max_tte_minutes": label, "variant": f"k{label}_se_reentry_reentries_only", **reentry_summary},
            ]
        )

    summary = pd.DataFrame(all_summaries)
    summary_out = REPORT_DIR / "k8_10_12_15_se_reentry_summary.csv"
    summary.to_csv(summary_out, index=False)
    print(summary.to_string(index=False))
    if last_trades_out is not None:
        print(f"\nSaved per-window trades, latest file: {last_trades_out}")
    print(f"Saved summary to: {summary_out}")


if __name__ == "__main__":
    main()

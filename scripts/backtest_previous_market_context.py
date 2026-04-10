from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "data" / "reports" / "30d_kxbtc15m_report_pack"
TRADES_PATH = REPORT_DIR / "k12_se_reentry_30d_trades.csv"
SNAPSHOTS_PATH = ROOT / "data" / "kxbtc15m_30d_history_complement.csv"
SUMMARY_PATH = REPORT_DIR / "previous_market_context_summary.csv"
ENRICHED_PATH = REPORT_DIR / "previous_market_context_trades.csv"


def compute_metrics(trades: pd.DataFrame) -> dict[str, float]:
    if trades.empty:
        return {
            "trades": 0.0,
            "wins": 0.0,
            "losses": 0.0,
            "win_rate": 0.0,
            "pnl_cents": 0.0,
            "roi_pct": 0.0,
            "max_drawdown_cents": 0.0,
        }
    pnl = trades["realized_pnl_cents"].astype(float)
    equity = pnl.cumsum()
    drawdown = equity - equity.cummax()
    wins = float((pnl > 0).sum())
    losses = float((pnl <= 0).sum())
    total_entry = trades["entry_price_cents"].astype(float).sum()
    return {
        "trades": float(len(trades)),
        "wins": wins,
        "losses": losses,
        "win_rate": float((wins / len(trades)) if len(trades) else 0.0),
        "pnl_cents": float(pnl.sum()),
        "roi_pct": float((pnl.sum() / total_entry * 100.0) if total_entry > 0 else 0.0),
        "max_drawdown_cents": float(drawdown.min()),
    }


def load_market_context() -> tuple[pd.DataFrame, pd.DataFrame]:
    snapshots = pd.read_csv(
        SNAPSHOTS_PATH,
        usecols=["market_ticker", "observed_at", "expiry", "spot_price", "threshold", "settlement_price"],
        parse_dates=["observed_at", "expiry"],
    )
    snapshots = snapshots.sort_values(["market_ticker", "observed_at"]).reset_index(drop=True)

    market_summary = (
        snapshots.groupby("market_ticker", as_index=False)
        .agg(
            expiry=("expiry", "last"),
            threshold=("threshold", "last"),
            settlement_price=("settlement_price", "last"),
            settle_spot=("spot_price", "last"),
        )
        .sort_values("expiry")
        .reset_index(drop=True)
    )
    market_summary["result_yes"] = (
        market_summary["settlement_price"].astype(float) >= market_summary["threshold"].astype(float)
    ).astype(int)
    market_summary["result_side"] = np.where(market_summary["result_yes"] == 1, "yes", "no")
    market_summary["prev_result_side"] = market_summary["result_side"].shift(1)
    market_summary["prev2_result_side"] = market_summary["result_side"].shift(2)
    market_summary["prev_two_same_side"] = (
        market_summary["prev_result_side"].notna()
        & (market_summary["prev_result_side"] == market_summary["prev2_result_side"])
    )
    market_summary["prev_two_streak_side"] = np.where(
        market_summary["prev_two_same_side"], market_summary["prev_result_side"], pd.NA
    )

    spot_series = (
        snapshots[["observed_at", "spot_price"]]
        .dropna()
        .sort_values("observed_at")
        .drop_duplicates(subset=["observed_at"], keep="last")
        .reset_index(drop=True)
    )
    return market_summary, spot_series


def build_trade_frame(market_summary: pd.DataFrame, spot_series: pd.DataFrame) -> pd.DataFrame:
    trades = pd.read_csv(
        TRADES_PATH,
        parse_dates=["entry_time", "exit_time"],
    )
    trades = trades[trades["trade_kind"] == "initial"].copy()
    trades = trades.sort_values("entry_time").reset_index(drop=True)

    trades["entry_price_cents"] = trades["entry_price_cents"].astype(int)
    trades["exit_price_cents"] = trades["exit_price_cents"].astype(int)
    trades["realized_pnl_cents"] = trades["realized_pnl_cents"].astype(int)
    trades["gbm_edge"] = trades["gbm_edge"].astype(float)
    trades["won"] = (trades["realized_pnl_cents"] > 0).astype(int)

    market_cols = ["market_ticker", "expiry", "result_side", "prev_result_side", "prev_two_same_side", "prev_two_streak_side"]
    trades = trades.merge(market_summary[market_cols], on="market_ticker", how="left")
    trades["prev_market_same_direction"] = trades["side"] == trades["prev_result_side"]
    trades["prev_two_streak_aligned"] = trades["side"] == trades["prev_two_streak_side"]

    entry_spot = pd.merge_asof(
        trades[["entry_time"]].sort_values("entry_time"),
        spot_series.rename(columns={"observed_at": "spot_observed_at", "spot_price": "entry_spot"}),
        left_on="entry_time",
        right_on="spot_observed_at",
        direction="backward",
    )
    trades = pd.concat([trades.reset_index(drop=True), entry_spot[["entry_spot"]].reset_index(drop=True)], axis=1)

    lookback = trades[["entry_time"]].copy()
    lookback["momentum_lookup_time"] = lookback["entry_time"] - pd.Timedelta(minutes=15)
    past_spot = pd.merge_asof(
        lookback.sort_values("momentum_lookup_time"),
        spot_series.rename(columns={"observed_at": "spot_observed_at", "spot_price": "spot_15m_ago"}),
        left_on="momentum_lookup_time",
        right_on="spot_observed_at",
        direction="backward",
    )
    past_spot = past_spot.sort_index()
    trades["spot_15m_ago"] = past_spot["spot_15m_ago"].to_numpy()
    trades["spot_momentum_15m"] = trades["entry_spot"] - trades["spot_15m_ago"]
    trades["btc_15m_momentum_aligned"] = (
        ((trades["side"] == "yes") & (trades["spot_momentum_15m"] > 0))
        | ((trades["side"] == "no") & (trades["spot_momentum_15m"] < 0))
    )
    trades["prev1_and_momentum"] = trades["prev_market_same_direction"] & trades["btc_15m_momentum_aligned"]
    trades["prev2_and_momentum"] = trades["prev_two_streak_aligned"] & trades["btc_15m_momentum_aligned"]
    return trades


def summarize_filters(trades: pd.DataFrame) -> pd.DataFrame:
    variants: list[tuple[str, pd.Series]] = [
        ("baseline_initial", pd.Series(True, index=trades.index)),
        ("prev_market_same_direction", trades["prev_market_same_direction"].fillna(False)),
        ("prev_two_streak_aligned", trades["prev_two_streak_aligned"].fillna(False)),
        ("btc_15m_momentum_aligned", trades["btc_15m_momentum_aligned"].fillna(False)),
        ("prev1_and_momentum", trades["prev1_and_momentum"].fillna(False)),
        ("prev2_and_momentum", trades["prev2_and_momentum"].fillna(False)),
    ]
    rows: list[dict[str, float | str]] = []
    for name, mask in variants:
        subset = trades.loc[mask].copy()
        metrics = compute_metrics(subset)
        rows.append(
            {
                "variant": name,
                **metrics,
                "avg_entry_cents": float(subset["entry_price_cents"].mean()) if not subset.empty else 0.0,
                "avg_gbm_edge": float(subset["gbm_edge"].mean()) if not subset.empty else 0.0,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    market_summary, spot_series = load_market_context()
    trades = build_trade_frame(market_summary, spot_series)
    summary = summarize_filters(trades)
    trades.to_csv(ENRICHED_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)
    print(summary.to_string(index=False))
    print(f"\nSaved summary to: {SUMMARY_PATH}")
    print(f"Saved enriched trades to: {ENRICHED_PATH}")


if __name__ == "__main__":
    main()

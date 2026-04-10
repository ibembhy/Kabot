from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import pandas as pd

from kabot.backtest.engine import BacktestConfig, BacktestEngine
from kabot.backtest.metrics import build_metrics
from kabot.models.base import ProbabilityModel
from kabot.signals.engine import SignalConfig
from kabot.trading.exits import ExitConfig
from kabot.types import BacktestResult


def compare_strategy_results(results: dict[str, BacktestResult]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name, result in results.items():
        row: dict[str, object] = {"strategy": name}
        row.update(result.summary)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("strategy").reset_index(drop=True) if rows else pd.DataFrame()


def edge_distribution_table(trades: pd.DataFrame, buckets: Sequence[float] = (-1.0, -0.05, 0.0, 0.05, 0.1, 1.0)) -> pd.DataFrame:
    if trades.empty or "edge" not in trades.columns:
        return pd.DataFrame(columns=["edge_bucket", "trade_count", "pnl", "win_rate", "average_edge"])
    frame = trades.copy()
    frame["edge_bucket"] = pd.cut(frame["edge"], bins=list(buckets), include_lowest=True)
    summary = (
        frame.groupby("edge_bucket", observed=False)
        .agg(
            trade_count=("market_ticker", "count"),
            pnl=("realized_pnl", "sum"),
            win_rate=("realized_pnl", lambda s: float((s > 0).mean()) if len(s) else 0.0),
            average_edge=("edge", "mean"),
        )
        .reset_index()
    )
    summary["edge_bucket"] = summary["edge_bucket"].astype(str)
    return summary


def early_exit_comparison(hold_result: BacktestResult, exit_result: BacktestResult) -> dict[str, float | int]:
    hold_trades = hold_result.trades if isinstance(hold_result.trades, pd.DataFrame) else pd.DataFrame()
    exit_trades = exit_result.trades if isinstance(exit_result.trades, pd.DataFrame) else pd.DataFrame()
    if hold_trades.empty or exit_trades.empty:
        return {
            "overlap_trade_count": 0,
            "hold_total_pnl": float(hold_result.summary.get("total_pnl", 0.0)),
            "exit_total_pnl": float(exit_result.summary.get("total_pnl", 0.0)),
            "pnl_delta": float(exit_result.summary.get("total_pnl", 0.0) - hold_result.summary.get("total_pnl", 0.0)),
            "early_exit_better_share": 0.0,
        }

    merged = hold_trades.merge(
        exit_trades,
        on=["market_ticker", "side"],
        suffixes=("_hold", "_exit"),
    )
    if merged.empty:
        return {
            "overlap_trade_count": 0,
            "hold_total_pnl": float(hold_result.summary.get("total_pnl", 0.0)),
            "exit_total_pnl": float(exit_result.summary.get("total_pnl", 0.0)),
            "pnl_delta": float(exit_result.summary.get("total_pnl", 0.0) - hold_result.summary.get("total_pnl", 0.0)),
            "early_exit_better_share": 0.0,
        }

    early_exit_better = (merged["realized_pnl_exit"] > merged["realized_pnl_hold"]).mean()
    return {
        "overlap_trade_count": int(len(merged)),
        "hold_total_pnl": float(hold_result.summary.get("total_pnl", 0.0)),
        "exit_total_pnl": float(exit_result.summary.get("total_pnl", 0.0)),
        "pnl_delta": float(exit_result.summary.get("total_pnl", 0.0) - hold_result.summary.get("total_pnl", 0.0)),
        "early_exit_better_share": float(early_exit_better),
    }


def volatility_sensitivity_analysis(
    *,
    snapshots: pd.DataFrame,
    features: pd.DataFrame,
    model: ProbabilityModel,
    signal_config: SignalConfig,
    exit_config: ExitConfig,
    backtest_config: BacktestConfig,
    volatility_multipliers: Sequence[float] = (0.75, 1.0, 1.25),
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for multiplier in volatility_multipliers:
        scaled_features = features.copy()
        if "realized_volatility" in scaled_features.columns:
            scaled_features["realized_volatility"] = scaled_features["realized_volatility"] * multiplier
        engine = BacktestEngine(
            model=model,
            signal_config=signal_config,
            exit_config=exit_config,
            backtest_config=replace(backtest_config),
        )
        result = engine.run(snapshots, scaled_features)
        row: dict[str, object] = {"vol_multiplier": float(multiplier)}
        row.update(result.summary)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("vol_multiplier").reset_index(drop=True) if rows else pd.DataFrame()


def trade_frequency_by_day(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["date", "trade_count", "pnl"])
    frame = trades.copy()
    frame["date"] = pd.to_datetime(frame["entry_time"], utc=True).dt.date
    return (
        frame.groupby("date", observed=False)
        .agg(trade_count=("market_ticker", "count"), pnl=("realized_pnl", "sum"))
        .reset_index()
        .sort_values("date")
    )


def performance_summary(trades: pd.DataFrame) -> dict[str, float | int]:
    return build_metrics(trades)

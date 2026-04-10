from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from itertools import product
import json
from pathlib import Path
from typing import Any

import pandas as pd

from kabot.backtest.engine import BacktestConfig, BacktestEngine
from kabot.backtest.metrics import build_metrics
from kabot.models.base import ProbabilityModel
from kabot.signals.engine import SignalConfig
from kabot.trading.exits import ExitConfig


def _coerce_ts(value: Any) -> pd.Timestamp:
    return pd.to_datetime(value, utc=True)


def run_rolling_windows(
    *,
    snapshots: pd.DataFrame,
    features: pd.DataFrame,
    model: ProbabilityModel,
    signal_config: SignalConfig,
    exit_config: ExitConfig,
    backtest_config: BacktestConfig,
    window_days: int = 7,
    step_days: int = 3,
) -> pd.DataFrame:
    if snapshots.empty or "observed_at" not in snapshots.columns:
        return pd.DataFrame()
    observed = pd.to_datetime(snapshots["observed_at"], utc=True)
    start = observed.min()
    end = observed.max()
    if pd.isna(start) or pd.isna(end):
        return pd.DataFrame()
    if window_days <= 0 or step_days <= 0:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    window = timedelta(days=window_days)
    step = timedelta(days=step_days)
    cursor = start
    while cursor + window <= end + timedelta(seconds=1):
        window_end = cursor + window
        window_snapshots = snapshots.loc[
            (pd.to_datetime(snapshots["observed_at"], utc=True) >= cursor)
            & (pd.to_datetime(snapshots["observed_at"], utc=True) < window_end)
        ].copy()
        if not window_snapshots.empty:
            feature_slice = features
            if not feature_slice.empty:
                feature_slice = feature_slice.loc[
                    (feature_slice.index >= cursor - timedelta(days=1))
                    & (feature_slice.index <= window_end + timedelta(days=1))
                ]
            result = BacktestEngine(
                model=model,
                signal_config=signal_config,
                exit_config=exit_config,
                backtest_config=replace(backtest_config),
            ).run(window_snapshots, feature_slice)
            row: dict[str, Any] = {
                "window_start": cursor.isoformat(),
                "window_end": window_end.isoformat(),
            }
            row.update(result.summary)
            rows.append(row)
        cursor += step
    return pd.DataFrame(rows)


def run_regime_splits(
    *,
    snapshots: pd.DataFrame,
    features: pd.DataFrame,
    model: ProbabilityModel,
    signal_config: SignalConfig,
    exit_config: ExitConfig,
    backtest_config: BacktestConfig,
) -> pd.DataFrame:
    if snapshots.empty or features.empty or "realized_volatility" not in features.columns:
        return pd.DataFrame()
    vol = features["realized_volatility"].dropna()
    if vol.empty:
        return pd.DataFrame()
    low_cut = float(vol.quantile(0.33))
    high_cut = float(vol.quantile(0.67))
    rows: list[dict[str, Any]] = []
    for label, mask in (
        ("low_vol", features["realized_volatility"] <= low_cut),
        ("mid_vol", (features["realized_volatility"] > low_cut) & (features["realized_volatility"] < high_cut)),
        ("high_vol", features["realized_volatility"] >= high_cut),
    ):
        regime_idx = features.loc[mask].index
        if len(regime_idx) == 0:
            continue
        start, end = regime_idx.min(), regime_idx.max()
        regime_snapshots = snapshots.loc[
            (pd.to_datetime(snapshots["observed_at"], utc=True) >= start)
            & (pd.to_datetime(snapshots["observed_at"], utc=True) <= end)
        ].copy()
        if regime_snapshots.empty:
            continue
        result = BacktestEngine(
            model=model,
            signal_config=signal_config,
            exit_config=exit_config,
            backtest_config=replace(backtest_config),
        ).run(regime_snapshots, features.loc[(features.index >= start) & (features.index <= end)])
        row: dict[str, Any] = {"regime": label, "vol_low": low_cut, "vol_high": high_cut}
        row.update(result.summary)
        rows.append(row)
    return pd.DataFrame(rows)


def run_parameter_sweep(
    *,
    snapshots: pd.DataFrame,
    features: pd.DataFrame,
    model: ProbabilityModel,
    signal_config: SignalConfig,
    exit_config: ExitConfig,
    backtest_config: BacktestConfig,
    min_edges: list[float],
    max_spreads: list[int],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for min_edge, max_spread in product(min_edges, max_spreads):
        result = BacktestEngine(
            model=model,
            signal_config=replace(signal_config, min_edge=float(min_edge), max_spread_cents=int(max_spread)),
            exit_config=exit_config,
            backtest_config=replace(backtest_config),
        ).run(snapshots, features)
        row: dict[str, Any] = {"min_edge": float(min_edge), "max_spread_cents": int(max_spread)}
        row.update(result.summary)
        rows.append(row)
    return pd.DataFrame(rows)


def run_cost_stress(
    *,
    snapshots: pd.DataFrame,
    features: pd.DataFrame,
    model: ProbabilityModel,
    signal_config: SignalConfig,
    exit_config: ExitConfig,
    backtest_config: BacktestConfig,
    stress_cases: list[dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for case in stress_cases:
        result = BacktestEngine(
            model=model,
            signal_config=signal_config,
            exit_config=exit_config,
            backtest_config=replace(
                backtest_config,
                entry_slippage_cents=int(case["entry_slippage_cents"]),
                exit_slippage_cents=int(case["exit_slippage_cents"]),
                fee_rate_bps=float(case["fee_rate_bps"]),
            ),
        ).run(snapshots, features)
        row: dict[str, Any] = {
            "scenario": str(case["name"]),
            "entry_slippage_cents": int(case["entry_slippage_cents"]),
            "exit_slippage_cents": int(case["exit_slippage_cents"]),
            "fee_rate_bps": float(case["fee_rate_bps"]),
        }
        row.update(result.summary)
        rows.append(row)
    return pd.DataFrame(rows)


def trade_tail_risk(trades: pd.DataFrame) -> dict[str, Any]:
    if trades.empty or "realized_pnl" not in trades.columns:
        return {"worst_trade": 0.0, "worst_5pct_avg": 0.0, "worst_day": 0.0}
    pnl = trades["realized_pnl"].astype(float).sort_values()
    worst_n = max(int(round(len(pnl) * 0.05)), 1)
    frame = trades.copy()
    frame["day"] = pd.to_datetime(frame["entry_time"], utc=True).dt.date
    by_day = frame.groupby("day", observed=False)["realized_pnl"].sum() if not frame.empty else pd.Series(dtype=float)
    return {
        "worst_trade": float(pnl.iloc[0]),
        "worst_5pct_avg": float(pnl.iloc[:worst_n].mean()),
        "worst_day": float(by_day.min()) if not by_day.empty else 0.0,
    }


def run_robustness_suite(
    *,
    snapshots: pd.DataFrame,
    features: pd.DataFrame,
    model: ProbabilityModel,
    signal_config: SignalConfig,
    exit_config: ExitConfig,
    backtest_config: BacktestConfig,
    rolling_window_days: int = 7,
    rolling_step_days: int = 3,
    min_edges: list[float] | None = None,
    max_spreads: list[int] | None = None,
    stress_cases: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    min_edges = min_edges or [0.03, 0.05, 0.07]
    max_spreads = max_spreads or [4, 6, 8]
    stress_cases = stress_cases or [
        {"name": "base", "entry_slippage_cents": backtest_config.entry_slippage_cents, "exit_slippage_cents": backtest_config.exit_slippage_cents, "fee_rate_bps": backtest_config.fee_rate_bps},
        {"name": "medium_cost", "entry_slippage_cents": backtest_config.entry_slippage_cents + 1, "exit_slippage_cents": backtest_config.exit_slippage_cents + 1, "fee_rate_bps": max(backtest_config.fee_rate_bps, 8.0)},
        {"name": "high_cost", "entry_slippage_cents": backtest_config.entry_slippage_cents + 2, "exit_slippage_cents": backtest_config.exit_slippage_cents + 2, "fee_rate_bps": max(backtest_config.fee_rate_bps, 15.0)},
    ]
    base_result = BacktestEngine(
        model=model,
        signal_config=signal_config,
        exit_config=exit_config,
        backtest_config=replace(backtest_config),
    ).run(snapshots, features)
    return {
        "base_summary": base_result.summary,
        "tail_risk": trade_tail_risk(base_result.trades),
        "rolling_windows": run_rolling_windows(
            snapshots=snapshots,
            features=features,
            model=model,
            signal_config=signal_config,
            exit_config=exit_config,
            backtest_config=backtest_config,
            window_days=rolling_window_days,
            step_days=rolling_step_days,
        ),
        "regime_splits": run_regime_splits(
            snapshots=snapshots,
            features=features,
            model=model,
            signal_config=signal_config,
            exit_config=exit_config,
            backtest_config=backtest_config,
        ),
        "parameter_sweep": run_parameter_sweep(
            snapshots=snapshots,
            features=features,
            model=model,
            signal_config=signal_config,
            exit_config=exit_config,
            backtest_config=backtest_config,
            min_edges=min_edges,
            max_spreads=max_spreads,
        ),
        "cost_stress": run_cost_stress(
            snapshots=snapshots,
            features=features,
            model=model,
            signal_config=signal_config,
            exit_config=exit_config,
            backtest_config=backtest_config,
            stress_cases=stress_cases,
        ),
    }


def write_robustness_outputs(*, payload: dict[str, Any], output_dir: str) -> dict[str, str]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    base_summary = pd.DataFrame([payload.get("base_summary", {})])
    base_path = target / "base_summary.csv"
    base_summary.to_csv(base_path, index=False)
    outputs["base_summary"] = str(base_path)

    tail = pd.DataFrame([payload.get("tail_risk", {})])
    tail_path = target / "tail_risk.csv"
    tail.to_csv(tail_path, index=False)
    outputs["tail_risk"] = str(tail_path)

    for key in ("rolling_windows", "regime_splits", "parameter_sweep", "cost_stress"):
        frame = payload.get(key)
        if isinstance(frame, pd.DataFrame):
            path = target / f"{key}.csv"
            frame.to_csv(path, index=False)
            outputs[key] = str(path)
    summary_path = target / "suite_summary.json"
    summary = {
        "base_summary": payload.get("base_summary", {}),
        "tail_risk": payload.get("tail_risk", {}),
        "rows": {
            key: int(len(payload[key]))
            for key in ("rolling_windows", "regime_splits", "parameter_sweep", "cost_stress")
            if isinstance(payload.get(key), pd.DataFrame)
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    outputs["suite_summary"] = str(summary_path)
    return outputs

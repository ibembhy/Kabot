from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def max_drawdown(pnl_series: Sequence[float]) -> float:
    if not pnl_series:
        return 0.0
    series = pd.Series(pnl_series, dtype=float).cumsum()
    running_max = series.cummax()
    drawdowns = series - running_max
    return float(drawdowns.min())


def build_metrics(trades: pd.DataFrame) -> dict[str, float | int]:
    if trades.empty:
        return {
            "trade_count": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "average_pnl": 0.0,
            "average_edge": 0.0,
            "average_hold_minutes": 0.0,
            "max_drawdown": 0.0,
            "roi": 0.0,
        }

    pnl = trades["realized_pnl"].astype(float)
    notional = trades["entry_notional"].astype(float)
    hold_minutes = trades["hold_minutes"].astype(float)
    edge = trades["edge"].astype(float)

    return {
        "trade_count": int(len(trades)),
        "win_rate": float((pnl > 0).mean()),
        "total_pnl": float(pnl.sum()),
        "average_pnl": float(pnl.mean()),
        "average_edge": float(edge.mean()),
        "average_hold_minutes": float(hold_minutes.mean()),
        "max_drawdown": float(max_drawdown(pnl.tolist())),
        "roi": float(pnl.sum() / notional.sum()) if notional.sum() else 0.0,
    }


def edge_distribution(trades: pd.DataFrame) -> dict[str, float]:
    if trades.empty:
        return {"p10": 0.0, "p50": 0.0, "p90": 0.0}
    values = trades["edge"].astype(float)
    return {
        "p10": float(np.percentile(values, 10)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
    }

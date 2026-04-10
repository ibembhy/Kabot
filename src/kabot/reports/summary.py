from __future__ import annotations

import pandas as pd

from kabot.backtest.metrics import build_metrics, edge_distribution


def summarize_backtest(trades: pd.DataFrame) -> dict[str, float | int]:
    summary = build_metrics(trades)
    summary.update(edge_distribution(trades))
    return summary

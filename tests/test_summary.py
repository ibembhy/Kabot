from __future__ import annotations

import pandas as pd

from kabot.reports.summary import summarize_backtest


def test_summarize_backtest_returns_core_metrics() -> None:
    trades = pd.DataFrame(
        [
            {
                "market_ticker": "A",
                "realized_pnl": 10.0,
                "entry_notional": 50.0,
                "hold_minutes": 5.0,
                "edge": 0.06,
            },
            {
                "market_ticker": "B",
                "realized_pnl": -5.0,
                "entry_notional": 40.0,
                "hold_minutes": 3.0,
                "edge": 0.02,
            },
        ]
    )
    summary = summarize_backtest(trades)
    assert summary["trade_count"] == 2
    assert "roi" in summary
    assert "p50" in summary

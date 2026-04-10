from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd

from kabot.backtest.engine import BacktestConfig, BacktestEngine
from kabot.models.gbm_threshold import GBMThresholdModel
from kabot.signals.engine import SignalConfig
from kabot.trading.exits import ExitConfig


def test_backtest_engine_runs_hold_to_settlement() -> None:
    base_time = datetime(2026, 4, 5, 12, 0, tzinfo=UTC)
    snapshots = pd.DataFrame(
        [
            {
                "source": "test",
                "series_ticker": "KXBTCD",
                "market_ticker": "KXBTCD-TEST",
                "contract_type": "threshold",
                "underlying_symbol": "BTC-USD",
                "observed_at": base_time,
                "expiry": base_time + timedelta(minutes=10),
                "spot_price": 68000.0,
                "threshold": 67950.0,
                "range_low": None,
                "range_high": None,
                "direction": "above",
                "yes_bid": 0.48,
                "yes_ask": 0.50,
                "no_bid": 0.47,
                "no_ask": 0.49,
                "mid_price": 0.49,
                "implied_probability": 0.49,
                "volume": 100.0,
                "open_interest": 10.0,
                "settlement_price": 68100.0,
                "metadata": {},
            },
            {
                "source": "test",
                "series_ticker": "KXBTCD",
                "market_ticker": "KXBTCD-TEST",
                "contract_type": "threshold",
                "underlying_symbol": "BTC-USD",
                "observed_at": base_time + timedelta(minutes=5),
                "expiry": base_time + timedelta(minutes=10),
                "spot_price": 68050.0,
                "threshold": 67950.0,
                "range_low": None,
                "range_high": None,
                "direction": "above",
                "yes_bid": 0.57,
                "yes_ask": 0.59,
                "no_bid": 0.40,
                "no_ask": 0.42,
                "mid_price": 0.585,
                "implied_probability": 0.585,
                "volume": 120.0,
                "open_interest": 12.0,
                "settlement_price": 68100.0,
                "metadata": {},
            },
        ]
    )
    features = pd.DataFrame(
        {"realized_volatility": [0.7, 0.7]},
        index=pd.to_datetime([base_time - timedelta(minutes=1), base_time + timedelta(minutes=4)], utc=True),
    )

    engine = BacktestEngine(
        model=GBMThresholdModel(),
        signal_config=SignalConfig(
            min_edge=0.01,
            min_contract_price_cents=35,
            max_contract_price_cents=65,
            max_spread_cents=10,
            max_near_money_bps=500,
        ),
        exit_config=ExitConfig(take_profit_cents=8, stop_loss_cents=8, fair_value_buffer_cents=0),
        backtest_config=BacktestConfig(strategy_mode="hold_to_settlement"),
    )

    result = engine.run(snapshots, features)
    assert result.summary["trade_count"] == 1
    assert not result.trades.empty

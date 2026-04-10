from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd

from kabot.backtest.engine import BacktestConfig, BacktestEngine
from kabot.models.gbm_threshold import GBMThresholdModel
from kabot.reports.evaluation import (
    compare_strategy_results,
    early_exit_comparison,
    edge_distribution_table,
    trade_frequency_by_day,
    volatility_sensitivity_analysis,
)
from kabot.signals.engine import SignalConfig
from kabot.trading.exits import ExitConfig


def _sample_snapshots() -> tuple[pd.DataFrame, pd.DataFrame]:
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
                "yes_bid": 0.62,
                "yes_ask": 0.64,
                "no_bid": 0.35,
                "no_ask": 0.37,
                "mid_price": 0.63,
                "implied_probability": 0.63,
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
    return snapshots, features


def _engine(strategy_mode: str) -> BacktestEngine:
    return BacktestEngine(
        model=GBMThresholdModel(),
        signal_config=SignalConfig(
            min_edge=0.01,
            min_contract_price_cents=35,
            max_contract_price_cents=65,
            max_spread_cents=20,
            max_near_money_bps=500,
        ),
        exit_config=ExitConfig(take_profit_cents=8, stop_loss_cents=8, fair_value_buffer_cents=0),
        backtest_config=BacktestConfig(strategy_mode=strategy_mode),
    )


def test_volatility_sensitivity_analysis_returns_rows() -> None:
    snapshots, features = _sample_snapshots()
    table = volatility_sensitivity_analysis(
        snapshots=snapshots,
        features=features,
        model=GBMThresholdModel(),
        signal_config=SignalConfig(
            min_edge=0.01,
            min_contract_price_cents=35,
            max_contract_price_cents=65,
            max_spread_cents=20,
            max_near_money_bps=500,
        ),
        exit_config=ExitConfig(take_profit_cents=8, stop_loss_cents=8, fair_value_buffer_cents=0),
        backtest_config=BacktestConfig(strategy_mode="hold_to_settlement"),
        volatility_multipliers=(0.8, 1.0, 1.2),
    )
    assert len(table) == 3
    assert "trade_count" in table.columns


def test_evaluation_helpers_compare_results_and_edges() -> None:
    snapshots, features = _sample_snapshots()
    hold = _engine("hold_to_settlement").run(snapshots, features)
    trade_exit = _engine("trade_exit").run(snapshots, features)

    comparison = compare_strategy_results({"hold": hold, "trade_exit": trade_exit})
    assert set(comparison["strategy"]) == {"hold", "trade_exit"}

    exit_delta = early_exit_comparison(hold, trade_exit)
    assert "pnl_delta" in exit_delta

    edge_table = edge_distribution_table(hold.trades)
    assert "trade_count" in edge_table.columns

    frequency = trade_frequency_by_day(hold.trades)
    assert len(frequency) == 1

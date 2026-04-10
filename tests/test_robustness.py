from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd

from kabot.backtest.engine import BacktestConfig
from kabot.models.gbm_threshold import GBMThresholdModel
from kabot.reports.robustness import run_robustness_suite
from kabot.signals.engine import SignalConfig
from kabot.trading.exits import ExitConfig


def _sample_snapshots() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    start = datetime(2026, 4, 1, tzinfo=UTC)
    for day in range(10):
        observed = start + timedelta(days=day)
        expiry = observed + timedelta(hours=4)
        spot = 68000.0 + day * 20.0
        threshold = 67950.0 + day * 20.0
        rows.append(
            {
                "observed_at": observed,
                "source": "test",
                "series_ticker": "KXBTC15M",
                "market_ticker": f"KXBTC15M-T{day}",
                "contract_type": "threshold",
                "underlying_symbol": "BTC-USD",
                "expiry": expiry,
                "spot_price": spot,
                "threshold": threshold,
                "range_low": None,
                "range_high": None,
                "direction": "up",
                "yes_bid": 0.46,
                "yes_ask": 0.49,
                "no_bid": 0.49,
                "no_ask": 0.52,
                "mid_price": 0.50,
                "implied_probability": 0.50,
                "volume": 10.0,
                "open_interest": 10.0,
                "settlement_price": threshold + (5.0 if day % 2 == 0 else -5.0),
                "metadata_json": {},
            }
        )
    return pd.DataFrame(rows)


def _sample_features() -> pd.DataFrame:
    start = datetime(2026, 3, 31, tzinfo=UTC)
    idx = pd.date_range(start=start, periods=60 * 24 * 12, freq="1min", tz="UTC")
    frame = pd.DataFrame(index=idx)
    frame["realized_volatility"] = 0.4 + (pd.Series(range(len(idx)), index=idx) % 10) * 0.01
    return frame


def test_run_robustness_suite_returns_expected_sections() -> None:
    payload = run_robustness_suite(
        snapshots=_sample_snapshots(),
        features=_sample_features(),
        model=GBMThresholdModel(),
        signal_config=SignalConfig(
            min_edge=0.01,
            min_contract_price_cents=30,
            max_contract_price_cents=70,
            max_spread_cents=10,
            max_near_money_bps=2000,
            min_confidence=0.0,
        ),
        exit_config=ExitConfig(
            take_profit_cents=8,
            stop_loss_cents=10,
            fair_value_buffer_cents=3,
            time_exit_minutes=2,
        ),
        backtest_config=BacktestConfig(strategy_mode="hold_to_settlement"),
        rolling_window_days=3,
        rolling_step_days=2,
    )
    assert "base_summary" in payload
    assert "tail_risk" in payload
    assert isinstance(payload["rolling_windows"], pd.DataFrame)
    assert isinstance(payload["regime_splits"], pd.DataFrame)
    assert isinstance(payload["parameter_sweep"], pd.DataFrame)
    assert isinstance(payload["cost_stress"], pd.DataFrame)
    assert len(payload["parameter_sweep"]) > 0
    assert len(payload["cost_stress"]) == 3

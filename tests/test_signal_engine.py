from __future__ import annotations

from datetime import UTC, datetime, timedelta

from kabot.signals.engine import SignalConfig, generate_signal
from kabot.types import MarketSnapshot, ProbabilityEstimate


def test_generate_signal_returns_buy_yes_for_positive_edge() -> None:
    observed_at = datetime(2026, 4, 5, 12, 0, tzinfo=UTC)
    snapshot = MarketSnapshot(
        source="test",
        series_ticker="KXBTCD",
        market_ticker="KXBTCD-TEST",
        contract_type="threshold",
        underlying_symbol="BTC-USD",
        observed_at=observed_at,
        expiry=observed_at + timedelta(minutes=10),
        spot_price=68000.0,
        threshold=68050.0,
        yes_bid=0.48,
        yes_ask=0.50,
        no_bid=0.47,
        no_ask=0.49,
        implied_probability=0.49,
    )
    estimate = ProbabilityEstimate(
        model_name="gbm_threshold",
        observed_at=observed_at,
        expiry=snapshot.expiry,
        spot_price=snapshot.spot_price,
        target_price=68050.0,
        volatility=0.6,
        drift=0.0,
        probability=0.62,
    )
    signal = generate_signal(
        snapshot,
        estimate,
        SignalConfig(
            min_edge=0.05,
            min_contract_price_cents=35,
            max_contract_price_cents=65,
            max_spread_cents=10,
            max_near_money_bps=500,
        ),
    )
    assert signal.action == "buy_yes"
    assert signal.side == "yes"

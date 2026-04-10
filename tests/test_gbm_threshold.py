from __future__ import annotations

from datetime import UTC, datetime, timedelta

from kabot.models.gbm_threshold import GBMThresholdModel, terminal_probability_above
from kabot.types import MarketSnapshot


def test_terminal_probability_above_is_bounded() -> None:
    probability = terminal_probability_above(spot_price=68000, target_price=68100, time_to_expiry_years=1 / 365, volatility=0.8)
    assert 0.0 <= probability <= 1.0


def test_gbm_model_estimates_threshold_probability() -> None:
    observed_at = datetime(2026, 4, 5, 12, 0, tzinfo=UTC)
    snapshot = MarketSnapshot(
        source="test",
        series_ticker="KXBTCD",
        market_ticker="KXBTCD-TEST",
        contract_type="threshold",
        underlying_symbol="BTC-USD",
        observed_at=observed_at,
        expiry=observed_at + timedelta(minutes=15),
        spot_price=68000.0,
        threshold=68100.0,
        yes_bid=0.42,
        yes_ask=0.45,
        no_bid=0.53,
        no_ask=0.56,
        implied_probability=0.445,
    )
    estimate = GBMThresholdModel().estimate(snapshot, volatility=0.7)
    assert estimate.model_name == "gbm_threshold"
    assert estimate.target_price == 68100.0
    assert 0.0 <= estimate.probability <= 1.0

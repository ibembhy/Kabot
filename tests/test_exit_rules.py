from __future__ import annotations

from datetime import UTC, datetime, timedelta

from kabot.trading.exits import ExitConfig, evaluate_exit
from kabot.types import MarketSnapshot


def test_evaluate_exit_triggers_take_profit() -> None:
    observed_at = datetime(2026, 4, 5, 12, 5, tzinfo=UTC)
    snapshot = MarketSnapshot(
        source="test",
        series_ticker="KXBTCD",
        market_ticker="KXBTCD-TEST",
        contract_type="threshold",
        underlying_symbol="BTC-USD",
        observed_at=observed_at,
        expiry=observed_at + timedelta(minutes=5),
        spot_price=68100.0,
        threshold=68050.0,
        yes_bid=0.61,
        yes_ask=0.63,
        no_bid=0.36,
        no_ask=0.38,
    )
    decision = evaluate_exit(
        snapshot,
        side="yes",
        entry_price_cents=50,
        fair_value_cents=60,
        contracts=1,
        config=ExitConfig(take_profit_cents=8, stop_loss_cents=8, fair_value_buffer_cents=0),
    )
    assert decision.action == "exit"
    assert decision.trigger == "take_profit"

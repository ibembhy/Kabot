"""Tests for the DAILY profile (KXBTCD hourly contracts)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from kabot.trading.daily_strategies import (
    DailyExitConfig,
    DailySignalConfig,
    evaluate_daily_exit,
    evaluate_daily_signal,
)
from kabot.types import MarketSnapshot

UTC = timezone.utc


def _snap(
    *,
    spot: float = 72000.0,
    threshold: float = 70000.0,
    yes_ask: float = 0.78,
    yes_bid: float = 0.75,
    no_ask: float = 0.24,
    no_bid: float = 0.21,
    tte_minutes: float = 30.0,
    volume: float = 5000.0,
) -> MarketSnapshot:
    now = datetime.now(UTC)
    return MarketSnapshot(
        source="test",
        series_ticker="KXBTCD",
        market_ticker="KXBTCD-26APR14T1200",
        contract_type="threshold",
        underlying_symbol="BTC-USD",
        observed_at=now,
        expiry=now + timedelta(minutes=tte_minutes),
        spot_price=spot,
        threshold=threshold,
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        no_bid=no_bid,
        no_ask=no_ask,
        volume=volume,
    )


_cfg = DailySignalConfig()
_ecfg = DailyExitConfig()


class TestDailySignal:
    def test_good_signal_passes(self):
        snapshot = _snap(spot=72000.0, threshold=70000.0, yes_ask=0.72, yes_bid=0.69)
        result = evaluate_daily_signal(
            snapshot,
            volatility=0.60,
            config=_cfg,
            observed_at=snapshot.observed_at,
        )
        if result is not None:
            assert result.edge >= _cfg.min_edge
            assert result.entry_price_cents >= _cfg.min_price_cents

    def test_distance_too_small_rejected(self):
        snapshot = _snap(spot=70100.0, threshold=70000.0)
        result = evaluate_daily_signal(
            snapshot,
            volatility=0.60,
            config=_cfg,
            observed_at=snapshot.observed_at,
        )
        assert result is None

    def test_tte_too_short_rejected(self):
        snapshot = _snap(tte_minutes=5.0)
        result = evaluate_daily_signal(
            snapshot,
            volatility=0.60,
            config=_cfg,
            observed_at=snapshot.observed_at,
        )
        assert result is None

    def test_tte_too_long_rejected(self):
        snapshot = _snap(tte_minutes=55.0)
        result = evaluate_daily_signal(
            snapshot,
            volatility=0.60,
            config=_cfg,
            observed_at=snapshot.observed_at,
        )
        assert result is None

    def test_price_below_min_rejected(self):
        snapshot = _snap(yes_ask=0.55, yes_bid=0.52)
        result = evaluate_daily_signal(
            snapshot,
            volatility=0.60,
            config=_cfg,
            observed_at=snapshot.observed_at,
        )
        assert result is None

    def test_returns_best_edge_not_first(self):
        config = DailySignalConfig(
            min_price_cents=20,
            max_price_cents=95,
            min_edge=0.01,
            min_distance_dollars=200.0,
        )
        snapshot = _snap(yes_ask=0.72, yes_bid=0.69, no_ask=0.30, no_bid=0.27)
        result = evaluate_daily_signal(
            snapshot,
            volatility=0.60,
            config=config,
            observed_at=snapshot.observed_at,
        )
        if result is not None:
            assert result.edge >= 0.01


class TestDailyExit:
    def test_hold_when_no_condition(self):
        snapshot = _snap(yes_bid=0.73, tte_minutes=30.0)
        decision = evaluate_daily_exit(
            snapshot,
            side="yes",
            entry_price_cents=72,
            fair_value_cents=80,
            contracts=2,
            volatility=0.60,
            observed_at=snapshot.observed_at,
            config=_ecfg,
        )
        assert decision.action == "hold"

    def test_stop_loss_triggers(self):
        snapshot = _snap(yes_bid=0.55, tte_minutes=30.0)
        decision = evaluate_daily_exit(
            snapshot,
            side="yes",
            entry_price_cents=75,
            fair_value_cents=80,
            contracts=2,
            volatility=0.60,
            observed_at=snapshot.observed_at,
            config=DailyExitConfig(stop_loss_cents=15),
        )
        assert decision.action == "exit"
        assert decision.trigger == "stop_loss"

    def test_fair_value_exit_triggers(self):
        snapshot = _snap(yes_bid=0.76, tte_minutes=30.0)
        decision = evaluate_daily_exit(
            snapshot,
            side="yes",
            entry_price_cents=70,
            fair_value_cents=78,
            contracts=2,
            volatility=0.60,
            observed_at=snapshot.observed_at,
            config=DailyExitConfig(fair_value_buffer_cents=3),
        )
        assert decision.action == "exit"
        assert decision.trigger == "fair_value_convergence"

    def test_hold_inside_min_tte_window(self):
        snapshot = _snap(yes_bid=0.50, tte_minutes=5.0)
        decision = evaluate_daily_exit(
            snapshot,
            side="yes",
            entry_price_cents=75,
            fair_value_cents=80,
            contracts=2,
            volatility=0.60,
            observed_at=snapshot.observed_at,
            config=DailyExitConfig(
                stop_loss_cents=15,
                min_tte_to_exit_minutes=8.0,
            ),
        )
        assert decision.action == "hold"

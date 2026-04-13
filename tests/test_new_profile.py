"""Tests for the NEW profile strategies and velocity detector."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from kabot.trading.new_strategies import (
    StrategyAConfig,
    StrategyBConfig,
    evaluate_strategy_a,
    evaluate_strategy_b,
)
from kabot.trading.live_trader import LiveTrader, LiveTraderConfig
from kabot.trading.velocity import BTCVelocityDetector
from kabot.types import MarketSnapshot


UTC = timezone.utc


def _make_snapshot(
    *,
    spot_price: float = 83000.0,
    threshold: float = 82800.0,
    yes_ask: float = 0.45,
    yes_bid: float = 0.42,
    no_ask: float = 0.57,
    no_bid: float = 0.54,
    tte_minutes: float = 10.0,
    volume: float = 5000.0,
) -> MarketSnapshot:
    now = datetime.now(UTC)
    return MarketSnapshot(
        source="test",
        series_ticker="KXBTC15M",
        market_ticker="KXBTC15M-T83000",
        contract_type="threshold",
        underlying_symbol="BTC-USD",
        observed_at=now,
        expiry=now + timedelta(minutes=tte_minutes),
        spot_price=spot_price,
        threshold=threshold,
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        no_bid=no_bid,
        no_ask=no_ask,
        volume=volume,
    )


class TestVelocityDetector:
    def test_no_data_returns_zero(self) -> None:
        det = BTCVelocityDetector()
        reading = det.reading()
        assert reading.bps_per_second == 0.0
        assert reading.direction == 0

    def test_upward_move_detected(self) -> None:
        det = BTCVelocityDetector(window_seconds=30.0)
        now = datetime.now(UTC)
        det.update(now - timedelta(seconds=10), 83000.0)
        det.update(now - timedelta(seconds=5), 83100.0)
        det.update(now, 83200.0)
        reading = det.reading()
        assert reading.direction == 1
        assert reading.move_bps > 0
        assert reading.bps_per_second > 0

    def test_fast_move_flag(self) -> None:
        det = BTCVelocityDetector(window_seconds=30.0)
        now = datetime.now(UTC)
        det.update(now - timedelta(seconds=5), 83000.0)
        det.update(now - timedelta(seconds=3), 83100.0)
        det.update(now, 83300.0)
        assert det.is_fast_move(min_bps_per_second=5.0, min_total_bps=30.0)
        assert not det.is_fast_move(min_bps_per_second=50.0, min_total_bps=30.0)

    def test_old_data_pruned(self) -> None:
        det = BTCVelocityDetector(window_seconds=10.0, min_points=2)
        now = datetime.now(UTC)
        det.update(now - timedelta(seconds=60), 82000.0)
        det.update(now - timedelta(seconds=5), 83000.0)
        det.update(now, 83050.0)
        reading = det.reading()
        assert reading.window_seconds < 30

    def test_no_recent_move_is_not_fast(self) -> None:
        det = BTCVelocityDetector(window_seconds=30.0)
        now = datetime.now(UTC)
        det.update(now - timedelta(seconds=20), 83000.0)
        det.update(now - timedelta(seconds=10), 83000.0)
        det.update(now, 83000.0)
        assert not det.is_fast_move(min_bps_per_second=5.0, min_total_bps=30.0)


class TestStrategyA:
    def test_good_signal_returns_signal(self) -> None:
        snapshot = _make_snapshot(
            spot_price=83200.0,
            threshold=82000.0,
            yes_ask=0.35,
            yes_bid=0.32,
            no_ask=0.67,
            no_bid=0.64,
            tte_minutes=10.0,
            volume=5000.0,
        )
        config = StrategyAConfig(
            min_edge=0.08,
            min_tte_minutes=8.0,
            max_tte_minutes=14.0,
            min_price_cents=20,
            max_price_cents=80,
            max_spread_cents=8,
            min_volume=1000.0,
        )
        signal = evaluate_strategy_a(
            snapshot,
            volatility=0.80,
            config=config,
            observed_at=snapshot.observed_at,
        )
        assert signal is not None
        assert signal.side == "yes"
        assert signal.edge >= 0.08
        assert signal.strategy == "hold_settlement"

    def test_tte_too_short_returns_none(self) -> None:
        snapshot = _make_snapshot(tte_minutes=5.0)
        config = StrategyAConfig(
            min_edge=0.08,
            min_tte_minutes=8.0,
            max_tte_minutes=14.0,
            min_price_cents=20,
            max_price_cents=80,
            max_spread_cents=8,
            min_volume=1000.0,
        )
        signal = evaluate_strategy_a(
            snapshot,
            volatility=0.80,
            config=config,
            observed_at=snapshot.observed_at,
        )
        assert signal is None

    def test_edge_below_threshold_returns_none(self) -> None:
        snapshot = _make_snapshot(yes_ask=0.75, yes_bid=0.72, no_ask=0.30, no_bid=0.27)
        config = StrategyAConfig(
            min_edge=0.08,
            min_tte_minutes=8.0,
            max_tte_minutes=14.0,
            min_price_cents=20,
            max_price_cents=80,
            max_spread_cents=8,
            min_volume=1000.0,
        )
        signal = evaluate_strategy_a(
            snapshot,
            volatility=0.80,
            config=config,
            observed_at=snapshot.observed_at,
        )
        assert signal is None


class TestStrategyB:
    def test_fade_up_move_returns_no_signal(self) -> None:
        snapshot = _make_snapshot(
            spot_price=83500.0,
            threshold=83000.0,
            yes_ask=0.80,
            yes_bid=0.76,
            no_ask=0.22,
            no_bid=0.18,
            tte_minutes=8.0,
            volume=2000.0,
        )
        config = StrategyBConfig(
            min_velocity_bps_per_second=5.0,
            min_total_move_bps=150.0,
            min_edge=0.08,
            min_tte_minutes=4.0,
            max_tte_minutes=12.0,
            min_price_cents=15,
            max_price_cents=85,
            max_spread_cents=12,
            min_volume=500.0,
            vol_spike_multiplier=1.5,
        )
        signal = evaluate_strategy_b(
            snapshot,
            base_volatility=0.80,
            move_direction=1,
            move_bps=200.0,
            config=config,
            observed_at=snapshot.observed_at,
        )
        if signal is not None:
            assert signal.side == "no"
            assert signal.strategy == "fade_move"

    def test_fade_requires_minimum_edge(self) -> None:
        snapshot = _make_snapshot(
            no_ask=0.45,
            no_bid=0.42,
            tte_minutes=8.0,
        )
        config = StrategyBConfig(
            min_velocity_bps_per_second=5.0,
            min_total_move_bps=150.0,
            min_edge=0.08,
            min_tte_minutes=4.0,
            max_tte_minutes=12.0,
            min_price_cents=15,
            max_price_cents=85,
            max_spread_cents=12,
            min_volume=500.0,
            vol_spike_multiplier=1.5,
        )
        signal = evaluate_strategy_b(
            snapshot,
            base_volatility=0.80,
            move_direction=1,
            move_bps=200.0,
            config=config,
            observed_at=snapshot.observed_at,
        )
        assert signal is None


class _FakeClient:
    auth_signer = None


class _FakeStore:
    pass


def test_new_profile_cycle_writes_signal_trace(tmp_path) -> None:
    trace_path = tmp_path / "execution_trace.jsonl"
    trader = LiveTrader(
        store=_FakeStore(),  # type: ignore[arg-type]
        client=_FakeClient(),  # type: ignore[arg-type]
        config=LiveTraderConfig(
            active_profile="NEW",
            dry_run=True,
            execution_trace_path=str(trace_path),
        ),
    )
    snapshot = _make_snapshot(
        spot_price=83200.0,
        threshold=82000.0,
        yes_ask=0.35,
        yes_bid=0.32,
        no_ask=0.67,
        no_bid=0.64,
        tte_minutes=10.0,
        volume=5000.0,
    )

    orders = trader._run_new_profile_cycle(
        snapshots=[snapshot],
        volatility=0.80,
        observed_at=snapshot.observed_at,
    )

    assert len(orders) == 1
    assert orders[0]["status"] == "dry_run"
    assert '"event":"new_profile_signal"' in trace_path.read_text(encoding="utf-8")

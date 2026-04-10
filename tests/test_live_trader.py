from __future__ import annotations
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from kabot.trading.execution_state import ExecutionStateStore
from kabot.trading.live_trader import (
    LocalRestingEntryLock,
    LiveTrader,
    LiveTraderConfig,
    SignalBreakReentryState,
    StrategyRule,
    StrategyCandidate,
    _available_contracts_at_price,
    _strategy_rules_with_max_tte,
    select_entry_candidates,
    summarize_rejections,
)
from kabot.types import MarketSnapshot
from kabot.types import Position


def _snapshot(
    *,
    market_ticker: str,
    minutes_to_expiry: float,
    spot_price: float,
    threshold: float,
    yes_bid: float,
    yes_ask: float,
    volume: float = 10000.0,
) -> MarketSnapshot:
    observed_at = datetime(2026, 4, 6, 12, 0, tzinfo=UTC)
    return MarketSnapshot(
        source="test",
        series_ticker="KXBTC15M",
        market_ticker=market_ticker,
        contract_type="threshold",
        underlying_symbol="BTC-USD",
        observed_at=observed_at,
        expiry=observed_at + timedelta(minutes=minutes_to_expiry),
        spot_price=spot_price,
        threshold=threshold,
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        no_bid=1.0 - yes_ask,
        no_ask=1.0 - yes_bid,
        implied_probability=yes_ask,
        volume=volume,
    )


def test_select_entry_candidates_applies_exact_live_rules() -> None:
    snapshots = [
        _snapshot(
            market_ticker="A-YES-MID",
            minutes_to_expiry=4.5,
            spot_price=68010.0,
            threshold=68000.0,
            yes_bid=0.46,
            yes_ask=0.48,
        ),
        _snapshot(
            market_ticker="B-NO-MID",
            minutes_to_expiry=4.0,
            spot_price=67988.0,
            threshold=68000.0,
            yes_bid=0.52,
            yes_ask=0.54,
        ),
        _snapshot(
            market_ticker="C-YES-WIDE",
            minutes_to_expiry=4.0,
            spot_price=68012.0,
            threshold=68000.0,
            yes_bid=0.34,
            yes_ask=0.36,
        ),
        _snapshot(
            market_ticker="D-NO-WIDE",
            minutes_to_expiry=4.0,
            spot_price=67988.0,
            threshold=68000.0,
            yes_bid=0.62,
            yes_ask=0.64,
        ),
        _snapshot(
            market_ticker="E-BAD-SPREAD",
            minutes_to_expiry=4.0,
            spot_price=68010.0,
            threshold=68000.0,
            yes_bid=0.35,
            yes_ask=0.48,
        ),
        _snapshot(
            market_ticker="F-BAD-DIRECTION",
            minutes_to_expiry=4.0,
            spot_price=68005.0,
            threshold=68000.0,
            yes_bid=0.30,
            yes_ask=0.32,
        ),
        _snapshot(
            market_ticker="G-BAD-BAND",
            minutes_to_expiry=4.0,
            spot_price=68010.0,
            threshold=68000.0,
            yes_bid=0.66,
            yes_ask=0.68,
        ),
    ]

    selected = select_entry_candidates(
        snapshots,
        blocked_markets={"BLOCKED"},
        current_position_contracts={},
        current_position_strategies={},
        active_strategy_counts={},
        open_market_count=0,
        max_open_markets=4,
        max_strategy_open_counts={
            "yes_continuation_mid": 3,
            "no_continuation_mid": 2,
            "yes_continuation_wide": 2,
            "no_continuation_wide": 2,
        },
    )

    assert {(candidate.strategy_name, candidate.snapshot.market_ticker, candidate.side) for candidate in selected} == {
        ("yes_continuation_mid", "A-YES-MID", "yes"),
        ("no_continuation_mid", "B-NO-MID", "no"),
    }


def test_select_entry_candidates_respects_open_market_cap() -> None:
    snapshots = [
        _snapshot(
            market_ticker="GOOD-1",
            minutes_to_expiry=4.5,
            spot_price=68010.0,
            threshold=68000.0,
            yes_bid=0.46,
            yes_ask=0.48,
        )
    ]

    selected = select_entry_candidates(
        snapshots,
        blocked_markets=set(),
        current_position_contracts={},
        current_position_strategies={},
        active_strategy_counts={},
        open_market_count=3,
        max_open_markets=3,
        max_strategy_open_counts={
            "yes_continuation_mid": 3,
            "no_continuation_mid": 2,
            "yes_continuation_wide": 2,
            "no_continuation_wide": 2,
        },
    )

    assert selected == []


def test_select_entry_candidates_respects_min_market_volume() -> None:
    snapshots = [
        _snapshot(
            market_ticker="LOW-VOL",
            minutes_to_expiry=4.5,
            spot_price=68010.0,
            threshold=68000.0,
            yes_bid=0.46,
            yes_ask=0.48,
            volume=4000.0,
        ),
        _snapshot(
            market_ticker="HIGH-VOL",
            minutes_to_expiry=4.5,
            spot_price=68010.0,
            threshold=68000.0,
            yes_bid=0.46,
            yes_ask=0.48,
            volume=7000.0,
        ),
    ]

    selected = select_entry_candidates(
        snapshots,
        blocked_markets=set(),
        current_position_contracts={},
        current_position_strategies={},
        active_strategy_counts={},
        open_market_count=0,
        max_open_markets=4,
        max_strategy_open_counts={
            "yes_continuation_mid": 3,
            "no_continuation_mid": 2,
            "yes_continuation_wide": 2,
            "no_continuation_wide": 2,
        },
        min_market_volume=5000.0,
    )

    assert [candidate.snapshot.market_ticker for candidate in selected] == ["HIGH-VOL"]


def test_select_entry_candidates_respects_strategy_caps() -> None:
    snapshots = [
        _snapshot(
            market_ticker="NO-MID",
            minutes_to_expiry=4.0,
            spot_price=67988.0,
            threshold=68000.0,
            yes_bid=0.52,
            yes_ask=0.54,
        ),
        _snapshot(
            market_ticker="YES-WIDE",
            minutes_to_expiry=4.0,
            spot_price=68012.0,
            threshold=68000.0,
            yes_bid=0.34,
            yes_ask=0.36,
        ),
        _snapshot(
            market_ticker="NO-WIDE",
            minutes_to_expiry=4.0,
            spot_price=67988.0,
            threshold=68000.0,
            yes_bid=0.34,
            yes_ask=0.36,
        ),
    ]

    selected = select_entry_candidates(
        snapshots,
        blocked_markets=set(),
        current_position_contracts={},
        current_position_strategies={},
        active_strategy_counts={"no_continuation_mid": 2, "yes_continuation_wide": 2, "no_continuation_wide": 2},
        open_market_count=0,
        max_open_markets=3,
        max_strategy_open_counts={
            "yes_continuation_mid": 3,
            "no_continuation_mid": 2,
            "yes_continuation_wide": 2,
            "no_continuation_wide": 2,
        },
    )

    assert selected == []


def test_select_entry_candidates_allows_single_add_on_slice_for_existing_market() -> None:
    snapshots = [
        _snapshot(
            market_ticker="NO-MID",
            minutes_to_expiry=4.0,
            spot_price=67980.0,
            threshold=68000.0,
            yes_bid=0.47,
            yes_ask=0.48,
        ),
    ]

    selected = select_entry_candidates(
        snapshots,
        blocked_markets=set(),
        current_position_contracts={"NO-MID": 2},
        current_position_strategies={"NO-MID": "no_continuation_mid"},
        active_strategy_counts={"no_continuation_mid": 1},
        open_market_count=1,
        max_open_markets=3,
        max_strategy_open_counts={"yes_continuation_mid": 3, "no_continuation_mid": 2, "yes_continuation_wide": 2},
        available_balance_cents=2_000,
        bankroll_cents=2_000,
        max_contracts_per_order=2,
        max_contracts_per_market=4,
    )

    assert len(selected) == 1
    assert selected[0].strategy_name == "no_continuation_mid"
    assert selected[0].contracts == 2


def test_select_entry_candidates_caps_expensive_entries_to_single_contract() -> None:
    snapshots = [
        _snapshot(
            market_ticker="EXPENSIVE-YES",
            minutes_to_expiry=4.0,
            spot_price=68020.0,
            threshold=68000.0,
            yes_bid=0.56,
            yes_ask=0.58,
        ),
    ]

    selected = select_entry_candidates(
        snapshots,
        blocked_markets=set(),
        current_position_contracts={},
        current_position_strategies={},
        active_strategy_counts={},
        open_market_count=0,
        max_open_markets=3,
        max_strategy_open_counts={
            "yes_continuation_mid": 3,
            "no_continuation_mid": 2,
            "yes_continuation_wide": 2,
            "no_continuation_wide": 2,
        },
        available_balance_cents=5_000,
        bankroll_cents=5_000,
        max_contracts_per_order=2,
        max_contracts_per_market=4,
    )

    assert len(selected) == 1
    assert selected[0].price_cents == 58
    assert selected[0].contracts == 1


def test_select_entry_candidates_rejects_entries_above_hard_cap() -> None:
    snapshots = [
        _snapshot(
            market_ticker="TOO-EXPENSIVE-YES",
            minutes_to_expiry=4.0,
            spot_price=68025.0,
            threshold=68000.0,
            yes_bid=0.60,
            yes_ask=0.62,
        ),
    ]

    selected = select_entry_candidates(
        snapshots,
        blocked_markets=set(),
        current_position_contracts={},
        current_position_strategies={},
        active_strategy_counts={},
        open_market_count=0,
        max_open_markets=3,
        max_strategy_open_counts={
            "yes_continuation_mid": 3,
            "no_continuation_mid": 2,
            "yes_continuation_wide": 2,
            "no_continuation_wide": 2,
        },
        available_balance_cents=5_000,
        bankroll_cents=5_000,
        max_contracts_per_order=2,
        max_contracts_per_market=4,
    )

    assert selected == []


def test_select_entry_candidates_can_use_profile_specific_max_tte() -> None:
    snapshots = [
        _snapshot(
            market_ticker="EARLY-ONLY",
            minutes_to_expiry=11.0,
            spot_price=68020.0,
            threshold=68000.0,
            yes_bid=0.50,
            yes_ask=0.52,
        ),
    ]

    baseline_selected = select_entry_candidates(
        snapshots,
        blocked_markets=set(),
        current_position_contracts={},
        current_position_strategies={},
        active_strategy_counts={},
        open_market_count=0,
        max_open_markets=3,
        max_strategy_open_counts={
            "yes_continuation_mid": 3,
            "no_continuation_mid": 2,
            "yes_continuation_wide": 2,
            "no_continuation_wide": 2,
        },
        available_balance_cents=5_000,
        bankroll_cents=5_000,
    )
    experiment_selected = select_entry_candidates(
        snapshots,
        blocked_markets=set(),
        current_position_contracts={},
        current_position_strategies={},
        active_strategy_counts={},
        open_market_count=0,
        max_open_markets=3,
        max_strategy_open_counts={
            "yes_continuation_mid": 3,
            "no_continuation_mid": 2,
            "yes_continuation_wide": 2,
            "no_continuation_wide": 2,
        },
        available_balance_cents=5_000,
        bankroll_cents=5_000,
        strategy_rules=_strategy_rules_with_max_tte(12.0),
    )

    assert baseline_selected == []
    assert len(experiment_selected) == 1
    assert experiment_selected[0].snapshot.market_ticker == "EARLY-ONLY"


def test_summarize_rejections_distinguishes_block_reasons() -> None:
    snapshot = _snapshot(
        market_ticker="BLOCKED",
        minutes_to_expiry=4.0,
        spot_price=68020.0,
        threshold=68000.0,
        yes_bid=0.49,
        yes_ask=0.50,
    )

    reasons = summarize_rejections(
        [snapshot],
        blocked_markets={"BLOCKED"},
        blocked_reason_sets={"signal_break_locked": {"BLOCKED"}},
        active_strategy_counts={},
        open_market_count=0,
        max_open_markets=3,
        max_strategy_open_counts={},
        distance_threshold_dollars=10.0,
        strategy_rules=_strategy_rules_with_max_tte(10.0),
    )

    assert reasons == {"signal_break_locked": 1}


def test_available_contracts_reads_empty_orderbook_fp_as_zero_depth() -> None:
    payload = {"orderbook_fp": {"yes_dollars": [], "no_dollars": []}}
    assert _available_contracts_at_price(payload, side="no", limit_price_cents=44) == 0


def test_available_contracts_uses_reciprocal_fp_book_for_yes_buys() -> None:
    payload = {
        "orderbook_fp": {
            "yes_dollars": [["0.5600", "10.00"]],
            "no_dollars": [["0.3200", "5.00"], ["0.3700", "7.00"], ["0.4400", "11.00"]],
        }
    }
    assert _available_contracts_at_price(payload, side="yes", limit_price_cents=63) == 18


def test_available_contracts_uses_reciprocal_fp_book_for_no_buys() -> None:
    payload = {
        "orderbook_fp": {
            "yes_dollars": [["0.3300", "4.00"], ["0.4100", "6.00"], ["0.5800", "9.00"]],
            "no_dollars": [["0.4400", "8.00"]],
        }
    }
    assert _available_contracts_at_price(payload, side="no", limit_price_cents=59) == 15


class _StubStore:
    def load_market_snapshots(self, **kwargs):
        import pandas as pd

        observed_to = kwargs["observed_to"]
        times = [observed_to - timedelta(minutes=3), observed_to - timedelta(minutes=2), observed_to - timedelta(minutes=1)]
        return pd.DataFrame(
            {
                "observed_at": times,
                "spot_price": [68000.0, 68010.0, 68020.0],
            }
        )


class _StubClient:
    def __init__(self) -> None:
        self.create_payloads: list[dict] = []
        self.cancelled_order_ids: list[str] = []
        self.created = 0

    def create_order(self, payload: dict) -> dict:
        self.created += 1
        self.create_payloads.append(payload)
        return {"order": {"order_id": f"order-{self.created}", "status": "resting"}}

    def get_order(self, order_id: str) -> dict:
        if order_id == "order-1":
            return {"order": {"order_id": order_id, "status": "resting"}}
        return {"order": {"order_id": order_id, "status": "executed"}}

    def get_orderbook(self, ticker: str) -> dict:
        return {
            "orderbook": {
                "yes": {"asks": [{"price": 45, "count": 10}]},
                "no": {"asks": [{"price": 56, "count": 10}]},
            }
        }

    def get_fills(self, *, order_id: str | None = None, ticker: str | None = None, limit: int = 200) -> dict:
        return {"fills": []}

    def cancel_order(self, order_id: str) -> dict:
        self.cancelled_order_ids.append(order_id)
        return {"order_id": order_id, "status": "canceled"}

    def get_market(self, ticker: str) -> dict:
        return {
            "market": {
                "ticker": ticker,
                "series_ticker": "KXBTC15M",
                "result": None,
                "status": "open",
                "market_type": "binary",
                "yes_bid_dollars": "0.44",
                "yes_ask_dollars": "0.45",
                "no_bid_dollars": "0.55",
                "no_ask_dollars": "0.56",
                "floor_strike": "68000",
                "expiration_time": "2026-04-06T12:04:00Z",
            }
        }


def test_signal_break_exit_closes_local_position_when_enabled() -> None:
    class _ExitClient(_StubClient):
        def create_order(self, payload: dict) -> dict:
            self.created += 1
            self.create_payloads.append(payload)
            return {
                "order": {
                    "order_id": f"exit-{self.created}",
                    "status": "executed",
                    "fill_count_fp": str(payload["count"]),
                }
            }

        def get_fills(self, *, order_id: str | None = None, ticker: str | None = None, limit: int = 200) -> dict:
            return {"fills": [{"count": 2, "price": 46}]}

    client = _ExitClient()
    trader = LiveTrader(
        store=_StubStore(),
        client=client,  # type: ignore[arg-type]
        config=LiveTraderConfig(
            enable_signal_break_exit=True,
            exit_cross_cents=4,
            signal_break_confirmation_cycles=1,
            gbm_min_points=3,
        ),
    )
    observed_at = datetime(2026, 4, 6, 12, 0, tzinfo=UTC)
    snapshot = MarketSnapshot(
        source="test",
        series_ticker="KXBTC15M",
        market_ticker="OPEN-YES",
        contract_type="threshold",
        underlying_symbol="BTC-USD",
        observed_at=observed_at,
        expiry=observed_at + timedelta(minutes=6),
        spot_price=67980.0,
        threshold=68000.0,
        yes_bid=0.47,
        yes_ask=0.49,
        no_bid=0.51,
        no_ask=0.53,
        volume=10000.0,
    )
    trader.local_positions["OPEN-YES"] = Position(
        position_id="p1",
        market_ticker="OPEN-YES",
        side="yes",
        contracts=2,
        entry_time=observed_at - timedelta(minutes=1),
        entry_price_cents=55,
        expiry=snapshot.expiry,
    )
    trader.position_strategies["OPEN-YES"] = "yes_continuation_mid"

    exits = trader._reconcile_signal_break_positions(
        observed_at=observed_at,
        snapshots=[snapshot],
        volatility=0.3,
    )

    assert len(exits) == 1
    assert exits[0]["status"] == "executed"
    assert exits[0]["filled_contracts"] == 2
    assert "OPEN-YES" not in trader.local_positions
    assert "OPEN-YES" in trader.signal_break_blocked_tickers
    assert trader.closed_trades[-1].market_ticker == "OPEN-YES"
    assert trader.closed_trades[-1].realized_pnl_cents == (43 - 55) * 2


def test_signal_break_exit_blocks_same_ticker_from_future_entries() -> None:
    class _ExitClient(_StubClient):
        def create_order(self, payload: dict) -> dict:
            self.created += 1
            self.create_payloads.append(payload)
            return {
                "order": {
                    "order_id": f"exit-{self.created}",
                    "status": "executed",
                    "fill_count_fp": str(payload["count"]),
                }
            }

        def get_fills(self, *, order_id: str | None = None, ticker: str | None = None, limit: int = 200) -> dict:
            return {"fills": [{"count": 2, "price": 46}]}

    client = _ExitClient()
    trader = LiveTrader(
        store=_StubStore(),
        client=client,  # type: ignore[arg-type]
        config=LiveTraderConfig(
            enable_signal_break_exit=True,
            signal_break_confirmation_cycles=1,
            gbm_min_points=3,
        ),
    )
    observed_at = datetime(2026, 4, 6, 12, 0, tzinfo=UTC)
    exiting_snapshot = MarketSnapshot(
        source="test",
        series_ticker="KXBTC15M",
        market_ticker="OPEN-YES",
        contract_type="threshold",
        underlying_symbol="BTC-USD",
        observed_at=observed_at,
        expiry=observed_at + timedelta(minutes=6),
        spot_price=67980.0,
        threshold=68000.0,
        yes_bid=0.47,
        yes_ask=0.49,
        no_bid=0.51,
        no_ask=0.53,
        volume=10000.0,
    )
    trader.local_positions["OPEN-YES"] = Position(
        position_id="p1",
        market_ticker="OPEN-YES",
        side="yes",
        contracts=2,
        entry_time=observed_at - timedelta(minutes=1),
        entry_price_cents=55,
        expiry=exiting_snapshot.expiry,
    )
    trader.position_strategies["OPEN-YES"] = "yes_continuation_mid"

    trader._reconcile_signal_break_positions(
        observed_at=observed_at,
        snapshots=[exiting_snapshot],
        volatility=0.3,
    )

    reentry_candidate_snapshot = _snapshot(
        market_ticker="OPEN-YES",
        minutes_to_expiry=4.0,
        spot_price=68025.0,
        threshold=68000.0,
        yes_bid=0.49,
        yes_ask=0.50,
    )
    selected = select_entry_candidates(
        [reentry_candidate_snapshot],
        blocked_markets=set(trader.signal_break_blocked_tickers),
        current_position_contracts={},
        current_position_strategies={},
        active_strategy_counts={},
        open_market_count=0,
        max_open_markets=3,
        max_strategy_open_counts={},
        distance_threshold_dollars=10.0,
        strategy_rules=_strategy_rules_with_max_tte(10.0),
    )

    assert selected == []


def test_signal_break_exit_requires_confirmation_cycles() -> None:
    trader = LiveTrader(
        store=_StubStore(),
        client=_StubClient(),  # type: ignore[arg-type]
        config=LiveTraderConfig(
            enable_signal_break_exit=True,
            signal_break_confirmation_cycles=2,
            gbm_min_points=3,
        ),
    )
    observed_at = datetime(2026, 4, 6, 12, 0, tzinfo=UTC)
    snapshot = MarketSnapshot(
        source="test",
        series_ticker="KXBTC15M",
        market_ticker="OPEN-YES",
        contract_type="threshold",
        underlying_symbol="BTC-USD",
        observed_at=observed_at,
        expiry=observed_at + timedelta(minutes=6),
        spot_price=68004.0,
        threshold=68000.0,
        yes_bid=0.47,
        yes_ask=0.49,
        no_bid=0.51,
        no_ask=0.53,
        volume=10000.0,
    )
    trader.local_positions["OPEN-YES"] = Position(
        position_id="p1",
        market_ticker="OPEN-YES",
        side="yes",
        contracts=2,
        entry_time=observed_at - timedelta(minutes=1),
        entry_price_cents=55,
        expiry=snapshot.expiry,
    )
    trader.position_strategies["OPEN-YES"] = "yes_continuation_mid"

    first = trader._reconcile_signal_break_positions(
        observed_at=observed_at,
        snapshots=[snapshot],
        volatility=0.3,
    )
    second = trader._reconcile_signal_break_positions(
        observed_at=observed_at + timedelta(seconds=1),
        snapshots=[snapshot],
        volatility=0.3,
    )

    assert first == []
    assert len(second) == 1


def test_price_stop_respects_grace_and_confirmation() -> None:
    class _ExitClient(_StubClient):
        def create_order(self, payload: dict) -> dict:
            self.created += 1
            self.create_payloads.append(payload)
            return {
                "order": {
                    "order_id": f"exit-{self.created}",
                    "status": "executed",
                    "fill_count_fp": str(payload["count"]),
                }
            }

        def get_fills(self, *, order_id: str | None = None, ticker: str | None = None, limit: int = 200) -> dict:
            return {"fills": [{"count": 2, "price": 40}]}

    client = _ExitClient()
    trader = LiveTrader(
        store=_StubStore(),
        client=client,  # type: ignore[arg-type]
        config=LiveTraderConfig(
            enable_signal_break_exit=True,
            signal_break_confirmation_cycles=1,
            price_stop_cents=15,
            price_stop_grace_seconds=90,
            price_stop_confirm_cycles=3,
            hard_stop_cents=0,
            gbm_min_points=3,
        ),
    )
    base_time = datetime(2026, 4, 6, 12, 0, tzinfo=UTC)
    trader.local_positions["OPEN-YES"] = Position(
        position_id="p1",
        market_ticker="OPEN-YES",
        side="yes",
        contracts=2,
        entry_time=base_time - timedelta(seconds=30),
        entry_price_cents=55,
        expiry=base_time + timedelta(minutes=6),
    )
    trader.position_strategies["OPEN-YES"] = "yes_continuation_mid"

    early_snapshot = MarketSnapshot(
        source="test",
        series_ticker="KXBTC15M",
        market_ticker="OPEN-YES",
        contract_type="threshold",
        underlying_symbol="BTC-USD",
        observed_at=base_time,
        expiry=base_time + timedelta(minutes=6),
        spot_price=68030.0,
        threshold=68000.0,
        yes_bid=0.40,
        yes_ask=0.42,
        no_bid=0.58,
        no_ask=0.60,
        volume=10000.0,
    )

    first = trader._reconcile_signal_break_positions(
        observed_at=base_time,
        snapshots=[early_snapshot],
        volatility=None,
    )

    assert first == []

    later_time = base_time + timedelta(seconds=100)
    later_snapshot = MarketSnapshot(
        source="test",
        series_ticker="KXBTC15M",
        market_ticker="OPEN-YES",
        contract_type="threshold",
        underlying_symbol="BTC-USD",
        observed_at=later_time,
        expiry=base_time + timedelta(minutes=6),
        spot_price=68030.0,
        threshold=68000.0,
        yes_bid=0.40,
        yes_ask=0.42,
        no_bid=0.58,
        no_ask=0.60,
        volume=10000.0,
    )

    second = trader._reconcile_signal_break_positions(
        observed_at=later_time,
        snapshots=[later_snapshot],
        volatility=None,
    )
    third = trader._reconcile_signal_break_positions(
        observed_at=later_time + timedelta(seconds=1),
        snapshots=[later_snapshot],
        volatility=None,
    )
    fourth = trader._reconcile_signal_break_positions(
        observed_at=later_time + timedelta(seconds=2),
        snapshots=[later_snapshot],
        volatility=None,
    )

    assert second == []
    assert third == []
    assert len(fourth) == 1


def test_hard_stop_ignores_grace_and_confirmation() -> None:
    class _ExitClient(_StubClient):
        def create_order(self, payload: dict) -> dict:
            self.created += 1
            self.create_payloads.append(payload)
            return {
                "order": {
                    "order_id": f"exit-{self.created}",
                    "status": "executed",
                    "fill_count_fp": str(payload["count"]),
                }
            }

        def get_fills(self, *, order_id: str | None = None, ticker: str | None = None, limit: int = 200) -> dict:
            return {"fills": [{"count": 2, "price": 33}]}

    client = _ExitClient()
    trader = LiveTrader(
        store=_StubStore(),
        client=client,  # type: ignore[arg-type]
        config=LiveTraderConfig(
            enable_signal_break_exit=True,
            signal_break_confirmation_cycles=3,
            price_stop_cents=15,
            price_stop_grace_seconds=90,
            price_stop_confirm_cycles=3,
            hard_stop_cents=22,
            gbm_min_points=3,
        ),
    )
    base_time = datetime(2026, 4, 6, 12, 0, tzinfo=UTC)
    trader.local_positions["OPEN-NO"] = Position(
        position_id="p2",
        market_ticker="OPEN-NO",
        side="no",
        contracts=2,
        entry_time=base_time - timedelta(seconds=10),
        entry_price_cents=55,
        expiry=base_time + timedelta(minutes=6),
    )
    trader.position_strategies["OPEN-NO"] = "no_continuation_mid"

    snapshot = MarketSnapshot(
        source="test",
        series_ticker="KXBTC15M",
        market_ticker="OPEN-NO",
        contract_type="threshold",
        underlying_symbol="BTC-USD",
        observed_at=base_time,
        expiry=base_time + timedelta(minutes=6),
        spot_price=67950.0,
        threshold=68000.0,
        yes_bid=0.67,
        yes_ask=0.69,
        no_bid=0.33,
        no_ask=0.35,
        volume=10000.0,
    )

    exits = trader._reconcile_signal_break_positions(
        observed_at=base_time,
        snapshots=[snapshot],
        volatility=None,
    )

    assert len(exits) == 1


def test_submit_order_can_use_execution_session_and_write_trace(monkeypatch, tmp_path: Path) -> None:
    class _ExecutionClient(_StubClient):
        def create_order(self, payload: dict) -> dict:
            self.created += 1
            self.create_payloads.append(payload)
            if self.created == 1:
                return {"order": {"order_id": f"order-{self.created}", "status": "canceled", "fill_count_fp": "0.00"}}
            return {"order": {"order_id": f"order-{self.created}", "status": "executed", "fill_count_fp": "2.00"}}

    client = _ExecutionClient()
    trace_path = tmp_path / "execution_trace.jsonl"
    trader = LiveTrader(
        store=_StubStore(),
        client=client,  # type: ignore[arg-type]
        config=LiveTraderConfig(
            enable_execution_sessions=True,
            execution_trace_path=str(trace_path),
            execution_session_attempts=2,
            execution_session_retry_delay_seconds=0.0,
            execution_cross_cents=1,
            gbm_min_points=3,
        ),
    )
    monkeypatch.setattr("kabot.trading.live_trader.time.sleep", lambda *_args, **_kwargs: None)
    snapshot = _snapshot(
        market_ticker="TEST",
        minutes_to_expiry=4.0,
        spot_price=67980.0,
        threshold=68000.0,
        yes_bid=0.44,
        yes_ask=0.45,
    )
    candidate = StrategyCandidate(
        strategy_name="no_continuation_mid",
        confidence="medium",
        snapshot=snapshot,
        side="no",
        price_cents=56,
        contracts=2,
        gbm_probability=0.7,
        gbm_edge=0.14,
    )
    monkeypatch.setattr(trader, "_refresh_candidate_from_live_state", lambda **_kwargs: candidate)

    result = trader._submit_order(candidate)

    assert result["status"] == "executed"
    assert result["filled_contracts"] == 2
    lines = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    events = [line["event"] for line in lines]
    assert "execution_session_start" in events
    assert "execution_attempt_submitted" in events
    assert "execution_session_complete" in events


def test_reentry_rules_require_stronger_edge_and_fresh_lag() -> None:
    trader = LiveTrader(
        store=_StubStore(),
        client=_StubClient(),  # type: ignore[arg-type]
        config=LiveTraderConfig(
            enable_signal_break_reentry=True,
            reentry_edge_premium=0.02,
            reentry_min_price_improvement_cents=3,
            gbm_min_points=3,
        ),
    )
    trader.reentry_states["TEST"] = SignalBreakReentryState(
        market_ticker="TEST",
        exited_at=datetime(2026, 4, 6, 12, 0, tzinfo=UTC),
        exit_side="no",
        exit_strategy_name="no_continuation_mid",
        exit_reference_price_cents=50,
        exit_execution_price_cents=47,
    )
    snapshot = _snapshot(
        market_ticker="TEST",
        minutes_to_expiry=4.0,
        spot_price=67980.0,
        threshold=68000.0,
        yes_bid=0.55,
        yes_ask=0.56,
    )
    weak = StrategyCandidate(
        strategy_name="no_continuation_mid",
        confidence="medium",
        snapshot=snapshot,
        side="no",
        price_cents=44,
        contracts=2,
        gbm_probability=0.70,
        gbm_edge=0.03,
    )
    strong = StrategyCandidate(
        strategy_name="no_continuation_mid",
        confidence="medium",
        snapshot=snapshot,
        side="no",
        price_cents=44,
        contracts=2,
        gbm_probability=0.75,
        gbm_edge=0.05,
    )
    reject_summary: dict[str, int] = {}

    filtered = trader._apply_reentry_rules(candidates=[weak, strong], reject_summary=reject_summary)

    assert len(filtered) == 1
    assert filtered[0].gbm_edge == 0.05
    assert filtered[0].contracts == 1
    assert reject_summary["reentry_edge_too_weak"] == 1


def test_opposite_side_after_exit_is_treated_as_fresh_candidate() -> None:
    trader = LiveTrader(
        store=_StubStore(),
        client=_StubClient(),  # type: ignore[arg-type]
        config=LiveTraderConfig(
            enable_signal_break_reentry=True,
            reentry_edge_premium=0.01,
            reentry_min_price_improvement_cents=1,
            gbm_min_points=3,
        ),
    )
    trader.reentry_states["TEST"] = SignalBreakReentryState(
        market_ticker="TEST",
        exited_at=datetime(2026, 4, 6, 12, 0, tzinfo=UTC),
        exit_side="yes",
        exit_strategy_name="yes_continuation_mid",
        exit_reference_price_cents=50,
        exit_execution_price_cents=46,
    )
    snapshot = _snapshot(
        market_ticker="TEST",
        minutes_to_expiry=4.0,
        spot_price=67980.0,
        threshold=68000.0,
        yes_bid=0.55,
        yes_ask=0.56,
    )
    opposite = StrategyCandidate(
        strategy_name="no_continuation_mid",
        confidence="medium",
        snapshot=snapshot,
        side="no",
        price_cents=44,
        contracts=2,
        gbm_probability=0.70,
        gbm_edge=0.03,
    )

    reject_summary: dict[str, int] = {}
    filtered = trader._apply_reentry_rules(candidates=[opposite], reject_summary=reject_summary)

    assert len(filtered) == 1
    assert filtered[0].side == "no"


def test_reentry_fill_increments_cap() -> None:
    trader = LiveTrader(
        store=_StubStore(),
        client=_StubClient(),  # type: ignore[arg-type]
        config=LiveTraderConfig(enable_signal_break_reentry=True, gbm_min_points=3),
    )
    trader.reentry_states["TEST"] = SignalBreakReentryState(
        market_ticker="TEST",
        exited_at=datetime(2026, 4, 6, 12, 0, tzinfo=UTC),
        exit_side="no",
        exit_strategy_name="no_continuation_mid",
        exit_reference_price_cents=50,
        exit_execution_price_cents=47,
    )
    snapshot = _snapshot(
        market_ticker="TEST",
        minutes_to_expiry=4.0,
        spot_price=67980.0,
        threshold=68000.0,
        yes_bid=0.55,
        yes_ask=0.56,
    )
    candidate = StrategyCandidate(
        strategy_name="no_continuation_mid",
        confidence="high",
        snapshot=snapshot,
        side="no",
        price_cents=44,
        contracts=2,
        gbm_probability=0.75,
        gbm_edge=0.05,
    )

    trader._register_successful_reentry(candidate)

    assert trader.reentry_states["TEST"].successful_reentries == 1


def test_submit_order_uses_ioc_micro_retries_until_fill(monkeypatch) -> None:
    class _IocRetryClient(_StubClient):
        def create_order(self, payload: dict) -> dict:
            self.created += 1
            self.create_payloads.append(payload)
            if self.created == 1:
                return {"order": {"order_id": f"order-{self.created}", "status": "canceled", "fill_count_fp": "0.00"}}
            return {"order": {"order_id": f"order-{self.created}", "status": "executed", "fill_count_fp": "2.00"}}

    client = _IocRetryClient()
    trader = LiveTrader(
        store=_StubStore(),
        client=client,  # type: ignore[arg-type]
        config=LiveTraderConfig(
            dry_run=False,
            execution_cross_cents=1,
            ioc_retry_delay_seconds=0.0,
            resting_order_retry_delay_seconds=0.0,
            max_entry_retries=1,
            gbm_min_points=3,
        ),
    )
    monkeypatch.setattr("kabot.trading.live_trader.time.sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(trader, "_refresh_candidate", lambda **kwargs: kwargs["current_candidate"])
    snapshot = _snapshot(
        market_ticker="TEST",
        minutes_to_expiry=4.0,
        spot_price=67980.0,
        threshold=68000.0,
        yes_bid=0.44,
        yes_ask=0.45,
    )
    candidate = StrategyCandidate(
        strategy_name="no_continuation_mid",
        confidence="medium",
        snapshot=snapshot,
        side="no",
        price_cents=56,
        contracts=4,
        gbm_probability=0.7,
        gbm_edge=0.14,
    )

    result = trader._submit_order(candidate)

    assert client.cancelled_order_ids == []
    assert len(client.create_payloads) == 2
    assert client.create_payloads[0]["time_in_force"] == "immediate_or_cancel"
    assert client.create_payloads[1]["time_in_force"] == "immediate_or_cancel"
    assert result["status"] == "executed"
    assert result["filled_contracts"] == 2
    assert len(result["attempts"]) == 2


def test_submit_order_reports_partial_fill_count_from_exchange() -> None:
    class _PartialFillClient(_StubClient):
        def create_order(self, payload: dict) -> dict:
            self.created += 1
            self.create_payloads.append(payload)
            return {"order": {"order_id": f"order-{self.created}", "status": "canceled", "fill_count_fp": "1.00"}}

    client = _PartialFillClient()
    trader = LiveTrader(
        store=_StubStore(),
        client=client,  # type: ignore[arg-type]
        config=LiveTraderConfig(
            dry_run=False,
            execution_cross_cents=1,
            ioc_retry_delay_seconds=0.0,
            resting_order_retry_delay_seconds=0.0,
            max_entry_retries=0,
            gbm_min_points=3,
        ),
    )
    snapshot = _snapshot(
        market_ticker="TEST",
        minutes_to_expiry=4.0,
        spot_price=67980.0,
        threshold=68000.0,
        yes_bid=0.44,
        yes_ask=0.45,
    )
    candidate = StrategyCandidate(
        strategy_name="no_continuation_mid",
        confidence="medium",
        snapshot=snapshot,
        side="no",
        price_cents=56,
        contracts=4,
        gbm_probability=0.7,
        gbm_edge=0.14,
    )

    result = trader._submit_order(candidate)

    assert len(client.create_payloads) == 1
    assert result["filled_contracts"] == 1


def test_submit_order_uses_exchange_fill_records_when_present() -> None:
    class _ExchangeFillClient(_StubClient):
        def create_order(self, payload: dict) -> dict:
            self.created += 1
            self.create_payloads.append(payload)
            return {"order": {"order_id": f"order-{self.created}", "status": "canceled", "fill_count_fp": "0.00"}}

        def get_fills(self, *, order_id: str | None = None, ticker: str | None = None, limit: int = 200) -> dict:
            return {"fills": [{"count": 2, "price": 56}]}

    client = _ExchangeFillClient()
    trader = LiveTrader(
        store=_StubStore(),
        client=client,  # type: ignore[arg-type]
        config=LiveTraderConfig(
            dry_run=False,
            execution_cross_cents=1,
            ioc_retry_delay_seconds=0.0,
            resting_order_retry_delay_seconds=0.0,
            max_entry_retries=0,
            gbm_min_points=3,
        ),
    )
    snapshot = _snapshot(
        market_ticker="TEST",
        minutes_to_expiry=4.0,
        spot_price=67980.0,
        threshold=68000.0,
        yes_bid=0.44,
        yes_ask=0.45,
    )
    candidate = StrategyCandidate(
        strategy_name="no_continuation_mid",
        confidence="medium",
        snapshot=snapshot,
        side="no",
        price_cents=56,
        contracts=2,
        gbm_probability=0.7,
        gbm_edge=0.14,
    )

    result = trader._submit_order(candidate)

    assert result["filled_contracts"] == 2
    assert result["exchange_filled_contracts"] == 2


def test_submit_order_skips_when_orderbook_has_no_depth() -> None:
    class _NoDepthClient(_StubClient):
        def get_orderbook(self, ticker: str) -> dict:
            return {
                "orderbook": {
                    "yes": {"asks": [{"price": 45, "count": 10}]},
                    "no": {"asks": [{"price": 70, "count": 1}]},
                }
            }

    client = _NoDepthClient()
    trader = LiveTrader(
        store=_StubStore(),
        client=client,  # type: ignore[arg-type]
        config=LiveTraderConfig(
            dry_run=False,
            execution_cross_cents=1,
            ioc_retry_delay_seconds=0.0,
            resting_order_retry_delay_seconds=0.0,
            max_entry_retries=0,
            gbm_min_points=3,
            min_orderbook_fill_fraction=1.0,
        ),
    )
    snapshot = _snapshot(
        market_ticker="TEST",
        minutes_to_expiry=4.0,
        spot_price=67980.0,
        threshold=68000.0,
        yes_bid=0.44,
        yes_ask=0.45,
    )
    candidate = StrategyCandidate(
        strategy_name="no_continuation_mid",
        confidence="medium",
        snapshot=snapshot,
        side="no",
        price_cents=56,
        contracts=2,
        gbm_probability=0.7,
        gbm_edge=0.14,
    )

    result = trader._submit_order(candidate)

    assert client.create_payloads == []
    assert result["status"] == "skipped_no_depth"
    assert result["orderbook_available_contracts"] == 0


def test_execution_session_can_submit_without_orderbook_precheck() -> None:
    class _NoOrderbookClient(_StubClient):
        def get_orderbook(self, ticker: str) -> dict:
            raise AssertionError("orderbook precheck should be disabled")

        def create_order(self, payload: dict) -> dict:
            self.created += 1
            self.create_payloads.append(payload)
            return {"order": {"order_id": f"order-{self.created}", "status": "executed", "fill_count_fp": "1.00"}}

    client = _NoOrderbookClient()
    trader = LiveTrader(
        store=_StubStore(),
        client=client,  # type: ignore[arg-type]
        config=LiveTraderConfig(
            dry_run=False,
            enable_execution_sessions=True,
            use_orderbook_precheck=False,
            execution_cross_cents=1,
            execution_session_attempts=1,
            execution_session_retry_delay_seconds=0.0,
            gbm_min_points=3,
        ),
    )
    snapshot = _snapshot(
        market_ticker="TEST",
        minutes_to_expiry=4.0,
        spot_price=68020.0,
        threshold=68000.0,
        yes_bid=0.44,
        yes_ask=0.45,
    )
    candidate = StrategyCandidate(
        strategy_name="yes_continuation_mid",
        confidence="medium",
        snapshot=snapshot,
        side="yes",
        price_cents=45,
        contracts=1,
        gbm_probability=0.7,
        gbm_edge=0.25,
    )

    result = trader._submit_order(candidate)

    assert len(client.create_payloads) == 1
    assert result["status"] == "executed"
    assert result["filled_contracts"] == 1
    assert result["orderbook_available_contracts"] is None


def test_execution_session_registers_local_resting_lock_for_gtc_entry() -> None:
    class _RestingGtcClient(_StubClient):
        def create_order(self, payload: dict) -> dict:
            self.created += 1
            self.create_payloads.append(payload)
            return {"order": {"order_id": f"order-{self.created}", "status": "resting", "fill_count_fp": "0.00"}}

    client = _RestingGtcClient()
    trader = LiveTrader(
        store=_StubStore(),
        client=client,  # type: ignore[arg-type]
        config=LiveTraderConfig(
            dry_run=False,
            enable_execution_sessions=True,
            use_orderbook_precheck=False,
            entry_time_in_force="good_till_canceled",
            execution_cross_cents=1,
            execution_session_attempts=1,
            max_contracts_per_order=4,
            gbm_min_points=3,
        ),
    )
    snapshot = _snapshot(
        market_ticker="TEST",
        minutes_to_expiry=4.0,
        spot_price=68020.0,
        threshold=68000.0,
        yes_bid=0.44,
        yes_ask=0.45,
    )
    candidate = StrategyCandidate(
        strategy_name="yes_continuation_mid",
        confidence="high",
        snapshot=snapshot,
        side="yes",
        price_cents=45,
        contracts=4,
        gbm_probability=0.7,
        gbm_edge=0.25,
    )

    result = trader._submit_order(candidate)

    assert len(client.create_payloads) == 1
    assert client.create_payloads[0]["count"] == 4
    assert result["status"] == "resting"
    assert "TEST" in trader.local_resting_entry_locks


def test_local_resting_entry_lock_expires_without_exchange_confirmation() -> None:
    trader = LiveTrader(
        store=_StubStore(),
        client=_StubClient(),  # type: ignore[arg-type]
        config=LiveTraderConfig(
            local_resting_entry_lock_seconds=5.0,
            gbm_min_points=3,
        ),
    )
    observed_at = datetime(2026, 4, 6, 12, 0, tzinfo=UTC)
    trader.local_resting_entry_locks["TEST"] = LocalRestingEntryLock(
        created_at=observed_at - timedelta(seconds=10)
    )

    active = trader._active_local_resting_entry_tickers(
        observed_at=observed_at,
        resting_order_tickers=set(),
        account_position_contracts={},
        local_position_contracts={},
    )

    assert active == set()
    assert "TEST" not in trader.local_resting_entry_locks


def test_execution_state_drops_stale_ws_quotes() -> None:
    store = ExecutionStateStore(series_ticker="KXBTC15M")
    observed_at = datetime(2026, 4, 6, 12, 0, tzinfo=UTC)
    store.update_metadata(
        raw_markets=[
            {
                "ticker": "TEST",
                "series_ticker": "KXBTC15M",
                "market_type": "binary",
                "status": "open",
                "result": None,
                "floor_strike": "68000",
                "expiration_time": "2026-04-06T12:10:00Z",
            }
        ],
        observed_at=observed_at,
    )
    snapshot = store.build_snapshot(
        ticker="TEST",
        ws_prices={
            "yes_bid": 0.44,
            "yes_ask": 0.45,
            "no_bid": 0.55,
            "no_ask": 0.56,
            "_updated_at": observed_at - timedelta(seconds=20),
        },
        spot_price=68010.0,
        observed_at=observed_at,
        max_quote_age_seconds=10.0,
    )
    assert snapshot is not None
    assert snapshot.yes_ask is None
    assert snapshot.no_ask is None


def test_failed_entry_attempts_start_backoff() -> None:
    trader = LiveTrader(
        store=_StubStore(),
        client=_StubClient(),  # type: ignore[arg-type]
        config=LiveTraderConfig(
            failed_entry_backoff_after_attempts=3,
            failed_entry_backoff_seconds=30.0,
            gbm_min_points=3,
        ),
    )
    ticker = "TEST"
    trader._record_failed_entry_attempt(ticker)
    trader._record_failed_entry_attempt(ticker)
    assert ticker not in trader.failed_entry_blocked_until
    trader._record_failed_entry_attempt(ticker)
    assert ticker in trader.failed_entry_blocked_until
    assert ticker in trader._blocked_failed_entry_tickers(datetime.now(UTC))


def test_submit_order_cancels_without_retry_if_setup_breaks(monkeypatch) -> None:
    class _BadRefreshClient(_StubClient):
        def get_market(self, ticker: str) -> dict:
            return {
                "market": {
                    "ticker": ticker,
                    "series_ticker": "KXBTC15M",
                    "result": None,
                    "status": "open",
                    "market_type": "binary",
                    "yes_bid_dollars": "0.20",
                    "yes_ask_dollars": "0.22",
                    "no_bid_dollars": "0.78",
                    "no_ask_dollars": "0.80",
                    "floor_strike": "68000",
                    "expiration_time": "2026-04-06T12:04:00Z",
                }
            }

    client = _BadRefreshClient()
    trader = LiveTrader(
        store=_StubStore(),
        client=client,  # type: ignore[arg-type]
        config=LiveTraderConfig(
            dry_run=False,
            execution_cross_cents=1,
            resting_order_retry_delay_seconds=0.0,
            max_entry_retries=1,
            gbm_min_points=3,
            entry_time_in_force="good_till_canceled",
        ),
    )
    monkeypatch.setattr("kabot.trading.live_trader.time.sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(trader, "_refresh_candidate", lambda **_kwargs: None)
    snapshot = _snapshot(
        market_ticker="TEST",
        minutes_to_expiry=4.0,
        spot_price=67980.0,
        threshold=68000.0,
        yes_bid=0.44,
        yes_ask=0.45,
    )
    candidate = StrategyCandidate(
        strategy_name="no_continuation_mid",
        confidence="medium",
        snapshot=snapshot,
        side="no",
        price_cents=56,
        contracts=4,
        gbm_probability=0.7,
        gbm_edge=0.14,
    )

    result = trader._submit_order(candidate)

    assert client.cancelled_order_ids == ["order-1"]
    assert len(client.create_payloads) == 1
    assert result["status"] == "canceled_not_retried"


def test_submit_order_ioc_stops_retrying_if_setup_breaks(monkeypatch) -> None:
    class _IocBreakClient(_StubClient):
        def create_order(self, payload: dict) -> dict:
            self.created += 1
            self.create_payloads.append(payload)
            return {"order": {"order_id": f"order-{self.created}", "status": "canceled", "fill_count_fp": "0.00"}}

    client = _IocBreakClient()
    trader = LiveTrader(
        store=_StubStore(),
        client=client,  # type: ignore[arg-type]
        config=LiveTraderConfig(
            dry_run=False,
            execution_cross_cents=1,
            ioc_retry_delay_seconds=0.0,
            resting_order_retry_delay_seconds=0.0,
            max_entry_retries=1,
            gbm_min_points=3,
            entry_time_in_force="immediate_or_cancel",
        ),
    )
    monkeypatch.setattr("kabot.trading.live_trader.time.sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(trader, "_refresh_candidate", lambda **_kwargs: None)
    snapshot = _snapshot(
        market_ticker="TEST",
        minutes_to_expiry=4.0,
        spot_price=67980.0,
        threshold=68000.0,
        yes_bid=0.44,
        yes_ask=0.45,
    )
    candidate = StrategyCandidate(
        strategy_name="no_continuation_mid",
        confidence="medium",
        snapshot=snapshot,
        side="no",
        price_cents=56,
        contracts=2,
        gbm_probability=0.7,
        gbm_edge=0.14,
    )

    result = trader._submit_order(candidate)

    assert len(client.create_payloads) == 1
    assert result["status"] == "ioc_not_retried"

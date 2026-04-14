from __future__ import annotations

import base64
import json
import math
import os
import time
from collections import deque
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from kabot.compat import UTC
from kabot.markets.normalize import normalize_market
from kabot.models.gbm_threshold import GBMThresholdModel, probability_for_snapshot
from kabot.models.latency_repricing import LatencyRepricingModel
from kabot.storage.postgres import PostgresStore
from kabot.trading.daily_strategies import (
    DailyExitConfig,
    DailySignal,
    DailySignalConfig,
    evaluate_daily_exit,
    evaluate_daily_signal,
    evaluate_daily_signal_debug,
)
from kabot.trading.daily_vol import bootstrap_hourly_vol, estimate_hourly_vol_from_db
from kabot.trading.execution_state import ExecutionStateStore
from kabot.trading.execution_trace import ExecutionTraceWriter
from kabot.trading.new_strategies import (
    NewSignal,
    StrategyAConfig,
    StrategyBConfig,
    evaluate_strategy_a,
    evaluate_strategy_b,
)
from kabot.trading.velocity import BTCVelocityDetector
from kabot.trading.ws_feeds import CoinbaseSpotFeed, KalshiFillFeed, KalshiTickerFeed
from kabot.types import MarketSnapshot, Position


def _probability_to_cents(value: float | None) -> int | None:
    if value is None:
        return None
    return int(round(max(0.0, min(1.0, float(value))) * 100.0))


def _time_to_expiry_minutes(snapshot: MarketSnapshot) -> float:
    return max((snapshot.expiry - snapshot.observed_at).total_seconds() / 60.0, 0.0)


def _yes_spread_cents(snapshot: MarketSnapshot) -> int | None:
    ask = _probability_to_cents(snapshot.yes_ask)
    bid = _probability_to_cents(snapshot.yes_bid)
    if ask is None or bid is None:
        return None
    return ask - bid


@dataclass(frozen=True)
class StrategyCandidate:
    strategy_name: str
    confidence: str
    snapshot: MarketSnapshot
    side: str
    price_cents: int
    contracts: int
    gbm_probability: float | None = None
    gbm_edge: float | None = None
    time_in_force: str | None = None


@dataclass(frozen=True)
class StrategyRule:
    name: str
    side: str
    min_tte_minutes: float
    max_tte_minutes: float
    min_price_cents: int
    max_price_cents: int
    max_spread_cents: int


STRATEGY_RULES: tuple[StrategyRule, ...] = (
    StrategyRule(
        name="yes_continuation_mid",
        side="yes",
        min_tte_minutes=0.0,
        max_tte_minutes=10.0,
        min_price_cents=38,
        max_price_cents=62,
        max_spread_cents=6,
    ),
    StrategyRule(
        name="no_continuation_mid",
        side="no",
        min_tte_minutes=0.0,
        max_tte_minutes=10.0,
        min_price_cents=38,
        max_price_cents=62,
        max_spread_cents=6,
    ),
    StrategyRule(
        name="yes_continuation_wide",
        side="yes",
        min_tte_minutes=0.0,
        max_tte_minutes=10.0,
        min_price_cents=38,
        max_price_cents=62,
        max_spread_cents=8,
    ),
    StrategyRule(
        name="no_continuation_wide",
        side="no",
        min_tte_minutes=0.0,
        max_tte_minutes=10.0,
        min_price_cents=38,
        max_price_cents=62,
        max_spread_cents=8,
    ),
)

SOFT_ENTRY_CAP_CENTS = 57
HARD_ENTRY_CAP_CENTS = 62
EXPENSIVE_ENTRY_EDGE_PREMIUM = 0.02


def _strategy_rules_with_max_tte(max_tte_minutes: float) -> tuple[StrategyRule, ...]:
    return tuple(replace(rule, max_tte_minutes=max_tte_minutes) for rule in STRATEGY_RULES)


@dataclass(frozen=True)
class ClosedTrade:
    market_ticker: str
    strategy_name: str
    closed_at: datetime
    realized_pnl_cents: int


@dataclass
class SignalBreakReentryState:
    market_ticker: str
    exited_at: datetime
    exit_side: str
    exit_strategy_name: str
    exit_reference_price_cents: int
    exit_execution_price_cents: int
    successful_reentries: int = 0


@dataclass
class PendingSignalBreakState:
    reason: str
    count: int = 1


@dataclass
class LocalRestingEntryLock:
    created_at: datetime
    side: str = ""
    strategy_name: str = ""
    price_cents: int | None = None
    gbm_edge: float | None = None
    gbm_probability: float | None = None
    observed_at: datetime | None = None
    contracts: int | None = None


def _side_prices(snapshot: MarketSnapshot, side: str) -> tuple[int | None, int | None]:
    if side == "yes":
        return _probability_to_cents(snapshot.yes_ask), _probability_to_cents(snapshot.yes_bid)
    return _probability_to_cents(snapshot.no_ask), _probability_to_cents(snapshot.no_bid)


def _increment_reason(counts: dict[str, int], reason: str) -> None:
    counts[reason] = counts.get(reason, 0) + 1


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _confidence_for_candidate(
    *,
    rule_name: str,
    tte_minutes: float,
    ask_cents: int,
    spread_cents: int,
) -> str:
    if rule_name in {"yes_continuation_mid", "no_continuation_mid"}:
        if spread_cents <= 2 and tte_minutes <= 3.0 and 45 <= ask_cents <= 55:
            return "high"
        return "medium"
    return "low"


def _effective_contracts_for_price(*, ask_cents: int, contracts: int) -> int:
    if ask_cents > SOFT_ENTRY_CAP_CENTS:
        return min(contracts, 1)
    return contracts


def _contracts_for_candidate(*, confidence: str, price_cents: int, available_balance_cents: int | None) -> int:
    target_contracts = 2 if confidence == "high" else 1
    if available_balance_cents is None:
        return target_contracts
    affordable = max(available_balance_cents // max(price_cents, 1), 0)
    return max(0, min(target_contracts, affordable))


def _dynamic_contracts_for_candidate(
    *,
    price_cents: int,
    current_market_contracts: int,
    bankroll_cents: int | None,
    available_balance_cents: int | None,
    deployed_cents: int,
    per_market_fraction: float,
    max_total_deployed_fraction: float,
    max_notional_per_market_cents: int,
    max_contracts_per_order: int,
    max_contracts_per_market: int,
) -> int:
    if price_cents <= 0:
        return 0
    if available_balance_cents is None:
        affordable = 1
    else:
        affordable = max(available_balance_cents // price_cents, 0)
    if affordable <= 0:
        return 0
    if bankroll_cents is None:
        target_notional_cents = price_cents
        remaining_deployable_cents = price_cents
    else:
        bankroll_cents = max(int(bankroll_cents), 0)
        target_notional_cents = min(
            int(bankroll_cents * max(per_market_fraction, 0.0)),
            max_notional_per_market_cents,
        )
        max_total_deployed_cents = int(bankroll_cents * max(max_total_deployed_fraction, 0.0))
        remaining_deployable_cents = max(max_total_deployed_cents - max(deployed_cents, 0), 0)
    effective_budget_cents = min(target_notional_cents, remaining_deployable_cents)
    if effective_budget_cents < price_cents:
        return 0
    contracts = effective_budget_cents // price_cents
    remaining_market_capacity = max(max_contracts_per_market - max(current_market_contracts, 0), 0)
    return max(0, min(int(contracts), affordable, max_contracts_per_order, remaining_market_capacity))


def select_entry_candidates(
    snapshots: list[MarketSnapshot],
    *,
    blocked_markets: set[str],
    current_position_contracts: dict[str, int],
    current_position_strategies: dict[str, str],
    active_strategy_counts: dict[str, int],
    open_market_count: int,
    max_open_markets: int,
    max_strategy_open_counts: dict[str, int],
    available_balance_cents: int | None = None,
    bankroll_cents: int | None = None,
    deployed_cents: int = 0,
    distance_threshold_dollars: float = 0.0,
    per_market_fraction: float = 0.08,
    max_total_deployed_fraction: float = 0.24,
    max_notional_per_market_cents: int = 500,
    max_contracts_per_order: int = 2,
    max_contracts_per_market: int = 4,
    min_market_volume: float = 0.0,
    strategy_rules: tuple[StrategyRule, ...] = STRATEGY_RULES,
) -> list[StrategyCandidate]:
    if open_market_count >= max_open_markets and not current_position_contracts:
        return []

    candidates: list[StrategyCandidate] = []
    allocated_cents = 0
    projected_open_market_count = open_market_count
    for snapshot in sorted(snapshots, key=lambda item: (item.expiry, item.market_ticker)):
        if snapshot.market_ticker in blocked_markets:
            continue
        if snapshot.contract_type != "threshold":
            continue
        if snapshot.threshold is None:
            continue
        if (snapshot.volume or 0.0) < min_market_volume:
            continue
        tte_minutes = _time_to_expiry_minutes(snapshot)
        existing_contracts = int(current_position_contracts.get(snapshot.market_ticker, 0))
        existing_strategy = current_position_strategies.get(snapshot.market_ticker)
        if existing_contracts <= 0 and projected_open_market_count >= max_open_markets:
            continue
        for rule in strategy_rules:
            current_strategy_count = int(active_strategy_counts.get(rule.name, 0))
            strategy_cap = int(max_strategy_open_counts.get(rule.name, max_open_markets))
            if existing_contracts > 0 and existing_strategy not in ("", None, rule.name):
                continue
            if existing_contracts <= 0 and current_strategy_count >= strategy_cap:
                continue
            if not (rule.min_tte_minutes < tte_minutes < rule.max_tte_minutes):
                continue
            if rule.side == "yes" and snapshot.spot_price < (snapshot.threshold + distance_threshold_dollars):
                continue
            if rule.side == "no" and snapshot.spot_price > (snapshot.threshold - distance_threshold_dollars):
                continue
            ask_cents, bid_cents = _side_prices(snapshot, rule.side)
            if ask_cents is None or bid_cents is None:
                continue
            spread_cents = ask_cents - bid_cents
            if not (rule.min_price_cents <= ask_cents <= rule.max_price_cents):
                continue
            if spread_cents >= rule.max_spread_cents:
                continue
            confidence = _confidence_for_candidate(
                rule_name=rule.name,
                tte_minutes=tte_minutes,
                ask_cents=ask_cents,
                spread_cents=spread_cents,
            )
            contracts = _dynamic_contracts_for_candidate(
                price_cents=ask_cents,
                current_market_contracts=existing_contracts,
                bankroll_cents=bankroll_cents,
                available_balance_cents=available_balance_cents,
                deployed_cents=deployed_cents + allocated_cents,
                per_market_fraction=per_market_fraction,
                max_total_deployed_fraction=max_total_deployed_fraction,
                max_notional_per_market_cents=max_notional_per_market_cents,
                max_contracts_per_order=max_contracts_per_order,
                max_contracts_per_market=max_contracts_per_market,
            )
            contracts = _effective_contracts_for_price(ask_cents=ask_cents, contracts=contracts)
            if contracts <= 0:
                continue
            candidates.append(
                StrategyCandidate(
                    strategy_name=rule.name,
                    confidence=confidence,
                    snapshot=snapshot,
                    side=rule.side,
                    price_cents=ask_cents,
                    contracts=contracts,
                )
            )
            allocated_cents += contracts * ask_cents
            blocked_markets.add(snapshot.market_ticker)
            current_position_contracts[snapshot.market_ticker] = existing_contracts + contracts
            current_position_strategies[snapshot.market_ticker] = rule.name
            if existing_contracts <= 0:
                active_strategy_counts[rule.name] = current_strategy_count + 1
                projected_open_market_count += 1
            break
    return candidates


def summarize_rejections(
    snapshots: list[MarketSnapshot],
    *,
    blocked_markets: set[str],
    blocked_reason_sets: dict[str, set[str]] | None = None,
    active_strategy_counts: dict[str, int],
    open_market_count: int,
    max_open_markets: int,
    max_strategy_open_counts: dict[str, int],
    distance_threshold_dollars: float = 0.0,
    min_market_volume: float = 0.0,
    strategy_rules: tuple[StrategyRule, ...] = STRATEGY_RULES,
) -> dict[str, int]:
    reason_counts: dict[str, int] = {}
    if open_market_count >= max_open_markets:
        return {"max_open_markets_reached": len(snapshots)}

    for snapshot in sorted(snapshots, key=lambda item: (item.expiry, item.market_ticker)):
        if snapshot.market_ticker in blocked_markets:
            blocked_reason = "market_already_active"
            if blocked_reason_sets:
                for reason, tickers in blocked_reason_sets.items():
                    if snapshot.market_ticker in tickers:
                        blocked_reason = reason
                        break
            _increment_reason(reason_counts, blocked_reason)
            continue
        if snapshot.contract_type != "threshold":
            _increment_reason(reason_counts, "unsupported_contract_type")
            continue
        if snapshot.threshold is None:
            _increment_reason(reason_counts, "missing_threshold")
            continue
        if (snapshot.volume or 0.0) < min_market_volume:
            _increment_reason(reason_counts, "volume_below_threshold")
            continue

        tte_minutes = _time_to_expiry_minutes(snapshot)
        matched = False
        saw_strategy_cap = False
        saw_tte_failure = False
        saw_direction_failure = False
        saw_missing_price = False
        saw_price_band_failure = False
        saw_spread_failure = False

        for rule in strategy_rules:
            current_strategy_count = int(active_strategy_counts.get(rule.name, 0))
            strategy_cap = int(max_strategy_open_counts.get(rule.name, max_open_markets))
            if current_strategy_count >= strategy_cap:
                saw_strategy_cap = True
                continue
            if not (rule.min_tte_minutes < tte_minutes < rule.max_tte_minutes):
                saw_tte_failure = True
                continue
            if rule.side == "yes" and snapshot.spot_price < (snapshot.threshold + distance_threshold_dollars):
                saw_direction_failure = True
                continue
            if rule.side == "no" and snapshot.spot_price > (snapshot.threshold - distance_threshold_dollars):
                saw_direction_failure = True
                continue
            ask_cents, bid_cents = _side_prices(snapshot, rule.side)
            if ask_cents is None or bid_cents is None:
                saw_missing_price = True
                continue
            spread_cents = ask_cents - bid_cents
            if not (rule.min_price_cents <= ask_cents <= rule.max_price_cents):
                saw_price_band_failure = True
                continue
            if spread_cents >= rule.max_spread_cents:
                saw_spread_failure = True
                continue
            matched = True
            break

        if matched:
            continue
        if saw_spread_failure:
            _increment_reason(reason_counts, "spread_too_wide")
        elif saw_price_band_failure:
            _increment_reason(reason_counts, "price_out_of_band")
        elif saw_missing_price:
            _increment_reason(reason_counts, "missing_prices")
        elif saw_direction_failure:
            _increment_reason(reason_counts, "spot_threshold_direction_mismatch")
        elif saw_tte_failure:
            _increment_reason(reason_counts, "tte_out_of_window")
        elif saw_strategy_cap:
            _increment_reason(reason_counts, "strategy_cap_reached")
        else:
            _increment_reason(reason_counts, "no_matching_rule")

    return reason_counts


@dataclass(frozen=True)
class LiveTraderConfig:
    active_profile: str = "baseline_live"
    series_ticker: str = "KXBTC15M"
    daily_vol_floor: float = 0.40
    daily_min_edge: float = 0.04
    daily_min_price_cents: int = 40
    daily_max_price_cents: int = 95
    daily_min_tte_minutes: float = 8.0
    daily_max_tte_minutes: float = 50.0
    daily_min_distance_dollars: float = 200.0
    daily_max_spread_cents: int = 6
    daily_min_volume: float = 0.0
    daily_stop_loss_cents: int = 15
    daily_fair_value_buffer_cents: int = 3
    daily_negative_edge_threshold: float = -0.04
    daily_min_tte_to_exit_minutes: float = 8.0
    daily_max_open_markets: int = 3
    daily_contracts_per_trade: int = 2
    poll_seconds: int = 10
    max_open_markets: int = 3
    contracts_per_trade: int = 1
    dry_run: bool = False
    daily_loss_stop_cents: int = 1000
    max_trades_per_day: int = 0
    max_strategy_open_counts: dict[str, int] | None = None
    cooldown_loss_streak: int = 3
    cooldown_minutes: int = 30
    max_spot_age_seconds: int = 30
    max_market_age_seconds: int = 30
    min_market_volume: float = 5000.0
    distance_threshold_dollars: float = 10.0
    gbm_min_edge_mid: float = 0.02
    gbm_min_edge_wide_yes: float = 0.04
    gbm_lookback_minutes: int = 90
    gbm_min_points: int = 20
    gbm_volatility_floor: float = 0.05
    position_fraction_per_market: float = 0.08
    max_total_deployed_fraction: float = 0.24
    max_notional_per_market_cents: int = 500
    max_contracts_per_order: int = 2
    max_contracts_per_market: int = 4
    entry_max_tte_minutes: float = 10.0
    enable_signal_break_exit: bool = False
    enable_execution_sessions: bool = False
    enable_signal_break_reentry: bool = False
    use_orderbook_precheck: bool = True
    execution_cross_cents: int = 6
    exit_cross_cents: int = 4
    execution_session_attempts: int = 3
    execution_session_retry_delay_seconds: float = 0.05
    execution_ladder_steps_cents: tuple[int, ...] | None = None
    execution_ladder_steps_cents_high: tuple[int, ...] | None = None
    high_conviction_edge_threshold: float = 0.0
    high_conviction_distance_threshold_dollars: float = 0.0
    enforce_positive_edge_on_execution: bool = False
    execution_min_edge_margin_low: float = 0.0
    execution_min_edge_margin_high: float = 0.0
    execution_session_use_rest_fallback: bool = True
    execution_spread_tight_cents: int = 2
    execution_spread_wide_cents: int = 6
    execution_spread_insane_cents: int = 12
    execution_spread_tight_min_cross_cents: int = 2
    execution_spread_wide_max_cross_cents: int = 1
    execution_ladder_delay_high_seconds: float = 5.0
    execution_ladder_delay_mid_seconds: float = 8.0
    execution_ladder_delay_low_seconds: float = 12.0
    fast_market_tte_minutes: float = 5.0
    fast_market_delay_ceiling_seconds: float = 6.0
    resting_entry_max_age_seconds: float = 45.0
    resting_entry_max_age_seconds_high: float = 90.0
    resting_entry_max_age_seconds_fast: float = 25.0
    resting_entry_max_age_seconds_low_depth: float = 25.0
    fast_cancel_tte_minutes: float = 3.0
    edge_decay_cancel_threshold: float = 0.02
    execution_trace_path: str | None = "data/execution_trace.jsonl"
    execution_heartbeat_seconds: float = 2.0
    event_loop_sleep_seconds: float = 0.1
    reentry_edge_premium: float = 0.02
    reentry_max_per_market: int = 3
    reentry_min_price_improvement_cents: int = 3
    exit_distance_threshold_dollars: float = 5.0
    signal_break_confirmation_cycles: int = 2
    exit_negative_edge_threshold: float = -0.01
    price_stop_cents: int = 0
    price_stop_grace_seconds: int = 0
    price_stop_confirm_cycles: int = 1
    hard_stop_cents: int = 0
    max_ws_quote_age_seconds: float = 10.0
    failed_entry_backoff_seconds: float = 30.0
    failed_entry_backoff_after_attempts: int = 3
    failed_entry_decay_seconds: float = 15.0
    local_resting_entry_lock_seconds: float = 2.0
    settlement_timeout_seconds: float = 300.0
    min_orderbook_fill_fraction: float = 0.5
    resting_order_retry_delay_seconds: float = 2.0
    ioc_retry_delay_seconds: float = 0.15
    max_entry_retries: int = 1
    entry_time_in_force: str = "immediate_or_cancel"
    exit_time_in_force: str = "immediate_or_cancel"
    hybrid_resting_entry_enabled: bool = False
    hybrid_resting_entry_seconds: float = 5.0
    metadata_refresh_seconds: int = 60
    whipsaw_crossing_window_minutes: float = 30.0
    whipsaw_max_crossings: int = 999999
    overnight_edge_multiplier: float = 1.0
    overnight_hours_utc: tuple[int, int] = (0, 7)


@dataclass(frozen=True)
class KalshiAuthConfig:
    api_key_id: str
    private_key_path: str


class KalshiAuthSigner:
    def __init__(self, config: KalshiAuthConfig) -> None:
        self.config = config
        self._private_key = self._load_private_key(config.private_key_path)

    @staticmethod
    def _load_private_key(path: str):
        private_key_bytes = Path(path).expanduser().read_bytes()
        return serialization.load_pem_private_key(private_key_bytes, password=None)

    @staticmethod
    def _timestamp_ms() -> str:
        return str(int(datetime.now(UTC).timestamp() * 1000))

    def _sign(self, *, timestamp: str, method: str, path: str) -> str:
        message = f"{timestamp}{method.upper()}{path}".encode("utf-8")
        signature = self._private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        return base64.b64encode(signature).decode("utf-8")

    def request_headers(self, *, method: str, path: str) -> dict[str, str]:
        normalized_path = path.split("?", 1)[0]
        if not normalized_path.startswith("/trade-api/"):
            normalized_path = f"/trade-api/v2{normalized_path if normalized_path.startswith('/') else f'/{normalized_path}'}"
        timestamp = self._timestamp_ms()
        signature = self._sign(timestamp=timestamp, method=method, path=normalized_path)
        return {
            "KALSHI-ACCESS-KEY": self.config.api_key_id,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
            "KALSHI-ACCESS-SIGNATURE": signature,
        }


@dataclass
class KabotKalshiClient:
    base_url: str = "https://api.elections.kalshi.com/trade-api/v2"
    timeout_seconds: int = 10
    auth_signer: KalshiAuthSigner | None = None

    def __post_init__(self) -> None:
        self.session = requests.Session()
        self.session.trust_env = False

    def list_markets(self, *, series_ticker: str, status: str = "open", limit: int = 200) -> list[dict[str, Any]]:
        markets: list[dict[str, Any]] = []
        cursor: str | None = None
        while True:
            params: dict[str, Any] = {
                "series_ticker": series_ticker,
                "status": status,
                "limit": limit,
            }
            if cursor:
                params["cursor"] = cursor
            response = self.session.get(f"{self.base_url}/markets", params=params, timeout=self.timeout_seconds)
            response.raise_for_status()
            payload = response.json()
            markets.extend(payload.get("markets", []))
            cursor = payload.get("cursor") or None
            if not cursor:
                break
        return markets

    def get_positions(self) -> dict[str, Any]:
        return self._request_json("GET", "/portfolio/positions")

    def list_orders(self, *, status: str | None = None, limit: int = 200) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": limit}
        if status:
            params["status"] = status
        return self._request_json("GET", "/portfolio/orders", params=params)

    def create_order(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request_json("POST", "/portfolio/orders", json_payload=payload)

    def get_order(self, order_id: str) -> dict[str, Any]:
        return self._request_json("GET", f"/portfolio/orders/{order_id}")

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        return self._request_json("DELETE", f"/portfolio/orders/{order_id}")

    def get_balance(self) -> dict[str, Any]:
        return self._request_json("GET", "/portfolio/balance")

    def get_market(self, ticker: str) -> dict[str, Any]:
        response = self.session.get(f"{self.base_url}/markets/{ticker}", timeout=self.timeout_seconds)
        response.raise_for_status()
        return response.json()

    def get_orderbook(self, ticker: str) -> dict[str, Any]:
        response = self.session.get(f"{self.base_url}/markets/{ticker}/orderbook", timeout=self.timeout_seconds)
        response.raise_for_status()
        return response.json()

    def get_fills(
        self,
        *,
        order_id: str | None = None,
        ticker: str | None = None,
        limit: int = 200,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": limit}
        if order_id:
            params["order_id"] = order_id
        if ticker:
            params["ticker"] = ticker
        return self._request_json("GET", "/portfolio/fills", params=params)

    def fetch_spot_price(self, product_id: str = "BTC-USD") -> tuple[float, datetime]:
        response = self.session.get(
            f"https://api.exchange.coinbase.com/products/{product_id}/ticker",
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        observed_at = datetime.now(UTC)
        time_raw = payload.get("time")
        if time_raw:
            try:
                observed_at = datetime.fromisoformat(str(time_raw).replace("Z", "+00:00")).astimezone(UTC)
            except ValueError:
                observed_at = datetime.now(UTC)
        return float(payload["price"]), observed_at

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        headers: dict[str, str] = {}
        if self.auth_signer is not None:
            headers.update(self.auth_signer.request_headers(method=method, path=path))
        response = self.session.request(
            method=method.upper(),
            url=f"{self.base_url}{path}",
            params=params,
            json=json_payload,
            headers=headers,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return response.json()


def _extract_position_count(position: dict[str, Any]) -> float:
    raw_position = position.get("position")
    if raw_position in (None, ""):
        raw_position = position.get("position_fp")
    if raw_position in (None, ""):
        raw_position = position.get("count")
    if raw_position in (None, ""):
        raw_position = position.get("count_fp")
    try:
        return float(raw_position or 0)
    except (TypeError, ValueError):
        return 0.0


def _extract_fill_count(order: dict[str, Any]) -> int:
    # Kalshi v2: filled quantity = count - remaining_count
    count_raw = order.get("count")
    remaining_raw = order.get("remaining_count")
    if count_raw not in (None, "") and remaining_raw not in (None, ""):
        try:
            return max(int(round(float(count_raw))) - int(round(float(remaining_raw))), 0)
        except (TypeError, ValueError):
            pass
    for key in ("fill_count_fp", "fill_count", "count_fp", "count"):
        raw = order.get(key)
        if raw in (None, ""):
            continue
        try:
            return int(round(float(raw)))
        except (TypeError, ValueError):
            continue
    return 0


def _extract_positions(payload: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("market_positions", "positions", "data"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _extract_orders(payload: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("orders", "data"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _extract_fills(payload: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("fills", "data"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _extract_order_status(order: dict[str, Any]) -> str:
    for key in ("status", "order_status"):
        value = order.get(key)
        if value not in (None, ""):
            return str(value).lower()
    return ""


def _extract_order_id(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    order = payload.get("order")
    if isinstance(order, dict):
        return str(order.get("order_id") or order.get("id") or "") or None
    return str(payload.get("order_id") or payload.get("id") or "") or None


def _extract_market_ticker(payload: dict[str, Any]) -> str:
    return str(payload.get("market_ticker", payload.get("ticker", "")) or "")


def _extract_price_cents(payload: dict[str, Any]) -> int | None:
    for key in ("price", "price_cents", "yes_price", "no_price"):
        raw = payload.get(key)
        if raw in (None, ""):
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if value <= 1.0:
            return int(round(value * 100.0))
        return int(round(value))
    return None


def _extract_quantity(payload: dict[str, Any]) -> int:
    for key in ("count", "count_fp", "quantity", "quantity_fp", "size", "size_fp"):
        raw = payload.get(key)
        if raw in (None, ""):
            continue
        try:
            return int(round(float(raw)))
        except (TypeError, ValueError):
            continue
    return 0


def _coerce_fp_orderbook_levels(raw_levels: Any) -> list[dict[str, Any]] | None:
    if not isinstance(raw_levels, list):
        return None
    levels: list[dict[str, Any]] = []
    for item in raw_levels:
        if isinstance(item, dict):
            levels.append(item)
            continue
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            levels.append({"price": item[0], "count": item[1]})
            continue
    return levels


def _side_order_levels(payload: dict[str, Any], side: str) -> list[dict[str, Any]]:
    book = payload.get("orderbook", payload) if isinstance(payload, dict) else {}
    if isinstance(payload, dict):
        fp_book = payload.get("orderbook_fp")
        if isinstance(fp_book, dict):
            fp_levels = _coerce_fp_orderbook_levels(fp_book.get(f"{side}_dollars"))
            if fp_levels is not None:
                return fp_levels
    candidate_keys = (
        [side, f"{side}_book", f"{side}_orderbook", f"{side}_asks", f"{side}_levels"]
        if side in {"yes", "no"}
        else []
    )
    for key in candidate_keys:
        value = book.get(key)
        if isinstance(value, dict):
            for nested_key in ("asks", "sell_orders", "levels", "orders"):
                nested = value.get(nested_key)
                if isinstance(nested, list):
                    return [item for item in nested if isinstance(item, dict)]
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    for nested_key in ("asks", "sell_orders", "orders"):
        nested = book.get(nested_key)
        if isinstance(nested, dict):
            side_levels = nested.get(side)
            if isinstance(side_levels, list):
                return [item for item in side_levels if isinstance(item, dict)]
    return []


def _has_explicit_orderbook(payload: dict[str, Any], side: str) -> bool:
    if not isinstance(payload, dict):
        return False
    fp_book = payload.get("orderbook_fp")
    if isinstance(fp_book, dict) and f"{side}_dollars" in fp_book:
        return True
    book = payload.get("orderbook", payload)
    if isinstance(book, dict):
        if side in book:
            return True
        for key in (f"{side}_book", f"{side}_orderbook", f"{side}_asks", f"{side}_levels"):
            if key in book:
                return True
        for nested_key in ("asks", "sell_orders", "orders"):
            nested = book.get(nested_key)
            if isinstance(nested, dict) and side in nested:
                return True
    return False


def _available_contracts_at_price(payload: dict[str, Any], *, side: str, limit_price_cents: int) -> int | None:
    fp_book = payload.get("orderbook_fp") if isinstance(payload, dict) else None
    if isinstance(fp_book, dict):
        # Kalshi orderbook_fp exposes only bids. For entry fillability we need
        # the reciprocal side because:
        # - YES buy fills against NO bids at prices >= (100 - yes_limit)
        # - NO buy fills against YES bids at prices >= (100 - no_limit)
        reciprocal_side = "no" if side == "yes" else "yes"
        reciprocal_levels = _coerce_fp_orderbook_levels(fp_book.get(f"{reciprocal_side}_dollars"))
        if reciprocal_levels is not None:
            if not reciprocal_levels:
                return 0
            reciprocal_min_bid_cents = max(100 - int(limit_price_cents), 0)
            total = 0
            usable_level_found = False
            for level in reciprocal_levels:
                price_cents = _extract_price_cents(level)
                quantity = _extract_quantity(level)
                if price_cents is None or quantity <= 0:
                    continue
                usable_level_found = True
                if price_cents >= reciprocal_min_bid_cents:
                    total += quantity
            if not usable_level_found:
                return None
            return total

    levels = _side_order_levels(payload, side)
    if not levels:
        if _has_explicit_orderbook(payload, side):
            return 0
        return None
    total = 0
    usable_level_found = False
    for level in levels:
        price_cents = _extract_price_cents(level)
        quantity = _extract_quantity(level)
        if price_cents is None or quantity <= 0:
            continue
        usable_level_found = True
        if price_cents <= limit_price_cents:
            total += quantity
    if not usable_level_found:
        return None
    return total


def _filled_contracts_from_fills(payload: dict[str, Any]) -> int | None:
    fills = _extract_fills(payload)
    if not fills:
        return None
    total = 0
    saw_any = False
    for fill in fills:
        quantity = _extract_quantity(fill)
        if quantity <= 0:
            continue
        total += quantity
        saw_any = True
    return total if saw_any else 0


class LiveTrader:
    def __init__(
        self,
        *,
        store: PostgresStore,
        client: KabotKalshiClient,
        config: LiveTraderConfig,
    ) -> None:
        self.store = store
        self.client = client
        self.config = config
        self.model = LatencyRepricingModel(
            volatility_floor=config.gbm_volatility_floor,
            persistence_factor=0.75,
            min_move_bps=3.0,
        )
        self.local_positions: dict[str, Position] = {}
        self.position_strategies: dict[str, str] = {}
        self.reentry_states: dict[str, SignalBreakReentryState] = {}
        self.pending_signal_breaks: dict[str, PendingSignalBreakState] = {}
        self.signal_break_blocked_tickers: set[str] = set()
        self.local_resting_entry_locks: dict[str, LocalRestingEntryLock] = {}
        self.failed_entry_attempts: dict[str, int] = {}
        self.failed_entry_blocked_until: dict[str, datetime] = {}
        self.failed_entry_last_attempt: dict[str, datetime] = {}
        self.failed_entry_last_price: dict[str, int] = {}
        self.failed_entry_last_edge: dict[str, float] = {}
        self.settlement_pending_since: dict[str, datetime] = {}
        self._prev_spot_price: float | None = None
        self._spot_price_history: deque[tuple[datetime, float]] = deque()
        self._loss_streak_path = Path("data/loss_streak.json")
        self.closed_trades: list[ClosedTrade] = []
        self.submitted_trade_times: list[datetime] = []
        self.consecutive_losses = 0
        self.cooldown_until: datetime | None = None
        self._load_loss_streak_state()
        self._velocity_detector = BTCVelocityDetector(window_seconds=30.0, min_points=3)
        self._bootstrapped_vol: float | None = None
        self._daily_vol: float | None = None
        self._daily_positions: dict[str, Position] = {}
        self._daily_position_fair_values: dict[str, int] = {}
        self._daily_closed_trades: list[ClosedTrade] = []
        self._last_daily_signal_debug_at: datetime | None = None
        # WebSocket feeds — started in run_forever()
        self.spot_feed = CoinbaseSpotFeed()
        auth_signer = getattr(client, "auth_signer", None)
        self.ticker_feed = KalshiTickerFeed(auth_signer) if auth_signer is not None else None
        self.fill_feed = KalshiFillFeed(auth_signer) if auth_signer is not None else None
        self.state_store = ExecutionStateStore(series_ticker=config.series_ticker)
        # Full REST market dicts keyed by ticker (metadata: expiry, threshold, etc.)
        self._market_metadata: dict[str, dict[str, Any]] = {}
        self._last_metadata_refresh: datetime | None = None
        self.trace_writer = ExecutionTraceWriter(self.config.execution_trace_path)

    def _strategy_rules(self) -> tuple[StrategyRule, ...]:
        return _strategy_rules_with_max_tte(self.config.entry_max_tte_minutes)

    def _recover_orphaned_positions(self) -> None:
        """On startup, sync any open exchange positions into local state so they track through settlement."""
        try:
            account_contracts = self._account_position_contracts()
        except Exception:
            return
        if not account_contracts:
            return
        observed_at = datetime.now(UTC)
        try:
            positions_payload = self.client.get_positions()
            positions_list = _extract_positions(positions_payload)
        except Exception:
            return
        positions_by_ticker: dict[str, dict[str, Any]] = {}
        for pos in positions_list:
            ticker = _extract_market_ticker(pos)
            if ticker:
                positions_by_ticker[ticker] = pos
        for ticker, contracts in account_contracts.items():
            if contracts <= 0 or ticker in self.local_positions:
                continue
            metadata = self._market_metadata.get(ticker)
            if metadata is None:
                continue
            pos = positions_by_ticker.get(ticker)
            if pos is None:
                continue
            count = _extract_position_count(pos)
            if count == 0:
                continue
            side: str = "yes" if count > 0 else "no"
            expiry_raw = metadata.get("close_time") or metadata.get("expiry") or metadata.get("expected_expiration_time")
            if expiry_raw is None:
                continue
            try:
                expiry = datetime.fromisoformat(str(expiry_raw).replace("Z", "+00:00")).astimezone(UTC)
            except Exception:
                continue
            self.local_positions[ticker] = Position(
                position_id=f"recovered-{ticker}",
                market_ticker=ticker,
                side=side,  # type: ignore[arg-type]
                contracts=int(abs(count)),
                entry_time=observed_at,
                entry_price_cents=50,
                expiry=expiry,
            )
            self.position_strategies[ticker] = "recovered"
            self._trace_execution_event({
                "event": "position_recovered_on_startup",
                "market_ticker": ticker,
                "side": side,
                "contracts": int(abs(count)),
                "expiry": expiry,
            })

    def run_forever(self) -> None:
        self._refresh_market_metadata()
        self._recover_orphaned_positions()
        if self.config.active_profile in {"NEW", "GOD"}:
            self._bootstrap_vol_from_coinbase()
        if self.config.active_profile == "DAILY":
            self._bootstrap_daily_vol()
        self.spot_feed.start()
        if self.ticker_feed is not None:
            self.ticker_feed.start()
        if self.fill_feed is not None:
            self.fill_feed.start()
        last_ticker_revision = self.ticker_feed.revision() if self.ticker_feed is not None else 0
        last_spot_revision = self.spot_feed.revision()
        last_heartbeat = datetime.min.replace(tzinfo=UTC)
        while True:
            now = datetime.now(UTC)
            if (
                self._last_metadata_refresh is None
                or (now - self._last_metadata_refresh).total_seconds() >= self.config.metadata_refresh_seconds
            ):
                self._refresh_market_metadata()
            should_run = False
            if self.config.enable_execution_sessions:
                current_ticker_revision = self.ticker_feed.revision() if self.ticker_feed is not None else 0
                current_spot_revision = self.spot_feed.revision()
                if current_ticker_revision > last_ticker_revision or current_spot_revision > last_spot_revision:
                    should_run = True
                if (now - last_heartbeat).total_seconds() >= max(float(self.config.execution_heartbeat_seconds), 0.1):
                    should_run = True
                if should_run:
                    last_ticker_revision = current_ticker_revision
                    last_spot_revision = current_spot_revision
                    last_heartbeat = now
                    result = self.run_once()
                    print(self._format_cycle_log(result), flush=True)
                time.sleep(max(float(self.config.event_loop_sleep_seconds), 0.01))
                continue
            result = self.run_once()
            print(self._format_cycle_log(result), flush=True)
            time.sleep(self.config.poll_seconds)

    def run_once(self) -> dict[str, Any]:
        observed_at = datetime.now(UTC)
        if self.config.active_profile == "DAILY" and self._daily_vol is None:
            self._bootstrap_daily_vol()
        closed_now = self._reconcile_settled_positions(observed_at)
        ws_spot, ws_spot_ts = self.spot_feed.get()
        if ws_spot is not None and ws_spot_ts is not None:
            spot_price, spot_timestamp = ws_spot, ws_spot_ts
        else:
            spot_price, spot_timestamp = self.client.fetch_spot_price()
        if (observed_at - spot_timestamp).total_seconds() > self.config.max_spot_age_seconds:
            return {
                "observed_at": observed_at.isoformat(),
                "active_profile": self.config.active_profile,
                "series_ticker": self.config.series_ticker,
                "status": "paused_stale_spot",
                "spot_age_seconds": (observed_at - spot_timestamp).total_seconds(),
                "closed_positions": [closed.__dict__ for closed in closed_now],
                "reject_summary": {"stale_spot": 1},
            }
        self._append_spot_price_history(observed_at=observed_at, spot_price=spot_price)
        if self.config.active_profile == "NEW":
            self._velocity_detector.update(observed_at, spot_price)

        if self.cooldown_until is not None and observed_at < self.cooldown_until:
            return {
                "observed_at": observed_at.isoformat(),
                "active_profile": self.config.active_profile,
                "series_ticker": self.config.series_ticker,
                "status": "cooldown",
                "cooldown_until": self.cooldown_until.isoformat(),
                "closed_positions": [closed.__dict__ for closed in closed_now],
                "reject_summary": {"cooldown": 1},
            }

        if self._daily_realized_pnl_cents(observed_at.date()) <= -abs(self.config.daily_loss_stop_cents):
            return {
                "observed_at": observed_at.isoformat(),
                "active_profile": self.config.active_profile,
                "series_ticker": self.config.series_ticker,
                "status": "daily_loss_stop",
                "daily_realized_pnl_cents": self._daily_realized_pnl_cents(observed_at.date()),
                "closed_positions": [closed.__dict__ for closed in closed_now],
                "reject_summary": {"daily_loss_stop": 1},
            }

        if self.config.max_trades_per_day > 0 and self._daily_trade_count(observed_at.date()) >= self.config.max_trades_per_day:
            return {
                "observed_at": observed_at.isoformat(),
                "active_profile": self.config.active_profile,
                "series_ticker": self.config.series_ticker,
                "status": "max_trades_reached",
                "daily_trade_count": self._daily_trade_count(observed_at.date()),
                "closed_positions": [closed.__dict__ for closed in closed_now],
                "reject_summary": {"max_trades_reached": 1},
            }

        snapshots = self._build_snapshots(spot_price=spot_price, observed_at=observed_at)
        self.store.insert_market_snapshots(snapshots)

        account_position_contracts = self._account_position_contracts()
        resting_order_tickers = self._resting_buy_market_tickers()
        self._sync_local_positions(
            account_position_contracts=account_position_contracts,
            resting_order_tickers=resting_order_tickers,
            observed_at=observed_at,
        )
        local_position_contracts = {
            ticker: position.contracts
            for ticker, position in self.local_positions.items()
            if position.status == "open"
        }
        external_position_tickers = {
            ticker for ticker, contracts in account_position_contracts.items() if contracts > 0 and ticker not in local_position_contracts
        }
        permanently_blocked_tickers = set(self.signal_break_blocked_tickers)
        local_open_tickers = set(local_position_contracts)
        local_resting_entry_tickers = self._active_local_resting_entry_tickers(
            observed_at=observed_at,
            resting_order_tickers=set(resting_order_tickers),
            account_position_contracts=account_position_contracts,
            local_position_contracts=local_position_contracts,
        )
        resting_buy_tickers = set(resting_order_tickers) | local_resting_entry_tickers
        open_tickers = local_open_tickers | external_position_tickers | resting_buy_tickers
        active_strategy_counts = self._active_strategy_counts()
        available_balance_cents: int | None = None
        try:
            balance_payload = self.client.get_balance()
            available_balance_cents = _safe_int(balance_payload.get("balance"))
        except Exception:
            available_balance_cents = None
        deployed_cents = self._local_open_notional_cents()
        bankroll_cents = None
        if available_balance_cents is not None:
            bankroll_cents = available_balance_cents + deployed_cents
        reject_summary: dict[str, int] = {}
        sample_market: dict[str, Any] | None = None
        if snapshots:
            future_snapshots = [snapshot for snapshot in snapshots if snapshot.expiry > observed_at]
            chosen_snapshot = min(
                future_snapshots or snapshots,
                key=lambda snapshot: max((snapshot.expiry - observed_at).total_seconds(), 0.0),
            )
            sample_market = self._sample_market_view(snapshot=chosen_snapshot)
        fresh_snapshots = list(snapshots)
        volatility = None
        if self.config.active_profile != "DAILY":
            volatility = self._estimate_live_volatility(observed_at=observed_at)
        if self.config.active_profile == "NEW":
            if volatility is None:
                volatility = self._bootstrapped_vol or self.config.gbm_volatility_floor
            new_orders = self._run_new_profile_cycle(
                snapshots=fresh_snapshots,
                volatility=volatility,
                observed_at=observed_at,
                reject_summary=reject_summary,
            )
            velocity = self._velocity_detector.reading()
            return {
                "observed_at": observed_at.isoformat(),
                "active_profile": self.config.active_profile,
                "series_ticker": self.config.series_ticker,
                "spot_price": spot_price,
                "spot_timestamp": spot_timestamp.isoformat(),
                "snapshots_seen": len(snapshots),
                "fresh_snapshots_seen": len(fresh_snapshots),
                "active_markets": len(self.local_positions),
                "candidates": len(new_orders),
                "orders_placed": len([order for order in new_orders if order.get("filled_contracts", 0) > 0]),
                "placed_orders": new_orders,
                "available_balance_cents": available_balance_cents,
                "estimated_bankroll_cents": bankroll_cents,
                "local_deployed_cents": deployed_cents,
                "velocity_bps_per_second": round(velocity.bps_per_second, 2),
                "velocity_move_bps": round(velocity.move_bps, 2),
                "gbm_volatility": volatility,
                "daily_realized_pnl_cents": self._daily_realized_pnl_cents(observed_at.date()),
                "daily_trade_count": self._daily_trade_count(observed_at.date()),
                "closed_positions": [closed.__dict__ for closed in closed_now],
                "reject_summary": reject_summary,
                "sample_market": sample_market,
                "dry_run": self.config.dry_run,
            }
        if self.config.active_profile == "DAILY":
            daily_orders = self._run_daily_profile_cycle(
                snapshots=fresh_snapshots,
                observed_at=observed_at,
            )
            daily_pnl_cents = sum(
                trade.realized_pnl_cents
                for trade in self._daily_closed_trades
                if trade.closed_at.date() == observed_at.date()
            )
            return {
                "observed_at": observed_at.isoformat(),
                "active_profile": self.config.active_profile,
                "series_ticker": self.config.series_ticker,
                "spot_price": spot_price,
                "spot_timestamp": spot_timestamp.isoformat(),
                "snapshots_seen": len(snapshots),
                "fresh_snapshots_seen": len(fresh_snapshots),
                "active_markets": len([
                    position
                    for position in self._daily_positions.values()
                    if position.status == "open"
                ]),
                "candidates": len(daily_orders),
                "orders_placed": len([
                    order
                    for order in daily_orders
                    if order.get("filled_contracts", 0) > 0
                ]),
                "placed_orders": daily_orders,
                "daily_vol": self._estimate_daily_vol(observed_at=observed_at),
                "daily_realized_pnl_cents": daily_pnl_cents,
                "closed_positions": [
                    trade.__dict__
                    for trade in self._daily_closed_trades
                    if trade.closed_at.date() == observed_at.date()
                ],
                "dry_run": self.config.dry_run,
            }
        strategy_rules = self._strategy_rules()
        signal_break_orders = self._reconcile_signal_break_positions(
            observed_at=observed_at,
            snapshots=fresh_snapshots,
            volatility=volatility,
        )
        self._cancel_stale_resting_entry_orders(
            observed_at=observed_at,
            snapshots=fresh_snapshots,
            volatility=volatility,
        )
        if signal_break_orders:
            account_position_contracts = self._account_position_contracts()
            resting_order_tickers = self._resting_buy_market_tickers()
            self._sync_local_positions(
                account_position_contracts=account_position_contracts,
                resting_order_tickers=resting_order_tickers,
                observed_at=observed_at,
            )
            local_position_contracts = {
                ticker: position.contracts
                for ticker, position in self.local_positions.items()
                if position.status == "open"
            }
            external_position_tickers = {
                ticker for ticker, contracts in account_position_contracts.items() if contracts > 0 and ticker not in local_position_contracts
            }
            local_resting_entry_tickers = self._active_local_resting_entry_tickers(
                observed_at=observed_at,
                resting_order_tickers=set(resting_order_tickers),
                account_position_contracts=account_position_contracts,
                local_position_contracts=local_position_contracts,
            )
            open_tickers = set(local_position_contracts) | external_position_tickers | set(resting_order_tickers) | local_resting_entry_tickers
            active_strategy_counts = self._active_strategy_counts()
            permanently_blocked_tickers = set(self.signal_break_blocked_tickers)
        candidates = select_entry_candidates(
            fresh_snapshots,
            blocked_markets=set(external_position_tickers)
            | set(resting_order_tickers)
            | local_resting_entry_tickers
            | permanently_blocked_tickers,
            current_position_contracts=dict(local_position_contracts),
            current_position_strategies=dict(self.position_strategies),
            active_strategy_counts=active_strategy_counts,
            open_market_count=len(open_tickers),
            max_open_markets=self.config.max_open_markets,
            max_strategy_open_counts=self.config.max_strategy_open_counts or {},
            available_balance_cents=available_balance_cents,
            bankroll_cents=bankroll_cents,
            deployed_cents=deployed_cents,
            distance_threshold_dollars=self.config.distance_threshold_dollars,
            per_market_fraction=self.config.position_fraction_per_market,
            max_total_deployed_fraction=self.config.max_total_deployed_fraction,
            max_notional_per_market_cents=self.config.max_notional_per_market_cents,
            max_contracts_per_order=self.config.max_contracts_per_order,
            max_contracts_per_market=self.config.max_contracts_per_market,
            min_market_volume=self.config.min_market_volume,
            strategy_rules=strategy_rules,
        )
        whipsaw_filtered: list[StrategyCandidate] = []
        for candidate in candidates:
            threshold = candidate.snapshot.threshold
            if threshold is not None and self._count_threshold_crossings(
                threshold=float(threshold),
                window_minutes=float(self.config.whipsaw_crossing_window_minutes),
            ) > int(self.config.whipsaw_max_crossings):
                _increment_reason(reject_summary, "whipsaw_skip")
                continue
            whipsaw_filtered.append(candidate)
        candidates = whipsaw_filtered
        gbm_filtered: list[StrategyCandidate] = []
        for candidate in candidates:
            if volatility is None:
                _increment_reason(reject_summary, "gbm_vol_unavailable")
                continue
            estimate = self.model.estimate(candidate.snapshot, volatility=volatility)
            market_probability = candidate.price_cents / 100.0
            side_probability = estimate.probability if candidate.side == "yes" else (1.0 - estimate.probability)
            edge = side_probability - market_probability
            min_edge = self._required_gbm_edge(
                strategy_name=candidate.strategy_name,
                ask_cents=candidate.price_cents,
            )
            if edge < min_edge:
                _increment_reason(reject_summary, "gbm_edge_below_threshold")
                continue
            if self.config.active_profile == "GOD" and side_probability >= 0.99 and candidate.price_cents < 95:
                _increment_reason(reject_summary, "impossible_edge_rejected")
                self._trace_execution_event(
                    {
                        "event": "impossible_edge_rejected",
                        "market_ticker": candidate.snapshot.market_ticker,
                        "side": candidate.side,
                        "strategy_name": candidate.strategy_name,
                        "price_cents": candidate.price_cents,
                        "gbm_probability": side_probability,
                        "gbm_edge": edge,
                        "threshold": candidate.snapshot.threshold,
                        "spot_price": candidate.snapshot.spot_price,
                    }
                )
                continue
            gbm_filtered.append(
                StrategyCandidate(
                    strategy_name=candidate.strategy_name,
                    confidence=candidate.confidence,
                    snapshot=candidate.snapshot,
                    side=candidate.side,
                    price_cents=candidate.price_cents,
                    contracts=candidate.contracts,
                    gbm_probability=side_probability,
                    gbm_edge=edge,
                )
            )
        candidates = self._apply_reentry_rules(candidates=gbm_filtered, reject_summary=reject_summary)
        failed_entry_blocked_tickers = self._blocked_failed_entry_tickers(observed_at)
        backoff_blocked_count = 0
        execution_candidates: list[StrategyCandidate] = []
        for candidate in candidates:
            if candidate.snapshot.market_ticker in failed_entry_blocked_tickers:
                if not self._allow_backoff_reattempt(candidate):
                    backoff_blocked_count += 1
                    continue
            execution_candidates.append(candidate)
        selection_rejections = summarize_rejections(
            fresh_snapshots,
            blocked_markets=set(open_tickers) | permanently_blocked_tickers,
            blocked_reason_sets={
                "has_local_position": local_open_tickers,
                "has_external_position": set(external_position_tickers),
                "has_resting_order": resting_buy_tickers,
                "signal_break_locked": permanently_blocked_tickers,
            },
            active_strategy_counts=dict(active_strategy_counts),
            open_market_count=len(open_tickers),
            max_open_markets=self.config.max_open_markets,
            max_strategy_open_counts=self.config.max_strategy_open_counts or {},
            distance_threshold_dollars=self.config.distance_threshold_dollars,
            min_market_volume=self.config.min_market_volume,
            strategy_rules=strategy_rules,
        )
        for reason, count in selection_rejections.items():
            reject_summary[reason] = reject_summary.get(reason, 0) + count
        if backoff_blocked_count:
            reject_summary["blocked_by_backoff"] = reject_summary.get("blocked_by_backoff", 0) + backoff_blocked_count

        placed: list[dict[str, Any]] = []
        for candidate in execution_candidates:
            order = self._submit_order(candidate)
            placed.append(order)
            filled_contracts = max(int(order.get("filled_contracts", 0) or 0), 0)
            if filled_contracts > 0:
                self._clear_failed_entry_attempts(candidate.snapshot.market_ticker)
                self.local_resting_entry_locks.pop(candidate.snapshot.market_ticker, None)
                self._register_successful_reentry(candidate)
                self._trace_execution_event(
                    {
                        "event": "entry_filled",
                        "market_ticker": candidate.snapshot.market_ticker,
                        "side": candidate.side,
                        "strategy_name": candidate.strategy_name,
                        "price_cents": candidate.price_cents,
                        "filled_contracts": filled_contracts,
                        "gbm_edge": candidate.gbm_edge,
                        "gbm_probability": candidate.gbm_probability,
                        "observed_at": candidate.snapshot.observed_at,
                        "fill_edge": (
                            float(candidate.gbm_probability) - (candidate.price_cents / 100.0)
                            if candidate.gbm_probability is not None
                            else None
                        ),
                    }
                )
                existing = self.local_positions.get(candidate.snapshot.market_ticker)
                if existing is None:
                    self.local_positions[candidate.snapshot.market_ticker] = Position(
                        position_id=str(uuid4()),
                        market_ticker=candidate.snapshot.market_ticker,
                        side=candidate.side,  # type: ignore[arg-type]
                        contracts=filled_contracts,
                        entry_time=candidate.snapshot.observed_at,
                        entry_price_cents=candidate.price_cents,
                        expiry=candidate.snapshot.expiry,
                    )
                else:
                    total_contracts = existing.contracts + filled_contracts
                    weighted_entry = int(
                        round(
                            (
                                existing.entry_price_cents * existing.contracts
                                + candidate.price_cents * filled_contracts
                            )
                            / max(total_contracts, 1)
                        )
                    )
                    existing.contracts = total_contracts
                    existing.entry_price_cents = weighted_entry
                    existing.expiry = candidate.snapshot.expiry
                self.position_strategies[candidate.snapshot.market_ticker] = candidate.strategy_name
            else:
                order_status = str(order.get("status") or "")
                if self._should_record_failed_entry(order_status=order_status, response=order.get("response")):
                    self._record_failed_entry_attempt(
                        candidate.snapshot.market_ticker,
                        price_cents=candidate.price_cents,
                        gbm_edge=candidate.gbm_edge,
                    )
            self.submitted_trade_times.append(observed_at)

        return {
            "observed_at": observed_at.isoformat(),
            "active_profile": self.config.active_profile,
            "series_ticker": self.config.series_ticker,
            "spot_price": spot_price,
            "spot_timestamp": spot_timestamp.isoformat(),
            "snapshots_seen": len(snapshots),
            "fresh_snapshots_seen": len(fresh_snapshots),
            "active_markets": len(open_tickers),
            "candidates": len(candidates),
            "orders_placed": len(placed),
            "placed_orders": placed,
            "signal_break_orders": signal_break_orders,
            "available_balance_cents": available_balance_cents,
            "estimated_bankroll_cents": bankroll_cents,
            "local_deployed_cents": deployed_cents,
            "gbm_volatility": volatility,
            "daily_realized_pnl_cents": self._daily_realized_pnl_cents(observed_at.date()),
            "daily_trade_count": self._daily_trade_count(observed_at.date()),
            "closed_positions": [closed.__dict__ for closed in closed_now],
            "reject_summary": reject_summary,
            "sample_market": sample_market,
            "dry_run": self.config.dry_run,
        }

    @staticmethod
    def _format_cycle_log(result: dict[str, Any]) -> str:
        observed_at = str(result.get("observed_at", ""))
        status = str(result.get("status", "ok"))
        active_profile = str(result.get("active_profile", "baseline_live"))
        series = str(result.get("series_ticker", ""))
        spot_price = result.get("spot_price")
        if isinstance(spot_price, (int, float)):
            spot_text = f"{float(spot_price):.2f}"
        else:
            spot_text = "na"
        snapshots_seen = int(result.get("snapshots_seen", 0) or 0)
        fresh_seen = int(result.get("fresh_snapshots_seen", 0) or 0)
        active_markets = int(result.get("active_markets", 0) or 0)
        candidates = int(result.get("candidates", 0) or 0)
        orders_placed = int(result.get("orders_placed", 0) or 0)
        available_balance_cents = _safe_int(result.get("available_balance_cents"))
        estimated_bankroll_cents = _safe_int(result.get("estimated_bankroll_cents"))
        local_deployed_cents = _safe_int(result.get("local_deployed_cents"))
        gbm_volatility = result.get("gbm_volatility")
        daily_pnl = int(result.get("daily_realized_pnl_cents", 0) or 0)
        daily_trades = int(result.get("daily_trade_count", 0) or 0)
        signal_break_orders = result.get("signal_break_orders")
        signal_exit_count = len(signal_break_orders) if isinstance(signal_break_orders, list) else 0
        reject_summary = result.get("reject_summary")
        reject_text = ""
        if isinstance(reject_summary, dict) and reject_summary:
            reject_text = " rejects=" + ",".join(
                f"{key}:{value}" for key, value in sorted(reject_summary.items())
            )
        sample_market = result.get("sample_market")
        sample_text = ""
        if isinstance(sample_market, dict) and sample_market:
            sample_bits = []
            for key in ("market_ticker", "volume", "open_interest", "yes_bid_c", "yes_ask_c", "no_bid_c", "no_ask_c", "threshold", "spot_price", "tte_s"):
                if key in sample_market:
                    sample_bits.append(f"{key}={sample_market[key]}")
            if sample_bits:
                sample_text = " sample[" + " ".join(sample_bits) + "]"
        order_text = ""
        placed_orders = result.get("placed_orders")
        if isinstance(placed_orders, list) and placed_orders:
            latest_order = placed_orders[-1]
            if isinstance(latest_order, dict):
                order_bits = []
                for key in (
                    "side",
                    "status",
                    "strategy",
                    "strategy_name",
                    "confidence",
                    "gbm_edge",
                    "signal_price_cents",
                    "execution_price_cents",
                    "filled_contracts",
                    "exchange_filled_contracts",
                    "orderbook_available_contracts",
                ):
                    if key in latest_order:
                        order_bits.append(f"{key}={latest_order[key]}")
                if order_bits:
                    order_text = " order[" + " ".join(order_bits) + "]"
        return (
            f"{observed_at} profile={active_profile} series={series} status={status} "
            f"spot={spot_text} snapshots={snapshots_seen} fresh={fresh_seen} "
            f"active={active_markets} candidates={candidates} orders={orders_placed} "
            f"signal_exits={signal_exit_count} "
            f"balance_cents={available_balance_cents if available_balance_cents is not None else 'na'} "
            f"bankroll_cents={estimated_bankroll_cents if estimated_bankroll_cents is not None else 'na'} "
            f"deployed_cents={local_deployed_cents if local_deployed_cents is not None else 'na'} "
            f"gbm_vol={round(float(gbm_volatility), 4) if isinstance(gbm_volatility, (int, float)) else 'na'} "
            f"day_pnl_cents={daily_pnl} day_trades={daily_trades}{reject_text}{sample_text}{order_text}"
        )

    def _refresh_market_metadata(self) -> None:
        """Fetch current open markets via REST and subscribe new tickers to the WS feed."""
        observed_at = datetime.now(UTC)
        try:
            raw_markets = self.client.list_markets(series_ticker=self.config.series_ticker, status="open")
        except Exception:
            return
        self.state_store.update_metadata(raw_markets=raw_markets, observed_at=observed_at)
        live_tickers: set[str] = set()
        new_tickers: list[str] = []
        for raw in raw_markets:
            ticker = str(raw.get("ticker") or raw.get("market_ticker") or "")
            if not ticker:
                continue
            live_tickers.add(ticker)
            self._market_metadata[ticker] = {**raw, "series_ticker": self.config.series_ticker}
            if ticker not in (self.ticker_feed._subscribed if self.ticker_feed is not None else set()):
                new_tickers.append(ticker)
        for ticker in list(self._market_metadata):
            if ticker not in live_tickers:
                self._market_metadata.pop(ticker, None)
                self.reentry_states.pop(ticker, None)
                self.pending_signal_breaks.pop(ticker, None)
                self.signal_break_blocked_tickers.discard(ticker)
                self.local_resting_entry_locks.pop(ticker, None)
                self.failed_entry_attempts.pop(ticker, None)
                self.failed_entry_blocked_until.pop(ticker, None)
                self.settlement_pending_since.pop(ticker, None)
        if new_tickers and self.ticker_feed is not None:
            self.ticker_feed.subscribe(new_tickers)
        self._last_metadata_refresh = datetime.now(UTC)

    def _build_snapshots(self, *, spot_price: float, observed_at: datetime) -> list[MarketSnapshot]:
        """Merge WS price updates with REST metadata to build MarketSnapshots."""
        recent_log_return = 0.0
        if self._prev_spot_price is not None and self._prev_spot_price > 0 and spot_price > 0:
            recent_log_return = math.log(spot_price / self._prev_spot_price)
        self._prev_spot_price = spot_price
        ws_prices = self.ticker_feed.snapshot() if self.ticker_feed is not None else {}
        snapshots = self.state_store.build_snapshots(
            ws_snapshot=ws_prices,
            spot_price=spot_price,
            observed_at=observed_at,
            max_quote_age_seconds=self.config.max_ws_quote_age_seconds,
        )
        if recent_log_return != 0.0:
            snapshots = [
                replace(s, metadata={**s.metadata, "recent_log_return": recent_log_return})
                for s in snapshots
            ]
        return snapshots

    def _snapshot_for_ticker(self, *, ticker: str, spot_price: float, observed_at: datetime) -> MarketSnapshot | None:
        ws_prices = self.ticker_feed.get_prices(ticker) if self.ticker_feed is not None else None
        return self.state_store.build_snapshot(
            ticker=ticker,
            ws_prices=ws_prices,
            spot_price=spot_price,
            observed_at=observed_at,
            max_quote_age_seconds=self.config.max_ws_quote_age_seconds,
        )

    def _trace_execution_event(self, event: dict[str, Any]) -> None:
        payload = {
            "ts": datetime.now(UTC),
            "profile": self.config.active_profile,
            **event,
        }
        self.trace_writer.write(payload)

    def _blocked_failed_entry_tickers(self, observed_at: datetime) -> set[str]:
        blocked: set[str] = set()
        for ticker, until in list(self.failed_entry_blocked_until.items()):
            if observed_at >= until:
                self.failed_entry_blocked_until.pop(ticker, None)
                self.failed_entry_attempts.pop(ticker, None)
                self.failed_entry_last_attempt.pop(ticker, None)
                self.failed_entry_last_price.pop(ticker, None)
                self.failed_entry_last_edge.pop(ticker, None)
                continue
            blocked.add(ticker)
        return blocked

    @staticmethod
    def _should_record_failed_entry(*, order_status: str, response: dict[str, Any] | None) -> bool:
        status = (order_status or "").strip().lower()
        if status in {"rejected", "error", "failed"}:
            return True
        if response is None:
            return False
        if isinstance(response, dict):
            if response.get("error") or response.get("errors"):
                return True
        return False

    def _allow_backoff_reattempt(self, candidate: StrategyCandidate) -> bool:
        ticker = candidate.snapshot.market_ticker
        last_price = self.failed_entry_last_price.get(ticker)
        last_edge = self.failed_entry_last_edge.get(ticker)
        price_improved = (
            last_price is not None and candidate.price_cents <= max(last_price - 1, 0)
        )
        edge_improved = (
            last_edge is not None
            and candidate.gbm_edge is not None
            and candidate.gbm_edge >= (last_edge + 0.01)
        )
        return bool(price_improved or edge_improved)

    def _active_local_resting_entry_tickers(
        self,
        *,
        observed_at: datetime,
        resting_order_tickers: set[str],
        account_position_contracts: dict[str, int],
        local_position_contracts: dict[str, int],
    ) -> set[str]:
        active: set[str] = set()
        grace_seconds = max(float(self.config.local_resting_entry_lock_seconds), 0.0)
        for ticker, lock in list(self.local_resting_entry_locks.items()):
            if (
                ticker in resting_order_tickers
                or int(account_position_contracts.get(ticker, 0)) > 0
                or int(local_position_contracts.get(ticker, 0)) > 0
            ):
                active.add(ticker)
                continue
            age_seconds = (observed_at - lock.created_at).total_seconds()
            if age_seconds <= grace_seconds:
                active.add(ticker)
                continue
            self.local_resting_entry_locks.pop(ticker, None)
        return active

    def _cancel_stale_resting_entry_orders(
        self,
        *,
        observed_at: datetime,
        snapshots: list[MarketSnapshot],
        volatility: float | None,
    ) -> None:
        """Cancel resting GTC entry orders whose signal has broken since posting."""
        if not self.local_resting_entry_locks:
            return
        snapshots_by_ticker = {s.market_ticker: s for s in snapshots}
        for ticker, lock in list(self.local_resting_entry_locks.items()):
            if not lock.side or not lock.strategy_name:
                continue
            snapshot = snapshots_by_ticker.get(ticker)
            if snapshot is None:
                continue
            if snapshot.threshold is None:
                continue
            cancel_reason: str | None = None
            # Time-to-fill cutoff (unless high conviction)
            age_seconds = (observed_at - lock.created_at).total_seconds()
            distance = snapshot.spot_price - snapshot.threshold if lock.side == "yes" else snapshot.threshold - snapshot.spot_price
            is_high_conviction = False
            if lock.gbm_edge is not None:
                is_high_conviction = (
                    lock.gbm_edge >= float(self.config.high_conviction_edge_threshold)
                    and distance >= float(self.config.high_conviction_distance_threshold_dollars)
                )
            max_age = float(self.config.resting_entry_max_age_seconds_high if is_high_conviction else self.config.resting_entry_max_age_seconds)
            if self.config.hybrid_resting_entry_enabled and self.config.entry_time_in_force == "immediate_or_cancel":
                max_age = min(max_age, float(self.config.hybrid_resting_entry_seconds))
            tte_minutes = max((snapshot.expiry - observed_at).total_seconds() / 60.0, 0.0)
            if tte_minutes <= float(self.config.fast_cancel_tte_minutes):
                max_age = min(max_age, float(self.config.resting_entry_max_age_seconds_fast))
                if max_age > 0 and age_seconds > max_age:
                    cancel_reason = "tte_cutoff_cancel"
            ask_cents, bid_cents = _side_prices(snapshot, lock.side)
            if ask_cents is not None and bid_cents is not None:
                spread_cents = max(ask_cents - bid_cents, 0)
                if spread_cents >= int(self.config.execution_spread_wide_cents):
                    max_age = min(max_age, float(self.config.resting_entry_max_age_seconds_low_depth))
            if cancel_reason is None and max_age > 0 and age_seconds > max_age:
                cancel_reason = "time_cutoff_cancel"
            dist = self.config.distance_threshold_dollars
            # Signal broken: spot no longer satisfies direction requirement
            if cancel_reason is None:
                if lock.side == "yes" and snapshot.spot_price < snapshot.threshold + dist:
                    cancel_reason = "signal_break_cancel"
                elif lock.side == "no" and snapshot.spot_price > snapshot.threshold - dist:
                    cancel_reason = "signal_break_cancel"
            # Also check GBM edge
            if cancel_reason is None and volatility is not None:
                ask_cents, _ = _side_prices(snapshot, lock.side)
                if ask_cents is not None:
                    estimate = self.model.estimate(snapshot, volatility=volatility)
                    side_prob = estimate.probability if lock.side == "yes" else (1.0 - estimate.probability)
                    edge = side_prob - ask_cents / 100.0
                    if edge < self._required_gbm_edge(strategy_name=lock.strategy_name, ask_cents=ask_cents):
                        cancel_reason = "signal_break_cancel"
                    if cancel_reason is None and lock.gbm_edge is not None and edge <= (lock.gbm_edge - float(self.config.edge_decay_cancel_threshold)):
                        cancel_reason = "edge_decay_cancel"
            if cancel_reason is None:
                continue
            # Resolve order_id: fill_feed first, then REST
            order_id: str | None = None
            if self.fill_feed is not None:
                order_id = self.fill_feed.get_resting_order_id(ticker)
            if not order_id:
                try:
                    orders_payload = self.client.list_orders(status="resting")
                    for order in _extract_orders(orders_payload):
                        if _extract_market_ticker(order) == ticker and str(order.get("action", "")).lower() == "buy":
                            order_id = _extract_order_id({"order": order})
                            break
                except Exception:
                    pass
            if order_id:
                try:
                    self.client.cancel_order(order_id)
                    self._trace_execution_event({
                        "event": "resting_entry_canceled",
                        "market_ticker": ticker,
                        "order_id": order_id,
                        "reason": cancel_reason,
                    })
                except Exception:
                    pass
            if self.fill_feed is not None:
                self.fill_feed.deregister_order(ticker)
            self.local_resting_entry_locks.pop(ticker, None)

    def _record_failed_entry_attempt(
        self,
        ticker: str,
        *,
        price_cents: int | None = None,
        gbm_edge: float | None = None,
    ) -> None:
        now = datetime.now(UTC)
        count = self.failed_entry_attempts.get(ticker, 0)
        last_attempt = self.failed_entry_last_attempt.get(ticker)
        decay_seconds = float(self.config.failed_entry_decay_seconds)
        if last_attempt is not None and decay_seconds > 0:
            age = (now - last_attempt).total_seconds()
            if age >= decay_seconds:
                count = max(count - 1, 0)
        count += 1
        self.failed_entry_attempts[ticker] = count
        self.failed_entry_last_attempt[ticker] = now
        if price_cents is not None:
            self.failed_entry_last_price[ticker] = int(price_cents)
        if gbm_edge is not None:
            self.failed_entry_last_edge[ticker] = float(gbm_edge)
        threshold = max(int(self.config.failed_entry_backoff_after_attempts), 1)
        if count >= threshold:
            self.failed_entry_blocked_until[ticker] = now + timedelta(
                seconds=max(float(self.config.failed_entry_backoff_seconds), 0.0)
            )
            self._trace_execution_event(
                {
                    "event": "failed_entry_backoff_started",
                    "market_ticker": ticker,
                    "failed_entry_attempts": count,
                    "blocked_until": self.failed_entry_blocked_until[ticker],
                }
            )

    def _clear_failed_entry_attempts(self, ticker: str) -> None:
        self.failed_entry_attempts.pop(ticker, None)
        self.failed_entry_blocked_until.pop(ticker, None)
        self.failed_entry_last_attempt.pop(ticker, None)
        self.failed_entry_last_price.pop(ticker, None)
        self.failed_entry_last_edge.pop(ticker, None)

    def _apply_reentry_rules(
        self,
        *,
        candidates: list[StrategyCandidate],
        reject_summary: dict[str, int],
    ) -> list[StrategyCandidate]:
        if not self.config.enable_signal_break_reentry:
            return candidates
        filtered: list[StrategyCandidate] = []
        for candidate in candidates:
            state = self.reentry_states.get(candidate.snapshot.market_ticker)
            if state is None:
                filtered.append(candidate)
                continue
            if candidate.side != state.exit_side:
                filtered.append(candidate)
                continue
            if state.successful_reentries >= int(self.config.reentry_max_per_market):
                _increment_reason(reject_summary, "reentry_cap_reached")
                continue
            required_edge = self._required_gbm_edge(
                strategy_name=candidate.strategy_name,
                ask_cents=candidate.price_cents,
            ) + float(self.config.reentry_edge_premium)
            if (candidate.gbm_edge or float("-inf")) < required_edge:
                _increment_reason(reject_summary, "reentry_edge_too_weak")
                continue
            if candidate.price_cents > (state.exit_reference_price_cents - int(self.config.reentry_min_price_improvement_cents)):
                _increment_reason(reject_summary, "reentry_no_fresh_lag")
                continue
            contracts = candidate.contracts if candidate.confidence == "high" else min(candidate.contracts, 1)
            if contracts <= 0:
                _increment_reason(reject_summary, "reentry_size_zero")
                continue
            filtered.append(
                StrategyCandidate(
                    strategy_name=candidate.strategy_name,
                    confidence=candidate.confidence,
                    snapshot=candidate.snapshot,
                    side=candidate.side,
                    price_cents=candidate.price_cents,
                    contracts=contracts,
                    gbm_probability=candidate.gbm_probability,
                    gbm_edge=candidate.gbm_edge,
                )
            )
        return filtered

    def _register_successful_reentry(self, candidate: StrategyCandidate) -> None:
        state = self.reentry_states.get(candidate.snapshot.market_ticker)
        if state is None:
            return
        if candidate.side != state.exit_side:
            self.reentry_states.pop(candidate.snapshot.market_ticker, None)
            return
        state.successful_reentries += 1
        self._trace_execution_event(
            {
                "event": "reentry_filled",
                "market_ticker": candidate.snapshot.market_ticker,
                "side": candidate.side,
                "strategy_name": candidate.strategy_name,
                "price_cents": candidate.price_cents,
                "successful_reentries": state.successful_reentries,
            }
        )

    def _latest_spot_price(self, fallback_spot_price: float) -> float:
        ws_spot, _ws_ts = self.spot_feed.get()
        if ws_spot is None:
            return fallback_spot_price
        return float(ws_spot)

    def _reconcile_signal_break_positions(
        self,
        *,
        observed_at: datetime,
        snapshots: list[MarketSnapshot],
        volatility: float | None,
    ) -> list[dict[str, Any]]:
        if not self.config.enable_signal_break_exit:
            return []
        snapshots_by_ticker = {snapshot.market_ticker: snapshot for snapshot in snapshots}
        exits: list[dict[str, Any]] = []
        for market_ticker, position in list(self.local_positions.items()):
            if position.status != "open" or position.contracts <= 0:
                continue
            snapshot = snapshots_by_ticker.get(market_ticker)
            if snapshot is None or snapshot.expiry <= observed_at:
                self.pending_signal_breaks.pop(market_ticker, None)
                continue
            strategy_name = self.position_strategies.get(market_ticker, "unknown")
            exit_reason = self._signal_break_reason(
                position=position,
                strategy_name=strategy_name,
                snapshot=snapshot,
                volatility=volatility,
            )
            if exit_reason is None:
                self.pending_signal_breaks.pop(market_ticker, None)
                continue
            if exit_reason == "hard_stop_loss":
                self.pending_signal_breaks.pop(market_ticker, None)
                order = self._submit_exit_order(position=position, snapshot=snapshot, reason=exit_reason)
                exits.append(order)
                filled_contracts = max(int(order.get("filled_contracts", 0) or 0), 0)
                if filled_contracts <= 0:
                    continue
                exit_price_cents = _safe_int(order.get("execution_price_cents"))
                if exit_price_cents is None:
                    continue
                exit_reference_price_cents = _probability_to_cents(
                    snapshot.yes_bid if position.side == "yes" else snapshot.no_bid
                )
                if exit_reference_price_cents is None:
                    exit_reference_price_cents = exit_price_cents
                realized_total_cents = int((exit_price_cents - position.entry_price_cents) * filled_contracts)
                closed = ClosedTrade(
                    market_ticker=market_ticker,
                    strategy_name=f"{strategy_name}:signal_break",
                    closed_at=observed_at,
                    realized_pnl_cents=realized_total_cents,
                )
                self.closed_trades.append(closed)
                self._update_loss_streak_from_closed_trade(closed, observed_at=observed_at)
                self.reentry_states[market_ticker] = SignalBreakReentryState(
                    market_ticker=market_ticker,
                    exited_at=observed_at,
                    exit_side=position.side,
                    exit_strategy_name=strategy_name,
                    exit_reference_price_cents=exit_reference_price_cents,
                    exit_execution_price_cents=exit_price_cents,
                )
                self._trace_execution_event(
                    {
                        "event": "signal_break_exit_recorded",
                        "market_ticker": market_ticker,
                        "side": position.side,
                        "strategy_name": strategy_name,
                        "exit_reference_price_cents": exit_reference_price_cents,
                        "exit_price_cents": exit_price_cents,
                    }
                )
                if not self.config.enable_signal_break_reentry:
                    self.signal_break_blocked_tickers.add(market_ticker)
                remaining_contracts = max(position.contracts - filled_contracts, 0)
                if remaining_contracts <= 0:
                    del self.local_positions[market_ticker]
                    self.position_strategies.pop(market_ticker, None)
                else:
                    position.contracts = remaining_contracts
                continue
            pending_state = self.pending_signal_breaks.get(market_ticker)
            required_cycles = int(self.config.signal_break_confirmation_cycles)
            if exit_reason == "price_stop_loss":
                required_cycles = int(self.config.price_stop_confirm_cycles)
            if pending_state is None or pending_state.reason != exit_reason:
                self.pending_signal_breaks[market_ticker] = PendingSignalBreakState(reason=exit_reason, count=1)
                if max(required_cycles, 1) <= 1:
                    pending_state = self.pending_signal_breaks[market_ticker]
                else:
                    continue
            else:
                pending_state.count += 1
            if pending_state.count < max(required_cycles, 1):
                continue
            order = self._submit_exit_order(position=position, snapshot=snapshot, reason=exit_reason)
            exits.append(order)
            filled_contracts = max(int(order.get("filled_contracts", 0) or 0), 0)
            if filled_contracts <= 0:
                continue
            exit_price_cents = _safe_int(order.get("execution_price_cents"))
            if exit_price_cents is None:
                continue
            exit_reference_price_cents = _probability_to_cents(snapshot.yes_bid if position.side == "yes" else snapshot.no_bid)
            if exit_reference_price_cents is None:
                exit_reference_price_cents = exit_price_cents
            realized_total_cents = int((exit_price_cents - position.entry_price_cents) * filled_contracts)
            closed = ClosedTrade(
                market_ticker=market_ticker,
                strategy_name=f"{strategy_name}:signal_break",
                closed_at=observed_at,
                realized_pnl_cents=realized_total_cents,
            )
            self.closed_trades.append(closed)
            self._update_loss_streak_from_closed_trade(closed, observed_at=observed_at)
            self.reentry_states[market_ticker] = SignalBreakReentryState(
                market_ticker=market_ticker,
                exited_at=observed_at,
                exit_side=position.side,
                exit_strategy_name=strategy_name,
                exit_reference_price_cents=exit_reference_price_cents,
                exit_execution_price_cents=exit_price_cents,
            )
            self._trace_execution_event(
                {
                    "event": "signal_break_exit_recorded",
                    "market_ticker": market_ticker,
                    "side": position.side,
                    "strategy_name": strategy_name,
                    "exit_reference_price_cents": exit_reference_price_cents,
                    "exit_price_cents": exit_price_cents,
                }
            )
            if not self.config.enable_signal_break_reentry:
                self.signal_break_blocked_tickers.add(market_ticker)
            remaining_contracts = max(position.contracts - filled_contracts, 0)
            self.pending_signal_breaks.pop(market_ticker, None)
            if remaining_contracts <= 0:
                del self.local_positions[market_ticker]
                self.position_strategies.pop(market_ticker, None)
            else:
                position.contracts = remaining_contracts
        return exits

    def _signal_break_reason(
        self,
        *,
        position: Position,
        strategy_name: str,
        snapshot: MarketSnapshot,
        volatility: float | None,
    ) -> str | None:
        if snapshot.contract_type != "threshold" or snapshot.threshold is None:
            return None
        exit_distance = float(self.config.exit_distance_threshold_dollars)
        if position.side == "yes" and snapshot.spot_price < (snapshot.threshold + exit_distance):
            return "spot_below_yes_threshold"
        if position.side == "no" and snapshot.spot_price > (snapshot.threshold - exit_distance):
            return "spot_above_no_threshold"
        bid_cents = _probability_to_cents(snapshot.yes_bid if position.side == "yes" else snapshot.no_bid)
        if bid_cents is not None:
            loss_cents = max(int(position.entry_price_cents - bid_cents), 0)
            hard_stop_cents = int(self.config.hard_stop_cents)
            if hard_stop_cents > 0 and loss_cents >= hard_stop_cents:
                return "hard_stop_loss"
            price_stop_cents = int(self.config.price_stop_cents)
            if price_stop_cents > 0:
                grace_seconds = max(int(self.config.price_stop_grace_seconds), 0)
                age_seconds = max(int((snapshot.observed_at - position.entry_time).total_seconds()), 0)
                if age_seconds >= grace_seconds and loss_cents >= price_stop_cents:
                    return "price_stop_loss"
        if volatility is None:
            return None
        if bid_cents is None:
            return None
        estimate = self.model.estimate(snapshot, volatility=volatility)
        side_probability = estimate.probability if position.side == "yes" else (1.0 - estimate.probability)
        current_edge = side_probability - (bid_cents / 100.0)
        if current_edge < float(self.config.exit_negative_edge_threshold):
            return "gbm_edge_negative"
        return None

    def _submit_exit_order(self, *, position: Position, snapshot: MarketSnapshot, reason: str) -> dict[str, Any]:
        bid_cents = _probability_to_cents(snapshot.yes_bid if position.side == "yes" else snapshot.no_bid)
        if bid_cents is None:
            return {
                "status": "missing_exit_bid",
                "market_ticker": position.market_ticker,
                "reason": reason,
                "filled_contracts": 0,
            }
        execution_price_cents = max(bid_cents - max(int(self.config.exit_cross_cents), 0), 1)
        payload = {
            "ticker": position.market_ticker,
            "action": "sell",
            "side": position.side,
            "count": int(position.contracts),
            "type": "limit",
            "yes_price": execution_price_cents if position.side == "yes" else None,
            "no_price": execution_price_cents if position.side == "no" else None,
            "client_order_id": f"kabot-exit-{position.market_ticker}-{int(snapshot.observed_at.timestamp())}",
            "time_in_force": self.config.exit_time_in_force,
        }
        payload = {key: value for key, value in payload.items() if value is not None}
        if self.config.dry_run:
            return {
                "status": "dry_run",
                "market_ticker": position.market_ticker,
                "reason": reason,
                "execution_price_cents": execution_price_cents,
                "filled_contracts": 0,
                "payload": payload,
            }
        response = self.client.create_order(payload)
        response_order = response.get("order", response) if isinstance(response, dict) else {}
        order_status = _extract_order_status(response_order if isinstance(response_order, dict) else {})
        order_id = _extract_order_id(response)
        filled_contracts = _extract_fill_count(response_order if isinstance(response_order, dict) else {})
        exchange_filled_contracts: int | None = None
        if order_id:
            try:
                fills_response = self.client.get_fills(order_id=order_id, ticker=position.market_ticker)
                exchange_filled_contracts = _filled_contracts_from_fills(fills_response)
            except Exception:
                exchange_filled_contracts = None
        if exchange_filled_contracts is not None:
            filled_contracts = exchange_filled_contracts
        return {
            "status": order_status or "submitted",
            "market_ticker": position.market_ticker,
            "reason": reason,
            "execution_price_cents": execution_price_cents,
            "filled_contracts": max(int(filled_contracts), 0),
            "exchange_filled_contracts": exchange_filled_contracts,
            "payload": payload,
            "response": response,
        }

    def _reconcile_settled_positions(self, observed_at: datetime) -> list[ClosedTrade]:
        closed_now: list[ClosedTrade] = []
        for market_ticker, position in list(self.local_positions.items()):
            if position.status != "open" or position.expiry is None or position.expiry > observed_at:
                continue
            try:
                payload = self.client.get_market(market_ticker)
                market = payload.get("market") if isinstance(payload.get("market"), dict) else payload
                settlement_price = self._extract_settlement_price(market)
                threshold = self._extract_threshold(market)
            except Exception:
                settlement_price = None
                threshold = None
            if settlement_price is None or threshold is None:
                if market_ticker not in self.settlement_pending_since:
                    self.settlement_pending_since[market_ticker] = observed_at
                wait_seconds = (observed_at - self.settlement_pending_since[market_ticker]).total_seconds()
                if wait_seconds < self.config.settlement_timeout_seconds:
                    continue
                self._trace_execution_event({
                    "event": "settlement_forced_timeout",
                    "market_ticker": market_ticker,
                    "wait_seconds": wait_seconds,
                    "settlement_price": settlement_price,
                    "threshold": threshold,
                })
                pnl_cents = 0
            else:
                self.settlement_pending_since.pop(market_ticker, None)
                exit_price_cents = 100 if (
                    (position.side == "yes" and settlement_price >= threshold)
                    or (position.side == "no" and settlement_price < threshold)
                ) else 0
                pnl_cents = int((exit_price_cents - position.entry_price_cents) * position.contracts)
            strategy_name = self.position_strategies.get(market_ticker, "unknown")
            closed = ClosedTrade(
                market_ticker=market_ticker,
                strategy_name=strategy_name,
                closed_at=observed_at,
                realized_pnl_cents=pnl_cents,
            )
            closed_now.append(closed)
            self.closed_trades.append(closed)
            self._update_loss_streak_from_closed_trade(closed, observed_at=observed_at)
            del self.local_positions[market_ticker]
            self.position_strategies.pop(market_ticker, None)
            self.reentry_states.pop(market_ticker, None)
            self.pending_signal_breaks.pop(market_ticker, None)
            self.signal_break_blocked_tickers.discard(market_ticker)
        return closed_now

    def _load_loss_streak_state(self) -> None:
        try:
            payload = json.loads(self._loss_streak_path.read_text())
        except FileNotFoundError:
            return
        except (OSError, json.JSONDecodeError):
            return
        try:
            self.consecutive_losses = max(int(payload.get("consecutive_losses", 0)), 0)
        except (TypeError, ValueError):
            self.consecutive_losses = 0
        cooldown_raw = payload.get("cooldown_until")
        if cooldown_raw:
            try:
                self.cooldown_until = datetime.fromisoformat(str(cooldown_raw).replace("Z", "+00:00")).astimezone(UTC)
            except ValueError:
                self.cooldown_until = None

    def _save_loss_streak_state(self) -> None:
        payload = {
            "consecutive_losses": int(self.consecutive_losses),
            "cooldown_until": self.cooldown_until.isoformat() if self.cooldown_until is not None else None,
        }
        try:
            self._loss_streak_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self._loss_streak_path.with_suffix(".json.tmp")
            tmp_path.write_text(json.dumps(payload, sort_keys=True) + "\n")
            tmp_path.replace(self._loss_streak_path)
        except OSError:
            return

    def _update_loss_streak_from_closed_trade(self, closed: ClosedTrade, *, observed_at: datetime) -> None:
        old_losses = self.consecutive_losses
        old_cooldown = self.cooldown_until
        if closed.realized_pnl_cents < 0:
            self.consecutive_losses += 1
            if self.consecutive_losses >= self.config.cooldown_loss_streak:
                self.cooldown_until = observed_at + timedelta(minutes=self.config.cooldown_minutes)
        else:
            self.consecutive_losses = 0
        if self.consecutive_losses != old_losses or self.cooldown_until != old_cooldown:
            self._save_loss_streak_state()

    def _append_spot_price_history(self, *, observed_at: datetime, spot_price: float) -> None:
        self._spot_price_history.append((observed_at, float(spot_price)))
        cutoff = observed_at - timedelta(minutes=float(self.config.whipsaw_crossing_window_minutes))
        while self._spot_price_history and self._spot_price_history[0][0] < cutoff:
            self._spot_price_history.popleft()

    def _count_threshold_crossings(self, threshold: float, window_minutes: float = 30.0) -> int:
        cutoff = datetime.now(UTC) - timedelta(minutes=float(window_minutes))
        recent = [
            (ts, spot)
            for ts, spot in self._spot_price_history
            if ts >= cutoff and spot != threshold
        ]
        crossings = 0
        previous_side: int | None = None
        for _ts, spot in recent:
            side = 1 if spot > threshold else -1
            if previous_side is not None and side != previous_side:
                crossings += 1
            previous_side = side
        return crossings

    def _daily_realized_pnl_cents(self, day) -> int:
        return int(sum(trade.realized_pnl_cents for trade in self.closed_trades if trade.closed_at.date() == day))

    def _daily_trade_count(self, day) -> int:
        return int(sum(1 for trade_time in self.submitted_trade_times if trade_time.date() == day))

    def _local_open_notional_cents(self) -> int:
        return int(
            sum(
                position.entry_price_cents * position.contracts
                for position in self.local_positions.values()
                if position.status == "open"
            )
        )

    def _active_strategy_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for market_ticker, position in self.local_positions.items():
            if position.status != "open":
                continue
            strategy_name = self.position_strategies.get(market_ticker, "unknown")
            counts[strategy_name] = counts.get(strategy_name, 0) + 1
        return counts

    def _gbm_min_edge_for_strategy(self, strategy_name: str) -> float:
        if strategy_name in {"yes_continuation_wide", "no_continuation_wide"}:
            return float(self.config.gbm_min_edge_wide_yes)
        return float(self.config.gbm_min_edge_mid)

    def _required_gbm_edge(self, *, strategy_name: str, ask_cents: int) -> float:
        base_edge = self._gbm_min_edge_for_strategy(strategy_name)
        if ask_cents > SOFT_ENTRY_CAP_CENTS:
            base_edge += EXPENSIVE_ENTRY_EDGE_PREMIUM
        if self._is_overnight_utc():
            base_edge *= float(self.config.overnight_edge_multiplier)
        return base_edge

    def _is_overnight_utc(self) -> bool:
        start_hour, end_hour = self.config.overnight_hours_utc
        current_hour = datetime.now(UTC).hour
        if start_hour <= end_hour:
            return start_hour <= current_hour < end_hour
        return current_hour >= start_hour or current_hour < end_hour

    @staticmethod
    def _extract_settlement_price(market: dict[str, Any]) -> float | None:
        for key in ("expiration_value", "settlement_price", "settlement_value", "final_price"):
            raw = market.get(key)
            if raw in (None, ""):
                continue
            try:
                return float(raw)
            except (TypeError, ValueError):
                continue
        return None

    @staticmethod
    def _extract_threshold(market: dict[str, Any]) -> float | None:
        for key in ("threshold", "floor_strike", "strike"):
            raw = market.get(key)
            if raw in (None, ""):
                continue
            try:
                return float(raw)
            except (TypeError, ValueError):
                continue
        return None

    @staticmethod
    def _sample_market_view(*, snapshot: MarketSnapshot) -> dict[str, Any]:
        yes_ask_c, yes_bid_c = _probability_to_cents(snapshot.yes_ask), _probability_to_cents(snapshot.yes_bid)
        no_ask_c, no_bid_c = _probability_to_cents(snapshot.no_ask), _probability_to_cents(snapshot.no_bid)
        return {
            "market_ticker": snapshot.market_ticker,
            "volume": round(float(snapshot.volume), 2) if snapshot.volume is not None else None,
            "open_interest": round(float(snapshot.open_interest), 2) if snapshot.open_interest is not None else None,
            "yes_bid_c": yes_bid_c,
            "yes_ask_c": yes_ask_c,
            "no_bid_c": no_bid_c,
            "no_ask_c": no_ask_c,
            "threshold": round(float(snapshot.threshold), 2) if snapshot.threshold is not None else None,
            "spot_price": round(float(snapshot.spot_price), 2),
            "tte_s": max(int((snapshot.expiry - snapshot.observed_at).total_seconds()), 0),
        }

    def _bootstrap_vol_from_coinbase(self) -> None:
        """Fetch recent Coinbase candles and compute realized vol for selected startup profiles."""
        end = datetime.now(UTC)
        start = end - timedelta(minutes=90)
        try:
            response = self.client.session.get(
                "https://api.exchange.coinbase.com/products/BTC-USD/candles",
                params={
                    "granularity": 60,
                    "start": start.isoformat().replace("+00:00", "Z"),
                    "end": end.isoformat().replace("+00:00", "Z"),
                },
                timeout=10,
            )
            response.raise_for_status()
            candles = response.json()
            if not candles or len(candles) < 10:
                return
            closes = [float(candle[4]) for candle in sorted(candles, key=lambda candle: candle[0])]
            log_returns = [
                math.log(closes[index] / closes[index - 1])
                for index in range(1, len(closes))
                if closes[index - 1] > 0 and closes[index] > 0
            ]
            if len(log_returns) < 5:
                return
            sigma_per_minute = float(np.std(np.array(log_returns), ddof=1))
            annualized = sigma_per_minute * math.sqrt(525_600.0)
            if math.isfinite(annualized) and annualized > 0:
                self._bootstrapped_vol = max(annualized, self.config.gbm_volatility_floor)
                print(f"[{self.config.active_profile}] Bootstrapped vol from Coinbase: {self._bootstrapped_vol:.4f}", flush=True)
        except Exception as exc:
            print(f"[{self.config.active_profile}] Vol bootstrap failed, using floor: {exc}", flush=True)

    def _bootstrap_daily_vol(self) -> None:
        session = getattr(self.client, "session", None)
        if session is None:
            self._daily_vol = self.config.daily_vol_floor
            print(f"[DAILY] Vol bootstrap failed, using floor: {self._daily_vol:.4f}", flush=True)
            return
        result = bootstrap_hourly_vol(
            session,
            floor=self.config.daily_vol_floor,
        )
        if result is not None:
            self._daily_vol = result
            print(f"[DAILY] Bootstrapped vol: {self._daily_vol:.4f}", flush=True)
        else:
            self._daily_vol = self.config.daily_vol_floor
            print(f"[DAILY] Vol bootstrap failed, using floor: {self._daily_vol:.4f}", flush=True)

    def _live_volatility_floor(self) -> float:
        if self.config.active_profile in {"NEW", "GOD"} and self._bootstrapped_vol is not None:
            return self._bootstrapped_vol
        return self.config.gbm_volatility_floor

    def _estimate_live_volatility(self, *, observed_at: datetime) -> float | None:
        lookback_start = observed_at - timedelta(minutes=self.config.gbm_lookback_minutes)
        history = self.store.load_market_snapshots(
            series_ticker=self.config.series_ticker,
            observed_from=lookback_start,
            observed_to=observed_at,
        )
        if history.empty or "observed_at" not in history.columns or "spot_price" not in history.columns:
            return self._live_volatility_floor()
        spot = (
            history[["observed_at", "spot_price"]]
            .dropna(subset=["observed_at", "spot_price"])
            .drop_duplicates(subset=["observed_at"])
            .sort_values("observed_at")
        )
        if len(spot) < self.config.gbm_min_points:
            return self._live_volatility_floor()
        price = spot["spot_price"].astype(float).to_numpy()
        times = pd.to_datetime(spot["observed_at"], utc=True)
        log_returns = np.diff(np.log(np.clip(price, 1e-12, None)))
        if len(log_returns) < max(self.config.gbm_min_points - 1, 2):
            return self._live_volatility_floor()
        deltas = np.diff(times.astype("int64")) / 1_000_000_000.0
        positive_deltas = deltas[deltas > 0]
        if len(positive_deltas) == 0:
            return self._live_volatility_floor()
        avg_dt_seconds = float(np.median(positive_deltas))
        if avg_dt_seconds <= 0:
            return self._live_volatility_floor()
        annualization = math.sqrt(31_536_000.0 / avg_dt_seconds)
        sigma = float(np.std(log_returns, ddof=1)) * annualization
        if not math.isfinite(sigma):
            return self._live_volatility_floor()
        return max(sigma, self._live_volatility_floor())

    def _estimate_daily_vol(self, *, observed_at: datetime) -> float:
        db_vol = estimate_hourly_vol_from_db(
            self.store,
            series_ticker=self.config.series_ticker,
            observed_at=observed_at,
            lookback_minutes=240,
            min_points=20,
            floor=self.config.daily_vol_floor,
        )
        if db_vol > self.config.daily_vol_floor:
            return db_vol
        if self._daily_vol is not None:
            return self._daily_vol
        return self.config.daily_vol_floor

    def _account_position_contracts(self) -> dict[str, int]:
        positions_by_ticker: dict[str, int] = {}
        try:
            positions_payload = self.client.get_positions()
            for position in _extract_positions(positions_payload):
                ticker = _extract_market_ticker(position)
                contracts = int(round(abs(_extract_position_count(position))))
                if ticker and contracts > 0:
                    positions_by_ticker[ticker] = contracts
        except Exception:
            pass
        return positions_by_ticker

    def _resting_buy_market_tickers(self) -> set[str]:
        # Use the WebSocket fill-feed cache when healthy — avoids a REST round-trip
        # every poll cycle.  Fall back to REST if the feed has not yet connected.
        ws_result: set[str] | None = None
        if self.fill_feed is not None:
            ws_result = self.fill_feed.get_resting_tickers()
            if ws_result is not None and not ws_result:
                return ws_result

        active: set[str] = set()
        try:
            orders_payload = self.client.list_orders(status="resting")
            for order in _extract_orders(orders_payload):
                ticker = _extract_market_ticker(order)
                status = _extract_order_status(order)
                action = str(order.get("action", "") or "").lower()
                if ticker and action == "buy" and status in {"resting", "open", "submitted", "pending"}:
                    active.add(ticker)
        except Exception:
            if ws_result is not None:
                return ws_result

        if self.fill_feed is not None and ws_result is not None:
            for ticker in ws_result - active:
                self.fill_feed.deregister_order(ticker)

        return active

    def _sync_local_positions(
        self,
        *,
        account_position_contracts: dict[str, int],
        resting_order_tickers: set[str],
        observed_at: datetime,
    ) -> None:
        for market_ticker, position in list(self.local_positions.items()):
            if position.status != "open":
                continue
            if position.expiry is not None and position.expiry <= observed_at:
                continue
            account_contracts = int(account_position_contracts.get(market_ticker, 0))
            if account_contracts > 0:
                position.contracts = account_contracts
                self.local_resting_entry_locks.pop(market_ticker, None)
                continue
            if market_ticker not in resting_order_tickers:
                del self.local_positions[market_ticker]
                self.position_strategies.pop(market_ticker, None)
                self.local_resting_entry_locks.pop(market_ticker, None)
        # Promote resting-entry locks into local positions once the account shows fills.
        for market_ticker, contracts in account_position_contracts.items():
            if contracts <= 0:
                continue
            if market_ticker in self.local_positions:
                continue
            lock = self.local_resting_entry_locks.get(market_ticker)
            if lock is None or lock.price_cents is None:
                continue
            self.local_positions[market_ticker] = Position(
                position_id=str(uuid4()),
                market_ticker=market_ticker,
                side=lock.side,  # type: ignore[arg-type]
                contracts=int(contracts),
                entry_time=lock.observed_at or observed_at,
                entry_price_cents=int(lock.price_cents),
            )
            self.position_strategies[market_ticker] = lock.strategy_name
            self.local_resting_entry_locks.pop(market_ticker, None)
            fill_age_seconds = (observed_at - lock.created_at).total_seconds()
            if fill_age_seconds < 5:
                fill_bucket = "0-5s"
            elif fill_age_seconds < 15:
                fill_bucket = "5-15s"
            elif fill_age_seconds < 30:
                fill_bucket = "15-30s"
            elif fill_age_seconds < 45:
                fill_bucket = "30-45s"
            elif fill_age_seconds < 90:
                fill_bucket = "45-90s"
            else:
                fill_bucket = "90s+"
            self._trace_execution_event(
                {
                    "event": "entry_filled",
                    "market_ticker": market_ticker,
                    "side": lock.side,
                    "strategy_name": lock.strategy_name,
                    "price_cents": int(lock.price_cents),
                    "filled_contracts": int(contracts),
                    "gbm_edge": lock.gbm_edge,
                    "gbm_probability": lock.gbm_probability,
                    "observed_at": lock.observed_at or observed_at,
                    "fill_age_seconds": fill_age_seconds,
                    "fill_bucket": fill_bucket,
                    "fill_edge": (
                        float(lock.gbm_probability) - (int(lock.price_cents) / 100.0)
                        if lock.gbm_probability is not None and lock.price_cents is not None
                        else None
                    ),
                }
            )

    def _submit_order_via_execution_session(self, candidate: StrategyCandidate) -> dict[str, Any]:
        attempts: list[dict[str, Any]] = []
        current_candidate = candidate
        total_filled_contracts = 0
        max_attempts = max(int(self.config.execution_session_attempts), 1)
        entry_time_in_force = candidate.time_in_force or self.config.entry_time_in_force
        ladder_steps = self.config.execution_ladder_steps_cents
        if ladder_steps is None or len(ladder_steps) == 0:
            ladder_steps = (int(self.config.execution_cross_cents),) * max_attempts
        distance_from_threshold = 0.0
        if current_candidate.snapshot.threshold is not None:
            if current_candidate.side == "yes":
                distance_from_threshold = current_candidate.snapshot.spot_price - current_candidate.snapshot.threshold
            else:
                distance_from_threshold = current_candidate.snapshot.threshold - current_candidate.snapshot.spot_price
        is_high_conviction = (
            current_candidate.gbm_edge is not None
            and current_candidate.gbm_edge >= float(self.config.high_conviction_edge_threshold)
            and distance_from_threshold >= float(self.config.high_conviction_distance_threshold_dollars)
        )
        if self.config.execution_ladder_steps_cents_high and is_high_conviction:
            ladder_steps = self.config.execution_ladder_steps_cents_high
        last_response: dict[str, Any] | None = None
        last_payload: dict[str, Any] | None = None
        last_status = "submitted"
        last_execution_price_cents = min(candidate.price_cents + max(self.config.execution_cross_cents, 0), 99)
        last_order_id: str | None = None
        self._trace_execution_event(
            {
                "event": "execution_session_start",
                "market_ticker": candidate.snapshot.market_ticker,
                "side": candidate.side,
                "strategy_name": candidate.strategy_name,
                "signal_price_cents": candidate.price_cents,
                "contracts": candidate.contracts,
            }
        )
        for attempt_index in range(max_attempts):
            remaining_contracts = max(int(current_candidate.contracts) - total_filled_contracts, 0)
            if remaining_contracts <= 0:
                break
            snapshot = current_candidate.snapshot
            step_cents = int(
                ladder_steps[min(attempt_index, len(ladder_steps) - 1)]
                if ladder_steps
                else self.config.execution_cross_cents
            )
            # Spread-aware execution
            ask_cents, bid_cents = _side_prices(snapshot, current_candidate.side)
            spread_cents: int | None = None
            if ask_cents is not None and bid_cents is not None:
                spread_cents = max(ask_cents - bid_cents, 0)
                if spread_cents >= int(self.config.execution_spread_insane_cents):
                    self._trace_execution_event(
                        {
                            "event": "execution_attempt_skipped_no_depth",
                            "market_ticker": snapshot.market_ticker,
                            "attempt": attempt_index + 1,
                            "side": current_candidate.side,
                            "strategy_name": current_candidate.strategy_name,
                            "signal_price_cents": current_candidate.price_cents,
                            "execution_price_cents": execution_price_cents,
                            "orderbook_available_contracts": None,
                            "min_fill_contracts": None,
                            "reason": "spread_insane_skip",
                            "spread_cents": spread_cents,
                        }
                    )
                    return {
                        "status": "skipped_no_depth",
                        "side": current_candidate.side,
                        "strategy_name": current_candidate.strategy_name,
                        "confidence": current_candidate.confidence,
                        "gbm_probability": current_candidate.gbm_probability,
                        "gbm_edge": current_candidate.gbm_edge,
                        "signal_price_cents": current_candidate.price_cents,
                        "execution_price_cents": execution_price_cents,
                        "filled_contracts": total_filled_contracts,
                        "payload": None,
                        "response": None,
                        "attempts": attempts,
                        "orderbook_available_contracts": None,
                    }
                if spread_cents <= int(self.config.execution_spread_tight_cents):
                    step_cents = max(step_cents, int(self.config.execution_spread_tight_min_cross_cents))
                if spread_cents >= int(self.config.execution_spread_wide_cents):
                    step_cents = min(step_cents, int(self.config.execution_spread_wide_max_cross_cents))
                    if attempt_index == 0 and step_cents <= 0:
                        step_cents = 1
            # Cap cross based on edge strength
            execution_price_cents = min(current_candidate.price_cents + max(step_cents, 0), 99)
            residual_edge = None
            max_cross_from_edge = None
            if current_candidate.gbm_probability is not None:
                residual_edge = float(current_candidate.gbm_probability) - (execution_price_cents / 100.0)
                max_cross_from_edge = max(int(math.floor(float(residual_edge) * 100.0)), 0)
                step_cents = min(step_cents, max_cross_from_edge, 3)
                execution_price_cents = min(current_candidate.price_cents + max(step_cents, 0), 99)
                residual_edge = float(current_candidate.gbm_probability) - (execution_price_cents / 100.0)
            if self.config.enforce_positive_edge_on_execution and residual_edge is not None:
                min_edge_margin = float(
                    self.config.execution_min_edge_margin_high if is_high_conviction else self.config.execution_min_edge_margin_low
                )
                if residual_edge < min_edge_margin:
                    self._trace_execution_event(
                        {
                            "event": "execution_attempt_skipped_edge_negative",
                            "market_ticker": snapshot.market_ticker,
                            "attempt": attempt_index + 1,
                            "side": current_candidate.side,
                            "strategy_name": current_candidate.strategy_name,
                            "signal_price_cents": current_candidate.price_cents,
                            "execution_price_cents": execution_price_cents,
                            "gbm_edge": current_candidate.gbm_edge,
                            "residual_edge": residual_edge,
                            "min_edge_margin": min_edge_margin,
                            "max_cross_from_edge": max_cross_from_edge,
                            "ladder_step_cents": step_cents,
                            "reason": "edge_cap_blocked",
                        }
                    )
                    return {
                        "status": "skipped_edge_negative",
                        "side": current_candidate.side,
                        "strategy_name": current_candidate.strategy_name,
                        "confidence": current_candidate.confidence,
                        "gbm_probability": current_candidate.gbm_probability,
                        "gbm_edge": current_candidate.gbm_edge,
                        "signal_price_cents": current_candidate.price_cents,
                        "execution_price_cents": execution_price_cents,
                        "filled_contracts": total_filled_contracts,
                        "payload": None,
                        "response": None,
                        "attempts": attempts,
                        "orderbook_available_contracts": None,
                    }
            orderbook_available_contracts: int | None = None
            orderbook_payload: dict[str, Any] | None = None
            if self.config.use_orderbook_precheck:
                try:
                    orderbook_payload = self.client.get_orderbook(snapshot.market_ticker)
                    orderbook_available_contracts = _available_contracts_at_price(
                        orderbook_payload,
                        side=current_candidate.side,
                        limit_price_cents=execution_price_cents,
                    )
                except Exception:
                    orderbook_payload = None
                    orderbook_available_contracts = None
            self._trace_execution_event(
                {
                    "event": "execution_attempt_ready",
                    "market_ticker": snapshot.market_ticker,
                    "attempt": attempt_index + 1,
                    "side": current_candidate.side,
                    "strategy_name": current_candidate.strategy_name,
                    "signal_price_cents": current_candidate.price_cents,
                    "execution_price_cents": execution_price_cents,
                    "remaining_contracts": remaining_contracts,
                    "orderbook_available_contracts": orderbook_available_contracts,
                    "ladder_step_cents": step_cents,
                    "max_cross_from_edge": max_cross_from_edge,
                    "residual_edge": residual_edge,
                    "signal_edge": current_candidate.gbm_edge,
                }
            )
            low_depth = False
            if self.config.use_orderbook_precheck:
                min_fill_contracts = max(
                    1,
                    int(math.ceil(remaining_contracts * max(float(self.config.min_orderbook_fill_fraction), 0.0))),
                )
                if (
                    orderbook_available_contracts is not None
                    and orderbook_available_contracts < min_fill_contracts
                ):
                    low_depth = True
                    attempts.append(
                        {
                            "attempt": attempt_index + 1,
                            "signal_price_cents": current_candidate.price_cents,
                            "execution_price_cents": execution_price_cents,
                            "orderbook_available_contracts": orderbook_available_contracts,
                            "min_fill_contracts": min_fill_contracts,
                            "status": "skipped_no_depth",
                        }
                    )
                    # Do not hard-skip on low depth; allow laddering unless spread is insane.
            if low_depth and attempt_index == 0 and step_cents <= 0:
                step_cents = 1
                execution_price_cents = min(current_candidate.price_cents + max(step_cents, 0), 99)
            if (
                entry_time_in_force != "immediate_or_cancel"
                and attempt_index > 0
                and last_order_id
            ):
                try:
                    self.client.cancel_order(last_order_id)
                    if self.fill_feed is not None:
                        self.fill_feed.deregister_order(snapshot.market_ticker)
                    self.local_resting_entry_locks.pop(snapshot.market_ticker, None)
                    self._trace_execution_event(
                        {
                            "event": "execution_attempt_canceled_previous",
                            "market_ticker": snapshot.market_ticker,
                            "attempt": attempt_index + 1,
                            "order_id": last_order_id,
                        }
                    )
                except Exception:
                    pass
            payload = {
                "ticker": snapshot.market_ticker,
                "action": "buy",
                "side": current_candidate.side,
                "count": remaining_contracts,
                "type": "limit",
                "yes_price": execution_price_cents if current_candidate.side == "yes" else None,
                "no_price": execution_price_cents if current_candidate.side == "no" else None,
                "client_order_id": f"kabot-exec-{snapshot.market_ticker}-{int(snapshot.observed_at.timestamp())}-{attempt_index}",
                "time_in_force": entry_time_in_force,
            }
            payload = {key: value for key, value in payload.items() if value is not None}
            if self.config.dry_run:
                self._trace_execution_event(
                    {
                        "event": "execution_attempt_dry_run",
                        "market_ticker": snapshot.market_ticker,
                        "attempt": attempt_index + 1,
                        "payload": payload,
                    }
                )
                return {
                    "status": "dry_run",
                    "side": current_candidate.side,
                    "strategy_name": current_candidate.strategy_name,
                    "confidence": current_candidate.confidence,
                    "gbm_probability": current_candidate.gbm_probability,
                    "gbm_edge": current_candidate.gbm_edge,
                    "signal_price_cents": current_candidate.price_cents,
                    "execution_price_cents": execution_price_cents,
                    "filled_contracts": 0,
                    "payload": payload,
                    "attempts": attempts,
                    "orderbook_available_contracts": orderbook_available_contracts,
                }
            submission_ts = datetime.now(UTC)
            signal_age_ms = int(round((submission_ts - snapshot.observed_at).total_seconds() * 1000.0))
            response = self.client.create_order(payload)
            order_id = _extract_order_id(response)
            response_order = response.get("order", response) if isinstance(response, dict) else {}
            order_status = _extract_order_status(response_order if isinstance(response_order, dict) else {})
            filled_contracts = _extract_fill_count(response_order if isinstance(response_order, dict) else {})
            fills_response: dict[str, Any] | None = None
            exchange_filled_contracts: int | None = None
            if order_id:
                try:
                    fills_response = self.client.get_fills(order_id=order_id, ticker=snapshot.market_ticker)
                    exchange_filled_contracts = _filled_contracts_from_fills(fills_response)
                except Exception:
                    fills_response = None
                    exchange_filled_contracts = None
            if exchange_filled_contracts is not None:
                filled_contracts = exchange_filled_contracts
            total_filled_contracts += filled_contracts
            last_status = order_status or "submitted"
            last_response = response
            last_payload = payload
            last_execution_price_cents = execution_price_cents
            last_order_id = order_id
            if (
                entry_time_in_force != "immediate_or_cancel"
                and order_status in {"resting", "open", "submitted", "pending"}
            ):
                self.local_resting_entry_locks[snapshot.market_ticker] = LocalRestingEntryLock(
                    created_at=datetime.now(UTC),
                    side=current_candidate.side,
                    strategy_name=current_candidate.strategy_name,
                    price_cents=current_candidate.price_cents,
                    gbm_edge=current_candidate.gbm_edge,
                    gbm_probability=current_candidate.gbm_probability,
                    observed_at=snapshot.observed_at,
                    contracts=remaining_contracts,
                )
                if order_id and self.fill_feed is not None:
                    self.fill_feed.register_order(snapshot.market_ticker, order_id)
            attempt_record: dict[str, Any] = {
                "attempt": attempt_index + 1,
                "signal_price_cents": current_candidate.price_cents,
                "execution_price_cents": execution_price_cents,
                "payload": payload,
                "response": response,
                "order_id": order_id,
                "status": order_status or "submitted",
                "filled_contracts": filled_contracts,
                "exchange_filled_contracts": exchange_filled_contracts,
                "orderbook_available_contracts": orderbook_available_contracts,
            }
            attempts.append(attempt_record)
            self._trace_execution_event(
                {
                    "event": "execution_attempt_submitted",
                    "market_ticker": snapshot.market_ticker,
                    "attempt": attempt_index + 1,
                    "side": current_candidate.side,
                    "strategy_name": current_candidate.strategy_name,
                    "signal_price_cents": current_candidate.price_cents,
                    "execution_price_cents": execution_price_cents,
                    "status": order_status or "submitted",
                    "filled_contracts": filled_contracts,
                    "exchange_filled_contracts": exchange_filled_contracts,
                    "orderbook_available_contracts": orderbook_available_contracts,
                    "ladder_step_cents": step_cents,
                    "max_cross_from_edge": max_cross_from_edge,
                    "residual_edge": residual_edge,
                    "signal_edge": current_candidate.gbm_edge,
                    "submit_edge": residual_edge,
                    "signal_age_ms": signal_age_ms,
                }
            )
            if total_filled_contracts >= int(candidate.contracts) or attempt_index >= max_attempts - 1:
                break
            delay_seconds = float(self.config.execution_session_retry_delay_seconds)
            if current_candidate.gbm_edge is not None:
                if current_candidate.gbm_edge >= 0.06:
                    delay_seconds = float(self.config.execution_ladder_delay_high_seconds)
                elif current_candidate.gbm_edge >= 0.04:
                    delay_seconds = float(self.config.execution_ladder_delay_mid_seconds)
                else:
                    delay_seconds = float(self.config.execution_ladder_delay_low_seconds)
            tte_minutes = max((snapshot.expiry - snapshot.observed_at).total_seconds() / 60.0, 0.0)
            if tte_minutes <= float(self.config.fast_market_tte_minutes):
                delay_seconds = min(delay_seconds, float(self.config.fast_market_delay_ceiling_seconds))
            time.sleep(max(delay_seconds, 0.0))
            refreshed_candidate = self._refresh_candidate_from_live_state(current_candidate=current_candidate)
            if refreshed_candidate is None:
                self._trace_execution_event(
                    {
                        "event": "execution_attempt_not_retried",
                        "market_ticker": snapshot.market_ticker,
                        "attempt": attempt_index + 1,
                        "reason": "live_state_refresh_failed_or_signal_broke",
                    }
                )
                break
            current_candidate = refreshed_candidate
        if (
            self.config.hybrid_resting_entry_enabled
            and entry_time_in_force == "immediate_or_cancel"
            and total_filled_contracts <= 0
            and attempts
            and current_candidate.snapshot.market_ticker not in self.local_resting_entry_locks
        ):
            refreshed_candidate = self._refresh_candidate_from_live_state(current_candidate=current_candidate)
            if refreshed_candidate is not None:
                current_candidate = refreshed_candidate
                snapshot = current_candidate.snapshot
                remaining_contracts = max(int(current_candidate.contracts) - total_filled_contracts, 0)
                execution_price_cents = min(
                    current_candidate.price_cents + max(int(self.config.execution_cross_cents), 0),
                    99,
                )
                residual_edge = None
                if current_candidate.gbm_probability is not None:
                    residual_edge = float(current_candidate.gbm_probability) - (execution_price_cents / 100.0)
                    max_cross_from_edge = max(int(math.floor(float(residual_edge) * 100.0)), 0)
                    cross_cents = min(max(int(self.config.execution_cross_cents), 0), max_cross_from_edge, 3)
                    execution_price_cents = min(current_candidate.price_cents + max(cross_cents, 0), 99)
                    residual_edge = float(current_candidate.gbm_probability) - (execution_price_cents / 100.0)
                if remaining_contracts > 0 and (
                    residual_edge is None or residual_edge >= float(self.config.execution_min_edge_margin_low)
                ):
                    payload = {
                        "ticker": snapshot.market_ticker,
                        "action": "buy",
                        "side": current_candidate.side,
                        "count": remaining_contracts,
                        "type": "limit",
                        "yes_price": execution_price_cents if current_candidate.side == "yes" else None,
                        "no_price": execution_price_cents if current_candidate.side == "no" else None,
                        "client_order_id": f"kabot-hybrid-{snapshot.market_ticker}-{int(snapshot.observed_at.timestamp())}",
                        "time_in_force": "good_till_canceled",
                    }
                    payload = {key: value for key, value in payload.items() if value is not None}
                    response = self.client.create_order(payload)
                    order_id = _extract_order_id(response)
                    response_order = response.get("order", response) if isinstance(response, dict) else {}
                    order_status = _extract_order_status(response_order if isinstance(response_order, dict) else {})
                    filled_contracts = _extract_fill_count(response_order if isinstance(response_order, dict) else {})
                    exchange_filled_contracts = None
                    if order_id:
                        try:
                            fills_response = self.client.get_fills(order_id=order_id, ticker=snapshot.market_ticker)
                            exchange_filled_contracts = _filled_contracts_from_fills(fills_response)
                        except Exception:
                            exchange_filled_contracts = None
                    if exchange_filled_contracts is not None:
                        filled_contracts = exchange_filled_contracts
                    total_filled_contracts += filled_contracts
                    last_status = order_status or "submitted"
                    last_response = response
                    last_payload = payload
                    last_execution_price_cents = execution_price_cents
                    attempt_record = {
                        "attempt": len(attempts) + 1,
                        "signal_price_cents": current_candidate.price_cents,
                        "execution_price_cents": execution_price_cents,
                        "payload": payload,
                        "response": response,
                        "order_id": order_id,
                        "status": order_status or "submitted",
                        "filled_contracts": filled_contracts,
                        "exchange_filled_contracts": exchange_filled_contracts,
                        "orderbook_available_contracts": None,
                        "hybrid_resting_fallback": True,
                    }
                    attempts.append(attempt_record)
                    self._trace_execution_event(
                        {
                            "event": "execution_hybrid_resting_submitted",
                            "market_ticker": snapshot.market_ticker,
                            "side": current_candidate.side,
                            "strategy_name": current_candidate.strategy_name,
                            "signal_price_cents": current_candidate.price_cents,
                            "execution_price_cents": execution_price_cents,
                            "status": order_status or "submitted",
                            "filled_contracts": filled_contracts,
                            "exchange_filled_contracts": exchange_filled_contracts,
                            "residual_edge": residual_edge,
                            "max_age_seconds": self.config.hybrid_resting_entry_seconds,
                        }
                    )
                    if order_status in {"resting", "open", "submitted", "pending"}:
                        self.local_resting_entry_locks[snapshot.market_ticker] = LocalRestingEntryLock(
                            created_at=datetime.now(UTC),
                            side=current_candidate.side,
                            strategy_name=current_candidate.strategy_name,
                            price_cents=execution_price_cents,
                            gbm_edge=current_candidate.gbm_edge,
                            gbm_probability=current_candidate.gbm_probability,
                            observed_at=snapshot.observed_at,
                            contracts=remaining_contracts,
                        )
                        if order_id and self.fill_feed is not None:
                            self.fill_feed.register_order(snapshot.market_ticker, order_id)
        self._trace_execution_event(
            {
                "event": "execution_session_complete",
                "market_ticker": candidate.snapshot.market_ticker,
                "side": candidate.side,
                "strategy_name": candidate.strategy_name,
                "status": last_status if attempts else "abandoned",
                "filled_contracts": total_filled_contracts,
                "attempt_count": len(attempts),
            }
        )
        return {
            "status": last_status if attempts else "abandoned",
            "side": candidate.side,
            "strategy_name": candidate.strategy_name,
            "confidence": candidate.confidence,
            "gbm_probability": candidate.gbm_probability,
            "gbm_edge": candidate.gbm_edge,
            "signal_price_cents": candidate.price_cents,
            "execution_price_cents": last_execution_price_cents,
            "filled_contracts": total_filled_contracts,
            "payload": last_payload,
            "response": last_response,
            "attempts": attempts,
            "exchange_filled_contracts": total_filled_contracts,
            "orderbook_available_contracts": attempts[-1].get("orderbook_available_contracts") if attempts else None,
        }

    def _refresh_candidate_from_live_state(
        self,
        *,
        current_candidate: StrategyCandidate,
    ) -> StrategyCandidate | None:
        observed_at = datetime.now(UTC)
        spot_price = self._latest_spot_price(current_candidate.snapshot.spot_price)
        refreshed_snapshot = self._snapshot_for_ticker(
            ticker=current_candidate.snapshot.market_ticker,
            spot_price=spot_price,
            observed_at=observed_at,
        )
        if refreshed_snapshot is None and self.config.execution_session_use_rest_fallback:
            try:
                latest_market = self.client.get_market(current_candidate.snapshot.market_ticker)
                latest_raw = latest_market.get("market") if isinstance(latest_market.get("market"), dict) else latest_market
                refreshed_snapshot = normalize_market(
                    {**latest_raw, "series_ticker": self.config.series_ticker},
                    spot_price=spot_price,
                    observed_at=observed_at,
                    source="kalshi_live_exec_fallback",
                )
            except Exception:
                refreshed_snapshot = None
        if refreshed_snapshot is None:
            return None
        return self._refresh_candidate(current_candidate=current_candidate, snapshot=refreshed_snapshot)

    def _run_daily_profile_cycle(
        self,
        *,
        snapshots: list[MarketSnapshot],
        observed_at: datetime,
    ) -> list[dict[str, Any]]:
        """DAILY profile: evaluate KXBTCD signals and manage open positions."""
        volatility = self._estimate_daily_vol(observed_at=observed_at)

        signal_config = DailySignalConfig(
            min_edge=self.config.daily_min_edge,
            min_price_cents=self.config.daily_min_price_cents,
            max_price_cents=self.config.daily_max_price_cents,
            min_tte_minutes=self.config.daily_min_tte_minutes,
            max_tte_minutes=self.config.daily_max_tte_minutes,
            min_distance_dollars=self.config.daily_min_distance_dollars,
            max_spread_cents=self.config.daily_max_spread_cents,
            min_volume=self.config.daily_min_volume,
        )

        exit_config = DailyExitConfig(
            fair_value_buffer_cents=self.config.daily_fair_value_buffer_cents,
            negative_edge_threshold=self.config.daily_negative_edge_threshold,
            min_tte_to_exit_minutes=self.config.daily_min_tte_to_exit_minutes,
            stop_loss_cents=self.config.daily_stop_loss_cents,
        )

        results: list[dict[str, Any]] = []
        snapshots_by_ticker = {snapshot.market_ticker: snapshot for snapshot in snapshots}
        if (
            self._last_daily_signal_debug_at is None
            or (observed_at - self._last_daily_signal_debug_at).total_seconds() >= 300.0
        ):
            self._last_daily_signal_debug_at = observed_at
            reason_counts: dict[str, int] = {}
            passed_tickers: list[str] = []
            for snapshot in snapshots:
                r = evaluate_daily_signal_debug(
                    snapshot,
                    volatility=volatility,
                    config=signal_config,
                    observed_at=observed_at,
                )
                reason = str(r.get("reason", "unknown"))
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
                if not r.get("rejected", True):
                    passed_tickers.append(snapshot.market_ticker)
            debug_event = {
                "event": "daily_signal_debug",
                "profile": self.config.active_profile,
                "daily_vol": round(volatility, 4),
                "result_count": len(snapshots),
                "reason_counts": {**reason_counts, "passed": len(passed_tickers)},
                "passed_tickers": passed_tickers,
            }
            self._trace_execution_event(debug_event)
            print(json.dumps(debug_event), flush=True)

        for ticker, position in list(self._daily_positions.items()):
            if position.status != "open" or position.contracts <= 0:
                continue

            if position.expiry is not None and position.expiry <= observed_at:
                result = self._reconcile_daily_settled_position(
                    ticker=ticker,
                    position=position,
                    observed_at=observed_at,
                )
                results.append(result)
                continue

            snapshot = snapshots_by_ticker.get(ticker)
            if snapshot is None:
                continue

            fair_value_cents = self._daily_position_fair_values.get(
                ticker,
                position.entry_price_cents,
            )
            exit_decision = evaluate_daily_exit(
                snapshot,
                side=position.side,
                entry_price_cents=position.entry_price_cents,
                fair_value_cents=fair_value_cents,
                contracts=position.contracts,
                volatility=volatility,
                observed_at=observed_at,
                config=exit_config,
            )

            self._trace_execution_event(
                {
                    "event": "daily_exit_evaluated",
                    "market_ticker": ticker,
                    "action": exit_decision.action,
                    "trigger": exit_decision.trigger,
                    "exit_price_cents": exit_decision.exit_price_cents,
                    "unrealized_pnl_cents": exit_decision.unrealized_pnl_cents,
                    "reason": exit_decision.reason,
                }
            )

            if exit_decision.action == "exit" and exit_decision.exit_price_cents is not None:
                order_result = self._submit_daily_exit_order(
                    position=position,
                    snapshot=snapshot,
                    reason=exit_decision.trigger or "unknown",
                )
                results.append(order_result)
                filled = max(int(order_result.get("filled_contracts", 0) or 0), 0)
                if filled > 0:
                    pnl_cents = int(
                        (exit_decision.exit_price_cents - position.entry_price_cents) * filled
                    )
                    self._daily_closed_trades.append(
                        ClosedTrade(
                            market_ticker=ticker,
                            strategy_name="daily",
                            closed_at=observed_at,
                            realized_pnl_cents=pnl_cents,
                        )
                    )
                    self._trace_execution_event(
                        {
                            "event": "daily_exit_filled",
                            "market_ticker": ticker,
                            "side": position.side,
                            "exit_price_cents": exit_decision.exit_price_cents,
                            "filled_contracts": filled,
                            "pnl_cents": pnl_cents,
                            "trigger": exit_decision.trigger,
                        }
                    )
                    remaining = max(position.contracts - filled, 0)
                    if remaining <= 0:
                        del self._daily_positions[ticker]
                        self._daily_position_fair_values.pop(ticker, None)
                    else:
                        position.contracts = remaining

        open_daily_count = sum(
            1 for position in self._daily_positions.values() if position.status == "open"
        )
        if open_daily_count >= self.config.daily_max_open_markets:
            return results

        daily_pnl = sum(
            trade.realized_pnl_cents
            for trade in self._daily_closed_trades
            if trade.closed_at.date() == observed_at.date()
        )
        if daily_pnl <= -abs(self.config.daily_loss_stop_cents):
            return results

        open_daily_tickers = {
            ticker
            for ticker, position in self._daily_positions.items()
            if position.status == "open"
        }
        if self.fill_feed is not None:
            resting_tickers = self.fill_feed.get_resting_tickers()
            if resting_tickers is not None:
                open_daily_tickers |= resting_tickers

        for snapshot in snapshots:
            if snapshot.market_ticker in open_daily_tickers:
                continue
            if open_daily_count >= self.config.daily_max_open_markets:
                break

            signal_result = evaluate_daily_signal(
                snapshot,
                volatility=volatility,
                config=signal_config,
                observed_at=observed_at,
            )

            if isinstance(signal_result, str):
                continue
            signal = signal_result

            self._trace_execution_event(
                {
                    "event": "daily_signal",
                    "market_ticker": signal.market_ticker,
                    "side": signal.side,
                    "entry_price_cents": signal.entry_price_cents,
                    "model_probability": signal.model_probability,
                    "market_probability": signal.market_probability,
                    "edge": signal.edge,
                    "tte_minutes": signal.tte_minutes,
                    "fair_value_cents": signal.fair_value_cents,
                    "daily_vol": volatility,
                    "reason": signal.reason,
                }
            )

            order_result = self._submit_daily_entry_order(
                signal=signal,
                observed_at=observed_at,
            )
            results.append(order_result)

            filled = max(int(order_result.get("filled_contracts", 0) or 0), 0)
            if filled > 0:
                open_daily_tickers.add(signal.market_ticker)
                open_daily_count += 1
                self._daily_positions[signal.market_ticker] = Position(
                    position_id=str(uuid4()),
                    market_ticker=signal.market_ticker,
                    side=signal.side,
                    contracts=filled,
                    entry_time=observed_at,
                    entry_price_cents=signal.entry_price_cents,
                    expiry=snapshot.expiry,
                )
                self._daily_position_fair_values[signal.market_ticker] = signal.fair_value_cents
                self._trace_execution_event(
                    {
                        "event": "daily_entry_filled",
                        "market_ticker": signal.market_ticker,
                        "side": signal.side,
                        "price_cents": signal.entry_price_cents,
                        "filled_contracts": filled,
                        "edge": signal.edge,
                        "fair_value_cents": signal.fair_value_cents,
                    }
                )

        return results

    def _run_new_profile_cycle(
        self,
        *,
        snapshots: list[MarketSnapshot],
        volatility: float,
        observed_at: datetime,
        reject_summary: dict[str, int] | None = None,
    ) -> list[dict[str, Any]]:
        """NEW profile: evaluate Strategy A and B signals and submit orders."""
        strategy_a_config = StrategyAConfig(
            min_edge=0.08,
            min_tte_minutes=8.0,
            max_tte_minutes=14.0,
            min_price_cents=20,
            max_price_cents=80,
            max_spread_cents=8,
            min_volume=1000.0,
        )
        strategy_b_config = StrategyBConfig(
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

        velocity = self._velocity_detector.reading()
        is_fast_move = self._velocity_detector.is_fast_move(
            min_bps_per_second=strategy_b_config.min_velocity_bps_per_second,
            min_total_bps=strategy_b_config.min_total_move_bps,
        )
        open_tickers = {
            ticker for ticker, position in self.local_positions.items()
            if position.status == "open"
        }
        orders_placed: list[dict[str, Any]] = []

        for snapshot in snapshots:
            if snapshot.market_ticker in open_tickers:
                if reject_summary is not None:
                    _increment_reason(reject_summary, "has_open_position")
                continue

            signal: NewSignal | None = None
            if not is_fast_move:
                signal = evaluate_strategy_a(
                    snapshot,
                    volatility=volatility,
                    config=strategy_a_config,
                    observed_at=observed_at,
                )
            if signal is None and is_fast_move:
                signal = evaluate_strategy_b(
                    snapshot,
                    base_volatility=volatility,
                    move_direction=velocity.direction,
                    move_bps=velocity.move_bps,
                    config=strategy_b_config,
                    observed_at=observed_at,
                )
            if signal is None:
                if reject_summary is not None:
                    _increment_reason(
                        reject_summary,
                        self._new_profile_reject_reason(
                            snapshot=snapshot,
                            volatility=volatility,
                            observed_at=observed_at,
                            is_fast_move=is_fast_move,
                            velocity_direction=velocity.direction,
                            strategy_a_config=strategy_a_config,
                            strategy_b_config=strategy_b_config,
                        ),
                    )
                continue

            self._trace_execution_event(
                {
                    "event": "new_profile_signal",
                    "strategy": signal.strategy,
                    "market_ticker": signal.market_ticker,
                    "side": signal.side,
                    "entry_price_cents": signal.entry_price_cents,
                    "model_probability": signal.model_probability,
                    "market_probability": signal.market_probability,
                    "edge": signal.edge,
                    "tte_minutes": signal.tte_minutes,
                    "velocity_bps_per_second": velocity.bps_per_second,
                    "velocity_move_bps": velocity.move_bps,
                    "reason": signal.reason,
                }
            )

            result = self._submit_new_profile_order(signal=signal, observed_at=observed_at)
            orders_placed.append(result)
            filled = max(int(result.get("filled_contracts", 0) or 0), 0)
            if filled > 0:
                open_tickers.add(signal.market_ticker)
                self.local_positions[signal.market_ticker] = Position(
                    position_id=str(uuid4()),
                    market_ticker=signal.market_ticker,
                    side=signal.side,
                    contracts=filled,
                    entry_time=observed_at,
                    entry_price_cents=signal.entry_price_cents,
                    expiry=snapshot.expiry,
                )
                self.position_strategies[signal.market_ticker] = signal.strategy

        return orders_placed

    def _new_profile_reject_reason(
        self,
        *,
        snapshot: MarketSnapshot,
        volatility: float,
        observed_at: datetime,
        is_fast_move: bool,
        velocity_direction: int,
        strategy_a_config: StrategyAConfig,
        strategy_b_config: StrategyBConfig,
    ) -> str:
        """Return the first NEW-profile gate that blocks a snapshot."""
        if snapshot.contract_type != "threshold" or snapshot.threshold is None:
            return "not_threshold"
        tte_minutes = max((snapshot.expiry - observed_at).total_seconds() / 60.0, 0.0)
        if not is_fast_move:
            return self._new_profile_strategy_a_reject_reason(
                snapshot=snapshot,
                volatility=volatility,
                tte_minutes=tte_minutes,
                config=strategy_a_config,
            )
        return self._new_profile_strategy_b_reject_reason(
            snapshot=snapshot,
            volatility=volatility,
            tte_minutes=tte_minutes,
            velocity_direction=velocity_direction,
            config=strategy_b_config,
        )

    @staticmethod
    def _new_profile_strategy_a_reject_reason(
        *,
        snapshot: MarketSnapshot,
        volatility: float,
        tte_minutes: float,
        config: StrategyAConfig,
    ) -> str:
        if (snapshot.volume or 0.0) < config.min_volume:
            return "volume_below_a"
        if not (config.min_tte_minutes < tte_minutes <= config.max_tte_minutes):
            return "tte_outside_a"
        prob_yes = probability_for_snapshot(snapshot, volatility=max(volatility, 0.05), drift=0.0)
        saw_quotes = False
        saw_price_band = False
        saw_spread = False
        for model_prob, ask, bid in [
            (prob_yes, snapshot.yes_ask, snapshot.yes_bid),
            (1.0 - prob_yes, snapshot.no_ask, snapshot.no_bid),
        ]:
            if ask is None or bid is None:
                continue
            saw_quotes = True
            ask_cents = int(round(ask * 100.0))
            bid_cents = int(round(bid * 100.0))
            if not (config.min_price_cents <= ask_cents <= config.max_price_cents):
                continue
            saw_price_band = True
            if ask_cents - bid_cents > config.max_spread_cents:
                continue
            saw_spread = True
            if model_prob - ask >= config.min_edge:
                return "unknown_a"
        if not saw_quotes:
            return "missing_quotes_a"
        if not saw_price_band:
            return "price_band_a"
        if not saw_spread:
            return "spread_wide_a"
        return "edge_below_a"

    @staticmethod
    def _new_profile_strategy_b_reject_reason(
        *,
        snapshot: MarketSnapshot,
        volatility: float,
        tte_minutes: float,
        velocity_direction: int,
        config: StrategyBConfig,
    ) -> str:
        if velocity_direction == 0:
            return "no_move_direction_b"
        if (snapshot.volume or 0.0) < config.min_volume:
            return "volume_below_b"
        if not (config.min_tte_minutes < tte_minutes <= config.max_tte_minutes):
            return "tte_outside_b"
        sigma = max(volatility * config.vol_spike_multiplier, 0.05)
        prob_yes = probability_for_snapshot(snapshot, volatility=sigma, drift=0.0)
        if velocity_direction > 0:
            model_prob = 1.0 - prob_yes
            ask = snapshot.no_ask
            bid = snapshot.no_bid
        else:
            model_prob = prob_yes
            ask = snapshot.yes_ask
            bid = snapshot.yes_bid
        if ask is None or bid is None:
            return "missing_quotes_b"
        ask_cents = int(round(ask * 100.0))
        bid_cents = int(round(bid * 100.0))
        if not (config.min_price_cents <= ask_cents <= config.max_price_cents):
            return "price_band_b"
        if ask_cents - bid_cents > config.max_spread_cents:
            return "spread_wide_b"
        if model_prob - ask < config.min_edge:
            return "edge_below_b"
        return "unknown_b"

    def _submit_daily_entry_order(
        self,
        *,
        signal: DailySignal,
        observed_at: datetime,
    ) -> dict[str, Any]:
        """Submit a GTC limit entry order for the DAILY profile."""
        payload: dict[str, Any] = {
            "ticker": signal.market_ticker,
            "action": "buy",
            "side": signal.side,
            "count": self.config.daily_contracts_per_trade,
            "type": "limit",
            "time_in_force": "good_till_canceled",
            "client_order_id": (
                f"kabot-daily-{signal.market_ticker}-{int(observed_at.timestamp())}"
            ),
        }
        if signal.side == "yes":
            payload["yes_price"] = signal.entry_price_cents
        else:
            payload["no_price"] = signal.entry_price_cents

        self._trace_execution_event(
            {
                "event": "daily_order_submit",
                "market_ticker": signal.market_ticker,
                "side": signal.side,
                "price_cents": signal.entry_price_cents,
                "contracts": self.config.daily_contracts_per_trade,
                "edge": signal.edge,
            }
        )

        if self.config.dry_run:
            return {
                "status": "dry_run",
                "market_ticker": signal.market_ticker,
                "side": signal.side,
                "price_cents": signal.entry_price_cents,
                "filled_contracts": 0,
                "edge": signal.edge,
                "payload": payload,
            }

        try:
            response = self.client.create_order(payload)
        except Exception as exc:
            self._trace_execution_event(
                {
                    "event": "daily_order_error",
                    "market_ticker": signal.market_ticker,
                    "error": str(exc),
                }
            )
            return {
                "status": "error",
                "market_ticker": signal.market_ticker,
                "error": str(exc),
                "filled_contracts": 0,
            }

        response_order = response.get("order", response) if isinstance(response, dict) else {}
        order = response_order if isinstance(response_order, dict) else {}
        order_status = _extract_order_status(order)
        order_id = _extract_order_id(response)
        filled_contracts = _extract_fill_count(order)

        if order_id and self.fill_feed is not None:
            self.fill_feed.register_order(signal.market_ticker, order_id)

        return {
            "status": order_status or "submitted",
            "market_ticker": signal.market_ticker,
            "side": signal.side,
            "price_cents": signal.entry_price_cents,
            "filled_contracts": max(int(filled_contracts), 0),
            "order_id": order_id,
            "edge": signal.edge,
            "response": response,
        }

    def _submit_daily_exit_order(
        self,
        *,
        position: Position,
        snapshot: MarketSnapshot,
        reason: str,
    ) -> dict[str, Any]:
        """Submit an IOC sell order for DAILY exit. When exiting, exit fast."""
        bid = snapshot.yes_bid if position.side == "yes" else snapshot.no_bid
        if bid is None:
            return {
                "status": "no_bid",
                "market_ticker": position.market_ticker,
                "filled_contracts": 0,
                "reason": reason,
            }

        exit_price_cents = int(round(bid * 100.0))

        payload: dict[str, Any] = {
            "ticker": position.market_ticker,
            "action": "sell",
            "side": position.side,
            "count": position.contracts,
            "type": "limit",
            "time_in_force": "immediate_or_cancel",
            "client_order_id": (
                f"kabot-daily-exit-{position.market_ticker}"
                f"-{int(snapshot.observed_at.timestamp())}"
            ),
        }
        if position.side == "yes":
            payload["yes_price"] = exit_price_cents
        else:
            payload["no_price"] = exit_price_cents

        if self.config.dry_run:
            return {
                "status": "dry_run",
                "market_ticker": position.market_ticker,
                "exit_price_cents": exit_price_cents,
                "filled_contracts": 0,
                "reason": reason,
                "payload": payload,
            }

        try:
            response = self.client.create_order(payload)
        except Exception as exc:
            return {
                "status": "error",
                "market_ticker": position.market_ticker,
                "error": str(exc),
                "filled_contracts": 0,
            }

        response_order = response.get("order", response) if isinstance(response, dict) else {}
        order = response_order if isinstance(response_order, dict) else {}
        order_status = _extract_order_status(order)
        filled_contracts = _extract_fill_count(order)

        return {
            "status": order_status or "submitted",
            "market_ticker": position.market_ticker,
            "exit_price_cents": exit_price_cents,
            "filled_contracts": max(int(filled_contracts), 0),
            "reason": reason,
            "response": response,
        }

    def _reconcile_daily_settled_position(
        self,
        *,
        ticker: str,
        position: Position,
        observed_at: datetime,
    ) -> dict[str, Any]:
        """Fetch settlement from Kalshi and book PnL for expired DAILY position."""
        try:
            payload = self.client.get_market(ticker)
            market = payload.get("market") if isinstance(payload.get("market"), dict) else payload
            settlement_price = self._extract_settlement_price(market)
            threshold = self._extract_threshold(market)
        except Exception:
            settlement_price = None
            threshold = None

        if settlement_price is None or threshold is None:
            return {
                "status": "settlement_pending",
                "market_ticker": ticker,
                "filled_contracts": 0,
            }

        exit_price_cents = 100 if (
            (position.side == "yes" and settlement_price >= threshold)
            or (position.side == "no" and settlement_price < threshold)
        ) else 0

        pnl_cents = int((exit_price_cents - position.entry_price_cents) * position.contracts)

        self._daily_closed_trades.append(
            ClosedTrade(
                market_ticker=ticker,
                strategy_name="daily",
                closed_at=observed_at,
                realized_pnl_cents=pnl_cents,
            )
        )
        del self._daily_positions[ticker]
        self._daily_position_fair_values.pop(ticker, None)

        self._trace_execution_event(
            {
                "event": "daily_settled",
                "market_ticker": ticker,
                "side": position.side,
                "settlement_price": settlement_price,
                "threshold": threshold,
                "exit_price_cents": exit_price_cents,
                "pnl_cents": pnl_cents,
                "contracts": position.contracts,
            }
        )

        return {
            "status": "settled",
            "market_ticker": ticker,
            "pnl_cents": pnl_cents,
            "exit_price_cents": exit_price_cents,
        }

    def _submit_new_profile_order(
        self,
        *,
        signal: NewSignal,
        observed_at: datetime,
    ) -> dict[str, Any]:
        """Submit an order for the NEW profile without REST precheck or blocking fill lookup."""
        cross_cents = 0 if signal.strategy == "hold_settlement" else 2
        execution_price_cents = min(signal.entry_price_cents + cross_cents, 99)
        payload: dict[str, Any] = {
            "ticker": signal.market_ticker,
            "action": "buy",
            "side": signal.side,
            "count": 1,
            "type": "limit",
            "time_in_force": "immediate_or_cancel",
            "client_order_id": f"kabot-new-{signal.market_ticker}-{int(observed_at.timestamp())}",
        }
        if signal.side == "yes":
            payload["yes_price"] = execution_price_cents
        else:
            payload["no_price"] = execution_price_cents

        self._trace_execution_event(
            {
                "event": "new_profile_order_submit",
                "strategy": signal.strategy,
                "market_ticker": signal.market_ticker,
                "side": signal.side,
                "signal_price_cents": signal.entry_price_cents,
                "execution_price_cents": execution_price_cents,
                "edge": signal.edge,
            }
        )

        if self.config.dry_run:
            return {
                "status": "dry_run",
                "strategy": signal.strategy,
                "market_ticker": signal.market_ticker,
                "side": signal.side,
                "signal_price_cents": signal.entry_price_cents,
                "execution_price_cents": execution_price_cents,
                "filled_contracts": 0,
                "edge": signal.edge,
                "payload": payload,
            }

        try:
            response = self.client.create_order(payload)
        except Exception as exc:
            self._trace_execution_event(
                {
                    "event": "new_profile_order_error",
                    "market_ticker": signal.market_ticker,
                    "error": str(exc),
                }
            )
            return {
                "status": "error",
                "market_ticker": signal.market_ticker,
                "error": str(exc),
                "filled_contracts": 0,
            }

        response_order = response.get("order", response) if isinstance(response, dict) else {}
        order = response_order if isinstance(response_order, dict) else {}
        order_status = _extract_order_status(order)
        order_id = _extract_order_id(response)
        filled_contracts = _extract_fill_count(order)

        if order_id and not (self.fill_feed is not None and self.fill_feed.is_healthy()):
            try:
                fills_response = self.client.get_fills(order_id=order_id, ticker=signal.market_ticker)
                exchange_filled = _filled_contracts_from_fills(fills_response)
                if exchange_filled is not None:
                    filled_contracts = exchange_filled
            except Exception:
                pass

        self._trace_execution_event(
            {
                "event": "new_profile_order_result",
                "strategy": signal.strategy,
                "market_ticker": signal.market_ticker,
                "side": signal.side,
                "signal_price_cents": signal.entry_price_cents,
                "execution_price_cents": execution_price_cents,
                "order_status": order_status,
                "filled_contracts": filled_contracts,
                "edge": signal.edge,
            }
        )

        return {
            "status": order_status or "submitted",
            "strategy": signal.strategy,
            "market_ticker": signal.market_ticker,
            "side": signal.side,
            "signal_price_cents": signal.entry_price_cents,
            "execution_price_cents": execution_price_cents,
            "filled_contracts": max(int(filled_contracts), 0),
            "order_id": order_id,
            "edge": signal.edge,
            "response": response,
        }

    def _submit_order(self, candidate: StrategyCandidate) -> dict[str, Any]:
        if self.config.enable_execution_sessions:
            return self._submit_order_via_execution_session(candidate)
        attempts: list[dict[str, Any]] = []
        current_candidate = candidate
        max_attempts = max(int(self.config.max_entry_retries), 0) + 1
        total_filled_contracts = 0
        last_status = "submitted"
        last_response: dict[str, Any] | None = None
        last_payload: dict[str, Any] | None = None
        last_execution_price_cents = min(candidate.price_cents + max(self.config.execution_cross_cents, 0), 99)

        for attempt_index in range(max_attempts):
            snapshot = current_candidate.snapshot
            remaining_contracts = max(int(current_candidate.contracts) - total_filled_contracts, 0)
            if remaining_contracts <= 0:
                break
            execution_price_cents = min(current_candidate.price_cents + max(self.config.execution_cross_cents, 0), 99)
            orderbook_available_contracts: int | None = None
            orderbook_payload: dict[str, Any] | None = None
            if self.config.use_orderbook_precheck:
                try:
                    orderbook_payload = self.client.get_orderbook(snapshot.market_ticker)
                    orderbook_available_contracts = _available_contracts_at_price(
                        orderbook_payload,
                        side=current_candidate.side,
                        limit_price_cents=execution_price_cents,
                    )
                except Exception:
                    orderbook_payload = None
                    orderbook_available_contracts = None
                min_fill_contracts = max(
                    1,
                    int(math.ceil(remaining_contracts * max(float(self.config.min_orderbook_fill_fraction), 0.0))),
                )
                if (
                    orderbook_available_contracts is not None
                    and orderbook_available_contracts < min_fill_contracts
                ):
                    attempts.append(
                        {
                            "attempt": attempt_index + 1,
                            "signal_price_cents": current_candidate.price_cents,
                            "execution_price_cents": execution_price_cents,
                            "orderbook_available_contracts": orderbook_available_contracts,
                            "min_fill_contracts": min_fill_contracts,
                            "status": "skipped_no_depth",
                            "orderbook": orderbook_payload,
                        }
                    )
                    return {
                        "status": "skipped_no_depth",
                        "side": current_candidate.side,
                        "strategy_name": current_candidate.strategy_name,
                        "confidence": current_candidate.confidence,
                        "gbm_probability": current_candidate.gbm_probability,
                        "gbm_edge": current_candidate.gbm_edge,
                        "signal_price_cents": current_candidate.price_cents,
                        "execution_price_cents": execution_price_cents,
                        "filled_contracts": total_filled_contracts,
                        "payload": None,
                        "response": None,
                        "attempts": attempts,
                        "orderbook_available_contracts": orderbook_available_contracts,
                    }
            payload = {
                "ticker": snapshot.market_ticker,
                "action": "buy",
                "side": current_candidate.side,
                "count": remaining_contracts,
                "type": "limit",
                "yes_price": execution_price_cents if current_candidate.side == "yes" else None,
                "no_price": execution_price_cents if current_candidate.side == "no" else None,
                "client_order_id": f"kabot-{snapshot.market_ticker}-{int(snapshot.observed_at.timestamp())}-{attempt_index}",
                "time_in_force": self.config.entry_time_in_force,
            }
            payload = {key: value for key, value in payload.items() if value is not None}

            if self.config.dry_run:
                return {
                    "status": "dry_run",
                    "side": current_candidate.side,
                    "strategy_name": current_candidate.strategy_name,
                    "confidence": current_candidate.confidence,
                    "gbm_probability": current_candidate.gbm_probability,
                    "gbm_edge": current_candidate.gbm_edge,
                    "signal_price_cents": current_candidate.price_cents,
                    "execution_price_cents": execution_price_cents,
                    "filled_contracts": 0,
                    "payload": payload,
                    "attempts": attempts,
                    "orderbook_available_contracts": orderbook_available_contracts,
                }

            response = self.client.create_order(payload)
            order_id = _extract_order_id(response)
            response_order = response.get("order", response) if isinstance(response, dict) else {}
            order_status = _extract_order_status(response_order if isinstance(response_order, dict) else {})
            filled_contracts = _extract_fill_count(response_order if isinstance(response_order, dict) else {})
            fills_response: dict[str, Any] | None = None
            exchange_filled_contracts: int | None = None
            if order_id:
                try:
                    fills_response = self.client.get_fills(order_id=order_id, ticker=snapshot.market_ticker)
                    exchange_filled_contracts = _filled_contracts_from_fills(fills_response)
                except Exception:
                    fills_response = None
                    exchange_filled_contracts = None
            if exchange_filled_contracts is not None:
                filled_contracts = exchange_filled_contracts
            total_filled_contracts += filled_contracts
            last_status = order_status or "submitted"
            last_response = response
            last_payload = payload
            last_execution_price_cents = execution_price_cents
            # If the order is resting (limit order sitting on the book), register it with
            # the fill feed so the next cycle skips the list_orders REST poll.
            if (
                order_id
                and order_status in {"resting", "open", "submitted", "pending"}
                and self.fill_feed is not None
            ):
                self.fill_feed.register_order(snapshot.market_ticker, order_id)
            if self.config.entry_time_in_force != "immediate_or_cancel" and order_status in {"resting", "open", "submitted", "pending"}:
                self.local_resting_entry_locks[snapshot.market_ticker] = LocalRestingEntryLock(
                    created_at=datetime.now(UTC),
                    side=current_candidate.side,
                    strategy_name=current_candidate.strategy_name,
                    price_cents=current_candidate.price_cents,
                    gbm_edge=current_candidate.gbm_edge,
                    gbm_probability=current_candidate.gbm_probability,
                    observed_at=snapshot.observed_at,
                    contracts=remaining_contracts,
                )
            attempt_record: dict[str, Any] = {
                "attempt": attempt_index + 1,
                "signal_price_cents": current_candidate.price_cents,
                "execution_price_cents": execution_price_cents,
                "payload": payload,
                "response": response,
                "order_id": order_id,
                "status": order_status or "submitted",
                "filled_contracts": filled_contracts,
                "exchange_filled_contracts": exchange_filled_contracts,
                "fills_response": fills_response,
                "orderbook_available_contracts": orderbook_available_contracts,
            }

            if self.config.entry_time_in_force == "immediate_or_cancel":
                attempts.append(attempt_record)
                remaining_contracts = max(int(current_candidate.contracts) - total_filled_contracts, 0)
                if remaining_contracts <= 0 or attempt_index >= max_attempts - 1:
                    return {
                        "status": order_status or "submitted",
                        "side": current_candidate.side,
                        "strategy_name": current_candidate.strategy_name,
                        "confidence": current_candidate.confidence,
                        "gbm_probability": current_candidate.gbm_probability,
                        "gbm_edge": current_candidate.gbm_edge,
                        "signal_price_cents": current_candidate.price_cents,
                        "execution_price_cents": execution_price_cents,
                        "filled_contracts": total_filled_contracts,
                        "payload": payload,
                        "response": response,
                        "attempts": attempts,
                        "exchange_filled_contracts": exchange_filled_contracts,
                        "orderbook_available_contracts": orderbook_available_contracts,
                    }
                time.sleep(max(float(self.config.ioc_retry_delay_seconds), 0.0))
                latest_market = self.client.get_market(snapshot.market_ticker)
                latest_raw = latest_market.get("market") if isinstance(latest_market.get("market"), dict) else latest_market
                refreshed_snapshot = normalize_market(
                    {**latest_raw, "series_ticker": self.config.series_ticker},
                    spot_price=snapshot.spot_price,
                    observed_at=datetime.now(UTC),
                    source="kalshi_live_ioc_retry",
                )
                refreshed_candidate = self._refresh_candidate(current_candidate=current_candidate, snapshot=refreshed_snapshot)
                if refreshed_candidate is None:
                    return {
                        "status": "ioc_not_retried",
                        "side": current_candidate.side,
                        "strategy_name": current_candidate.strategy_name,
                        "confidence": current_candidate.confidence,
                        "gbm_probability": current_candidate.gbm_probability,
                        "gbm_edge": current_candidate.gbm_edge,
                        "signal_price_cents": current_candidate.price_cents,
                        "execution_price_cents": execution_price_cents,
                        "filled_contracts": total_filled_contracts,
                        "payload": payload,
                        "response": response,
                        "attempts": attempts,
                        "exchange_filled_contracts": exchange_filled_contracts,
                        "orderbook_available_contracts": orderbook_available_contracts,
                    }
                current_candidate = StrategyCandidate(
                    strategy_name=refreshed_candidate.strategy_name,
                    confidence=refreshed_candidate.confidence,
                    snapshot=refreshed_candidate.snapshot,
                    side=refreshed_candidate.side,
                    price_cents=refreshed_candidate.price_cents,
                    contracts=max(int(refreshed_candidate.contracts), total_filled_contracts + remaining_contracts),
                    gbm_probability=refreshed_candidate.gbm_probability,
                    gbm_edge=refreshed_candidate.gbm_edge,
                )
                continue

            if attempt_index >= max_attempts - 1 or not order_id:
                attempts.append(attempt_record)
                return {
                    "status": order_status or "submitted",
                    "side": current_candidate.side,
                    "strategy_name": current_candidate.strategy_name,
                    "confidence": current_candidate.confidence,
                    "gbm_probability": current_candidate.gbm_probability,
                    "gbm_edge": current_candidate.gbm_edge,
                    "signal_price_cents": current_candidate.price_cents,
                    "execution_price_cents": execution_price_cents,
                    "filled_contracts": filled_contracts,
                    "payload": payload,
                    "response": response,
                    "attempts": attempts,
                    "exchange_filled_contracts": exchange_filled_contracts,
                    "orderbook_available_contracts": orderbook_available_contracts,
                }

            time.sleep(max(float(self.config.resting_order_retry_delay_seconds), 0.0))
            refreshed = self.client.get_order(order_id)
            refreshed_order = refreshed.get("order", refreshed) if isinstance(refreshed, dict) else {}
            refreshed_status = _extract_order_status(refreshed_order if isinstance(refreshed_order, dict) else {})
            refreshed_filled_contracts = _extract_fill_count(refreshed_order if isinstance(refreshed_order, dict) else {})
            try:
                refreshed_fills = self.client.get_fills(order_id=order_id, ticker=snapshot.market_ticker)
                refreshed_exchange_fills = _filled_contracts_from_fills(refreshed_fills)
                if refreshed_exchange_fills is not None:
                    refreshed_filled_contracts = refreshed_exchange_fills
            except Exception:
                pass
            attempt_record["post_wait_status"] = refreshed_status or "unknown"
            attempt_record["post_wait_response"] = refreshed
            attempt_record["post_wait_filled_contracts"] = refreshed_filled_contracts
            if refreshed_status not in {"resting", "open", "submitted", "pending"}:
                attempts.append(attempt_record)
                return {
                    "status": refreshed_status or order_status or "submitted",
                    "strategy_name": current_candidate.strategy_name,
                    "confidence": current_candidate.confidence,
                    "gbm_probability": current_candidate.gbm_probability,
                    "gbm_edge": current_candidate.gbm_edge,
                    "signal_price_cents": current_candidate.price_cents,
                    "execution_price_cents": execution_price_cents,
                    "filled_contracts": refreshed_filled_contracts,
                    "payload": payload,
                    "response": response,
                    "attempts": attempts,
                    "exchange_filled_contracts": exchange_filled_contracts,
                    "orderbook_available_contracts": orderbook_available_contracts,
                }

            cancel_response = self.client.cancel_order(order_id)
            attempt_record["cancel_response"] = cancel_response
            attempts.append(attempt_record)
            latest_market = self.client.get_market(snapshot.market_ticker)
            latest_raw = latest_market.get("market") if isinstance(latest_market.get("market"), dict) else latest_market
            refreshed_snapshot = normalize_market(
                {**latest_raw, "series_ticker": self.config.series_ticker},
                spot_price=snapshot.spot_price,
                observed_at=datetime.now(UTC),
                source="kalshi_live_retry",
            )
            refreshed_candidate = self._refresh_candidate(current_candidate=current_candidate, snapshot=refreshed_snapshot)
            if refreshed_candidate is None:
                return {
                    "status": "canceled_not_retried",
                    "side": current_candidate.side,
                    "strategy_name": current_candidate.strategy_name,
                    "confidence": current_candidate.confidence,
                    "gbm_probability": current_candidate.gbm_probability,
                    "gbm_edge": current_candidate.gbm_edge,
                    "signal_price_cents": current_candidate.price_cents,
                    "execution_price_cents": execution_price_cents,
                    "filled_contracts": refreshed_filled_contracts,
                    "payload": payload,
                    "response": response,
                    "attempts": attempts,
                    "exchange_filled_contracts": exchange_filled_contracts,
                    "orderbook_available_contracts": orderbook_available_contracts,
                }
            current_candidate = refreshed_candidate

        return {
            "status": last_status if attempts else "abandoned",
            "side": candidate.side,
            "strategy_name": candidate.strategy_name,
            "confidence": candidate.confidence,
            "gbm_probability": candidate.gbm_probability,
            "gbm_edge": candidate.gbm_edge,
            "signal_price_cents": candidate.price_cents,
            "execution_price_cents": last_execution_price_cents,
            "filled_contracts": total_filled_contracts,
            "payload": last_payload,
            "response": last_response,
            "attempts": attempts,
            "exchange_filled_contracts": total_filled_contracts,
        }

    def _refresh_candidate(
        self,
        *,
        current_candidate: StrategyCandidate,
        snapshot: MarketSnapshot,
    ) -> StrategyCandidate | None:
        if snapshot.contract_type != "threshold" or snapshot.threshold is None:
            return None
        if (snapshot.volume or 0.0) < self.config.min_market_volume:
            return None
        rule = next((item for item in self._strategy_rules() if item.name == current_candidate.strategy_name), None)
        if rule is None:
            return None
        tte_minutes = _time_to_expiry_minutes(snapshot)
        if not (rule.min_tte_minutes < tte_minutes < rule.max_tte_minutes):
            return None
        if rule.side == "yes" and snapshot.spot_price < (snapshot.threshold + self.config.distance_threshold_dollars):
            return None
        if rule.side == "no" and snapshot.spot_price > (snapshot.threshold - self.config.distance_threshold_dollars):
            return None
        ask_cents, bid_cents = _side_prices(snapshot, rule.side)
        if ask_cents is None or bid_cents is None:
            return None
        spread_cents = ask_cents - bid_cents
        if not (rule.min_price_cents <= ask_cents <= rule.max_price_cents):
            return None
        if spread_cents >= rule.max_spread_cents:
            return None
        volatility = self._estimate_live_volatility(observed_at=snapshot.observed_at)
        if volatility is None:
            return None
        estimate = self.model.estimate(snapshot, volatility=volatility)
        market_probability = ask_cents / 100.0
        side_probability = estimate.probability if rule.side == "yes" else (1.0 - estimate.probability)
        edge = side_probability - market_probability
        if edge < self._required_gbm_edge(strategy_name=rule.name, ask_cents=ask_cents):
            return None
        confidence = _confidence_for_candidate(
            rule_name=rule.name,
            tte_minutes=tte_minutes,
            ask_cents=ask_cents,
            spread_cents=spread_cents,
        )
        return StrategyCandidate(
            strategy_name=rule.name,
            confidence=confidence,
            snapshot=snapshot,
            side=rule.side,
            price_cents=ask_cents,
            contracts=_effective_contracts_for_price(ask_cents=ask_cents, contracts=current_candidate.contracts),
            gbm_probability=side_probability,
            gbm_edge=edge,
        )


def build_live_trader(*, store: PostgresStore, config: LiveTraderConfig) -> LiveTrader:
    api_key_id = os.getenv("KABOT_KALSHI_API_KEY_ID", "").strip()
    private_key_path = os.getenv("KABOT_KALSHI_PRIVATE_KEY_PATH", "").strip()
    if not api_key_id or not private_key_path:
        raise ValueError("Missing KABOT_KALSHI_API_KEY_ID or KABOT_KALSHI_PRIVATE_KEY_PATH")
    signer = KalshiAuthSigner(KalshiAuthConfig(api_key_id=api_key_id, private_key_path=private_key_path))
    client = KabotKalshiClient(auth_signer=signer)
    return LiveTrader(store=store, client=client, config=config)

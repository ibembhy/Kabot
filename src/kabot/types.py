from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal


ContractType = Literal["threshold", "range", "direction"]
SignalAction = Literal["buy_yes", "buy_no", "no_action"]
PositionStatus = Literal["open", "closed"]
ExitAction = Literal["hold", "exit"]
ExitTrigger = Literal["take_profit", "stop_loss", "fair_value_convergence", "time_exit", "settlement"]


@dataclass(frozen=True)
class MarketSnapshot:
    source: str
    series_ticker: str
    market_ticker: str
    contract_type: ContractType
    underlying_symbol: str
    observed_at: datetime
    expiry: datetime
    spot_price: float
    threshold: float | None = None
    range_low: float | None = None
    range_high: float | None = None
    direction: str | None = None
    yes_bid: float | None = None
    yes_ask: float | None = None
    no_bid: float | None = None
    no_ask: float | None = None
    mid_price: float | None = None
    implied_probability: float | None = None
    volume: float | None = None
    open_interest: float | None = None
    settlement_price: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def time_to_expiry_years(self) -> float:
        seconds = max((self.expiry - self.observed_at).total_seconds(), 0.0)
        return seconds / (365.0 * 24.0 * 60.0 * 60.0)


@dataclass(frozen=True)
class ProbabilityEstimate:
    model_name: str
    observed_at: datetime
    expiry: datetime
    spot_price: float
    target_price: float
    volatility: float
    drift: float
    probability: float
    inputs: dict[str, Any] = field(default_factory=dict)
    notes: str | None = None


@dataclass(frozen=True)
class TradingSignal:
    market_ticker: str
    action: SignalAction
    side: Literal["yes", "no"] | None
    model_probability: float
    market_probability: float | None
    edge: float | None
    entry_price_cents: int | None
    fair_value_cents: int | None
    expected_value_cents: float | None
    reason: str


@dataclass
class Position:
    position_id: str
    market_ticker: str
    side: Literal["yes", "no"]
    contracts: int
    entry_time: datetime
    entry_price_cents: int
    expiry: datetime | None = None
    status: PositionStatus = "open"
    exit_time: datetime | None = None
    exit_price_cents: int | None = None
    exit_trigger: ExitTrigger | None = None
    realized_pnl: float | None = None


@dataclass(frozen=True)
class ExitDecision:
    action: ExitAction
    trigger: ExitTrigger | None
    exit_price_cents: int | None
    unrealized_pnl: float | None
    reason: str


@dataclass(frozen=True)
class BacktestResult:
    strategy_mode: str
    trades: Any
    summary: dict[str, Any]


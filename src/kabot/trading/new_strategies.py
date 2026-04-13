"""NEW profile strategy logic.

Strategy A: Hold-to-settlement on high-confidence GBM signals.
Strategy B: Fade the fast BTC move.

Both strategies are pure functions over market state.
No HTTP calls, no side effects.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from kabot.models.gbm_threshold import probability_for_snapshot
from kabot.types import MarketSnapshot


@dataclass(frozen=True)
class NewSignal:
    strategy: Literal["hold_settlement", "fade_move"]
    market_ticker: str
    side: Literal["yes", "no"]
    entry_price_cents: int
    model_probability: float
    market_probability: float
    edge: float
    tte_minutes: float
    reason: str


@dataclass(frozen=True)
class StrategyAConfig:
    min_edge: float = 0.08
    min_tte_minutes: float = 8.0
    max_tte_minutes: float = 14.0
    min_price_cents: int = 20
    max_price_cents: int = 80
    max_spread_cents: int = 8
    min_volume: float = 1000.0


def evaluate_strategy_a(
    snapshot: MarketSnapshot,
    *,
    volatility: float,
    config: StrategyAConfig,
    observed_at: datetime,
) -> NewSignal | None:
    """Return a signal if Strategy A conditions are met, else None."""
    if snapshot.contract_type != "threshold" or snapshot.threshold is None:
        return None
    if (snapshot.volume or 0.0) < config.min_volume:
        return None

    tte_minutes = max((snapshot.expiry - observed_at).total_seconds() / 60.0, 0.0)
    if not (config.min_tte_minutes < tte_minutes <= config.max_tte_minutes):
        return None

    sigma = max(volatility, 0.05)
    prob_yes = probability_for_snapshot(snapshot, volatility=sigma, drift=0.0)

    best_signal: NewSignal | None = None
    for side, model_prob, ask, bid in [
        ("yes", prob_yes, snapshot.yes_ask, snapshot.yes_bid),
        ("no", 1.0 - prob_yes, snapshot.no_ask, snapshot.no_bid),
    ]:
        if ask is None or bid is None:
            continue
        ask_cents = int(round(ask * 100.0))
        bid_cents = int(round(bid * 100.0))
        spread_cents = ask_cents - bid_cents
        if not (config.min_price_cents <= ask_cents <= config.max_price_cents):
            continue
        if spread_cents > config.max_spread_cents:
            continue
        edge = model_prob - ask
        signal = NewSignal(
            strategy="hold_settlement",
            market_ticker=snapshot.market_ticker,
            side=side,  # type: ignore[arg-type]
            entry_price_cents=ask_cents,
            model_probability=model_prob,
            market_probability=ask,
            edge=edge,
            tte_minutes=tte_minutes,
            reason=f"StrategyA edge={edge:.4f} tte={tte_minutes:.1f}min",
        )
        if best_signal is None or signal.edge > best_signal.edge:
            best_signal = signal

    if best_signal is None or best_signal.edge < config.min_edge:
        return None
    return best_signal


@dataclass(frozen=True)
class StrategyBConfig:
    min_velocity_bps_per_second: float = 5.0
    min_total_move_bps: float = 150.0
    min_edge: float = 0.08
    min_tte_minutes: float = 4.0
    max_tte_minutes: float = 12.0
    min_price_cents: int = 15
    max_price_cents: int = 85
    max_spread_cents: int = 12
    min_volume: float = 500.0
    vol_spike_multiplier: float = 1.5


def evaluate_strategy_b(
    snapshot: MarketSnapshot,
    *,
    base_volatility: float,
    move_direction: int,
    move_bps: float,
    config: StrategyBConfig,
    observed_at: datetime,
) -> NewSignal | None:
    """Return a fade signal if Strategy B conditions are met, else None."""
    if snapshot.contract_type != "threshold" or snapshot.threshold is None:
        return None
    if move_direction == 0:
        return None
    if (snapshot.volume or 0.0) < config.min_volume:
        return None

    tte_minutes = max((snapshot.expiry - observed_at).total_seconds() / 60.0, 0.0)
    if not (config.min_tte_minutes < tte_minutes <= config.max_tte_minutes):
        return None

    sigma = max(base_volatility * config.vol_spike_multiplier, 0.05)
    prob_yes = probability_for_snapshot(snapshot, volatility=sigma, drift=0.0)

    if move_direction > 0:
        fade_side: Literal["yes", "no"] = "no"
        fade_model_prob = 1.0 - prob_yes
        fade_ask = snapshot.no_ask
        fade_bid = snapshot.no_bid
    else:
        fade_side = "yes"
        fade_model_prob = prob_yes
        fade_ask = snapshot.yes_ask
        fade_bid = snapshot.yes_bid

    if fade_ask is None or fade_bid is None:
        return None

    ask_cents = int(round(fade_ask * 100.0))
    bid_cents = int(round(fade_bid * 100.0))
    spread_cents = ask_cents - bid_cents

    if not (config.min_price_cents <= ask_cents <= config.max_price_cents):
        return None
    if spread_cents > config.max_spread_cents:
        return None

    edge = fade_model_prob - fade_ask
    if edge < config.min_edge:
        return None

    return NewSignal(
        strategy="fade_move",
        market_ticker=snapshot.market_ticker,
        side=fade_side,
        entry_price_cents=ask_cents,
        model_probability=fade_model_prob,
        market_probability=fade_ask,
        edge=edge,
        tte_minutes=tte_minutes,
        reason=(
            f"StrategyB fade={'NO' if move_direction > 0 else 'YES'} "
            f"move={move_bps:.0f}bps edge={edge:.4f} spiked_vol={sigma:.3f}"
        ),
    )

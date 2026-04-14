"""DAILY profile strategy logic for KXBTCD (60-minute BTC contracts).

Strategy: buy high-probability contracts (65c+) using GTC limit orders
as a maker. On a 60-minute contract there is time to fill and exit.

Exit early when:
- Market reprices to our fair value (take the profit)
- GBM edge flips negative (model now disagrees)
- Stop loss hit

Hold to settlement when TTE < 8 minutes.

Academic basis: Kalshi high-price contracts (65c+) systematically
outperform their implied probability across 300k+ contracts studied.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from kabot.models.gbm_threshold import probability_for_snapshot
from kabot.types import MarketSnapshot


@dataclass(frozen=True)
class DailySignal:
    market_ticker: str
    side: Literal["yes", "no"]
    entry_price_cents: int
    model_probability: float
    market_probability: float
    edge: float
    tte_minutes: float
    fair_value_cents: int
    reason: str


@dataclass(frozen=True)
class DailySignalConfig:
    min_edge: float = 0.04
    min_price_cents: int = 40
    max_price_cents: int = 95
    min_tte_minutes: float = 8.0
    max_tte_minutes: float = 50.0
    min_distance_dollars: float = 200.0
    max_spread_cents: int = 6
    min_volume: float = 0.0


def evaluate_daily_signal(
    snapshot: MarketSnapshot,
    *,
    volatility: float,
    config: DailySignalConfig,
    observed_at: datetime,
) -> DailySignal | str:
    """Return the best-edge signal if DAILY conditions are met, else a rejection reason.

    Evaluates both YES and NO. Returns the side with the highest edge
    that passes all filters. Never returns the first side found.
    """
    if snapshot.contract_type != "threshold" or snapshot.threshold is None:
        return "not_threshold"
    if (snapshot.volume or 0.0) < config.min_volume:
        return "low_volume"

    tte_minutes = max((snapshot.expiry - observed_at).total_seconds() / 60.0, 0.0)
    if not (config.min_tte_minutes < tte_minutes <= config.max_tte_minutes):
        return "tte_out_of_window"

    distance = abs(snapshot.spot_price - snapshot.threshold)
    if distance < config.min_distance_dollars:
        return "distance_too_small"

    from kabot.trading.daily_vol import DAILY_VOL_FLOOR

    sigma = max(volatility, DAILY_VOL_FLOOR)
    prob_yes = probability_for_snapshot(snapshot, volatility=sigma, drift=0.0)

    best_signal: DailySignal | None = None
    saw_price_range = False
    saw_acceptable_spread = False

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
        saw_price_range = True
        if spread_cents > config.max_spread_cents:
            continue
        saw_acceptable_spread = True

        edge = model_prob - ask
        if edge < config.min_edge:
            continue

        fair_value_cents = int(round(model_prob * 100.0))
        signal = DailySignal(
            market_ticker=snapshot.market_ticker,
            side=side,
            entry_price_cents=ask_cents,
            model_probability=model_prob,
            market_probability=ask,
            edge=edge,
            tte_minutes=tte_minutes,
            fair_value_cents=fair_value_cents,
            reason=(
                f"DAILY {side.upper()} edge={edge:.4f} "
                f"tte={tte_minutes:.0f}min dist=${distance:.0f}"
            ),
        )
        if best_signal is None or signal.edge > best_signal.edge:
            best_signal = signal

    if best_signal is None:
        if not saw_price_range:
            return "no_side_in_price_range"
        if not saw_acceptable_spread:
            return "spread_too_wide"
        return "edge_too_low"
    return best_signal


def evaluate_daily_signal_debug(
    snapshot: MarketSnapshot,
    *,
    volatility: float,
    config: DailySignalConfig,
    observed_at: datetime,
) -> dict[str, object]:
    """Return DAILY signal gating details for trace/debug logs."""
    tte_minutes = max((snapshot.expiry - observed_at).total_seconds() / 60.0, 0.0)
    distance_dollars = (
        abs(snapshot.spot_price - snapshot.threshold)
        if snapshot.threshold is not None
        else 0.0
    )
    debug: dict[str, object] = {
        "market_ticker": snapshot.market_ticker,
        "rejected": True,
        "reason": "not_threshold",
        "tte_minutes": tte_minutes,
        "distance_dollars": distance_dollars,
        "best_ask_cents": None,
        "best_edge": None,
    }

    if snapshot.contract_type != "threshold" or snapshot.threshold is None:
        return debug
    if (snapshot.volume or 0.0) < config.min_volume:
        debug["reason"] = "low_volume"
        return debug
    if not (config.min_tte_minutes < tte_minutes <= config.max_tte_minutes):
        debug["reason"] = "tte_out_of_window"
        return debug
    if distance_dollars < config.min_distance_dollars:
        debug["reason"] = "distance_too_small"
        return debug

    from kabot.trading.daily_vol import DAILY_VOL_FLOOR

    sigma = max(volatility, DAILY_VOL_FLOOR)
    prob_yes = probability_for_snapshot(snapshot, volatility=sigma, drift=0.0)

    saw_price_range = False
    saw_acceptable_spread = False
    best_ask_cents: int | None = None
    best_edge: float | None = None

    for model_prob, ask, bid in [
        (prob_yes, snapshot.yes_ask, snapshot.yes_bid),
        (1.0 - prob_yes, snapshot.no_ask, snapshot.no_bid),
    ]:
        if ask is None or bid is None:
            continue
        ask_cents = int(round(ask * 100.0))
        bid_cents = int(round(bid * 100.0))
        edge = model_prob - ask
        if best_edge is None or edge > best_edge:
            best_edge = edge
            best_ask_cents = ask_cents
        if not (config.min_price_cents <= ask_cents <= config.max_price_cents):
            continue
        saw_price_range = True
        if ask_cents - bid_cents > config.max_spread_cents:
            continue
        saw_acceptable_spread = True
        if edge >= config.min_edge:
            debug.update(
                {
                    "rejected": False,
                    "reason": "passed",
                    "best_ask_cents": ask_cents,
                    "best_edge": edge,
                }
            )
            return debug

    debug["best_ask_cents"] = best_ask_cents
    debug["best_edge"] = best_edge
    if not saw_price_range:
        debug["reason"] = "no_side_in_price_range"
    elif not saw_acceptable_spread:
        debug["reason"] = "spread_too_wide"
    else:
        debug["reason"] = "edge_too_low"
    return debug


@dataclass(frozen=True)
class DailyExitConfig:
    fair_value_buffer_cents: int = 3
    negative_edge_threshold: float = -0.04
    min_tte_to_exit_minutes: float = 8.0
    stop_loss_cents: int = 15


@dataclass(frozen=True)
class DailyExitDecision:
    action: Literal["hold", "exit"]
    trigger: str | None
    exit_price_cents: int | None
    unrealized_pnl_cents: int
    reason: str


def evaluate_daily_exit(
    snapshot: MarketSnapshot,
    *,
    side: str,
    entry_price_cents: int,
    fair_value_cents: int,
    contracts: int,
    volatility: float,
    observed_at: datetime,
    config: DailyExitConfig,
) -> DailyExitDecision:
    """Evaluate whether to exit a DAILY position early."""
    bid = snapshot.yes_bid if side == "yes" else snapshot.no_bid
    if bid is None:
        return DailyExitDecision(
            action="hold",
            trigger=None,
            exit_price_cents=None,
            unrealized_pnl_cents=0,
            reason="No bid available.",
        )

    current_bid_cents = int(round(bid * 100.0))
    pnl_per_contract = current_bid_cents - entry_price_cents
    total_pnl = pnl_per_contract * contracts
    tte_minutes = max((snapshot.expiry - observed_at).total_seconds() / 60.0, 0.0)

    if tte_minutes <= config.min_tte_to_exit_minutes:
        return DailyExitDecision(
            action="hold",
            trigger=None,
            exit_price_cents=current_bid_cents,
            unrealized_pnl_cents=total_pnl,
            reason=f"TTE {tte_minutes:.1f}min; holding to settlement.",
        )

    if pnl_per_contract <= -config.stop_loss_cents:
        return DailyExitDecision(
            action="exit",
            trigger="stop_loss",
            exit_price_cents=current_bid_cents,
            unrealized_pnl_cents=total_pnl,
            reason=f"Stop loss: {pnl_per_contract}c per contract.",
        )

    if current_bid_cents >= fair_value_cents - config.fair_value_buffer_cents and pnl_per_contract > 0:
        return DailyExitDecision(
            action="exit",
            trigger="fair_value_convergence",
            exit_price_cents=current_bid_cents,
            unrealized_pnl_cents=total_pnl,
            reason=f"Fair value reached: bid={current_bid_cents}c fv={fair_value_cents}c.",
        )

    if snapshot.contract_type == "threshold" and snapshot.threshold is not None:
        from kabot.trading.daily_vol import DAILY_VOL_FLOOR

        sigma = max(volatility, DAILY_VOL_FLOOR)
        prob_yes = probability_for_snapshot(snapshot, volatility=sigma, drift=0.0)
        side_prob = prob_yes if side == "yes" else (1.0 - prob_yes)
        current_edge = side_prob - bid
        if current_edge < config.negative_edge_threshold:
            return DailyExitDecision(
                action="exit",
                trigger="negative_edge",
                exit_price_cents=current_bid_cents,
                unrealized_pnl_cents=total_pnl,
                reason=f"Edge flipped: {current_edge:.4f} < {config.negative_edge_threshold}.",
            )

    return DailyExitDecision(
        action="hold",
        trigger=None,
        exit_price_cents=current_bid_cents,
        unrealized_pnl_cents=total_pnl,
        reason="No exit condition met.",
    )

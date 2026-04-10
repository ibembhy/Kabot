from __future__ import annotations

from dataclasses import dataclass

from kabot.signals.engine import probability_to_cents
from kabot.types import ExitDecision, MarketSnapshot


@dataclass(frozen=True)
class ExitConfig:
    take_profit_cents: int | None = None
    stop_loss_cents: int | None = None
    fair_value_buffer_cents: int = 0
    time_exit_minutes: int | None = None


def exit_price_cents(snapshot: MarketSnapshot, side: str) -> int | None:
    probability = snapshot.yes_bid if side == "yes" else snapshot.no_bid
    if probability is None:
        return None
    return probability_to_cents(probability)


def evaluate_exit(
    snapshot: MarketSnapshot,
    *,
    side: str,
    entry_price_cents: int,
    fair_value_cents: int,
    contracts: int,
    config: ExitConfig,
) -> ExitDecision:
    current_price_cents = exit_price_cents(snapshot, side)
    if current_price_cents is None:
        return ExitDecision(
            action="hold",
            trigger=None,
            exit_price_cents=None,
            unrealized_pnl=None,
            reason="No bid available for exit.",
        )

    pnl_per_contract = current_price_cents - entry_price_cents
    unrealized_pnl = pnl_per_contract * contracts

    if config.take_profit_cents is not None and pnl_per_contract >= config.take_profit_cents:
        return ExitDecision(
            action="exit",
            trigger="take_profit",
            exit_price_cents=current_price_cents,
            unrealized_pnl=unrealized_pnl,
            reason="Take-profit threshold reached.",
        )
    if config.stop_loss_cents is not None and pnl_per_contract <= -config.stop_loss_cents:
        return ExitDecision(
            action="exit",
            trigger="stop_loss",
            exit_price_cents=current_price_cents,
            unrealized_pnl=unrealized_pnl,
            reason="Stop-loss threshold reached.",
        )
    if current_price_cents >= fair_value_cents - config.fair_value_buffer_cents and pnl_per_contract > 0:
        return ExitDecision(
            action="exit",
            trigger="fair_value_convergence",
            exit_price_cents=current_price_cents,
            unrealized_pnl=unrealized_pnl,
            reason="Market converged to fair value.",
        )
    if config.time_exit_minutes is not None:
        minutes_to_expiry = max((snapshot.expiry - snapshot.observed_at).total_seconds() / 60.0, 0.0)
        if minutes_to_expiry <= config.time_exit_minutes:
            return ExitDecision(
                action="exit",
                trigger="time_exit",
                exit_price_cents=current_price_cents,
                unrealized_pnl=unrealized_pnl,
                reason="Time exit threshold reached.",
            )

    return ExitDecision(
        action="hold",
        trigger=None,
        exit_price_cents=current_price_cents,
        unrealized_pnl=unrealized_pnl,
        reason="No exit condition met.",
    )

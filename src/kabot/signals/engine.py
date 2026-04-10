from __future__ import annotations

from dataclasses import dataclass

from kabot.types import MarketSnapshot, ProbabilityEstimate, TradingSignal


@dataclass(frozen=True)
class SignalConfig:
    min_edge: float
    min_contract_price_cents: int
    max_contract_price_cents: int
    max_spread_cents: int = 99
    max_near_money_bps: float | None = None
    min_confidence: float = 0.0


def probability_to_cents(probability: float) -> int:
    return int(round(max(0.0, min(1.0, probability)) * 100.0))


def _entry_price_and_probability(snapshot: MarketSnapshot, side: str) -> tuple[int | None, float | None]:
    if side == "yes":
        if snapshot.yes_ask is None:
            return None, None
        return probability_to_cents(snapshot.yes_ask), snapshot.yes_ask
    if snapshot.no_ask is None:
        return None, None
    return probability_to_cents(snapshot.no_ask), snapshot.no_ask


def _exit_probability(snapshot: MarketSnapshot, side: str) -> float | None:
    return snapshot.yes_bid if side == "yes" else snapshot.no_bid


def _spread_cents(snapshot: MarketSnapshot, side: str) -> int | None:
    entry_price_cents, _ = _entry_price_and_probability(snapshot, side)
    exit_probability = _exit_probability(snapshot, side)
    if entry_price_cents is None or exit_probability is None:
        return None
    return entry_price_cents - probability_to_cents(exit_probability)


def generate_signal(snapshot: MarketSnapshot, estimate: ProbabilityEstimate, config: SignalConfig) -> TradingSignal:
    if (
        config.max_near_money_bps is not None
        and snapshot.threshold is not None
        and snapshot.spot_price > 0
    ):
        distance_bps = abs(snapshot.threshold - snapshot.spot_price) / snapshot.spot_price * 10_000.0
        if distance_bps > config.max_near_money_bps:
            return TradingSignal(
                market_ticker=snapshot.market_ticker,
                action="no_action",
                side=None,
                model_probability=estimate.probability,
                market_probability=snapshot.implied_probability,
                edge=None,
                entry_price_cents=None,
                fair_value_cents=None,
                expected_value_cents=None,
                reason="Contract is too far from spot.",
            )

    best_signal: TradingSignal | None = None
    for side, model_probability in (("yes", estimate.probability), ("no", 1.0 - estimate.probability)):
        entry_price_cents, market_probability = _entry_price_and_probability(snapshot, side)
        if entry_price_cents is None or market_probability is None:
            continue
        if entry_price_cents < config.min_contract_price_cents or entry_price_cents > config.max_contract_price_cents:
            continue
        spread_cents = _spread_cents(snapshot, side)
        if spread_cents is not None and spread_cents > config.max_spread_cents:
            continue
        edge = model_probability - market_probability
        fair_value_cents = probability_to_cents(model_probability)
        expected_value_cents = round(fair_value_cents - entry_price_cents, 2)
        action = "no_action"
        reason = "Edge below threshold."
        if edge >= config.min_edge and model_probability >= config.min_confidence and expected_value_cents > 0:
            action = "buy_yes" if side == "yes" else "buy_no"
            reason = f"Edge {edge:.4f} exceeds threshold."
        signal = TradingSignal(
            market_ticker=snapshot.market_ticker,
            action=action,  # type: ignore[arg-type]
            side=side,  # type: ignore[arg-type]
            model_probability=model_probability,
            market_probability=market_probability,
            edge=edge,
            entry_price_cents=entry_price_cents,
            fair_value_cents=fair_value_cents,
            expected_value_cents=expected_value_cents,
            reason=reason,
        )
        if best_signal is None:
            best_signal = signal
        elif (signal.edge or -999.0, signal.expected_value_cents or -999.0) > (
            best_signal.edge or -999.0,
            best_signal.expected_value_cents or -999.0,
        ):
            best_signal = signal

    if best_signal is not None:
        return best_signal
    return TradingSignal(
        market_ticker=snapshot.market_ticker,
        action="no_action",
        side=None,
        model_probability=estimate.probability,
        market_probability=snapshot.implied_probability,
        edge=None,
        entry_price_cents=None,
        fair_value_cents=None,
        expected_value_cents=None,
        reason="No tradable prices inside configured entry range.",
    )

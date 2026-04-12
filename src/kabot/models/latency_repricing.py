from __future__ import annotations

import math
from dataclasses import dataclass

from kabot.models.base import ProbabilityModel
from kabot.models.gbm_threshold import clamp, probability_for_snapshot
from kabot.types import MarketSnapshot, ProbabilityEstimate


@dataclass
class LatencyRepricingModel(ProbabilityModel):
    drift: float = 0.0
    volatility_floor: float = 0.05
    persistence_factor: float = 0.75
    min_move_bps: float = 3.0
    max_move_bps: float = 100.0
    model_name: str = "latency_repricing"

    def estimate(self, snapshot: MarketSnapshot, volatility: float) -> ProbabilityEstimate:
        sigma = max(volatility, self.volatility_floor)
        recent_log_return = float(snapshot.metadata.get("recent_log_return", 0.0) or 0.0)
        recent_move_bps = abs(recent_log_return) * 10_000.0
        base_probability = probability_for_snapshot(
            snapshot,
            volatility=sigma,
            drift=self.drift,
        )

        effective_spot = snapshot.spot_price
        blended_probability = base_probability
        repriced_probability = base_probability
        blend_weight = 0.0

        if recent_move_bps >= self.min_move_bps:
            capped_move_bps = min(recent_move_bps, self.max_move_bps)
            impulse_return = math.copysign(capped_move_bps / 10_000.0, recent_log_return)
            effective_spot = snapshot.spot_price * math.exp(impulse_return)
            repriced_probability = probability_for_snapshot(
                snapshot,
                spot_price=effective_spot,
                volatility=sigma,
                drift=self.drift,
            )
            blend_weight = clamp((capped_move_bps / self.max_move_bps) * self.persistence_factor, 0.0, 1.0)
            blended_probability = clamp(
                base_probability + blend_weight * (repriced_probability - base_probability),
                0.0,
                1.0,
            )

        target = snapshot.threshold
        if snapshot.contract_type == "range":
            target = snapshot.range_high if snapshot.range_high is not None else snapshot.range_low
        elif snapshot.contract_type == "direction":
            target = snapshot.threshold if snapshot.threshold is not None else snapshot.spot_price

        return ProbabilityEstimate(
            model_name=self.model_name,
            observed_at=snapshot.observed_at,
            expiry=snapshot.expiry,
            spot_price=snapshot.spot_price,
            target_price=float(target if target is not None else snapshot.spot_price),
            volatility=sigma,
            drift=self.drift,
            probability=blended_probability,
            inputs={
                "contract_type": snapshot.contract_type,
                "threshold": snapshot.threshold,
                "range_low": snapshot.range_low,
                "range_high": snapshot.range_high,
                "direction": snapshot.direction,
                "recent_log_return": recent_log_return,
                "recent_move_bps": round(recent_move_bps, 4),
                "effective_spot": round(effective_spot, 4),
                "base_probability": round(base_probability, 6),
                "repriced_probability": round(repriced_probability, 6),
                "blend_weight": round(blend_weight, 6),
            },
            notes="GBM baseline repriced with recent BTC move persistence.",
        )

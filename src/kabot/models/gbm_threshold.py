from __future__ import annotations

import math
from dataclasses import dataclass

from kabot.models.base import ProbabilityModel
from kabot.types import MarketSnapshot, ProbabilityEstimate


def norm_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def terminal_probability_above(
    spot_price: float,
    target_price: float,
    time_to_expiry_years: float,
    volatility: float,
    drift: float = 0.0,
) -> float:
    if target_price <= 0 or spot_price <= 0:
        raise ValueError("spot_price and target_price must be positive")
    if time_to_expiry_years <= 0:
        return 1.0 if spot_price >= target_price else 0.0
    sigma = max(volatility, 1e-12)
    denom = sigma * math.sqrt(time_to_expiry_years)
    numerator = math.log(spot_price / target_price) + (drift - 0.5 * sigma * sigma) * time_to_expiry_years
    return clamp(norm_cdf(numerator / denom), 0.0, 1.0)


def probability_for_snapshot(
    snapshot: MarketSnapshot,
    *,
    volatility: float,
    drift: float,
    spot_price: float | None = None,
) -> float:
    s = spot_price if spot_price is not None else snapshot.spot_price
    t = snapshot.time_to_expiry_years
    if snapshot.contract_type == "threshold":
        if snapshot.threshold is None:
            raise ValueError("threshold contract missing threshold")
        above = terminal_probability_above(s, snapshot.threshold, t, volatility, drift)
        if snapshot.direction == "below":
            return 1.0 - above
        return above
    if snapshot.contract_type == "range":
        if snapshot.range_low is None or snapshot.range_high is None:
            raise ValueError("range contract missing bounds")
        lower = terminal_probability_above(s, snapshot.range_low, t, volatility, drift)
        upper = terminal_probability_above(s, snapshot.range_high, t, volatility, drift)
        return clamp(lower - upper, 0.0, 1.0)
    if snapshot.contract_type == "direction":
        reference = snapshot.threshold if snapshot.threshold is not None else s
        above = terminal_probability_above(s, reference, t, volatility, drift)
        if snapshot.direction == "down":
            return 1.0 - above
        return above
    raise ValueError(f"Unsupported contract type: {snapshot.contract_type}")


@dataclass
class GBMThresholdModel(ProbabilityModel):
    drift: float = 0.0
    volatility_floor: float = 0.05
    model_name: str = "gbm_threshold"

    def estimate(self, snapshot: MarketSnapshot, volatility: float) -> ProbabilityEstimate:
        sigma = max(volatility, self.volatility_floor)
        probability = probability_for_snapshot(snapshot, volatility=sigma, drift=self.drift)
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
            probability=probability,
            inputs={
                "contract_type": snapshot.contract_type,
                "threshold": snapshot.threshold,
                "range_low": snapshot.range_low,
                "range_high": snapshot.range_high,
                "direction": snapshot.direction,
            },
            notes="Baseline GBM threshold probability.",
        )

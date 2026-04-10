from __future__ import annotations

from abc import ABC, abstractmethod

from kabot.types import MarketSnapshot, ProbabilityEstimate


class ProbabilityModel(ABC):
    @abstractmethod
    def estimate(self, snapshot: MarketSnapshot, volatility: float) -> ProbabilityEstimate:
        raise NotImplementedError

"""BTC price velocity detector.

Measures how fast BTC is moving in bps/second over a short window.
Used by Strategy B (Fade the Fast Move) to detect fast-move events.
"""
from __future__ import annotations

import math
from collections import deque
from datetime import datetime, timedelta
from typing import NamedTuple


class VelocityReading(NamedTuple):
    bps_per_second: float
    direction: int
    move_bps: float
    window_seconds: float


class BTCVelocityDetector:
    """Maintains a rolling window of BTC spot prices and computes velocity."""

    def __init__(self, window_seconds: float = 30.0, min_points: int = 3) -> None:
        self._window_seconds = window_seconds
        self._min_points = min_points
        self._history: deque[tuple[datetime, float]] = deque()

    def update(self, observed_at: datetime, spot_price: float) -> None:
        """Record a new spot price observation."""
        self._history.append((observed_at, float(spot_price)))
        cutoff = observed_at - timedelta(seconds=self._window_seconds)
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()

    def reading(self) -> VelocityReading:
        """Return current velocity. Returns zero reading if insufficient data."""
        if len(self._history) < self._min_points:
            return VelocityReading(0.0, 0, 0.0, 0.0)
        oldest_ts, oldest_price = self._history[0]
        newest_ts, newest_price = self._history[-1]
        window_seconds = max((newest_ts - oldest_ts).total_seconds(), 1e-6)
        if oldest_price <= 0:
            return VelocityReading(0.0, 0, 0.0, window_seconds)
        log_return = math.log(newest_price / oldest_price)
        move_bps = log_return * 10_000.0
        bps_per_second = abs(move_bps) / window_seconds
        direction = 1 if move_bps > 0 else (-1 if move_bps < 0 else 0)
        return VelocityReading(
            bps_per_second=bps_per_second,
            direction=direction,
            move_bps=move_bps,
            window_seconds=window_seconds,
        )

    def is_fast_move(self, *, min_bps_per_second: float, min_total_bps: float) -> bool:
        """Return True if the current velocity exceeds both thresholds."""
        reading = self.reading()
        return reading.bps_per_second >= min_bps_per_second and abs(reading.move_bps) >= min_total_bps

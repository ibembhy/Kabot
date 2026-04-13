"""Volatility estimation for the DAILY profile (KXBTCD hourly contracts).

KXBTCD expires every 60 minutes. The correct vol input is intraday
realized vol over a 4-hour lookback of 1-minute Coinbase candles.
The annualization formula is identical to the existing GOD vol estimator.
The only differences are a longer lookback and a higher vol floor to
reflect the wider 60-minute price range vs 15-minute.
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

UTC = timezone.utc

# BTC 60-minute realized vol is rarely below 40% annualized.
# Higher floor than GOD (0.05) because 60-minute contracts have wider
# expected price ranges.
DAILY_VOL_FLOOR = 0.40


def bootstrap_hourly_vol(session: Any, *, floor: float = DAILY_VOL_FLOOR) -> float | None:
    """Fetch last 4 hours of 1-minute Coinbase candles and compute vol.

    Returns annualized intraday vol suitable for 60-minute GBM contracts.
    Returns None on failure; caller must use floor.

    Uses the same Coinbase endpoint and formula as GOD bootstrap, but
    fetches 4 hours instead of 90 minutes.
    """
    try:
        end = datetime.now(UTC)
        start = end - timedelta(hours=4)
        response = session.get(
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
            logger.warning("[DAILY] Vol bootstrap: insufficient candles (%d)", len(candles))
            return None
        closes = [float(candle[4]) for candle in sorted(candles, key=lambda candle: candle[0])]
        log_returns = [
            math.log(closes[index] / closes[index - 1])
            for index in range(1, len(closes))
            if closes[index - 1] > 0 and closes[index] > 0
        ]
        if len(log_returns) < 5:
            return None
        arr = np.array(log_returns, dtype=float)
        sigma_per_minute = float(np.std(arr, ddof=1))
        annualized = sigma_per_minute * math.sqrt(525_600.0)
        if not math.isfinite(annualized) or annualized <= 0:
            return None
        result = max(annualized, floor)
        logger.info("[DAILY] Bootstrapped vol: %.4f (from %d candles)", result, len(closes))
        return result
    except Exception as exc:
        logger.warning("[DAILY] Vol bootstrap failed: %s", exc)
        return None


def estimate_hourly_vol_from_db(
    store: Any,
    *,
    series_ticker: str,
    observed_at: datetime,
    lookback_minutes: int = 240,
    min_points: int = 20,
    floor: float = DAILY_VOL_FLOOR,
) -> float:
    """Estimate 60-minute contract vol from DB spot price history.

    Same log-return plus annualization as GOD vol estimator but with a
    4-hour (240-minute) lookback. Falls back to floor if insufficient.
    """
    import pandas as pd

    lookback_start = observed_at - timedelta(minutes=lookback_minutes)
    try:
        history = store.load_market_snapshots(
            series_ticker=series_ticker,
            observed_from=lookback_start,
            observed_to=observed_at,
        )
    except Exception:
        return floor

    if history.empty or "observed_at" not in history.columns or "spot_price" not in history.columns:
        return floor

    spot = (
        history[["observed_at", "spot_price"]]
        .dropna(subset=["observed_at", "spot_price"])
        .drop_duplicates(subset=["observed_at"])
        .sort_values("observed_at")
    )

    if len(spot) < min_points:
        return floor

    price = spot["spot_price"].astype(float).to_numpy()
    times = pd.to_datetime(spot["observed_at"], utc=True)
    log_returns = np.diff(np.log(np.clip(price, 1e-12, None)))

    if len(log_returns) < max(min_points - 1, 2):
        return floor

    deltas = np.diff(times.astype("int64")) / 1_000_000_000.0
    positive_deltas = deltas[deltas > 0]
    if len(positive_deltas) == 0:
        return floor

    avg_dt_seconds = float(np.median(positive_deltas))
    if avg_dt_seconds <= 0:
        return floor

    annualization = math.sqrt(31_536_000.0 / avg_dt_seconds)
    sigma = float(np.std(log_returns, ddof=1)) * annualization

    if not math.isfinite(sigma) or sigma <= 0:
        return floor

    return max(sigma, floor)

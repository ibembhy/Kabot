from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone

from kabot.types import MarketSnapshot


def _scaled_probability(value: float | int | None) -> float | None:
    if value is None:
        return None
    number = float(value)
    if number > 1.0:
        number = number / 100.0
    return min(max(number, 0.0), 1.0)


def _coalesce(mapping: Mapping[str, object], *keys: str) -> object | None:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _ensure_utc(value: object) -> datetime:
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _infer_contract_type(raw_market: Mapping[str, object]) -> str:
    contract_type = str(_coalesce(raw_market, "contract_type", "type", "market_type", "subcategory") or "threshold").lower()
    if "range" in contract_type:
        return "range"
    if "direction" in contract_type or "updown" in contract_type:
        return "direction"
    return "threshold"


def _infer_direction(raw_market: Mapping[str, object]) -> str | None:
    direction = _coalesce(raw_market, "direction", "settlement_rule")
    if isinstance(direction, str):
        normalized = direction.lower()
        if any(token in normalized for token in ("below", "down")):
            return "below"
        if any(token in normalized for token in ("above", "up", "higher")):
            return "above"
    return None


def _is_btc_market(raw_market: Mapping[str, object], underlying_symbol: str) -> bool:
    identifiers = [
        underlying_symbol,
        str(_coalesce(raw_market, "series_ticker", "seriesTicker", "series") or ""),
        str(_coalesce(raw_market, "ticker", "market_ticker") or ""),
    ]
    return any("BTC" in value.upper() for value in identifiers)


def _validate_threshold(
    *,
    raw_market: Mapping[str, object],
    threshold: float | None,
    spot_price: float,
    underlying_symbol: str,
) -> None:
    if threshold is None or not _is_btc_market(raw_market, underlying_symbol):
        return
    if threshold < spot_price * 0.5 or threshold > spot_price * 1.5:
        ticker = str(_coalesce(raw_market, "ticker", "market_ticker") or "")
        raise ValueError(
            f"Invalid BTC threshold for {ticker or 'market'}: "
            f"threshold={threshold} spot_price={spot_price}"
        )


def normalize_market(
    raw_market: Mapping[str, object],
    *,
    spot_price: float,
    observed_at: datetime,
    underlying_symbol: str = "BTC-USD",
    source: str = "kalshi",
) -> MarketSnapshot:
    contract_type = _infer_contract_type(raw_market)
    yes_bid = _scaled_probability(_coalesce(raw_market, "yes_bid", "yes_bid_price", "yes_bid_dollars"))
    yes_ask = _scaled_probability(_coalesce(raw_market, "yes_ask", "yes_ask_price", "yes_ask_dollars"))
    no_bid = _scaled_probability(_coalesce(raw_market, "no_bid", "no_bid_price", "no_bid_dollars"))
    no_ask = _scaled_probability(_coalesce(raw_market, "no_ask", "no_ask_price", "no_ask_dollars"))
    mid_price = None
    if yes_bid is not None and yes_ask is not None:
        mid_price = (yes_bid + yes_ask) / 2.0
    implied_probability = yes_ask if yes_ask is not None else mid_price
    expiry_raw = _coalesce(raw_market, "close_time", "expected_expiration_time", "expiry", "expiration_time", "settlement_time")
    if expiry_raw is None:
        raise ValueError("Market missing expiry")
    threshold = _coalesce(raw_market, "threshold", "strike", "floor_strike")
    threshold_value = None if threshold is None else float(threshold)
    _validate_threshold(
        raw_market=raw_market,
        threshold=threshold_value,
        spot_price=float(spot_price),
        underlying_symbol=underlying_symbol,
    )
    range_low = _coalesce(raw_market, "range_low", "floor", "lower_bound")
    range_high = _coalesce(raw_market, "range_high", "cap", "upper_bound")
    settlement_price_raw = _coalesce(raw_market, "settlement_price", "settlement_value", "final_price")
    volume_raw = _coalesce(raw_market, "volume", "volume_fp", "volume_24h", "volume_24h_fp")
    open_interest_raw = _coalesce(raw_market, "open_interest", "openInterest", "open_interest_fp")
    return MarketSnapshot(
        source=source,
        series_ticker=str(_coalesce(raw_market, "series_ticker", "seriesTicker", "series") or "CRYPTO"),
        market_ticker=str(_coalesce(raw_market, "ticker", "market_ticker") or ""),
        contract_type=contract_type,  # type: ignore[arg-type]
        underlying_symbol=underlying_symbol,
        observed_at=_ensure_utc(observed_at),
        expiry=_ensure_utc(expiry_raw),
        spot_price=float(spot_price),
        threshold=threshold_value,
        range_low=None if range_low is None else float(range_low),
        range_high=None if range_high is None else float(range_high),
        direction=_infer_direction(raw_market),
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        no_bid=no_bid,
        no_ask=no_ask,
        mid_price=mid_price,
        implied_probability=implied_probability,
        volume=None if volume_raw is None else float(volume_raw),
        open_interest=None if open_interest_raw is None else float(open_interest_raw),
        settlement_price=None if settlement_price_raw is None else float(settlement_price_raw),
        metadata=dict(raw_market),
    )

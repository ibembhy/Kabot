from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from kabot.types import Position


def open_position(
    *,
    market_ticker: str,
    side: str,
    contracts: int,
    entry_time: datetime,
    entry_price_cents: int,
    expiry: datetime | None = None,
) -> Position:
    return Position(
        position_id=str(uuid4()),
        market_ticker=market_ticker,
        side=side,  # type: ignore[arg-type]
        contracts=contracts,
        entry_time=entry_time,
        entry_price_cents=entry_price_cents,
        expiry=expiry,
    )

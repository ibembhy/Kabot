from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from kabot.data.kalshi_history import _spot_at_or_before, snapshots_to_frame
from kabot.types import MarketSnapshot


def test_spot_at_or_before_returns_latest_known_value() -> None:
    series = pd.Series(
        [68000.0, 68100.0],
        index=pd.to_datetime(
            [datetime(2026, 4, 1, 20, 40, tzinfo=UTC), datetime(2026, 4, 1, 20, 41, tzinfo=UTC)],
            utc=True,
        ),
    )
    observed_at = datetime(2026, 4, 1, 20, 41, 30, tzinfo=UTC)
    assert _spot_at_or_before(series, observed_at) == 68100.0


def test_snapshots_to_frame_sorts_rows() -> None:
    snapshots = [
        MarketSnapshot(
            source="test",
            series_ticker="KXBTC15M",
            market_ticker="B",
            contract_type="threshold",
            underlying_symbol="BTC-USD",
            observed_at=datetime(2026, 4, 1, 20, 42, tzinfo=UTC),
            expiry=datetime(2026, 4, 1, 20, 45, tzinfo=UTC),
            spot_price=68000.0,
            threshold=68050.0,
            metadata={},
        ),
        MarketSnapshot(
            source="test",
            series_ticker="KXBTC15M",
            market_ticker="A",
            contract_type="threshold",
            underlying_symbol="BTC-USD",
            observed_at=datetime(2026, 4, 1, 20, 41, tzinfo=UTC),
            expiry=datetime(2026, 4, 1, 20, 45, tzinfo=UTC),
            spot_price=67990.0,
            threshold=68000.0,
            metadata={},
        ),
    ]
    frame = snapshots_to_frame(snapshots)
    assert list(frame["market_ticker"]) == ["A", "B"]

from datetime import UTC, datetime

import pytest

from kabot.markets.normalize import normalize_market


def test_normalize_rejects_impossible_btc_threshold_scale() -> None:
    raw_market = {
        "ticker": "KXBTC15M-26APR131115-15",
        "series_ticker": "KXBTC15M",
        "market_type": "binary",
        "yes_bid_dollars": "0.54",
        "yes_ask_dollars": "0.60",
        "no_bid_dollars": "0.40",
        "no_ask_dollars": "0.46",
        "floor_strike": "7.243729",
        "volume": "335246.60",
        "close_time": "2026-04-13T15:15:00Z",
    }

    with pytest.raises(ValueError, match="Invalid BTC threshold"):
        normalize_market(
            raw_market,
            spot_price=72010.52,
            observed_at=datetime(2026, 4, 13, 15, 8, 39, tzinfo=UTC),
        )


def test_normalize_accepts_valid_btc_threshold() -> None:
    snapshot = normalize_market(
        {
            "ticker": "KXBTC15M-26APR131115-15",
            "series_ticker": "KXBTC15M",
            "market_type": "binary",
            "yes_bid_dollars": "0.54",
            "yes_ask_dollars": "0.60",
            "no_bid_dollars": "0.40",
            "no_ask_dollars": "0.46",
            "floor_strike": "72437.29",
            "volume": "335246.60",
            "close_time": "2026-04-13T15:15:00Z",
        },
        spot_price=72010.52,
        observed_at=datetime(2026, 4, 13, 15, 8, 39, tzinfo=UTC),
    )

    assert snapshot.threshold == 72437.29

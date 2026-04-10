from __future__ import annotations

import json
from datetime import UTC, datetime

import pandas as pd

from kabot.cli import _derive_candles_from_snapshots, _merge_snapshots_with_settlements, main


def test_show_config_prints_json(capsys) -> None:
    main(["show-config"])
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["app"]["name"] == "Kabot"


def test_derive_candles_from_snapshots_builds_ohlcv() -> None:
    frame = pd.DataFrame(
        {
            "observed_at": pd.to_datetime(
                [datetime(2026, 4, 5, 12, 0, tzinfo=UTC), datetime(2026, 4, 5, 12, 1, tzinfo=UTC)], utc=True
            ),
            "spot_price": [68000.0, 68010.0],
        }
    )
    candles = _derive_candles_from_snapshots(frame, "1min")
    assert len(candles) == 2
    assert set(candles.columns) == {"open", "high", "low", "close", "volume"}


def test_merge_snapshots_with_settlements_fills_settlement_price() -> None:
    snapshots = pd.DataFrame(
        {
            "market_ticker": ["M1"],
            "settlement_price": [None],
        }
    )
    settlements = pd.DataFrame(
        {
            "market_ticker": ["M1"],
            "expiration_value": [68123.45],
        }
    )
    merged = _merge_snapshots_with_settlements(snapshots, settlements)
    assert float(merged.loc[0, "settlement_price"]) == 68123.45

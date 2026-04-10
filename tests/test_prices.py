from __future__ import annotations

from pathlib import Path

from kabot.data.prices import load_ohlcv_csv


def test_load_ohlcv_csv_reads_kaggle_style_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "btc.csv"
    csv_path.write_text(
        "Timestamp,Open,High,Low,Close,Volume_(BTC)\n"
        "2026-04-05 12:00:00+00:00,68000,68100,67950,68050,12.5\n"
        "2026-04-05 12:01:00+00:00,68050,68120,68010,68100,10.0\n",
        encoding="utf-8",
    )
    frame = load_ohlcv_csv(csv_path)
    assert len(frame) == 2
    assert list(frame.columns) == ["open", "high", "low", "close", "volume"]
    assert float(frame.iloc[0]["close"]) == 68050.0


def test_load_ohlcv_csv_accepts_volume_btc_alias(tmp_path: Path) -> None:
    csv_path = tmp_path / "btc_alias.csv"
    csv_path.write_text(
        "Timestamp,Open,High,Low,Close,Volume_(BTC)\n"
        "1712318400,68000,68100,67950,68050,12.5\n",
        encoding="utf-8",
    )
    frame = load_ohlcv_csv(csv_path)
    assert len(frame) == 1
    assert float(frame.iloc[0]["volume"]) == 12.5

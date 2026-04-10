from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any

import pandas as pd
from psycopg import connect
from psycopg.rows import dict_row

from kabot.types import MarketSnapshot


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        return {key: _jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    return value


class PostgresStore:
    def __init__(self, dsn: str) -> None:
        self.dsn = dsn

    def execute_sql_file(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as handle:
            sql = handle.read()
        with connect(self.dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
            conn.commit()

    def insert_market_snapshots(self, snapshots: list[MarketSnapshot]) -> None:
        if not snapshots:
            return
        rows = []
        for snapshot in snapshots:
            rows.append(
                (
                    snapshot.observed_at,
                    snapshot.source,
                    snapshot.series_ticker,
                    snapshot.market_ticker,
                    snapshot.contract_type,
                    snapshot.underlying_symbol,
                    snapshot.expiry,
                    snapshot.spot_price,
                    snapshot.threshold,
                    snapshot.range_low,
                    snapshot.range_high,
                    snapshot.direction,
                    snapshot.yes_bid,
                    snapshot.yes_ask,
                    snapshot.no_bid,
                    snapshot.no_ask,
                    snapshot.mid_price,
                    snapshot.implied_probability,
                    snapshot.volume,
                    snapshot.open_interest,
                    snapshot.settlement_price,
                    json.dumps(_jsonable(snapshot.metadata)),
                )
            )
        with connect(self.dsn) as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO kalshi_market_snapshots (
                        observed_at, source, series_ticker, market_ticker, contract_type,
                        underlying_symbol, expiry, spot_price, threshold, range_low, range_high,
                        direction, yes_bid, yes_ask, no_bid, no_ask, mid_price,
                        implied_probability, volume, open_interest, settlement_price, metadata_json
                    )
                    VALUES (
                        %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s::jsonb
                    )
                    ON CONFLICT (market_ticker, observed_at) DO UPDATE SET
                        source = EXCLUDED.source,
                        yes_bid = EXCLUDED.yes_bid,
                        yes_ask = EXCLUDED.yes_ask,
                        no_bid = EXCLUDED.no_bid,
                        no_ask = EXCLUDED.no_ask,
                        mid_price = EXCLUDED.mid_price,
                        implied_probability = EXCLUDED.implied_probability,
                        volume = EXCLUDED.volume,
                        open_interest = EXCLUDED.open_interest,
                        settlement_price = EXCLUDED.settlement_price,
                        metadata_json = EXCLUDED.metadata_json
                    """,
                    rows,
                )
            conn.commit()

    def insert_settlements(self, settlements: pd.DataFrame) -> None:
        if settlements.empty:
            return
        frame = settlements.copy()
        rows = []
        for _, row in frame.iterrows():
            metadata = row.get("metadata_json")
            if pd.isna(metadata) or metadata is None:
                metadata = {}
            rows.append(
                (
                    row["market_ticker"],
                    row.get("settled_at"),
                    row.get("result"),
                    None if pd.isna(row.get("expiration_value")) else float(row["expiration_value"]),
                    json.dumps(_jsonable(metadata)),
                )
            )
        with connect(self.dsn) as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO kalshi_settlements (
                        market_ticker, settled_at, result, expiration_value, metadata_json
                    )
                    VALUES (%s, %s, %s, %s, %s::jsonb)
                    ON CONFLICT (market_ticker) DO UPDATE SET
                        settled_at = EXCLUDED.settled_at,
                        result = EXCLUDED.result,
                        expiration_value = EXCLUDED.expiration_value,
                        metadata_json = EXCLUDED.metadata_json
                    """,
                    rows,
                )
            conn.commit()

    def insert_btc_candles(self, candles: pd.DataFrame, *, exchange: str, symbol: str, timeframe: str) -> None:
        if candles.empty:
            return
        frame = candles.copy().reset_index().rename(columns={"index": "ts"})
        rows = [
            (
                row["ts"],
                exchange,
                symbol,
                timeframe,
                float(row["open"]),
                float(row["high"]),
                float(row["low"]),
                float(row["close"]),
                None if pd.isna(row.get("volume")) else float(row["volume"]),
            )
            for _, row in frame.iterrows()
        ]
        with connect(self.dsn) as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO btc_candles (
                        ts, exchange, symbol, timeframe, open, high, low, close, volume
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (ts, exchange, symbol, timeframe) DO UPDATE SET
                        open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        volume = EXCLUDED.volume
                    """,
                    rows,
                )
            conn.commit()

    def load_market_snapshots(
        self,
        *,
        series_ticker: str | None = None,
        market_ticker: str | None = None,
        observed_from: datetime | None = None,
        observed_to: datetime | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        clauses: list[str] = []
        params: list[Any] = []
        if series_ticker:
            clauses.append("series_ticker = %s")
            params.append(series_ticker)
        if market_ticker:
            clauses.append("market_ticker = %s")
            params.append(market_ticker)
        if observed_from:
            clauses.append("observed_at >= %s")
            params.append(observed_from)
        if observed_to:
            clauses.append("observed_at <= %s")
            params.append(observed_to)
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        limit_sql = "LIMIT %s" if limit is not None else ""
        if limit is not None:
            params.append(limit)
        query = f"""
            SELECT *
            FROM kalshi_market_snapshots
            {where_sql}
            ORDER BY observed_at, market_ticker
            {limit_sql}
        """
        with connect(self.dsn, row_factory=dict_row) as conn:
            rows = conn.execute(query, params).fetchall()
        return pd.DataFrame(rows)

    def load_settlements(self, *, market_tickers: list[str] | None = None) -> pd.DataFrame:
        clauses: list[str] = []
        params: list[Any] = []
        if market_tickers:
            clauses.append("market_ticker = ANY(%s)")
            params.append(market_tickers)
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"""
            SELECT *
            FROM kalshi_settlements
            {where_sql}
            ORDER BY market_ticker
        """
        with connect(self.dsn, row_factory=dict_row) as conn:
            rows = conn.execute(query, params).fetchall()
        return pd.DataFrame(rows)

    def load_btc_candles(
        self,
        *,
        exchange: str,
        symbol: str,
        timeframe: str,
        ts_from: datetime | None = None,
        ts_to: datetime | None = None,
    ) -> pd.DataFrame:
        clauses = ["exchange = %s", "symbol = %s", "timeframe = %s"]
        params: list[Any] = [exchange, symbol, timeframe]
        if ts_from is not None:
            clauses.append("ts >= %s")
            params.append(ts_from)
        if ts_to is not None:
            clauses.append("ts <= %s")
            params.append(ts_to)
        query = f"""
            SELECT ts, open, high, low, close, volume
            FROM btc_candles
            WHERE {' AND '.join(clauses)}
            ORDER BY ts
        """
        with connect(self.dsn, row_factory=dict_row) as conn:
            rows = conn.execute(query, params).fetchall()
        frame = pd.DataFrame(rows)
        if frame.empty:
            return frame
        frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
        return frame.set_index("ts").sort_index()

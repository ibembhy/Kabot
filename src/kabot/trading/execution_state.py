from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from kabot.markets.normalize import normalize_market
from kabot.types import MarketSnapshot


@dataclass(frozen=True)
class ExecutionMarketState:
    ticker: str
    snapshot: MarketSnapshot
    metadata_updated_at: datetime | None
    quote_updated_at: datetime | None
    revision: int


class ExecutionStateStore:
    def __init__(self, *, series_ticker: str) -> None:
        self.series_ticker = series_ticker
        self._metadata: dict[str, dict[str, Any]] = {}
        self._metadata_updated_at: dict[str, datetime] = {}

    def update_metadata(self, *, raw_markets: list[dict[str, Any]], observed_at: datetime) -> set[str]:
        live_tickers: set[str] = set()
        for raw in raw_markets:
            ticker = str(raw.get("ticker") or raw.get("market_ticker") or "")
            if not ticker:
                continue
            live_tickers.add(ticker)
            self._metadata[ticker] = {**raw, "series_ticker": self.series_ticker}
            self._metadata_updated_at[ticker] = observed_at
        for ticker in list(self._metadata):
            if ticker not in live_tickers:
                self._metadata.pop(ticker, None)
                self._metadata_updated_at.pop(ticker, None)
        return live_tickers

    def tickers(self) -> set[str]:
        return set(self._metadata.keys())

    def build_snapshot(
        self,
        *,
        ticker: str,
        ws_prices: dict[str, Any] | None,
        spot_price: float,
        observed_at: datetime,
        max_quote_age_seconds: float | None = None,
    ) -> MarketSnapshot | None:
        metadata = self._metadata.get(ticker)
        if metadata is None:
            return None
        effective_ws_prices = ws_prices
        if ws_prices and max_quote_age_seconds is not None:
            updated_at = ws_prices.get("_updated_at")
            if isinstance(updated_at, datetime):
                age_seconds = (observed_at - updated_at).total_seconds()
                if age_seconds > max(float(max_quote_age_seconds), 0.0):
                    effective_ws_prices = {
                        key: value
                        for key, value in ws_prices.items()
                        if key not in {"yes_bid", "yes_ask", "no_bid", "no_ask", "volume", "open_interest", "last_price"}
                    }
                    if all(key.startswith("_") for key in effective_ws_prices):
                        effective_ws_prices = None
        raw = {**metadata, **(effective_ws_prices or {})}
        source = "kalshi_ws" if effective_ws_prices else "kalshi_live"
        try:
            return normalize_market(raw, spot_price=spot_price, observed_at=observed_at, source=source)
        except Exception:
            return None

    def build_state(
        self,
        *,
        ticker: str,
        ws_prices: dict[str, Any] | None,
        spot_price: float,
        observed_at: datetime,
        revision: int = 0,
        quote_updated_at: datetime | None = None,
        max_quote_age_seconds: float | None = None,
    ) -> ExecutionMarketState | None:
        snapshot = self.build_snapshot(
            ticker=ticker,
            ws_prices=ws_prices,
            spot_price=spot_price,
            observed_at=observed_at,
            max_quote_age_seconds=max_quote_age_seconds,
        )
        if snapshot is None:
            return None
        return ExecutionMarketState(
            ticker=ticker,
            snapshot=snapshot,
            metadata_updated_at=self._metadata_updated_at.get(ticker),
            quote_updated_at=quote_updated_at,
            revision=revision,
        )

    def build_snapshots(
        self,
        *,
        ws_snapshot: dict[str, dict[str, Any]],
        spot_price: float,
        observed_at: datetime,
        max_quote_age_seconds: float | None = None,
    ) -> list[MarketSnapshot]:
        snapshots: list[MarketSnapshot] = []
        for ticker in sorted(self._metadata):
            snapshot = self.build_snapshot(
                ticker=ticker,
                ws_prices=ws_snapshot.get(ticker),
                spot_price=spot_price,
                observed_at=observed_at,
                max_quote_age_seconds=max_quote_age_seconds,
            )
            if snapshot is not None:
                snapshots.append(snapshot)
        return snapshots

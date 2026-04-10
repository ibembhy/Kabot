from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime
from typing import Any

from kabot.compat import UTC

logger = logging.getLogger(__name__)

_RECONNECT_DELAY_SECONDS = 5.0


class CoinbaseSpotFeed:
    """Maintains a live BTC-USD spot price via Coinbase WebSocket."""

    WS_URL = "wss://ws-feed.exchange.coinbase.com"

    def __init__(self, product_id: str = "BTC-USD") -> None:
        self._product_id = product_id
        self._price: float | None = None
        self._timestamp: datetime | None = None
        self._revision = 0
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="coinbase-spot-feed")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def get(self) -> tuple[float, datetime] | tuple[None, None]:
        with self._lock:
            return self._price, self._timestamp

    def revision(self) -> int:
        with self._lock:
            return self._revision

    def _run_loop(self) -> None:
        import websocket  # type: ignore[import-untyped]

        while not self._stop_event.is_set():
            try:
                ws = websocket.WebSocketApp(
                    self.WS_URL,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=lambda ws, e: logger.warning("Coinbase WS error: %s", e),
                    on_close=lambda ws, code, msg: None,
                )
                ws.run_forever()
            except Exception as exc:
                logger.warning("Coinbase WS exception: %s", exc)
            if not self._stop_event.is_set():
                time.sleep(_RECONNECT_DELAY_SECONDS)

    def _on_open(self, ws: Any) -> None:
        ws.send(json.dumps({
            "type": "subscribe",
            "channels": [{"name": "ticker", "product_ids": [self._product_id]}],
        }))

    def _on_message(self, ws: Any, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except Exception:
            return
        if msg.get("type") != "ticker":
            return
        price_raw = msg.get("price")
        if price_raw is None:
            return
        try:
            price = float(price_raw)
        except (TypeError, ValueError):
            return
        ts = datetime.now(UTC)
        time_raw = msg.get("time")
        if time_raw:
            try:
                ts = datetime.fromisoformat(str(time_raw).replace("Z", "+00:00")).astimezone(UTC)
            except ValueError:
                pass
        with self._lock:
            self._price = price
            self._timestamp = ts
            self._revision += 1


class KalshiTickerFeed:
    """Maintains a live cache of Kalshi bid/ask prices via WebSocket."""

    def __init__(self, auth_signer: Any, ws_url: str = "wss://api.elections.kalshi.com/trade-api/ws/v2") -> None:
        self._auth_signer = auth_signer
        self._ws_url = ws_url
        # ticker -> {yes_bid, yes_ask, no_bid, no_ask, ...} — values in cents (0-100)
        self._prices: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._subscribed: set[str] = set()
        self._pending: set[str] = set()
        self._revision = 0
        self._ws: Any = None
        self._ws_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._msg_id = 0
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="kalshi-ticker-feed")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        with self._ws_lock:
            if self._ws is not None:
                try:
                    self._ws.close()
                except Exception:
                    pass

    def subscribe(self, tickers: list[str]) -> None:
        """Subscribe to price updates for tickers. Safe to call at any time."""
        new = [t for t in tickers if t not in self._subscribed]
        if not new:
            return
        with self._ws_lock:
            ws = self._ws
        if ws is not None:
            self._send_subscribe(ws, new)
        else:
            with self._lock:
                self._pending.update(new)

    def get_prices(self, ticker: str) -> dict[str, Any] | None:
        with self._lock:
            entry = self._prices.get(ticker)
            return dict(entry) if entry is not None else None

    def snapshot(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {k: dict(v) for k, v in self._prices.items()}

    def revision(self) -> int:
        with self._lock:
            return self._revision

    def dirty_tickers_since(self, revision: int) -> set[str]:
        with self._lock:
            return {
                ticker
                for ticker, value in self._prices.items()
                if int(value.get("_revision", 0) or 0) > revision
            }

    def _next_id(self) -> int:
        self._msg_id += 1
        return self._msg_id

    def _auth_headers(self) -> dict[str, str]:
        return self._auth_signer.request_headers(method="GET", path="/trade-api/ws/v2")

    def _send_subscribe(self, ws: Any, tickers: list[str]) -> None:
        try:
            ws.send(json.dumps({
                "id": self._next_id(),
                "cmd": "subscribe",
                "params": {"channels": ["ticker"], "market_tickers": tickers},
            }))
            self._subscribed.update(tickers)
        except Exception as exc:
            logger.warning("Kalshi WS subscribe error: %s", exc)
            # put them back in pending so they're retried on reconnect
            with self._lock:
                self._pending.update(tickers)

    def _run_loop(self) -> None:
        import websocket  # type: ignore[import-untyped]

        while not self._stop_event.is_set():
            self._subscribed.clear()
            try:
                ws = websocket.WebSocketApp(
                    self._ws_url,
                    header=self._auth_headers(),
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=lambda ws, e: logger.warning("Kalshi WS error: %s", e),
                    on_close=lambda ws, code, msg: None,
                )
                with self._ws_lock:
                    self._ws = ws
                ws.run_forever()
            except Exception as exc:
                logger.warning("Kalshi WS exception: %s", exc)
            finally:
                with self._ws_lock:
                    self._ws = None
            if not self._stop_event.is_set():
                time.sleep(_RECONNECT_DELAY_SECONDS)

    def _on_open(self, ws: Any) -> None:
        with self._lock:
            pending = list(self._pending)
            self._pending.clear()
        if pending:
            self._send_subscribe(ws, pending)

    def _on_message(self, ws: Any, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except Exception:
            return
        msg_type = msg.get("type")
        if msg_type not in ("ticker", "ticker_snapshot"):
            return
        ticker = str(msg.get("market_ticker") or msg.get("ticker") or "")
        if not ticker:
            return
        update: dict[str, Any] = {}
        for field in ("yes_bid", "yes_ask", "no_bid", "no_ask", "volume", "open_interest", "last_price"):
            val = msg.get(field)
            if val is not None:
                update[field] = val
        if not update:
            return
        with self._lock:
            existing = self._prices.get(ticker, {})
            self._revision += 1
            self._prices[ticker] = {
                **existing,
                **update,
                "_updated_at": datetime.now(UTC),
                "_revision": self._revision,
            }


class KalshiFillFeed:
    """Tracks resting orders via Kalshi WebSocket fill/order_group_updates channels.

    Subscribes to account-level private events so the live trader can skip the
    ``list_orders(status="resting")`` REST poll on every cycle.  Falls back
    gracefully: ``get_resting_tickers()`` returns ``None`` whenever the feed is
    not yet connected, letting callers fall back to REST.
    """

    # Terminal order states — remove from resting set on any of these.
    _TERMINAL_STATUSES = {"canceled", "filled", "expired", "settled"}

    def __init__(self, auth_signer: Any, ws_url: str = "wss://api.elections.kalshi.com/trade-api/ws/v2") -> None:
        self._auth_signer = auth_signer
        self._ws_url = ws_url
        # market_ticker -> order_id for orders we know are resting
        self._resting: dict[str, str] = {}
        # reverse index: order_id -> market_ticker
        self._order_ticker: dict[str, str] = {}
        self._lock = threading.Lock()
        self._ws: Any = None
        self._ws_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._msg_id = 0
        self._healthy = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="kalshi-fill-feed")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        with self._ws_lock:
            if self._ws is not None:
                try:
                    self._ws.close()
                except Exception:
                    pass

    def register_order(self, market_ticker: str, order_id: str) -> None:
        """Register a resting order after placement so the feed can track its lifecycle."""
        with self._lock:
            self._resting[market_ticker] = order_id
            self._order_ticker[order_id] = market_ticker

    def get_resting_order_id(self, market_ticker: str) -> str | None:
        """Return the resting order ID for a ticker, or None if not tracked."""
        with self._lock:
            return self._resting.get(market_ticker)

    def deregister_order(self, market_ticker: str) -> None:
        """Explicitly remove a ticker from the resting set (e.g. after a manual cancel)."""
        with self._lock:
            order_id = self._resting.pop(market_ticker, None)
            if order_id:
                self._order_ticker.pop(order_id, None)

    def get_resting_tickers(self) -> set[str] | None:
        """Return the set of tickers with resting orders, or ``None`` if the feed is unhealthy.

        Callers should fall back to the REST ``list_orders`` endpoint when this
        returns ``None``.
        """
        if not self._healthy:
            return None
        with self._lock:
            return set(self._resting.keys())

    def is_healthy(self) -> bool:
        return self._healthy

    def _next_id(self) -> int:
        self._msg_id += 1
        return self._msg_id

    def _auth_headers(self) -> dict[str, str]:
        return self._auth_signer.request_headers(method="GET", path="/trade-api/ws/v2")

    def _run_loop(self) -> None:
        import websocket  # type: ignore[import-untyped]

        while not self._stop_event.is_set():
            self._healthy = False
            try:
                ws = websocket.WebSocketApp(
                    self._ws_url,
                    header=self._auth_headers(),
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=lambda ws, e: logger.warning("Kalshi fill-feed WS error: %s", e),
                    on_close=lambda ws, code, msg: None,
                )
                with self._ws_lock:
                    self._ws = ws
                ws.run_forever()
            except Exception as exc:
                logger.warning("Kalshi fill-feed WS exception: %s", exc)
            finally:
                self._healthy = False
                with self._ws_lock:
                    self._ws = None
            if not self._stop_event.is_set():
                time.sleep(_RECONNECT_DELAY_SECONDS)

    def _on_open(self, ws: Any) -> None:
        try:
            ws.send(json.dumps({
                "id": self._next_id(),
                "cmd": "subscribe",
                "params": {"channels": ["fill", "order_group_updates"]},
            }))
            self._healthy = True
        except Exception as exc:
            logger.warning("Kalshi fill-feed subscribe error: %s", exc)

    def _on_message(self, ws: Any, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except Exception:
            return
        msg_type = msg.get("type")
        if msg_type == "fill":
            order_id = str(msg.get("order_id") or "")
            if order_id:
                with self._lock:
                    market_ticker = self._order_ticker.pop(order_id, None)
                    if market_ticker:
                        self._resting.pop(market_ticker, None)
        elif msg_type == "order_group_updates":
            order_id = str(msg.get("order_id") or "")
            status = str(msg.get("status") or "").lower()
            if order_id and status in self._TERMINAL_STATUSES:
                with self._lock:
                    market_ticker = self._order_ticker.pop(order_id, None)
                    if market_ticker:
                        self._resting.pop(market_ticker, None)

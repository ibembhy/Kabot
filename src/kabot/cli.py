from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Sequence

import pandas as pd

from kabot.backtest.engine import BacktestConfig, BacktestEngine
from kabot.data.features import build_feature_frame, resample_ohlcv
from kabot.data.kalshi_history import (
    HistoricalBackfillConfig,
    KalshiHistoricalClient,
    backfill_series_to_snapshots,
    snapshots_to_frame,
)
from kabot.data.kalshi import snapshots_from_frame
from kabot.data.prices import CoinbasePriceClient, load_ohlcv_csv
from kabot.models.gbm_threshold import GBMThresholdModel
from kabot.reports.evaluation import compare_strategy_results, early_exit_comparison
from kabot.reports.robustness import run_robustness_suite, write_robustness_outputs
from kabot.signals.engine import SignalConfig
from kabot.settings import load_settings
from kabot.trading.exits import ExitConfig
from kabot.trading.live_trader import LiveTraderConfig, build_live_trader


def _default_sql_path() -> Path:
    return Path(__file__).resolve().parents[2] / "sql" / "001_init.sql"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kabot", description="Kabot research CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    show_config = subparsers.add_parser("show-config", help="Print effective configuration as JSON")
    show_config.add_argument("--config", help="Optional TOML config override path")

    init_db = subparsers.add_parser("init-db", help="Initialize the Postgres schema")
    init_db.add_argument("--config", help="Optional TOML config override path")
    init_db.add_argument("--sql-file", default=str(_default_sql_path()), help="Schema SQL file to execute")

    load_snapshots = subparsers.add_parser("load-snapshots", help="Load Kalshi snapshots from an existing SQLite file")
    load_snapshots.add_argument("--config", help="Optional TOML config override path")
    load_snapshots.add_argument("--sqlite-path", required=True, help="Path to source SQLite snapshot DB")
    load_snapshots.add_argument("--series", help="Optional series ticker filter, e.g. KXBTCD")
    load_snapshots.add_argument("--limit", type=int, help="Optional row limit")

    load_settlements = subparsers.add_parser("load-settlements", help="Load Kalshi settlements from CSV")
    load_settlements.add_argument("--config", help="Optional TOML config override path")
    load_settlements.add_argument("--csv-path", required=True, help="Path to settlement CSV")

    derive_btc = subparsers.add_parser("derive-btc-candles", help="Derive BTC candles from snapshot spot prices")
    derive_btc.add_argument("--config", help="Optional TOML config override path")
    derive_btc.add_argument("--sqlite-path", required=True, help="Path to source SQLite snapshot DB")
    derive_btc.add_argument("--series", help="Optional series ticker filter")
    derive_btc.add_argument("--timeframe", default="1min", help="Target candle timeframe, default 1min")

    load_btc_csv = subparsers.add_parser("load-btc-csv", help="Load BTC candles from a CSV dataset")
    load_btc_csv.add_argument("--config", help="Optional TOML config override path")
    load_btc_csv.add_argument("--csv-path", required=True, help="Path to BTC OHLCV CSV")
    load_btc_csv.add_argument("--timeframe", default="1m", help="Stored timeframe label, default 1m")
    load_btc_csv.add_argument("--exchange", default="csv_import", help="Exchange/source label")
    load_btc_csv.add_argument("--timestamp-column", default="Timestamp")
    load_btc_csv.add_argument("--open-column", default="Open")
    load_btc_csv.add_argument("--high-column", default="High")
    load_btc_csv.add_argument("--low-column", default="Low")
    load_btc_csv.add_argument("--close-column", default="Close")
    load_btc_csv.add_argument("--volume-column", default="Volume")

    load_btc_coinbase = subparsers.add_parser("load-btc-coinbase", help="Fetch BTC candles from Coinbase and store them")
    load_btc_coinbase.add_argument("--config", help="Optional TOML config override path")
    load_btc_coinbase.add_argument("--start", required=True, help="Inclusive UTC start timestamp")
    load_btc_coinbase.add_argument("--end", required=True, help="Exclusive UTC end timestamp")
    load_btc_coinbase.add_argument("--granularity-seconds", type=int, default=60)
    load_btc_coinbase.add_argument("--product-id", default="BTC-USD")
    load_btc_coinbase.add_argument("--exchange", default="coinbase")
    load_btc_coinbase.add_argument("--timeframe", default="1m")

    export_kalshi_history = subparsers.add_parser("export-kalshi-history", help="Backfill settled Kalshi markets from the public API into a CSV")
    export_kalshi_history.add_argument("--series", required=True, help="Series ticker, e.g. KXBTC15M")
    export_kalshi_history.add_argument("--start", required=True, help="Inclusive UTC start timestamp")
    export_kalshi_history.add_argument("--end", required=True, help="Inclusive UTC end timestamp")
    export_kalshi_history.add_argument("--btc-csv-path", required=True, help="Path to BTC minute OHLCV CSV")
    export_kalshi_history.add_argument("--output-csv", required=True, help="Destination CSV path")
    export_kalshi_history.add_argument("--lookback-minutes", type=int, default=15)
    export_kalshi_history.add_argument("--max-near-money-bps", type=float, default=1000.0)
    export_kalshi_history.add_argument("--max-markets-per-event", type=int, help="Optional cap on nearest markets fetched per event")
    export_kalshi_history.add_argument("--limit-events", type=int)
    export_kalshi_history.add_argument("--sleep-seconds", type=float, default=0.0)
    export_kalshi_history.add_argument("--progress", action="store_true", help="Print incremental progress while exporting")

    backtest = subparsers.add_parser("run-backtest", help="Run a real backtest from Postgres-loaded data or a snapshots CSV")
    backtest.add_argument("--config", help="Optional TOML config override path")
    backtest.add_argument("--series", help="Series ticker, e.g. KXBTCD (required unless --snapshots-csv is set)")
    backtest.add_argument("--snapshots-csv", help="Path to a snapshots CSV file (skips Postgres)")
    backtest.add_argument("--strategy-mode", choices=["hold_to_settlement", "trade_exit"], default="hold_to_settlement")
    backtest.add_argument("--observed-from", help="Optional inclusive UTC timestamp")
    backtest.add_argument("--observed-to", help="Optional inclusive UTC timestamp")
    backtest.add_argument("--volatility-override", type=float, help="Use this fixed annualized volatility for every backtest signal")

    robustness = subparsers.add_parser("run-robustness-suite", help="Run robustness diagnostics on existing backtest data")
    robustness.add_argument("--config", help="Optional TOML config override path")
    robustness.add_argument("--series", help="Series ticker, e.g. KXBTC15M (required unless --snapshots-csv is set)")
    robustness.add_argument("--snapshots-csv", help="Path to a snapshots CSV file (skips Postgres)")
    robustness.add_argument("--strategy-mode", choices=["hold_to_settlement", "trade_exit"], default="hold_to_settlement")
    robustness.add_argument("--observed-from", help="Optional inclusive UTC timestamp")
    robustness.add_argument("--observed-to", help="Optional inclusive UTC timestamp")
    robustness.add_argument("--rolling-window-days", type=int, default=7, help="Rolling window size in days")
    robustness.add_argument("--rolling-step-days", type=int, default=3, help="Rolling window step in days")
    robustness.add_argument("--output-dir", help="Optional output directory for CSV/JSON artifacts")

    live_trade = subparsers.add_parser("live-trade", help="Run the simple KXBTC15M live trading loop")
    live_trade.add_argument("--config", help="Optional TOML config override path")
    live_trade.add_argument(
        "--profile",
        default="baseline_live",
        choices=(
            "baseline_live",
            "exp_12m_signal_break",
            "exp_12m_signal_break_execution",
            "exp_fills2",
            "GOD",
            "NEW",
            "DAILY",
        ),
        help=(
            "Named live profile. baseline_live keeps the current bot. "
            "exp_12m_signal_break enables the 12-minute strategy with signal-break exits. "
            "exp_12m_signal_break_execution runs that same strategy with the execution-session engine and no REST depth precheck. "
            "exp_fills2 uses a 10-minute window, looser distance, and a controlled execution ladder. "
            "GOD mirrors exp_fills2 with Coinbase volatility bootstrap and no orderbook precheck. "
            "NEW runs both hold-to-settlement (Strategy A) and fade-the-fast-move (Strategy B) "
            "with a bootstrapped volatility floor and no orderbook precheck. "
            "DAILY trades KXBTCD (hourly BTC contracts expiring at :00 each hour) "
            "using GTC limit orders as maker. Targets 65c+ contracts with early "
            "exit logic. Runs as separate service alongside GOD."
        ),
    )
    live_trade.add_argument("--series", default="KXBTC15M", help="Series ticker, default KXBTC15M")
    live_trade.add_argument("--poll-seconds", type=int, default=10, help="Polling interval, default 10 seconds")
    live_trade.add_argument("--max-open-markets", type=int, default=3, help="Maximum concurrently active markets")
    live_trade.add_argument("--dry-run", action="store_true", help="Preview orders without submitting them")
    live_trade.add_argument("--once", action="store_true", help="Run one loop iteration and exit")
    live_trade.add_argument("--daily-loss-stop-dollars", type=float, default=10.0, help="Pause for the day after this realized loss")
    live_trade.add_argument("--max-trades-per-day", type=int, default=0, help="Maximum entries per day; 0 disables the cap")
    live_trade.add_argument("--cooldown-loss-streak", type=int, default=3, help="Trigger cooldown after this many realized losses in a row")
    live_trade.add_argument("--cooldown-minutes", type=int, default=30, help="Cooldown duration in minutes")
    live_trade.add_argument("--max-spot-age-seconds", type=int, default=30, help="Maximum allowed BTC spot staleness")
    live_trade.add_argument("--max-market-age-seconds", type=int, default=30, help="Maximum allowed Kalshi market staleness")

    return parser


def _load_snapshot_frame(sqlite_path: str, *, series: str | None = None, limit: int | None = None) -> pd.DataFrame:
    conn = sqlite3.connect(sqlite_path)
    query = "SELECT * FROM market_snapshots"
    clauses: list[str] = []
    params: list[object] = []
    if series:
        clauses.append("series_ticker = ?")
        params.append(series)
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY observed_at"
    if limit is not None:
        query += f" LIMIT {int(limit)}"
    frame = pd.read_sql_query(query, conn)
    conn.close()
    if frame.empty:
        return frame
    for column in ("observed_at", "expiry"):
        frame[column] = pd.to_datetime(frame[column], utc=True, format="ISO8601")
    if "metadata_json" in frame.columns:
        frame["metadata_json"] = frame["metadata_json"].fillna("{}").map(json.loads)
    return frame


def _derive_candles_from_snapshots(frame: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    spot = (
        frame[["observed_at", "spot_price"]]
        .dropna(subset=["observed_at", "spot_price"])
        .drop_duplicates(subset=["observed_at"])
        .sort_values("observed_at")
        .rename(columns={"observed_at": "ts", "spot_price": "close"})
        .set_index("ts")
    )
    spot["open"] = spot["close"]
    spot["high"] = spot["close"]
    spot["low"] = spot["close"]
    spot["volume"] = 0.0
    candles = spot[["open", "high", "low", "close", "volume"]]
    if timeframe.lower() not in {"1min", "1m"}:
        return resample_ohlcv(candles, timeframe)
    return candles


def _load_settlements_csv(csv_path: str) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    if frame.empty:
        return frame
    rename_map = {"ticker": "market_ticker", "close_time": "settled_at"}
    frame = frame.rename(columns=rename_map)
    if "settled_at" in frame.columns:
        frame["settled_at"] = pd.to_datetime(frame["settled_at"], utc=True, errors="coerce", format="ISO8601")
    frame["metadata_json"] = frame.apply(
        lambda row: {
            "volume": None if pd.isna(row.get("volume")) else float(row["volume"]),
            "open_interest": None if pd.isna(row.get("open_interest")) else float(row["open_interest"]),
        },
        axis=1,
    )
    return frame[["market_ticker", "settled_at", "result", "expiration_value", "metadata_json"]]


def _load_snapshots_csv(csv_path: str, *, observed_from: pd.Timestamp | None = None, observed_to: pd.Timestamp | None = None) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    if frame.empty:
        return frame
    for col in ("observed_at", "expiry"):
        if col in frame.columns:
            frame[col] = pd.to_datetime(frame[col], utc=True, errors="coerce")
    if observed_from is not None:
        frame = frame.loc[frame["observed_at"] >= observed_from]
    if observed_to is not None:
        frame = frame.loc[frame["observed_at"] <= observed_to]
    return frame.reset_index(drop=True)


def _features_from_snapshots(snapshots: pd.DataFrame, *, volatility_window: int, annualization_factor: float) -> pd.DataFrame:
    """Derive a feature frame from the spot_price column of a snapshot frame."""
    if snapshots.empty or "spot_price" not in snapshots.columns or "observed_at" not in snapshots.columns:
        return pd.DataFrame()
    spot = (
        snapshots[["observed_at", "spot_price"]]
        .dropna(subset=["observed_at", "spot_price"])
        .drop_duplicates(subset=["observed_at"])
        .sort_values("observed_at")
        .rename(columns={"observed_at": "ts", "spot_price": "close"})
        .set_index("ts")
    )
    spot["open"] = spot["close"]
    spot["high"] = spot["close"]
    spot["low"] = spot["close"]
    spot["volume"] = 0.0
    return build_feature_frame(spot, volatility_window=volatility_window, annualization_factor=annualization_factor)


def _constant_volatility_features(snapshots: pd.DataFrame, *, volatility: float) -> pd.DataFrame:
    """Build a feature frame that forces one realized volatility value."""
    if snapshots.empty or "observed_at" not in snapshots.columns:
        return pd.DataFrame()
    observed_at = (
        pd.to_datetime(snapshots["observed_at"], utc=True, errors="coerce")
        .dropna()
        .drop_duplicates()
        .sort_values()
    )
    if observed_at.empty:
        return pd.DataFrame()
    return pd.DataFrame(
        {"realized_volatility": float(volatility)},
        index=pd.DatetimeIndex(observed_at, name="ts"),
    )


def _merge_snapshots_with_settlements(snapshots: pd.DataFrame, settlements: pd.DataFrame) -> pd.DataFrame:
    if snapshots.empty:
        return snapshots
    if settlements.empty:
        return snapshots
    merged = snapshots.merge(
        settlements[["market_ticker", "expiration_value"]].rename(columns={"expiration_value": "settlement_price"}),
        on="market_ticker",
        how="left",
        suffixes=("", "_settlement"),
    )
    if "settlement_price_settlement" in merged.columns:
        mask = merged["settlement_price"].isna()
        merged.loc[mask, "settlement_price"] = merged.loc[mask, "settlement_price_settlement"]
        merged = merged.drop(columns=["settlement_price_settlement"])
    return merged


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "show-config":
        settings = load_settings(args.config)
        print(json.dumps(settings.raw, indent=2, sort_keys=True))
        return

    if args.command == "init-db":
        from kabot.storage.postgres import PostgresStore

        settings = load_settings(args.config)
        store = PostgresStore(settings.storage["db_dsn"])
        store.execute_sql_file(args.sql_file)
        print(f"Initialized database using {args.sql_file}")
        return

    if args.command == "load-snapshots":
        from kabot.storage.postgres import PostgresStore

        settings = load_settings(args.config)
        store = PostgresStore(settings.storage["db_dsn"])
        frame = _load_snapshot_frame(args.sqlite_path, series=args.series, limit=args.limit)
        snapshots = snapshots_from_frame(frame, source="kalshi_sqlite")
        store.insert_market_snapshots(snapshots)
        print(f"Loaded {len(snapshots)} market snapshots from {args.sqlite_path}")
        return

    if args.command == "load-settlements":
        from kabot.storage.postgres import PostgresStore

        settings = load_settings(args.config)
        store = PostgresStore(settings.storage["db_dsn"])
        frame = _load_settlements_csv(args.csv_path)
        store.insert_settlements(frame)
        print(f"Loaded {len(frame)} settlement rows from {args.csv_path}")
        return

    if args.command == "derive-btc-candles":
        from kabot.storage.postgres import PostgresStore

        settings = load_settings(args.config)
        store = PostgresStore(settings.storage["db_dsn"])
        frame = _load_snapshot_frame(args.sqlite_path, series=args.series)
        candles = _derive_candles_from_snapshots(frame, args.timeframe)
        store.insert_btc_candles(
            candles,
            exchange="derived_kalshi",
            symbol=settings.data["underlying_symbol"],
            timeframe="1m" if args.timeframe.lower() in {"1min", "1m"} else args.timeframe,
        )
        print(f"Loaded {len(candles)} BTC candles into Postgres from snapshot spot prices")
        return

    if args.command == "load-btc-csv":
        from kabot.storage.postgres import PostgresStore

        settings = load_settings(args.config)
        store = PostgresStore(settings.storage["db_dsn"])
        candles = load_ohlcv_csv(
            args.csv_path,
            timestamp_column=args.timestamp_column,
            open_column=args.open_column,
            high_column=args.high_column,
            low_column=args.low_column,
            close_column=args.close_column,
            volume_column=args.volume_column,
        )
        store.insert_btc_candles(
            candles,
            exchange=args.exchange,
            symbol=settings.data["underlying_symbol"],
            timeframe=args.timeframe,
        )
        print(f"Loaded {len(candles)} BTC candles from {args.csv_path}")
        return

    if args.command == "load-btc-coinbase":
        from kabot.storage.postgres import PostgresStore

        settings = load_settings(args.config)
        store = PostgresStore(settings.storage["db_dsn"])
        client = CoinbasePriceClient(product_id=args.product_id)
        start = pd.to_datetime(args.start, utc=True).to_pydatetime()
        end = pd.to_datetime(args.end, utc=True).to_pydatetime()
        candles = client.fetch_candles_range(start, end, granularity_seconds=args.granularity_seconds)
        store.insert_btc_candles(
            candles,
            exchange=args.exchange,
            symbol=settings.data["underlying_symbol"],
            timeframe=args.timeframe,
        )
        print(f"Loaded {len(candles)} Coinbase BTC candles from {args.start} to {args.end}")
        return

    if args.command == "export-kalshi-history":
        start = pd.to_datetime(args.start, utc=True).to_pydatetime()
        end = pd.to_datetime(args.end, utc=True).to_pydatetime()
        candles = load_ohlcv_csv(args.btc_csv_path)
        spot_history = candles["close"].astype(float).sort_index()
        client = KalshiHistoricalClient()
        def _print_progress(payload: dict[str, object]) -> None:
            stage = str(payload.get("stage", ""))
            if stage == "events_loaded":
                print(
                    f"[progress] series={payload.get('series_ticker')} events={payload.get('event_count')} snapshots={payload.get('snapshots_written')}",
                    flush=True,
                )
                return
            if stage == "event_started":
                print(
                    f"[progress] event {payload.get('event_index')}/{payload.get('event_count')} "
                    f"{payload.get('event_ticker')} markets={payload.get('market_count')} eligible={payload.get('eligible_market_count')} "
                    f"snapshots={payload.get('snapshots_written')}",
                    flush=True,
                )
                return
            if stage == "market_finished":
                print(
                    f"[progress] market {payload.get('processed_markets')} "
                    f"{payload.get('market_ticker')} snapshots={payload.get('snapshots_written')}",
                    flush=True,
                )
                return
            if stage == "complete":
                print(
                    f"[progress] complete series={payload.get('series_ticker')} events={payload.get('event_count')} "
                    f"markets={payload.get('processed_markets')} snapshots={payload.get('snapshots_written')}",
                    flush=True,
                )
        snapshots = backfill_series_to_snapshots(
            client=client,
            config=HistoricalBackfillConfig(
                series_ticker=args.series,
                start=start,
                end=end,
                lookback_minutes=args.lookback_minutes,
                max_near_money_bps=args.max_near_money_bps,
                max_markets_per_event=args.max_markets_per_event,
                limit_events=args.limit_events,
                sleep_seconds=args.sleep_seconds,
            ),
            spot_history=spot_history,
            progress_callback=_print_progress if args.progress else None,
        )
        frame = snapshots_to_frame(snapshots)
        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.output_csv, index=False)
        print(f"Exported {len(frame)} snapshots to {args.output_csv}")
        return

    if args.command == "run-backtest":
        settings = load_settings(args.config)
        observed_from = pd.to_datetime(args.observed_from, utc=True) if args.observed_from else None
        observed_to = pd.to_datetime(args.observed_to, utc=True) if args.observed_to else None

        if getattr(args, "snapshots_csv", None):
            snapshots = _load_snapshots_csv(args.snapshots_csv, observed_from=observed_from, observed_to=observed_to)
            features = _features_from_snapshots(
                snapshots,
                volatility_window=int(settings.data["volatility_window"]),
                annualization_factor=float(settings.data["annualization_factor"]),
            )
        else:
            if not args.series:
                raise SystemExit("--series is required when not using --snapshots-csv")
            from kabot.storage.postgres import PostgresStore
            store = PostgresStore(settings.storage["db_dsn"])
            snapshots = store.load_market_snapshots(
                series_ticker=args.series,
                observed_from=observed_from.to_pydatetime() if observed_from is not None else None,
                observed_to=observed_to.to_pydatetime() if observed_to is not None else None,
            )
            settlements = store.load_settlements(
                market_tickers=snapshots["market_ticker"].dropna().astype(str).unique().tolist() if not snapshots.empty else None
            )
            candles = store.load_btc_candles(
                exchange="derived_kalshi",
                symbol=settings.data["underlying_symbol"],
                timeframe="1m",
                ts_from=observed_from.to_pydatetime() if observed_from is not None else None,
                ts_to=observed_to.to_pydatetime() if observed_to is not None else None,
            )
            snapshots = _merge_snapshots_with_settlements(snapshots, settlements)
            features = build_feature_frame(
                candles,
                volatility_window=int(settings.data["volatility_window"]),
                annualization_factor=float(settings.data["annualization_factor"]),
            )
        if args.volatility_override is not None:
            features = _constant_volatility_features(snapshots, volatility=float(args.volatility_override))
        engine = BacktestEngine(
            model=GBMThresholdModel(
                drift=float(settings.model["drift"]),
                volatility_floor=float(settings.model["volatility_floor"]),
            ),
            signal_config=SignalConfig(
                min_edge=float(settings.signal["min_edge"]),
                min_contract_price_cents=int(settings.signal["min_contract_price_cents"]),
                max_contract_price_cents=int(settings.signal["max_contract_price_cents"]),
                max_spread_cents=int(settings.signal["max_spread_cents"]),
                max_near_money_bps=float(settings.signal["max_near_money_bps"]),
                min_confidence=float(settings.signal["min_confidence"]),
            ),
            exit_config=ExitConfig(
                take_profit_cents=int(settings.trading["take_profit_cents"]),
                stop_loss_cents=int(settings.trading["stop_loss_cents"]),
                fair_value_buffer_cents=int(settings.trading["fair_value_buffer_cents"]),
                time_exit_minutes=int(settings.trading["time_exit_minutes_before_expiry"]),
            ),
            backtest_config=BacktestConfig(
                strategy_mode=args.strategy_mode,
                entry_slippage_cents=int(settings.backtest["entry_slippage_cents"]),
                exit_slippage_cents=int(settings.backtest["exit_slippage_cents"]),
                fee_rate_bps=float(settings.backtest["fee_rate_bps"]),
                allow_reentry=bool(settings.trading["allow_reentry"]),
            ),
        )
        result = engine.run(snapshots, features)
        payload = {
            "strategy_mode": result.strategy_mode,
            "summary": result.summary,
            "comparison": None,
        }
        if args.strategy_mode == "trade_exit":
            hold_result = BacktestEngine(
                model=GBMThresholdModel(
                    drift=float(settings.model["drift"]),
                    volatility_floor=float(settings.model["volatility_floor"]),
                ),
                signal_config=SignalConfig(
                    min_edge=float(settings.signal["min_edge"]),
                    min_contract_price_cents=int(settings.signal["min_contract_price_cents"]),
                    max_contract_price_cents=int(settings.signal["max_contract_price_cents"]),
                    max_spread_cents=int(settings.signal["max_spread_cents"]),
                    max_near_money_bps=float(settings.signal["max_near_money_bps"]),
                    min_confidence=float(settings.signal["min_confidence"]),
                ),
                exit_config=ExitConfig(
                    take_profit_cents=int(settings.trading["take_profit_cents"]),
                    stop_loss_cents=int(settings.trading["stop_loss_cents"]),
                    fair_value_buffer_cents=int(settings.trading["fair_value_buffer_cents"]),
                    time_exit_minutes=int(settings.trading["time_exit_minutes_before_expiry"]),
                ),
                backtest_config=BacktestConfig(
                    strategy_mode="hold_to_settlement",
                    entry_slippage_cents=int(settings.backtest["entry_slippage_cents"]),
                    exit_slippage_cents=int(settings.backtest["exit_slippage_cents"]),
                    fee_rate_bps=float(settings.backtest["fee_rate_bps"]),
                    allow_reentry=bool(settings.trading["allow_reentry"]),
                ),
            ).run(snapshots, features)
            payload["comparison"] = early_exit_comparison(hold_result, result)
            payload["strategies"] = compare_strategy_results({"hold": hold_result, "trade_exit": result}).to_dict(orient="records")
        print(json.dumps(payload, indent=2, default=str))
        return

    if args.command == "run-robustness-suite":
        settings = load_settings(args.config)
        observed_from = pd.to_datetime(args.observed_from, utc=True) if args.observed_from else None
        observed_to = pd.to_datetime(args.observed_to, utc=True) if args.observed_to else None

        if getattr(args, "snapshots_csv", None):
            snapshots = _load_snapshots_csv(args.snapshots_csv, observed_from=observed_from, observed_to=observed_to)
            features = _features_from_snapshots(
                snapshots,
                volatility_window=int(settings.data["volatility_window"]),
                annualization_factor=float(settings.data["annualization_factor"]),
            )
        else:
            if not args.series:
                raise SystemExit("--series is required when not using --snapshots-csv")
            from kabot.storage.postgres import PostgresStore
            store = PostgresStore(settings.storage["db_dsn"])
            snapshots = store.load_market_snapshots(
                series_ticker=args.series,
                observed_from=observed_from.to_pydatetime() if observed_from is not None else None,
                observed_to=observed_to.to_pydatetime() if observed_to is not None else None,
            )
            settlements = store.load_settlements(
                market_tickers=snapshots["market_ticker"].dropna().astype(str).unique().tolist() if not snapshots.empty else None
            )
            candles = store.load_btc_candles(
                exchange="derived_kalshi",
                symbol=settings.data["underlying_symbol"],
                timeframe="1m",
                ts_from=observed_from.to_pydatetime() if observed_from is not None else None,
                ts_to=observed_to.to_pydatetime() if observed_to is not None else None,
            )
            snapshots = _merge_snapshots_with_settlements(snapshots, settlements)
            features = build_feature_frame(
                candles,
                volatility_window=int(settings.data["volatility_window"]),
                annualization_factor=float(settings.data["annualization_factor"]),
            )
        payload = run_robustness_suite(
            snapshots=snapshots,
            features=features,
            model=GBMThresholdModel(
                drift=float(settings.model["drift"]),
                volatility_floor=float(settings.model["volatility_floor"]),
            ),
            signal_config=SignalConfig(
                min_edge=float(settings.signal["min_edge"]),
                min_contract_price_cents=int(settings.signal["min_contract_price_cents"]),
                max_contract_price_cents=int(settings.signal["max_contract_price_cents"]),
                max_spread_cents=int(settings.signal["max_spread_cents"]),
                max_near_money_bps=float(settings.signal["max_near_money_bps"]),
                min_confidence=float(settings.signal["min_confidence"]),
            ),
            exit_config=ExitConfig(
                take_profit_cents=int(settings.trading["take_profit_cents"]),
                stop_loss_cents=int(settings.trading["stop_loss_cents"]),
                fair_value_buffer_cents=int(settings.trading["fair_value_buffer_cents"]),
                time_exit_minutes=int(settings.trading["time_exit_minutes_before_expiry"]),
            ),
            backtest_config=BacktestConfig(
                strategy_mode=args.strategy_mode,
                entry_slippage_cents=int(settings.backtest["entry_slippage_cents"]),
                exit_slippage_cents=int(settings.backtest["exit_slippage_cents"]),
                fee_rate_bps=float(settings.backtest["fee_rate_bps"]),
                allow_reentry=bool(settings.trading["allow_reentry"]),
            ),
            rolling_window_days=int(args.rolling_window_days),
            rolling_step_days=int(args.rolling_step_days),
        )
        summary = {
            "base_summary": payload["base_summary"],
            "tail_risk": payload["tail_risk"],
            "rows": {
                "rolling_windows": len(payload["rolling_windows"]),
                "regime_splits": len(payload["regime_splits"]),
                "parameter_sweep": len(payload["parameter_sweep"]),
                "cost_stress": len(payload["cost_stress"]),
            },
        }
        if args.output_dir:
            summary["outputs"] = write_robustness_outputs(payload=payload, output_dir=args.output_dir)
        print(json.dumps(summary, indent=2, default=str))
        return

    if args.command == "live-trade":
        from kabot.storage.postgres import PostgresStore

        settings = load_settings(args.config)
        store = PostgresStore(settings.storage["db_dsn"])
        profile = str(args.profile)
        uses_fills2_profile = profile in {"exp_fills2", "GOD"}
        uses_12m_strategy = profile in {"exp_12m_signal_break", "exp_12m_signal_break_execution"}
        uses_fills2 = uses_fills2_profile
        entry_max_tte_minutes = 14.0 if uses_fills2 else (12.0 if uses_12m_strategy else 10.0)
        enable_signal_break_exit = uses_12m_strategy
        enable_execution_sessions = profile in {"exp_12m_signal_break_execution", "exp_fills2", "GOD"}
        enable_signal_break_reentry = False
        use_orderbook_precheck = profile not in {"exp_12m_signal_break_execution", "exp_fills2", "GOD"}
        entry_time_in_force = "good_till_canceled" if profile in {"exp_12m_signal_break_execution", "exp_fills2", "GOD"} else "immediate_or_cancel"
        execution_cross_cents = 0 if profile in {"exp_12m_signal_break_execution", "exp_fills2", "GOD"} else 6
        min_market_volume = 1000.0 if profile in {"exp_12m_signal_break_execution", "exp_fills2", "GOD"} else 5000.0
        exit_cross_cents = 4 if (uses_12m_strategy or uses_fills2) else 1
        exit_distance_threshold_dollars = 0.0 if uses_fills2 else (3.0 if uses_12m_strategy else 10.0)
        signal_break_confirmation_cycles = 3 if (uses_12m_strategy or uses_fills2) else 1
        exit_negative_edge_threshold = -0.05 if uses_fills2 else (-0.02 if uses_12m_strategy else 0.0)
        price_stop_cents = 15 if (uses_12m_strategy or uses_fills2) else 0
        price_stop_grace_seconds = 90 if (uses_12m_strategy or uses_fills2) else 0
        price_stop_confirm_cycles = 3 if (uses_12m_strategy or uses_fills2) else 1
        hard_stop_cents = 22 if (uses_12m_strategy or uses_fills2) else 0
        reentry_edge_premium = 0.01 if profile in {"exp_12m_signal_break_execution", "exp_fills2", "GOD"} else 0.02
        reentry_min_price_improvement_cents = 1 if profile in {"exp_12m_signal_break_execution", "exp_fills2", "GOD"} else 3
        distance_threshold_dollars = 6.0 if uses_fills2 else 10.0
        execution_ladder_steps_cents = (1, 2, 3, 4) if uses_fills2 else None
        execution_ladder_steps_cents_high = (2, 3, 4, 5) if uses_fills2 else None
        high_conviction_edge_threshold = 0.06 if uses_fills2 else 0.0
        enforce_positive_edge_on_execution = False
        high_conviction_distance_threshold_dollars = 8.0 if uses_fills2 else 0.0
        execution_min_edge_margin_low = 0.03 if uses_fills2 else 0.0
        execution_min_edge_margin_high = 0.0 if uses_fills2 else 0.0
        gbm_min_edge_mid = 0.06 if uses_fills2 else 0.02
        gbm_min_edge_wide_yes = 0.03 if uses_fills2 else 0.04
        execution_spread_tight_cents = 2 if uses_fills2 else 2
        execution_spread_wide_cents = 6 if uses_fills2 else 6
        execution_spread_tight_min_cross_cents = 2 if uses_fills2 else 2
        execution_spread_wide_max_cross_cents = 1 if uses_fills2 else 1
        execution_ladder_delay_high_seconds = 0.5 if profile == "GOD" else (5.0 if uses_fills2 else 5.0)
        execution_ladder_delay_mid_seconds = 0.8 if profile == "GOD" else (8.0 if uses_fills2 else 8.0)
        execution_ladder_delay_low_seconds = 1.2 if profile == "GOD" else (12.0 if uses_fills2 else 12.0)
        fast_market_tte_minutes = 5.0 if uses_fills2 else 5.0
        fast_market_delay_ceiling_seconds = 6.0 if uses_fills2 else 6.0
        resting_entry_max_age_seconds = 20.0 if uses_fills2 else 45.0
        resting_entry_max_age_seconds_high = 40.0 if uses_fills2 else 90.0
        edge_decay_cancel_threshold = 0.02 if uses_fills2 else 0.02
        is_new_profile = profile == "NEW"
        if is_new_profile:
            entry_max_tte_minutes = 14.0
            use_orderbook_precheck = False
            entry_time_in_force = "immediate_or_cancel"
            enable_signal_break_exit = False
            enable_execution_sessions = False
            gbm_min_edge_mid = 0.08
            gbm_min_edge_wide_yes = 0.08
            min_market_volume = 1000.0
            execution_cross_cents = 0
            distance_threshold_dollars = 0.0
        is_daily_profile = profile == "DAILY"
        if is_daily_profile:
            args.series = "KXBTCD"
            entry_time_in_force = "good_till_canceled"
            use_orderbook_precheck = False
            enable_execution_sessions = False
            enable_signal_break_exit = False
            min_market_volume = 500.0
            gbm_min_edge_mid = 0.06
            gbm_min_edge_wide_yes = 0.06
        trader = build_live_trader(
            store=store,
            config=LiveTraderConfig(
                active_profile=profile,
                series_ticker=args.series,
                poll_seconds=args.poll_seconds,
                max_open_markets=args.max_open_markets,
                dry_run=bool(args.dry_run),
                daily_loss_stop_cents=int(round(float(args.daily_loss_stop_dollars) * 100.0)),
                max_trades_per_day=args.max_trades_per_day,
                max_strategy_open_counts={
                    "yes_continuation_mid": args.max_open_markets,
                    "no_continuation_mid": 2,
                    "yes_continuation_wide": 2,
                },
                cooldown_loss_streak=args.cooldown_loss_streak,
                cooldown_minutes=args.cooldown_minutes,
                max_spot_age_seconds=args.max_spot_age_seconds,
                max_market_age_seconds=args.max_market_age_seconds,
                entry_max_tte_minutes=entry_max_tte_minutes,
                distance_threshold_dollars=distance_threshold_dollars,
                enable_signal_break_exit=enable_signal_break_exit,
                enable_execution_sessions=enable_execution_sessions,
                enable_signal_break_reentry=enable_signal_break_reentry,
                use_orderbook_precheck=use_orderbook_precheck,
                entry_time_in_force=entry_time_in_force,
                hybrid_resting_entry_enabled=False,
                hybrid_resting_entry_seconds=5.0,
                execution_cross_cents=execution_cross_cents,
                execution_ladder_steps_cents=execution_ladder_steps_cents,
                execution_ladder_steps_cents_high=execution_ladder_steps_cents_high,
                high_conviction_edge_threshold=high_conviction_edge_threshold,
                high_conviction_distance_threshold_dollars=high_conviction_distance_threshold_dollars,
                enforce_positive_edge_on_execution=enforce_positive_edge_on_execution,
                execution_min_edge_margin_low=execution_min_edge_margin_low,
                execution_min_edge_margin_high=execution_min_edge_margin_high,
                gbm_min_edge_mid=gbm_min_edge_mid,
                gbm_min_edge_wide_yes=gbm_min_edge_wide_yes,
                min_market_volume=min_market_volume,
                exit_cross_cents=exit_cross_cents,
                exit_distance_threshold_dollars=exit_distance_threshold_dollars,
                signal_break_confirmation_cycles=signal_break_confirmation_cycles,
                exit_negative_edge_threshold=exit_negative_edge_threshold,
                price_stop_cents=price_stop_cents,
                price_stop_grace_seconds=price_stop_grace_seconds,
                price_stop_confirm_cycles=price_stop_confirm_cycles,
                hard_stop_cents=hard_stop_cents,
                reentry_edge_premium=reentry_edge_premium,
                reentry_min_price_improvement_cents=reentry_min_price_improvement_cents,
                execution_session_attempts=5 if profile == "GOD" else (4 if uses_fills2 else 1),
                execution_session_retry_delay_seconds=2.0 if uses_fills2 else 0.05,
                failed_entry_backoff_seconds=10.0 if uses_fills2 else 30.0,
                failed_entry_backoff_after_attempts=5 if uses_fills2 else 3,
                execution_spread_tight_cents=execution_spread_tight_cents,
                execution_spread_wide_cents=execution_spread_wide_cents,
                execution_spread_tight_min_cross_cents=execution_spread_tight_min_cross_cents,
                execution_spread_wide_max_cross_cents=execution_spread_wide_max_cross_cents,
                execution_ladder_delay_high_seconds=execution_ladder_delay_high_seconds,
                execution_ladder_delay_mid_seconds=execution_ladder_delay_mid_seconds,
                execution_ladder_delay_low_seconds=execution_ladder_delay_low_seconds,
                fast_market_tte_minutes=fast_market_tte_minutes,
                fast_market_delay_ceiling_seconds=fast_market_delay_ceiling_seconds,
                resting_entry_max_age_seconds=resting_entry_max_age_seconds,
                resting_entry_max_age_seconds_high=resting_entry_max_age_seconds_high,
                edge_decay_cancel_threshold=edge_decay_cancel_threshold,
                max_contracts_per_order=2,
                max_contracts_per_market=4,
                min_orderbook_fill_fraction=0.0 if uses_fills2 else 0.5,
                daily_vol_floor=0.40,
                daily_min_edge=0.06,
                daily_min_price_cents=65,
                daily_max_price_cents=95,
                daily_min_tte_minutes=8.0,
                daily_max_tte_minutes=50.0,
                daily_min_distance_dollars=200.0,
                daily_max_spread_cents=6,
                daily_min_volume=500.0,
                daily_stop_loss_cents=15,
                daily_fair_value_buffer_cents=3,
                daily_negative_edge_threshold=-0.04,
                daily_min_tte_to_exit_minutes=8.0,
                daily_max_open_markets=3,
                daily_contracts_per_trade=2,
            ),
        )
        if args.once:
            print(json.dumps(trader.run_once(), indent=2, default=str))
            return
        trader.run_forever()
        return

    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()

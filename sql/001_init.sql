CREATE TABLE IF NOT EXISTS btc_candles (
    ts TIMESTAMPTZ NOT NULL,
    exchange TEXT NOT NULL,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    open DOUBLE PRECISION NOT NULL,
    high DOUBLE PRECISION NOT NULL,
    low DOUBLE PRECISION NOT NULL,
    close DOUBLE PRECISION NOT NULL,
    volume DOUBLE PRECISION,
    PRIMARY KEY (ts, exchange, symbol, timeframe)
);

CREATE TABLE IF NOT EXISTS kalshi_market_snapshots (
    observed_at TIMESTAMPTZ NOT NULL,
    source TEXT NOT NULL,
    series_ticker TEXT NOT NULL,
    market_ticker TEXT NOT NULL,
    contract_type TEXT NOT NULL,
    underlying_symbol TEXT NOT NULL,
    expiry TIMESTAMPTZ NOT NULL,
    spot_price DOUBLE PRECISION NOT NULL,
    threshold DOUBLE PRECISION,
    range_low DOUBLE PRECISION,
    range_high DOUBLE PRECISION,
    direction TEXT,
    yes_bid DOUBLE PRECISION,
    yes_ask DOUBLE PRECISION,
    no_bid DOUBLE PRECISION,
    no_ask DOUBLE PRECISION,
    mid_price DOUBLE PRECISION,
    implied_probability DOUBLE PRECISION,
    volume DOUBLE PRECISION,
    open_interest DOUBLE PRECISION,
    settlement_price DOUBLE PRECISION,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (market_ticker, observed_at)
);

CREATE TABLE IF NOT EXISTS kalshi_settlements (
    market_ticker TEXT PRIMARY KEY,
    settled_at TIMESTAMPTZ,
    result TEXT,
    expiration_value DOUBLE PRECISION,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS backtest_runs (
    run_id TEXT PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    strategy_mode TEXT NOT NULL,
    config_json JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS backtest_trades (
    run_id TEXT NOT NULL REFERENCES backtest_runs(run_id),
    trade_id TEXT NOT NULL,
    market_ticker TEXT NOT NULL,
    side TEXT NOT NULL,
    entry_time TIMESTAMPTZ NOT NULL,
    exit_time TIMESTAMPTZ,
    entry_price_cents INTEGER NOT NULL,
    exit_price_cents INTEGER,
    contracts INTEGER NOT NULL,
    exit_trigger TEXT,
    realized_pnl DOUBLE PRECISION,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (run_id, trade_id)
);


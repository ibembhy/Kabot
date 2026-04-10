# Kabot

Kabot is a Bitcoin prediction-market research and live-trading system for Kalshi crypto markets.

Its first job is not to predict "BTC up or down." Its first job is to estimate:

- `P(price_T >= threshold)`

Then it compares that model probability to Kalshi's market-implied probability and asks:

- is there real edge?
- does that edge survive fees and slippage?
- does exiting before settlement help or hurt?

This project is intentionally separate from `CryptoBot` so the architecture can stay small, readable, and probability-first.

## Status

Core system complete, with active live profiles in use:

- Phase 1: discovery / reuse audit
- Phase 2: architecture design
- Phase 3: MVP implementation
- Phase 4: evaluation helpers
- Phase 5: tests + docs + usable CLI

## What Kabot Includes

- Postgres-backed storage for BTC candles and Kalshi market snapshots
- Kalshi market normalization into one internal `MarketSnapshot`
- realized-volatility feature generation
- baseline GBM threshold model
- signal generation from model-vs-market edge
- position and exit logic (signal-break + price-based stops)
- live trading profiles and execution tracing
- backtesting for:
  - hold to settlement
  - pre-expiry exit trading
- evaluation helpers for:
  - volatility sensitivity
  - edge distribution
  - early-exit comparison
  - trade frequency

## Project Structure

```text
Kabot/
|-- config/
|-- sql/
|-- src/kabot/
|   |-- data/
|   |-- storage/
|   |-- markets/
|   |-- models/
|   |-- signals/
|   |-- trading/
|   |-- backtest/
|   `-- reports/
`-- tests/
```

## Core Flow

1. Load BTC candles.
2. Compute realized volatility and basic return features.
3. Normalize Kalshi market snapshots into a consistent shape.
4. Estimate `P(price_T >= threshold)` with the baseline model.
5. Compare model probability vs market probability.
6. Generate:
   - `buy_yes`
   - `buy_no`
   - `no_action`
7. Simulate either:
   - hold to settlement
   - exit before expiry
8. Evaluate PnL, edge, drawdown, hold time, and exit benefit.

## Anti-Leakage Rules

- only use data available at `observed_at`
- volatility is computed only from candles at or before `observed_at`
- settlement values are never used during signal generation
- exit logic only sees snapshots after entry
- backtests use ask for entry and bid for exit

## Quick Start

### 1. Install

```bash
pip install -e .
```

For dev/test work:

```bash
pip install -e .[dev]
```

### 2. Configure environment

Copy `.env.example` and set your DSN:

```text
KABOT_DB_DSN=postgresql://kabot:kabot@localhost/kabot
```

### 3. Initialize the database

```bash
kabot init-db
```

Or with Python directly:

```bash
python -m kabot.cli init-db
```

### 4. Inspect effective config

```bash
kabot show-config
```

### 5. Load real historical data from existing CryptoBot files

Load Kalshi snapshots from the existing SQLite archive:

```bash
kabot load-snapshots --sqlite-path "C:\\Users\\cbemb\\Documents\\CryptoBot\\data\\server_snapshots.sqlite3" --series KXBTCD
```

Load settlement outcomes from CSV:

```bash
kabot load-settlements --csv-path "C:\\Users\\cbemb\\Documents\\CryptoBot\\data\\settled_markets_kxbtcd.csv"
```

Derive BTC candles from the same historical snapshot store:

```bash
kabot derive-btc-candles --sqlite-path "C:\\Users\\cbemb\\Documents\\CryptoBot\\data\\server_snapshots.sqlite3" --series KXBTCD --timeframe 1min
```

Or, better, load a proper BTC minute-candle CSV:

```bash
kabot load-btc-csv --csv-path "C:\\path\\to\\bitcoin-historical-data.csv" --exchange kaggle_bitstamp --timeframe 1m
```

For the Kaggle-style dataset, the defaults already expect columns like:

- `Timestamp`
- `Open`
- `High`
- `Low`
- `Close`
- `Volume_(BTC)`

You can also export settled Kalshi history directly from the public API into a CSV:

```bash
kabot export-kalshi-history --series KXBTC15M --start 2026-03-06T00:00:00Z --end 2026-04-05T23:59:59Z --btc-csv-path "C:\\Users\\cbemb\\Downloads\\archive (2)\\btcusd_1-min_data.csv" --output-csv "C:\\Users\\cbemb\\Documents\\Kabot\\data\\kxbtc15m_30d_history.csv"
```

This uses the BTC CSV as the spot reference and backfills settled Kalshi market candlesticks for near-money contracts.

### 6. Run a first real backtest

Hold-to-settlement:

```bash
kabot run-backtest --series KXBTCD --strategy-mode hold_to_settlement
```

Pre-expiry exit mode:

```bash
kabot run-backtest --series KXBTCD --strategy-mode trade_exit
```

The CLI prints JSON summaries so the first question is easy to answer:

- did the model find trades?
- what was total PnL?
- did early exits beat holding?

### 7. Run tests

```bash
python -m pytest -q
```

## Live Profiles

Kabot has named live profiles for clarity:

- `Kabot` (baseline)
- `K12+SE` (12-minute entry window + signal-break exits)
- `+Fills` (12-minute + signal-break + better fills execution path)

The `+Fills` profile currently uses resting GTC entry orders, and logs entry fills with GBM edge data so we can evaluate whether model edge correlates with real outcomes.

## Configuration

The default config lives in `config/default.toml`.


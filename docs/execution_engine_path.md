# Execution Engine Path

## Goal

Improve **live fill capture** without continuing to patch the existing poll-loop logic one tweak at a time.

The current evidence suggests:

- signal detection is often good enough to find trades
- live fills are the weak point
- the current architecture mixes:
  - market discovery
  - signal evaluation
  - order submission
  - retry logic
  - position management
- that makes it hard to reason about latency and hard to measure where fills are lost

This document describes the next architecture path.

## What We Know

From current live measurements:

- new market pickup can be delayed by tens of seconds
- once a market is active, Kabot samples roughly every 2 seconds
- valid signals often persist for some time
- fills still fail repeatedly even after:
  - IOC
  - larger crossing
  - retries
  - slicing
  - websocket data

That means the next gain is likely to come from a **better execution path**, not another small filter tweak.

## Target Design

Split the live system into 4 layers:

1. **Market State Layer**
2. **Signal Layer**
3. **Execution Layer**
4. **Observability Layer**

The current `LiveTrader` does parts of all 4. The new path should separate them.

## 1. Market State Layer

Own a live in-memory state per ticker.

Per market we want:

- ticker
- threshold
- expiry
- last spot price
- last yes/no bid/ask
- last orderbook depth
- volume
- open interest
- timestamps for each update source

Key idea:

- websocket updates mutate this state continuously
- metadata refresh only fills in fields websocket does not provide
- signal/execution logic reads from this in-memory state instead of rebuilding everything inside each loop

Suggested type:

- `ExecutionMarketState`

Suggested new file:

- `src/kabot/trading/execution_state.py`

## 2. Signal Layer

Signal generation should become a pure function over market state.

Inputs:

- `ExecutionMarketState`
- model volatility
- strategy configuration

Outputs:

- eligible / not eligible
- side
- strategy name
- signal price
- GBM edge
- confidence
- reason if rejected

Key idea:

- the signal layer should **not** place orders
- it should only say:
  - this market is tradable now
  - or not

Suggested type:

- `ExecutionSignal`

Suggested new file:

- `src/kabot/trading/signal_engine.py`

## 3. Execution Layer

This is the main new path.

When a signal becomes tradable:

- start an execution routine for that ticker
- keep it isolated from the rest of the universe
- do not wait for the next full poll cycle

This routine owns:

- entry attempt start time
- attempt count
- per-attempt limit price
- IOC / fill results
- partial fills
- abort conditions

### Execution Loop

For one ticker:

1. read freshest market state
2. confirm signal still valid
3. compute execution ladder price
4. submit IOC
5. collect fill / cancel result
6. if partially or fully filled:
   - update execution state
7. if no fill and signal still valid:
   - refresh state immediately
   - reprice
   - retry
8. stop when:
   - target size filled
   - signal breaks
   - max ladder reached
   - time budget exceeded

The important difference from today:

- retries happen inside a **dedicated execution routine**
- not inside the outer whole-bot polling cycle

Suggested type:

- `ExecutionSession`

Suggested new file:

- `src/kabot/trading/execution_engine.py`

## 4. Observability Layer

This is mandatory.

We need hard timestamps for:

- signal first detected
- execution session started
- each quote used
- each order submitted
- exchange acknowledgement
- fill time
- cancel time
- final outcome

Without this, execution work becomes guessing.

Suggested event log record:

- `ExecutionTraceEvent`

Suggested storage target:

- write structured JSON lines locally first
- later optionally add Postgres table

Suggested new file:

- `src/kabot/trading/execution_trace.py`

## Event Flow

### Today

`run_once()`

- build snapshots
- generate candidates
- submit order immediately
- maybe retry
- sleep

### Target

`market state update`

- websocket or metadata refresh updates ticker state

`signal evaluation`

- recompute eligibility for that ticker

`execution session spawn`

- if ticker transitions from not-eligible -> eligible
- and no active execution session exists
- start execution routine for that ticker

`execution session`

- attacks only that ticker
- uses fresh state and short retry ladder

`position manager`

- tracks fills and open positions
- handles exit logic separately

## First Safe Implementation Slice

Do **not** rewrite the whole bot first.

Implement the smallest slice that proves whether better execution helps:

### Slice 1

Build an **entry execution session** for one ticker while leaving the rest of `LiveTrader` intact.

Specifically:

1. keep current signal rules
2. keep current filters
3. keep current config/profile system
4. replace only `_submit_order()` behavior with a dedicated execution session object

This session should:

- receive a single `StrategyCandidate`
- own a short time budget, e.g. 400-800ms
- attempt 2-3 IOC orders using fresh state each time
- log every step

If this slice does not materially improve fills, we stop patching and conclude the market is not capturable enough with current tech.

### Why this slice first

- small blast radius
- baseline logic remains available
- easy rollback
- high information value

## Rollout Strategy

Keep the current profile system.

Profiles:

- `baseline_live`
- `exp_12m_signal_break`
- future:
  - `exp_execution_session`

Do not overwrite baseline behavior.

Each experiment should be selectable by profile so rollback stays one command.

## Concrete Build Order

### Phase 1

- add `ExecutionMarketState`
- add `ExecutionSignal`
- add `ExecutionTraceEvent`
- add trace logging around current `_submit_order()`

### Phase 2

- implement `ExecutionSession`
- use it only for entry order handling
- keep current selection loop intact

### Phase 3

- move eligibility evaluation to event-driven updates for already-known tickers
- reduce dependence on outer 2-second loop for active markets

### Phase 4

- consider a dedicated exit execution session if entry improvements are real

## What Not To Do

Avoid these for now:

- more random cross increases without measurement
- more ad hoc retry rules in `run_once()`
- complex averaging-down logic
- big strategy rewrites before execution is measured

## Success Criteria

The new execution path is worth keeping only if it improves at least one of these in real live data:

- fill rate per valid signal
- fill rate per order
- realized profit capture on signals the old path missed
- lower ratio of repeated zero-fill attempts on the same ticker

If it does not improve those, the issue is probably:

- market structure
- liquidity
- or exchange/latency limits beyond current tech


"""
Projected compounding growth curve from real backtest trade data.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kabot.backtest.engine import BacktestConfig, BacktestEngine
from kabot.data.features import build_feature_frame
from kabot.models.gbm_threshold import GBMThresholdModel
from kabot.signals.engine import SignalConfig
from kabot.trading.exits import ExitConfig

# ── Config ────────────────────────────────────────────────────────────────────
STARTING_BANKROLL = 45.00
N_SIMULATIONS = 500
OUTPUT_PATH = Path("data/reports/growth_projection.png")

# ── Load and run backtest ─────────────────────────────────────────────────────
csv_path = Path(__file__).resolve().parents[1] / "data" / "kxbtc15m_30d_history.csv"
snapshots = pd.read_csv(csv_path)
for col in ("observed_at", "expiry"):
    snapshots[col] = pd.to_datetime(snapshots[col], utc=True, errors="coerce")

spot = (
    snapshots[["observed_at", "spot_price"]]
    .dropna().drop_duplicates("observed_at")
    .sort_values("observed_at")
    .rename(columns={"observed_at": "ts", "spot_price": "close"})
    .set_index("ts")
)
for c in ("open", "high", "low"):
    spot[c] = spot["close"]
spot["volume"] = 0.0
features = build_feature_frame(spot, volatility_window=12, annualization_factor=105120.0)

result = BacktestEngine(
    model=GBMThresholdModel(drift=0.0, volatility_floor=0.05),
    signal_config=SignalConfig(
        min_edge=0.05, min_contract_price_cents=35, max_contract_price_cents=65,
        max_spread_cents=6, max_near_money_bps=500.0, min_confidence=0.0,
    ),
    exit_config=ExitConfig(take_profit_cents=8, stop_loss_cents=10,
                           fair_value_buffer_cents=3, time_exit_minutes=2),
    backtest_config=BacktestConfig(
        strategy_mode="hold_to_settlement",
        entry_slippage_cents=1, exit_slippage_cents=1, fee_rate_bps=0.0,
    ),
).run(snapshots, features)

trades = result.trades.copy()
trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
trades["day"] = trades["entry_time"].dt.date

# ── Daily PnL (backtest: 1 contract per trade at ~$0.52 avg) ─────────────────
daily_pnl = (
    trades.groupby("day")["realized_pnl"]
    .sum()
    .reset_index()
    .sort_values("day")
)
# Per-contract daily PnL in dollars
daily_pnl_dollars = daily_pnl["realized_pnl"].values / 100.0
n_days = len(daily_pnl_dollars)

# ── Contract scaling by bankroll ──────────────────────────────────────────────
# IOC orders: max 2 contracts per order (current live setting)
# As bankroll grows: contracts step up at natural thresholds
AVG_PRICE_DOLLARS = 0.50
PER_MARKET_FRACTION = 0.08
MAX_CONTRACTS_ORDER = 2    # current live setting (max_contracts_per_order)
MAX_CONTRACTS_MARKET = 4   # ceiling (max_contracts_per_market)

def contracts_at(bankroll: float, use_market_max: bool = False) -> int:
    cap = MAX_CONTRACTS_MARKET if use_market_max else MAX_CONTRACTS_ORDER
    target = bankroll * PER_MARKET_FRACTION
    c = int(target // AVG_PRICE_DOLLARS)
    return max(1, min(c, cap))

# ── Simulate ──────────────────────────────────────────────────────────────────
def simulate(bankroll_start: float, daily_pnl_arr: np.ndarray,
             shuffle: bool = False, use_market_max: bool = False) -> np.ndarray:
    rng = np.random.default_rng()
    seq = rng.permutation(daily_pnl_arr) if shuffle else daily_pnl_arr.copy()
    equity = [bankroll_start]
    bk = bankroll_start
    for day_pnl in seq:
        scale = contracts_at(bk, use_market_max=use_market_max)
        bk = max(bk + day_pnl * scale, 0.01)
        equity.append(bk)
    return np.array(equity)

# Conservative (2 contracts max / IOC)
baseline_cons = simulate(STARTING_BANKROLL, daily_pnl_dollars, shuffle=False, use_market_max=False)
sims_cons = np.array([simulate(STARTING_BANKROLL, daily_pnl_dollars, shuffle=True, use_market_max=False)
                      for _ in range(N_SIMULATIONS)])

# Optimistic (up to 4 contracts / market max)
baseline_opt = simulate(STARTING_BANKROLL, daily_pnl_dollars, shuffle=False, use_market_max=True)
sims_opt = np.array([simulate(STARTING_BANKROLL, daily_pnl_dollars, shuffle=True, use_market_max=True)
                     for _ in range(N_SIMULATIONS)])

p10_cons, p50_cons, p90_cons = [np.percentile(sims_cons, p, axis=0) for p in (10, 50, 90)]
p10_opt,  p50_opt,  p90_opt  = [np.percentile(sims_opt,  p, axis=0) for p in (10, 50, 90)]
day_idx = np.arange(n_days + 1)

# ── Summary ───────────────────────────────────────────────────────────────────
summary = {
    "starting_bankroll_usd": STARTING_BANKROLL,
    "trading_days_in_data": int(n_days),
    "total_trades_in_data": int(len(trades)),
    "win_rate_pct": round(float((trades["realized_pnl"] > 0).mean()) * 100, 1),
    "contracts_at_45usd_conservative": contracts_at(45, use_market_max=False),
    "contracts_at_45usd_optimistic": contracts_at(45, use_market_max=True),
    "contracts_at_100usd": contracts_at(100, use_market_max=True),
    "conservative_30d": {
        "p10": round(float(p10_cons[-1]), 2),
        "median": round(float(p50_cons[-1]), 2),
        "p90": round(float(p90_cons[-1]), 2),
    },
    "optimistic_30d": {
        "p10": round(float(p10_opt[-1]), 2),
        "median": round(float(p50_opt[-1]), 2),
        "p90": round(float(p90_opt[-1]), 2),
    },
}
print(json.dumps(summary, indent=2))

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(13, 10))

# Top: portfolio projection
ax = axes[0]
ax.fill_between(day_idx, p10_cons, p90_cons, alpha=0.15, color="#FF5722", label="_nolegend_")
ax.plot(day_idx, p50_cons, color="#FF5722", linewidth=2,
        label=f"Conservative (2 contracts/order) median: ${p50_cons[-1]:.0f}")
ax.fill_between(day_idx, p10_opt, p90_opt, alpha=0.15, color="#2196F3", label="_nolegend_")
ax.plot(day_idx, p50_opt, color="#2196F3", linewidth=2,
        label=f"Optimistic (up to 4 contracts) median: ${p50_opt[-1]:.0f}")
ax.axhline(STARTING_BANKROLL, color="gray", linewidth=1, linestyle=":", alpha=0.8,
           label=f"Start: ${STARTING_BANKROLL:.0f}")
ax.set_title(
    f"30-Day Compounding Projection from ${STARTING_BANKROLL:.0f}  |  "
    f"Win rate {summary['win_rate_pct']}%  |  {summary['total_trades_in_data']} historical trades  |  "
    f"{N_SIMULATIONS} simulations",
    fontsize=11, pad=10
)
ax.set_xlabel("Trading Day", fontsize=10)
ax.set_ylabel("Portfolio Value ($)", fontsize=10)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}"))
ax.legend(fontsize=9, loc="upper left")
ax.grid(True, alpha=0.3)
# Annotate
for val, label, color in [
    (p90_opt[-1],  f"Best 10% (opt): ${p90_opt[-1]:.0f}",   "#1565C0"),
    (p50_opt[-1],  f"Median (opt): ${p50_opt[-1]:.0f}",      "#2196F3"),
    (p50_cons[-1], f"Median (cons): ${p50_cons[-1]:.0f}",    "#FF5722"),
    (p10_cons[-1], f"Worst 10% (cons): ${p10_cons[-1]:.0f}", "#BF360C"),
]:
    ax.annotate(label, xy=(day_idx[-1], val), xytext=(day_idx[-1] - 0.5, val),
                fontsize=8, color=color, fontweight="bold", ha="right", va="center")

# Bottom: contracts vs bankroll
ax2 = axes[1]
bk_range = np.linspace(5, 300, 500)
ax2.plot(bk_range, [contracts_at(b, use_market_max=False) for b in bk_range],
         color="#FF5722", linewidth=2, label="Conservative (max 2 per order)")
ax2.plot(bk_range, [contracts_at(b, use_market_max=True) for b in bk_range],
         color="#2196F3", linewidth=2, label="Optimistic (max 4 per market)")
ax2.axvline(STARTING_BANKROLL, color="gray", linestyle="--", linewidth=1.5,
            label=f"Current bankroll: ${STARTING_BANKROLL:.0f}")
ax2.set_title("Contracts per Trade vs Portfolio Size  (how compounding accelerates)", fontsize=11)
ax2.set_xlabel("Portfolio Value ($)", fontsize=10)
ax2.set_ylabel("Contracts per Trade", fontsize=10)
ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}"))
ax2.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout(pad=2.5)
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUTPUT_PATH, dpi=150)
print(f"Chart saved: {OUTPUT_PATH}")

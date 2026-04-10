"""
Time-of-day analysis for Kabot.

Sources:
  • btcusd_1min.csv         — BTC volatility, volume, direction bias by EST hour
  • kxbtc15m_30d_history.csv — backtest PnL/win-rate/edge by EST hour
  • Same CSV for Kalshi spread/activity by hour

Output: data/reports/time_of_day_analysis.png  +  terminal table
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TZ = "US/Eastern"

# ── BTC 1-min helpers ────────────────────────────────────────────────────────

def load_1min_btc(csv_path: Path, days: int = 90) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    df["ts"] = pd.to_datetime(df["Timestamp"], unit="s", utc=True).dt.tz_convert(TZ)
    cutoff = df["ts"].max() - pd.Timedelta(days=days)
    df = df[df["ts"] >= cutoff].copy()
    df["hour"] = df["ts"].dt.hour
    df["ret"] = df["Close"].pct_change()
    return df


def btc_hourly_stats(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby("hour")
    # annualise from 1-min returns: sqrt(60 * 24 * 365) ≈ 730
    vol     = grp["ret"].std() * np.sqrt(60 * 24 * 365)
    up_pct  = grp["ret"].apply(lambda x: (x > 0).mean() * 100)
    avg_abs = grp["ret"].apply(lambda x: x.abs().mean() * 100)
    return pd.DataFrame({
        "ann_vol":    vol,
        "up_pct":     up_pct,
        "avg_abs_ret": avg_abs,
    }).reset_index()

# ── Backtest helpers ─────────────────────────────────────────────────────────

def build_engine():
    from kabot.backtest.engine import BacktestConfig, BacktestEngine
    from kabot.models.gbm_threshold import GBMThresholdModel
    from kabot.settings import load_settings
    from kabot.signals.engine import SignalConfig
    from kabot.trading.exits import ExitConfig

    s = load_settings()
    return BacktestEngine(
        model=GBMThresholdModel(
            drift=float(s.model["drift"]),
            volatility_floor=float(s.model["volatility_floor"]),
        ),
        signal_config=SignalConfig(
            min_edge=float(s.signal["min_edge"]),
            min_contract_price_cents=int(s.signal["min_contract_price_cents"]),
            max_contract_price_cents=int(s.signal["max_contract_price_cents"]),
            max_spread_cents=int(s.signal["max_spread_cents"]),
            max_near_money_bps=float(s.signal["max_near_money_bps"]),
            min_confidence=float(s.signal["min_confidence"]),
        ),
        exit_config=ExitConfig(
            take_profit_cents=int(s.trading["take_profit_cents"]),
            stop_loss_cents=int(s.trading["stop_loss_cents"]),
            fair_value_buffer_cents=int(s.trading["fair_value_buffer_cents"]),
            time_exit_minutes=int(s.trading["time_exit_minutes_before_expiry"]),
        ),
        backtest_config=BacktestConfig(
            strategy_mode="hold_to_settlement",
            entry_slippage_cents=int(s.backtest["entry_slippage_cents"]),
            exit_slippage_cents=int(s.backtest["exit_slippage_cents"]),
            fee_rate_bps=float(s.backtest["fee_rate_bps"]),
            allow_reentry=bool(s.trading["allow_reentry"]),
        ),
    )


def run_backtest_trades(csv_path: Path) -> pd.DataFrame:
    from kabot.data.features import build_feature_frame
    from kabot.settings import load_settings

    s = load_settings()

    snaps = pd.read_csv(csv_path)
    for col in ("observed_at", "expiry"):
        if col in snaps.columns:
            snaps[col] = pd.to_datetime(snaps[col], utc=True, errors="coerce")

    # derive features from spot_price column (same approach as CLI --snapshots-csv)
    spot = (
        snaps[["observed_at", "spot_price"]]
        .dropna(subset=["observed_at", "spot_price"])
        .drop_duplicates(subset=["observed_at"])
        .sort_values("observed_at")
        .rename(columns={"observed_at": "ts", "spot_price": "close"})
        .set_index("ts")
    )
    spot["open"] = spot["close"]
    spot["high"] = spot["close"]
    spot["low"]  = spot["close"]
    spot["volume"] = 0.0
    features = build_feature_frame(
        spot,
        volatility_window=int(s.data["volatility_window"]),
        annualization_factor=float(s.data["annualization_factor"]),
    )

    engine = build_engine()
    result = engine.run(snaps, features)
    trades = result.trades
    if trades is None or trades.empty:
        return pd.DataFrame()

    if "entry_time" in trades.columns:
        trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True, errors="coerce").dt.tz_convert(TZ)
        trades["hour"] = trades["entry_time"].dt.hour
    return trades

# ── Snapshot helpers ─────────────────────────────────────────────────────────

def snapshot_hourly(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["observed_at"] = pd.to_datetime(df["observed_at"], utc=True, errors="coerce").dt.tz_convert(TZ)
    df["hour"]   = df["observed_at"].dt.hour
    df["spread"] = df["yes_ask"] - df["yes_bid"]
    return df.groupby("hour").agg(
        snapshot_count=("yes_ask", "count"),
        avg_spread=("spread", "mean"),
        avg_implied_prob=("implied_probability", "mean"),
    ).reset_index()

# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    snap_csv = ROOT / "data" / "kxbtc15m_30d_history.csv"
    btc_csv  = ROOT / "data" / "btcusd_1min.csv"

    print("Loading BTC 1-min data (last 90 days)…")
    btc_raw = load_1min_btc(btc_csv, days=90)
    btc_h   = btc_hourly_stats(btc_raw)

    print("Loading Kalshi snapshot activity…")
    snap_h = snapshot_hourly(snap_csv)

    print("Running backtest for trade-level data…")
    trades = run_backtest_trades(snap_csv)

    if trades.empty or "hour" not in trades.columns:
        print("ERROR: no trades from backtest.")
        sys.exit(1)

    pnl_col = "realized_pnl"
    trades["win"] = trades[pnl_col] > 0

    trade_h = trades.groupby("hour").agg(
        trade_count=(pnl_col, "count"),
        win_rate=("win", "mean"),
        avg_pnl=(pnl_col, "mean"),
        total_pnl=(pnl_col, "sum"),
        pnl_std=(pnl_col, "std"),
    ).reset_index()
    trade_h["win_rate_pct"] = trade_h["win_rate"] * 100

    hours  = pd.DataFrame({"hour": range(24)})
    merged = (hours
              .merge(btc_h,   on="hour", how="left")
              .merge(snap_h,  on="hour", how="left")
              .merge(trade_h, on="hour", how="left"))

    # ── text table ────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print(f"{'Hr(EST)':>8} {'Trades':>7} {'WinRate':>8} {'AvgPnL':>8} {'TotPnL':>9}"
          f" {'AnnVol':>8} {'UpPct':>7} {'Spread':>7}")
    print("-" * 78)
    for _, r in merged.iterrows():
        tc = r.get("trade_count")
        if pd.isna(tc) or tc == 0:
            av = r["ann_vol"] * 100 if not pd.isna(r.get("ann_vol", float("nan"))) else 0
            up = r["up_pct"] if not pd.isna(r.get("up_pct", float("nan"))) else 0
            print(f"{int(r['hour']):>3}  {'—':>7}  {'—':>8}  {'—':>8}  {'—':>9}"
                  f"  {av:>7.0f}%  {up:>6.1f}%  {'—':>7}")
            continue
        av = r["ann_vol"] * 100 if not pd.isna(r.get("ann_vol", float("nan"))) else 0
        up = r["up_pct"]   if not pd.isna(r.get("up_pct",  float("nan"))) else 0
        sp = r["avg_spread"] if not pd.isna(r.get("avg_spread", float("nan"))) else 0
        print(
            f"{int(r['hour']):>3}"
            f"  {int(tc):>7}"
            f"  {r['win_rate_pct']:>7.1f}%"
            f"  {r['avg_pnl']:>7.1f}c"
            f"  {r['total_pnl']:>8.0f}c"
            f"  {av:>7.0f}%"
            f"  {up:>6.1f}%"
            f"  {sp:>6.2f}c"
        )
    print("=" * 78)

    active = trade_h[trade_h["trade_count"] > 0]
    best_h  = int(active.loc[active["total_pnl"].idxmax(),  "hour"])
    worst_h = int(active.loc[active["total_pnl"].idxmin(),  "hour"])
    best_wr = int(active.loc[active["win_rate"].idxmax(),   "hour"])
    most_t  = int(active.loc[active["trade_count"].idxmax(),"hour"])
    print(f"\nBest  total PnL hour  : {best_h:02d}:00 UTC")
    print(f"Worst total PnL hour  : {worst_h:02d}:00 UTC")
    print(f"Highest win-rate hour : {best_wr:02d}:00 UTC")
    print(f"Most trades hour      : {most_t:02d}:00 UTC")

    # Correlation: does higher BTC vol correlate with better win-rate?
    comb = active.merge(btc_h, on="hour")
    corr_vol_wr  = comb["ann_vol"].corr(comb["win_rate"])
    corr_vol_pnl = comb["ann_vol"].corr(comb["avg_pnl"])
    print(f"\nCorrelation (BTC vol vs win-rate) : {corr_vol_wr:+.3f}")
    print(f"Correlation (BTC vol vs avg PnL)  : {corr_vol_pnl:+.3f}")

    # ── plot ──────────────────────────────────────────────────────────────
    hr  = merged["hour"].values
    fig = plt.figure(figsize=(18, 15))
    fig.suptitle("Kabot — Time-of-Day Analysis (EST hours, 30-day backtest)", fontsize=15, fontweight="bold")
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.48, wspace=0.30)

    vlines = [(4, "gray", "--", "EU open 04:00 EST"),
              (8, "darkorange", "--", "US open 08:00 EST"),
              (16, "gray", ":", "US close 16:00 EST")]

    def annotate(ax):
        for x, col, ls, _ in vlines:
            ax.axvline(x=x, color=col, linestyle=ls, alpha=0.45, linewidth=1.1)
        ax.set_xticks(range(0, 24, 2))
        ax.set_xlabel("EST Hour", fontsize=9)

    # 1 — trade count
    ax1 = fig.add_subplot(gs[0, 0])
    tc_vals = merged["trade_count"].fillna(0).values
    ax1.bar(hr, tc_vals, color="steelblue", alpha=0.8, width=0.7)
    ax1.set_title("Trade Count by Hour", fontsize=11, fontweight="bold")
    ax1.set_ylabel("# Trades")
    annotate(ax1)

    # 2 — win rate
    ax2 = fig.add_subplot(gs[0, 1])
    wr_vals = merged["win_rate_pct"].values.astype(float)
    ax2.bar(hr, np.nan_to_num(wr_vals), color="seagreen", alpha=0.8, width=0.7)
    mean_wr = float(np.nanmean(wr_vals))
    ax2.axhline(y=mean_wr, color="red", linestyle="--", linewidth=1.3, label=f"mean {mean_wr:.1f}%")
    ax2.set_title("Win Rate by Hour", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Win Rate (%)")
    ax2.legend(fontsize=8)
    annotate(ax2)

    # 3 — avg PnL per trade
    ax3 = fig.add_subplot(gs[1, 0])
    ap_vals = merged["avg_pnl"].values.astype(float)
    colors3 = ["green" if (not np.isnan(v) and v > 0) else "red" for v in ap_vals]
    ax3.bar(hr, np.nan_to_num(ap_vals), color=colors3, alpha=0.8, width=0.7)
    ax3.axhline(y=0, color="black", linewidth=0.8)
    ax3.set_title("Avg PnL per Trade (cents)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Avg PnL (¢)")
    annotate(ax3)

    # 4 — total PnL
    ax4 = fig.add_subplot(gs[1, 1])
    tp_vals = merged["total_pnl"].values.astype(float)
    colors4 = ["green" if (not np.isnan(v) and v > 0) else "red" for v in tp_vals]
    ax4.bar(hr, np.nan_to_num(tp_vals), color=colors4, alpha=0.8, width=0.7)
    ax4.axhline(y=0, color="black", linewidth=0.8)
    ax4.set_title("Total PnL by Hour (cents)", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Total PnL (¢)")
    annotate(ax4)

    # 5 — BTC annualised vol
    ax5 = fig.add_subplot(gs[2, 0])
    av_vals = merged["ann_vol"].fillna(0).values * 100
    ax5.bar(hr, av_vals, color="darkorange", alpha=0.8, width=0.7)
    ax5.set_title("BTC Annualised Volatility by Hour (90-day, 1-min, EST)", fontsize=11, fontweight="bold")
    ax5.set_ylabel("Ann. Vol (%)")
    annotate(ax5)

    # 6 — Kalshi spread
    ax6 = fig.add_subplot(gs[2, 1])
    sp_vals = merged["avg_spread"].fillna(0).values
    ax6.bar(hr, sp_vals, color="mediumpurple", alpha=0.8, width=0.7)
    ax6.set_title("Kalshi Avg YES Spread by Hour", fontsize=11, fontweight="bold")
    ax6.set_ylabel("Spread (¢)")
    annotate(ax6)

    from matplotlib.lines import Line2D
    legend_els = [Line2D([0],[0], color=c, linestyle=ls, label=lbl) for _,c,ls,lbl in vlines]
    fig.legend(handles=legend_els, loc="lower center", ncol=3, fontsize=9, frameon=True)

    out = ROOT / "data" / "reports" / "time_of_day_analysis.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nChart saved: {out}")


if __name__ == "__main__":
    main()

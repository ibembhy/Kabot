from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRADES = ROOT / "data" / "reports" / "30d_kxbtc15m_report_pack" / "k12_se_reentry_30d_trades.csv"
DEFAULT_SNAPSHOTS = ROOT / "data" / "kxbtc15m_30d_history_complement.csv"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "reports" / "30d_kxbtc15m_report_pack"


def _load_settlement_frame(snapshots_csv: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        snapshots_csv,
        usecols=["market_ticker", "threshold", "settlement_price", "direction"],
    )
    frame = frame.dropna(subset=["market_ticker", "threshold", "settlement_price"]).copy()
    return (
        frame.sort_values("market_ticker")
        .groupby("market_ticker", as_index=False)
        .last()
    )


def _hold_exit_price_cents(row: pd.Series) -> int:
    settles_yes = float(row["settlement_price"]) >= float(row["threshold"])
    if str(row.get("direction", "")).lower() in {"below", "down"}:
        settles_yes = float(row["settlement_price"]) < float(row["threshold"])
    is_win = (row["side"] == "yes" and settles_yes) or (row["side"] == "no" and not settles_yes)
    return 100 if is_win else 0


def _summary_stats(frame: pd.DataFrame, *, actual_col: str, hold_col: str) -> dict[str, float | int]:
    if frame.empty:
        return {
            "count": 0,
            "hold_better_count": 0,
            "hold_worse_count": 0,
            "tie_count": 0,
            "hold_better_rate_pct": 0.0,
            "actual_pnl_cents": 0.0,
            "hold_pnl_cents": 0.0,
            "net_hold_minus_actual_cents": 0.0,
            "median_hold_minus_actual_cents": 0.0,
            "mean_hold_minus_actual_cents": 0.0,
        }
    delta = frame[hold_col].astype(float) - frame[actual_col].astype(float)
    return {
        "count": int(len(frame)),
        "hold_better_count": int((delta > 0).sum()),
        "hold_worse_count": int((delta < 0).sum()),
        "tie_count": int((delta == 0).sum()),
        "hold_better_rate_pct": float((delta > 0).mean() * 100.0),
        "actual_pnl_cents": float(frame[actual_col].sum()),
        "hold_pnl_cents": float(frame[hold_col].sum()),
        "net_hold_minus_actual_cents": float(delta.sum()),
        "median_hold_minus_actual_cents": float(delta.median()),
        "mean_hold_minus_actual_cents": float(delta.mean()),
    }


def build_analysis(trades_csv: Path, snapshots_csv: Path) -> tuple[dict[str, dict[str, float | int]], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trades = pd.read_csv(trades_csv)
    settlements = _load_settlement_frame(snapshots_csv)

    enriched = trades.merge(settlements, on="market_ticker", how="left")
    enriched["hold_exit_price_cents"] = enriched.apply(_hold_exit_price_cents, axis=1)
    enriched["hold_pnl_cents"] = enriched["hold_exit_price_cents"] - enriched["entry_price_cents"]
    enriched["hold_minus_actual_cents"] = enriched["hold_pnl_cents"] - enriched["realized_pnl_cents"]
    enriched["is_early_exit"] = enriched["exit_trigger"] != "settlement"

    early = enriched[enriched["is_early_exit"]].copy()
    initial_early = early[early["trade_kind"] == "initial"].copy()
    reentry_early = early[early["trade_kind"] == "reentry"].copy()

    market_rows: list[dict[str, object]] = []
    ordered = enriched.sort_values(["market_ticker", "entry_time", "trade_kind"])
    for market_ticker, group in ordered.groupby("market_ticker", sort=False):
        first_trade = group.iloc[0]
        market_rows.append(
            {
                "market_ticker": market_ticker,
                "trade_count": int(len(group)),
                "had_early_exit": bool((group["is_early_exit"]).any()),
                "had_reentry": bool((group["trade_kind"] == "reentry").any()),
                "actual_total_pnl_cents": float(group["realized_pnl_cents"].sum()),
                "hold_from_first_pnl_cents": float(first_trade["hold_pnl_cents"]),
                "hold_minus_actual_cents": float(first_trade["hold_pnl_cents"] - group["realized_pnl_cents"].sum()),
            }
        )
    market_level = pd.DataFrame(market_rows)

    summary = {
        "decision_level_all_early_exits": _summary_stats(early, actual_col="realized_pnl_cents", hold_col="hold_pnl_cents"),
        "decision_level_initial_early_exits": _summary_stats(initial_early, actual_col="realized_pnl_cents", hold_col="hold_pnl_cents"),
        "decision_level_reentry_early_exits": _summary_stats(reentry_early, actual_col="realized_pnl_cents", hold_col="hold_pnl_cents"),
        "market_level_markets_with_early_exit": _summary_stats(
            market_level[market_level["had_early_exit"]],
            actual_col="actual_total_pnl_cents",
            hold_col="hold_from_first_pnl_cents",
        ),
        "market_level_markets_with_reentry": _summary_stats(
            market_level[market_level["had_reentry"]],
            actual_col="actual_total_pnl_cents",
            hold_col="hold_from_first_pnl_cents",
        ),
        "market_level_early_exit_without_reentry": _summary_stats(
            market_level[(market_level["had_early_exit"]) & (~market_level["had_reentry"])],
            actual_col="actual_total_pnl_cents",
            hold_col="hold_from_first_pnl_cents",
        ),
    }

    reason_breakdown = (
        early.groupby(["trade_kind", "exit_trigger"], observed=False)
        .agg(
            exits=("market_ticker", "size"),
            hold_better_count=("hold_minus_actual_cents", lambda s: int((s > 0).sum())),
            hold_worse_count=("hold_minus_actual_cents", lambda s: int((s < 0).sum())),
            tie_count=("hold_minus_actual_cents", lambda s: int((s == 0).sum())),
            actual_pnl_cents=("realized_pnl_cents", "sum"),
            hold_pnl_cents=("hold_pnl_cents", "sum"),
            net_hold_minus_actual_cents=("hold_minus_actual_cents", "sum"),
            avg_hold_minus_actual_cents=("hold_minus_actual_cents", "mean"),
        )
        .reset_index()
        .sort_values(["trade_kind", "exit_trigger"])
    )

    return summary, early, market_level, reason_breakdown


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare early exits against hold-to-settlement outcomes.")
    parser.add_argument("--trades-csv", default=str(DEFAULT_TRADES), help="Path to trade export CSV.")
    parser.add_argument("--snapshots-csv", default=str(DEFAULT_SNAPSHOTS), help="Path to snapshots CSV with settlement prices.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for summary artifacts.")
    args = parser.parse_args()

    trades_csv = Path(args.trades_csv)
    snapshots_csv = Path(args.snapshots_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary, early, market_level, reason_breakdown = build_analysis(trades_csv=trades_csv, snapshots_csv=snapshots_csv)

    stem = trades_csv.stem.replace("_trades", "")
    summary_path = output_dir / f"{stem}_hold_vs_settlement_summary.json"
    early_path = output_dir / f"{stem}_early_exit_hold_vs_settlement.csv"
    markets_path = output_dir / f"{stem}_market_level_hold_vs_settlement.csv"
    reason_path = output_dir / f"{stem}_hold_vs_settlement_by_reason.csv"

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    early.sort_values("hold_minus_actual_cents", ascending=False).to_csv(early_path, index=False)
    market_level.sort_values("hold_minus_actual_cents", ascending=False).to_csv(markets_path, index=False)
    reason_breakdown.to_csv(reason_path, index=False)

    print(json.dumps(summary, indent=2))
    print(f"Saved summary to: {summary_path}")
    print(f"Saved early-exit detail to: {early_path}")
    print(f"Saved market-level detail to: {markets_path}")
    print(f"Saved reason breakdown to: {reason_path}")


if __name__ == "__main__":
    main()

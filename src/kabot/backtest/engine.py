from __future__ import annotations

from dataclasses import dataclass
from math import erf, sqrt

import numpy as np
import pandas as pd

from kabot.models.base import ProbabilityModel
from kabot.signals.engine import SignalConfig, generate_signal
from kabot.trading.exits import ExitConfig, evaluate_exit
from kabot.trading.positions import open_position
from kabot.types import BacktestResult, MarketSnapshot, ProbabilityEstimate

from .metrics import build_metrics, edge_distribution


@dataclass(frozen=True)
class BacktestConfig:
    strategy_mode: str = "hold_to_settlement"
    contracts_per_trade: int = 1
    entry_slippage_cents: int = 0
    exit_slippage_cents: int = 0
    fee_rate_bps: float = 0.0
    allow_reentry: bool = False


class BacktestEngine:
    def __init__(
        self,
        *,
        model: ProbabilityModel,
        signal_config: SignalConfig,
        exit_config: ExitConfig,
        backtest_config: BacktestConfig,
    ) -> None:
        self.model = model
        self.signal_config = signal_config
        self.exit_config = exit_config
        self.backtest_config = backtest_config

    def run(self, snapshots: pd.DataFrame, features: pd.DataFrame) -> BacktestResult:
        if snapshots.empty:
            return BacktestResult(strategy_mode=self.backtest_config.strategy_mode, trades=pd.DataFrame(), summary={})

        if self.backtest_config.strategy_mode == "hold_to_settlement":
            trades_frame = self._run_hold_to_settlement_vectorized(snapshots, features)
        else:
            trades: list[dict[str, object]] = []
            for _, market_frame in snapshots.sort_values(["market_ticker", "observed_at"]).groupby("market_ticker", sort=False):
                trades.extend(self._simulate_market(market_frame.reset_index(drop=True), features))
            trades_frame = pd.DataFrame(trades)
        summary = build_metrics(trades_frame)
        summary.update(edge_distribution(trades_frame))
        return BacktestResult(
            strategy_mode=self.backtest_config.strategy_mode,
            trades=trades_frame,
            summary=summary,
        )

    def _run_hold_to_settlement_vectorized(self, snapshots: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        frame = snapshots.sort_values(["observed_at", "market_ticker"]).copy()
        frame["observed_at"] = pd.to_datetime(frame["observed_at"], utc=True)
        frame["expiry"] = pd.to_datetime(frame["expiry"], utc=True)
        frame = self._attach_volatility(frame, features)
        frame["volatility"] = frame["volatility"].fillna(0.5)

        tte_seconds = (frame["expiry"] - frame["observed_at"]).dt.total_seconds()
        frame["tte_years"] = np.maximum(tte_seconds.to_numpy(dtype=float), 0.0) / 31_536_000.0
        frame["prob_yes"] = self._vectorized_probability(frame)
        frame["prob_no"] = 1.0 - frame["prob_yes"]

        candidates = self._vectorized_signal_candidates(frame)
        if candidates.empty:
            return pd.DataFrame()

        if not self.backtest_config.allow_reentry:
            candidates = candidates.sort_values(["market_ticker", "observed_at"]).groupby("market_ticker", as_index=False, sort=False).first()

        exit_price_cents = np.where(
            candidates["side"].eq("yes"),
            self._vectorized_settlement_yes(candidates) * 100.0,
            (1.0 - self._vectorized_settlement_yes(candidates)) * 100.0,
        )
        exit_price_cents = np.rint(exit_price_cents).astype(int)
        entry_price_cents = candidates["entry_price_cents"].astype(int).to_numpy() + int(self.backtest_config.entry_slippage_cents)
        contracts = int(self.backtest_config.contracts_per_trade)
        entry_notional = entry_price_cents * contracts
        fee_cost = entry_notional * (float(self.backtest_config.fee_rate_bps) / 10_000.0)
        realized_pnl = (exit_price_cents - entry_price_cents) * contracts - fee_cost
        hold_minutes = np.maximum(
            (pd.to_datetime(candidates["expiry"], utc=True) - pd.to_datetime(candidates["observed_at"], utc=True)).dt.total_seconds().to_numpy() / 60.0,
            0.0,
        )

        return pd.DataFrame(
            {
                "market_ticker": candidates["market_ticker"].to_numpy(),
                "side": candidates["side"].to_numpy(),
                "entry_time": pd.to_datetime(candidates["observed_at"], utc=True).to_numpy(),
                "exit_time": pd.to_datetime(candidates["expiry"], utc=True).to_numpy(),
                "entry_price_cents": entry_price_cents,
                "exit_price_cents": exit_price_cents,
                "entry_notional": entry_notional.astype(float),
                "edge": candidates["edge"].astype(float).to_numpy(),
                "model_probability": candidates["model_probability"].astype(float).to_numpy(),
                "market_probability": candidates["market_probability"].astype(float).to_numpy(),
                "hold_minutes": hold_minutes.astype(float),
                "realized_pnl": realized_pnl.astype(float),
                "exit_trigger": "settlement",
            }
        )

    def _simulate_market(self, market_frame: pd.DataFrame, features: pd.DataFrame) -> list[dict[str, object]]:
        trades: list[dict[str, object]] = []
        has_entered = False
        for entry_index, row in market_frame.iterrows():
            if has_entered and not self.backtest_config.allow_reentry:
                break

            snapshot = self._snapshot_from_row(row)
            volatility = self._lookup_volatility(features, snapshot.observed_at)
            estimate = self.model.estimate(snapshot, volatility=volatility)
            signal = generate_signal(snapshot, estimate, self.signal_config)
            if signal.action == "no_action" or signal.side is None or signal.entry_price_cents is None or signal.edge is None:
                continue

            position = open_position(
                market_ticker=snapshot.market_ticker,
                side=signal.side,
                contracts=self.backtest_config.contracts_per_trade,
                entry_time=snapshot.observed_at,
                entry_price_cents=signal.entry_price_cents + self.backtest_config.entry_slippage_cents,
                expiry=snapshot.expiry,
            )
            trade = self._close_position(
                position=position,
                signal=signal,
                estimate=estimate,
                market_frame=market_frame,
                entry_index=entry_index,
            )
            trades.append(trade)
            has_entered = True
        return trades

    def _attach_volatility(self, snapshots: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        if features.empty or "realized_volatility" not in features.columns:
            result = snapshots.copy()
            result["volatility"] = 0.5
            return result
        feature_frame = features.copy()
        feature_frame = feature_frame.reset_index()
        time_column = feature_frame.columns[0]
        feature_frame = feature_frame.rename(columns={time_column: "observed_at"})
        feature_frame["observed_at"] = pd.to_datetime(feature_frame["observed_at"], utc=True)
        feature_frame = feature_frame.sort_values("observed_at")[["observed_at", "realized_volatility"]]
        merged = pd.merge_asof(
            snapshots.sort_values("observed_at"),
            feature_frame,
            on="observed_at",
            direction="backward",
        )
        merged = merged.rename(columns={"realized_volatility": "volatility"})
        return merged

    def _vectorized_probability(self, frame: pd.DataFrame) -> np.ndarray:
        contract_type = frame["contract_type"].fillna("threshold").astype(str).str.lower()
        volatility = np.maximum(frame["volatility"].astype(float).to_numpy(), 1e-12)
        tte_years = frame["tte_years"].astype(float).to_numpy()
        spot = np.maximum(frame["spot_price"].astype(float).to_numpy(), 1e-12)
        drift = float(getattr(self.model, "drift", 0.0))
        vol_floor = float(getattr(self.model, "volatility_floor", 0.05))
        volatility = np.maximum(volatility, vol_floor)

        def cdf(values: np.ndarray) -> np.ndarray:
            erf_vec = np.frompyfunc(lambda x: erf(float(x) / sqrt(2.0)), 1, 1)
            return 0.5 * (1.0 + np.asarray(erf_vec(values), dtype=float))

        def prob_above(target: np.ndarray) -> np.ndarray:
            target = np.maximum(target.astype(float), 1e-12)
            immediate = (spot >= target).astype(float)
            valid = tte_years > 0
            denom = volatility * np.sqrt(np.maximum(tte_years, 1e-18))
            numerator = np.log(spot / target) + (drift - 0.5 * volatility * volatility) * tte_years
            d = np.divide(numerator, denom, out=np.zeros_like(numerator, dtype=float), where=denom > 0)
            probs = cdf(d)
            return np.where(valid, np.clip(probs, 0.0, 1.0), immediate)

        threshold = frame["threshold"].astype(float).fillna(frame["spot_price"].astype(float)).to_numpy()
        range_low = frame["range_low"].astype(float).fillna(np.nan).to_numpy()
        range_high = frame["range_high"].astype(float).fillna(np.nan).to_numpy()
        direction = frame["direction"].fillna("").astype(str).str.lower()

        result = prob_above(threshold)

        is_threshold = contract_type.eq("threshold").to_numpy()
        is_range = contract_type.eq("range").to_numpy()
        is_direction = contract_type.eq("direction").to_numpy()

        threshold_prob = prob_above(threshold)
        below_mask = direction.isin(["below", "down"]).to_numpy()
        result = np.where(is_threshold & below_mask, 1.0 - threshold_prob, result)
        result = np.where(is_threshold & ~below_mask, threshold_prob, result)

        valid_range = is_range & ~np.isnan(range_low) & ~np.isnan(range_high)
        if valid_range.any():
            lower_prob = prob_above(np.where(np.isnan(range_low), spot, range_low))
            upper_prob = prob_above(np.where(np.isnan(range_high), spot, range_high))
            range_prob = np.clip(lower_prob - upper_prob, 0.0, 1.0)
            result = np.where(valid_range, range_prob, result)

        if is_direction.any():
            direction_prob = threshold_prob
            result = np.where(is_direction & below_mask, 1.0 - direction_prob, result)
            result = np.where(is_direction & ~below_mask, direction_prob, result)

        return np.clip(result, 0.0, 1.0)

    def _vectorized_signal_candidates(self, frame: pd.DataFrame) -> pd.DataFrame:
        yes_ask = frame["yes_ask"].astype(float)
        no_ask = frame["no_ask"].astype(float)
        yes_bid = frame["yes_bid"].astype(float)
        no_bid = frame["no_bid"].astype(float)
        prob_yes = frame["prob_yes"].astype(float)
        prob_no = frame["prob_no"].astype(float)

        near_money_mask = pd.Series(True, index=frame.index)
        if self.signal_config.max_near_money_bps is not None and "threshold" in frame.columns:
            threshold = frame["threshold"].astype(float)
            valid_threshold = threshold.notna() & frame["spot_price"].astype(float).gt(0)
            distance_bps = pd.Series(np.inf, index=frame.index, dtype=float)
            distance_bps.loc[valid_threshold] = (
                (threshold.loc[valid_threshold] - frame.loc[valid_threshold, "spot_price"].astype(float)).abs()
                / frame.loc[valid_threshold, "spot_price"].astype(float)
                * 10_000.0
            )
            near_money_mask = distance_bps.le(float(self.signal_config.max_near_money_bps))

        yes_entry_cents = np.rint(yes_ask.to_numpy(dtype=float) * 100.0)
        no_entry_cents = np.rint(no_ask.to_numpy(dtype=float) * 100.0)
        yes_spread = yes_entry_cents - np.rint(yes_bid.fillna(-1.0).to_numpy(dtype=float) * 100.0)
        no_spread = no_entry_cents - np.rint(no_bid.fillna(-1.0).to_numpy(dtype=float) * 100.0)

        yes_edge = prob_yes - yes_ask
        no_edge = prob_no - no_ask
        yes_fair = np.rint(prob_yes.to_numpy(dtype=float) * 100.0)
        no_fair = np.rint(prob_no.to_numpy(dtype=float) * 100.0)
        yes_expected = yes_fair - yes_entry_cents
        no_expected = no_fair - no_entry_cents

        base_mask = near_money_mask.to_numpy() & (frame["tte_years"].to_numpy(dtype=float) > 0)
        yes_mask = (
            base_mask
            & yes_ask.notna().to_numpy()
            & yes_bid.notna().to_numpy()
            & (yes_entry_cents >= int(self.signal_config.min_contract_price_cents))
            & (yes_entry_cents <= int(self.signal_config.max_contract_price_cents))
            & (yes_spread <= int(self.signal_config.max_spread_cents))
            & (yes_edge.to_numpy(dtype=float) >= float(self.signal_config.min_edge))
            & (prob_yes.to_numpy(dtype=float) >= float(self.signal_config.min_confidence))
            & (yes_expected > 0)
        )
        no_mask = (
            base_mask
            & no_ask.notna().to_numpy()
            & no_bid.notna().to_numpy()
            & (no_entry_cents >= int(self.signal_config.min_contract_price_cents))
            & (no_entry_cents <= int(self.signal_config.max_contract_price_cents))
            & (no_spread <= int(self.signal_config.max_spread_cents))
            & (no_edge.to_numpy(dtype=float) >= float(self.signal_config.min_edge))
            & (prob_no.to_numpy(dtype=float) >= float(self.signal_config.min_confidence))
            & (no_expected > 0)
        )

        base_columns = [
            "market_ticker",
            "observed_at",
            "expiry",
            "contract_type",
            "settlement_price",
            "threshold",
            "range_low",
            "range_high",
            "direction",
            "spot_price",
        ]
        yes_candidates = frame.loc[yes_mask, base_columns].copy()
        yes_candidates["side"] = "yes"
        yes_candidates["entry_price_cents"] = yes_entry_cents[yes_mask].astype(int)
        yes_candidates["edge"] = yes_edge.to_numpy(dtype=float)[yes_mask]
        yes_candidates["model_probability"] = prob_yes.to_numpy(dtype=float)[yes_mask]
        yes_candidates["market_probability"] = yes_ask.to_numpy(dtype=float)[yes_mask]
        yes_candidates["expected_value_cents"] = yes_expected[yes_mask].astype(float)

        no_candidates = frame.loc[no_mask, base_columns].copy()
        no_candidates["side"] = "no"
        no_candidates["entry_price_cents"] = no_entry_cents[no_mask].astype(int)
        no_candidates["edge"] = no_edge.to_numpy(dtype=float)[no_mask]
        no_candidates["model_probability"] = prob_no.to_numpy(dtype=float)[no_mask]
        no_candidates["market_probability"] = no_ask.to_numpy(dtype=float)[no_mask]
        no_candidates["expected_value_cents"] = no_expected[no_mask].astype(float)

        candidates = pd.concat([yes_candidates, no_candidates], ignore_index=True)
        if candidates.empty:
            return candidates

        candidates = candidates.sort_values(
            ["market_ticker", "observed_at", "edge", "expected_value_cents"],
            ascending=[True, True, False, False],
        )
        candidates = candidates.groupby(["market_ticker", "observed_at"], as_index=False, sort=False).first()
        return candidates

    @staticmethod
    def _vectorized_settlement_yes(frame: pd.DataFrame) -> np.ndarray:
        contract_type = frame["contract_type"].fillna("threshold").astype(str).str.lower()
        settlement_price = frame["settlement_price"].astype(float)
        threshold = frame["threshold"].astype(float)
        range_low = frame["range_low"].astype(float)
        range_high = frame["range_high"].astype(float)
        direction = frame["direction"].fillna("").astype(str).str.lower()
        spot_price = frame["spot_price"].astype(float)

        result = np.zeros(len(frame), dtype=float)

        threshold_mask = contract_type.eq("threshold")
        above_mask = ~direction.isin(["below", "down"])
        below_mask = direction.isin(["below", "down"])
        result = np.where(
            threshold_mask & threshold.notna().to_numpy() & settlement_price.notna().to_numpy() & above_mask.to_numpy(),
            (settlement_price.to_numpy(dtype=float) >= threshold.to_numpy(dtype=float)).astype(float),
            result,
        )
        result = np.where(
            threshold_mask & threshold.notna().to_numpy() & settlement_price.notna().to_numpy() & below_mask.to_numpy(),
            (settlement_price.to_numpy(dtype=float) < threshold.to_numpy(dtype=float)).astype(float),
            result,
        )

        range_mask = contract_type.eq("range")
        valid_range = range_mask & range_low.notna().to_numpy() & range_high.notna().to_numpy() & settlement_price.notna().to_numpy()
        result = np.where(
            valid_range,
            (
                (settlement_price.to_numpy(dtype=float) >= range_low.to_numpy(dtype=float))
                & (settlement_price.to_numpy(dtype=float) < range_high.to_numpy(dtype=float))
            ).astype(float),
            result,
        )

        direction_mask = contract_type.eq("direction")
        reference = threshold.fillna(spot_price).to_numpy(dtype=float)
        result = np.where(
            direction_mask & below_mask.to_numpy() & settlement_price.notna().to_numpy(),
            (settlement_price.to_numpy(dtype=float) < reference).astype(float),
            result,
        )
        result = np.where(
            direction_mask & above_mask.to_numpy() & settlement_price.notna().to_numpy(),
            (settlement_price.to_numpy(dtype=float) >= reference).astype(float),
            result,
        )

        return result

    def _close_position(
        self,
        *,
        position,
        signal,
        estimate: ProbabilityEstimate,
        market_frame: pd.DataFrame,
        entry_index: int,
    ) -> dict[str, object]:
        contracts = self.backtest_config.contracts_per_trade
        entry_notional = position.entry_price_cents * contracts
        fee_cost = entry_notional * (self.backtest_config.fee_rate_bps / 10_000.0)

        subsequent = market_frame.iloc[entry_index + 1 :]
        if self.backtest_config.strategy_mode == "trade_exit":
            for _, row in subsequent.iterrows():
                snapshot = self._snapshot_from_row(row)
                exit_decision = evaluate_exit(
                    snapshot,
                    side=position.side,
                    entry_price_cents=position.entry_price_cents,
                    fair_value_cents=signal.fair_value_cents or position.entry_price_cents,
                    contracts=contracts,
                    config=self.exit_config,
                )
                if exit_decision.action == "exit" and exit_decision.exit_price_cents is not None:
                    exit_price_cents = max(exit_decision.exit_price_cents - self.backtest_config.exit_slippage_cents, 0)
                    realized_pnl = (exit_price_cents - position.entry_price_cents) * contracts - fee_cost
                    hold_minutes = max((snapshot.observed_at - position.entry_time).total_seconds() / 60.0, 0.0)
                    return {
                        "market_ticker": position.market_ticker,
                        "side": position.side,
                        "entry_time": position.entry_time,
                        "exit_time": snapshot.observed_at,
                        "entry_price_cents": position.entry_price_cents,
                        "exit_price_cents": exit_price_cents,
                        "entry_notional": float(entry_notional),
                        "edge": float(signal.edge or 0.0),
                        "model_probability": float(signal.model_probability),
                        "market_probability": float(signal.market_probability or 0.0),
                        "hold_minutes": float(hold_minutes),
                        "realized_pnl": float(realized_pnl),
                        "exit_trigger": exit_decision.trigger,
                    }

        final_row = market_frame.iloc[-1]
        final_snapshot = self._snapshot_from_row(final_row)
        settlement_value = self._settlement_value(final_snapshot, position.side)
        exit_price_cents = int(round(settlement_value * 100.0))
        realized_pnl = (exit_price_cents - position.entry_price_cents) * contracts - fee_cost
        hold_minutes = max((final_snapshot.expiry - position.entry_time).total_seconds() / 60.0, 0.0)
        return {
            "market_ticker": position.market_ticker,
            "side": position.side,
            "entry_time": position.entry_time,
            "exit_time": final_snapshot.expiry,
            "entry_price_cents": position.entry_price_cents,
            "exit_price_cents": exit_price_cents,
            "entry_notional": float(entry_notional),
            "edge": float(signal.edge or 0.0),
            "model_probability": float(signal.model_probability),
            "market_probability": float(signal.market_probability or 0.0),
            "hold_minutes": float(hold_minutes),
            "realized_pnl": float(realized_pnl),
            "exit_trigger": "settlement",
        }

    @staticmethod
    def _lookup_volatility(features: pd.DataFrame, observed_at: pd.Timestamp) -> float:
        if features.empty or "realized_volatility" not in features.columns:
            return 0.5
        trimmed = features.loc[features.index <= observed_at]
        if trimmed.empty:
            return 0.5
        latest = trimmed["realized_volatility"].dropna()
        if latest.empty:
            return 0.5
        return float(latest.iloc[-1])

    @staticmethod
    def _snapshot_from_row(row: pd.Series) -> MarketSnapshot:
        return MarketSnapshot(**row.to_dict())

    @staticmethod
    def _settlement_value(snapshot: MarketSnapshot, side: str) -> float:
        if snapshot.contract_type == "threshold":
            if snapshot.threshold is None:
                return 0.0
            settled_yes = 1.0 if snapshot.settlement_price is not None and snapshot.settlement_price >= snapshot.threshold else 0.0
        elif snapshot.contract_type == "range":
            if snapshot.range_low is None or snapshot.range_high is None or snapshot.settlement_price is None:
                settled_yes = 0.0
            else:
                settled_yes = 1.0 if snapshot.range_low <= snapshot.settlement_price < snapshot.range_high else 0.0
        else:
            reference = snapshot.threshold if snapshot.threshold is not None else snapshot.spot_price
            if snapshot.direction == "down":
                settled_yes = 1.0 if snapshot.settlement_price is not None and snapshot.settlement_price < reference else 0.0
            else:
                settled_yes = 1.0 if snapshot.settlement_price is not None and snapshot.settlement_price >= reference else 0.0
        return settled_yes if side == "yes" else 1.0 - settled_yes

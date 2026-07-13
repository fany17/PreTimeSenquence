from __future__ import annotations

import math
import unittest

import numpy as np
import pandas as pd

from pretimesequence.v1.backtest import run_event_backtest
from pretimesequence.v1.spec import AmbiguousBarPolicy, CostSpec, StrategySpec
from pretimesequence.v1.splits import purged_chronological_split
from pretimesequence.v1.targets import _make_path_targets_reference, make_path_targets


def make_bars(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=len(rows), freq="min", tz="UTC"),
            "open": [row[0] for row in rows],
            "high": [row[1] for row in rows],
            "low": [row[2] for row in rows],
            "close": [row[3] for row in rows],
        }
    )


def make_spec(
    *,
    horizon_bars: int = 2,
    take_profit_return: float = 0.02,
    stop_loss_return: float = 0.01,
    costs: CostSpec | None = None,
    embargo_bars: int = 1,
    leverage_cap: float = 20.0,
) -> StrategySpec:
    return StrategySpec(
        horizon_bars=horizon_bars,
        max_holding_bars=horizon_bars,
        take_profit_return=take_profit_return,
        stop_loss_return=stop_loss_return,
        costs=costs or CostSpec(fee_rate=0.0, spread_rate=0.0, slippage_rate=0.0),
        ambiguous_bar_policy=AmbiguousBarPolicy.CONSERVATIVE,
        embargo_bars=embargo_bars,
        leverage_cap=leverage_cap,
    )


class StrategySpecTests(unittest.TestCase):
    def test_default_contract_uses_next_bar_and_15_minute_horizon(self) -> None:
        spec = StrategySpec()

        self.assertEqual(spec.base_interval_minutes, 1)
        self.assertEqual(spec.entry_delay_bars, 1)
        self.assertEqual(spec.horizon_bars, 15)
        self.assertEqual(spec.max_holding_bars, 15)
        self.assertEqual(spec.max_concurrent_positions, 1)

    def test_invalid_contract_fails_fast(self) -> None:
        with self.assertRaisesRegex(ValueError, "entry_delay_bars"):
            StrategySpec(entry_delay_bars=0)
        with self.assertRaisesRegex(ValueError, "max_holding_bars"):
            StrategySpec(horizon_bars=15, max_holding_bars=16)
        with self.assertRaisesRegex(ValueError, "fee_rate"):
            CostSpec(fee_rate=-0.001)
        with self.assertRaisesRegex(ValueError, "fee_rate"):
            CostSpec(fee_rate=float("nan"))

    def test_mapping_round_trip_preserves_the_validated_contract(self) -> None:
        original = make_spec(
            horizon_bars=5,
            costs=CostSpec(
                fee_rate=0.001,
                spread_rate=0.002,
                slippage_rate=0.0005,
                funding_rate_per_bar=0.0001,
            ),
            embargo_bars=3,
        )

        restored = StrategySpec.from_mapping(original.to_mapping())

        self.assertEqual(restored, original)


class PathTargetTests(unittest.TestCase):
    def test_entry_is_next_bar_open_and_path_metrics_use_future_window(self) -> None:
        bars = make_bars(
            [
                (10.0, 55.0, 9.0, 50.0),
                (100.0, 105.0, 95.0, 101.0),
                (101.0, 110.0, 90.0, 105.0),
            ]
        )
        spec = make_spec(horizon_bars=2, take_profit_return=0.50, stop_loss_return=0.50)

        targets = make_path_targets(bars, spec)
        first = targets.iloc[0]

        self.assertEqual(len(targets), 1)
        self.assertEqual(first["signal_index"], 0)
        self.assertEqual(first["entry_index"], 1)
        self.assertEqual(first["entry_price"], 100.0)
        self.assertAlmostEqual(first["return_h"], math.log(105.0 / 100.0))
        self.assertAlmostEqual(first["mfe_h"], 0.10)
        self.assertAlmostEqual(first["mae_h"], -0.10)
        self.assertEqual(first["long_exit_reason"], "horizon")
        self.assertEqual(first["long_holding_bars"], 2)

    def test_ambiguous_bar_is_flagged_and_conservatively_stopped(self) -> None:
        bars = make_bars(
            [
                (100.0, 100.0, 100.0, 100.0),
                (100.0, 103.0, 97.0, 100.0),
                (100.0, 100.0, 100.0, 100.0),
            ]
        )

        first = make_path_targets(bars, make_spec()).iloc[0]

        self.assertTrue(first["ambiguous_bar"])
        self.assertTrue(first["long_ambiguous"])
        self.assertTrue(first["short_ambiguous"])
        self.assertEqual(first["long_exit_reason"], "stop_loss")
        self.assertEqual(first["short_exit_reason"], "stop_loss")
        self.assertAlmostEqual(first["long_net_return"], -0.01)
        self.assertAlmostEqual(first["short_net_return"], -0.01)

    def test_target_values_do_not_depend_on_leverage_cap(self) -> None:
        bars = make_bars(
            [
                (100.0, 100.0, 100.0, 100.0),
                (100.0, 101.0, 100.0, 100.5),
                (100.5, 101.0, 100.0, 100.5),
            ]
        )

        low_leverage = make_path_targets(bars, make_spec(leverage_cap=1.0)).iloc[0]
        high_leverage = make_path_targets(bars, make_spec(leverage_cap=20.0)).iloc[0]

        self.assertAlmostEqual(low_leverage["long_net_return"], high_leverage["long_net_return"])
        self.assertAlmostEqual(low_leverage["short_net_return"], high_leverage["short_net_return"])

    def test_unsorted_or_duplicate_timestamps_are_rejected(self) -> None:
        bars = make_bars(
            [
                (100.0, 100.0, 100.0, 100.0),
                (100.0, 101.0, 99.0, 100.0),
                (100.0, 101.0, 99.0, 100.0),
            ]
        )
        bars.loc[2, "timestamp"] = bars.loc[1, "timestamp"]

        with self.assertRaisesRegex(ValueError, "timestamp"):
            make_path_targets(bars, make_spec())

    def test_naive_timestamps_and_cadence_gaps_are_rejected(self) -> None:
        bars = make_bars(
            [
                (100.0, 100.0, 100.0, 100.0),
                (100.0, 101.0, 99.0, 100.0),
                (100.0, 101.0, 99.0, 100.0),
            ]
        )
        naive = bars.copy()
        naive["timestamp"] = naive["timestamp"].dt.tz_localize(None)
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            make_path_targets(naive, make_spec())

        with_gap = bars.copy()
        with_gap.loc[2, "timestamp"] += pd.Timedelta(minutes=1)
        with self.assertRaisesRegex(ValueError, "cadence"):
            make_path_targets(with_gap, make_spec())

    def test_funding_payment_is_applied_only_on_the_explicit_payment_bar(self) -> None:
        bars = make_bars(
            [
                (100.0, 100.0, 100.0, 100.0),
                (100.0, 100.0, 100.0, 100.0),
            ]
        )
        bars["funding_payment_rate"] = [0.0, 0.001]
        spec = make_spec(
            horizon_bars=1,
            take_profit_return=0.50,
            stop_loss_return=0.50,
        )

        target = make_path_targets(bars, spec).iloc[0]

        self.assertAlmostEqual(target["long_net_return"], -0.001)
        self.assertAlmostEqual(target["short_net_return"], 0.001)

    def test_gap_open_exit_prices_are_conservative_and_explicit(self) -> None:
        gap_down = make_bars(
            [
                (100.0, 100.0, 100.0, 100.0),
                (100.0, 100.5, 99.5, 100.0),
                (95.0, 96.0, 94.0, 95.0),
            ]
        )
        gap_up = gap_down.copy()
        gap_up.loc[2, ["open", "high", "low", "close"]] = [105.0, 106.0, 104.0, 105.0]
        spec = make_spec(horizon_bars=2, take_profit_return=0.02, stop_loss_return=0.01)

        down_target = make_path_targets(gap_down, spec).iloc[0]
        up_target = make_path_targets(gap_up, spec).iloc[0]

        self.assertEqual(down_target["long_exit_reason"], "stop_loss")
        self.assertEqual(down_target["long_exit_price"], 95.0)
        self.assertEqual(up_target["long_exit_reason"], "take_profit")
        self.assertEqual(up_target["long_exit_price"], 102.0)

    def test_time_exit_uses_max_holding_close_but_path_metrics_use_full_horizon(self) -> None:
        bars = make_bars(
            [
                (100.0, 100.0, 100.0, 100.0),
                (100.0, 101.0, 99.0, 101.0),
                (101.0, 102.0, 100.0, 102.0),
                (102.0, 111.0, 101.0, 110.0),
            ]
        )
        spec = StrategySpec(
            horizon_bars=3,
            max_holding_bars=2,
            take_profit_return=0.50,
            stop_loss_return=0.50,
            costs=CostSpec(fee_rate=0.0, spread_rate=0.0, slippage_rate=0.0),
        )

        target = make_path_targets(bars, spec).iloc[0]

        self.assertEqual(target["long_exit_reason"], "time_exit")
        self.assertEqual(target["long_holding_bars"], 2)
        self.assertEqual(target["long_exit_price"], 102.0)
        self.assertAlmostEqual(target["return_h"], math.log(110.0 / 100.0))
        self.assertAlmostEqual(target["mfe_h"], 0.11)

    def test_ambiguous_exclude_and_early_exit_funding_are_preserved(self) -> None:
        ambiguous = make_bars(
            [
                (100.0, 100.0, 100.0, 100.0),
                (100.0, 103.0, 97.0, 100.0),
                (100.0, 100.0, 100.0, 100.0),
            ]
        )
        exclude_spec = StrategySpec(
            horizon_bars=2,
            max_holding_bars=2,
            take_profit_return=0.02,
            stop_loss_return=0.01,
            costs=CostSpec(fee_rate=0.0, spread_rate=0.0, slippage_rate=0.0),
            ambiguous_bar_policy=AmbiguousBarPolicy.EXCLUDE,
        )
        excluded = make_path_targets(ambiguous, exclude_spec).iloc[0]
        self.assertEqual(excluded["long_exit_reason"], "ambiguous_excluded")
        self.assertTrue(excluded["long_excluded"])
        self.assertTrue(pd.isna(excluded["long_net_return"]))

        early_exit = make_bars(
            [
                (100.0, 100.0, 100.0, 100.0),
                (100.0, 102.0, 99.5, 101.0),
                (101.0, 101.0, 100.0, 101.0),
            ]
        )
        early_exit["funding_payment_rate"] = [0.0, 0.001, 0.005]
        funding_target = make_path_targets(
            early_exit,
            make_spec(horizon_bars=2, take_profit_return=0.01, stop_loss_return=0.50),
        ).iloc[0]
        self.assertEqual(funding_target["long_holding_bars"], 1)
        self.assertAlmostEqual(funding_target["long_funding_return"], 0.001)
        self.assertAlmostEqual(funding_target["long_net_return"], 0.009)

    def test_vectorized_targets_match_reference_across_costs_and_policies(self) -> None:
        rng = np.random.default_rng(20260714)
        n_rows = 180
        base = 100.0 + np.cumsum(rng.normal(0.0, 0.18, size=n_rows))
        open_ = base + rng.normal(0.0, 0.04, size=n_rows)
        close = base + rng.normal(0.0, 0.04, size=n_rows)
        high = np.maximum(open_, close) + rng.uniform(0.02, 0.35, size=n_rows)
        low = np.minimum(open_, close) - rng.uniform(0.02, 0.35, size=n_rows)
        bars = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-04-01", periods=n_rows, freq="min", tz="UTC"),
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "funding_payment_rate": np.where(np.arange(n_rows) % 17 == 0, 0.0001, 0.0),
            }
        )

        for policy in (AmbiguousBarPolicy.CONSERVATIVE, AmbiguousBarPolicy.EXCLUDE):
            spec = StrategySpec(
                entry_delay_bars=2,
                horizon_bars=7,
                max_holding_bars=5,
                take_profit_return=0.004,
                stop_loss_return=0.003,
                costs=CostSpec(fee_rate=0.0005, spread_rate=0.0002, slippage_rate=0.0001),
                ambiguous_bar_policy=policy,
            )
            expected = _make_path_targets_reference(bars, spec)
            actual = make_path_targets(bars, spec)

            pd.testing.assert_frame_equal(
                actual,
                expected,
                check_dtype=True,
                check_exact=False,
                rtol=1e-12,
                atol=1e-12,
            )

    def test_vectorized_target_dtypes_match_reference_for_all_hit_and_no_hit(self) -> None:
        all_hit = make_bars([(100.0, 103.0, 97.0, 100.0)] * 12)
        no_hit = make_bars([(100.0, 100.1, 99.9, 100.0)] * 12)
        spec = make_spec(horizon_bars=2, take_profit_return=0.02, stop_loss_return=0.01)

        for bars in (all_hit, no_hit):
            pd.testing.assert_frame_equal(
                make_path_targets(bars, spec),
                _make_path_targets_reference(bars, spec),
                check_dtype=True,
                check_exact=False,
                rtol=1e-12,
                atol=1e-12,
            )


class PurgedSplitTests(unittest.TestCase):
    def test_purge_and_embargo_remove_overlapping_boundary_rows(self) -> None:
        spec = make_spec(horizon_bars=5, embargo_bars=2)

        split = purged_chronological_split(
            n_samples=100,
            spec=spec,
            train_ratio=0.70,
            validation_ratio=0.15,
        )

        self.assertEqual(split.train_indices[0], 0)
        self.assertEqual(split.train_indices[-1], 64)
        self.assertEqual(split.validation_indices[0], 72)
        self.assertEqual(split.validation_indices[-1], 79)
        self.assertEqual(split.test_indices[0], 87)
        self.assertEqual(split.test_indices[-1], 99)
        self.assertLess(split.train_indices[-1] + spec.horizon_bars, split.validation_indices[0])
        self.assertLess(split.validation_indices[-1] + spec.horizon_bars, split.test_indices[0])

    def test_too_small_dataset_fails_instead_of_returning_empty_fold(self) -> None:
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            purged_chronological_split(
                n_samples=20,
                spec=make_spec(horizon_bars=10, embargo_bars=2),
            )


class EventBacktestTests(unittest.TestCase):
    def test_default_notional_is_derived_from_risk_budget_and_stop_distance(self) -> None:
        bars = make_bars(
            [
                (100.0, 100.0, 100.0, 100.0),
                (100.0, 102.0, 98.0, 100.0),
            ]
        )
        decisions = pd.DataFrame({"signal_index": [0], "action": ["long"]})
        spec = make_spec(
            horizon_bars=1,
            take_profit_return=0.50,
            stop_loss_return=0.01,
        )

        trade = run_event_backtest(bars, decisions, spec).trades.iloc[0]

        self.assertAlmostEqual(trade["notional"], spec.risk_fraction_per_trade / spec.stop_loss_return)

    def test_tp_exit_restores_flat_and_allows_a_later_signal(self) -> None:
        bars = make_bars(
            [
                (100.0, 100.0, 100.0, 100.0),
                (100.0, 102.0, 100.0, 101.0),
                (100.0, 102.0, 100.0, 101.0),
                (101.0, 101.0, 101.0, 101.0),
            ]
        )
        decisions = pd.DataFrame({"signal_index": [0, 1], "action": ["long", "long"]})

        result = run_event_backtest(
            bars,
            decisions,
            make_spec(take_profit_return=0.01, stop_loss_return=0.01),
        )

        self.assertEqual(result.trades["entry_index"].tolist(), [1, 2])
        self.assertEqual(result.trades["exit_reason"].tolist(), ["take_profit", "take_profit"])
        self.assertEqual(result.equity_curve.iloc[-1]["state"], "flat")

    def test_ambiguous_bar_uses_conservative_stop_loss(self) -> None:
        bars = make_bars(
            [
                (100.0, 100.0, 100.0, 100.0),
                (100.0, 102.0, 98.0, 100.0),
                (100.0, 100.0, 100.0, 100.0),
            ]
        )
        decisions = pd.DataFrame({"signal_index": [0], "action": ["long"]})

        trade = run_event_backtest(
            bars,
            decisions,
            make_spec(take_profit_return=0.01, stop_loss_return=0.01),
        ).trades.iloc[0]

        self.assertTrue(trade["ambiguous_bar"])
        self.assertEqual(trade["exit_reason"], "stop_loss")
        self.assertAlmostEqual(trade["net_return"], -0.01)

    def test_maximum_holding_time_exits_at_the_expected_close(self) -> None:
        bars = make_bars(
            [
                (100.0, 100.0, 100.0, 100.0),
                (100.0, 100.5, 99.5, 100.0),
                (100.0, 100.5, 99.5, 100.2),
                (100.2, 100.5, 100.0, 100.2),
            ]
        )
        decisions = pd.DataFrame({"signal_index": [0], "action": ["long"]})

        trade = run_event_backtest(
            bars,
            decisions,
            make_spec(horizon_bars=2, take_profit_return=0.10, stop_loss_return=0.10),
        ).trades.iloc[0]

        self.assertEqual(trade["entry_index"], 1)
        self.assertEqual(trade["exit_index"], 2)
        self.assertEqual(trade["holding_bars"], 2)
        self.assertEqual(trade["exit_reason"], "time_exit")

    def test_fold_end_forces_close_without_using_later_bars(self) -> None:
        bars = make_bars(
            [
                (100.0, 100.0, 100.0, 100.0),
                (100.0, 100.5, 99.5, 100.0),
                (100.0, 100.5, 99.5, 100.2),
                (150.0, 160.0, 140.0, 155.0),
            ]
        )
        decisions = pd.DataFrame({"signal_index": [0], "action": ["long"]})
        spec = make_spec(horizon_bars=3, take_profit_return=0.50, stop_loss_return=0.50)

        trade = run_event_backtest(bars, decisions, spec, fold_end_index=2).trades.iloc[0]

        self.assertEqual(trade["exit_index"], 2)
        self.assertEqual(trade["exit_reason"], "fold_end")
        self.assertEqual(trade["exit_raw_price"], 100.2)

    def test_flat_price_trade_loses_explicit_execution_costs(self) -> None:
        bars = make_bars(
            [
                (100.0, 100.0, 100.0, 100.0),
                (100.0, 100.0, 100.0, 100.0),
            ]
        )
        decisions = pd.DataFrame({"signal_index": [0], "action": ["long"]})
        costs = CostSpec(fee_rate=0.001, spread_rate=0.002, slippage_rate=0.001)
        spec = make_spec(
            horizon_bars=1,
            take_profit_return=0.50,
            stop_loss_return=0.50,
            costs=costs,
        )

        trade = run_event_backtest(bars, decisions, spec).trades.iloc[0]

        self.assertEqual(trade["exit_reason"], "time_exit")
        self.assertLess(trade["net_return"], -0.005)
        self.assertGreater(trade["fees"], 0.0)

    def test_target_and_backtest_share_cost_and_funding_semantics(self) -> None:
        bars = make_bars(
            [
                (100.0, 100.0, 100.0, 100.0),
                (100.0, 100.0, 100.0, 100.0),
            ]
        )
        costs = CostSpec(
            fee_rate=0.001,
            spread_rate=0.002,
            slippage_rate=0.001,
            funding_rate_per_bar=0.0005,
        )
        spec = make_spec(
            horizon_bars=1,
            take_profit_return=0.50,
            stop_loss_return=0.50,
            costs=costs,
        )
        target = make_path_targets(bars, spec).iloc[0]
        decisions = pd.DataFrame({"signal_index": [0], "action": ["long"]})

        trade = run_event_backtest(bars, decisions, spec).trades.iloc[0]

        self.assertAlmostEqual(trade["net_return"], target["long_net_return"])
        self.assertGreater(target["short_net_return"], target["long_net_return"])


if __name__ == "__main__":
    unittest.main()

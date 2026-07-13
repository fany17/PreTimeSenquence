from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from pretimesequence.features import FEATURE_COLUMNS, add_features
from pretimesequence.v1.dataset import SupervisedDataset, build_supervised_dataset
from pretimesequence.v1.spec import AmbiguousBarPolicy, StrategySpec
from pretimesequence.v1.splits import inner_walk_forward_folds, outer_walk_forward_folds


def make_market_bars(n_rows: int = 145) -> pd.DataFrame:
    base = 100.0 + np.arange(n_rows, dtype=float) * 0.03
    open_ = base
    close = base + 0.02
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=n_rows, freq="min", tz="UTC"),
            "open": open_,
            "high": np.maximum(open_, close) + 0.05,
            "low": np.minimum(open_, close) - 0.05,
            "close": close,
            "volume": 1_000.0 + np.arange(n_rows, dtype=float) ** 1.1,
        },
        index=pd.Index(10_000 + np.arange(n_rows) * 7, name="arbitrary_source_index"),
    )


def make_meta(n_rows: int = 80, horizon: int = 5) -> pd.DataFrame:
    signal_index = np.arange(n_rows, dtype=int)
    return pd.DataFrame(
        {
            "signal_index": signal_index,
            "signal_time": pd.date_range("2026-02-01", periods=n_rows, freq="min", tz="UTC"),
            "entry_index": signal_index + 1,
            "horizon_end_index": signal_index + horizon,
            "feature_valid": True,
            "target_valid": True,
        },
        index=pd.Index(signal_index, name="signal_index"),
    )


class SupervisedDatasetTests(unittest.TestCase):
    def test_non_range_index_is_aligned_by_global_signal_position_without_imputation(self) -> None:
        bars = make_market_bars()

        dataset = build_supervised_dataset(bars, StrategySpec())

        self.assertIsInstance(dataset, SupervisedDataset)
        expected_index = pd.Index(range(len(bars) - 15), name="signal_index")
        self.assertTrue(dataset.X.index.equals(expected_index))
        self.assertTrue(dataset.targets.index.equals(expected_index))
        self.assertTrue(dataset.sample_meta.index.equals(expected_index))
        self.assertTrue(dataset.X.loc[0].isna().any())
        self.assertFalse(bool(dataset.sample_meta.loc[0, "feature_valid"]))

        expected_features = add_features(bars.reset_index(drop=True)).loc[:, FEATURE_COLUMNS].iloc[125]
        pd.testing.assert_series_equal(
            dataset.X.loc[125],
            expected_features,
            check_names=False,
        )
        self.assertTrue(bool(dataset.sample_meta.loc[125, "feature_valid"]))

    def test_future_bar_perturbation_does_not_change_signal_time_features(self) -> None:
        bars = make_market_bars()
        signal_index = 125
        changed = bars.copy()
        future_mask = np.arange(len(changed)) > signal_index
        for column in ("open", "high", "low", "close"):
            changed.loc[future_mask, column] = changed.loc[future_mask, column] * 1.7
        changed.loc[future_mask, "volume"] = changed.loc[future_mask, "volume"] * 9.0

        original_dataset = build_supervised_dataset(bars, StrategySpec())
        changed_dataset = build_supervised_dataset(changed, StrategySpec())

        pd.testing.assert_series_equal(
            original_dataset.X.loc[signal_index],
            changed_dataset.X.loc[signal_index],
        )

    def test_ambiguous_excluded_target_is_retained_and_marked_invalid(self) -> None:
        bars = make_market_bars()
        entry_label = bars.index[1]
        bars.loc[entry_label, "high"] = bars.loc[entry_label, "open"] * 1.05
        bars.loc[entry_label, "low"] = bars.loc[entry_label, "open"] * 0.95
        spec = StrategySpec(ambiguous_bar_policy=AmbiguousBarPolicy.EXCLUDE)

        dataset = build_supervised_dataset(bars, spec)

        self.assertIn(0, dataset.targets.index)
        self.assertTrue(bool(dataset.targets.loc[0, "ambiguous_bar"]))
        self.assertFalse(bool(dataset.sample_meta.loc[0, "target_valid"]))
        self.assertTrue(pd.isna(dataset.targets.loc[0, "long_net_return"]))

    def test_missing_or_invalid_volume_fails_fast(self) -> None:
        bars = make_market_bars()
        with self.assertRaisesRegex(ValueError, "volume"):
            build_supervised_dataset(bars.drop(columns="volume"), StrategySpec())

        invalid = bars.copy()
        invalid.loc[invalid.index[3], "volume"] = -1.0
        with self.assertRaisesRegex(ValueError, "volume"):
            build_supervised_dataset(invalid, StrategySpec())


class IntervalAwareSplitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = StrategySpec(
            horizon_bars=5,
            max_holding_bars=5,
            embargo_bars=2,
        )

    def test_outer_and_inner_folds_use_real_target_intervals(self) -> None:
        meta = make_meta()

        outer_folds = outer_walk_forward_folds(
            meta,
            self.spec,
            min_development_bars=20,
            test_bars=15,
            step_bars=15,
            max_folds=2,
        )

        self.assertEqual(len(outer_folds), 2)
        for fold in outer_folds:
            development = meta.loc[fold.development_indices]
            test = meta.loc[fold.test_indices]
            self.assertLess(
                int(development["horizon_end_index"].max()),
                int(test["signal_index"].min()),
            )
            self.assertTrue((development["horizon_end_index"] < fold.boundary_signal_index).all())
            self.assertGreaterEqual(
                int(test["signal_index"].min()),
                fold.boundary_signal_index + self.spec.embargo_bars,
            )
            self.assertTrue((test["horizon_end_index"] <= fold.test_end_bar_index).all())

        inner_folds = inner_walk_forward_folds(
            meta,
            outer_folds[1].development_indices,
            self.spec,
            min_train_bars=10,
            validation_bars=10,
            step_bars=10,
            max_folds=1,
        )
        self.assertEqual(len(inner_folds), 1)
        inner = inner_folds[0]
        train = meta.loc[inner.train_indices]
        validation = meta.loc[inner.validation_indices]
        self.assertLess(
            int(train["horizon_end_index"].max()),
            int(validation["signal_index"].min()),
        )
        self.assertTrue((train["horizon_end_index"] < inner.boundary_signal_index).all())
        self.assertTrue((validation["horizon_end_index"] <= inner.validation_end_bar_index).all())

    def test_invalid_rows_are_filtered_after_global_boundaries_are_fixed(self) -> None:
        clean_meta = make_meta()
        invalid_meta = clean_meta.copy()
        invalid_meta.loc[5, "feature_valid"] = False
        invalid_meta.loc[25, "target_valid"] = False

        clean_fold = outer_walk_forward_folds(
            clean_meta,
            self.spec,
            min_development_bars=20,
            test_bars=15,
            max_folds=1,
        )[0]
        invalid_fold = outer_walk_forward_folds(
            invalid_meta,
            self.spec,
            min_development_bars=20,
            test_bars=15,
            max_folds=1,
        )[0]

        self.assertEqual(invalid_fold.boundary_signal_index, clean_fold.boundary_signal_index)
        self.assertEqual(invalid_fold.test_start_signal_index, clean_fold.test_start_signal_index)
        self.assertNotIn(5, invalid_fold.development_indices)
        self.assertNotIn(25, invalid_fold.test_indices)
        self.assertLess(
            int(invalid_meta.loc[invalid_fold.development_indices, "horizon_end_index"].max()),
            int(invalid_meta.loc[invalid_fold.test_indices, "signal_index"].min()),
        )

    def test_insufficient_samples_fail_fast(self) -> None:
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            outer_walk_forward_folds(
                make_meta(n_rows=10),
                self.spec,
                min_development_bars=20,
                test_bars=15,
            )

        meta = make_meta(n_rows=30)
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            inner_walk_forward_folds(
                meta,
                np.arange(0, 10, dtype=int),
                self.spec,
                min_train_bars=20,
                validation_bars=15,
            )


if __name__ == "__main__":
    unittest.main()

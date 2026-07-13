from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from pretimesequence.v1.dataset import build_supervised_dataset
from pretimesequence.v1.policy import decisions_from_action_values
from pretimesequence.v1.spec import StrategySpec
from pretimesequence.v1.training import fit_ridge_action_value, predict_action_values
from pretimesequence.v1.walk_forward import (
    AllCandidatesFailedError,
    NestedWalkForwardConfig,
    run_nested_walk_forward,
)


def make_research_bars(n_rows: int = 260) -> pd.DataFrame:
    position = np.arange(n_rows, dtype=float)
    base = 100.0 + 0.025 * position + 0.25 * np.sin(position / 11.0)
    open_ = base + 0.015 * np.sin(position / 5.0)
    close = base + 0.02 * np.cos(position / 7.0)
    high = np.maximum(open_, close) + 0.08 + 0.01 * np.sin(position / 3.0) ** 2
    low = np.minimum(open_, close) - 0.08 - 0.01 * np.cos(position / 3.0) ** 2
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-03-01", periods=n_rows, freq="min", tz="UTC"),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1_500.0 + 2.0 * position + 30.0 * np.sin(position / 13.0) ** 2,
        },
        index=pd.Index(50_000 + np.arange(n_rows) * 3, name="source_index"),
    )


def make_walk_config() -> NestedWalkForwardConfig:
    return NestedWalkForwardConfig(
        min_development_bars=200,
        test_bars=35,
        outer_step_bars=35,
        max_outer_folds=1,
        inner_min_train_bars=30,
        inner_validation_bars=20,
        inner_step_bars=20,
        max_inner_folds=1,
        alphas=(0.1, 1.0),
        thresholds=(0.0, 0.001),
    )


class PolicyAndTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bars = make_research_bars()
        cls.spec = StrategySpec()
        cls.dataset = build_supervised_dataset(cls.bars, cls.spec)

    def test_policy_uses_strict_threshold_and_ties_are_flat(self) -> None:
        predictions = pd.DataFrame(
            {
                "signal_index": [10, 11, 12, 13, 14],
                "pred_long_net_return": [0.02, 0.01, 0.03, -0.01, 0.01],
                "pred_short_net_return": [0.01, 0.02, 0.03, -0.02, 0.00],
            }
        )

        decisions = decisions_from_action_values(predictions, threshold=0.01)

        self.assertEqual(decisions["action"].tolist(), ["long", "short", "flat", "flat", "flat"])
        self.assertEqual(decisions["signal_index"].tolist(), predictions["signal_index"].tolist())

    def test_ridge_scaler_fits_train_only_and_prediction_schema_is_strict(self) -> None:
        train_indices = np.arange(120, 140, dtype=int)
        model = fit_ridge_action_value(
            self.dataset.X,
            self.dataset.targets,
            train_indices,
            alpha=0.1,
        )

        np.testing.assert_allclose(
            model.scaler.mean_,
            self.dataset.X.loc[train_indices].mean(axis=0).to_numpy(dtype=float),
        )
        self.assertEqual(model.train_indices, tuple(train_indices))

        reordered = self.dataset.X.loc[[150], list(reversed(self.dataset.feature_names))]
        with self.assertRaisesRegex(ValueError, "schema"):
            predict_action_values(model, reordered)

        with self.assertRaisesRegex(ValueError, "sorted"):
            fit_ridge_action_value(
                self.dataset.X,
                self.dataset.targets,
                np.array([121, 120], dtype=int),
                alpha=0.1,
            )


class NestedWalkForwardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bars = make_research_bars()
        cls.spec = StrategySpec()
        cls.dataset = build_supervised_dataset(cls.bars, cls.spec)
        cls.config = make_walk_config()

    def test_oof_is_chronological_outer_predictions_are_unique_and_horizons_fit(self) -> None:
        result = run_nested_walk_forward(
            self.bars,
            self.spec,
            self.config,
            dataset=self.dataset,
        )

        self.assertFalse(result.inner_oof_predictions.empty)
        for row in result.inner_oof_predictions.itertuples(index=False):
            self.assertLessEqual(row.train_start_signal_index, row.train_end_signal_index)
            self.assertGreater(row.train_samples, 0)
            self.assertLess(row.train_end_signal_index, row.signal_index)
            self.assertLess(row.train_horizon_end_max, row.signal_index)

        self.assertFalse(result.outer_predictions.duplicated(["outer_fold", "signal_index"]).any())
        self.assertTrue(
            (result.outer_predictions["horizon_end_index"] <= result.outer_predictions["test_end_bar_index"]).all()
        )
        self.assertEqual(len(result.fold_metrics), 1)
        required_metrics = {
            "long_mae",
            "short_mae",
            "long_rmse",
            "short_rmse",
            "trades",
            "net_return",
            "max_drawdown",
            "expectancy",
            "chosen_alpha",
            "chosen_threshold",
            "always_flat_net_return",
        }
        self.assertTrue(required_metrics.issubset(result.fold_metrics.columns))
        self.assertEqual(float(result.fold_metrics.iloc[0]["always_flat_net_return"]), 0.0)

    def test_outer_price_and_label_perturbation_does_not_change_selection(self) -> None:
        baseline = run_nested_walk_forward(
            self.bars,
            self.spec,
            self.config,
            dataset=self.dataset,
        )
        changed_bars = self.bars.copy()
        future_mask = np.arange(len(changed_bars)) >= 202
        for column in ("open", "high", "low", "close"):
            changed_bars.loc[future_mask, column] = changed_bars.loc[future_mask, column] * 1.25
        changed_bars.loc[future_mask, "volume"] = changed_bars.loc[future_mask, "volume"] * 4.0
        changed_dataset = build_supervised_dataset(changed_bars, self.spec)

        changed = run_nested_walk_forward(
            changed_bars,
            self.spec,
            self.config,
            dataset=changed_dataset,
        )

        baseline_choice = baseline.fold_metrics.loc[0, ["chosen_alpha", "chosen_threshold"]].tolist()
        changed_choice = changed.fold_metrics.loc[0, ["chosen_alpha", "chosen_threshold"]].tolist()
        self.assertEqual(baseline_choice, changed_choice)

    def test_prebuilt_dataset_must_match_the_exact_supplied_bars(self) -> None:
        changed_bars = self.bars.copy()
        changed_bars.loc[changed_bars.index[150], "volume"] *= 2.0

        with self.assertRaisesRegex(ValueError, "fingerprint"):
            run_nested_walk_forward(
                changed_bars,
                self.spec,
                self.config,
                dataset=self.dataset,
            )

    def test_outer_backtest_is_called_once_for_the_frozen_selection(self) -> None:
        from pretimesequence.v1.backtest import run_event_backtest as real_backtest

        calls: list[tuple[int, int]] = []

        def recording_backtest(*args, **kwargs):
            calls.append((kwargs["fold_start_index"], kwargs["fold_end_index"]))
            return real_backtest(*args, **kwargs)

        with patch("pretimesequence.v1.walk_forward.run_event_backtest", side_effect=recording_backtest):
            result = run_nested_walk_forward(
                self.bars,
                self.spec,
                self.config,
                dataset=self.dataset,
            )

        outer_bounds = (
            int(result.fold_metrics.iloc[0]["test_start_signal_index"]),
            int(result.fold_metrics.iloc[0]["test_end_bar_index"]),
        )
        self.assertEqual(calls.count(outer_bounds), 1)

    def test_all_candidates_and_partial_failure_reasons_are_retained(self) -> None:
        from pretimesequence.v1.training import fit_ridge_action_value as real_fit

        def selective_fit(*args, **kwargs):
            if kwargs["alpha"] == 1.0:
                raise RuntimeError("synthetic candidate failure")
            return real_fit(*args, **kwargs)

        with patch("pretimesequence.v1.walk_forward.fit_ridge_action_value", side_effect=selective_fit):
            result = run_nested_walk_forward(
                self.bars,
                self.spec,
                self.config,
                dataset=self.dataset,
            )

        self.assertEqual(len(result.candidate_metrics), 4)
        failed = result.candidate_metrics[result.candidate_metrics["status"] == "failed"]
        successful = result.candidate_metrics[result.candidate_metrics["status"] == "ok"]
        self.assertEqual(len(failed), 2)
        self.assertEqual(len(successful), 2)
        self.assertTrue(failed["failure_reason"].str.contains("synthetic candidate failure").all())
        self.assertTrue((result.fold_metrics["chosen_alpha"] == 0.1).all())

    def test_all_candidates_failed_raises_auditable_error(self) -> None:
        with patch(
            "pretimesequence.v1.walk_forward.fit_ridge_action_value",
            side_effect=RuntimeError("all fits failed"),
        ):
            with self.assertRaisesRegex(AllCandidatesFailedError, "All candidates failed") as caught:
                run_nested_walk_forward(
                    self.bars,
                    self.spec,
                    self.config,
                    dataset=self.dataset,
                )

        self.assertEqual(len(caught.exception.candidate_metrics), 4)
        self.assertTrue((caught.exception.candidate_metrics["status"] == "failed").all())


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from pretimesequence.v1.cli import (
    RUN_ARTIFACT_NAMES,
    build_run_artifacts,
    load_input_bars,
    normalize_input_bars,
    parse_float_candidates,
)
from pretimesequence.v1.spec import StrategySpec
from pretimesequence.v1.walk_forward import NestedWalkForwardConfig, NestedWalkForwardResult


def make_input_bars(*, timezone_aware: bool) -> pd.DataFrame:
    timestamps = pd.date_range(
        "2026-01-01",
        periods=6,
        freq="min",
        tz="UTC" if timezone_aware else None,
    )
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [100.0, 100.1, 100.2, 100.3, 100.4, 100.5],
            "high": [100.2, 100.3, 100.4, 100.5, 100.6, 100.7],
            "low": [99.8, 99.9, 100.0, 100.1, 100.2, 100.3],
            "close": [100.1, 100.2, 100.3, 100.4, 100.5, 100.6],
            "volume": [1_000.0] * 6,
        }
    )


def make_result() -> NestedWalkForwardResult:
    config = NestedWalkForwardConfig(
        min_development_bars=200,
        test_bars=35,
        inner_min_train_bars=30,
        inner_validation_bars=20,
        alphas=(0.1,),
        thresholds=(0.0,),
        max_outer_folds=1,
        max_inner_folds=1,
    )
    return NestedWalkForwardResult(
        fold_metrics=pd.DataFrame(
            [
                {
                    "outer_fold": 0,
                    "trades": 2,
                    "net_return": 0.01,
                    "max_drawdown": 0.005,
                    "chosen_alpha": 0.1,
                    "chosen_threshold": 0.0,
                }
            ]
        ),
        candidate_metrics=pd.DataFrame([{"outer_fold": 0, "status": "ok", "selected": True}]),
        inner_oof_predictions=pd.DataFrame([{"outer_fold": 0, "signal_index": 10}]),
        outer_predictions=pd.DataFrame([{"outer_fold": 0, "signal_index": 20}]),
        trades=pd.DataFrame([{"outer_fold": 0, "side": "long"}, {"outer_fold": 0, "side": "short"}]),
        spec=StrategySpec(),
        config=config,
    )


class InputNormalizationTests(unittest.TestCase):
    def test_naive_timestamps_require_explicit_utc_assumption(self) -> None:
        bars = make_input_bars(timezone_aware=False)

        with self.assertRaisesRegex(ValueError, "assume-naive-utc"):
            normalize_input_bars(bars, assume_naive_utc=False)

        normalized = normalize_input_bars(bars, assume_naive_utc=True)
        self.assertEqual(str(normalized["timestamp"].dt.tz), "UTC")

    def test_time_filter_and_smoke_limit_are_chronological_and_explicit(self) -> None:
        bars = make_input_bars(timezone_aware=True)

        normalized = normalize_input_bars(
            bars,
            assume_naive_utc=False,
            start="2026-01-01T00:01:00Z",
            end="2026-01-01T00:04:00Z",
            smoke_rows=2,
        )

        self.assertEqual(len(normalized), 2)
        self.assertEqual(normalized["timestamp"].iloc[0], pd.Timestamp("2026-01-01T00:01:00Z"))
        self.assertEqual(normalized["timestamp"].iloc[-1], pd.Timestamp("2026-01-01T00:02:00Z"))

    def test_candidate_parser_rejects_duplicates_and_negative_values(self) -> None:
        self.assertEqual(parse_float_candidates("0.1,1,10", "alphas"), (0.1, 1.0, 10.0))
        with self.assertRaisesRegex(ValueError, "duplicates"):
            parse_float_candidates("0.1,0.1", "alphas")
        with self.assertRaisesRegex(ValueError, "non-negative"):
            parse_float_candidates("0,-1", "thresholds")

    def test_pickle_requires_explicit_trust_flag_before_deserialization(self) -> None:
        with patch.object(Path, "is_file", return_value=True):
            with self.assertRaisesRegex(ValueError, "allow-unsafe-pickle"):
                load_input_bars("untrusted.pkl", assume_naive_utc=True)


class RunArtifactTests(unittest.TestCase):
    def test_artifacts_include_full_audit_tables_and_non_claim_boundary(self) -> None:
        artifacts = build_run_artifacts(
            make_result(),
            metadata={
                "experiment_id": "smoke-sol-001",
                "mode": "smoke",
                "input": {"path": "data/SOL.pkl", "rows": 500},
            },
        )

        self.assertEqual(set(artifacts), set(RUN_ARTIFACT_NAMES))
        manifest = json.loads(artifacts["run.json"])
        self.assertEqual(manifest["experiment_id"], "smoke-sol-001")
        self.assertEqual(manifest["mode"], "smoke")
        self.assertFalse(manifest["strategy_claim_allowed"])
        self.assertEqual(manifest["result_summary"]["outer_folds"], 1)
        self.assertEqual(manifest["result_summary"]["trades"], 2)
        fold_record = manifest["artifacts"]["fold_metrics.csv"]
        self.assertEqual(fold_record["rows"], 1)
        self.assertEqual(len(fold_record["sha256"]), 64)
        self.assertIn("net_return", fold_record["columns"])
        self.assertIn("outer_fold", artifacts["fold_metrics.csv"])
        self.assertIn("signal_index", artifacts["outer_predictions.csv"])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import sklearn

from .execution import validate_bar_cadence, validate_bars
from .dataset import fingerprint_bars
from .spec import StrategySpec
from .walk_forward import (
    NestedWalkForwardConfig,
    NestedWalkForwardResult,
    run_nested_walk_forward,
)


RUN_ARTIFACT_NAMES = (
    "run.json",
    "fold_metrics.csv",
    "candidate_metrics.csv",
    "inner_oof_predictions.csv",
    "outer_predictions.csv",
    "trades.csv",
)


def parse_float_candidates(value: str, name: str) -> tuple[float, ...]:
    """Parse a pre-registered comma-separated candidate list."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must contain at least one value.")
    try:
        candidates = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise ValueError(f"{name} must be a comma-separated list of numbers.") from exc
    if any(not math.isfinite(item) or item < 0 for item in candidates):
        raise ValueError(f"{name} values must be finite and non-negative.")
    if len(set(candidates)) != len(candidates):
        raise ValueError(f"{name} must not contain duplicates.")
    return candidates


def _utc_boundary(value: str | None, name: str) -> pd.Timestamp | None:
    if value is None:
        return None
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a valid timestamp.") from exc
    if pd.isna(timestamp):
        raise ValueError(f"{name} must be a valid timestamp.")
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def normalize_input_bars(
    df: pd.DataFrame,
    *,
    assume_naive_utc: bool,
    start: str | None = None,
    end: str | None = None,
    smoke_rows: int | None = None,
    base_interval_minutes: int = 1,
) -> pd.DataFrame:
    """Normalize user-selected bars without silently sorting or dropping rows."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("The input file must contain a pandas DataFrame.")
    required = ("timestamp", "open", "high", "low", "close", "volume")
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {missing}")
    if smoke_rows is not None and (
        not isinstance(smoke_rows, int) or isinstance(smoke_rows, bool) or smoke_rows <= 0
    ):
        raise ValueError("smoke_rows must be a positive integer when provided.")

    bars = df.copy()
    try:
        timestamps = pd.to_datetime(bars["timestamp"], errors="coerce")
    except (TypeError, ValueError) as exc:
        raise ValueError("timestamp values could not be parsed consistently.") from exc
    if timestamps.isna().any():
        raise ValueError("timestamp contains missing or invalid values.")
    timestamp_timezone = timestamps.dt.tz
    if timestamp_timezone is None:
        if not assume_naive_utc:
            raise ValueError(
                "timestamp is timezone-naive; confirm its meaning with --assume-naive-utc."
            )
        timestamps = timestamps.dt.tz_localize("UTC")
    else:
        timestamps = timestamps.dt.tz_convert("UTC")
    bars["timestamp"] = timestamps

    start_time = _utc_boundary(start, "start")
    end_time = _utc_boundary(end, "end")
    if start_time is not None and end_time is not None and start_time > end_time:
        raise ValueError("start must not be later than end.")
    if start_time is not None:
        bars = bars.loc[bars["timestamp"] >= start_time]
    if end_time is not None:
        bars = bars.loc[bars["timestamp"] <= end_time]
    if smoke_rows is not None:
        bars = bars.head(smoke_rows)
    if bars.empty:
        raise ValueError("No bars remain after applying the requested time range.")

    normalized = validate_bars(bars)
    normalized["volume"] = pd.to_numeric(normalized["volume"], errors="coerce")
    volume = normalized["volume"].to_numpy(dtype=float)
    if not np.isfinite(volume).all() or (volume < 0).any():
        raise ValueError("volume values must be finite and non-negative.")
    validate_bar_cadence(normalized, base_interval_minutes)
    return normalized


def load_input_bars(
    path: str | Path,
    *,
    assume_naive_utc: bool,
    allow_unsafe_pickle: bool = False,
    start: str | None = None,
    end: str | None = None,
    smoke_rows: int | None = None,
    base_interval_minutes: int = 1,
) -> pd.DataFrame:
    input_path = Path(path)
    if not input_path.is_file():
        raise ValueError(f"Data file does not exist: {input_path}")
    suffix = input_path.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        if not allow_unsafe_pickle:
            raise ValueError(
                "Pickle deserialization can execute code. Re-run with "
                "--allow-unsafe-pickle only for a trusted local file."
            )
        frame = pd.read_pickle(input_path)
    elif suffix == ".csv":
        frame = pd.read_csv(input_path)
    else:
        raise ValueError(f"Unsupported data file type: {suffix}")
    return normalize_input_bars(
        frame,
        assume_naive_utc=assume_naive_utc,
        start=start,
        end=end,
        smoke_rows=smoke_rows,
        base_interval_minutes=base_interval_minutes,
    )


def _frame_csv(frame: pd.DataFrame) -> str:
    return frame.to_csv(index=False, lineterminator="\n")


def _result_summary(result: NestedWalkForwardResult) -> dict[str, float | int | None]:
    fold_metrics = result.fold_metrics
    net_returns = pd.to_numeric(fold_metrics.get("net_return"), errors="coerce")
    drawdowns = pd.to_numeric(fold_metrics.get("max_drawdown"), errors="coerce")
    return {
        "outer_folds": int(len(fold_metrics)),
        "candidates": int(len(result.candidate_metrics)),
        "inner_oof_predictions": int(len(result.inner_oof_predictions)),
        "outer_predictions": int(len(result.outer_predictions)),
        "trades": int(len(result.trades)),
        "mean_outer_net_return": None if net_returns.empty else float(net_returns.mean()),
        "median_outer_net_return": None if net_returns.empty else float(net_returns.median()),
        "worst_outer_net_return": None if net_returns.empty else float(net_returns.min()),
        "max_outer_drawdown": None if drawdowns.empty else float(drawdowns.max()),
    }


def _result_frames(result: NestedWalkForwardResult) -> dict[str, pd.DataFrame]:
    return {
        "fold_metrics.csv": result.fold_metrics,
        "candidate_metrics.csv": result.candidate_metrics,
        "inner_oof_predictions.csv": result.inner_oof_predictions,
        "outer_predictions.csv": result.outer_predictions,
        "trades.csv": result.trades,
    }


def _artifact_record(name: str, payload: bytes, frame: pd.DataFrame) -> dict[str, object]:
    return {
        "name": name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "rows": int(len(frame)),
        "columns": [str(column) for column in frame.columns],
    }


def _artifact_record_from_file(
    name: str,
    path: Path,
    frame: pd.DataFrame,
) -> dict[str, object]:
    return {
        "name": name,
        "sha256": _sha256_file(path),
        "bytes": int(path.stat().st_size),
        "rows": int(len(frame)),
        "columns": [str(column) for column in frame.columns],
    }


def _run_manifest(
    result: NestedWalkForwardResult,
    metadata: Mapping[str, object],
    artifact_records: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    required_metadata = {"experiment_id", "mode", "input"}
    missing = required_metadata - set(metadata)
    if missing:
        raise ValueError(f"Missing run metadata: {sorted(missing)}")
    manifest = dict(metadata)
    manifest.update(
        {
            "schema_version": 1,
            "status": "completed",
            "created_at_utc": metadata.get(
                "created_at_utc",
                datetime.now(timezone.utc).isoformat(),
            ),
            "strategy_claim_allowed": False,
            "scope": "v1_deterministic_ridge_nested_walk_forward_baseline",
            "strategy_spec": result.spec.to_mapping(),
            "walk_forward_config": asdict(result.config),
            "result_summary": _result_summary(result),
            "artifacts": dict(artifact_records),
            "limitations": [
                "This run does not validate strategy profitability.",
                "Calibration, exchange liquidation and complete contract rules are not implemented.",
                "Paper trading and live trading are outside this command's scope.",
            ],
        }
    )
    return manifest


def build_run_artifacts(
    result: NestedWalkForwardResult,
    *,
    metadata: Mapping[str, object],
) -> dict[str, str]:
    """Serialize a complete audit bundle before any filesystem write."""

    artifacts = {name: _frame_csv(frame) for name, frame in _result_frames(result).items()}
    artifact_records = {
        name: _artifact_record(name, content.encode("utf-8"), _result_frames(result)[name])
        for name, content in artifacts.items()
    }
    manifest = _run_manifest(result, metadata, artifact_records)
    return {
        "run.json": json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        **artifacts,
    }


def ensure_output_directory_available(output_dir: str | Path) -> Path:
    output = Path(output_dir)
    if output.exists():
        raise ValueError(f"Output path already exists: {output}. Use a new experiment directory.")
    return output


def write_run_artifacts(output_dir: str | Path, artifacts: Mapping[str, str]) -> Path:
    output = ensure_output_directory_available(output_dir)
    unknown = set(artifacts) - set(RUN_ARTIFACT_NAMES)
    missing = set(RUN_ARTIFACT_NAMES) - set(artifacts)
    if unknown or missing:
        raise ValueError(f"Artifact set mismatch; missing={sorted(missing)}, unknown={sorted(unknown)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.incomplete-{uuid.uuid4().hex}"
    staging.mkdir(exist_ok=False)
    write_order = tuple(name for name in RUN_ARTIFACT_NAMES if name != "run.json") + ("run.json",)
    for name in write_order:
        (staging / name).write_bytes(artifacts[name].encode("utf-8"))
    staging.rename(output)
    return output.resolve()


def write_run_result(
    output_dir: str | Path,
    result: NestedWalkForwardResult,
    *,
    metadata: Mapping[str, object],
) -> Path:
    """Stream audit tables to a sibling staging directory, then atomically publish."""

    output = ensure_output_directory_available(output_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.incomplete-{uuid.uuid4().hex}"
    staging.mkdir(exist_ok=False)
    artifact_records: dict[str, dict[str, object]] = {}
    for name, frame in _result_frames(result).items():
        artifact_path = staging / name
        frame.to_csv(artifact_path, index=False, lineterminator="\n")
        artifact_records[name] = _artifact_record_from_file(name, artifact_path, frame)

    manifest = _run_manifest(result, metadata, artifact_records)
    (staging / "run.json").write_bytes(
        (json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode(
            "utf-8"
        )
    )
    staging.rename(output)
    return output.resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_metadata(repo_root: Path) -> dict[str, object]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def _source_tree_sha256(repo_root: Path) -> str:
    source_files = sorted((repo_root / "pretimesequence" / "v1").glob("*.py"))
    source_files.append(repo_root / "pretimesequence" / "features.py")
    digest = hashlib.sha256()
    for path in source_files:
        if not path.is_file():
            raise ValueError(f"Required source file does not exist: {path}")
        digest.update(path.relative_to(repo_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _load_strategy_spec(path: str | None) -> StrategySpec:
    if path is None:
        return StrategySpec()
    config_path = Path(path)
    if not config_path.is_file():
        raise ValueError(f"Strategy config does not exist: {config_path}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Strategy config JSON must contain an object.")
    return StrategySpec.from_mapping(payload)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pretimesequence.v1",
        description="Train and evaluate the correctness-first v1 Ridge baseline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train", help="Run nested walk-forward training/evaluation.")
    train.add_argument("--data", required=True, help="Trusted local .pkl/.pickle/.csv OHLCV file.")
    train.add_argument("--output", required=True, help="New audit output directory.")
    train.add_argument("--experiment-id", required=True, help="Unique human-readable run identifier.")
    train.add_argument("--strategy-config", default=None, help="Optional StrategySpec JSON file.")
    train.add_argument("--start", default=None, help="Inclusive UTC start timestamp.")
    train.add_argument("--end", default=None, help="Inclusive UTC end timestamp.")
    train.add_argument(
        "--assume-naive-utc",
        action="store_true",
        help="Explicitly interpret timezone-naive source timestamps as UTC.",
    )
    train.add_argument(
        "--allow-unsafe-pickle",
        action="store_true",
        help="Allow pickle only when the local file is trusted; pickle can execute code.",
    )
    train.add_argument(
        "--smoke-rows",
        type=int,
        default=None,
        help="Use only the first N selected rows and mark the run as smoke/debug.",
    )
    train.add_argument("--min-development-bars", type=int, default=300_000)
    train.add_argument("--test-bars", type=int, default=43_200)
    train.add_argument("--outer-step-bars", type=int, default=43_200)
    train.add_argument("--max-outer-folds", type=int, default=5)
    train.add_argument("--inner-min-train-bars", type=int, default=100_000)
    train.add_argument("--inner-validation-bars", type=int, default=43_200)
    train.add_argument("--inner-step-bars", type=int, default=43_200)
    train.add_argument("--max-inner-folds", type=int, default=3)
    train.add_argument("--alphas", default="0.1,1,10")
    train.add_argument("--thresholds", default="0,0.0005,0.001")
    train.add_argument("--initial-equity", type=float, default=1.0)
    train.add_argument("--notional-fraction", type=float, default=None)
    return parser


def _source_file_snapshot(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise ValueError(f"Data file does not exist: {path}")
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
        "bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _input_metadata(
    bars: pd.DataFrame,
    spec: StrategySpec,
    source_snapshot: Mapping[str, object],
) -> dict[str, object]:
    return {
        **source_snapshot,
        "selected_rows": int(len(bars)),
        "first_timestamp": bars["timestamp"].iloc[0].isoformat(),
        "last_timestamp": bars["timestamp"].iloc[-1].isoformat(),
        "selected_bars_sha256": fingerprint_bars(bars, spec),
    }


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command != "train":
        raise ValueError(f"Unsupported command: {args.command}")
    if not args.experiment_id.strip():
        raise ValueError("experiment-id must not be empty.")

    output = ensure_output_directory_available(args.output)
    spec = _load_strategy_spec(args.strategy_config)
    alphas = parse_float_candidates(args.alphas, "alphas")
    thresholds = parse_float_candidates(args.thresholds, "thresholds")
    config = NestedWalkForwardConfig(
        min_development_bars=args.min_development_bars,
        test_bars=args.test_bars,
        outer_step_bars=args.outer_step_bars,
        max_outer_folds=args.max_outer_folds,
        inner_min_train_bars=args.inner_min_train_bars,
        inner_validation_bars=args.inner_validation_bars,
        inner_step_bars=args.inner_step_bars,
        max_inner_folds=args.max_inner_folds,
        alphas=alphas,
        thresholds=thresholds,
        initial_equity=args.initial_equity,
        notional_fraction=args.notional_fraction,
    )
    data_path = Path(args.data)
    source_before = _source_file_snapshot(data_path)
    bars = load_input_bars(
        data_path,
        assume_naive_utc=args.assume_naive_utc,
        allow_unsafe_pickle=args.allow_unsafe_pickle,
        start=args.start,
        end=args.end,
        smoke_rows=args.smoke_rows,
        base_interval_minutes=spec.base_interval_minutes,
    )
    source_after = _source_file_snapshot(data_path)
    if source_before != source_after:
        raise RuntimeError("The source data file changed while it was being loaded.")
    input_metadata = _input_metadata(bars, spec, source_after)
    mode = "smoke" if args.smoke_rows is not None else "research_baseline"
    # Keep subprocess output ASCII-safe because some Windows ``conda run``
    # installations decode captured stdout with the active legacy code page.
    print(f"Data check passed: {len(bars)} bars; starting v1 training ({mode})...", flush=True)
    result = run_nested_walk_forward(bars, spec, config)
    source_root = Path(__file__).resolve().parents[2]
    metadata = {
        "experiment_id": args.experiment_id,
        "mode": mode,
        "input": input_metadata,
        "selection": {
            "start": args.start,
            "end": args.end,
            "smoke_rows": args.smoke_rows,
            "assume_naive_utc": bool(args.assume_naive_utc),
        },
        "environment": {
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "code": {
            **_git_metadata(source_root),
            "source_tree_sha256": _source_tree_sha256(source_root),
        },
    }
    written_to = write_run_result(output, result, metadata=metadata)
    summary = _result_summary(result)
    print(
        f"Training evaluation completed: outer_folds={summary['outer_folds']}, "
        f"outer_predictions={summary['outer_predictions']}, trades={summary['trades']}"
    )
    print(f"Audit output: {written_to}")
    print("Boundary: baseline research only; this does not validate profitability or live trading.")
    return 0


def cli_entrypoint() -> None:
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Training did not complete: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

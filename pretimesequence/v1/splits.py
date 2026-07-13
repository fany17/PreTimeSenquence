from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .spec import StrategySpec


@dataclass(frozen=True)
class PurgedChronologicalSplit:
    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray
    train_boundary: int
    validation_boundary: int
    purge_bars: int
    embargo_bars: int


@dataclass(frozen=True)
class OuterWalkForwardFold:
    """One expanding outer fold expressed in global signal/bar indices."""

    fold_id: int
    development_indices: np.ndarray
    test_indices: np.ndarray
    boundary_signal_index: int
    test_start_signal_index: int
    test_end_bar_index: int


@dataclass(frozen=True)
class InnerWalkForwardFold:
    """One expanding inner train/validation fold inside outer development."""

    fold_id: int
    train_indices: np.ndarray
    validation_indices: np.ndarray
    boundary_signal_index: int
    validation_start_signal_index: int
    validation_end_bar_index: int


def purged_chronological_split(
    n_samples: int,
    spec: StrategySpec,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
) -> PurgedChronologicalSplit:
    """Return ordered train/validation/test indices with purge and embargo.

    At each boundary, rows immediately before the next partition are purged so
    their target paths cannot reach into it. Rows immediately after the nominal
    boundary are embargoed. The outer test is never returned as an input to any
    parameter-selection helper.
    """

    if not isinstance(n_samples, int) or isinstance(n_samples, bool) or n_samples <= 0:
        raise ValueError("n_samples must be a positive integer.")
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be in (0, 1).")
    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be in (0, 1).")
    if train_ratio + validation_ratio >= 1:
        raise ValueError("train_ratio + validation_ratio must be less than 1.")

    train_boundary = int(n_samples * train_ratio)
    validation_boundary = int(n_samples * (train_ratio + validation_ratio))
    purge = spec.effective_purge_bars
    embargo = spec.embargo_bars

    train_stop = train_boundary - purge
    validation_start = train_boundary + embargo
    validation_stop = validation_boundary - purge
    test_start = validation_boundary + embargo

    train_indices = np.arange(0, train_stop, dtype=int)
    validation_indices = np.arange(validation_start, validation_stop, dtype=int)
    test_indices = np.arange(test_start, n_samples, dtype=int)
    if not len(train_indices) or not len(validation_indices) or not len(test_indices):
        raise ValueError(
            "Insufficient samples after applying the configured purge and embargo "
            f"(n={n_samples}, purge={purge}, embargo={embargo})."
        )

    return PurgedChronologicalSplit(
        train_indices=train_indices,
        validation_indices=validation_indices,
        test_indices=test_indices,
        train_boundary=train_boundary,
        validation_boundary=validation_boundary,
        purge_bars=purge,
        embargo_bars=embargo,
    )


_INTERVAL_META_COLUMNS = (
    "signal_index",
    "signal_time",
    "entry_index",
    "horizon_end_index",
    "feature_valid",
    "target_valid",
)


def _positive_integer(name: str, value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _optional_positive_integer(name: str, value: int | None) -> int | None:
    if value is None:
        return None
    return _positive_integer(name, value)


def _validate_interval_meta(sample_meta: pd.DataFrame, spec: StrategySpec) -> pd.DataFrame:
    if not isinstance(sample_meta, pd.DataFrame):
        raise TypeError("sample_meta must be a pandas DataFrame.")
    missing = [column for column in _INTERVAL_META_COLUMNS if column not in sample_meta.columns]
    if missing:
        raise ValueError(f"Missing interval metadata columns: {missing}")
    if sample_meta.empty:
        raise ValueError("sample_meta must not be empty.")

    frame = sample_meta.copy()
    for column in ("signal_index", "entry_index", "horizon_end_index"):
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.isna().any() or ((numeric % 1) != 0).any():
            raise ValueError(f"{column} values must be integers.")
        frame[column] = numeric.astype(int)

    if frame["signal_index"].duplicated().any():
        raise ValueError("signal_index values must be unique.")
    if not frame["signal_index"].is_monotonic_increasing:
        raise ValueError("signal_index values must be sorted in increasing order.")
    index_values = pd.to_numeric(pd.Series(frame.index), errors="coerce")
    if index_values.isna().any() or not np.array_equal(
        index_values.to_numpy(dtype=int),
        frame["signal_index"].to_numpy(dtype=int),
    ):
        raise ValueError("sample_meta index must equal its global signal_index column.")

    expected_entry = frame["signal_index"] + spec.entry_delay_bars
    expected_horizon_end = frame["signal_index"] + spec.horizon_bars
    if not frame["entry_index"].equals(expected_entry):
        raise ValueError("entry_index does not match StrategySpec.entry_delay_bars.")
    if not frame["horizon_end_index"].equals(expected_horizon_end):
        raise ValueError("horizon_end_index does not match StrategySpec.horizon_bars.")

    for column in ("feature_valid", "target_valid"):
        if not pd.api.types.is_bool_dtype(frame[column].dtype) or frame[column].isna().any():
            raise ValueError(f"{column} must contain non-null boolean values.")

    for value in frame["signal_time"]:
        if pd.isna(value):
            raise ValueError("signal_time values must be valid and timezone-aware.")
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("signal_time values must be timezone-aware.")
    signal_times = pd.to_datetime(frame["signal_time"], utc=True, errors="coerce")
    if signal_times.isna().any() or signal_times.duplicated().any():
        raise ValueError("signal_time values must be valid and unique.")
    if not signal_times.is_monotonic_increasing:
        raise ValueError("signal_time values must be sorted in increasing order.")
    frame["signal_time"] = signal_times
    return frame


def _eligible_mask(frame: pd.DataFrame, valid_only: bool) -> pd.Series:
    if not isinstance(valid_only, bool):
        raise ValueError("valid_only must be a boolean.")
    if not valid_only:
        return pd.Series(True, index=frame.index, dtype=bool)
    return frame["feature_valid"] & frame["target_valid"]


def _assert_real_interval_isolation(
    frame: pd.DataFrame,
    left_indices: np.ndarray,
    right_indices: np.ndarray,
) -> None:
    left_end = int(frame.loc[left_indices, "horizon_end_index"].max())
    right_start = int(frame.loc[right_indices, "signal_index"].min())
    if left_end >= right_start:
        raise RuntimeError(
            "Target interval isolation failed: a left target reaches the right partition."
        )


def outer_walk_forward_folds(
    sample_meta: pd.DataFrame,
    spec: StrategySpec,
    *,
    min_development_bars: int,
    test_bars: int,
    step_bars: int | None = None,
    max_folds: int | None = None,
    valid_only: bool = True,
) -> tuple[OuterWalkForwardFold, ...]:
    """Build expanding outer folds without compressing global bar positions.

    The nominal boundary is fixed on the global ``signal_index`` timeline.
    Development targets must end before that boundary. The outer test begins
    only after the configured embargo and includes only targets whose complete
    horizon fits inside the test bar window.
    """

    frame = _validate_interval_meta(sample_meta, spec)
    min_development_bars = _positive_integer("min_development_bars", min_development_bars)
    test_bars = _positive_integer("test_bars", test_bars)
    resolved_step = test_bars if step_bars is None else _positive_integer("step_bars", step_bars)
    max_folds = _optional_positive_integer("max_folds", max_folds)
    eligible = _eligible_mask(frame, valid_only)

    first_signal = int(frame["signal_index"].iloc[0])
    last_available_bar = int(frame["horizon_end_index"].max())
    boundary = first_signal + min_development_bars
    folds: list[OuterWalkForwardFold] = []

    while max_folds is None or len(folds) < max_folds:
        test_start = boundary + spec.embargo_bars
        test_end_bar = test_start + test_bars - 1
        if test_end_bar > last_available_bar:
            break

        development_mask = (
            eligible
            & (frame["signal_index"] < boundary)
            & (frame["horizon_end_index"] < boundary)
            & (frame["signal_index"] + spec.effective_purge_bars < boundary)
        )
        test_mask = (
            eligible
            & (frame["signal_index"] >= test_start)
            & (frame["horizon_end_index"] <= test_end_bar)
        )
        development_indices = frame.loc[development_mask, "signal_index"].to_numpy(dtype=int)
        test_indices = frame.loc[test_mask, "signal_index"].to_numpy(dtype=int)
        if not len(development_indices) or not len(test_indices):
            raise ValueError(
                "Insufficient valid samples for an outer fold after interval purge and embargo."
            )
        _assert_real_interval_isolation(frame, development_indices, test_indices)
        folds.append(
            OuterWalkForwardFold(
                fold_id=len(folds),
                development_indices=development_indices,
                test_indices=test_indices,
                boundary_signal_index=boundary,
                test_start_signal_index=test_start,
                test_end_bar_index=test_end_bar,
            )
        )
        boundary += resolved_step

    if not folds:
        raise ValueError(
            "Insufficient samples to construct any outer walk-forward fold with the configured windows."
        )
    return tuple(folds)


def inner_walk_forward_folds(
    sample_meta: pd.DataFrame,
    development_indices: np.ndarray,
    spec: StrategySpec,
    *,
    min_train_bars: int,
    validation_bars: int,
    step_bars: int | None = None,
    max_folds: int | None = None,
    valid_only: bool = True,
) -> tuple[InnerWalkForwardFold, ...]:
    """Build expanding inner folds inside one outer development partition."""

    frame = _validate_interval_meta(sample_meta, spec)
    min_train_bars = _positive_integer("min_train_bars", min_train_bars)
    validation_bars = _positive_integer("validation_bars", validation_bars)
    resolved_step = validation_bars if step_bars is None else _positive_integer("step_bars", step_bars)
    max_folds = _optional_positive_integer("max_folds", max_folds)

    raw_indices = np.asarray(development_indices)
    if raw_indices.ndim != 1 or not len(raw_indices):
        raise ValueError("development_indices must be a non-empty one-dimensional array.")
    numeric_indices = pd.to_numeric(pd.Series(raw_indices), errors="coerce")
    if numeric_indices.isna().any() or ((numeric_indices % 1) != 0).any():
        raise ValueError("development_indices must contain integers.")
    development = numeric_indices.to_numpy(dtype=int)
    if len(np.unique(development)) != len(development):
        raise ValueError("development_indices must be unique.")
    known_indices = set(frame["signal_index"].to_numpy(dtype=int))
    unknown = sorted(set(development) - known_indices)
    if unknown:
        raise ValueError(f"development_indices are outside sample_meta: {unknown[:5]}")

    allowed = frame["signal_index"].isin(development)
    eligible = allowed & _eligible_mask(frame, valid_only)
    allowed_frame = frame.loc[allowed]
    first_signal = int(allowed_frame["signal_index"].min())
    last_available_bar = int(allowed_frame["horizon_end_index"].max())
    boundary = first_signal + min_train_bars
    folds: list[InnerWalkForwardFold] = []

    while max_folds is None or len(folds) < max_folds:
        validation_start = boundary + spec.embargo_bars
        validation_end_bar = validation_start + validation_bars - 1
        if validation_end_bar > last_available_bar:
            break

        train_mask = (
            eligible
            & (frame["signal_index"] < boundary)
            & (frame["horizon_end_index"] < boundary)
            & (frame["signal_index"] + spec.effective_purge_bars < boundary)
        )
        validation_mask = (
            eligible
            & (frame["signal_index"] >= validation_start)
            & (frame["horizon_end_index"] <= validation_end_bar)
        )
        train_indices = frame.loc[train_mask, "signal_index"].to_numpy(dtype=int)
        validation_indices = frame.loc[validation_mask, "signal_index"].to_numpy(dtype=int)
        if not len(train_indices) or not len(validation_indices):
            raise ValueError(
                "Insufficient valid samples for an inner fold after interval purge and embargo."
            )
        _assert_real_interval_isolation(frame, train_indices, validation_indices)
        folds.append(
            InnerWalkForwardFold(
                fold_id=len(folds),
                train_indices=train_indices,
                validation_indices=validation_indices,
                boundary_signal_index=boundary,
                validation_start_signal_index=validation_start,
                validation_end_bar_index=validation_end_bar,
            )
        )
        boundary += resolved_step

    if not folds:
        raise ValueError(
            "Insufficient samples to construct any inner walk-forward fold with the configured windows."
        )
    return tuple(folds)

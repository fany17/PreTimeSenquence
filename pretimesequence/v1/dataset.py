from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..features import FEATURE_COLUMNS, add_features
from .execution import validate_bar_cadence, validate_bars
from .spec import StrategySpec
from .targets import make_path_targets


TARGET_VALUE_COLUMNS = (
    "return_h",
    "mfe_h",
    "mae_h",
    "long_net_return",
    "short_net_return",
)


@dataclass(frozen=True)
class SupervisedDataset:
    """Aligned v1 feature, target and interval metadata tables.

    Every table is indexed by the positional ``signal_index`` of the validated,
    reset bar frame. Invalid warm-up or target rows remain present and are
    identified by ``sample_meta.feature_valid`` and ``target_valid``.
    """

    X: pd.DataFrame
    targets: pd.DataFrame
    sample_meta: pd.DataFrame
    feature_names: tuple[str, ...]
    spec: StrategySpec
    bar_fingerprint: str


def _validate_and_normalize_bars(df: pd.DataFrame, spec: StrategySpec) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("bars must be a pandas DataFrame.")
    if "volume" not in df.columns:
        raise ValueError("Missing required bar column: volume")

    bars = validate_bars(df)
    validate_bar_cadence(bars, spec.base_interval_minutes)
    volume = pd.to_numeric(bars["volume"], errors="coerce")
    volume_values = volume.to_numpy(dtype=float)
    if not np.isfinite(volume_values).all() or (volume_values < 0).any():
        raise ValueError("volume values must be finite and non-negative.")
    bars["volume"] = volume
    return bars


def _target_validity(targets: pd.DataFrame) -> pd.Series:
    missing = [column for column in TARGET_VALUE_COLUMNS if column not in targets.columns]
    if missing:
        raise ValueError(f"Missing required target value columns: {missing}")
    numeric_targets = targets.loc[:, TARGET_VALUE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    finite_values = np.isfinite(numeric_targets.to_numpy(dtype=float)).all(axis=1)
    excluded = targets["long_excluded"].astype(bool) | targets["short_excluded"].astype(bool)
    return pd.Series(finite_values & ~excluded.to_numpy(), index=targets.index, dtype=bool)


def _fingerprint_normalized_bars(bars: pd.DataFrame) -> str:
    columns = ["timestamp", "open", "high", "low", "close", "volume"]
    if "funding_payment_rate" in bars.columns:
        columns.append("funding_payment_rate")
    values = bars.loc[:, columns].copy()
    hashes = pd.util.hash_pandas_object(values, index=False).to_numpy(dtype=np.uint64)
    digest = hashlib.sha256()
    digest.update("\x1f".join(columns).encode("utf-8"))
    digest.update(hashes.tobytes())
    return digest.hexdigest()


def fingerprint_bars(df: pd.DataFrame, spec: StrategySpec) -> str:
    """Return an in-memory provenance fingerprint for v1 OHLCV inputs."""

    return _fingerprint_normalized_bars(_validate_and_normalize_bars(df, spec))


def _validate_alignment(
    bars: pd.DataFrame,
    X: pd.DataFrame,
    targets: pd.DataFrame,
    sample_meta: pd.DataFrame,
    spec: StrategySpec,
) -> None:
    expected_index = pd.Index(targets["signal_index"].to_numpy(dtype=int), name="signal_index")
    if not expected_index.is_unique or not expected_index.is_monotonic_increasing:
        raise ValueError("signal_index must be unique and strictly increasing.")
    if not X.index.equals(expected_index) or not targets.index.equals(expected_index):
        raise ValueError("Feature and target signal_index alignment failed.")
    if not sample_meta.index.equals(expected_index):
        raise ValueError("sample_meta signal_index alignment failed.")
    if tuple(X.columns) != tuple(FEATURE_COLUMNS):
        raise ValueError("Feature schema does not match FEATURE_COLUMNS.")

    signal_indices = expected_index.to_numpy(dtype=int)
    if (signal_indices < 0).any() or (signal_indices >= len(bars)).any():
        raise ValueError("signal_index is outside the normalized bar frame.")
    expected_entry = signal_indices + spec.entry_delay_bars
    expected_horizon_end = signal_indices + spec.horizon_bars
    if not np.array_equal(sample_meta["entry_index"].to_numpy(dtype=int), expected_entry):
        raise ValueError("entry_index does not match StrategySpec.entry_delay_bars.")
    if not np.array_equal(
        sample_meta["horizon_end_index"].to_numpy(dtype=int),
        expected_horizon_end,
    ):
        raise ValueError("horizon_end_index does not match StrategySpec.horizon_bars.")

    expected_signal_times = bars.iloc[signal_indices]["timestamp"].reset_index(drop=True)
    actual_signal_times = sample_meta["signal_time"].reset_index(drop=True)
    if not expected_signal_times.equals(actual_signal_times):
        raise ValueError("signal_time does not match the feature cutoff bar.")


def build_supervised_dataset(df: pd.DataFrame, spec: StrategySpec) -> SupervisedDataset:
    """Build a non-imputed causal baseline dataset for v1 training.

    The v0 ``add_features`` implementation and ``FEATURE_COLUMNS`` schema are
    reused as the current causal baseline. No forward fill or zero fill is
    applied: warm-up rows remain in the returned tables and are marked invalid.
    """

    bars = _validate_and_normalize_bars(df, spec)
    featured = add_features(bars)
    missing_features = [column for column in FEATURE_COLUMNS if column not in featured.columns]
    if missing_features:
        raise ValueError(f"Feature generation did not produce required columns: {missing_features}")
    if len(featured) != len(bars):
        raise ValueError("Feature generation changed the number of bar rows.")

    raw_targets = make_path_targets(bars, spec)
    signal_indices = raw_targets["signal_index"].to_numpy(dtype=int)
    signal_index = pd.Index(signal_indices, name="signal_index")

    # Positional extraction is deliberate. The caller's original pandas index
    # has no role in the v1 sample identity.
    X = featured.loc[:, FEATURE_COLUMNS].iloc[signal_indices].copy()
    X.index = signal_index

    targets = raw_targets.copy()
    targets.index = signal_index
    target_valid = _target_validity(targets)
    feature_values = X.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    feature_valid = pd.Series(
        np.isfinite(feature_values).all(axis=1),
        index=signal_index,
        dtype=bool,
    )

    metadata_columns = [
        "signal_index",
        "signal_time",
        "entry_index",
        "entry_time",
        "horizon_end_index",
        "horizon_end_time",
        "ambiguous_bar",
        "long_excluded",
        "short_excluded",
    ]
    sample_meta = targets.loc[:, metadata_columns].copy()
    sample_meta["feature_valid"] = feature_valid
    sample_meta["target_valid"] = target_valid

    _validate_alignment(bars, X, targets, sample_meta, spec)
    return SupervisedDataset(
        X=X,
        targets=targets,
        sample_meta=sample_meta,
        feature_names=tuple(FEATURE_COLUMNS),
        spec=spec,
        bar_fingerprint=_fingerprint_normalized_bars(bars),
    )

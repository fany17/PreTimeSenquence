from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


ACTION_TARGET_COLUMNS = ("long_net_return", "short_net_return")
PREDICTION_COLUMNS = (
    "signal_index",
    "pred_long_net_return",
    "pred_short_net_return",
)


@dataclass(frozen=True)
class RidgeActionValueModel:
    """A deterministic scaled Ridge baseline fitted on one chronological slice."""

    scaler: StandardScaler
    estimator: Ridge
    feature_names: tuple[str, ...]
    alpha: float
    train_indices: tuple[int, ...]


def _validated_signal_indices(indices: np.ndarray, available_index: pd.Index, name: str) -> np.ndarray:
    raw = np.asarray(indices)
    if raw.ndim != 1 or not len(raw):
        raise ValueError(f"{name} must be a non-empty one-dimensional array.")
    numeric = pd.to_numeric(pd.Series(raw), errors="coerce")
    if numeric.isna().any() or ((numeric % 1) != 0).any():
        raise ValueError(f"{name} must contain integer global signal indices.")
    resolved = numeric.to_numpy(dtype=int)
    if len(np.unique(resolved)) != len(resolved):
        raise ValueError(f"{name} must contain unique signal indices.")
    if not pd.Index(resolved).is_monotonic_increasing:
        raise ValueError(f"{name} must be sorted in increasing chronological order.")
    missing = sorted(set(resolved) - set(available_index.to_numpy(dtype=int)))
    if missing:
        raise ValueError(f"{name} contains indices outside the feature table: {missing[:5]}")
    return resolved


def _validate_feature_frame(X: pd.DataFrame, expected_schema: tuple[str, ...] | None = None) -> tuple[str, ...]:
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")
    if X.empty:
        raise ValueError("X must not be empty.")
    if X.columns.duplicated().any():
        raise ValueError("Feature schema contains duplicate columns.")
    schema = tuple(str(column) for column in X.columns)
    if expected_schema is not None and schema != expected_schema:
        raise ValueError("Feature schema or column order does not match the fitted model.")
    if X.index.duplicated().any():
        raise ValueError("Feature signal indices must be unique.")
    numeric_index = pd.to_numeric(pd.Series(X.index), errors="coerce")
    if numeric_index.isna().any() or ((numeric_index % 1) != 0).any():
        raise ValueError("Feature index must contain integer global signal indices.")
    return schema


def fit_ridge_action_value(
    X: pd.DataFrame,
    targets: pd.DataFrame,
    train_indices: np.ndarray,
    *,
    alpha: float,
) -> RidgeActionValueModel:
    """Fit scaler and multi-output Ridge using only ``train_indices`` rows."""

    schema = _validate_feature_frame(X)
    if not isinstance(targets, pd.DataFrame):
        raise TypeError("targets must be a pandas DataFrame.")
    missing_targets = [column for column in ACTION_TARGET_COLUMNS if column not in targets.columns]
    if missing_targets:
        raise ValueError(f"Missing action-value target columns: {missing_targets}")
    if not np.isfinite(float(alpha)) or float(alpha) < 0:
        raise ValueError("alpha must be finite and non-negative.")

    resolved_indices = _validated_signal_indices(train_indices, X.index, "train_indices")
    missing_from_targets = sorted(set(resolved_indices) - set(targets.index.to_numpy(dtype=int)))
    if missing_from_targets:
        raise ValueError(f"Training indices are outside the target table: {missing_from_targets[:5]}")

    X_train = X.loc[resolved_indices]
    y_train = targets.loc[resolved_indices, ACTION_TARGET_COLUMNS]
    X_values = X_train.to_numpy(dtype=float)
    y_values = y_train.to_numpy(dtype=float)
    if not np.isfinite(X_values).all():
        raise ValueError("Training features must be finite; filter feature_valid rows first.")
    if not np.isfinite(y_values).all():
        raise ValueError("Training targets must be finite; filter target_valid rows first.")

    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(X_values)
    estimator = Ridge(alpha=float(alpha))
    estimator.fit(scaled_train, y_values)
    return RidgeActionValueModel(
        scaler=scaler,
        estimator=estimator,
        feature_names=schema,
        alpha=float(alpha),
        train_indices=tuple(int(index) for index in resolved_indices),
    )


def predict_action_values(model: RidgeActionValueModel, X: pd.DataFrame) -> pd.DataFrame:
    """Predict long/short unit-notional net returns with strict schema checks."""

    if not isinstance(model, RidgeActionValueModel):
        raise TypeError("model must be a RidgeActionValueModel.")
    _validate_feature_frame(X, model.feature_names)
    X_values = X.to_numpy(dtype=float)
    if not np.isfinite(X_values).all():
        raise ValueError("Prediction features must be finite.")
    predicted = np.asarray(model.estimator.predict(model.scaler.transform(X_values)), dtype=float)
    if predicted.shape != (len(X), 2) or not np.isfinite(predicted).all():
        raise RuntimeError("Ridge action-value prediction returned an invalid shape or value.")
    signal_indices = pd.to_numeric(pd.Series(X.index), errors="raise").to_numpy(dtype=int)
    return pd.DataFrame(
        {
            "signal_index": signal_indices,
            "pred_long_net_return": predicted[:, 0],
            "pred_short_net_return": predicted[:, 1],
        }
    )

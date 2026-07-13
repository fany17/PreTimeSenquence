from __future__ import annotations

import numpy as np
import pandas as pd

from .training import PREDICTION_COLUMNS


def decisions_from_action_values(predictions: pd.DataFrame, *, threshold: float) -> pd.DataFrame:
    """Convert frozen action-value predictions to long/short/flat decisions.

    This pure policy receives no targets or evaluation metrics. An action must
    be the unique maximum and must be strictly greater than ``threshold``;
    exact long/short ties are always flat.
    """

    if not isinstance(predictions, pd.DataFrame):
        raise TypeError("predictions must be a pandas DataFrame.")
    if tuple(predictions.columns) != PREDICTION_COLUMNS:
        raise ValueError(
            "Prediction schema must contain only signal_index and the ordered "
            "long/short action-value predictions."
        )
    resolved_threshold = float(threshold)
    if not np.isfinite(resolved_threshold) or resolved_threshold < 0:
        raise ValueError("threshold must be finite and non-negative.")

    signal_index = pd.to_numeric(predictions["signal_index"], errors="coerce")
    if signal_index.isna().any() or ((signal_index % 1) != 0).any():
        raise ValueError("signal_index values must be integers.")
    if signal_index.duplicated().any():
        raise ValueError("signal_index values must be unique.")
    values = predictions.loc[:, PREDICTION_COLUMNS[1:]].apply(pd.to_numeric, errors="coerce")
    value_array = values.to_numpy(dtype=float)
    if not np.isfinite(value_array).all():
        raise ValueError("Action-value predictions must be finite.")

    long_value = value_array[:, 0]
    short_value = value_array[:, 1]
    actions = np.full(len(predictions), "flat", dtype=object)
    actions[(long_value > resolved_threshold) & (long_value > short_value)] = "long"
    actions[(short_value > resolved_threshold) & (short_value > long_value)] = "short"
    return pd.DataFrame(
        {
            "signal_index": signal_index.astype(int).to_numpy(),
            "action": actions,
        }
    )

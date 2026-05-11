from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LabelConfig:
    """Configuration for tradeable trend labels.

    The label at row t assumes the signal is formed after bar t closes and
    execution happens at the next bar open. Future bars are only used to build
    training labels, never as prediction features.
    """

    horizon: int = 20
    atr_window: int = 14
    atr_multiple: float = 4.0
    min_return: float = 0.005
    fee_rate: float = 0.0005
    slippage_rate: float = 0.0002
    neutral_return: float = 0.0005

    @property
    def round_trip_cost(self) -> float:
        return 2 * (self.fee_rate + self.slippage_rate)


def add_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")
    true_range = pd.concat(
        [(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def make_triple_barrier_labels(df: pd.DataFrame, config: LabelConfig | None = None) -> pd.DataFrame:
    """Create long/short/flat labels using a finite-horizon triple barrier.

    Label semantics:
    - long: upper barrier is hit first, or horizon return is materially positive.
    - short: lower barrier is hit first, or horizon return is materially negative.
    - flat: neither side clears the cost-adjusted neutral band.
    """

    config = config or LabelConfig()
    if len(df) < config.horizon + 2:
        raise ValueError("Not enough rows to create labels for the requested horizon.")

    out = df.copy()
    out["atr"] = add_atr(out, config.atr_window)
    out["future_entry_price"] = out["open"].shift(-1).fillna(out["close"].shift(-1))
    out["barrier_return"] = (
        (config.atr_multiple * out["atr"] / out["close"]).clip(lower=config.min_return)
        + config.round_trip_cost
    )
    out["label"] = "flat"
    out["label_code"] = 1
    out["label_reason"] = "no_edge"
    out["forward_return"] = np.nan

    last_label_idx = len(out) - config.horizon - 1
    for i in range(max(config.atr_window, 1), last_label_idx + 1):
        entry = float(out.at[i, "future_entry_price"])
        barrier = float(out.at[i, "barrier_return"])
        if not np.isfinite(entry) or not np.isfinite(barrier) or entry <= 0:
            continue

        upper = entry * (1 + barrier)
        lower = entry * (1 - barrier)
        future = out.iloc[i + 1 : i + 1 + config.horizon]

        upper_hits = future.index[future["high"] >= upper]
        lower_hits = future.index[future["low"] <= lower]
        first_upper = upper_hits[0] if len(upper_hits) else None
        first_lower = lower_hits[0] if len(lower_hits) else None

        horizon_close = float(future["close"].iloc[-1])
        forward_return = horizon_close / entry - 1 - config.round_trip_cost
        out.at[i, "forward_return"] = forward_return

        if first_upper is not None and (first_lower is None or first_upper < first_lower):
            out.at[i, "label"] = "long"
            out.at[i, "label_code"] = 2
            out.at[i, "label_reason"] = "upper_barrier"
        elif first_lower is not None and (first_upper is None or first_lower < first_upper):
            out.at[i, "label"] = "short"
            out.at[i, "label_code"] = 0
            out.at[i, "label_reason"] = "lower_barrier"
        elif forward_return > config.neutral_return + config.round_trip_cost:
            out.at[i, "label"] = "long"
            out.at[i, "label_code"] = 2
            out.at[i, "label_reason"] = "horizon_return"
        elif forward_return < -(config.neutral_return + config.round_trip_cost):
            out.at[i, "label"] = "short"
            out.at[i, "label_code"] = 0
            out.at[i, "label_reason"] = "horizon_return"

    valid = out.index <= last_label_idx
    valid &= out["atr"].notna() & out["future_entry_price"].notna()
    return out.loc[valid].reset_index(drop=True)


def label_summary(labels: pd.DataFrame) -> pd.DataFrame:
    summary = labels["label"].value_counts(dropna=False).rename_axis("label").reset_index(name="count")
    summary["share"] = summary["count"] / summary["count"].sum()
    return summary

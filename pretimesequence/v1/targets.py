from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .execution import (
    Side,
    barrier_levels,
    calculate_return_breakdown,
    detect_barrier_exit,
    entry_fill_price,
    exit_fill_price,
    funding_rate_for_bar,
    validate_bar_cadence,
    validate_bars,
)
from .spec import StrategySpec


@dataclass(frozen=True)
class _PathOutcome:
    entry_fill: float
    exit_raw: float
    exit_fill: float
    exit_reason: str
    holding_bars: int
    ambiguous: bool
    excluded: bool
    gross_return: float
    fee_return: float
    funding_return: float
    net_return: float


def _simulate_path(path: pd.DataFrame, entry_raw: float, side: Side, spec: StrategySpec) -> _PathOutcome:
    entry_fill = entry_fill_price(entry_raw, side, spec.costs)
    levels = barrier_levels(entry_fill, side, spec)
    cumulative_funding_rate = 0.0
    exit_raw = float(path.iloc[-1]["close"])
    exit_reason = "horizon" if len(path) == spec.horizon_bars - spec.entry_delay_bars + 1 else "time_exit"
    holding_bars = len(path)
    ambiguous = False
    excluded = False

    for offset, (_, bar) in enumerate(path.iterrows(), start=1):
        cumulative_funding_rate += funding_rate_for_bar(bar, spec.costs)
        trigger = detect_barrier_exit(bar, side, levels, spec.ambiguous_bar_policy)
        if trigger is None:
            continue
        exit_raw = trigger.raw_price
        exit_reason = trigger.reason
        holding_bars = offset
        ambiguous = trigger.ambiguous
        excluded = trigger.excluded
        break

    exit_fill = exit_fill_price(exit_raw, side, spec.costs)
    breakdown = calculate_return_breakdown(
        side=side,
        entry_fill=entry_fill,
        exit_fill=exit_fill,
        fee_rate=spec.costs.fee_rate,
        cumulative_funding_rate=cumulative_funding_rate,
    )
    net_return = np.nan if excluded else breakdown.net_return
    return _PathOutcome(
        entry_fill=entry_fill,
        exit_raw=exit_raw,
        exit_fill=exit_fill,
        exit_reason=exit_reason,
        holding_bars=holding_bars,
        ambiguous=ambiguous,
        excluded=excluded,
        gross_return=breakdown.gross_return,
        fee_return=breakdown.fee_return,
        funding_return=breakdown.funding_return,
        net_return=net_return,
    )


def _hit_value(reason: str) -> object:
    if reason == "take_profit":
        return True
    if reason == "stop_loss":
        return False
    return pd.NA


def _first_hit_bar(reason: str, holding_bars: int) -> object:
    if reason in {"take_profit", "stop_loss", "ambiguous_excluded"}:
        return holding_bars
    return pd.NA


def _make_path_targets_reference(df: pd.DataFrame, spec: StrategySpec) -> pd.DataFrame:
    """Build unit-notional path and action-outcome targets.

    A signal at bar ``t`` uses the next bar open as its raw entry and observes
    no data after ``t + horizon_bars``. The returned action outcomes are not
    multiplied by account leverage and no best-side label is selected.
    """

    bars = validate_bars(df)
    validate_bar_cadence(bars, spec.base_interval_minutes)
    if len(bars) < spec.horizon_bars + 1:
        raise ValueError("Not enough rows for the configured target horizon.")

    rows: list[dict[str, object]] = []
    last_signal_index = len(bars) - spec.horizon_bars - 1
    for signal_index in range(last_signal_index + 1):
        entry_index = signal_index + spec.entry_delay_bars
        horizon_end_index = signal_index + spec.horizon_bars
        full_path = bars.iloc[entry_index : horizon_end_index + 1]
        outcome_path = full_path.iloc[: spec.max_holding_bars]
        entry_raw = float(full_path.iloc[0]["open"])
        horizon_close = float(bars.iloc[horizon_end_index]["close"])
        long_outcome = _simulate_path(outcome_path, entry_raw, "long", spec)
        short_outcome = _simulate_path(outcome_path, entry_raw, "short", spec)

        long_exit_index = entry_index + long_outcome.holding_bars - 1
        short_exit_index = entry_index + short_outcome.holding_bars - 1
        rows.append(
            {
                "signal_index": signal_index,
                "signal_time": bars.iloc[signal_index]["timestamp"],
                "entry_index": entry_index,
                "entry_time": bars.iloc[entry_index]["timestamp"],
                "horizon_end_index": horizon_end_index,
                "horizon_end_time": bars.iloc[horizon_end_index]["timestamp"],
                "entry_price": entry_raw,
                "return_h": float(np.log(horizon_close / entry_raw)),
                "mfe_h": float(full_path["high"].max() / entry_raw - 1),
                "mae_h": float(full_path["low"].min() / entry_raw - 1),
                "long_entry_price": long_outcome.entry_fill,
                "long_exit_index": long_exit_index,
                "long_exit_time": bars.iloc[long_exit_index]["timestamp"],
                "long_exit_price": long_outcome.exit_fill,
                "long_exit_reason": long_outcome.exit_reason,
                "long_holding_bars": long_outcome.holding_bars,
                "long_tp_before_sl": _hit_value(long_outcome.exit_reason),
                "long_first_hit_bar": _first_hit_bar(long_outcome.exit_reason, long_outcome.holding_bars),
                "long_ambiguous": long_outcome.ambiguous,
                "long_excluded": long_outcome.excluded,
                "long_gross_return": long_outcome.gross_return,
                "long_fee_return": long_outcome.fee_return,
                "long_funding_return": long_outcome.funding_return,
                "long_net_return": long_outcome.net_return,
                "short_entry_price": short_outcome.entry_fill,
                "short_exit_index": short_exit_index,
                "short_exit_time": bars.iloc[short_exit_index]["timestamp"],
                "short_exit_price": short_outcome.exit_fill,
                "short_exit_reason": short_outcome.exit_reason,
                "short_holding_bars": short_outcome.holding_bars,
                "short_tp_before_sl": _hit_value(short_outcome.exit_reason),
                "short_first_hit_bar": _first_hit_bar(short_outcome.exit_reason, short_outcome.holding_bars),
                "short_ambiguous": short_outcome.ambiguous,
                "short_excluded": short_outcome.excluded,
                "short_gross_return": short_outcome.gross_return,
                "short_fee_return": short_outcome.fee_return,
                "short_funding_return": short_outcome.funding_return,
                "short_net_return": short_outcome.net_return,
                "ambiguous_bar": long_outcome.ambiguous or short_outcome.ambiguous,
            }
        )
    return pd.DataFrame(rows)


def _resolved_funding_rates(bars: pd.DataFrame, spec: StrategySpec) -> np.ndarray | None:
    """Return explicit per-bar settlement rates, or None for a constant fallback."""

    used = slice(spec.entry_delay_bars, None)
    if "funding_payment_rate" not in bars.columns:
        if "funding_rate" in bars.columns and bars["funding_rate"].iloc[used].notna().any():
            raise ValueError(
                "funding_rate is ambiguous for account settlement; provide "
                "funding_payment_rate only on actual payment bars."
            )
        return None

    raw = bars["funding_payment_rate"]
    numeric = pd.to_numeric(raw, errors="coerce")
    if (raw.notna() & numeric.isna()).iloc[used].any():
        raise ValueError("funding_payment_rate must be numeric when provided.")
    if "funding_rate" in bars.columns:
        ambiguous_fallback = numeric.isna() & bars["funding_rate"].notna()
        if ambiguous_fallback.iloc[used].any():
            raise ValueError(
                "funding_rate is ambiguous for account settlement; provide "
                "funding_payment_rate only on actual payment bars."
            )
    resolved = numeric.fillna(float(spec.costs.funding_rate_per_bar)).to_numpy(dtype=float)
    if not np.isfinite(resolved[used]).all():
        raise ValueError("funding_payment_rate must be finite when provided.")
    return resolved


def _vectorized_side_outcomes(
    *,
    open_path: np.ndarray,
    high_path: np.ndarray,
    low_path: np.ndarray,
    close_path: np.ndarray,
    funding_path: np.ndarray | None,
    entry_raw: np.ndarray,
    side: Side,
    spec: StrategySpec,
) -> dict[str, object]:
    impact = spec.costs.half_spread_rate + spec.costs.slippage_rate
    if side == "long":
        entry_fill = entry_raw * (1.0 + impact)
        target = entry_fill * (1.0 + spec.take_profit_return)
        stop = entry_fill * (1.0 - spec.stop_loss_return)
        gap_stop = open_path <= stop[:, None]
        gap_target = (~gap_stop) & (open_path >= target[:, None])
        stop_hit = low_path <= stop[:, None]
        target_hit = high_path >= target[:, None]
    else:
        entry_fill = entry_raw * (1.0 - impact)
        target = entry_fill * (1.0 - spec.take_profit_return)
        stop = entry_fill * (1.0 + spec.stop_loss_return)
        gap_stop = open_path >= stop[:, None]
        gap_target = (~gap_stop) & (open_path <= target[:, None])
        stop_hit = high_path >= stop[:, None]
        target_hit = low_path <= target[:, None]

    no_gap = ~(gap_stop | gap_target)
    ambiguous_event = no_gap & stop_hit & target_hit
    stop_event = gap_stop | (no_gap & stop_hit)
    target_event = gap_target | (no_gap & ~stop_hit & target_hit)
    event = stop_event | target_event
    has_event = event.any(axis=1)
    first_offset = np.argmax(event, axis=1)
    row_indices = np.arange(len(entry_raw), dtype=np.intp)
    holding_bars = np.where(has_event, first_offset + 1, spec.max_holding_bars).astype(np.int64)

    event_raw = np.where(
        gap_stop,
        open_path,
        np.where(
            gap_target,
            target[:, None],
            np.where(stop_event, stop[:, None], target[:, None]),
        ),
    )
    first_event_raw = event_raw[row_indices, first_offset]
    exit_raw = np.where(has_event, first_event_raw, close_path[:, -1])
    first_stop = stop_event[row_indices, first_offset]
    first_target = target_event[row_indices, first_offset]
    first_ambiguous = ambiguous_event[row_indices, first_offset] & has_event

    default_reason = (
        "horizon"
        if spec.max_holding_bars == spec.horizon_bars - spec.entry_delay_bars + 1
        else "time_exit"
    )
    exit_reason = np.full(len(entry_raw), default_reason, dtype=object)
    exit_reason[has_event & first_stop] = "stop_loss"
    exit_reason[has_event & first_target] = "take_profit"
    excluded = first_ambiguous & (spec.ambiguous_bar_policy.value == "exclude")
    exit_reason[excluded] = "ambiguous_excluded"

    if side == "long":
        exit_fill = exit_raw * (1.0 - impact)
        gross_return = (exit_fill - entry_fill) / entry_fill
    else:
        exit_fill = exit_raw * (1.0 + impact)
        gross_return = (entry_fill - exit_fill) / entry_fill
    fee_return = spec.costs.fee_rate * (1.0 + exit_fill / entry_fill)

    if funding_path is None:
        cumulative_funding = holding_bars * float(spec.costs.funding_rate_per_bar)
    else:
        funding_cumulative_path = np.cumsum(funding_path, axis=1)
        cumulative_funding = funding_cumulative_path[row_indices, holding_bars - 1]
    funding_return = cumulative_funding if side == "long" else -cumulative_funding
    net_return = gross_return - fee_return - funding_return
    net_return = net_return.astype(float, copy=False)
    net_return[excluded] = np.nan

    tp_before_sl = np.empty(len(entry_raw), dtype=object)
    tp_before_sl[:] = pd.NA
    tp_before_sl[exit_reason == "take_profit"] = True
    tp_before_sl[exit_reason == "stop_loss"] = False
    first_hit_bar = np.empty(len(entry_raw), dtype=object)
    first_hit_bar[:] = pd.NA
    hit = np.isin(exit_reason, ["take_profit", "stop_loss", "ambiguous_excluded"])
    first_hit_bar[hit] = holding_bars[hit]
    if not pd.isna(tp_before_sl).any():
        tp_before_sl = tp_before_sl.astype(bool)
    if not pd.isna(first_hit_bar).any():
        first_hit_bar = first_hit_bar.astype(np.int64)
    return {
        "entry_fill": entry_fill,
        "exit_raw": exit_raw,
        "exit_fill": exit_fill,
        "exit_reason": exit_reason,
        "holding_bars": holding_bars,
        "tp_before_sl": tp_before_sl,
        "first_hit_bar": first_hit_bar,
        "ambiguous": first_ambiguous,
        "excluded": excluded,
        "gross_return": gross_return,
        "fee_return": fee_return,
        "funding_return": funding_return,
        "net_return": net_return,
    }


def make_path_targets(df: pd.DataFrame, spec: StrategySpec) -> pd.DataFrame:
    """Build v1 path targets with vectorized bounded-horizon simulation."""

    bars = validate_bars(df)
    validate_bar_cadence(bars, spec.base_interval_minutes)
    if len(bars) < spec.horizon_bars + 1:
        raise ValueError("Not enough rows for the configured target horizon.")

    n_samples = len(bars) - spec.horizon_bars
    signal_index = np.arange(n_samples, dtype=np.int64)
    entry_index = signal_index + spec.entry_delay_bars
    horizon_end_index = signal_index + spec.horizon_bars
    full_path_bars = spec.horizon_bars - spec.entry_delay_bars + 1
    holding_bars = spec.max_holding_bars

    open_values = bars["open"].to_numpy(dtype=float)
    high_values = bars["high"].to_numpy(dtype=float)
    low_values = bars["low"].to_numpy(dtype=float)
    close_values = bars["close"].to_numpy(dtype=float)

    def path_windows(values: np.ndarray, width: int) -> np.ndarray:
        windows = np.lib.stride_tricks.sliding_window_view(values, width)
        return windows[spec.entry_delay_bars : spec.entry_delay_bars + n_samples]

    full_high_path = path_windows(high_values, full_path_bars)
    full_low_path = path_windows(low_values, full_path_bars)
    open_path = path_windows(open_values, holding_bars)
    high_path = path_windows(high_values, holding_bars)
    low_path = path_windows(low_values, holding_bars)
    close_path = path_windows(close_values, holding_bars)
    entry_raw = open_values[entry_index]

    funding_rates = _resolved_funding_rates(bars, spec)
    funding_path = None if funding_rates is None else path_windows(funding_rates, holding_bars)
    long_outcome = _vectorized_side_outcomes(
        open_path=open_path,
        high_path=high_path,
        low_path=low_path,
        close_path=close_path,
        funding_path=funding_path,
        entry_raw=entry_raw,
        side="long",
        spec=spec,
    )
    short_outcome = _vectorized_side_outcomes(
        open_path=open_path,
        high_path=high_path,
        low_path=low_path,
        close_path=close_path,
        funding_path=funding_path,
        entry_raw=entry_raw,
        side="short",
        spec=spec,
    )

    timestamps = bars["timestamp"]
    long_exit_index = entry_index + np.asarray(long_outcome["holding_bars"], dtype=np.int64) - 1
    short_exit_index = entry_index + np.asarray(short_outcome["holding_bars"], dtype=np.int64) - 1
    return pd.DataFrame(
        {
            "signal_index": signal_index,
            "signal_time": timestamps.iloc[signal_index].reset_index(drop=True),
            "entry_index": entry_index,
            "entry_time": timestamps.iloc[entry_index].reset_index(drop=True),
            "horizon_end_index": horizon_end_index,
            "horizon_end_time": timestamps.iloc[horizon_end_index].reset_index(drop=True),
            "entry_price": entry_raw,
            "return_h": np.log(close_values[horizon_end_index] / entry_raw),
            "mfe_h": full_high_path.max(axis=1) / entry_raw - 1.0,
            "mae_h": full_low_path.min(axis=1) / entry_raw - 1.0,
            "long_entry_price": long_outcome["entry_fill"],
            "long_exit_index": long_exit_index,
            "long_exit_time": timestamps.iloc[long_exit_index].reset_index(drop=True),
            "long_exit_price": long_outcome["exit_fill"],
            "long_exit_reason": long_outcome["exit_reason"],
            "long_holding_bars": long_outcome["holding_bars"],
            "long_tp_before_sl": long_outcome["tp_before_sl"],
            "long_first_hit_bar": long_outcome["first_hit_bar"],
            "long_ambiguous": long_outcome["ambiguous"],
            "long_excluded": long_outcome["excluded"],
            "long_gross_return": long_outcome["gross_return"],
            "long_fee_return": long_outcome["fee_return"],
            "long_funding_return": long_outcome["funding_return"],
            "long_net_return": long_outcome["net_return"],
            "short_entry_price": short_outcome["entry_fill"],
            "short_exit_index": short_exit_index,
            "short_exit_time": timestamps.iloc[short_exit_index].reset_index(drop=True),
            "short_exit_price": short_outcome["exit_fill"],
            "short_exit_reason": short_outcome["exit_reason"],
            "short_holding_bars": short_outcome["holding_bars"],
            "short_tp_before_sl": short_outcome["tp_before_sl"],
            "short_first_hit_bar": short_outcome["first_hit_bar"],
            "short_ambiguous": short_outcome["ambiguous"],
            "short_excluded": short_outcome["excluded"],
            "short_gross_return": short_outcome["gross_return"],
            "short_fee_return": short_outcome["fee_return"],
            "short_funding_return": short_outcome["funding_return"],
            "short_net_return": short_outcome["net_return"],
            "ambiguous_bar": np.asarray(long_outcome["ambiguous"], dtype=bool)
            | np.asarray(short_outcome["ambiguous"], dtype=bool),
        }
    )

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


@dataclass(frozen=True)
class TradeOutcomeConfig:
    """Configuration for labels aligned with the account backtest.

    Prices use the current bar close as the entry, matching the current
    account-backtest implementation. Future bars are used only for training
    labels.
    """

    horizon: int = 240
    leverage: float = 20.0
    take_profit_rate: float = 0.38
    stop_loss_rate: float = 0.28
    fee_rate: float = 0.0005
    min_net_profit: float = 0.20

    @property
    def take_profit_move(self) -> float:
        return self.take_profit_rate / self.leverage

    @property
    def stop_loss_move(self) -> float:
        return self.stop_loss_rate / self.leverage


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


def _simulate_side_outcome(
    future: pd.DataFrame,
    entry: float,
    side: str,
    config: TradeOutcomeConfig,
) -> tuple[float, str, int, float]:
    position_size = config.leverage
    if side == "long":
        target = entry * (1 + config.take_profit_move)
        stop = entry * (1 - config.stop_loss_move)
    else:
        target = entry * (1 - config.take_profit_move)
        stop = entry * (1 + config.stop_loss_move)

    close_price = float(future["close"].iloc[-1])
    reason = "horizon"
    holding_bars = len(future)
    for offset, row in enumerate(future.itertuples(index=False), start=1):
        high = float(row.high)
        low = float(row.low)
        if side == "long":
            if low <= stop:
                close_price, reason, holding_bars = stop, "stop_loss", offset
                break
            if high >= target:
                close_price, reason, holding_bars = target, "take_profit", offset
                break
        else:
            if high >= stop:
                close_price, reason, holding_bars = stop, "stop_loss", offset
                break
            if low <= target:
                close_price, reason, holding_bars = target, "take_profit", offset
                break

    gross = (close_price - entry) * position_size / entry
    if side == "short":
        gross = -gross
    fees = position_size * config.fee_rate + close_price * position_size / entry * config.fee_rate
    return float(gross - fees), reason, holding_bars, close_price


def make_trade_outcome_labels(df: pd.DataFrame, config: TradeOutcomeConfig | None = None) -> pd.DataFrame:
    """Create trade/flat plus long/short labels from executable TP/SL outcomes.

    A row is labelled with the side whose simulated net profit is larger,
    provided it clears ``min_net_profit``. Otherwise it is labelled flat. This
    better matches the account backtest than the older ATR triple-barrier GT.
    """

    config = config or TradeOutcomeConfig()
    if len(df) < config.horizon + 1:
        raise ValueError("Not enough rows to create trade outcome labels for the requested horizon.")

    out = df.copy()
    out["label"] = "flat"
    out["label_code"] = 1
    out["label_reason"] = "no_trade_edge"
    out["trade_label"] = 0
    out["side_label"] = pd.NA
    out["long_net_profit"] = pd.NA
    out["short_net_profit"] = pd.NA
    out["best_net_profit"] = pd.NA
    out["best_holding_bars"] = pd.NA

    last_label_idx = len(out) - config.horizon - 1
    n = last_label_idx + 1
    close = pd.to_numeric(out["close"], errors="coerce").to_numpy(dtype=float)
    high = pd.to_numeric(out["high"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(out["low"], errors="coerce").to_numpy(dtype=float)
    entry = close[:n]

    long_target = entry * (1 + config.take_profit_move)
    long_stop = entry * (1 - config.stop_loss_move)
    short_target = entry * (1 - config.take_profit_move)
    short_stop = entry * (1 + config.stop_loss_move)

    long_close = close[np.arange(n) + config.horizon]
    short_close = long_close.copy()
    long_reason = np.full(n, "horizon", dtype=object)
    short_reason = np.full(n, "horizon", dtype=object)
    long_bars = np.full(n, config.horizon, dtype=int)
    short_bars = np.full(n, config.horizon, dtype=int)
    long_open = np.isfinite(entry) & (entry > 0)
    short_open = long_open.copy()

    for offset in range(1, config.horizon + 1):
        future_high = high[offset : offset + n]
        future_low = low[offset : offset + n]

        long_stop_hit = long_open & (future_low <= long_stop)
        long_close[long_stop_hit] = long_stop[long_stop_hit]
        long_reason[long_stop_hit] = "stop_loss"
        long_bars[long_stop_hit] = offset
        long_open[long_stop_hit] = False

        long_target_hit = long_open & (future_high >= long_target)
        long_close[long_target_hit] = long_target[long_target_hit]
        long_reason[long_target_hit] = "take_profit"
        long_bars[long_target_hit] = offset
        long_open[long_target_hit] = False

        short_stop_hit = short_open & (future_high >= short_stop)
        short_close[short_stop_hit] = short_stop[short_stop_hit]
        short_reason[short_stop_hit] = "stop_loss"
        short_bars[short_stop_hit] = offset
        short_open[short_stop_hit] = False

        short_target_hit = short_open & (future_low <= short_target)
        short_close[short_target_hit] = short_target[short_target_hit]
        short_reason[short_target_hit] = "take_profit"
        short_bars[short_target_hit] = offset
        short_open[short_target_hit] = False

    position_size = config.leverage
    long_gross = (long_close - entry) * position_size / entry
    short_gross = -(short_close - entry) * position_size / entry
    long_fees = position_size * config.fee_rate + long_close * position_size / entry * config.fee_rate
    short_fees = position_size * config.fee_rate + short_close * position_size / entry * config.fee_rate
    long_net = long_gross - long_fees
    short_net = short_gross - short_fees
    invalid = ~np.isfinite(entry) | (entry <= 0)
    long_net[invalid] = np.nan
    short_net[invalid] = np.nan

    long_is_best = np.nan_to_num(long_net, nan=-np.inf) >= np.nan_to_num(short_net, nan=-np.inf)
    best_net = np.where(long_is_best, long_net, short_net)
    trade_mask = np.isfinite(best_net) & (best_net > config.min_net_profit)

    idx = out.index[:n]
    out.loc[idx, "long_net_profit"] = long_net
    out.loc[idx, "short_net_profit"] = short_net
    out.loc[idx, "best_net_profit"] = best_net
    out.loc[idx, "best_holding_bars"] = np.where(long_is_best, long_bars, short_bars)
    out.loc[idx[trade_mask & long_is_best], "label"] = "long"
    out.loc[idx[trade_mask & long_is_best], "label_code"] = 2
    out.loc[idx[trade_mask & long_is_best], "label_reason"] = long_reason[trade_mask & long_is_best]
    out.loc[idx[trade_mask & long_is_best], "trade_label"] = 1
    out.loc[idx[trade_mask & long_is_best], "side_label"] = 1
    out.loc[idx[trade_mask & ~long_is_best], "label"] = "short"
    out.loc[idx[trade_mask & ~long_is_best], "label_code"] = 0
    out.loc[idx[trade_mask & ~long_is_best], "label_reason"] = short_reason[trade_mask & ~long_is_best]
    out.loc[idx[trade_mask & ~long_is_best], "trade_label"] = 1
    out.loc[idx[trade_mask & ~long_is_best], "side_label"] = 0

    return out.loc[out.index <= last_label_idx].reset_index(drop=True)


def label_summary(labels: pd.DataFrame) -> pd.DataFrame:
    summary = labels["label"].value_counts(dropna=False).rename_axis("label").reset_index(name="count")
    summary["share"] = summary["count"] / summary["count"].sum()
    return summary

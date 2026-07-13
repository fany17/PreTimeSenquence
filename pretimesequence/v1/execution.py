from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .spec import AmbiguousBarPolicy, CostSpec, StrategySpec


Side = Literal["long", "short"]
OrderSide = Literal["buy", "sell"]

REQUIRED_BAR_COLUMNS = ("timestamp", "open", "high", "low", "close")


@dataclass(frozen=True)
class BarrierLevels:
    target: float
    stop: float


@dataclass(frozen=True)
class ExitTrigger:
    raw_price: float
    reason: str
    ambiguous: bool = False
    excluded: bool = False


@dataclass(frozen=True)
class ReturnBreakdown:
    gross_return: float
    fee_return: float
    funding_return: float
    net_return: float


def validate_bars(df: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_BAR_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required bar columns: {missing}")
    if df.empty:
        raise ValueError("Bars must not be empty.")

    bars = df.copy().reset_index(drop=True)
    for value in bars["timestamp"]:
        if pd.isna(value):
            raise ValueError("timestamp values must be valid and timezone-aware.")
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("timestamp values must be timezone-aware.")
    timestamps = pd.to_datetime(bars["timestamp"], utc=True, errors="coerce")
    if timestamps.isna().any():
        raise ValueError("timestamp values must be valid and timezone-normalizable.")
    if timestamps.duplicated().any():
        raise ValueError("timestamp values must be unique.")
    if not timestamps.is_monotonic_increasing:
        raise ValueError("timestamp values must be sorted in increasing order.")
    bars["timestamp"] = timestamps

    for column in ("open", "high", "low", "close"):
        bars[column] = pd.to_numeric(bars[column], errors="coerce")
    prices = bars.loc[:, ["open", "high", "low", "close"]].to_numpy(dtype=float)
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("OHLC prices must be finite and strictly positive.")
    invalid_ohlc = (
        (bars["high"] < bars[["open", "close"]].max(axis=1))
        | (bars["low"] > bars[["open", "close"]].min(axis=1))
        | (bars["high"] < bars["low"])
    )
    if invalid_ohlc.any():
        raise ValueError("OHLC relationships are invalid.")
    return bars


def validate_bar_cadence(bars: pd.DataFrame, base_interval_minutes: int) -> None:
    if len(bars) < 2:
        return
    expected = pd.Timedelta(minutes=base_interval_minutes)
    differences = bars["timestamp"].diff().iloc[1:]
    if not differences.eq(expected).all():
        bad_count = int((differences != expected).sum())
        raise ValueError(
            f"Bar cadence must be exactly {base_interval_minutes} minute(s); "
            f"found {bad_count} gap or irregular interval(s)."
        )


def market_fill_price(raw_price: float, order_side: OrderSide, costs: CostSpec) -> float:
    impact = costs.half_spread_rate + costs.slippage_rate
    multiplier = 1 + impact if order_side == "buy" else 1 - impact
    fill = float(raw_price) * multiplier
    if fill <= 0:
        raise ValueError("Execution costs produced a non-positive fill price.")
    return fill


def entry_fill_price(raw_price: float, side: Side, costs: CostSpec) -> float:
    return market_fill_price(raw_price, "buy" if side == "long" else "sell", costs)


def exit_fill_price(raw_price: float, side: Side, costs: CostSpec) -> float:
    return market_fill_price(raw_price, "sell" if side == "long" else "buy", costs)


def barrier_levels(entry_fill: float, side: Side, spec: StrategySpec) -> BarrierLevels:
    if side == "long":
        return BarrierLevels(
            target=entry_fill * (1 + spec.take_profit_return),
            stop=entry_fill * (1 - spec.stop_loss_return),
        )
    return BarrierLevels(
        target=entry_fill * (1 - spec.take_profit_return),
        stop=entry_fill * (1 + spec.stop_loss_return),
    )


def detect_barrier_exit(
    bar: pd.Series,
    side: Side,
    levels: BarrierLevels,
    policy: AmbiguousBarPolicy,
) -> ExitTrigger | None:
    open_price = float(bar["open"])
    high = float(bar["high"])
    low = float(bar["low"])

    if side == "long":
        if open_price <= levels.stop:
            return ExitTrigger(raw_price=open_price, reason="stop_loss")
        if open_price >= levels.target:
            return ExitTrigger(raw_price=levels.target, reason="take_profit")
        stop_hit = low <= levels.stop
        target_hit = high >= levels.target
    else:
        if open_price >= levels.stop:
            return ExitTrigger(raw_price=open_price, reason="stop_loss")
        if open_price <= levels.target:
            return ExitTrigger(raw_price=levels.target, reason="take_profit")
        stop_hit = high >= levels.stop
        target_hit = low <= levels.target

    if stop_hit and target_hit:
        if policy == AmbiguousBarPolicy.EXCLUDE:
            return ExitTrigger(
                raw_price=levels.stop,
                reason="ambiguous_excluded",
                ambiguous=True,
                excluded=True,
            )
        return ExitTrigger(raw_price=levels.stop, reason="stop_loss", ambiguous=True)
    if stop_hit:
        return ExitTrigger(raw_price=levels.stop, reason="stop_loss")
    if target_hit:
        return ExitTrigger(raw_price=levels.target, reason="take_profit")
    return None


def funding_rate_for_bar(bar: pd.Series, costs: CostSpec) -> float:
    if "funding_payment_rate" not in bar.index or pd.isna(bar["funding_payment_rate"]):
        if "funding_rate" in bar.index and not pd.isna(bar["funding_rate"]):
            raise ValueError(
                "funding_rate is ambiguous for account settlement; provide "
                "funding_payment_rate only on actual payment bars."
            )
        return float(costs.funding_rate_per_bar)
    value = float(bar["funding_payment_rate"])
    if not np.isfinite(value):
        raise ValueError("funding_payment_rate must be finite when provided.")
    return value


def calculate_return_breakdown(
    *,
    side: Side,
    entry_fill: float,
    exit_fill: float,
    fee_rate: float,
    cumulative_funding_rate: float,
) -> ReturnBreakdown:
    price_return = (exit_fill - entry_fill) / entry_fill
    gross_return = price_return if side == "long" else -price_return
    fee_return = fee_rate * (1 + exit_fill / entry_fill)
    funding_return = cumulative_funding_rate if side == "long" else -cumulative_funding_rate
    return ReturnBreakdown(
        gross_return=float(gross_return),
        fee_return=float(fee_return),
        funding_return=float(funding_return),
        net_return=float(gross_return - fee_return - funding_return),
    )

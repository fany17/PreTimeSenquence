from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .execution import (
    BarrierLevels,
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
from .spec import FoldEndPolicy, StrategySpec


TRADE_COLUMNS = [
    "signal_index",
    "signal_time",
    "entry_index",
    "entry_time",
    "exit_index",
    "exit_time",
    "side",
    "entry_raw_price",
    "entry_fill_price",
    "exit_raw_price",
    "exit_fill_price",
    "exit_reason",
    "holding_bars",
    "ambiguous_bar",
    "notional",
    "gross_profit",
    "fees",
    "funding",
    "net_profit",
    "gross_return",
    "net_return",
    "start_equity",
    "end_equity",
]


@dataclass(frozen=True)
class EventBacktestResult:
    trades: pd.DataFrame
    equity_curve: pd.DataFrame


@dataclass(frozen=True)
class _PendingOrder:
    signal_index: int
    entry_index: int
    side: Side


@dataclass
class _Position:
    signal_index: int
    entry_index: int
    side: Side
    entry_raw_price: float
    entry_fill_price: float
    levels: BarrierLevels
    notional: float
    start_equity: float
    holding_bars: int = 0
    cumulative_funding_rate: float = 0.0


def _decision_map(decisions: pd.DataFrame, n_bars: int) -> dict[int, Side | str]:
    required = {"signal_index", "action"}
    missing = required - set(decisions.columns)
    if missing:
        raise ValueError(f"Missing decision columns: {sorted(missing)}")
    frame = decisions.loc[:, ["signal_index", "action"]].copy()
    numeric_indices = pd.to_numeric(frame["signal_index"], errors="coerce")
    if numeric_indices.isna().any() or ((numeric_indices % 1) != 0).any():
        raise ValueError("signal_index values must be integers.")
    frame["signal_index"] = numeric_indices.astype(int)
    if frame["signal_index"].duplicated().any():
        raise ValueError("Each signal_index may have at most one decision.")
    if ((frame["signal_index"] < 0) | (frame["signal_index"] >= n_bars)).any():
        raise ValueError("signal_index is outside the bar range.")
    frame["action"] = frame["action"].astype(str).str.lower()
    allowed = {"long", "short", "flat"}
    unknown = sorted(set(frame["action"]) - allowed)
    if unknown:
        raise ValueError(f"Unknown decision actions: {unknown}")
    return dict(zip(frame["signal_index"], frame["action"]))


def run_event_backtest(
    bars: pd.DataFrame,
    decisions: pd.DataFrame,
    spec: StrategySpec,
    *,
    initial_equity: float = 1.0,
    notional_fraction: float | None = None,
    fold_start_index: int = 0,
    fold_end_index: int | None = None,
) -> EventBacktestResult:
    """Run a sequential single-symbol, single-position OHLC backtest.

    Decisions are formed at ``signal_index`` close. Orders become eligible only
    after ``entry_delay_bars`` and fill at that bar's open. The account state is
    updated immediately after every TP, SL, time exit or fold-end exit.
    """

    frame = validate_bars(bars)
    validate_bar_cadence(frame, spec.base_interval_minutes)
    action_by_index = _decision_map(decisions, len(frame))
    if not np.isfinite(initial_equity) or initial_equity <= 0:
        raise ValueError("initial_equity must be finite and positive.")
    resolved_notional_fraction = spec.default_notional_fraction if notional_fraction is None else notional_fraction
    if not np.isfinite(resolved_notional_fraction) or not 0 < resolved_notional_fraction <= spec.leverage_cap:
        raise ValueError("notional_fraction must be positive and no greater than leverage_cap.")

    end_index = len(frame) - 1 if fold_end_index is None else fold_end_index
    if not 0 <= fold_start_index <= end_index < len(frame):
        raise ValueError("Invalid fold_start_index/fold_end_index boundaries.")
    if spec.fold_end_policy != FoldEndPolicy.FORCE_CLOSE:
        raise ValueError(f"Unsupported fold_end_policy: {spec.fold_end_policy}")

    realized_equity = float(initial_equity)
    pending: _PendingOrder | None = None
    position: _Position | None = None
    trade_rows: list[dict[str, object]] = []
    equity_rows: list[dict[str, object]] = []

    def close_position(exit_index: int, exit_raw: float, reason: str, ambiguous: bool = False) -> None:
        nonlocal position, realized_equity
        if position is None:
            raise RuntimeError("Cannot close a missing position.")
        exit_fill = exit_fill_price(exit_raw, position.side, spec.costs)
        breakdown = calculate_return_breakdown(
            side=position.side,
            entry_fill=position.entry_fill_price,
            exit_fill=exit_fill,
            fee_rate=spec.costs.fee_rate,
            cumulative_funding_rate=position.cumulative_funding_rate,
        )
        net_profit = position.notional * breakdown.net_return
        end_equity = realized_equity + net_profit
        trade_rows.append(
            {
                "signal_index": position.signal_index,
                "signal_time": frame.iloc[position.signal_index]["timestamp"],
                "entry_index": position.entry_index,
                "entry_time": frame.iloc[position.entry_index]["timestamp"],
                "exit_index": exit_index,
                "exit_time": frame.iloc[exit_index]["timestamp"],
                "side": position.side,
                "entry_raw_price": position.entry_raw_price,
                "entry_fill_price": position.entry_fill_price,
                "exit_raw_price": float(exit_raw),
                "exit_fill_price": exit_fill,
                "exit_reason": reason,
                "holding_bars": position.holding_bars,
                "ambiguous_bar": ambiguous,
                "notional": position.notional,
                "gross_profit": position.notional * breakdown.gross_return,
                "fees": position.notional * breakdown.fee_return,
                "funding": position.notional * breakdown.funding_return,
                "net_profit": net_profit,
                "gross_return": breakdown.gross_return,
                "net_return": breakdown.net_return,
                "start_equity": position.start_equity,
                "end_equity": end_equity,
            }
        )
        realized_equity = end_equity
        position = None

    for bar_index in range(fold_start_index, end_index + 1):
        bar = frame.iloc[bar_index]

        if pending is not None and pending.entry_index == bar_index:
            if position is not None:
                raise RuntimeError("Pending order reached entry while another position was open.")
            raw_entry = float(bar["open"])
            fill_entry = entry_fill_price(raw_entry, pending.side, spec.costs)
            position = _Position(
                signal_index=pending.signal_index,
                entry_index=bar_index,
                side=pending.side,
                entry_raw_price=raw_entry,
                entry_fill_price=fill_entry,
                levels=barrier_levels(fill_entry, pending.side, spec),
                notional=realized_equity * resolved_notional_fraction,
                start_equity=realized_equity,
            )
            pending = None

        if position is not None:
            position.holding_bars += 1
            position.cumulative_funding_rate += funding_rate_for_bar(bar, spec.costs)
            trigger = detect_barrier_exit(
                bar,
                position.side,
                position.levels,
                spec.ambiguous_bar_policy,
            )
            if trigger is not None:
                if trigger.excluded:
                    raise ValueError(
                        "AmbiguousBarPolicy.EXCLUDE cannot produce an account PnL; "
                        "use CONSERVATIVE for event backtests."
                    )
                close_position(bar_index, trigger.raw_price, trigger.reason, trigger.ambiguous)
            elif position.holding_bars >= spec.max_holding_bars:
                close_position(bar_index, float(bar["close"]), "time_exit")

        if bar_index == end_index and position is not None:
            close_position(bar_index, float(bar["close"]), "fold_end")

        if bar_index < end_index and position is None and pending is None:
            action = action_by_index.get(bar_index, "flat")
            entry_index = bar_index + spec.entry_delay_bars
            if action in {"long", "short"} and entry_index <= end_index:
                pending = _PendingOrder(
                    signal_index=bar_index,
                    entry_index=entry_index,
                    side=action,  # type: ignore[arg-type]
                )

        if position is None:
            marked_equity = realized_equity
            state = "pending" if pending is not None else "flat"
        else:
            marked_exit = exit_fill_price(float(bar["close"]), position.side, spec.costs)
            marked = calculate_return_breakdown(
                side=position.side,
                entry_fill=position.entry_fill_price,
                exit_fill=marked_exit,
                fee_rate=spec.costs.fee_rate,
                cumulative_funding_rate=position.cumulative_funding_rate,
            )
            marked_equity = realized_equity + position.notional * marked.net_return
            state = position.side
        equity_rows.append(
            {
                "bar_index": bar_index,
                "timestamp": bar["timestamp"],
                "realized_equity": realized_equity,
                "equity": marked_equity,
                "state": state,
            }
        )

    trades = pd.DataFrame(trade_rows, columns=TRADE_COLUMNS)
    equity_curve = pd.DataFrame(equity_rows)
    return EventBacktestResult(trades=trades, equity_curve=equity_curve)

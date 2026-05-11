from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .model import TrendPredictor
from .strategy import attach_trade_signals


@dataclass(frozen=True)
class BacktestConfig:
    leverage: float = 20.0
    margin: float = 1.0
    take_profit_rate: float = 0.38
    stop_loss_rate: float = 0.28
    fee_rate: float = 0.0005
    min_confidence: float = 0.55
    initial_balance: float = 1.0


def run_backtest(df: pd.DataFrame, predictor: TrendPredictor, config: BacktestConfig | None = None) -> pd.DataFrame:
    config = config or BacktestConfig()
    predicted = predictor.predict_frame(df)
    predicted.loc[predicted["trend_score"] < config.min_confidence, "trend"] = "flat"
    signals = attach_trade_signals(predicted)
    trades: list[dict] = []

    for open_idx, row in signals[signals["action"].str.startswith(("open_", "flip_to_"), na=False)].iterrows():
        side = "long" if row["action"].endswith("long") else "short"
        open_price = float(row["close"])
        position_size = config.margin * config.leverage
        if side == "long":
            target = open_price * (1 + config.take_profit_rate / config.leverage)
            stop = open_price * (1 - config.stop_loss_rate / config.leverage)
        else:
            target = open_price * (1 - config.take_profit_rate / config.leverage)
            stop = open_price * (1 + config.stop_loss_rate / config.leverage)

        close_idx = len(df) - 1
        close_price = float(df.loc[close_idx, "close"])
        reason = "end"
        for idx in range(open_idx + 1, len(df)):
            high = float(df.loc[idx, "high"])
            low = float(df.loc[idx, "low"])
            if side == "long" and low <= stop:
                close_idx, close_price, reason = idx, stop, "stop_loss"
                break
            if side == "long" and high >= target:
                close_idx, close_price, reason = idx, target, "take_profit"
                break
            if side == "short" and high >= stop:
                close_idx, close_price, reason = idx, stop, "stop_loss"
                break
            if side == "short" and low <= target:
                close_idx, close_price, reason = idx, target, "take_profit"
                break

        gross = (close_price - open_price) * position_size / open_price
        if side == "short":
            gross = -gross
        fees = position_size * config.fee_rate + close_price * position_size / open_price * config.fee_rate
        net = max(gross - fees, -config.margin)
        trades.append(
            {
                "open_time": row["timestamp"],
                "close_time": df.loc[close_idx, "timestamp"],
                "side": side,
                "open_price": open_price,
                "close_price": close_price,
                "reason": reason,
                "net_profit": net,
            }
        )

    result = pd.DataFrame(trades)
    if not result.empty:
        result["cum_profit"] = result["net_profit"].cumsum()
    return result


def run_account_backtest(
    df: pd.DataFrame,
    predictor: TrendPredictor,
    config: BacktestConfig | None = None,
    start_time: str | pd.Timestamp | None = None,
    end_time: str | pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sequential account simulation.

    Uses full df for indicators and future exits, but only opens new trades whose
    signal timestamp is inside [start_time, end_time]. At most one position is
    open at a time.
    """

    config = config or BacktestConfig()
    predicted = predictor.predict_frame(df).copy()
    predicted["timestamp"] = pd.to_datetime(predicted["timestamp"])
    predicted.loc[predicted["trend_score"] < config.min_confidence, "trend"] = "flat"
    signals = attach_trade_signals(predicted)

    start_ts = pd.Timestamp(start_time) if start_time else signals["timestamp"].min()
    end_ts = pd.Timestamp(end_time) if end_time else signals["timestamp"].max()
    balance = float(config.initial_balance)
    cursor = 0
    trades: list[dict] = []

    while cursor < len(signals) - 1 and balance > 0:
        row = signals.iloc[cursor]
        ts = pd.Timestamp(row["timestamp"])
        if ts < start_ts:
            cursor += 1
            continue
        if ts > end_ts:
            break
        action = str(row.get("action", "hold"))
        if not action.startswith(("open_", "flip_to_")):
            cursor += 1
            continue

        side = "long" if action.endswith("long") else "short"
        open_idx = int(row.name)
        open_price = float(row["close"])
        margin = min(float(config.margin), balance)
        position_size = margin * config.leverage
        fee_open = position_size * config.fee_rate
        if fee_open >= balance:
            break

        if side == "long":
            target = open_price * (1 + config.take_profit_rate / config.leverage)
            stop = open_price * (1 - config.stop_loss_rate / config.leverage)
        else:
            target = open_price * (1 - config.take_profit_rate / config.leverage)
            stop = open_price * (1 + config.stop_loss_rate / config.leverage)

        close_idx = len(signals) - 1
        close_price = float(signals.iloc[close_idx]["close"])
        reason = "end"
        for idx in range(open_idx + 1, len(signals)):
            high = float(signals.iloc[idx]["high"])
            low = float(signals.iloc[idx]["low"])
            if side == "long" and low <= stop:
                close_idx, close_price, reason = idx, stop, "stop_loss"
                break
            if side == "long" and high >= target:
                close_idx, close_price, reason = idx, target, "take_profit"
                break
            if side == "short" and high >= stop:
                close_idx, close_price, reason = idx, stop, "stop_loss"
                break
            if side == "short" and low <= target:
                close_idx, close_price, reason = idx, target, "take_profit"
                break

        gross = (close_price - open_price) * position_size / open_price
        if side == "short":
            gross = -gross
        fee_close = close_price * position_size / open_price * config.fee_rate
        net = max(gross - fee_open - fee_close, -balance)
        start_balance = balance
        balance = max(0.0, balance + net)
        close_row = signals.iloc[close_idx]
        trades.append(
            {
                "open_time": row["timestamp"],
                "close_time": close_row["timestamp"],
                "side": side,
                "action": action,
                "confidence": float(row["trend_score"]),
                "open_price": open_price,
                "close_price": close_price,
                "reason": reason,
                "margin": margin,
                "leverage": config.leverage,
                "gross_profit": gross,
                "fees": fee_open + fee_close,
                "net_profit": net,
                "start_balance": start_balance,
                "end_balance": balance,
                "return_pct": net / start_balance if start_balance else 0.0,
            }
        )
        cursor = close_idx + 1

    trade_df = pd.DataFrame(trades)
    if trade_df.empty:
        daily = pd.DataFrame(columns=["date", "trades", "net_profit", "end_balance"])
        return trade_df, daily

    trade_df["open_date"] = pd.to_datetime(trade_df["open_time"]).dt.date.astype(str)
    daily = (
        trade_df.groupby("open_date", as_index=False)
        .agg(trades=("net_profit", "size"), net_profit=("net_profit", "sum"), end_balance=("end_balance", "last"))
        .rename(columns={"open_date": "date"})
    )
    return trade_df, daily

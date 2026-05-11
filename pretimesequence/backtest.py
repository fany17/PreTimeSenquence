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

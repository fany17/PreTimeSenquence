from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "MA5",
    "MA20",
    "MA50",
    "MA100",
    "BB_upper",
    "BB_lower",
    "BB_width",
    "log_return",
    "momentum_5",
    "momentum_10",
    "momentum_20",
    "roc_5",
    "roc_10",
    "roc_20",
    "roc_30",
    "roc_60",
    "TSI",
    "MACD",
    "MACD_signal",
    "MACD_hist",
    "stochastic_k",
    "stochastic_d",
    "Tenkan_sen",
    "Kijun_sen",
    "Senkou_Span_A",
    "Senkou_Span_B",
    "PSAR_af",
    "PSAR_reversal",
    "ADX",
    "PDI",
    "NDI",
    "DX",
    "Volume_MA5",
    "Volume_MA20",
    "price_to_volume",
    "volume_acceleration",
    "OBV",
    "OBV_change",
    "rolling_skew",
    "rolling_kurtosis",
    "is_hammer",
    "MA20_lag_1",
    "MA20_lag_2",
    "MA20_lag_3",
    "MA20_lag_4",
    "MA20_lag_5",
]


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def _rma(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["close"]
    high = out["high"]
    low = out["low"]
    open_ = out["open"]
    volume = out["volume"]

    out["log_return"] = np.log(close / close.shift(1))
    for window in (5, 20, 50, 100):
        out[f"MA{window}"] = close.rolling(window=window, min_periods=window).mean()

    out["std_20"] = close.rolling(window=20, min_periods=20).std()
    out["BB_upper"] = out["MA20"] + 2 * out["std_20"]
    out["BB_lower"] = out["MA20"] - 2 * out["std_20"]
    out["BB_width"] = out["BB_upper"] - out["BB_lower"]

    for window in (5, 10, 20):
        out[f"momentum_{window}"] = close - close.shift(window)
    for window in (5, 10, 20, 30, 60):
        out[f"roc_{window}"] = close.pct_change(window)

    momentum = close.diff()
    abs_momentum = momentum.abs()
    out["TSI"] = 100 * (_ema(_ema(momentum, 25), 13) / _ema(_ema(abs_momentum, 25), 13))

    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    out["MACD"] = ema12 - ema26
    out["MACD_signal"] = _ema(out["MACD"], 9)
    out["MACD_hist"] = out["MACD"] - out["MACD_signal"]

    low14 = low.rolling(14, min_periods=14).min()
    high14 = high.rolling(14, min_periods=14).max()
    out["stochastic_k"] = 100 * (close - low14) / (high14 - low14)
    out["stochastic_d"] = out["stochastic_k"].rolling(3, min_periods=3).mean()

    out["Tenkan_sen"] = (high.rolling(9, min_periods=9).max() + low.rolling(9, min_periods=9).min()) / 2
    out["Kijun_sen"] = (high.rolling(26, min_periods=26).max() + low.rolling(26, min_periods=26).min()) / 2
    out["Senkou_Span_A"] = ((out["Tenkan_sen"] + out["Kijun_sen"]) / 2).shift(26)
    out["Senkou_Span_B"] = ((high.rolling(52, min_periods=52).max() + low.rolling(52, min_periods=52).min()) / 2).shift(26)

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=out.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=out.index)
    true_range = pd.concat(
        [(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()],
        axis=1,
    ).max(axis=1)
    atr = _rma(true_range, 14)
    out["PDI"] = 100 * _rma(plus_dm, 14) / atr
    out["NDI"] = 100 * _rma(minus_dm, 14) / atr
    out["DX"] = (out["PDI"] - out["NDI"]).abs() / (out["PDI"] + out["NDI"]) * 100
    out["ADX"] = _rma(out["DX"], 14)

    out["PSAR_af"] = 0.02
    out["PSAR_reversal"] = np.where(close.diff().fillna(0).mul(close.diff().shift().fillna(0)) < 0, 1, 0)

    out["Volume_MA5"] = volume.rolling(5, min_periods=5).mean()
    out["Volume_MA20"] = volume.rolling(20, min_periods=20).mean()
    out["price_to_volume"] = close / (volume + 1e-9)
    out["volume_acceleration"] = volume.diff().diff()

    direction = np.sign(close.diff()).fillna(0)
    out["OBV"] = (direction * volume).cumsum()
    out["OBV_change"] = out["OBV"].diff()
    out["rolling_skew"] = out["log_return"].rolling(20, min_periods=20).skew()
    out["rolling_kurtosis"] = out["log_return"].rolling(20, min_periods=20).kurt()

    body = (close - open_).abs()
    candle_range = high - low
    lower_shadow = np.minimum(close, open_) - low
    out["is_hammer"] = ((lower_shadow > 2 * body) & ((body / candle_range.replace(0, np.nan)) < 0.3)).astype(int)

    for lag in range(1, 6):
        out[f"MA20_lag_{lag}"] = out["MA20"].shift(lag)

    out[FEATURE_COLUMNS] = out[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    return out


def make_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    featured = add_features(df)
    X = featured[FEATURE_COLUMNS].ffill().fillna(0.0)
    return X, featured

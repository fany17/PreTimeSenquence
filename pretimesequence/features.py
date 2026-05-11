from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "log_return",
    "range_pct",
    "body_pct",
    "upper_shadow_pct",
    "lower_shadow_pct",
    "atr_pct",
    "realized_vol_5",
    "realized_vol_20",
    "realized_vol_60",
    "ret_3",
    "roc_5",
    "roc_10",
    "roc_20",
    "roc_30",
    "roc_60",
    "close_pos_20",
    "close_pos_60",
    "ma5_rel",
    "ma20_rel",
    "ma50_rel",
    "ma100_rel",
    "ma5_ma20_rel",
    "ma20_ma50_rel",
    "bb_z",
    "bb_width_pct",
    "TSI",
    "MACD_pct",
    "MACD_signal_pct",
    "MACD_hist_pct",
    "stochastic_k",
    "stochastic_d",
    "tenkan_rel",
    "kijun_rel",
    "span_a_rel",
    "span_b_rel",
    "PSAR_reversal",
    "ADX",
    "PDI",
    "NDI",
    "DX",
    "volume_log_change",
    "volume_z_20",
    "volume_ratio_5_20",
    "volume_acceleration_z",
    "obv_roc_20",
    "rolling_skew",
    "rolling_kurtosis",
    "is_hammer",
    "minute_sin",
    "minute_cos",
    "hour_sin",
    "hour_cos",
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
    eps = 1e-12

    out["log_return"] = np.log(close / close.shift(1))
    out["range_pct"] = (high - low) / close.shift(1)
    out["body_pct"] = (close - open_) / open_
    out["upper_shadow_pct"] = (high - np.maximum(open_, close)) / open_
    out["lower_shadow_pct"] = (np.minimum(open_, close) - low) / open_

    for window in (5, 20, 50, 100):
        out[f"MA{window}"] = close.rolling(window=window, min_periods=window).mean()

    out["std_20"] = close.rolling(window=20, min_periods=20).std()
    out["BB_upper"] = out["MA20"] + 2 * out["std_20"]
    out["BB_lower"] = out["MA20"] - 2 * out["std_20"]
    out["BB_width"] = out["BB_upper"] - out["BB_lower"]

    for window in (5, 20, 60):
        out[f"realized_vol_{window}"] = out["log_return"].rolling(window, min_periods=window).std()
    out["ret_3"] = close.pct_change(3)
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
    out["MACD_pct"] = out["MACD"] / close
    out["MACD_signal_pct"] = out["MACD_signal"] / close
    out["MACD_hist_pct"] = out["MACD_hist"] / close

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
    out["atr_pct"] = atr / close
    out["PDI"] = 100 * _rma(plus_dm, 14) / atr
    out["NDI"] = 100 * _rma(minus_dm, 14) / atr
    out["DX"] = (out["PDI"] - out["NDI"]).abs() / (out["PDI"] + out["NDI"]) * 100
    out["ADX"] = _rma(out["DX"], 14)

    out["PSAR_reversal"] = np.where(close.diff().fillna(0).mul(close.diff().shift().fillna(0)) < 0, 1, 0)

    out["Volume_MA5"] = volume.rolling(5, min_periods=5).mean()
    out["Volume_MA20"] = volume.rolling(20, min_periods=20).mean()
    out["volume_log_change"] = np.log((volume + eps) / (volume.shift(1) + eps))
    out["volume_z_20"] = (volume - out["Volume_MA20"]) / (volume.rolling(20, min_periods=20).std() + eps)
    out["volume_ratio_5_20"] = out["Volume_MA5"] / (out["Volume_MA20"] + eps) - 1
    out["volume_acceleration_z"] = volume.diff().diff() / (volume.rolling(20, min_periods=20).std() + eps)

    direction = np.sign(close.diff()).fillna(0)
    out["OBV"] = (direction * volume).cumsum()
    out["obv_roc_20"] = out["OBV"].pct_change(20)
    out["rolling_skew"] = out["log_return"].rolling(20, min_periods=20).skew()
    out["rolling_kurtosis"] = out["log_return"].rolling(20, min_periods=20).kurt()

    body = (close - open_).abs()
    candle_range = high - low
    lower_shadow = np.minimum(close, open_) - low
    out["is_hammer"] = ((lower_shadow > 2 * body) & ((body / candle_range.replace(0, np.nan)) < 0.3)).astype(int)

    low20 = low.rolling(20, min_periods=20).min()
    high20 = high.rolling(20, min_periods=20).max()
    low60 = low.rolling(60, min_periods=60).min()
    high60 = high.rolling(60, min_periods=60).max()
    out["close_pos_20"] = (close - low20) / (high20 - low20 + eps)
    out["close_pos_60"] = (close - low60) / (high60 - low60 + eps)
    out["ma5_rel"] = close / (out["MA5"] + eps) - 1
    out["ma20_rel"] = close / (out["MA20"] + eps) - 1
    out["ma50_rel"] = close / (out["MA50"] + eps) - 1
    out["ma100_rel"] = close / (out["MA100"] + eps) - 1
    out["ma5_ma20_rel"] = out["MA5"] / (out["MA20"] + eps) - 1
    out["ma20_ma50_rel"] = out["MA20"] / (out["MA50"] + eps) - 1
    out["bb_z"] = (close - out["MA20"]) / (out["std_20"] + eps)
    out["bb_width_pct"] = out["BB_width"] / close
    out["tenkan_rel"] = close / (out["Tenkan_sen"] + eps) - 1
    out["kijun_rel"] = close / (out["Kijun_sen"] + eps) - 1
    out["span_a_rel"] = close / (out["Senkou_Span_A"] + eps) - 1
    out["span_b_rel"] = close / (out["Senkou_Span_B"] + eps) - 1

    if "timestamp" in out.columns:
        ts = pd.to_datetime(out["timestamp"], errors="coerce")
        minute_of_day = ts.dt.hour.fillna(0) * 60 + ts.dt.minute.fillna(0)
        out["minute_sin"] = np.sin(2 * np.pi * minute_of_day / 1440)
        out["minute_cos"] = np.cos(2 * np.pi * minute_of_day / 1440)
        out["hour_sin"] = np.sin(2 * np.pi * ts.dt.hour.fillna(0) / 24)
        out["hour_cos"] = np.cos(2 * np.pi * ts.dt.hour.fillna(0) / 24)
    else:
        out["minute_sin"] = 0.0
        out["minute_cos"] = 1.0
        out["hour_sin"] = 0.0
        out["hour_cos"] = 1.0

    out[FEATURE_COLUMNS] = out[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    return out


def make_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    featured = add_features(df)
    X = featured[FEATURE_COLUMNS].ffill().fillna(0.0)
    return X, featured

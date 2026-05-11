from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from .config import BinanceConfig, load_binance_config

OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
BINANCE_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
    "ignore",
]


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in OHLCV_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")

    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=OHLCV_COLUMNS)
    out = out.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    return out


def load_market_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".pkl", ".pickle"}:
        df = pd.read_pickle(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported data file type: {path.suffix}")
    return normalize_ohlcv(df)


def save_market_data(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".pkl", ".pickle"}:
        df.to_pickle(path)
    elif path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported data file type: {path.suffix}")


def create_binance_client(config: BinanceConfig | None = None):
    try:
        from binance.client import Client
    except ImportError as exc:
        raise RuntimeError("python-binance is required for API data fetching.") from exc

    config = config or load_binance_config()
    requests_params = None
    if config.proxy_url:
        requests_params = {"proxies": {"https": config.proxy_url, "http": config.proxy_url}}
    if config.has_credentials:
        return Client(config.api_key, config.api_secret, testnet=config.testnet, requests_params=requests_params)
    return Client(requests_params=requests_params)


def fetch_klines(
    symbol: str = "BTCUSDT",
    interval: str = "1m",
    limit: int = 500,
    start_time: str | pd.Timestamp | None = None,
    futures: bool = True,
    sleep_seconds: float = 0.2,
) -> pd.DataFrame:
    client = create_binance_client()
    start_ms = None if start_time is None else int(pd.Timestamp(start_time).timestamp() * 1000)
    all_rows: list[list] = []

    while True:
        kwargs = {"symbol": symbol, "interval": interval, "limit": min(limit, 1000)}
        if start_ms is not None:
            kwargs["startTime"] = start_ms
        rows = client.futures_klines(**kwargs) if futures else client.get_klines(**kwargs)
        if not rows:
            break
        all_rows.extend(rows)
        if start_time is None or len(rows) < kwargs["limit"] or len(all_rows) >= limit:
            break
        start_ms = int(rows[-1][0]) + 1
        time.sleep(sleep_seconds)

    df = pd.DataFrame(all_rows[:limit], columns=BINANCE_COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return normalize_ohlcv(df)

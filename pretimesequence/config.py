from __future__ import annotations

import importlib
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class BinanceConfig:
    api_key: str | None = None
    api_secret: str | None = None
    proxy_url: str | None = None
    testnet: bool = False

    @property
    def has_credentials(self) -> bool:
        return bool(self.api_key and self.api_secret)


def _bool_env(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def load_binance_config(allow_legacy_key_modules: bool = True) -> BinanceConfig:
    """Load credentials from environment, with optional legacy Getkey fallback."""
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")

    if allow_legacy_key_modules and not (api_key and api_secret):
        for module_name in ("GetkeyReal", "Getkey"):
            try:
                module = importlib.import_module(module_name)
            except Exception:
                continue
            api_key = api_key or getattr(module, "api_key", None)
            api_secret = api_secret or getattr(module, "api_secret", None)
            if api_key and api_secret:
                break

    return BinanceConfig(
        api_key=api_key,
        api_secret=api_secret,
        proxy_url=os.getenv("BINANCE_PROXY_URL") or None,
        testnet=_bool_env(os.getenv("BINANCE_TESTNET"), default=False),
    )

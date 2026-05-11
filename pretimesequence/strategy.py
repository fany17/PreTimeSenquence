from __future__ import annotations

import pandas as pd


def signal_from_trend(
    trend: str,
    current_position: str = "flat",
    allow_flip: bool = True,
) -> str:
    if trend not in {"long", "short", "flat"}:
        raise ValueError(f"Unknown trend: {trend}")
    if trend == "flat":
        return "hold"
    if current_position == "flat":
        return f"open_{trend}"
    if current_position == trend:
        return "hold"
    return f"flip_to_{trend}" if allow_flip else "hold"


def attach_trade_signals(df: pd.DataFrame, allow_flip: bool = True) -> pd.DataFrame:
    out = df.copy()
    position = "flat"
    actions: list[str] = []
    for trend in out["trend"].fillna("flat"):
        action = signal_from_trend(str(trend), current_position=position, allow_flip=allow_flip)
        actions.append(action)
        if action.startswith("open_"):
            position = action.replace("open_", "")
        elif action.startswith("flip_to_"):
            position = action.replace("flip_to_", "")
    out["action"] = actions
    return out

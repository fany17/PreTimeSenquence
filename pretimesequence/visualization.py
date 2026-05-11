from __future__ import annotations

from pathlib import Path

import pandas as pd

from .model import TrendPredictor
from .strategy import attach_trade_signals


def build_signal_frame(df: pd.DataFrame, predictor: TrendPredictor, min_confidence: float = 0.55) -> pd.DataFrame:
    frame = predictor.predict_frame(df)
    frame["confidence"] = frame["trend_score"].astype(float)
    frame.loc[frame["confidence"] < min_confidence, "trend"] = "flat"
    return attach_trade_signals(frame)


def write_signal_html(
    df: pd.DataFrame,
    predictor: TrendPredictor,
    output_path: str | Path = "outputs/signals.html",
    min_confidence: float = 0.55,
    title: str = "Trend Signals",
) -> pd.DataFrame:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise RuntimeError("plotly is required for HTML visualization.") from exc

    signals = build_signal_frame(df, predictor, min_confidence=min_confidence)
    long_points = signals[signals["action"].isin(["open_long", "flip_to_long"])]
    short_points = signals[signals["action"].isin(["open_short", "flip_to_short"])]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.72, 0.28],
        subplot_titles=(title, "Model confidence and trend"),
    )
    fig.add_trace(
        go.Candlestick(
            x=signals["timestamp"],
            open=signals["open"],
            high=signals["high"],
            low=signals["low"],
            close=signals["close"],
            name="OHLC",
            increasing_line_color="#1f9d55",
            decreasing_line_color="#d64545",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=long_points["timestamp"],
            y=long_points["low"] * 0.998,
            mode="markers",
            name="Open / flip long",
            marker=dict(symbol="triangle-up", size=10, color="#13a538"),
            customdata=long_points[["trend_score", "action"]],
            hovertemplate="%{x}<br>price=%{y:.6f}<br>confidence=%{customdata[0]:.3f}<br>%{customdata[1]}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=short_points["timestamp"],
            y=short_points["high"] * 1.002,
            mode="markers",
            name="Open / flip short",
            marker=dict(symbol="triangle-down", size=10, color="#c53030"),
            customdata=short_points[["trend_score", "action"]],
            hovertemplate="%{x}<br>price=%{y:.6f}<br>confidence=%{customdata[0]:.3f}<br>%{customdata[1]}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=signals["timestamp"],
            y=signals["trend_score"],
            mode="lines",
            name="confidence",
            line=dict(color="#2b6cb0", width=1.5),
        ),
        row=2,
        col=1,
    )
    fig.add_hline(y=min_confidence, line_dash="dash", line_color="#666", row=2, col=1)
    fig.update_layout(
        template="plotly_white",
        height=900,
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=60, r=30, t=70, b=45),
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Confidence", row=2, col=1, range=[0, 1])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path, include_plotlyjs="cdn")
    return signals

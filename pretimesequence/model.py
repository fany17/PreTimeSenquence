from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .features import make_feature_matrix

CODE_TO_TREND = {0: "short", 1: "flat", 2: "long"}


@dataclass(frozen=True)
class PredictionResult:
    timestamp: pd.Timestamp
    close: float
    score: float
    trend: str
    confidence: float


class TrendPredictor:
    def __init__(self, model_path: str | Path | None = "data/xgboost_model.json", low_threshold: float = 0.3, high_threshold: float = 0.7):
        self.model_path = Path(model_path) if model_path else None
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self._model = None
        self._last_score_is_probability = False

    def _load_xgboost(self):
        if self._model is not None:
            return self._model
        if not self.model_path or not self.model_path.exists():
            return None
        try:
            import xgboost as xgb
        except ImportError:
            return None
        model = xgb.Booster()
        model.load_model(str(self.model_path))
        self._model = model
        return model

    def predict_scores(self, df: pd.DataFrame) -> pd.Series:
        X, _ = make_feature_matrix(df)
        model = self._load_xgboost()
        if model is not None:
            import xgboost as xgb

            scores = model.predict(xgb.DMatrix(X))
            self._last_score_is_probability = False
            return pd.Series(scores, index=df.index, name="trend_score").clip(0, 1)

        score = 0.5 + 2.5 * df["close"].pct_change(20).fillna(0)
        self._last_score_is_probability = False
        return score.clip(0, 1).rename("trend_score")

    def predict_trends(self, df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        X, _ = make_feature_matrix(df)
        model = self._load_xgboost()
        if model is None:
            scores = self.predict_scores(df)
            return scores, self.classify_scores(scores)

        import xgboost as xgb

        raw = model.predict(xgb.DMatrix(X))
        if raw.ndim == 2 and raw.shape[1] == 3:
            codes = raw.argmax(axis=1)
            confidence = raw.max(axis=1)
            trends = pd.Series([CODE_TO_TREND[int(code)] for code in codes], index=df.index, name="trend")
            self._last_score_is_probability = True
            return pd.Series(confidence, index=df.index, name="trend_score"), trends

        scores = pd.Series(raw, index=df.index, name="trend_score").clip(0, 1)
        self._last_score_is_probability = False
        return scores, self.classify_scores(scores)

    def classify_scores(self, scores: pd.Series) -> pd.Series:
        return pd.Series(
            np.where(scores >= self.high_threshold, "long", np.where(scores <= self.low_threshold, "short", "flat")),
            index=scores.index,
            name="trend",
        )

    def predict_latest(self, df: pd.DataFrame) -> PredictionResult:
        scores, trends = self.predict_trends(df)
        idx = scores.last_valid_index()
        if idx is None:
            raise ValueError("No valid rows for prediction.")
        score = float(scores.loc[idx])
        trend = str(trends.loc[idx])
        confidence = score if self._last_score_is_probability else abs(score - 0.5) * 2
        return PredictionResult(
            timestamp=pd.Timestamp(df.loc[idx, "timestamp"]),
            close=float(df.loc[idx, "close"]),
            score=score,
            trend=trend,
            confidence=float(confidence),
        )

    def predict_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["trend_score"], out["trend"] = self.predict_trends(df)
        return out

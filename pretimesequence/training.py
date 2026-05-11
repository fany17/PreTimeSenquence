from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .features import make_feature_matrix
from .targets import LabelConfig, make_triple_barrier_labels


@dataclass(frozen=True)
class TrainConfig:
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    max_depth: int = 3
    learning_rate: float = 0.05
    n_estimators: int = 300
    subsample: float = 0.8
    colsample_bytree: float = 0.8


def chronological_split(X: pd.DataFrame, y: pd.Series, train_ratio: float = 0.70, val_ratio: float = 0.15):
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    if train_end <= 0 or val_end <= train_end or val_end >= n:
        raise ValueError("Invalid split ratios or insufficient data.")
    return (
        X.iloc[:train_end],
        X.iloc[train_end:val_end],
        X.iloc[val_end:],
        y.iloc[:train_end],
        y.iloc[train_end:val_end],
        y.iloc[val_end:],
    )


def train_xgboost_classifier(
    df: pd.DataFrame,
    output_path: str | Path = "data/xgboost_trend_model.json",
    label_config: LabelConfig | None = None,
    train_config: TrainConfig | None = None,
) -> dict:
    try:
        from sklearn.metrics import balanced_accuracy_score, classification_report
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise RuntimeError("xgboost and scikit-learn are required for training.") from exc

    label_config = label_config or LabelConfig()
    train_config = train_config or TrainConfig()
    labelled = make_triple_barrier_labels(df, label_config)
    X, featured = make_feature_matrix(labelled)
    y = labelled["label_code"].astype(int)
    X_train, X_val, X_test, y_train, y_val, y_test = chronological_split(
        X, y, train_config.train_ratio, train_config.val_ratio
    )

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        max_depth=train_config.max_depth,
        learning_rate=train_config.learning_rate,
        n_estimators=train_config.n_estimators,
        subsample=train_config.subsample,
        colsample_bytree=train_config.colsample_bytree,
        eval_metric="mlogloss",
        random_state=42,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(output_path)

    y_pred = pd.Series(model.predict(X_test), index=y_test.index)
    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1, 2],
        target_names=["short", "flat", "long"],
        zero_division=0,
        output_dict=True,
    )
    return {
        "rows": len(labelled),
        "model_path": str(output_path),
        "label_counts": labelled["label"].value_counts().to_dict(),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "test_report": report,
        "last_timestamp": str(featured["timestamp"].iloc[-1]),
    }

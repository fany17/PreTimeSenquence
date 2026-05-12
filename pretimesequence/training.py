from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .features import make_feature_matrix
from .targets import LabelConfig, TradeOutcomeConfig, make_trade_outcome_labels, make_triple_barrier_labels


@dataclass(frozen=True)
class TrainConfig:
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    max_depth: int = 3
    learning_rate: float = 0.05
    n_estimators: int = 300
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_class_weight: float = 0.5
    max_class_weight: float = 3.0


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


def class_balanced_sample_weight(y: pd.Series, min_weight: float = 0.5, max_weight: float = 3.0) -> pd.Series:
    counts = y.value_counts()
    total = len(y)
    n_classes = max(len(counts), 1)
    weights = {label: total / (n_classes * count) for label, count in counts.items()}
    weights = {label: min(max(weight, min_weight), max_weight) for label, weight in weights.items()}
    return y.map(weights).astype(float)


def train_xgboost_classifier(
    df: pd.DataFrame,
    output_path: str | Path = "data/xgboost_trend_model.json",
    label_config: LabelConfig | None = None,
    train_config: TrainConfig | None = None,
    train_until: str | pd.Timestamp | None = None,
    context_frames: dict[str, pd.DataFrame] | None = None,
) -> dict:
    try:
        from sklearn.metrics import balanced_accuracy_score, classification_report
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise RuntimeError("xgboost and scikit-learn are required for training.") from exc

    label_config = label_config or LabelConfig()
    train_config = train_config or TrainConfig()
    if train_until is not None:
        cutoff = pd.Timestamp(train_until)
        df = df[pd.to_datetime(df["timestamp"]) <= cutoff].copy()
        if df.empty:
            raise ValueError(f"No training rows before train_until={train_until}")
    labelled = make_triple_barrier_labels(df, label_config)
    X, featured = make_feature_matrix(labelled, context_frames=context_frames)
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
    sample_weight = class_balanced_sample_weight(
        y_train,
        min_weight=train_config.min_class_weight,
        max_weight=train_config.max_class_weight,
    )
    model.fit(X_train, y_train, sample_weight=sample_weight, eval_set=[(X_val, y_val)], verbose=False)

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
        "train_until": None if train_until is None else str(pd.Timestamp(train_until)),
    }


def train_two_stage_xgboost_classifier(
    df: pd.DataFrame,
    output_path: str | Path = "data/xgboost_two_stage_trend_model.json",
    outcome_config: TradeOutcomeConfig | None = None,
    train_config: TrainConfig | None = None,
    train_until: str | pd.Timestamp | None = None,
    context_frames: dict[str, pd.DataFrame] | None = None,
) -> dict:
    try:
        from sklearn.metrics import balanced_accuracy_score, classification_report
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise RuntimeError("xgboost and scikit-learn are required for training.") from exc

    outcome_config = outcome_config or TradeOutcomeConfig()
    train_config = train_config or TrainConfig()
    if train_until is not None:
        cutoff = pd.Timestamp(train_until)
        df = df[pd.to_datetime(df["timestamp"]) <= cutoff].copy()
        if df.empty:
            raise ValueError(f"No training rows before train_until={train_until}")

    labelled = make_trade_outcome_labels(df, outcome_config)
    X, featured = make_feature_matrix(labelled, context_frames=context_frames)
    y_trade = labelled["trade_label"].astype(int)
    y_side = labelled["side_label"]
    y_final = labelled["label_code"].astype(int)

    X_train, X_val, X_test, y_trade_train, y_trade_val, y_trade_test = chronological_split(
        X, y_trade, train_config.train_ratio, train_config.val_ratio
    )
    _, _, _, y_side_train, y_side_val, y_side_test = chronological_split(
        X, y_side, train_config.train_ratio, train_config.val_ratio
    )
    _, _, _, y_final_train, y_final_val, y_final_test = chronological_split(
        X, y_final, train_config.train_ratio, train_config.val_ratio
    )

    trade_model = XGBClassifier(
        objective="binary:logistic",
        max_depth=train_config.max_depth,
        learning_rate=train_config.learning_rate,
        n_estimators=train_config.n_estimators,
        subsample=train_config.subsample,
        colsample_bytree=train_config.colsample_bytree,
        eval_metric="logloss",
        random_state=42,
    )
    trade_weight = class_balanced_sample_weight(
        y_trade_train,
        min_weight=train_config.min_class_weight,
        max_weight=train_config.max_class_weight,
    )
    trade_model.fit(
        X_train,
        y_trade_train,
        sample_weight=trade_weight,
        eval_set=[(X_val, y_trade_val)],
        verbose=False,
    )

    side_train_mask = y_side_train.notna()
    side_val_mask = y_side_val.notna()
    side_test_mask = y_side_test.notna()
    if side_train_mask.sum() < 100 or y_side_train[side_train_mask].nunique() < 2:
        raise ValueError("Not enough long/short training samples for the side model.")

    side_model = XGBClassifier(
        objective="binary:logistic",
        max_depth=train_config.max_depth,
        learning_rate=train_config.learning_rate,
        n_estimators=train_config.n_estimators,
        subsample=train_config.subsample,
        colsample_bytree=train_config.colsample_bytree,
        eval_metric="logloss",
        random_state=43,
    )
    side_y_train = y_side_train[side_train_mask].astype(int)
    side_weight = class_balanced_sample_weight(
        side_y_train,
        min_weight=train_config.min_class_weight,
        max_weight=train_config.max_class_weight,
    )
    eval_set = None
    if side_val_mask.sum() and y_side_val[side_val_mask].nunique() > 1:
        eval_set = [(X_val[side_val_mask], y_side_val[side_val_mask].astype(int))]
    side_model.fit(
        X_train[side_train_mask],
        side_y_train,
        sample_weight=side_weight,
        eval_set=eval_set,
        verbose=False,
    )

    trade_prob = trade_model.predict_proba(X_test)[:, 1]
    side_prob = side_model.predict_proba(X_test)[:, 1]
    trade_pred = (trade_prob >= 0.5).astype(int)
    side_pred = np.where(side_prob >= 0.5, 2, 0)
    final_pred = np.where(trade_pred == 1, side_pred, 1)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trade_path = output_path.with_name(output_path.stem + "_trade.json")
    side_path = output_path.with_name(output_path.stem + "_side.json")
    trade_model.save_model(trade_path)
    side_model.save_model(side_path)

    manifest = {
        "model_type": "two_stage_xgboost",
        "trade_model": trade_path.name,
        "side_model": side_path.name,
        "trade_threshold": 0.5,
        "side_threshold": 0.5,
        "classes": {"short": 0, "flat": 1, "long": 2},
        "outcome_config": outcome_config.__dict__,
    }
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    trade_report = classification_report(
        y_trade_test,
        trade_pred,
        labels=[0, 1],
        target_names=["flat", "trade"],
        zero_division=0,
        output_dict=True,
    )
    side_report = {}
    side_balanced_accuracy = None
    if side_test_mask.sum() and y_side_test[side_test_mask].nunique() > 1:
        side_test_pred = (side_prob[side_test_mask.to_numpy()] >= 0.5).astype(int)
        side_balanced_accuracy = float(balanced_accuracy_score(y_side_test[side_test_mask].astype(int), side_test_pred))
        side_report = classification_report(
            y_side_test[side_test_mask].astype(int),
            side_test_pred,
            labels=[0, 1],
            target_names=["short", "long"],
            zero_division=0,
            output_dict=True,
        )

    final_report = classification_report(
        y_final_test,
        final_pred,
        labels=[0, 1, 2],
        target_names=["short", "flat", "long"],
        zero_division=0,
        output_dict=True,
    )
    return {
        "rows": len(labelled),
        "model_path": str(output_path),
        "trade_model_path": str(trade_path),
        "side_model_path": str(side_path),
        "label_counts": labelled["label"].value_counts().to_dict(),
        "trade_counts": y_trade.value_counts().to_dict(),
        "test_trade_balanced_accuracy": float(balanced_accuracy_score(y_trade_test, trade_pred)),
        "test_side_balanced_accuracy": side_balanced_accuracy,
        "test_final_balanced_accuracy": float(balanced_accuracy_score(y_final_test, final_pred)),
        "trade_report": trade_report,
        "side_report": side_report,
        "final_report": final_report,
        "last_timestamp": str(featured["timestamp"].iloc[-1]),
        "train_until": None if train_until is None else str(pd.Timestamp(train_until)),
    }

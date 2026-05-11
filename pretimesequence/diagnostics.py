from __future__ import annotations

from pathlib import Path

import pandas as pd

from .features import make_feature_matrix
from .targets import LabelConfig, make_triple_barrier_labels
from .training import TrainConfig, chronological_split, class_balanced_sample_weight

LABEL_NAMES = {0: "short", 1: "flat", 2: "long"}


def _balanced_accuracy(y_true: pd.Series, y_pred: pd.Series) -> float:
    from sklearn.metrics import balanced_accuracy_score

    return float(balanced_accuracy_score(y_true, y_pred))


def _cadence_summary(df: pd.DataFrame) -> dict:
    ts = pd.to_datetime(df["timestamp"]).sort_values()
    diffs = ts.diff().dropna()
    if diffs.empty:
        return {"median_seconds": None, "large_gaps": 0, "max_gap": None}
    median = diffs.median()
    large = diffs[diffs > median * 3]
    return {
        "median_seconds": float(median.total_seconds()),
        "large_gaps": int(len(large)),
        "max_gap": str(diffs.max()),
    }


def _baseline_predictions(X_test: pd.DataFrame, y_train: pd.Series, df_test: pd.DataFrame) -> dict[str, pd.Series]:
    majority = int(y_train.value_counts().idxmax())
    momentum = df_test["close"].pct_change(20).fillna(0)
    return {
        "majority": pd.Series(majority, index=X_test.index),
        "momentum20": pd.Series(
            momentum.map(lambda r: 2 if r > 0.001 else (0 if r < -0.001 else 1)).astype(int).values,
            index=X_test.index,
        ),
    }


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    headers = [str(col) for col in df.columns]
    rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(row[col]) for col in df.columns) + " |")
    return "\n".join(rows)


def _fit_xgb(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, train_config: TrainConfig):
    from xgboost import XGBClassifier

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
    return model


def _walk_forward_scores(X: pd.DataFrame, y: pd.Series, train_config: TrainConfig, folds: int = 4) -> pd.DataFrame:
    n = len(X)
    min_train = int(n * 0.45)
    fold_size = int((n - min_train) / folds)
    rows = []
    for fold in range(folds):
        train_end = min_train + fold * fold_size
        test_start = train_end
        test_end = n if fold == folds - 1 else min(train_end + fold_size, n)
        val_start = max(0, int(train_end * 0.85))
        if test_end <= test_start or train_end - val_start < 100:
            continue
        X_train, y_train = X.iloc[:val_start], y.iloc[:val_start]
        X_val, y_val = X.iloc[val_start:train_end], y.iloc[val_start:train_end]
        X_test, y_test = X.iloc[test_start:test_end], y.iloc[test_start:test_end]
        model = _fit_xgb(X_train, y_train, X_val, y_val, train_config)
        pred = pd.Series(model.predict(X_test), index=y_test.index)
        pred_share = {LABEL_NAMES[int(k)]: float(v) for k, v in pred.value_counts(normalize=True).items()}
        rows.append(
            {
                "fold": fold + 1,
                "train_rows": len(X_train),
                "test_rows": len(X_test),
                "balanced_accuracy": _balanced_accuracy(y_test, pred),
                "pred_share": pred_share,
            }
        )
    return pd.DataFrame(rows)


def run_diagnostics(
    df: pd.DataFrame,
    output_path: str | Path = "outputs/diagnostics.md",
    label_config: LabelConfig | None = None,
    train_config: TrainConfig | None = None,
    sample_for_mi: int = 20000,
) -> dict:
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.metrics import classification_report

    label_config = label_config or LabelConfig()
    train_config = train_config or TrainConfig(n_estimators=200)
    labelled = make_triple_barrier_labels(df, label_config)
    X, featured = make_feature_matrix(labelled)
    y = labelled["label_code"].astype(int)
    X_train, X_val, X_test, y_train, y_val, y_test = chronological_split(
        X, y, train_config.train_ratio, train_config.val_ratio
    )
    test_rows = labelled.iloc[X_test.index]

    baselines = {
        name: _balanced_accuracy(y_test, pred)
        for name, pred in _baseline_predictions(X_test, y_train, test_rows).items()
    }

    model = _fit_xgb(X_train, y_train, X_val, y_val, train_config)
    y_pred = pd.Series(model.predict(X_test), index=y_test.index)
    model_balanced = _balanced_accuracy(y_test, y_pred)
    walk_forward = _walk_forward_scores(X, y, TrainConfig(n_estimators=120), folds=4)

    mi_X = X_train
    mi_y = y_train
    if len(mi_X) > sample_for_mi:
        mi_X = mi_X.iloc[-sample_for_mi:]
        mi_y = mi_y.iloc[-sample_for_mi:]
    mi = mutual_info_classif(mi_X, mi_y, random_state=42, discrete_features=False)
    mi_table = (
        pd.DataFrame({"feature": X.columns, "mutual_info": mi})
        .sort_values("mutual_info", ascending=False)
        .head(15)
        .reset_index(drop=True)
    )

    label_counts = labelled["label"].value_counts().to_dict()
    cadence = _cadence_summary(df)
    days = (pd.to_datetime(df["timestamp"]).max() - pd.to_datetime(df["timestamp"]).min()).total_seconds() / 86400
    model_report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1, 2],
        target_names=["short", "flat", "long"],
        zero_division=0,
        output_dict=True,
    )

    diagnosis = []
    if len(df) < 100000 or days < 120:
        diagnosis.append("数据量偏少：当前更像单币种短样本研究，难以覆盖足够多市场 regime。")
    if model_balanced <= max(baselines.values()) + 0.03:
        diagnosis.append("算法没有明显超过简单基线：优先怀疑特征/标签可预测性，而不是换更复杂模型。")
    if mi_table["mutual_info"].max() < 0.02:
        diagnosis.append("单特征信息量很弱：传统技术指标对该 GT 的边际解释力有限。")
    pred_share = y_pred.value_counts(normalize=True).to_dict()
    if max(pred_share.values()) > 0.85:
        dominant = LABEL_NAMES[int(max(pred_share, key=pred_share.get))]
        diagnosis.append(f"测试期预测坍缩到单一方向 `{dominant}`：这是 regime shift 或类别代价未校准的信号。")
    if not walk_forward.empty and walk_forward["balanced_accuracy"].mean() <= max(baselines.values()) + 0.03:
        diagnosis.append("walk-forward 平均表现没有超过基线：当前特征组还不足以形成稳定 alpha。")
    if label_counts.get("flat", 0) / len(labelled) < 0.15:
        diagnosis.append("flat 比例偏低：出手过滤可能仍不够严格。")
    if not diagnosis:
        diagnosis.append("未发现单一主因，需要继续做 walk-forward 和跨币种稳定性检验。")

    lines = [
        "# 预测效果诊断",
        "",
        f"- 原始行数: {len(df)}",
        f"- 可标注行数: {len(labelled)}",
        f"- 时间跨度: {days:.1f} 天",
        f"- 中位 K 线间隔: {cadence['median_seconds']} 秒",
        f"- 大间隔数量: {cadence['large_gaps']}, 最大间隔: {cadence['max_gap']}",
        f"- 标签分布: {label_counts}",
        "",
        "## 验证表现",
        "",
        f"- majority baseline balanced accuracy: {baselines['majority']:.3f}",
        f"- momentum20 baseline balanced accuracy: {baselines['momentum20']:.3f}",
        f"- XGBoost chronological test balanced accuracy: {model_balanced:.3f}",
        f"- XGBoost test prediction share: { {LABEL_NAMES[int(k)]: float(v) for k, v in pred_share.items()} }",
        "",
        "## Walk-forward",
        "",
        _markdown_table(walk_forward),
        "",
        "## Top Mutual Information Features",
        "",
        _markdown_table(mi_table),
        "",
        "## 判断",
        "",
    ]
    lines.extend(f"- {item}" for item in diagnosis)
    lines.extend(
        [
            "",
            "## Test Classification Report",
            "",
            _markdown_table(pd.DataFrame(model_report).transpose().reset_index(names="class")),
            "",
        ]
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")

    return {
        "rows": len(df),
        "labelled_rows": len(labelled),
        "days": days,
        "label_counts": label_counts,
        "baselines": baselines,
        "model_balanced_accuracy": model_balanced,
        "top_features": mi_table.to_dict("records"),
        "walk_forward": walk_forward.to_dict("records"),
        "diagnosis": diagnosis,
        "output_path": str(output_path),
    }

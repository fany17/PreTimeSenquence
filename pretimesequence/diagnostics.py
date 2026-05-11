from __future__ import annotations

from pathlib import Path

import pandas as pd

from .features import make_feature_matrix
from .targets import LabelConfig, make_triple_barrier_labels
from .training import TrainConfig, chronological_split

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


def run_diagnostics(
    df: pd.DataFrame,
    output_path: str | Path = "outputs/diagnostics.md",
    label_config: LabelConfig | None = None,
    train_config: TrainConfig | None = None,
    sample_for_mi: int = 20000,
) -> dict:
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.metrics import classification_report
    from xgboost import XGBClassifier

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
    y_pred = pd.Series(model.predict(X_test), index=y_test.index)
    model_balanced = _balanced_accuracy(y_test, y_pred)

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
    top_feature_names = set(mi_table.head(8)["feature"])
    level_features = {"close", "open", "high", "low", "MA5", "MA20", "MA50", "MA100", "BB_upper", "BB_lower", "Kijun_sen", "Tenkan_sen", "Senkou_Span_A", "Senkou_Span_B"}
    if len(top_feature_names & level_features) >= 5:
        diagnosis.append("高排名特征主要是价格水平/均线类非平稳变量：模型可能在记忆价格阶段，而不是学习可迁移的收益结构。")
    pred_share = y_pred.value_counts(normalize=True).to_dict()
    if max(pred_share.values()) > 0.85:
        dominant = LABEL_NAMES[int(max(pred_share, key=pred_share.get))]
        diagnosis.append(f"测试期预测坍缩到单一方向 `{dominant}`：这是 regime shift 或类别代价未校准的信号。")
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
        "diagnosis": diagnosis,
        "output_path": str(output_path),
    }

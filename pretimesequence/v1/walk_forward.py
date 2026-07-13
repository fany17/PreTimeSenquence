from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .backtest import TRADE_COLUMNS, EventBacktestResult, run_event_backtest
from .dataset import SupervisedDataset, build_supervised_dataset, fingerprint_bars
from .execution import validate_bar_cadence, validate_bars
from .policy import decisions_from_action_values
from .spec import StrategySpec
from .splits import inner_walk_forward_folds, outer_walk_forward_folds
from .training import fit_ridge_action_value, predict_action_values


@dataclass(frozen=True)
class NestedWalkForwardConfig:
    min_development_bars: int
    test_bars: int
    inner_min_train_bars: int
    inner_validation_bars: int
    alphas: tuple[float, ...]
    thresholds: tuple[float, ...]
    outer_step_bars: int | None = None
    max_outer_folds: int | None = None
    inner_step_bars: int | None = None
    max_inner_folds: int | None = None
    initial_equity: float = 1.0
    notional_fraction: float | None = None

    def __post_init__(self) -> None:
        required_positive = (
            "min_development_bars",
            "test_bars",
            "inner_min_train_bars",
            "inner_validation_bars",
        )
        for name in required_positive:
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        for name in ("outer_step_bars", "max_outer_folds", "inner_step_bars", "max_inner_folds"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer or None.")

        if not self.alphas:
            raise ValueError("alphas must contain at least one pre-registered candidate.")
        resolved_alphas = tuple(float(alpha) for alpha in self.alphas)
        if any(not math.isfinite(alpha) or alpha < 0 for alpha in resolved_alphas):
            raise ValueError("alphas must be finite and non-negative.")
        if len(set(resolved_alphas)) != len(resolved_alphas):
            raise ValueError("alphas must not contain duplicates.")
        object.__setattr__(self, "alphas", resolved_alphas)

        if not self.thresholds:
            raise ValueError("thresholds must contain at least one pre-registered candidate.")
        resolved_thresholds = tuple(float(threshold) for threshold in self.thresholds)
        if any(not math.isfinite(threshold) or threshold < 0 for threshold in resolved_thresholds):
            raise ValueError("thresholds must be finite and non-negative.")
        if len(set(resolved_thresholds)) != len(resolved_thresholds):
            raise ValueError("thresholds must not contain duplicates.")
        object.__setattr__(self, "thresholds", resolved_thresholds)

        if not math.isfinite(float(self.initial_equity)) or self.initial_equity <= 0:
            raise ValueError("initial_equity must be finite and positive.")
        if self.notional_fraction is not None and (
            not math.isfinite(float(self.notional_fraction)) or self.notional_fraction <= 0
        ):
            raise ValueError("notional_fraction must be finite and positive when provided.")


@dataclass(frozen=True)
class NestedWalkForwardResult:
    fold_metrics: pd.DataFrame
    candidate_metrics: pd.DataFrame
    inner_oof_predictions: pd.DataFrame
    outer_predictions: pd.DataFrame
    trades: pd.DataFrame
    spec: StrategySpec
    config: NestedWalkForwardConfig


class AllCandidatesFailedError(ValueError):
    def __init__(self, candidate_metrics: pd.DataFrame) -> None:
        super().__init__("All candidates failed inside inner walk-forward evaluation.")
        self.candidate_metrics = candidate_metrics


def _validate_dataset_against_bars(
    bars: pd.DataFrame,
    dataset: SupervisedDataset,
    spec: StrategySpec,
) -> None:
    if not isinstance(dataset, SupervisedDataset):
        raise TypeError("dataset must be a SupervisedDataset.")
    if dataset.spec != spec:
        raise ValueError("dataset StrategySpec does not match the walk-forward StrategySpec.")
    if dataset.bar_fingerprint != fingerprint_bars(bars, spec):
        raise ValueError("dataset bar fingerprint does not match the supplied bars.")
    if tuple(dataset.X.columns) != dataset.feature_names:
        raise ValueError("dataset feature schema does not match feature_names.")
    if not dataset.X.index.equals(dataset.targets.index) or not dataset.X.index.equals(dataset.sample_meta.index):
        raise ValueError("dataset X, targets and sample_meta indices must align exactly.")

    frame = validate_bars(bars)
    validate_bar_cadence(frame, spec.base_interval_minutes)
    signal_indices = dataset.sample_meta["signal_index"].to_numpy(dtype=int)
    if (signal_indices < 0).any() or (signal_indices >= len(frame)).any():
        raise ValueError("dataset signal_index is outside the supplied bars.")
    expected_times = frame.iloc[signal_indices]["timestamp"].reset_index(drop=True)
    actual_times = dataset.sample_meta["signal_time"].reset_index(drop=True)
    if not expected_times.equals(actual_times):
        raise ValueError("dataset signal_time does not match the supplied bars.")
    if dataset.sample_meta["horizon_end_index"].max() >= len(frame):
        raise ValueError("dataset target horizon extends outside the supplied bars.")


def _backtest_metrics(result: EventBacktestResult, initial_equity: float) -> dict[str, float | int]:
    if result.equity_curve.empty:
        equity = np.array([float(initial_equity)], dtype=float)
    else:
        equity = np.concatenate(
            ([float(initial_equity)], result.equity_curve["equity"].to_numpy(dtype=float))
        )
    if not np.isfinite(equity).all():
        raise RuntimeError("Backtest equity curve contains non-finite values.")
    running_peak = np.maximum.accumulate(equity)
    drawdown = 1.0 - equity / running_peak
    final_equity = float(equity[-1])
    trades = len(result.trades)
    expectancy = 0.0 if not trades else float(result.trades["net_return"].mean())
    return {
        "trades": int(trades),
        "net_return": float(final_equity / initial_equity - 1.0),
        "max_drawdown": float(max(0.0, np.max(drawdown))),
        "expectancy": expectancy,
    }


def _failure_text(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def _candidate_failure_row(
    outer_fold: int,
    alpha: float,
    threshold: float,
    candidate_order: int,
    reason: str,
) -> dict[str, object]:
    return {
        "outer_fold": outer_fold,
        "alpha": alpha,
        "threshold": threshold,
        "candidate_order": candidate_order,
        "status": "failed",
        "failure_reason": reason,
        "inner_folds": 0,
        "median_net_return": np.nan,
        "worst_fold_net_return": np.nan,
        "max_drawdown": np.nan,
        "trades": 0,
        "expectancy": np.nan,
        "selected": False,
    }


def run_nested_walk_forward(
    bars: pd.DataFrame,
    spec: StrategySpec,
    config: NestedWalkForwardConfig,
    *,
    dataset: SupervisedDataset | None = None,
) -> NestedWalkForwardResult:
    """Run deterministic nested evaluation without writing models or reports.

    Every alpha/threshold pair is selected only from inner validation
    backtests. The selected alpha is retrained once on outer development, and
    the frozen threshold is applied to the outer test exactly once.
    """

    resolved_dataset = build_supervised_dataset(bars, spec) if dataset is None else dataset
    _validate_dataset_against_bars(bars, resolved_dataset, spec)
    if config.notional_fraction is not None and config.notional_fraction > spec.leverage_cap:
        raise ValueError("notional_fraction must not exceed StrategySpec.leverage_cap.")

    outer_folds = outer_walk_forward_folds(
        resolved_dataset.sample_meta,
        spec,
        min_development_bars=config.min_development_bars,
        test_bars=config.test_bars,
        step_bars=config.outer_step_bars,
        max_folds=config.max_outer_folds,
    )

    all_candidate_rows: list[dict[str, object]] = []
    all_oof_frames: list[pd.DataFrame] = []
    all_outer_frames: list[pd.DataFrame] = []
    all_trade_frames: list[pd.DataFrame] = []
    fold_metric_rows: list[dict[str, object]] = []

    for outer in outer_folds:
        inner_folds = inner_walk_forward_folds(
            resolved_dataset.sample_meta,
            outer.development_indices,
            spec,
            min_train_bars=config.inner_min_train_bars,
            validation_bars=config.inner_validation_bars,
            step_bars=config.inner_step_bars,
            max_folds=config.max_inner_folds,
        )
        fold_candidate_rows: list[dict[str, object]] = []

        for alpha_position, alpha in enumerate(config.alphas):
            predictions_by_inner: dict[int, pd.DataFrame] = {}
            alpha_failure: str | None = None
            for inner in inner_folds:
                try:
                    model = fit_ridge_action_value(
                        resolved_dataset.X,
                        resolved_dataset.targets,
                        inner.train_indices,
                        alpha=alpha,
                    )
                    raw_predictions = predict_action_values(
                        model,
                        resolved_dataset.X.loc[inner.validation_indices],
                    )
                    predictions_by_inner[inner.fold_id] = raw_predictions
                    actual = resolved_dataset.targets.loc[
                        inner.validation_indices,
                        ["long_net_return", "short_net_return"],
                    ]
                    train_start_signal_index = int(inner.train_indices.min())
                    train_end_signal_index = int(inner.train_indices.max())
                    train_samples = int(len(inner.train_indices))
                    train_horizon_end_max = int(
                        resolved_dataset.sample_meta.loc[
                            inner.train_indices,
                            "horizon_end_index",
                        ].max()
                    )
                    oof_frame = raw_predictions.copy()
                    oof_frame.insert(0, "alpha", alpha)
                    oof_frame.insert(0, "inner_fold", inner.fold_id)
                    oof_frame.insert(0, "outer_fold", outer.fold_id)
                    oof_frame["actual_long_net_return"] = actual[
                        "long_net_return"
                    ].to_numpy(dtype=float)
                    oof_frame["actual_short_net_return"] = actual[
                        "short_net_return"
                    ].to_numpy(dtype=float)
                    oof_frame["train_start_signal_index"] = train_start_signal_index
                    oof_frame["train_end_signal_index"] = train_end_signal_index
                    oof_frame["train_samples"] = train_samples
                    oof_frame["train_horizon_end_max"] = train_horizon_end_max
                    all_oof_frames.append(oof_frame)
                except Exception as exc:  # candidate failures are audit artifacts
                    alpha_failure = f"inner_fold={inner.fold_id}: {_failure_text(exc)}"
                    break

            if alpha_failure is not None:
                for threshold_position, threshold in enumerate(config.thresholds):
                    candidate_order = alpha_position * len(config.thresholds) + threshold_position
                    fold_candidate_rows.append(
                        _candidate_failure_row(
                            outer.fold_id,
                            alpha,
                            threshold,
                            candidate_order,
                            alpha_failure,
                        )
                    )
                continue

            for threshold_position, threshold in enumerate(config.thresholds):
                candidate_order = alpha_position * len(config.thresholds) + threshold_position
                inner_metrics: list[dict[str, float | int]] = []
                threshold_failure: str | None = None
                for inner in inner_folds:
                    try:
                        prediction_frame = predictions_by_inner[inner.fold_id]
                        decisions = decisions_from_action_values(
                            prediction_frame,
                            threshold=threshold,
                        )
                        if not (
                            resolved_dataset.sample_meta.loc[
                                inner.validation_indices,
                                "horizon_end_index",
                            ]
                            <= inner.validation_end_bar_index
                        ).all():
                            raise RuntimeError("Inner validation target extends beyond the fold end.")
                        backtest = run_event_backtest(
                            bars,
                            decisions,
                            spec,
                            initial_equity=config.initial_equity,
                            notional_fraction=config.notional_fraction,
                            fold_start_index=inner.validation_start_signal_index,
                            fold_end_index=inner.validation_end_bar_index,
                        )
                        inner_metrics.append(_backtest_metrics(backtest, config.initial_equity))
                    except Exception as exc:  # preserve this threshold as failed
                        threshold_failure = f"inner_fold={inner.fold_id}: {_failure_text(exc)}"
                        break

                if threshold_failure is not None:
                    fold_candidate_rows.append(
                        _candidate_failure_row(
                            outer.fold_id,
                            alpha,
                            threshold,
                            candidate_order,
                            threshold_failure,
                        )
                    )
                    continue

                net_returns = np.array([item["net_return"] for item in inner_metrics], dtype=float)
                total_trades = int(sum(int(item["trades"]) for item in inner_metrics))
                if total_trades:
                    expectancy = float(
                        sum(float(item["expectancy"]) * int(item["trades"]) for item in inner_metrics)
                        / total_trades
                    )
                else:
                    expectancy = 0.0
                fold_candidate_rows.append(
                    {
                        "outer_fold": outer.fold_id,
                        "alpha": alpha,
                        "threshold": threshold,
                        "candidate_order": candidate_order,
                        "status": "ok",
                        "failure_reason": "",
                        "inner_folds": len(inner_metrics),
                        "median_net_return": float(np.median(net_returns)),
                        "worst_fold_net_return": float(np.min(net_returns)),
                        "max_drawdown": float(max(float(item["max_drawdown"]) for item in inner_metrics)),
                        "trades": total_trades,
                        "expectancy": expectancy,
                        "selected": False,
                    }
                )

        fold_candidate_frame = pd.DataFrame(fold_candidate_rows)
        successful = fold_candidate_frame[fold_candidate_frame["status"] == "ok"]
        if successful.empty:
            all_candidate_rows.extend(fold_candidate_rows)
            raise AllCandidatesFailedError(pd.DataFrame(all_candidate_rows))
        ranked = successful.sort_values(
            by=[
                "median_net_return",
                "worst_fold_net_return",
                "max_drawdown",
                "trades",
                "candidate_order",
            ],
            ascending=[False, False, True, True, True],
            kind="mergesort",
        )
        chosen = ranked.iloc[0]
        chosen_order = int(chosen["candidate_order"])
        for row in fold_candidate_rows:
            if row["status"] == "ok" and int(row["candidate_order"]) == chosen_order:
                row["selected"] = True
        all_candidate_rows.extend(fold_candidate_rows)

        chosen_alpha = float(chosen["alpha"])
        chosen_threshold = float(chosen["threshold"])
        final_model = fit_ridge_action_value(
            resolved_dataset.X,
            resolved_dataset.targets,
            outer.development_indices,
            alpha=chosen_alpha,
        )
        outer_raw_predictions = predict_action_values(
            final_model,
            resolved_dataset.X.loc[outer.test_indices],
        )
        outer_decisions = decisions_from_action_values(
            outer_raw_predictions,
            threshold=chosen_threshold,
        )
        outer_meta = resolved_dataset.sample_meta.loc[outer.test_indices]
        if not (outer_meta["horizon_end_index"] <= outer.test_end_bar_index).all():
            raise RuntimeError("Outer test target extends beyond the fold end.")

        outer_backtest = run_event_backtest(
            bars,
            outer_decisions,
            spec,
            initial_equity=config.initial_equity,
            notional_fraction=config.notional_fraction,
            fold_start_index=outer.test_start_signal_index,
            fold_end_index=outer.test_end_bar_index,
        )
        trading_metrics = _backtest_metrics(outer_backtest, config.initial_equity)

        outer_frame = outer_raw_predictions.copy()
        outer_frame.insert(0, "outer_fold", outer.fold_id)
        outer_frame["actual_long_net_return"] = resolved_dataset.targets.loc[
            outer.test_indices,
            "long_net_return",
        ].to_numpy(dtype=float)
        outer_frame["actual_short_net_return"] = resolved_dataset.targets.loc[
            outer.test_indices,
            "short_net_return",
        ].to_numpy(dtype=float)
        outer_frame["action"] = outer_decisions["action"].to_numpy()
        outer_frame["chosen_alpha"] = chosen_alpha
        outer_frame["chosen_threshold"] = chosen_threshold
        outer_frame["horizon_end_index"] = outer_meta["horizon_end_index"].to_numpy(dtype=int)
        outer_frame["test_end_bar_index"] = outer.test_end_bar_index
        all_outer_frames.append(outer_frame)

        if not outer_backtest.trades.empty:
            fold_trades = outer_backtest.trades.copy()
            fold_trades.insert(0, "outer_fold", outer.fold_id)
            all_trade_frames.append(fold_trades)

        long_error = (
            outer_frame["pred_long_net_return"] - outer_frame["actual_long_net_return"]
        ).to_numpy(dtype=float)
        short_error = (
            outer_frame["pred_short_net_return"] - outer_frame["actual_short_net_return"]
        ).to_numpy(dtype=float)
        fold_metric_rows.append(
            {
                "outer_fold": outer.fold_id,
                "development_samples": len(outer.development_indices),
                "test_samples": len(outer.test_indices),
                "long_mae": float(np.mean(np.abs(long_error))),
                "short_mae": float(np.mean(np.abs(short_error))),
                "long_rmse": float(np.sqrt(np.mean(np.square(long_error)))),
                "short_rmse": float(np.sqrt(np.mean(np.square(short_error)))),
                "trades": int(trading_metrics["trades"]),
                "net_return": float(trading_metrics["net_return"]),
                "max_drawdown": float(trading_metrics["max_drawdown"]),
                "expectancy": float(trading_metrics["expectancy"]),
                "chosen_alpha": chosen_alpha,
                "chosen_threshold": chosen_threshold,
                "always_flat_net_return": 0.0,
                "test_start_signal_index": outer.test_start_signal_index,
                "test_end_bar_index": outer.test_end_bar_index,
            }
        )

    trades = (
        pd.concat(all_trade_frames, ignore_index=True)
        if all_trade_frames
        else pd.DataFrame(columns=["outer_fold", *TRADE_COLUMNS])
    )
    return NestedWalkForwardResult(
        fold_metrics=pd.DataFrame(fold_metric_rows),
        candidate_metrics=pd.DataFrame(all_candidate_rows),
        inner_oof_predictions=pd.concat(all_oof_frames, ignore_index=True),
        outer_predictions=pd.concat(all_outer_frames, ignore_index=True),
        trades=trades,
        spec=spec,
        config=config,
    )

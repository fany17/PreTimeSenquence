"""Correctness-first v1 research core.

The existing top-level modules remain the v0 baseline. This subpackage is an
incremental implementation of the normative v1 contracts and is not yet a
paper-trading system.
"""

from .spec import AmbiguousBarPolicy, CostSpec, FoldEndPolicy, StrategySpec
from .backtest import EventBacktestResult, run_event_backtest
from .dataset import SupervisedDataset, build_supervised_dataset
from .policy import decisions_from_action_values
from .splits import (
    InnerWalkForwardFold,
    OuterWalkForwardFold,
    PurgedChronologicalSplit,
    inner_walk_forward_folds,
    outer_walk_forward_folds,
    purged_chronological_split,
)
from .targets import make_path_targets
from .training import RidgeActionValueModel, fit_ridge_action_value, predict_action_values
from .walk_forward import (
    AllCandidatesFailedError,
    NestedWalkForwardConfig,
    NestedWalkForwardResult,
    run_nested_walk_forward,
)

__all__ = [
    "AmbiguousBarPolicy",
    "AllCandidatesFailedError",
    "CostSpec",
    "EventBacktestResult",
    "FoldEndPolicy",
    "InnerWalkForwardFold",
    "NestedWalkForwardConfig",
    "NestedWalkForwardResult",
    "OuterWalkForwardFold",
    "PurgedChronologicalSplit",
    "RidgeActionValueModel",
    "StrategySpec",
    "SupervisedDataset",
    "build_supervised_dataset",
    "decisions_from_action_values",
    "fit_ridge_action_value",
    "inner_walk_forward_folds",
    "make_path_targets",
    "outer_walk_forward_folds",
    "predict_action_values",
    "purged_chronological_split",
    "run_event_backtest",
    "run_nested_walk_forward",
]

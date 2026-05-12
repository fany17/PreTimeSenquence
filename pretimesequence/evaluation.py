from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .backtest import BacktestConfig, run_account_backtest
from .model import TrendPredictor
from .targets import TradeOutcomeConfig
from .training import TrainConfig, train_two_stage_xgboost_classifier


@dataclass(frozen=True)
class WalkForwardConfig:
    start: str
    periods: int = 5
    period_days: int = 7
    min_confidences: tuple[float, ...] = (0.55, 0.60, 0.65)
    initial_balance: float = 1.0
    margin: float = 1.0


def run_two_stage_walk_forward(
    df: pd.DataFrame,
    model_dir: str | Path,
    output_path: str | Path,
    walk_config: WalkForwardConfig,
    outcome_config: TradeOutcomeConfig,
    backtest_config: BacktestConfig,
    train_config: TrainConfig | None = None,
    context_frames: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Train before each period and backtest only that period."""

    model_dir = Path(model_dir)
    output_path = Path(output_path)
    model_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    start = pd.Timestamp(walk_config.start)
    results: list[dict] = []
    for fold in range(walk_config.periods):
        test_start = start + pd.Timedelta(days=fold * walk_config.period_days)
        test_end = test_start + pd.Timedelta(days=walk_config.period_days) - pd.Timedelta(minutes=1)
        train_until = test_start - pd.Timedelta(minutes=1)
        model_path = model_dir / f"two_stage_fold{fold + 1}_{train_until:%Y%m%d_%H%M}.json"

        train_metrics = train_two_stage_xgboost_classifier(
            df,
            output_path=model_path,
            outcome_config=outcome_config,
            train_config=train_config,
            train_until=train_until,
            context_frames=context_frames,
        )

        for min_confidence in walk_config.min_confidences:
            predictor = TrendPredictor(model_path, context_frames=context_frames)
            config = BacktestConfig(
                leverage=backtest_config.leverage,
                margin=walk_config.margin,
                take_profit_rate=backtest_config.take_profit_rate,
                stop_loss_rate=backtest_config.stop_loss_rate,
                fee_rate=backtest_config.fee_rate,
                min_confidence=min_confidence,
                initial_balance=walk_config.initial_balance,
            )
            trades, _ = run_account_backtest(
                df,
                predictor,
                config,
                start_time=test_start,
                end_time=test_end,
            )
            final_balance = walk_config.initial_balance if trades.empty else float(trades["end_balance"].iloc[-1])
            wins = int((trades["net_profit"] > 0).sum()) if not trades.empty else 0
            losses = int((trades["net_profit"] <= 0).sum()) if not trades.empty else 0
            results.append(
                {
                    "fold": fold + 1,
                    "train_until": str(train_until),
                    "test_start": str(test_start),
                    "test_end": str(test_end),
                    "min_confidence": min_confidence,
                    "trades": len(trades),
                    "wins": wins,
                    "losses": losses,
                    "initial_balance": walk_config.initial_balance,
                    "final_balance": final_balance,
                    "profit": final_balance - walk_config.initial_balance,
                    "label_counts": train_metrics["label_counts"],
                    "trade_balanced_accuracy": train_metrics["test_trade_balanced_accuracy"],
                    "side_balanced_accuracy": train_metrics["test_side_balanced_accuracy"],
                    "final_balanced_accuracy": train_metrics["test_final_balanced_accuracy"],
                    "model_path": str(model_path),
                }
            )

    result = pd.DataFrame(results)
    result.to_csv(output_path, index=False)
    return result

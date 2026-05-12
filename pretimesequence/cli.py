from __future__ import annotations

import argparse
from pathlib import Path

from .backtest import BacktestConfig, run_account_backtest, run_backtest
from .data import fetch_klines, load_market_data, save_market_data
from .diagnostics import run_diagnostics
from .evaluation import WalkForwardConfig, run_two_stage_walk_forward
from .model import TrendPredictor
from .targets import LabelConfig, TradeOutcomeConfig, label_summary, make_triple_barrier_labels
from .training import train_two_stage_xgboost_classifier, train_xgboost_classifier
from .visualization import write_signal_html


def _parse_context_args(values: list[str] | None) -> dict[str, object]:
    if not values:
        return {}
    contexts = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Context must be SYMBOL=PATH, got: {value}")
        symbol, path = value.split("=", 1)
        contexts[symbol.strip()] = load_market_data(path.strip())
    return contexts


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pretimesequence")
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch", help="Fetch recent Binance OHLCV data.")
    fetch.add_argument("--symbol", default="BTCUSDT")
    fetch.add_argument("--interval", default="1m")
    fetch.add_argument("--limit", type=int, default=5000)
    fetch.add_argument("--start-time", default=None)
    fetch.add_argument("--spot", action="store_true", help="Use spot klines instead of futures klines.")
    fetch.add_argument("--output", default="data/latest.pkl")

    predict = sub.add_parser("predict", help="Predict the latest trend and action.")
    predict.add_argument("--data", required=True)
    predict.add_argument("--model", default="data/xgboost_model.json")
    predict.add_argument("--low", type=float, default=0.3)
    predict.add_argument("--high", type=float, default=0.7)
    predict.add_argument("--context", action="append", default=None, help="Context market data as SYMBOL=PATH.")

    label = sub.add_parser("label", help="Create finite-horizon triple-barrier labels.")
    label.add_argument("--data", required=True)
    label.add_argument("--output", default="data/labelled_market_data.csv")
    label.add_argument("--horizon", type=int, default=20)
    label.add_argument("--atr-multiple", type=float, default=4.0)
    label.add_argument("--min-return", type=float, default=0.005)

    train = sub.add_parser("train", help="Train a chronological multi-class trend model.")
    train.add_argument("--data", required=True)
    train.add_argument("--model", default="data/xgboost_trend_model.json")
    train.add_argument("--horizon", type=int, default=20)
    train.add_argument("--atr-multiple", type=float, default=4.0)
    train.add_argument("--min-return", type=float, default=0.005)
    train.add_argument("--train-until", default=None, help="Only use rows at or before this timestamp for training.")
    train.add_argument("--context", action="append", default=None, help="Context market data as SYMBOL=PATH.")

    train_two = sub.add_parser("train-two-stage", help="Train trade/flat and long/short XGBoost models.")
    train_two.add_argument("--data", required=True)
    train_two.add_argument("--model", default="data/xgboost_two_stage_trend_model.json")
    train_two.add_argument("--horizon", type=int, default=240)
    train_two.add_argument("--leverage", type=float, default=20.0)
    train_two.add_argument("--take-profit-rate", type=float, default=0.38)
    train_two.add_argument("--stop-loss-rate", type=float, default=0.28)
    train_two.add_argument("--fee-rate", type=float, default=0.0005)
    train_two.add_argument("--min-net-profit", type=float, default=0.20)
    train_two.add_argument("--train-until", default=None, help="Only use rows at or before this timestamp for training.")
    train_two.add_argument("--context", action="append", default=None, help="Context market data as SYMBOL=PATH.")

    walk_two = sub.add_parser("walk-forward-two-stage", help="Walk-forward train/backtest two-stage XGBoost.")
    walk_two.add_argument("--data", required=True)
    walk_two.add_argument("--model-dir", default="data/walk_forward_two_stage")
    walk_two.add_argument("--output", default="outputs/walk_forward_two_stage.csv")
    walk_two.add_argument("--start", required=True, help="First test period start timestamp.")
    walk_two.add_argument("--periods", type=int, default=5)
    walk_two.add_argument("--period-days", type=int, default=7)
    walk_two.add_argument("--min-confidence", action="append", type=float, default=None)
    walk_two.add_argument("--horizon", type=int, default=120)
    walk_two.add_argument("--leverage", type=float, default=20.0)
    walk_two.add_argument("--take-profit-rate", type=float, default=0.38)
    walk_two.add_argument("--stop-loss-rate", type=float, default=0.28)
    walk_two.add_argument("--fee-rate", type=float, default=0.0005)
    walk_two.add_argument("--min-net-profit", type=float, default=0.20)
    walk_two.add_argument("--initial-balance", type=float, default=1.0)
    walk_two.add_argument("--margin", type=float, default=1.0)
    walk_two.add_argument("--context", action="append", default=None, help="Context market data as SYMBOL=PATH.")

    backtest = sub.add_parser("backtest", help="Backtest trend signals on local OHLCV data.")
    backtest.add_argument("--data", required=True)
    backtest.add_argument("--model", default="data/xgboost_model.json")
    backtest.add_argument("--output", default="outputs/backtest_trades.csv")
    backtest.add_argument("--leverage", type=float, default=20.0)
    backtest.add_argument("--margin", type=float, default=1.0)
    backtest.add_argument("--min-confidence", type=float, default=0.55)
    backtest.add_argument("--context", action="append", default=None, help="Context market data as SYMBOL=PATH.")

    account = sub.add_parser("account-backtest", help="Sequential account backtest for selected dates.")
    account.add_argument("--data", required=True)
    account.add_argument("--model", default="data/xgboost_trend_model.json")
    account.add_argument("--output", default="outputs/account_backtest_trades.csv")
    account.add_argument("--daily-output", default="outputs/account_backtest_daily.csv")
    account.add_argument("--start", default=None)
    account.add_argument("--end", default=None)
    account.add_argument("--initial-balance", type=float, default=1.0)
    account.add_argument("--margin", type=float, default=1.0)
    account.add_argument("--leverage", type=float, default=20.0)
    account.add_argument("--take-profit-rate", type=float, default=0.38)
    account.add_argument("--stop-loss-rate", type=float, default=0.28)
    account.add_argument("--fee-rate", type=float, default=0.0005)
    account.add_argument("--min-confidence", type=float, default=0.55)
    account.add_argument("--context", action="append", default=None, help="Context market data as SYMBOL=PATH.")

    plot = sub.add_parser("plot", help="Write an HTML candlestick chart with long/short signal markers.")
    plot.add_argument("--data", required=True)
    plot.add_argument("--model", default="data/xgboost_trend_model.json")
    plot.add_argument("--output", default="outputs/signals.html")
    plot.add_argument("--min-confidence", type=float, default=0.55)
    plot.add_argument("--title", default="Trend Signals")
    plot.add_argument("--context", action="append", default=None, help="Context market data as SYMBOL=PATH.")

    diagnose = sub.add_parser("diagnose", help="Diagnose data, label, feature and model quality.")
    diagnose.add_argument("--data", required=True)
    diagnose.add_argument("--output", default="outputs/diagnostics.md")
    diagnose.add_argument("--horizon", type=int, default=20)
    diagnose.add_argument("--atr-multiple", type=float, default=4.0)
    diagnose.add_argument("--min-return", type=float, default=0.005)
    diagnose.add_argument("--context", action="append", default=None, help="Context market data as SYMBOL=PATH.")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "fetch":
        df = fetch_klines(
            symbol=args.symbol,
            interval=args.interval,
            limit=args.limit,
            start_time=args.start_time,
            futures=not args.spot,
        )
        save_market_data(df, args.output)
        print(f"saved {len(df)} rows to {args.output}")
        return 0

    if args.command == "predict":
        df = load_market_data(args.data)
        predictor = TrendPredictor(
            args.model,
            low_threshold=args.low,
            high_threshold=args.high,
            context_frames=_parse_context_args(args.context),
        )
        result = predictor.predict_latest(df)
        print(
            f"{result.timestamp} close={result.close:.8g} "
            f"score={result.score:.3f} trend={result.trend} confidence={result.confidence:.3f}"
        )
        return 0

    if args.command == "label":
        df = load_market_data(args.data)
        labels = make_triple_barrier_labels(
            df,
            LabelConfig(horizon=args.horizon, atr_multiple=args.atr_multiple, min_return=args.min_return),
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        labels.to_csv(output, index=False)
        print(label_summary(labels).to_string(index=False))
        print(f"saved={output}")
        return 0

    if args.command == "train":
        df = load_market_data(args.data)
        metrics = train_xgboost_classifier(
            df,
            output_path=args.model,
            label_config=LabelConfig(horizon=args.horizon, atr_multiple=args.atr_multiple, min_return=args.min_return),
            train_until=args.train_until,
            context_frames=_parse_context_args(args.context),
        )
        print(f"rows={metrics['rows']} model={metrics['model_path']}")
        print(f"label_counts={metrics['label_counts']}")
        print(f"test_balanced_accuracy={metrics['test_balanced_accuracy']:.3f}")
        return 0

    if args.command == "train-two-stage":
        df = load_market_data(args.data)
        metrics = train_two_stage_xgboost_classifier(
            df,
            output_path=args.model,
            outcome_config=TradeOutcomeConfig(
                horizon=args.horizon,
                leverage=args.leverage,
                take_profit_rate=args.take_profit_rate,
                stop_loss_rate=args.stop_loss_rate,
                fee_rate=args.fee_rate,
                min_net_profit=args.min_net_profit,
            ),
            train_until=args.train_until,
            context_frames=_parse_context_args(args.context),
        )
        print(f"rows={metrics['rows']} model={metrics['model_path']}")
        print(f"label_counts={metrics['label_counts']}")
        print(f"trade_counts={metrics['trade_counts']}")
        print(f"test_trade_balanced_accuracy={metrics['test_trade_balanced_accuracy']:.3f}")
        side_acc = metrics["test_side_balanced_accuracy"]
        print(f"test_side_balanced_accuracy={side_acc:.3f}" if side_acc is not None else "test_side_balanced_accuracy=NA")
        print(f"test_final_balanced_accuracy={metrics['test_final_balanced_accuracy']:.3f}")
        return 0

    if args.command == "walk-forward-two-stage":
        df = load_market_data(args.data)
        min_confidences = tuple(args.min_confidence or [0.55, 0.60, 0.65])
        result = run_two_stage_walk_forward(
            df,
            model_dir=args.model_dir,
            output_path=args.output,
            walk_config=WalkForwardConfig(
                start=args.start,
                periods=args.periods,
                period_days=args.period_days,
                min_confidences=min_confidences,
                initial_balance=args.initial_balance,
                margin=args.margin,
            ),
            outcome_config=TradeOutcomeConfig(
                horizon=args.horizon,
                leverage=args.leverage,
                take_profit_rate=args.take_profit_rate,
                stop_loss_rate=args.stop_loss_rate,
                fee_rate=args.fee_rate,
                min_net_profit=args.min_net_profit,
            ),
            backtest_config=BacktestConfig(
                leverage=args.leverage,
                margin=args.margin,
                take_profit_rate=args.take_profit_rate,
                stop_loss_rate=args.stop_loss_rate,
                fee_rate=args.fee_rate,
                initial_balance=args.initial_balance,
            ),
            context_frames=_parse_context_args(args.context),
        )
        print(result[["fold", "min_confidence", "trades", "wins", "losses", "final_balance", "profit"]].to_string(index=False))
        print(f"saved={args.output}")
        return 0

    if args.command == "backtest":
        df = load_market_data(args.data)
        predictor = TrendPredictor(args.model, context_frames=_parse_context_args(args.context))
        trades = run_backtest(
            df,
            predictor,
            BacktestConfig(leverage=args.leverage, margin=args.margin, min_confidence=args.min_confidence),
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        trades.to_csv(output, index=False)
        total = 0.0 if trades.empty else float(trades["net_profit"].sum())
        print(f"trades={len(trades)} total_profit={total:.4f} saved={output}")
        return 0

    if args.command == "account-backtest":
        df = load_market_data(args.data)
        predictor = TrendPredictor(args.model, context_frames=_parse_context_args(args.context))
        config = BacktestConfig(
            leverage=args.leverage,
            margin=args.margin,
            take_profit_rate=args.take_profit_rate,
            stop_loss_rate=args.stop_loss_rate,
            fee_rate=args.fee_rate,
            min_confidence=args.min_confidence,
            initial_balance=args.initial_balance,
        )
        trades, daily = run_account_backtest(df, predictor, config, start_time=args.start, end_time=args.end)
        output = Path(args.output)
        daily_output = Path(args.daily_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        daily_output.parent.mkdir(parents=True, exist_ok=True)
        trades.to_csv(output, index=False)
        daily.to_csv(daily_output, index=False)
        final_balance = args.initial_balance if trades.empty else float(trades["end_balance"].iloc[-1])
        total_profit = final_balance - args.initial_balance
        print(f"trades={len(trades)} initial={args.initial_balance:.4f} final={final_balance:.4f} profit={total_profit:.4f}")
        if not daily.empty:
            print(daily.to_string(index=False))
        print(f"saved={output}")
        print(f"daily_saved={daily_output}")
        return 0

    if args.command == "plot":
        df = load_market_data(args.data)
        predictor = TrendPredictor(args.model, context_frames=_parse_context_args(args.context))
        signals = write_signal_html(
            df,
            predictor,
            output_path=args.output,
            min_confidence=args.min_confidence,
            title=args.title,
        )
        actions = signals["action"].value_counts().to_dict()
        print(f"saved={args.output}")
        print(f"actions={actions}")
        return 0

    if args.command == "diagnose":
        df = load_market_data(args.data)
        result = run_diagnostics(
            df,
            output_path=args.output,
            label_config=LabelConfig(horizon=args.horizon, atr_multiple=args.atr_multiple, min_return=args.min_return),
            context_frames=_parse_context_args(args.context),
        )
        print(f"rows={result['rows']} labelled_rows={result['labelled_rows']} days={result['days']:.1f}")
        print(f"label_counts={result['label_counts']}")
        print(f"baselines={result['baselines']}")
        print(f"model_balanced_accuracy={result['model_balanced_accuracy']:.3f}")
        print(f"saved={result['output_path']}")
        return 0

    raise ValueError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
from pathlib import Path

from .backtest import BacktestConfig, run_backtest
from .data import fetch_klines, load_market_data, save_market_data
from .model import TrendPredictor


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pretimesequence")
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch", help="Fetch recent Binance OHLCV data.")
    fetch.add_argument("--symbol", default="BTCUSDT")
    fetch.add_argument("--interval", default="1m")
    fetch.add_argument("--limit", type=int, default=500)
    fetch.add_argument("--output", default="data/latest.pkl")

    predict = sub.add_parser("predict", help="Predict the latest trend and action.")
    predict.add_argument("--data", required=True)
    predict.add_argument("--model", default="data/xgboost_model.json")
    predict.add_argument("--low", type=float, default=0.3)
    predict.add_argument("--high", type=float, default=0.7)

    backtest = sub.add_parser("backtest", help="Backtest trend signals on local OHLCV data.")
    backtest.add_argument("--data", required=True)
    backtest.add_argument("--model", default="data/xgboost_model.json")
    backtest.add_argument("--output", default="outputs/backtest_trades.csv")
    backtest.add_argument("--leverage", type=float, default=20.0)
    backtest.add_argument("--margin", type=float, default=1.0)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "fetch":
        df = fetch_klines(symbol=args.symbol, interval=args.interval, limit=args.limit)
        save_market_data(df, args.output)
        print(f"saved {len(df)} rows to {args.output}")
        return 0

    if args.command == "predict":
        df = load_market_data(args.data)
        predictor = TrendPredictor(args.model, low_threshold=args.low, high_threshold=args.high)
        result = predictor.predict_latest(df)
        print(
            f"{result.timestamp} close={result.close:.8g} "
            f"score={result.score:.3f} trend={result.trend} confidence={result.confidence:.3f}"
        )
        return 0

    if args.command == "backtest":
        df = load_market_data(args.data)
        predictor = TrendPredictor(args.model)
        trades = run_backtest(df, predictor, BacktestConfig(leverage=args.leverage, margin=args.margin))
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        trades.to_csv(output, index=False)
        total = 0.0 if trades.empty else float(trades["net_profit"].sum())
        print(f"trades={len(trades)} total_profit={total:.4f} saved={output}")
        return 0

    raise ValueError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())

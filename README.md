# PreTimeSequence

用于获取币安 OHLCV 序列、预测最近趋势，并转换成多空操作信号的研究项目。

## 安全配置

不要把真实 API key 写入代码。复制 `.env.example` 到本地环境变量或手动设置：

```powershell
$env:BINANCE_API_KEY="..."
$env:BINANCE_API_SECRET="..."
$env:BINANCE_PROXY_URL="http://127.0.0.1:7897"
```

历史的 `Getkey.py` / `GetkeyReal.py` 仍可被本地 fallback 读取，但已被 `.gitignore` 排除。

## 常用命令

使用已有 conda 环境：

```powershell
conda activate bitc
```

```powershell
python -m pretimesequence.cli label --data data/market_data_DOGE_new.pkl --output data/market_data_DOGE_new_labels.csv
python -m pretimesequence.cli train --data data/market_data_DOGE_new.pkl --model data/xgboost_trend_model.json
python -m pretimesequence.cli predict --data data/market_data_DOGE_new.pkl --model data/xgboost_model.json
python -m pretimesequence.cli backtest --data data/market_data_DOGE_new.pkl --model data/xgboost_model.json --output outputs/backtest_trades.csv
python -m pretimesequence.cli fetch --symbol BTCUSDT --interval 1m --limit 500 --output data/BTCUSDT_latest.pkl
```

当前默认 GT 是三分类 `short/flat/long`，使用 20 根 K 线 horizon、4ATR barrier、0.5% 最小收益门槛，并扣除手续费和滑点。这个默认值只是初始研究参数，不应该直接视为实盘参数。

## 结构

- `pretimesequence/config.py`: API key、代理、testnet 配置读取。
- `pretimesequence/data.py`: 本地数据读取、标准化、Binance K 线获取。
- `pretimesequence/features.py`: 趋势预测特征工程。
- `pretimesequence/targets.py`: 有限 horizon、成本修正的 triple-barrier GT 标签。
- `pretimesequence/training.py`: 按时间顺序切分的三分类训练流程。
- `pretimesequence/model.py`: XGBoost 或 fallback 动量预测。
- `pretimesequence/strategy.py`: 趋势到 `open_long/open_short/hold` 动作。
- `pretimesequence/backtest.py`: 简化止盈止损回测。
- `pretimesequence/cli.py`: 命令行入口。

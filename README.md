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
python -m pretimesequence.cli predict --data data/market_data_DOGE_new.pkl --model data/xgboost_trend_model.json
python -m pretimesequence.cli plot --data data/market_data_DOGE_new.pkl --model data/xgboost_trend_model.json --output outputs/signals.html
python -m pretimesequence.cli diagnose --data data/market_data_DOGE_new.pkl --output outputs/diagnostics.md
python -m pretimesequence.cli backtest --data data/market_data_DOGE_new.pkl --model data/xgboost_trend_model.json --output outputs/backtest_trades.csv --min-confidence 0.55
python -m pretimesequence.cli account-backtest --data data/DOGEUSDT_1m_2024.pkl --model data/xgboost_trend_model_2024.json --start "2024-06-29 00:00:00" --end "2024-07-26 23:59:00" --initial-balance 1 --margin 1 --leverage 20 --take-profit-rate 0.38 --stop-loss-rate 0.28 --fee-rate 0.0005 --min-confidence 0.45
python -m pretimesequence.cli train --data data/DOGEUSDT_1m_recent_600k.pkl --model data/xgboost_trend_model_recent_preholdout.json --train-until "2026-05-04 15:59:00"
python -m pretimesequence.cli account-backtest --data data/DOGEUSDT_1m_recent_600k.pkl --model data/xgboost_trend_model_recent_preholdout.json --start "2026-05-04 16:00:00" --end "2026-05-11 15:59:00" --initial-balance 1 --margin 1 --leverage 20 --take-profit-rate 0.38 --stop-loss-rate 0.28 --fee-rate 0.0005 --min-confidence 0.45
python -m pretimesequence.cli train --data data/DOGEUSDT_1m_recent_600k.pkl --model data/xgboost_trend_model_recent_context_preholdout.json --train-until "2026-05-04 15:59:00" --context BTC=data/BTCUSDT_1m_recent_600k.pkl --context ETH=data/ETHUSDT_1m_recent_600k.pkl --context SOL=data/SOLUSDT_1m_recent_600k.pkl
python -m pretimesequence.cli fetch --symbol DOGEUSDT --interval 1m --start-time "2024-01-01 00:00:00" --limit 300000 --output data/DOGEUSDT_1m_2024.pkl
python -m pretimesequence.cli train --data data/DOGEUSDT_1m_2024.pkl --model data/xgboost_trend_model_2024.json
python -m pretimesequence.cli diagnose --data data/DOGEUSDT_1m_2024.pkl --output outputs/diagnostics_DOGEUSDT_2024.md
python -m pretimesequence.cli plot --data data/DOGEUSDT_1m_2024.pkl --model data/xgboost_trend_model_2024.json --output outputs/signals_DOGEUSDT_2024.html
```

当前默认 GT 是三分类 `short/flat/long`，使用 20 根 K 线 horizon、4ATR barrier、0.5% 最小收益门槛，并扣除手续费和滑点。这个默认值只是初始研究参数，不应该直接视为实盘参数。

当前特征默认使用收益率、波动率、ATR、均线相对偏离、布林 z-score、量能 z-score 和日内周期项，避免把价格水平、均线绝对值这类非平稳变量直接作为主要输入。诊断命令会额外输出 walk-forward 分段验证，优先看它而不是单次随机切分。

回测和 HTML 图默认使用 `min-confidence=0.55` 过滤低置信度信号。低置信度分类结果只作为 `flat/hold` 处理，避免模型被迫每根 K 线都给出交易动作。

`--context SYMBOL=PATH` 可为训练、诊断、预测、回测加入 BTC/ETH/SOL 等市场背景特征。当前实验见 `docs/context_feature_experiment.md`：简单拼接 context 特征未明显改善 XGBoost。

本地 API key 可来自环境变量，也可 fallback 读取 `oldversion/GetkeyReal.py` 或 `oldversion/Getkey.py`。这些文件被 `.gitignore` 排除，不会提交到远端。

## 结构

- `pretimesequence/config.py`: API key、代理、testnet 配置读取。
- `pretimesequence/data.py`: 本地数据读取、标准化、Binance K 线获取。
- `pretimesequence/features.py`: 趋势预测特征工程。
- `pretimesequence/targets.py`: 有限 horizon、成本修正的 triple-barrier GT 标签。
- `pretimesequence/training.py`: 按时间顺序切分的三分类训练流程。
- `pretimesequence/model.py`: XGBoost 或 fallback 动量预测。
- `pretimesequence/strategy.py`: 趋势到 `open_long/open_short/hold` 动作。
- `pretimesequence/backtest.py`: 简化止盈止损回测。
- `pretimesequence/visualization.py`: HTML K 线图和多空信号点。
- `pretimesequence/diagnostics.py`: 数据量、标签、特征和模型质量诊断。
- `pretimesequence/cli.py`: 命令行入口。

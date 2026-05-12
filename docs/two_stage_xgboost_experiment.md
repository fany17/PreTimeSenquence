# 两阶段 XGBoost 与交易口径 GT 实验

## 为什么改 GT

旧标签使用 ATR triple-barrier：

- barrier = `ATR * atr_multiple / close`
- 默认 horizon = `20` 根 1m K 线
- 标签目标是价格趋势分类

但账户回测使用的是另一套交易规则：

- entry = 信号 K 线 close
- 杠杆 = `20`
- 止盈 = `0.38 / 20 = 1.9%` 价格移动
- 止损 = `0.28 / 20 = 1.4%` 价格移动
- 手续费 = 开平双边计入

这两个目标不一致。旧模型的分类正确率即使提高，也不一定对应交易赚钱。

## 新 GT

新增 `TradeOutcomeConfig` 和 `make_trade_outcome_labels`。

每个时点分别模拟 long 和 short：

1. 在未来 `horizon` 根 K 线内检查固定 TP/SL。
2. 同一根 K 线同时触发时，保守地优先按止损处理。
3. 计算扣除手续费后的 net profit。
4. 如果最佳方向的 net profit 大于 `min_net_profit`，标为 `long` 或 `short`。
5. 否则标为 `flat`。

这让 GT 和账户回测口径一致。

## 两阶段模型

新增命令：

```bash
python -m pretimesequence.cli train-two-stage --data data/SOLUSDT_1m_recent_600k.pkl --model data/xgboost_two_stage_SOL_recent_context_h120_edge020.json --train-until "2026-05-04 15:59:00" --horizon 120 --leverage 20 --take-profit-rate 0.38 --stop-loss-rate 0.28 --fee-rate 0.0005 --min-net-profit 0.20 --context BTC=data/BTCUSDT_1m_recent_600k.pkl --context ETH=data/ETHUSDT_1m_recent_600k.pkl --context DOGE=data/DOGEUSDT_1m_recent_600k.pkl
```

模型结构：

- 第一阶段：`trade / flat`
- 第二阶段：在 `trade` 样本内判断 `short / long`

预测时，`trend_score` 使用第一阶段的 `trade_prob`，用于账户回测的 `min-confidence` 过滤。

## SOL 实验结果

训练和测试边界：

- 主标的：`SOLUSDT`
- context：`BTCUSDT / ETHUSDT / DOGEUSDT`
- 训练截止：`2026-05-04 15:59:00`
- 最近一周 holdout：`2026-05-04 16:00:00` 到 `2026-05-11 15:59:00`

初始参数 `horizon=240, min_net_profit=0.02`：

- label counts: `short=243836`, `long=242287`, `flat=103557`
- trade 占比约 `82%`
- trade balanced accuracy: `0.542`
- side balanced accuracy: `0.515`
- final balanced accuracy: `0.368`
- 最近一周 `min-confidence=0.40/0.45/0.50` 都是 `1U -> 0.3896U`

判断：`min_net_profit=0.02` 太宽松，几乎把大部分时点都标成可交易，第一阶段没有足够过滤意义。

收紧后的参数比较：

| model | label counts | trade balanced acc | side balanced acc | final balanced acc | recent week result |
| --- | --- | ---: | ---: | ---: | --- |
| `h120 edge010` | `flat=319361, short=135361, long=135078` | `0.600` | `0.503` | `0.401` | conf `0.55`: `1U -> 2.8404U` |
| `h240 edge010` | `flat=234674, short=178199, long=176807` | `0.584` | `0.499` | `0.389` | conf `0.55`: `1U -> 1.8923U` |
| `h120 edge020` | `flat=440747, short=75241, long=73812` | `0.633` | `0.504` | `0.430` | conf `0.55`: `1U -> 2.9928U` |

`h120 edge020` 阈值敏感性：

| min-confidence | trades | final | profit |
| --- | ---: | ---: | ---: |
| `0.55` | 10 | `2.9928U` | `+1.9928U` |
| `0.60` | 11 | `0.8904U` | `-0.1096U` |
| `0.65` | 5 | `0.3772U` | `-0.6228U` |

HTML 信号图：

- `outputs/signals_SOLUSDT_recent_week_two_stage_h120_edge020_conf055.html`

## 当前判断

两阶段结构和交易口径 GT 是正确方向，但当前结果不能直接当成稳定策略：

1. `h120 edge020` 是在观察最近一周结果后挑出的参数，存在选择偏差。
2. 阈值从 `0.55` 到 `0.60` 结果反转，说明概率校准仍然不好。
3. 第二阶段 `short / long` balanced accuracy 约 `0.50`，方向判断仍接近随机。
4. 近期盈利主要来自交易过滤改善，不是方向模型明显变强。

下一步应该做：

1. 固定 `horizon=120, min_net_profit=0.20`，在更早的多段时间做 walk-forward 账户回测。
2. 增加 side 模型的特征或样本权重，否则方向判断没有明显优势。
3. 做概率校准，避免 `0.55/0.60` 这种阈值不稳定。
4. 在这些步骤前，不建议上神经网络；当前瓶颈仍然是 GT 和校准。

# SOL 两阶段 XGBoost walk-forward 验证

## 设置

目标：验证 `horizon=120, min_net_profit=0.20` 的两阶段 XGBoost 是否只是在最近一周偶然盈利。

主标的：

- `SOLUSDT`

context：

- `BTCUSDT`
- `ETHUSDT`
- `DOGEUSDT`

每个 fold 的流程：

1. 只使用测试周开始前的数据训练。
2. 测试接下来 7 天。
3. 初始资金固定为 `1U`。
4. 每个 fold 单独重置资金，避免前一个 fold 的资金状态影响下一个 fold。
5. 同时测试 `min-confidence=0.55 / 0.60 / 0.65`。

命令：

```bash
python -m pretimesequence.cli walk-forward-two-stage --data data/SOLUSDT_1m_recent_600k.pkl --model-dir data/walk_forward_two_stage_SOL_h120_edge020 --output outputs/walk_forward_SOLUSDT_two_stage_h120_edge020.csv --start "2026-04-07 00:00:00" --periods 5 --period-days 7 --horizon 120 --leverage 20 --take-profit-rate 0.38 --stop-loss-rate 0.28 --fee-rate 0.0005 --min-net-profit 0.20 --initial-balance 1 --margin 1 --min-confidence 0.55 --min-confidence 0.60 --min-confidence 0.65 --context BTC=data/BTCUSDT_1m_recent_600k.pkl --context ETH=data/ETHUSDT_1m_recent_600k.pkl --context DOGE=data/DOGEUSDT_1m_recent_600k.pkl
```

## 结果

| fold | min-confidence | trades | wins | losses | final |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.55 | 14 | 3 | 11 | 0.0497 |
| 1 | 0.60 | 15 | 5 | 10 | 0.1313 |
| 1 | 0.65 | 13 | 5 | 8 | 0.2680 |
| 2 | 0.55 | 18 | 6 | 12 | 0.0876 |
| 2 | 0.60 | 13 | 6 | 7 | 1.0595 |
| 2 | 0.65 | 11 | 5 | 6 | 0.9141 |
| 3 | 0.55 | 8 | 3 | 5 | 0.4229 |
| 3 | 0.60 | 7 | 2 | 5 | 0.3109 |
| 3 | 0.65 | 3 | 1 | 2 | 0.6662 |
| 4 | 0.55 | 5 | 4 | 1 | 2.1395 |
| 4 | 0.60 | 5 | 3 | 2 | 1.4794 |
| 4 | 0.65 | 2 | 1 | 1 | 1.0597 |
| 5 | 0.55 | 10 | 9 | 1 | 3.9397 |
| 5 | 0.60 | 10 | 7 | 3 | 2.3373 |
| 5 | 0.65 | 4 | 1 | 3 | 0.5390 |

按阈值汇总：

| min-confidence | mean final | median final | min final | max final | total trades | win rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.55 | 1.3279 | 0.4229 | 0.0497 | 3.9397 | 55 | 0.455 |
| 0.60 | 1.0637 | 1.0595 | 0.1313 | 2.3373 | 50 | 0.460 |
| 0.65 | 0.6894 | 0.6662 | 0.2680 | 1.0597 | 33 | 0.394 |

如果把每周结果复利相乘：

| min-confidence | compound final |
| ---: | ---: |
| 0.55 | 0.0155 |
| 0.60 | 0.1496 |
| 0.65 | 0.0932 |

## 判断

最近一周 `1U -> 2.9928U` 不是稳定策略证据。walk-forward 显示：

1. 前 3 周多数阈值亏损，最近 2 周盈利很强。
2. `0.55` 的平均 final 最高，但中位数只有 `0.4229`，说明结果被最后一周拉高。
3. `0.60` 的中位数接近 `1.06`，但最差周只有 `0.1313`，风险仍很高。
4. 所有阈值复利结果都低于 `1U`，说明跨 regime 连续运行会亏。

当前结论：

- 两阶段 + 交易口径 GT 比旧模型更接近真实交易目标。
- 但当前模型不稳定，不能直接进入实盘。
- 主要问题是 regime 过滤缺失：模型在部分周能够抓住趋势，但在错误 regime 会连续止损。

下一步应该做：

1. 增加 regime filter，例如 BTC/SOL 波动率状态、趋势强度、成交量状态。
2. 训练一个第三阶段过滤器，专门判断当前 regime 是否允许开仓。
3. 把评估目标从单周收益改为 walk-forward 的中位数、最大回撤和复利收益。
4. 在这些稳定性指标改善前，不应上神经网络。

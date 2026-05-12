# Forecast features 实验

## 动机

当前模型的核心问题不只是分类模型，而是开仓依据。GT 应保持客观，也就是未来按真实 TP/SL 和手续费能否赚钱；但模型输入需要包含“基于当前及过去数据对未来的预测项”。

因此本次没有把预测混入 GT，而是新增 past-only forecast features：

- 主标的 EWMA 预期收益：`forecast_ewm_ret_5 / 20 / 60`
- 主标的 momentum 组合预测：`forecast_mom_ret_20 / 60`
- 波动率归一化预测：`forecast_z_20 / 60`
- 上涨/下跌 proxy 概率：`forecast_up_prob_proxy_20 / forecast_down_prob_proxy_20`
- 预测收益相对 ATR 成本：`forecast_edge_to_cost_20`
- context 币种预测项：`btc/eth/doge_forecast_ret_20 / 60` 和 `forecast_z_20`

这些特征只使用当前和过去 K 线，不使用未来收益，因此不会泄漏。

## 验证设置

主标的：

- `SOLUSDT`

context：

- `BTCUSDT`
- `ETHUSDT`
- `DOGEUSDT`

模型：

- 两阶段 XGBoost
- GT: `horizon=120`, `min_net_profit=0.20`
- walk-forward: 5 个 7 天测试段
- 每个测试段开始前重新训练
- 阈值：`0.55 / 0.60 / 0.65`

输出：

- `outputs/walk_forward_SOLUSDT_two_stage_h120_edge020_forecast.csv`

## 结果对比

原两阶段模型：

| min-confidence | mean final | median final | min final | max final | trades | win rate | compound final |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.55 | 1.3279 | 0.4229 | 0.0497 | 3.9397 | 55 | 0.4545 | 0.0155 |
| 0.60 | 1.0637 | 1.0595 | 0.1313 | 2.3373 | 50 | 0.4600 | 0.1496 |
| 0.65 | 0.6894 | 0.6662 | 0.2680 | 1.0597 | 33 | 0.3939 | 0.0932 |

加入 forecast features 后：

| min-confidence | mean final | median final | min final | max final | trades | win rate | compound final |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.55 | 1.2658 | 0.2073 | 0.0332 | 3.8782 | 56 | 0.4107 | 0.0040 |
| 0.60 | 1.0760 | 0.4229 | 0.1380 | 2.3374 | 53 | 0.4717 | 0.0998 |
| 0.65 | 0.6883 | 0.5211 | 0.2682 | 1.4197 | 42 | 0.4048 | 0.0587 |

## 判断

简单加入 forecast proxy 没有改善稳定性，复利结果反而更差。

这不否定“预测项作为输入”的方向，但说明当前预测项太粗糙：

1. EWMA/momentum forecast 仍然本质上是线性动量外推。
2. 它没有直接预测 TP/SL 命中概率。
3. 它没有学习不同 regime 下预测项是否有效。
4. 它增加了特征数量，可能让 XGBoost 更容易贴合近期 regime。

下一步更合理的方向：

1. 先训练独立预测模型，输出未来 `5/20/60/120` 分钟收益分布或分位数。
2. 把该预测模型的 out-of-fold 预测结果作为二阶段交易模型的输入。
3. 预测项应该包括：
   - `P(long TP before SL)`
   - `P(short TP before SL)`
   - expected net profit long / short
   - forecast uncertainty
4. 只有用 out-of-fold 预测项，才能避免同一训练集内的预测泄漏。

当前结论：

- GT 改成交易口径是必要的。
- “预测项加入指标”也是必要方向。
- 但预测项不能只是手工 proxy，应该是独立预测模型的 out-of-fold 输出。

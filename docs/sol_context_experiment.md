# SOL context 特征实验

## 目的

DOGE 最近成交和波动状态不理想，本实验把主交易标的切到 `SOLUSDT`，仍然先使用 XGBoost，不直接增加模型复杂度。

## 数据

主交易标的：

- `SOLUSDT`: `data/SOLUSDT_1m_recent_600k.pkl`

市场背景：

- `BTCUSDT`: `data/BTCUSDT_1m_recent_600k.pkl`
- `ETHUSDT`: `data/ETHUSDT_1m_recent_600k.pkl`
- `DOGEUSDT`: `data/DOGEUSDT_1m_recent_600k.pkl`

四个数据文件对齐范围：

- 起始: `2025-03-21 00:00:00`
- 结束: `2026-05-11 15:59:00`
- 行数: `600000`

## 训练边界

- 训练截止: `2026-05-04 15:59:00`
- 最近一周 holdout: `2026-05-04 16:00:00` 到 `2026-05-11 15:59:00`

训练命令：

```bash
python -m pretimesequence.cli train --data data/SOLUSDT_1m_recent_600k.pkl --model data/xgboost_trend_model_SOL_recent_context_preholdout.json --train-until "2026-05-04 15:59:00" --context BTC=data/BTCUSDT_1m_recent_600k.pkl --context ETH=data/ETHUSDT_1m_recent_600k.pkl --context DOGE=data/DOGEUSDT_1m_recent_600k.pkl
```

## 分类结果

- 训练样本行数: `589887`
- 标签分布: `short=254943`, `flat=233414`, `long=101530`
- 训练命令输出的 chronological test balanced accuracy: `0.438`
- 诊断报告中的 XGBoost balanced accuracy: `0.432`
- majority baseline balanced accuracy: `0.333`
- momentum20 baseline balanced accuracy: `0.348`

诊断报告显示测试集预测分布：

- `flat`: `0.709`
- `long`: `0.226`
- `short`: `0.066`

主要问题是 `short` recall 很低：

- `short` precision: `0.448`
- `short` recall: `0.073`
- `long` precision: `0.223`
- `long` recall: `0.440`

## 最近一周账户回测

共同参数：

- 初始资金: `1U`
- 单笔保证金上限: `1U`
- 杠杆: `20`
- 止盈: `0.38`
- 止损: `0.28`
- 手续费率: `0.0005`

结果：

| min-confidence | trades | final | profit |
| --- | ---: | ---: | ---: |
| `0.40` | 9 | `0.7183U` | `-0.2817U` |
| `0.45` | 2 | `0.4900U` | `-0.5100U` |
| `0.50` | 0 | `1.0000U` | `0.0000U` |

`0.40` 阈值下交易明细摘要：

- 9 笔交易，4 笔止盈，5 笔止损。
- 亏损主要发生在 `2026-05-08` 到 `2026-05-10` 的连续反向信号。
- 单看 balanced accuracy 改善不够，交易端仍然会被高杠杆止损放大。

`0.45` 阈值下只触发两笔交易，均在 `2026-05-10` 止损：

- `short`: `93.45 -> 94.7583`
- `long`: `95.45 -> 94.1137`

## HTML 图

信号图已生成到：

- `outputs/signals_SOLUSDT_recent_week_context_conf040.html`

该文件是本地输出，按规则不提交到 Git。

## 结论

把 base 从 DOGE 改为 SOL 后，分类指标确实改善，说明 DOGE 最近不是一个很好的主交易标的。但最近一周账户回测仍然亏损，问题不在于单纯缺少 context 指标。

当前最主要的问题：

1. `short` 类召回过低，模型错过或误判空头 regime。
2. `long` 的 precision 偏低，错误多头在高杠杆下亏损很快。
3. 置信度没有很好校准，`0.40` 到 `0.45` 的筛选没有稳定提升收益。
4. 三分类模型直接输出交易方向，和真实交易目标不完全一致。

下一步建议先改模型结构，不急着上神经网络：

1. 两阶段 XGBoost：先预测 `trade/flat`，再在可交易样本中预测 `long/short`。
2. 对概率做 calibration，用校准后的概率做仓位和入场过滤。
3. 加入滚动训练或 regime 分组，避免一个模型覆盖所有市场状态。
4. 等上述结构验证后，再考虑轻量神经网络，例如 TCN/LSTM/Transformer encoder。

是否使用神经网络：现在还不是第一优先级。当前数据量足够尝试神经网络，但交易失败的核心更像是目标定义、概率校准和 regime 处理问题。直接换神经网络可能只是把错误目标学得更复杂。

# 多币种 context 特征实验

## 数据

主交易标的：

- `DOGEUSDT`: `data/DOGEUSDT_1m_recent_600k.pkl`

市场背景：

- `BTCUSDT`: `data/BTCUSDT_1m_recent_600k.pkl`
- `ETHUSDT`: `data/ETHUSDT_1m_recent_600k.pkl`
- `SOLUSDT`: `data/SOLUSDT_1m_recent_600k.pkl`

四个数据文件完全对齐：

- 起始: `2025-03-21 00:00:00`
- 结束: `2026-05-11 15:59:00`
- 行数: `600000`

## 新增指标

对每个 context symbol 增加：

- `ret_1 / ret_5 / ret_20 / ret_60`
- `vol_20 / vol_60`
- `range_pct`
- DOGE 相对该币种的 `ret_20 / ret_60` 相对强弱

这些指标通过 timestamp 对齐，用过去可见的 context K 线，避免用未来数据。

## 训练边界

- 训练截止: `2026-05-04 15:59:00`
- 最近一周 holdout: `2026-05-04 16:00:00` 到 `2026-05-11 15:59:00`

## 结果

不带 context 的 recent 模型：

- 内部测试 balanced accuracy: `0.405`
- 最近一周 `min-confidence=0.40`: final `0.8967U`
- 最近一周 `min-confidence=0.45`: final `1.7204U`

带 BTC/ETH/SOL context 的模型：

- 内部测试 balanced accuracy: `0.404`
- 最近一周 `min-confidence=0.40`: final `0.9637U`
- 最近一周 `min-confidence=0.45`: final `0.4900U`
- 最近一周 `min-confidence=0.50`: final `1.0000U`，不交易

## 诊断

context 诊断报告中，`btc_vol_60` 进入 top mutual-information 特征，说明市场背景确实有信息。但整体模型没有改善，最近一周交易也更差。

当前结论：

1. 数据量增加是有效的。
2. 简单拼接 BTC/ETH/SOL context 特征没有直接改善 XGBoost。
3. 下一步更应该改模型结构或训练目标，而不是继续简单堆指标。

建议下一步：

- 两阶段模型：先判断 `trade/flat`，再判断 `long/short`。
- 按时间滚动训练，不用单一固定模型覆盖所有 regime。
- 单独做 probability calibration，避免 `0.40/0.45` 阈值含义漂移。
- 对 context 特征做 regime 特征，例如 BTC 高/低波动分组，而不是只把收益率直接拼进去。

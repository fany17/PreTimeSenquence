# Project Status

更新时间：2026-07-11

## 一句话判断

仓库已经具备量化研究原型的主要模块，但 Ground Truth、交易时序和回测状态不统一，现有结果不能支持实盘；当前工作重点是定义并实现统一的 15 分钟研究合同，而不是继续增加指标或更换复杂模型。

## 当前保留的 v0 能力

- Binance futures/spot K 线获取与本地 PKL/CSV 读写；
- 1 分钟 OHLCV 标准化；
- 价格、波动率、趋势、成交量、日内周期和多币种 context 特征；
- ATR triple-barrier 三分类标签；
- 基于固定 TP/SL 的 trade outcome 标签；
- 单阶段和两阶段 XGBoost；
- chronological split、诊断和部分 walk-forward；
- 简化交易与账户回测；
- DOGE、SOL 和 context/forecast proxy 的历史实验记录。

这些内容作为 v0 基线保留，不等于 v1 已完成。

## 已确认的关键问题

### P0：会影响结论有效性

1. 项目同时存在 20、120 和 240 分钟预测目标，与“约 15 分钟”目标不一致。
2. 两阶段标签和账户回测使用当前 K 线 close 入场，存在不可执行的同收盘价成交假设。
3. ATR GT 与 trade outcome GT 的入场、horizon、成本和滑点口径不同。
4. trade outcome 同时模拟 long/short 后选择事后最优方向，第一阶段与方向阶段目标脱节。
5. 回测预先生成 position action，无法在 TP/SL 后同步恢复 flat 状态。
6. 内部时间切分没有按标签 horizon purge/embargo。
7. fold 末持仓可能使用测试区间之外的数据退出。

### P1：会影响稳定性与复现

1. 数据、模型和输出仅在本地，缺少 manifest、checksum 和配置快照。
2. 概率未校准，置信度阈值对结果高度敏感。
3. 缺少 mark price、funding、open interest 和更细成交信息。
4. 缺少自动化测试、CI、固定依赖和实验追踪。
5. 主要评价仍偏重 balanced accuracy，账户风险指标不足。

## 已有结果的正确解读

- 扩展到约 600,000 根 1m K 线后，分类指标较短样本改善，说明数据覆盖很重要。
- 简单拼接 BTC/ETH/SOL context 没有稳定改善结果。
- 手工 forecast proxy 和 forecast confidence 没有改善 walk-forward 稳定性。
- 两阶段模型改善了部分时期的交易过滤，但方向模型约等于随机。
- SOL 五周 walk-forward 的盈利主要集中在后两周，所有阈值的报告复利结果均低于初始资金。

这些结果支持“先修正任务、标签、验证和回测，再讨论模型复杂度”。

## 已冻结的项目决策

| ID | 决策 | 状态 |
| --- | --- | --- |
| D-001 | 使用 1m 数据研究约 15m 的未来路径 | 已确定 |
| D-002 | 信号在 bar t close 形成，最早在 bar t+1 成交 | 已确定 |
| D-003 | 杠杆不进入基础市场预测目标，进入策略成本和风险层 | 已确定 |
| D-004 | flat 主要由正期望值和置信度门槛产生，不强制作为原始市场类别 | 已确定 |
| D-005 | v0 代码与旧实验暂时保留，待 v1 验证后再归档 | 已确定 |
| D-006 | XGBoost/线性模型作为第一阶段基线，暂不上深度学习 | 已确定 |
| D-007 | paper trading 前必须通过 nested walk-forward 与冻结 holdout | 已确定 |

## 当前阶段

当前处于 `Phase 0: specification freeze`：

- [x] 完成仓库审查；
- [x] 重写标准文档；
- [x] 定义目标策略、数据、GT 和验证原则；
- [ ] 建立单一配置对象 `StrategySpec`；
- [ ] 重写标签和事件驱动回测；
- [ ] 建立测试与可复现数据清单；
- [ ] 重跑基线和 outer walk-forward。

详细计划见 [docs/ROADMAP.md](docs/ROADMAP.md)。


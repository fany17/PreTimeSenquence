# Project Status

更新时间：2026-07-14

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

## 当前已实现的 v1 correctness core

以下内容属于 **current implementation**，位于 `pretimesequence/v1/`，不替代现有 v0 CLI：

- 冻结、强类型且 fail-fast 校验的单一 `StrategySpec`；
- 共享的 next-bar fill、TP/SL、ambiguous bar、fee、spread、slippage 和 funding payment 语义；
- 向量化生成 15 分钟 `return_h`、MFE、MAE，以及不含杠杆、不事后选择最优方向的 long/short action outcomes；保留慢速 reference 实现用于语义差分测试；
- 基于 horizon 的 purged/embargoed chronological split；
- 逐 bar 维护 pending/flat/long/short 状态的单持仓事件回测；
- TP/SL 后立即恢复 flat、最大持有期退出和 fold-end forced close；
- 基于风险预算、止损距离和成本的默认名义仓位，并受 leverage cap 约束；
- 以全局 `signal_index` 对齐且不做隐式填补的 v1 supervised dataset，以及 OHLCV 内存指纹防错配；
- 基于真实 `horizon_end_index` 的 outer/inner interval-aware walk-forward folds；
- deterministic Ridge long/short action-value baseline、inner chronological OOF、inner-only alpha/threshold 选择和 outer 单次评估；
- 内存中的 fold、候选、OOF、outer prediction 和 trade 审计表；
- 独立 v1 CLI，可从可信本地 PKL/CSV 一条命令启动 Ridge nested walk-forward，并原子持久化 run manifest、输入/代码/产物 checksum、配置快照和全部审计 CSV；当前仍不持久化模型。

以上语义已由 43 项合成数据单元测试验证，命令为：

```bash
conda run -n bitc python -B -m unittest discover -s tests -v
```

此外已对本地 SOL 1m 数据的一段连续 500 bars 完成单-fold CLI smoke，验证 4 个预注册候选、inner OOF、outer prediction、事件回测及六类审计文件能够贯通；最终输出保存在被 Git 忽略的 `outputs/v1_smoke_sol_20260714_v6/`，五个 CSV 的 hash、bytes 和 rows 均已回读核验，并与性能优化前的 v4 输出数值等价，不属于策略有效性实验。

本地 SOL 600,000 bars 的默认训练窗口全量预检已通过：599,857 个有效样本、5 个 outer folds、每个 outer fold 3 个 inner folds、每个 outer test 43,185 个样本；在当前 `bitc` 环境下数据加载、dataset/target 构造和 folds 生成合计约 18.17 秒。该预检证明正式 baseline 已具备启动条件，但尚未把完整 600,000-bar 模型结果作为 validated 策略结论。

这里的 **validated** 仅指合成边界条件和最小真实数据 smoke 下的代码行为，不表示模型有效、策略盈利或已达到 paper trading 门槛。

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

1. CLI run 已记录输入 checksum、配置和代码状态，但仍缺少独立数据集版本 manifest、模型 artifact manifest 和锁定依赖。
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

当前处于 `Phase 0` 收尾并进入 `Phase 1: correctness first` 的首批实现：

- [x] 完成仓库审查；
- [x] 重写标准文档；
- [x] 定义目标策略、数据、GT 和验证原则；
- [x] 建立单一配置对象 `StrategySpec` 及字段校验；
- [x] 实现独立 v1 path targets、purged split 和事件回测核心；
- [x] 为 next-bar、TP/SL、ambiguous bar、成本、funding、fold-end 和数据契约建立合成测试；
- [x] 将 v1 核心接入 deterministic Ridge action-value 训练和 interval-aware nested walk-forward；
- [x] 为当前 Ridge baseline 持久化 run manifest、输入 checksum、配置快照和统一审计 CSV；
- [ ] 实现概率校准、更多预注册基线以及数据集/模型 artifact manifest；
- [ ] 实现 maintenance margin、liquidation、交易所规则和真实 funding timestamp 对齐；
- [ ] 建立可复现数据 manifest 与小型版本化 fixture；
- [ ] 重跑基线和 outer walk-forward。

详细计划见 [docs/ROADMAP.md](docs/ROADMAP.md)。


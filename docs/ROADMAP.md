# Roadmap

状态：**Current plan**

## Phase 0 — Specification freeze

目标：停止继续堆叠特征和模型，冻结唯一研究问题。

- [x] 审查 v0 数据、标签、训练和回测；
- [x] 明确 1m 数据、15m horizon 和 next-bar execution；
- [x] 将杠杆从基础 target 移至风险层；
- [x] 建立项目状态、架构、数据、GT 和验证文档；
- [x] 将规范参数实现为单一 `StrategySpec`；
- [x] 为当前 v1 核心配置字段建立类型 schema、mapping 和 fail-fast validation。

当前边界：v1 target、split 和 backtest 已引用同一配置对象；paper trading 尚未实现，因此 Phase 0 的最终完成门槛尚未全部满足。

## Phase 1 — Correctness first

目标：修复会使研究结论无效的逻辑。

- [x] 在 v1 correctness core 实现 next-bar entry；
- [x] 实现 15m return/MFE/MAE/action outcome targets；
- [x] v1 target 不再事后选择最优方向，也不存在 long tie bias；v0 历史实现继续保留；
- [x] 实现 purged/embargoed chronological split；
- [x] 实现单 symbol、单持仓事件驱动回测状态机；
- [x] 实现 fold 末强制平仓；
- [x] 实现 fee、spread、slippage 和实际 payment bar funding 规则；
- [ ] 实现 maintenance margin、liquidation、tick size、step size 和 minimum notional；
- [x] 为当前已实现的关键时序和边界建立合成单元测试；
- [x] 将 v1 correctness core 接入 deterministic Ridge action-value 训练和 interval-aware nested walk-forward；
- [x] 为当前 Ridge baseline 实现一键训练入口和持久化 run/checksum/config/fold/candidate/OOF/outer/trade 审计 bundle；
- [ ] 实现 calibration、更多基线、独立数据集 manifest 和模型 artifact manifest。

当前验证：43 项合成测试和一段连续 500-bar 的真实数据 CLI smoke 已通过，六类带完整性记录的审计文件已原子落盘；600,000-bar 默认窗口预检已在约 18.17 秒内构造 5 个 outer folds × 每折 3 个 inner folds。liquidation、交易所规则、calibration 和模型持久化仍未完成，因此 Phase 1 尚未整体完成。

## Phase 2 — Reproducible data

目标：建立可追溯的多币种期货数据集。

- [ ] raw/validated/features/targets 分层；
- [ ] manifest、schema、checksum；
- [ ] 多年份 BTC/ETH/SOL/DOGE 1m 数据；
- [ ] mark/index price、funding、OI、taker statistics；
- [ ] 数据缺口和合约规则报告；
- [ ] 小型合成测试 fixture。

完成门槛：任一实验可以由 manifest、配置和脚本重新生成。

## Phase 3 — Baselines and honest evaluation

目标：建立最小但可信的研究基线。

- [x] 在 v1 fold report 中加入 always-flat 基线；
- [ ] matched-turnover random、momentum、mean-reversion、breakout；
- [x] 实现 deterministic Ridge long/short action-value baseline；
- [ ] logistic 与 XGBoost baseline；
- [x] 实现 inner OOF、inner-only 选择和 outer 单次评估的 nested walk-forward API；
- [ ] 多币种、多 regime 分层；
- [ ] 成本和参数敏感性；
- [x] 持久化当前 Ridge baseline 的 forecast/trading 审计表和 run manifest；
- [ ] 增加 calibration report、模型 artifact manifest 及其跨模型统一 schema。

完成门槛：所有候选模型在同一 outer folds 上与基线比较，不允许按测试结果选择参数。

## Phase 4 — Forecast and meta decision

目标：从手工 forecast proxy 转向独立可校准预测。

- [ ] 预测 5/15/30/60m return distribution；
- [ ] 预测 MFE/MAE；
- [ ] 预测 long/short TP-before-SL probability；
- [ ] chronological OOF predictions；
- [ ] probability calibration；
- [ ] direction/action value -> meta trade filter；
- [ ] regime gate；
- [ ] 风险预算与仓位模型。

完成门槛：净收益与稳定性提升来自预注册 outer evaluation，而非单一近期窗口。

## Phase 5 — Paper trading

目标：验证实时数据、状态同步和执行差异。

- [ ] 实时 feed 与完整 bar 处理；
- [ ] 模型/feature schema 校验；
- [ ] order/position/account state machine；
- [ ] disconnect/reconnect 与 idempotency；
- [ ] daily loss、drawdown、consecutive-loss kill switch；
- [ ] paper trading 报告与回测偏差分析。

完成门槛：连续 paper trading 与同区间仿真差异可解释，且风险规则全部触发正确。

## 暂缓事项

在 Phase 3 前暂不优先：

- LSTM/TCN/Transformer；
- 强化学习；
- 自动实盘下单；
- 大规模超参数搜索；
- 更多手工技术指标。


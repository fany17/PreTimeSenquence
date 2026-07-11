# Roadmap

状态：**Current plan**

## Phase 0 — Specification freeze

目标：停止继续堆叠特征和模型，冻结唯一研究问题。

- [x] 审查 v0 数据、标签、训练和回测；
- [x] 明确 1m 数据、15m horizon 和 next-bar execution；
- [x] 将杠杆从基础 target 移至风险层；
- [x] 建立项目状态、架构、数据、GT 和验证文档；
- [ ] 将规范参数实现为单一 `StrategySpec`；
- [ ] 为所有配置字段建立 schema 与 validation。

完成门槛：target、backtest 和未来 paper trading 均引用同一配置对象。

## Phase 1 — Correctness first

目标：修复会使研究结论无效的逻辑。

- [ ] 实现 next-bar entry；
- [ ] 实现 15m return/MFE/MAE/action outcome targets；
- [ ] 删除事后多空最优选择及 long tie bias；
- [ ] 实现 purged/embargoed splits；
- [ ] 重写事件驱动回测状态机；
- [ ] fold 末强制平仓；
- [ ] 添加成本、滑点、funding 和 liquidation 规则；
- [ ] 为关键边界建立单元测试。

完成门槛：合成数据上的 entry、TP/SL、账户权益和 split 测试全部通过。

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

- [ ] always-flat、random、momentum、mean-reversion、breakout；
- [ ] logistic/linear 与 XGBoost；
- [ ] nested walk-forward；
- [ ] 多币种、多 regime 分层；
- [ ] 成本和参数敏感性；
- [ ] 统一 forecast、calibration 和 trading report。

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


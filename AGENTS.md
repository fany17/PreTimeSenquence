# AGENTS.md

本文件规定 AI/Codex 在本仓库中的工作方式。

## 开始任何任务前

按顺序读取：

1. `README.md`
2. `PROJECT_STATUS.md`
3. `docs/STRATEGY_SPEC.md`
4. `docs/GROUND_TRUTH_AND_VALIDATION.md`
5. `docs/ARCHITECTURE.md`
6. `docs/ROADMAP.md`

若任务只涉及某个模块，再读取相应代码和历史实验文档。

## 工作原则

1. 除非用户明确要求写入，否则只检查、分析和提出计划，不修改 Git。
2. 用户要求修改时，先确认范围；不得顺带重写未授权模块。
3. v0 代码是历史基线。在 v1 对应模块完成、测试通过并获得确认前，不删除 v0。
4. 文档必须区分 `current implementation`、`planned` 和 `validated`，不得把规划写成已实现结果。
5. 不得根据单次 holdout 或单一币种挑选参数后宣称策略有效。
6. 不得以 balanced accuracy 代替交易端收益、回撤和校准评估。
7. 不得把 API key、账户信息、原始大数据、模型文件或运行输出提交到 Git。

## 量化研究硬约束

- 特征截止于 bar `t` close；默认成交不得早于 bar `t+1`。
- 任何 target 都必须写明 entry、horizon、path、cost 和 ambiguous-bar 规则。
- 时间序列切分必须考虑 horizon 重叠，并使用 purge/embargo。
- 阈值和超参数只能在内层训练/验证数据上选择；outer test 不参与选择。
- 回测必须逐事件维护账户和持仓状态，不得用预生成的虚拟 position 代替真实状态机。
- 杠杆属于仓位和风险层；基础预测应尽量保持对杠杆配置独立。
- 任何新增 forecast 特征若来自模型，必须使用 chronological OOF 预测，禁止 in-sample stacking。

## 修改代码时

- 先为关键时序和边界条件写测试；
- 保持配置单一来源，避免 target/backtest/live 各自复制参数；
- 使用类型标注和明确的数据契约；
- 对同一 bar 同时触发 TP/SL、数据缺口、fold 末持仓、成交滑点等情况写显式测试；
- 修改后运行相关测试并记录命令和结果。

## 实验记录

新增实验必须基于 `docs/EXPERIMENT_TEMPLATE.md`，至少记录：

- 数据版本与时间范围；
- 训练/验证/测试边界；
- StrategySpec 与成本；
- 基线；
- 全部候选参数，而不只记录最优参数；
- fold 级结果、失败结果和结论边界。

## GLM 外部审阅 agent

本机可通过 Claude Code CLI 调用 GLM 获取独立复核意见。完整命令、配置和边界见 `docs/GLM_EXTERNAL_REVIEWER.md`。

- 只在用户明确要求 GLM 复核或外部第二意见时调用；
- 默认命令为 `claude -p --permission-mode plan --no-session-persistence "<task>"`；
- 该调用不是 Codex 原生 subagent，不得对其输出免除本地证据复核；
- 默认只读，不授权外部 agent 修改项目文件；
- 不得传入 API key、账户信息、原始大数据或未脱敏日志；
- 调用失败时显式报告，不得静默换模型并宣称为 GLM 结果。


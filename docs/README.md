# Documentation Index

## 规范性文档

以下文档定义 v1 重构目标，优先级高于 v0 代码中的默认参数：

1. [STRATEGY_SPEC.md](STRATEGY_SPEC.md)：唯一交易任务、时序和风险边界。
2. [DATA_SPEC.md](DATA_SPEC.md)：数据来源、schema、质量控制和版本记录。
3. [GROUND_TRUTH_AND_VALIDATION.md](GROUND_TRUTH_AND_VALIDATION.md)：标签、切分、校准、回测和评价标准。
4. [ARCHITECTURE.md](ARCHITECTURE.md)：目标代码结构、模块边界和数据流。
5. [ROADMAP.md](ROADMAP.md)：从 v0 到 v1 的分阶段实施顺序。
6. [EXPERIMENT_TEMPLATE.md](EXPERIMENT_TEMPLATE.md)：统一实验记录格式。

## 开发与审阅工具

- [GLM_EXTERNAL_REVIEWER.md](GLM_EXTERNAL_REVIEWER.md)：通过 Claude Code CLI 调用 GLM 外部审阅 agent 的已验证环境、安全命令和使用边界。

## v0 历史实验

下列文档是既有代码的实验记录，不代表 v1 规范：

- `logic_review.md`：旧标签和时间切分问题审查；
- `account_backtest_DOGEUSDT_2024_07.md`：DOGE 单月账户回测；
- `context_feature_experiment.md`：DOGE 多币种 context 实验；
- `sol_context_experiment.md`：SOL context 与近期 holdout；
- `two_stage_xgboost_experiment.md`：两阶段 XGBoost 与交易标签；
- `walk_forward_two_stage_sol.md`：SOL 五周 walk-forward；
- `forecast_feature_experiment.md`：手工 forecast proxy/confidence 实验。

历史实验可以用于说明问题和建立基线，但不得直接复制其 horizon、entry、杠杆和阈值作为 v1 默认值。

## 文档状态约定

- **Normative**：定义 v1 必须遵守的规范；
- **Current**：描述现有代码真实状态；
- **Planned**：计划但尚未实现；
- **Experimental**：一次实验结果，不能自动升级为项目结论；
- **Deprecated**：仅为历史追溯保留。


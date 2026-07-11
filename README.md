# PreTimeSequence

面向加密货币永续合约的短周期量化研究项目。项目目标是使用 1 分钟市场数据，在每根 K 线关闭后评估未来约 15 分钟的可交易机会，并在严格的成本、回测和风险约束下形成多空或不交易决策。

> [!WARNING]
> 本仓库当前是研究项目，不是可直接运行的实盘交易系统。现有 v0 代码和历史实验尚未证明稳定 alpha，不得据此直接使用真实资金或高杠杆下单。

## 项目状态

项目正在从早期趋势分类原型重构为可复现的 15 分钟决策研究框架：

- **v0（当前代码）**：保留已有 Binance 数据获取、技术指标、XGBoost、两阶段分类和账户回测代码，作为历史基线与问题证据。
- **v1（目标架构）**：统一交易时序、Ground Truth、验证和回测口径；预测收益分布与多空 action value；将杠杆移至仓位和风险层。
- **当前结论**：已有 walk-forward 结果跨市场状态不稳定，尚不满足实盘门槛。

完整进度与已知问题见 [PROJECT_STATUS.md](PROJECT_STATUS.md)。

## 统一研究问题

在第 `t` 根 1 分钟 K 线关闭后，仅使用当时及此前可见的信息，能否对从第 `t+1` 根 K 线开始的未来 15 分钟价格路径进行有效预测，并在扣除手续费、点差、滑点和风险惩罚后，识别具有正期望值的 long、short 或 flat 决策？

v1 默认交易时序：

```text
bar t close
  -> 构建特征与预测
  -> 生成 long / short / flat 决策
  -> bar t+1 open 或可成交报价入场
  -> 最多持有 15 分钟
  -> TP / SL / time exit / risk exit
```

10–20 倍仅作为允许的杠杆上限。模型首先预测市场机会，风险层再根据止损距离、账户风险预算和流动性决定实际仓位。

## 当前代码与目标架构

| 模块 | v0 当前实现 | v1 目标 |
| --- | --- | --- |
| 数据 | Binance 1m K 线、本地 PKL/CSV、context 币种 | 可校验、可追溯、分层存储的期货市场数据 |
| 特征 | 技术指标、收益率、波动、context、手工 forecast proxy | 因果特征、微观结构、独立 OOF forecast 输出 |
| GT | 20 分钟 ATR 三分类与 120/240 分钟交易标签并存 | 统一 15 分钟收益、MFE/MAE、TP/SL 概率和 action value |
| 模型 | 单阶段/两阶段 XGBoost | 基线 -> forecast model -> calibration -> meta filter |
| 验证 | 时间切分、部分 walk-forward | purged/embargoed nested walk-forward 与冻结 holdout |
| 回测 | 简化账户回测 | 逐事件状态机、真实成交与风险约束 |
| 实盘 | 未形成完整执行系统 | 仅在研究门槛通过后进入 paper trading |

## 仓库结构

```text
PreTimeSenquence/
├── README.md                         项目入口与统一口径
├── PROJECT_STATUS.md                 当前状态、问题与决策记录
├── AGENTS.md                         AI/Codex 工作规则
├── CONTRIBUTING.md                   协作与提交规范
├── docs/
│   ├── README.md                     文档索引
│   ├── ARCHITECTURE.md               v1 目标架构
│   ├── STRATEGY_SPEC.md              唯一策略合同
│   ├── DATA_SPEC.md                  数据规范
│   ├── GROUND_TRUTH_AND_VALIDATION.md 标签、验证与回测规范
│   ├── ROADMAP.md                    分阶段重构计划
│   └── EXPERIMENT_TEMPLATE.md        实验记录模板
├── pretimesequence/                  v0 Python 代码，暂时保留
├── oldversion/                       更早期历史代码，不作为当前入口
└── requirements.txt                  v0 依赖
```

## 文档阅读顺序

1. [PROJECT_STATUS.md](PROJECT_STATUS.md)：先确认什么已完成、什么尚未完成。
2. [docs/STRATEGY_SPEC.md](docs/STRATEGY_SPEC.md)：确认唯一交易任务和时序。
3. [docs/GROUND_TRUTH_AND_VALIDATION.md](docs/GROUND_TRUTH_AND_VALIDATION.md)：确认标签、切分和回测标准。
4. [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)：查看目标模块和数据流。
5. [docs/ROADMAP.md](docs/ROADMAP.md)：按阶段推进代码重构。

## v0 运行方式

以下命令仅用于复现现有基线，不代表 v1 设计已经实现：

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python -m pretimesequence.cli diagnose --data data/DOGEUSDT_1m_2024.pkl --output outputs/diagnostics.md
python -m pretimesequence.cli walk-forward-two-stage --help
```

本地密钥只允许通过环境变量或未跟踪的本地文件提供，不得提交到 Git。数据、模型和运行输出同样不进入普通 Git 历史。

## 研究通过门槛

任何模型进入 paper trading 前必须同时满足：

- 数据和特征不存在未来信息泄漏；
- 标签、模型、成交和回测使用同一 `StrategySpec`；
- outer walk-forward 在保守成本后优于无交易及简单基线；
- 结果不由单一币种、单一周或单一阈值主导；
- 概率经过时间外校准，阈值仅在内层验证集选择；
- 最大回撤、最差 fold 和连续亏损风险处于预设预算内；
- 最终冻结 holdout 在参数确定前从未参与选择。

## 贡献与修改

修改前请先阅读 [CONTRIBUTING.md](CONTRIBUTING.md) 和 [AGENTS.md](AGENTS.md)。新增实验应复制 [docs/EXPERIMENT_TEMPLATE.md](docs/EXPERIMENT_TEMPLATE.md)，记录数据版本、时间边界、参数、成本、基线和完整结果。

## 免责声明

本项目仅用于研究和软件验证，不构成投资建议。加密货币衍生品和高杠杆交易可能导致快速且超过预期的损失。


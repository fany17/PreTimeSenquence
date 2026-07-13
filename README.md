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

当前项目的 Conda 环境名为 `bitc`。2026-07-13 已验证的环境为 Python 3.9.20，且 `requirements.txt` 中的核心依赖均可正常导入。环境名是开发约定，不应将某台机器的 Conda 安装路径写入脚本。

已验证的核心版本快照：

| Package | Version |
| --- | --- |
| Python | 3.9.20 |
| pandas | 2.2.3 |
| numpy | 1.24.3 |
| python-binance | 1.0.21 |
| xgboost | 2.1.2 |
| scikit-learn | 1.5.2 |
| matplotlib | 3.9.2 |
| plotly | 5.24.1 |
| seaborn | 0.13.2 |

进入环境并验证项目入口：

```bash
conda activate bitc
python --version
python -m pretimesequence.cli --help
```

也可以不切换当前 shell 环境，显式通过 `bitc` 运行：

```bash
conda run -n bitc python -m pretimesequence.cli --help
```

以下命令仅用于复现现有 v0 基线，不代表 v1 设计已经实现：

```bash
python -m pretimesequence.cli diagnose --data data/DOGEUSDT_1m_2024.pkl --output outputs/diagnostics.md
python -m pretimesequence.cli walk-forward-two-stage --help
```

当前 v1 Ridge 基线已经可以用一条命令启动完整 nested walk-forward 训练评估：

```powershell
conda run -n bitc python -B -m pretimesequence.v1 train --data data/SOLUSDT_1m_recent_600k.pkl --output outputs/v1_sol_baseline_20260714 --experiment-id v1-sol-baseline-20260714 --assume-naive-utc --allow-unsafe-pickle
```

`--output` 必须指向尚不存在的新目录，命令不会覆盖既有实验。`--allow-unsafe-pickle` 只应用于确认可信的本地 pickle；协作者或下载来源不明的数据应先转换成 CSV。命令会保存输入 checksum、配置、fold/candidate 指标、OOF/outer predictions 和 trades，并记录每个 CSV 的 hash、字节数、行数和列 schema；当前仍是研究基线训练，不表示策略已经有效，也不能用于实盘。

上表是当前可运行环境的快照，不是已锁定的可复现依赖规范；`requirements.txt` 目前仍未固定版本。

本地密钥只允许通过环境变量或未跟踪的本地文件提供，不得提交到 Git。数据、模型和运行输出同样不进入普通 Git 历史。

## GLM 外部审阅 agent

本机已验证可通过 Claude Code CLI 调用 GLM，用于独立代码审查和研究方案复核。该路径是外部审阅工具，不是 Codex 原生 subagent；默认只读，其输出必须由主 agent 回到源码、数据和测试复核。配置、安全命令和调用边界见 [docs/GLM_EXTERNAL_REVIEWER.md](docs/GLM_EXTERNAL_REVIEWER.md)。

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


# GLM External Reviewer

状态：**Current / locally validated**

## 定位

本项目可通过 Claude Code CLI 调用 GLM，用于独立代码审查、实验设计复核和第二意见。这是一个外部审阅 agent，不是 Codex 原生 subagent；Codex 通过本地 `claude` 命令发起请求，再对返回结果进行核验和整合。

## 已验证环境

2026-07-13 在 Windows 上完成了最小无工具调用验证：

| Item | Validated value |
| --- | --- |
| Node.js | 24.18.0 |
| Claude Code CLI | 2.1.207 |
| Endpoint | `https://open.bigmodel.cn/api/anthropic` |
| Main model observed | `glm-5.2[1m]` |
| Auxiliary model observed | `glm-4.5-air` |
| Expected response | `GLM_SUBAGENT_OK` |

该验证只证明当时本机配置可成功调用，不证明模型结论正确，也不保证未来的模型名、计费或 endpoint 不变。

验证命令：

```powershell
claude -p --permission-mode plan --output-format json --no-session-persistence "不要调用任何工具，不要读取或修改文件。只输出：GLM_SUBAGENT_OK"
```

当次返回为 `success` 且 `result` 为 `GLM_SUBAGENT_OK`。CLI 报告主模型输入 14,377 tokens、输出 105 tokens，并报告费用 0.080274 USD。这些数值只是当次调用的 CLI 报告，实际扣费、上下文加载量和模型路由可能变化；即使是最小请求也不应假设成本可忽略。

## 本地配置

Claude 配置位于：

```text
C:\Users\<username>\.claude\settings.json
```

配置中至少需要 `ANTHROPIC_BASE_URL` 和 `ANTHROPIC_AUTH_TOKEN`。该文件是机器本地秘密配置，不属于本仓库，不得复制到项目、日志、实验记录或 Git 历史。

## 安全调用

默认使用只读 plan 模式，并关闭会话持久化：

```powershell
claude -p --permission-mode plan --no-session-persistence "只读审查指定问题，不修改文件；返回结论、证据位置和不确定性。"
```

需要结构化捕获返回值时：

```powershell
claude -p --permission-mode plan --output-format json --no-session-persistence "<task>"
```

调用前应先收窄范围，例如只给出目标模块、文件或 diff。不得向外部 agent 提供 API key、账户信息、原始大数据、未脱敏运行日志或与任务无关的整个用户目录。

## 使用规则

1. 只在用户明确要求 GLM 复核、外部第二意见或该文档规定的审查任务中调用。
2. 默认只读；外部 agent 不直接修改项目文件。
3. GLM 输出只是候选意见，必须回到源码、数据、测试和规范文档复核。
4. 调用会消耗独立的 GLM 额度或 API 费用；不得在循环中无界重试。
5. 调用失败时显式报告失败，不得静默换成其他模型后宣称是 GLM 结果。
6. 记录时只保留任务范围、模型标识、成功/失败和结论摘要，不记录密钥或完整机密上下文。

## 适合与不适合的任务

适合：

- 对已定义范围的代码或文档做独立复核；
- 查找可能的逻辑遗漏、测试缺口和反例；
- 对量化研究的时序、泄漏、回测和结论边界提供第二意见。

不适合：

- 代替本地测试、数据核对或人工批准；
- 直接获取本机秘密或未授权外部数据；
- 在没有 outer evaluation 的情况下宣称策略有效。

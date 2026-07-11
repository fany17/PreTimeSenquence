# Strategy Specification

状态：**Normative / v1 唯一策略合同**

## 研究目标

使用 1 分钟加密货币永续合约数据，在每个决策时点预测未来约 15 分钟的价格路径，并选择 long、short 或 flat。项目研究的是短周期预测与交易过滤，不以高频做市或毫秒级执行为目标。

## 默认时序

| 项目 | v1 默认值 |
| --- | --- |
| Base interval | 1 minute |
| Decision interval | 1 minute；可在实验中比较 5 minutes |
| Feature cutoff | bar `t` 完全关闭后 |
| Earliest entry | bar `t+1` open 或下一可成交 bid/ask |
| Primary horizon | 15 minutes |
| Auxiliary horizons | 5 / 30 / 60 minutes |
| Maximum holding | 15 minutes，除非实验明确覆盖其他 horizon |
| Concurrent position | 每个 symbol 最多一笔 |
| Position mode | isolated research accounting |

任何偏离默认值的实验必须在实验文档中显式声明，不得沿用 v0 的 20/120/240 分钟默认值而不解释。

## 决策变量

模型不直接输出“必须交易”的三分类，而应优先输出：

- future return distribution；
- MFE/MAE；
- `P(long TP before SL)`；
- `P(short TP before SL)`；
- `E[pnl_long]` 与 `E[pnl_short]`；
- uncertainty 与 calibrated confidence。

策略层计算：

```text
Q_long  = expected_net_pnl_long  - risk_penalty_long
Q_short = expected_net_pnl_short - risk_penalty_short
Q_flat  = 0
```

只有最大 action value 超过预先定义的阈值并通过 regime/risk filter 时才开仓，否则为 flat。

## 成交与退出

### Entry

- 禁止使用形成信号的同一 bar close 作为无摩擦成交价；
- 默认使用下一 bar open，并加入 side-dependent spread/slippage；
- 若使用逐笔或 order-book 数据，应显式模拟决策延迟。

### Exit

允许的退出原因：

- take profit；
- stop loss；
- maximum holding time；
- regime/risk exit；
- fold-end forced close；
- liquidation，仅作为风险失败事件，不作为正常策略退出。

若同一 1m bar 同时触发 TP 和 SL，默认采用保守顺序；正式研究应使用更细数据消除歧义，或将该事件标记为 ambiguous 并单独报告。

## 杠杆与仓位

- 10–20x 是最大允许杠杆，不是默认全仓暴露；
- 基础 market target 不依赖杠杆；
- 仓位由账户风险预算和 stop distance 反推；
- 初始研究建议每笔最大权益风险为 0.25%–0.50%，最终数值必须经风险实验确认；
- 必须限制最大名义敞口、单日亏损、连续止损和最大回撤；
- funding、手续费和维持保证金进入账户层。

## Universe

第一阶段建议使用：

- 主交易标的：SOLUSDT；
- 稳健性标的：BTCUSDT、ETHUSDT、DOGEUSDT；
- context 只允许使用同一时刻已经关闭或更早的数据。

单币种调参结果不得被视为跨币种有效。

## 非目标

v1 初期不解决：

- 毫秒级做市；
- 多交易所套利；
- 强化学习自动下单；
- 在未建立稳定基线前使用大型深度网络；
- 直接真实资金交易。


# Data Specification

状态：**Normative target**

## 数据层级

```text
raw       交易所原始响应，只追加不修改
validated schema、时区、重复和缺口检查后的标准数据
features  由 validated 数据因果生成的特征
targets   独立生成的未来路径与 action outcome
```

模型训练不得直接依赖无法追溯的手工 PKL 文件。

## 必需数据

### Core bars

- `timestamp_open_utc`
- `timestamp_close_utc`
- `symbol`
- `venue`
- `contract_type`
- `open/high/low/close`
- `base_volume`
- `quote_volume`
- `trade_count`
- `taker_buy_base_volume`
- `taker_buy_quote_volume`

### Futures context

- mark price 与 index price；
- funding rate 和 funding timestamp；
- open interest；
- taker buy/sell statistics；
- exchange information：tick size、step size、minimum notional；
- 可获得时增加 aggregate trades、liquidation 和 order-book snapshot。

## 时间语义

- 所有持久化时间统一为 UTC 且必须带 timezone；
- 明确 timestamp 表示 bar open 还是 bar close；
- bar `t` 的 close、volume 和 high/low 只有在该 bar 结束后才可作为特征；
- context 对齐只能 backward/as-of，不允许最近邻向未来匹配；
- 训练、验证和回测必须使用同一交易日历与缺口规则。

## 质量检查

每个数据集生成前必须检查：

1. 时间戳单调递增；
2. symbol 内无重复 bar；
3. OHLC 逻辑成立；
4. 价格和成交量为有限非负数；
5. 1m cadence 缺口数量、持续时间和原因；
6. 不完整的最后一根 bar 被排除；
7. 不同币种/context 对齐覆盖率；
8. 合约上线、下线和规则变化；
9. 异常尖峰与交易所维护期；
10. checksum 与行数一致。

缺口不得默认为零成交。是否填补、保留或切断序列必须由配置决定并记录。

## Dataset manifest

每个数据版本必须保存一个轻量 manifest：

```yaml
dataset_id: binance_usdm_solusdt_1m_2021_2026_v1
source: binance_usdm
symbol: SOLUSDT
interval: 1m
timezone: UTC
start: 2021-01-01T00:00:00Z
end: 2026-06-30T23:59:00Z
rows: 0
schema_version: 1
checksum: pending
created_by: pending
created_at: pending
```

manifest 可进入 Git；实际大数据文件不进入普通 Git。

## 数据覆盖

正式 walk-forward 数据应覆盖多个上涨、下跌、震荡、高波动和低波动阶段。优先使用多年份数据，并在 BTC/ETH/SOL/DOGE 上采用同一研究口径。

## 小型测试数据

仓库可以提交少量合成 fixture，用于验证：

- next-bar entry；
- TP-first、SL-first 和 ambiguous bar；
- 数据缺口；
- fold-end forced close；
- funding 和手续费计算。


# Experiment: <short title>

状态：**Experimental**

## 1. Question

本实验要回答的单一问题是什么？预期结果和可证伪条件是什么？

## 2. Registration

- Experiment ID:
- Created at:
- Code commit:
- StrategySpec version:
- Data manifest:
- Feature version:
- Target version:

## 3. Data

| Item | Value |
| --- | --- |
| Venue/contract | |
| Symbols | |
| Interval | |
| Start/end | |
| Rows | |
| Missing bars | |
| Context data | |

## 4. Time split

记录每个 fold 的 train、calibration、purge/embargo 和 outer test 边界。

## 5. Targets and execution

- Feature cutoff:
- Entry:
- Horizon:
- Target outputs:
- TP/SL:
- Ambiguous-bar rule:
- Fold-end rule:

## 6. Costs and risk

- Fee:
- Spread:
- Slippage:
- Funding:
- Leverage cap:
- Risk per trade:
- Position limits:

## 7. Models and baselines

列出全部模型、基线、特征组、参数范围和随机种子。不得只记录表现最好的配置。

## 8. Selection protocol

说明 calibration、threshold 和 hyperparameter 如何仅在 inner validation 中选择。

## 9. Results

### Forecast metrics

填写每个 outer fold 的误差、概率校准和 selected-trade precision。

### Trading metrics

填写净收益、交易数、expectancy、profit factor、最大回撤、worst fold、CVaR 和连续亏损。

### Baseline comparison

所有结果必须与相同 fold、相同成本下的基线比较。

## 10. Sensitivity and failures

记录成本、阈值、horizon、币种和 regime 敏感性，以及失败或异常结果。

## 11. Conclusion

- 哪个结论被支持？
- 哪个结论没有被支持？
- 是否存在选择偏差或泄漏风险？
- 下一实验是什么？
- 本结果是否改变项目状态？默认答案应为“否”，除非已通过预注册验证。


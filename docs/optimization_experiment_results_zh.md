# 方案优化实验结果记录

> 本文记录方案优化阶段的逐项结果。当前 Fold 6 outer test 已被用于前期诊断，因此以下数字属于开发性对照，不是新的独立盲测结果。

## 1. O0：设备响应基线

配置：

```text
sparse_extreme
+ initial_baseline_delta_with_device
+ 初始正常基线 30 个窗口
+ MLP hidden=32
+ 三分类响应分支 + direct expert 后覆盖
```

Fold 6 的 direct override 结果：

| 指标 | O0 基线 |
|---|---:|
| Macro-F1 | 0.8529 |
| FAR | 2.85% |
| abnormal recall | 91.73% |
| anomaly recall | 97.52% |
| direct recall | 65.36% |
| direct threshold | 0.1 |

该结果作为 R1 的唯一参考点。R1 只改变是否保留设备 one-hot，其他训练、校准和评估流程保持一致。

## 2. R1：去掉设备身份 one-hot

### 2.1 配置

```text
sparse_extreme
+ initial_baseline_delta_only
+ 初始正常基线 30 个窗口
+ MLP hidden=32
+ 三分类响应分支 + direct expert 后覆盖
```

输入从基线的 51 维减少为 45 维，删除：

```text
device_is_0 ... device_is_5
```

### 2.2 Fold 6 结果

| 指标 | O0 基线 | R1 去设备编码 | 变化 |
|---|---:|---:|---:|
| Macro-F1 | 0.8529 | **0.8679** | +0.0150 |
| FAR | 2.85% | **0.87%** | -1.98 个百分点 |
| abnormal recall | **91.73%** | 81.34% | -10.39 个百分点 |
| anomaly recall | **97.52%** | 88.39% | -9.13 个百分点 |
| direct recall | 65.36% | **68.09%** | +2.73 个百分点 |
| direct threshold | 0.1 | 0.2 | — |

### 2.3 逐设备结果

| 设备 | O0 abnormal | R1 abnormal | O0 direct | R1 direct |
|---|---:|---:|---:|---:|
| Google Pixel6 | 82.94% | 46.11% | 10.14% | 21.11% |
| Google Pixel Watch1 | 96.51% | 78.52% | n/a | n/a |
| Google Pixel Watch2 | 98.66% | 98.66% | n/a | n/a |
| HUAWEI Mate40 | 97.99% | 98.26% | 96.24% | 95.30% |
| RedMi K60 | 97.37% | 95.67% | 91.65% | 93.97% |
| XiaoMi MI8 | 36.15% | 25.82% | 30.99% | 24.88% |

### 2.4 结论

R1 不能直接替代 O0，原因是：

1. FAR 明显下降，说明去掉设备身份后模型的误报更少；
2. direct recall 小幅提高，RedMi K60 也有改善；
3. 但 abnormal recall 和 anomaly recall 明显下降，Pixel6、Watch1 和 MI8 受到较大影响；
4. Macro-F1 的提升主要来自 FAR/precision 改善，不能解释为整体异常检测能力提升。

因此，设备 one-hot 不是简单的“无用设备捷径”。在当前特征设计下，它承担了一部分设备条件化作用，但也牺牲了跨设备泛化的风险。R1 的正确定位是：

> 一个低误报、较保守的无设备身份对照版本，而不是当前最终响应模型。

### 2.5 后续决策

不立即把 R1 扩展到全部六个 fold，先继续做两个更有解释力的变体：

1. **R2：去掉跨频段差值和耦合特征**，判断 anomaly 识别是否主要依赖这些特征；
2. **R3/R4：设备能力掩码版本**，保留“是否有 L5/AGC”等可解释能力信息，但不直接输入设备身份 one-hot。

如果能力掩码能够恢复 R1 丢失的 abnormal/anomaly recall，同时保持较低 FAR，就优先采用“能力条件化”而不是“设备身份条件化”。

## 3. R2：去掉跨频段差值与耦合特征

### 3.1 配置

```text
sparse_extreme
+ initial_baseline_delta_no_cross
+ 保留 device_is_*
+ 初始正常基线 30 个窗口
+ MLP hidden=32
+ 三分类响应分支 + direct expert 后覆盖
```

R2 删除所有 `initial_baseline_delta_l5_minus_*` 和 `initial_baseline_delta_coupled_*` 特征，保留其余基础聚合特征和设备 one-hot。

### 3.2 Fold 6 结果

| 指标 | O0 基线 | R1 去设备编码 | R2 去跨频段特征 |
|---|---:|---:|---:|
| Macro-F1 | 0.8529 | 0.8679 | **0.8760** |
| FAR | 2.85% | **0.87%** | **1.45%** |
| abnormal recall | 91.73% | 81.34% | **91.78%** |
| anomaly recall | 97.52% | 88.39% | **98.26%** |
| direct recall | 65.36% | 68.09% | **68.59%** |
| direct threshold | 0.1 | 0.2 | 0.1 |

### 3.3 逐设备结果

| 设备 | O0 abnormal | R2 abnormal | O0 direct | R2 direct |
|---|---:|---:|---:|---:|
| Google Pixel6 | 82.94% | 66.39% | 10.14% | 2.36% |
| Google Pixel Watch1 | 96.51% | 98.26% | n/a | n/a |
| Google Pixel Watch2 | 98.66% | 98.39% | n/a | n/a |
| HUAWEI Mate40 | 97.99% | 97.99% | 96.24% | 95.57% |
| RedMi K60 | 97.37% | 97.06% | 91.65% | 96.14% |
| XiaoMi MI8 | 36.15% | **78.87%** | 30.99% | **74.65%** |

### 3.4 结论

R2 在 Fold 6 上优于 O0 的 pooled 结果：三类 recall 均不下降，FAR 降低，Macro-F1 提升。但逐设备结果仍不均衡：

- MI8 和 RedMi 获得明显改善；
- Watch 基本稳定；
- Pixel6 的 direct 和 abnormal 明显下降；
- 因此不能只根据 Fold 6 的 pooled 分数把 R2 定为最终版本。

当前判断是：跨频段差值/耦合特征在现有响应分支中可能引入设备相关噪声，尤其影响 MI8；但它们对 Pixel6 可能提供了有用信息。R2 应作为候选版本扩展到其余有效 fold 复验，之后再决定是否进入能力掩码或物理特征实验。

### 3.5 六折复验

R2 已在 `fold_1、fold_2、fold_4、fold_5、fold_6、fold_7` 完成复验；`fold_3` 仍因无法建立初始正常基线而排除。

| fold | O0 Macro-F1 | R2 Macro-F1 | O0 FAR | R2 FAR | O0 abnormal | R2 abnormal | O0 direct | R2 direct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fold_1 | 0.6604 | 0.6652 | 3.41% | 0.50% | 99.88% | 100.00% | 99.88% | 100.00% |
| fold_2 | 0.8670 | 0.8663 | 0.36% | 0.24% | 76.13% | 75.69% | 98.62% | 97.79% |
| fold_4 | 0.6648 | 0.6645 | 0.50% | 0.61% | 100.00% | 100.00% | 100.00% | 100.00% |
| fold_5 | 0.6627 | 0.6630 | 0.31% | 0.23% | 99.00% | 99.00% | 99.00% | 99.00% |
| fold_6 | 0.8529 | 0.8760 | 2.85% | 1.45% | 91.73% | 91.78% | 65.36% | 68.59% |
| fold_7 | 0.6593 | 0.6606 | 0.28% | 0.23% | 97.51% | 97.96% | 97.51% | 97.96% |

按六折 test 混淆矩阵 pooled 汇总：

| 指标 | O0 基线 | R2 去跨频段特征 | 变化 |
|---|---:|---:|---:|
| samples | 49,963 | 49,963 | — |
| Macro-F1 | 0.8946 | **0.9122** | +0.0176 |
| FAR | 1.17% | **0.49%** | -0.68 个百分点 |
| abnormal recall | 93.95% | **93.99%** | +0.04 个百分点 |
| anomaly recall | 75.88% | **76.28%** | +0.40 个百分点 |
| direct recall | 93.96% | **94.47%** | +0.51 个百分点 |

六折结果支持以下判断：

1. 在当前响应特征设计下，跨频段差值/耦合特征不是必需信息，删除后没有损害整体异常发现能力；
2. 删除后 FAR 稳定下降，说明这些特征可能携带了部分设备或 Session 相关噪声；
3. R2 在 pooled 指标和六折平均上优于 O0，可以进入下一阶段候选集；
4. Fold 2 的 anomaly recall 仍只有 50%，Pixel6 在 Fold 6 的 direct recall 仍很低，说明 R2 还没有解决所有设备差异问题。

因此，R2 暂定为当前设备响应分支的最佳候选特征集，但下一步仍需做“设备能力掩码”实验，而不是直接冻结为最终方案。

## 4. R3 实验口径修正

第一次运行的 `initial_baseline_delta_with_capability` 实际保留了完整跨频段差值/耦合特征，因此它表示“**O0 + 能力掩码**”，不是计划中的“**R2 + 能力掩码**”。该运行结果不纳入 R2/R3 对照，也不用于方案选择。

已补充严格版本：

```text
initial_baseline_delta_no_cross_with_capability
```

它只在 R2 特征基础上追加 `capability_has_l5` 和 `capability_has_agc`，将作为正式 R3 重新运行。

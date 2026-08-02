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


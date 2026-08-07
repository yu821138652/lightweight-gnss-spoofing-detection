# 场景条件化频段响应诊断：标签重构与六折覆盖审计

> 状态：截至 2026-08-07 的当前可审计方案。本文档替代旧设备级 `normal/anomaly/direct` 三分类链路作为频段响应部分的正式结论；原始张量和审计 JSON 位于本地 `output/label_repair_v1/`，不纳入 Git。

## 1. 结论摘要

静态数据的正式方案调整为：

```text
多设备 GNSS 观测
  ├─ 场景分支（学习模型）：normal / L1 / L5 / L1+L5
  └─ 场景条件化频段响应诊断
       ├─ 场景目标频段 + 当前可观测 → direct
       └─ 非目标频段 + 攻击前基线能力 → normal / associated anomaly 的诊断证据
```

场景分支仍是当前唯一完成严格静态六折泛化评估的学习模块：四维 C/N0-only、T=5、TCN 的 pooled Macro-F1 为 **0.9921**，Accuracy 为 **0.9944**。

频段响应部分已完成真值重构和样本覆盖审计，但当前静态录制不足以支持一个可报告六折泛化性能的 `normal / associated_anomaly` 通用分类器。因此，它当前定位为**可解释的频段响应诊断层**，而非另一个宣称高分泛化能力的黑盒模型。

## 2. 为什么废弃旧设备级三分类

旧张量的 `y_response_state` 为每个设备窗口只分配一个互斥状态：

```text
normal / attack-associated anomaly / direct spoof
```

但同一双频设备可同时具有：

```text
L5 = direct spoof
L1 = associated anomaly
```

旧构建逻辑中 direct 会覆盖 anomaly，且旧 CSV 未覆盖的异常会默认落为 normal。因此旧完整链路的 `0.9802` Macro-F1、`98.76%` anomaly recall 等指标仅可作为**旧标签体系下的开发性历史对照**，不能再作为本文频段关联异常联合诊断的最终主结果。

## 3. 新真值和标签语义

人工复核真值保存在 `docs/device_band_association_intervals.csv`，每行精确到：

```text
Environment / Scenario / Session / DeviceName / response_band / TOW interval
```

其中：

- 新主楼与操场的 `st_L5` 攻击区间内，六台设备的 L1 均标为关联异常；两只无 L5 的 Watch 也包括在内。
- 操场 `st_L1` 攻击区间内，`Google_Pixel6`、`XiaoMi_MI8` 的 L5 标为关联异常；其他设备的 L5 为 normal。
- 新主楼 `st_L1` 与两个 `st_L1+L5` 场景未观察到关联异常。

`direct` 不是由旧设备分类器预测，而是由以下语义共同确定：

```text
reviewed attack active
+ scene target band
+ device currently observes that band
→ direct
```

## 4. L5 消失也是关联异常

关联异常不只包括“频段仍可观测但 C/N0、斜率或质量发生变化”，还包括：

```text
攻击前该设备稳定具备 L5
攻击中 L5 成批消失或显著减少
→ L5 availability / tracking-loss associated anomaly
```

操场 `st_L1` 中，Pixel6 与 MI8 正是这一类型：攻击前存在 L5，攻击区间内 L5 大量消失，攻击后恢复。因而不能用“当前是否有 L5”排除它们；正确的资格条件是**攻击前已审核正常基线是否具备该频段**。

当前构建器以每条源流前 30 个已审核正常设备窗口建立固定基线，并输出：

- `baseline_has_l1` / `baseline_has_l5`：该频段在攻击前是否可用；
- `has_l1` / `has_l5`：当前窗口是否仍可观测；
- `y_associated_anomaly_l1` / `y_associated_anomaly_l5`：独立的频段级关联异常真值；
- `y_direct_l1` / `y_direct_l5`：独立的目标频段 direct 语义标签。

因此，`initial_baseline_delta_l5_log_signal_count` 可作为 L5 消失的直接证据；无 L5 硬件能力的 Watch 因 `baseline_has_l5=0` 不会被误纳入 L5 响应任务。

## 5. 六折样本覆盖审计

审计脚本：`pipeline_total/64_audit_scene_conditioned_band_labels.py`。

| Fold | L1 攻击 → L5：train/val | L1 攻击 → L5：test | L5 攻击 → L1：test | 可作严格响应模型评估？ |
|---|---|---|---|---|
| 1 | 两类都有 | 仅 normal（0/1232） | 无 | 否，仅可测 L5 假警报 |
| 2 | 两类都有 | 无 | 仅 anomaly（2955/0） | 否，仅可测 L1 漏检 |
| 4 | 两类都有 | 无 | 无 | 否 |
| 5 | 仅 normal | anomaly/normal = 1266/1266 | 无 | 否，开发集无 L5 正类 |
| 6 | 两类都有 | 无 | 仅 anomaly（3886/0） | 否，仅可测 L1 漏检 |
| 7 | 两类都有 | 无 | 无 | 否 |

fold 6 说明了 L5 消失型标签已正确进入训练与验证范围：

```text
train L1→L5: 895 anomaly + 1861 normal
val   L1→L5: 347 anomaly + 589 normal
```

然而，在任何一个 outer fold 中都不存在“开发集有正负类、测试集也有正负类”的同一关联异常子任务。`L5→L1` 还在所有已审核静态样本中只有 anomaly、没有 normal 反例。

因此，禁止将这些数据训练出的响应分类结果汇总为六折 Macro-F1、recall 或“跨设备泛化性能”。

## 6. 当前运行时诊断语义与边界

运行时应先输出攻击场景，再在对应频段生成可解释状态：

| 场景 | 频段状态 |
|---|---|
| `normal` | 不输出攻击相关响应诊断 |
| `L1` | 可观测 L1 为 direct；对基线具备 L5 的设备检查 L5 下降/消失证据 |
| `L5` | 可观测 L5 为 direct；对基线具备 L1 的设备检查 L1 压制、可用性下降等关联异常证据 |
| `L1+L5` | 两个可观测目标频段均为 direct；当前静态人工复核未确认关联异常 |

这里的“关联异常”是观测到的响应语义，不是对芯片内部因果机理的断言。尤其 `L5→L1` 在现有静态数据中是全体异常的模式，不能表述为所有终端、所有 L5 攻击下必然发生。

## 7. 论文中可报告与不可报告的内容

可以作为主结果报告：

- 场景分支的静态六折性能；
- AGC 等设备强相关特征损害跨设备泛化、四维 C/N0-only 更稳健的消融结论；
- 频段级人工复核发现：L5 攻击下 L1 关联异常，以及 L1 攻击下部分设备 L5 可用性丢失；
- 新标签体系能够表达 `direct` 与另一频段 `associated anomaly` 同时发生。

不可作为主结果报告：

- 旧设备三分类完整链路的 `0.9802` Macro-F1；
- 当前关联异常响应分支的六折泛化准确率、Macro-F1 或泛化 recall；
- “L5 攻击必然导致 L1 异常”之类的物理普适性断言；
- 动态端到端联合诊断性能。

## 8. 下一步数据与实验要求

要让关联异常分支成为可严格评估的学习模型，至少需要：

1. 独立的、含 L5 消失型正例的 `st_L1` 录制，使 outer-development 和 outer-test 都可出现正负类；
2. 含 L1 normal 反例的 `st_L5` 录制，用于验证 L5 攻击下 L1 关联异常的边界；
3. 动态场景的逐设备、逐频段人工复核真值；
4. 在冻结的开发集上固定可用性下降/质量下降阈值，再做独立测试。


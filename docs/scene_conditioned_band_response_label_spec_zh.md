# 场景条件化的频段响应标签规范

状态：已根据人工复核于 2026-08-07 确认；用于替代旧的设备互斥三分类标签。

## 1. 为什么不再使用设备级 `normal/anomaly/direct` 三分类

同一双频设备可在同一时刻同时具有：

```text
L5: direct_spoof
L1: associated_anomaly
```

旧标签将设备压缩为一个互斥状态，并在构建时让 `direct_spoof` 覆盖 `attack_associated_anomaly`。因此它无法表达跨频段联合响应；未在旧 CSV 中显式列出的关联异常还会默认落入 `normal`。

## 2. 新的树结构

场景分支先输出 `normal / L1 / L5 / L1+L5`。响应诊断不再与 direct 竞争类别，而是只对**非目标且攻击前基线具备该频段能力**的频段判断：

```text
scene=normal: 不触发攻击响应诊断
scene=L1:    L1 为 target/direct；L5 做 normal vs associated_anomaly
scene=L5:    L5 为 target/direct；L1 做 normal vs associated_anomaly
scene=L1+L5: L1、L5 都为 target/direct；当前人工复核无关联异常
```

`direct` 由场景目标频段与当前频段可用性共同给出；`associated_anomaly` 是独立的频段级标签。当前频段不可观测并不自动排除关联异常：若攻击前基线具备该频段，攻击中频段消失/显著减少本身就是可用性（跟踪丢失）型关联异常。Watch 在 L5 攻击下虽然不具备 L5，仍作为 L1 非目标频段的异常响应设备参与判断；但它们的 `baseline_has_l5=0`，不会被误纳入 L5 响应任务。

## 3. 已确认的关联异常真值

真值文件：`docs/device_band_association_intervals.csv`。

| 环境 | 场景 | 关联异常频段与设备 | 区间 |
|---|---|---|---|
| 新主楼 | st_L1 | 无 | — |
| 新主楼 | st_L5 | 全部六台设备的 L1 | 217683–218305 |
| 新主楼 | st_L1+L5 | 无 | — |
| 操场 | st_L1 | `Google_Pixel6`、`XiaoMi_MI8` 的 L5 | 262228–262860 |
| 操场 | st_L5 | 全部六台设备的 L1 | 266310–267054 |
| 操场 | st_L1+L5 | 无 | — |

全部六台设备为：`Google_Pixel6`、`Google_Pixel_Watch1`、`Google_Pixel_Watch2`、`HUAWEI_Mate40`、`RedMi_K60`、`XiaoMi_MI8`。

## 4. 对旧结果的影响

旧 `y_response_state` 只显式标注过两次 st_L5 下的 Watch L1 异常，其他关联异常默认被视为 normal，且 direct 会覆盖 anomaly。因此其三分类结果只能作为旧标签体系下的开发性对照，不能再作为本文“跨频段关联异常联合诊断”的最终主结果。

场景分支的频段攻击识别结果不受该标签修订影响。构建器现已输出频段级 `y_associated_anomaly_l1`、`y_associated_anomaly_l5`、target-band direct 掩码、当前可用性与基线可用性掩码。是否训练场景条件化二分类响应分支必须先通过每折类别覆盖审计；当前静态数据不满足严格六折泛化评估条件，详见 `docs/scene_conditioned_response_audit_20260807.md`。

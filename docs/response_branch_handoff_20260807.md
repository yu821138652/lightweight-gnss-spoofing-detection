# 响应分支交接说明：频段级标签重构后的当前进度

> 更新时间：2026-08-07。适用范围：仓库现有、已审核的静态 GNSS 数据。原始张量、审计 JSON、checkpoint 和图像均在本地 `output/`，不提交 Git。

## 1. 当前结论

响应分支已经完成**标签语义修复、张量构建和六折类别覆盖审计**；但由于现有静态录制的正负样本分布不完整，不能再将其训练或汇报为一个经过严格六折泛化验证的 `normal / associated_anomaly` 分类器。

当前正式系统应理解为：

```text
场景分支（已完成六折泛化评估）
  → normal / L1 / L5 / L1+L5

场景条件化频段响应诊断（可解释证据层）
  → 目标频段：direct
  → 非目标频段：关联异常证据 / 无关联异常证据 / 不适用
```

场景分支仍是论文的主学习模型：四维 C/N0-only、T=5、TCN，在静态六折 pooled 上 Macro-F1 为 **0.9921**、Accuracy 为 **0.9944**。

## 2. 旧响应分支为什么失效

旧方案以每个设备窗口的互斥三分类作为真值：

```text
normal / anomaly / direct
```

这与真实频段响应不一致。双频设备可以在同一攻击时刻同时出现：

```text
目标 L5：direct
非目标 L1：associated anomaly
```

旧构建器会让 direct 覆盖 anomaly，且早期 CSV 只标了部分 Watch 异常，其他人工复核到的关联异常会默认落为 normal。因此旧完整链路的 `0.9802` Macro-F1、`98.76%` anomaly recall 等结果只保留为历史开发对照，不能进入论文最终主结果。

历史链路和边界见 `docs/complete_scene_response_diagnosis_20260806.md`。

## 3. 新的频段级真值

真值文件：`docs/device_band_association_intervals.csv`。

每条标签由以下字段定位：

```text
Environment / Scenario / Session / DeviceName / response_band / start_tow / end_tow
```

人工复核结论：

| 场景 | 关联异常真值 |
|---|---|
| 新主楼 `st_L1` | 无 |
| 新主楼 `st_L5` | 六台设备的 L1 均异常 |
| 新主楼 `st_L1+L5` | 无 |
| 操场 `st_L1` | Pixel6、MI8 的 L5 异常；其余设备 L5 正常 |
| 操场 `st_L5` | 六台设备的 L1 均异常 |
| 操场 `st_L1+L5` | 无 |

两只 Watch 没有 L5；但它们在 `st_L5` 中的 L1 仍是关联异常，因此不应删除或标成 L5 direct。

## 4. L5 消失型关联异常

关联异常包括两类：

1. **质量变化型**：频段仍可观测，但 C/N0、斜率、可见信号数或跟踪质量出现异常变化；
2. **可用性丢失型**：攻击前稳定具备该频段，攻击中该频段消失或显著减少。

操场 `st_L1` 中的 Pixel6、MI8 属于第二类。原始张量复核显示：两台设备攻击前都有 L5；攻击期间 L5 大量消失；攻击后恢复。因此“当前没有 L5”不能作为排除条件，反而是最强的 L5 关联异常证据之一。

新张量以每个源流前 30 个已审核正常窗口为固定基线，输出：

| 字段 | 含义 |
|---|---|
| `baseline_has_l1/l5` | 攻击前基线是否具备该频段能力 |
| `has_l1/l5` | 当前窗口是否仍可观测该频段 |
| `y_associated_anomaly_l1/l5` | 独立的频段关联异常标签 |
| `y_target_l1/l5` | 场景定义的目标频段掩码 |
| `y_direct_l1/l5` | 攻击区间内、目标频段且当前可观测的 direct 语义标签 |

`initial_baseline_delta_l5_log_signal_count` 可直接描述 L5 从可用到消失的变化。Watch 的 `baseline_has_l5=0`，因此不会被错误纳入 L5 异常判断。

## 5. 代码和运行入口

| 文件 | 当前作用 |
|---|---|
| `pipeline_total/36_build_device_attack_event_tensors.py` | 构建设备窗口特征及独立频段标签；支持 L1/L5 基线能力与可用性字段 |
| `pipeline_total/37_train_device_attack_event.py` | 保留二分类训练入口；`--sample-scope associated_l1/l5` 会只保留攻击中、非目标且**基线具备该频段**的窗口 |
| `pipeline_total/64_audit_scene_conditioned_band_labels.py` | 在任何训练前审计 split × 场景 × 设备 × 频段的类别覆盖、可用性丢失和 direct 标签 |
| `docs/scene_conditioned_band_response_label_spec_zh.md` | 标签语义规范 |
| `docs/scene_conditioned_response_audit_20260807.md` | 当前可审计方案与六折汇总 |

构建与审计命令模板：

```powershell
$py = 'H:\GNSS\program\Release_Package\Release_Package\venv\Scripts\python.exe'

& $py pipeline_total/36_build_device_attack_event_tensors.py `
  --signal-data-dir output/tensors/static_timeblock_outer_v2/fold_6 `
  --output-dir output/label_repair_v1/fold_6/device_tensors_scene_conditioned_cn0_extreme `
  --feature-set initial_baseline_delta_cn0_extreme `
  --device-aggregate-profile sparse_extreme `
  --initial-baseline-windows 30 `
  --initial-baseline-policy exclude_stream `
  --overwrite

& $py pipeline_total/64_audit_scene_conditioned_band_labels.py `
  --data-dir output/label_repair_v1/fold_6/device_tensors_scene_conditioned_cn0_extreme `
  --output-json output/label_repair_v1/fold_6/scene_conditioned_band_label_audit.json
```

## 6. 已完成的六折审计

静态 outer folds：`1 / 2 / 4 / 5 / 6 / 7`。结果如下：

| Fold | `L1→L5` 开发集 | `L1→L5` 测试集 | `L5→L1` 测试集 | 是否可作严格泛化分类评估 |
|---|---|---|---|---|
| 1 | 两类都有 | 仅 normal（0/1232） | 无 | 否 |
| 2 | 两类都有 | 无 | 仅 anomaly（2955/0） | 否 |
| 4 | 两类都有 | 无 | 无 | 否 |
| 5 | 仅 normal | anomaly/normal = 1266/1266 | 无 | 否，开发集无正类 |
| 6 | 两类都有 | 无 | 仅 anomaly（3886/0） | 否 |
| 7 | 两类都有 | 无 | 无 | 否 |

fold 6 对 L5 消失型异常的标签覆盖正确：

```text
train：895 anomaly + 1861 normal
val：  347 anomaly + 589 normal
```

但所有 fold 都不存在“开发集含正负类且对应测试集也含正负类”的同一关联异常任务；`L5→L1` 还完全没有 normal 反例。因此：

```text
禁止报告响应分类器的六折 Macro-F1、泛化 recall 或跨设备泛化结论。
```

## 7. 当前可用的系统语义

| 场景输出 | 可解释频段响应 |
|---|---|
| `normal` | 不输出攻击相关响应诊断 |
| `L1` | 当前可观测 L1 为 direct；对 `baseline_has_l5=1` 的设备检查 L5 消失/下降证据 |
| `L5` | 当前可观测 L5 为 direct；对 `baseline_has_l1=1` 的设备检查 L1 压制、可用性下降等证据 |
| `L1+L5` | 两个可观测目标频段均为 direct；当前静态人工复核未确认关联异常 |

其中 L5 攻击下的 L1 异常是当前两处静态录制中一致观察到的模式，不应写成所有终端必然发生的物理定律；可使用“关联异常证据”或“已观测模式”表述。

## 8. 后续应做什么

仓库内可用数据已经全部纳入本次审计，因此下一步不再继续尝试通过换模型、调参、过采样或重复切分来制造不存在的泛化评估条件。推荐工作顺序：

1. 以场景分支性能作为论文的正式模型主结果；
2. 对全部静态 Session 输出设备 × 频段的基线—攻击—恢复效应量表，包括 C/N0、频段可用信号数、下降比例和恢复时序；
3. 将 L5 消失、L1 压制写成可解释的响应诊断案例，而不是未被严格验证的通用分类器；
4. 在论文中公开标签覆盖审计和适用边界，避免旧三分类指标造成误导；
5. 动态数据仅在完成逐设备、逐频段人工标签后再纳入响应层结论。

## 9. 提交记录

- `05c896f 重构频段响应标签并补充覆盖审计`
  - 新增频段级关联异常标签、基线能力掩码和审计工具；
  - 更新 README、论文主线与历史完整链路的定位。


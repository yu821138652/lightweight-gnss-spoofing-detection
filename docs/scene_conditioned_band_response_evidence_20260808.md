# 场景条件化频段响应证据汇总（2026-08-08）

## 1. 目的与边界

本报告不是新的响应分类器，也不报告响应分类的泛化 Accuracy、Macro-F1 或 Recall。

它使用已经修复的**频段独立标签**，在每个静态 outer-test Session 中按“设备 × 频段”汇总：攻击前基线、攻击期间、攻击后恢复、频段可用性以及人工复核的 direct/associated-anomaly 语义。其用途是把跨频段响应现象变成可复核的量化证据，替代旧互斥三分类链中不成立的泛化指标。

当前的场景条件来自 response tensor 内嵌的**reviewed 场景真值**，并非新的 mixed 四折场景模型预测。因此：

- 该报告可用于验证响应标签与原始观测是否自洽；
- 不能作为“场景模型 + 响应模型端到端性能”；
- 后续若接入新场景分支，必须使用其按 `(recording, device, endpoint_tow)` 导出的预测后验，另行报告场景误差传递。

## 2. 输入与方法

输入为 `05c896f` 后构建的修复张量：

```text
output/label_repair_v1/fold_{1,2,4,5,6,7}/device_tensors_scene_conditioned_cn0_extreme/test.npz
```

每个设备频段按下列三段汇总：

```text
pre_attack  : 第一个 reviewed 攻击窗口之前的 normal 窗口
attack      : reviewed 攻击窗口
post_attack : 最后一个 reviewed 攻击窗口之后的 normal 窗口
```

对 L1、L5 分别输出：

- `log_signal_count`、C/N0 median、C/N0 q25 的三段中位数；
- `attack_minus_pre_*`：攻击期中位数减攻击前中位数；
- `attack_present_rate`：攻击期间该频段仍可观测的比例；
- `first_post_available_delay_s`：攻击结束后首次重新可观测的延迟；
- `reviewed_target_band`、`reviewed_direct_observed`、`reviewed_associated_anomaly` 和 `baseline_capable`。

脚本：`pipeline_total/75_summarize_scene_conditioned_band_response.py`。

## 3. 可复现命令

```powershell
$py = 'H:\GNSS\program\Release_Package\Release_Package\venv\Scripts\python.exe'

& $py pipeline_total/75_summarize_scene_conditioned_band_response.py `
  --data-root output/label_repair_v1 `
  --folds 1 2 4 5 6 7 `
  --split test `
  --output-dir output/analysis/scene_conditioned_band_response_static_v1 `
  --overwrite
```

输出文件：

```text
output/analysis/scene_conditioned_band_response_static_v1/
├── device_band_response_summary.csv
├── device_band_response_windows.csv
├── manual_association_label_coverage.csv
└── response_evidence_report.json
```

## 4. 当前静态 outer-test 汇总

当前运行覆盖 fold `1/2/4/5/6/7` 的 6 个可建立攻击前基线的静态 outer-test Session；fold 3 仍因录制开始于攻击中、没有合法初始正常基线而不进入本报告。

```text
recordings:               6
device-band summaries:   58
device-band window rows: 99,926
```

按诊断语义的设备×频段汇总数为：

| 语义 | 数量 | 解释 |
|---|---:|---|
| `target_band_direct_observed` | 29 | 目标频段在攻击期实际仍可观测，具有 direct 语义 |
| `target_band_unobserved` | 8 | 全局目标为该频段，但设备未观测到该频段，不能伪标为 direct |
| `non_target_associated_anomaly` | 13 | 有攻击前能力且在人工复核中确认关联异常 |
| `non_target_no_associated_anomaly_label` | 4 | 有攻击前能力、但人工复核未标记关联异常 |
| `non_target_not_applicable_no_baseline_capability` | 4 | 例如 Watch 的 L5：攻击前即无该能力，不适用 L5 响应判断 |

## 5. 标签覆盖审计结果

`docs/device_band_association_intervals.csv` 当前共有 14 条人工频段关联异常区间：

```text
完整覆盖：13
未覆盖：  1
```

唯一未覆盖项是：

```text
new_building / st_L5 / 2025.07.29.20.16_新主楼
Google_Pixel6 / L1
```

其 `coverage_status` 为 `stream_not_present_in_requested_outer_tests`。这表示当前外层测试张量中不存在该设备流，而不是“模型将其预测为正常”或“人工标签被否定”。论文或后续统计不能将该项记为响应检测漏检。

## 6. 已量化的代表性现象

### 6.1 L5 欺骗下的 L1 关联异常

在有覆盖的 `st_L5 → L1` 人工关联异常中，多数设备的攻击期 L1 C/N0 中位数相对攻击前下降。以本次汇总的 `attack_minus_pre_cn0_last_median` 为例：

- 新主楼的 Watch1/Watch2/Huawei/RedMi/MI8 分别约为 `-0.26/-0.73/-1.62/-1.39/-1.65 dB-Hz`；
- 操场的 Pixel6/Watch1/Watch2/Huawei/RedMi 分别约为 `-2.01/-0.74/-1.49/-2.53/-0.89 dB-Hz`。

操场 MI8 的 L1 是一个重要反例：攻击期 L1 可用率降至约 `49.4%`，信号数变化约为 `-5.98`，而残留可见信号的 C/N0 中位数并未同步下降。这说明不能只用“C/N0 中位数下降”定义关联异常；频段可用性和可见信号数同样是必要证据。

### 6.2 L1 欺骗下的 L5 消失型关联异常

操场 `st_L1` 中，Pixel6 与 MI8 的 L5 均为人工确认的关联异常：

- Pixel6：攻击期 L5 可用率约 `10.1%`，攻击后约 `99.6%`；
- MI8：攻击期 L5 可用率约 `1.3%`，攻击后约 `99.7%`；
- 两者均在攻击结束约 `3 s` 后再次出现可观测 L5。

这验证了“攻击中 L5 消失”是关联异常的可解释证据，而不是应被当作“不支持 L5、无需判断”的情形。对比之下，Watch 的 `baseline_has_l5=0`，因此从一开始就不应进入 L5 消失型诊断。

## 7. 与新场景分支的衔接

新的场景分支冻结协议是 `mixed_timeblock_outer_cv4_w5_v2 + TCN32`，输出为：

```text
global_scene = normal / L1 / L5 / L1+L5
global_scene_confidence
```

当前证据层在评估时以 reviewed 场景真值决定“目标频段/非目标频段”的语义；实际部署时应替换为满足置信度要求的 `global_scene`。推荐的在线解释规则是：

```text
normal -> 不输出攻击相关频段响应结论
L1     -> L1 可观测时报告 direct；检查 baseline_has_l5=1 的 L5 消失/下降证据
L5     -> L5 可观测时报告 direct；检查 baseline_has_l1=1 的 L1 压制/可用性下降证据
L1+L5  -> 两个可观测目标频段均报告 direct；当前静态人工复核未确认额外关联异常
```

这里的“检查”是基于设备自身攻击前基线的规则化证据诊断，不等同于已经通过完整正负 outer-test 验证的通用二分类器。

## 8. 下一步与限制

1. 新场景分支文档中声明的 `70–74` 单频路由脚本和 `output/frozen/...` 工件目前未出现在本地 checkout；在其代码和按窗口导出的预测文件同步前，不能把本报告强接为新的端到端融合结果。
2. 需要统一场景预测导出字段（至少含 fold、recording、device、endpoint_tow、四类 posterior / scene、confidence），再实现严格的一对一窗口对齐与场景误差传递审计。
3. 动态数据还没有逐设备、逐频段人工响应标签，不能写入本响应证据层的结论。
4. 现有静态数据没有为每个关联异常子任务提供同时具备 train/validation/test 正负样本的 outer-CV 覆盖；因此继续追求“关联异常分类器六折高分”不成立。当前正确产物是可审计的效应量、可用性变化与恢复案例。

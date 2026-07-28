# 设备响应状态分层检测方案同步

本文用于同步当前 GNSS 欺骗检测实验的新方案、已观察到的问题、已有结果和后续运行命令。

## 1. 为什么要改方案

原来的设备事件二分类大致是：

```text
normal / attack
```

也就是只要 Session 处在人工审核的攻击区间内，设备窗口就被看作正样本。

这个标签对 `L5-only` 场景不够准确。两个 Pixel Watch 都没有 L5 频段，Watch1 还没有 AGC 数据，因此它们不可能表现为“L5 直接被欺骗”。但是在 `st_L5` 审核图里，Watch 的 L1 C/N0 确实会在攻击区间内明显下降，表现为攻击关联异常。

因此当前更合理的标签语义是：

```text
0 = normal / no observable response
1 = attack-associated anomaly
2 = direct spoof
```

直观理解：

- `normal`：设备没有可观测异常响应。
- `attack-associated anomaly`：设备没有被该频段直接欺骗，但出现攻击关联异常，例如 L5 攻击下 Watch 的 L1 压制。
- `direct spoof`：设备/频段有直接受骗证据，例如目标频段 C/N0 上升。

重要原则：不修改原始 CSV 的观测值。人工响应区间单独写在 `docs/device_response_intervals.csv`，直接欺骗标签仍来自已有 target-band signal 标签。

## 2. 当前模型结构

当前最好的方向不是硬树：

```text
normal vs abnormal -> anomaly vs direct
```

硬树在 fold_6 能提高 direct recall，但会伤 anomaly recall；在 fold_7 还会降低 direct recall。

当前推荐方案是“平坦三分类 + 直接欺骗专家覆盖”：

```text
base model:
  normal / anomaly / direct 三分类

direct expert:
  direct vs non-direct 二分类

inference:
  先用 base model 输出三分类；
  再用 direct expert 的概率判断是否覆盖成 direct。
```

direct override 的阈值不手动看 test 调，而是在 validation 上自动选择：

```text
max_val_far = 0.05
min_val_abnormal_recall = 0.90
在满足约束的阈值里优先选 macro-F1 / direct recall 更好的。
```

## 3. 指标含义

不要只看 Watch L5 recall。需要同时看整体指标。

- `abnormal recall`：真实为 anomaly 或 direct 的窗口，有多少被模型判成非 normal。它衡量“有没有发现异常”。
- `anomaly recall`：真实为 attack-associated anomaly 的窗口，有多少被正确判成 anomaly。它衡量 Watch L5 这类攻击关联异常是否被正确建模。
- `direct recall`：真实为 direct spoof 的窗口，有多少被正确判成 direct。它衡量真正被欺骗的设备/频段有没有被正确识别。
- `FAR`：真实 normal 的窗口里，有多少被误报成异常。

如果只做异常告警，目前结果已经比较好；如果要区分“直接欺骗”和“攻击关联异常”，还需要 direct expert。

## 4. 当前已有结果

本地目前至少已有 `fold_1`、`fold_6`、`fold_7` 的 signal tensor 目录。此前完整跑完并记录的是 `fold_6` 和 `fold_7`。

平坦三分类结果：

| fold | outer 场景 | Macro-F1 | FAR | abnormal recall | anomaly recall | direct recall |
|---|---|---:|---:|---:|---:|---:|
| fold_6 | st_L5 | 0.7766 | 2.83% | 90.64% | 97.58% | 47.06% |
| fold_7 | st_L1+L5 | 0.6588 | 0.22% | 97.19% | n/a | 97.19% |

validation 校准 direct override 后：

| fold | outer 场景 | Macro-F1 | FAR | abnormal recall | anomaly recall | direct recall |
|---|---|---:|---:|---:|---:|---:|
| fold_6 | st_L5 | 0.8529 | 2.85% | 91.73% | 97.52% | 65.36% |
| fold_7 | st_L1+L5 | 0.6593 | 0.28% | 97.51% | n/a | 97.51% |

两折平均：

```text
Macro-F1: 0.7561
FAR: 1.57%
abnormal recall: 94.62%
direct recall: 81.43%
anomaly recall: 97.52%  # 只在有 anomaly support 的 fold_6 上统计
```

结论：

- 设备异常检测本身已经比较强。
- Watch L5 的攻击关联异常能够被检出，不是单纯靠误报。
- direct spoof 在 fold_6 原本明显漏检，加入 direct expert 后有明显改善。
- 仍需补齐完整静态 7-fold，不能只凭 fold_6/fold_7 下最终结论。

## 5. 需要继续做什么

目标是完成静态 7-fold 响应状态实验。

先查看哪些 signal tensor 已存在：

```powershell
Get-ChildItem output/tensors/static_timeblock_outer_v2 -Directory | Select-Object Name
```

缺哪个 fold，就先构建哪个 fold 的 signal tensor。例如构建 `fold_2`：

```powershell
$PY = "H:\GNSS\program\Release_Package\Release_Package\venv\Scripts\python.exe"

& $PY pipeline_total/20_build_static_timeblock_tensors.py `
  --outer-manifest output/protocols/static_time_block_outer_v2/fold_2/recording_split_manifest.csv `
  --block-manifest output/protocols/static_time_block_outer_v2/fold_2/epoch_split_manifest.csv `
  --output-dir output/tensors/static_timeblock_outer_v2/fold_2 `
  --time-steps 5 `
  --block-size 256
```

然后对该 fold 运行响应状态完整流水：

```powershell
& $PY pipeline_total/43_run_static_response_state_fold.py `
  --fold fold_2 `
  --python-exe $PY `
  --overwrite-device-tensors
```

`43_run_static_response_state_fold.py` 会自动完成：

1. 构建设备响应状态张量；
2. 训练三分类 base model；
3. 测试三分类 base model；
4. 训练 direct expert；
5. 在 validation 上校准 direct override 阈值；
6. 在 outer test 上评估 override 结果。

## 6. 一次性补齐缺失 folds 的参考命令

如果当前只缺 `fold_2` 到 `fold_5`，可以运行：

```powershell
$PY = "H:\GNSS\program\Release_Package\Release_Package\venv\Scripts\python.exe"

foreach ($F in 2..5) {
  & $PY pipeline_total/20_build_static_timeblock_tensors.py `
    --outer-manifest "output/protocols/static_time_block_outer_v2/fold_$F/recording_split_manifest.csv" `
    --block-manifest "output/protocols/static_time_block_outer_v2/fold_$F/epoch_split_manifest.csv" `
    --output-dir "output/tensors/static_timeblock_outer_v2/fold_$F" `
    --time-steps 5 `
    --block-size 256
}
```

然后：

```powershell
foreach ($F in 2..5) {
  & $PY pipeline_total/43_run_static_response_state_fold.py `
    --fold "fold_$F" `
    --python-exe $PY `
    --overwrite-device-tensors
}
```

如果某个 fold 已经存在，但不确定是否完整，先不要删除，先把报错发出来。

## 7. 汇总结果

等 7 个 fold 都跑完后，运行：

```powershell
$metrics = 1..7 | ForEach-Object {
  "output/hierarchical_event_v1/static_response_state_v1/fold_$_/direct_override_mlp_h32_valcal_all/test_metrics_response_state_direct_override.json"
}

& $PY pipeline_total/40_summarize_response_state_metrics.py `
  --metrics $metrics `
  --output-csv output/hierarchical_event_v1/static_response_state_v1/static_response_state_direct_override_7fold_summary.csv
```

输出 CSV：

```text
output/hierarchical_event_v1/static_response_state_v1/static_response_state_direct_override_7fold_summary.csv
```

## 8. 当前代码提交

与该方案相关的近期提交包括：

```text
5ecded2 新增(分层检测): 支持设备响应状态诊断实验
00f9466 新增(分层检测): 记录静态响应状态试跑结果
9de107e 新增(分层检测): 加入直接欺骗专家覆盖评估
74fe5ac 新增(分层检测): 支持直接欺骗覆盖阈值校准
d518aac 新增(分层检测): 加入静态响应状态折级流水脚本
```

## 9. 给论文叙事的建议

可以把这个现象写成论文中的一个关键发现：

```text
同一攻击场景下，不同设备由于频段能力和芯片链路不同，会出现不同响应。
L5-only 攻击中，双频设备可能出现 L5 直接欺骗，而 L1-only Watch 不会出现 L5 直接欺骗，
但会在 L1 上表现出攻击关联压制异常。
```

所以检测任务不应该只有 `normal/spoof`，而应该区分：

```text
normal / no observable response
attack-associated anomaly
direct spoof
```

这既能解释 Watch L5 的现象，也能避免为了救 Watch 而牺牲整体检测指标。

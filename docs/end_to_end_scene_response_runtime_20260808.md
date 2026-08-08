# 场景预测驱动的端到端融合与轻量化指标（2026-08-08）

## 1. 评估范围

本次评估不再使用人工场景标签作为融合输入，而是读取场景分支在 outer-test 上导出的模型预测：

```text
场景分支：static_scene_response_fusion_s1_cn0_only / band_scene
输入：4 维 C/N0 + presence，窗口长度 5，TCN32
响应分支：response_extreme_cv_v1，44 维 C/N0 extreme，MLP-h16
后融合：direct_override_global_t020_mlp_h16
fold：1、2、4、5、6、7
```

严格融合脚本为 `pipeline_total/50_fuse_static_scene_response_predictions.py`。它按
`(fold, endpoint_tow)` 对齐场景预测，并将同一时刻的场景后验附加到每台设备的响应预测上；如果没有对应场景预测，则保留为缺失，不使用人工场景真值补齐。

## 2. 端到端融合结果

产物：

```text
output/hierarchical_event_v1/static_scene_response_fusion_s1_cn0_only_response44_t020/fusion_recheck/
```

结果：

| 指标 | 数值 |
|---|---:|
| 响应窗口数 | 49,963 |
| 场景上下文覆盖率 | 99.8599% |
| 场景 Macro-F1 | 0.99399 |
| 设备异常（normal 以外）Recall | 0.93726 |
| 场景与设备状态联合 Recall | 0.91451 |

这里的联合 Recall 要求设备响应状态和全局场景同时正确，因而比单独的场景或响应 Recall 更严格。

## 3. 模型规模与推理时间

测量环境：Windows CPU，Python 3.8.10，PyTorch 1.7.1+cpu，单线程；输入张量和标准化均已预构建，因此以下是模型前向推理时间，不包含原始 CSV 解析。

### 3.1 参数量

| 模块 | 配置 | 参数量 | checkpoint |
|---|---|---:|---:|
| 场景分支 | TCN32，4 维输入，5 步窗口，4 类 | 4,772 | 23,935 B |
| 响应 backbone | MLP-h16，44 维输入，3 类 | 771 | 5,135 B |
| direct expert | MLP-h16，44 维输入，2 类 | 754 | 5,071 B |
| 合计 | 三个轻量前向模块 | 6,297 | 34,141 B |

### 3.2 单窗口前向时间

fold 6 代表性测量结果：

| 模块 | P50 | P95 |
|---|---:|---:|
| 场景 TCN32 | 0.253 ms | 0.453 ms |
| 响应 backbone | 0.035 ms | 0.048 ms |
| direct expert | 0.042 ms | 0.054 ms |

三阶段顺序执行的模型前向时间按阶段测量值相加约为：

```text
P50 ≈ 0.330 ms
P95 ≈ 0.555 ms
```

该值不包含数据聚合、规则判断和文件 I/O，论文中应标注为“模型前向延迟”。

## 4. 检测延迟（TTD）

使用 `pipeline_total/52_measure_response_ttd.py`，`hold_windows=1`、`max_gap_tow=1.5`，在融合预测 CSV 上统计：

| 目标 | 事件数 | 检出率 | 中位 TTD | P90 TTD |
|---|---:|---:|---:|---:|
| abnormal | 29 | 96.55% | 6 TOW | 11.3 TOW |
| anomaly | 4 | 75.00% | 6 TOW | 6 TOW |
| direct | 25 | 100.00% | 6 TOW | 42 TOW |

TTD 使用 TOW 差值表示；只有在确认采样间隔为 1 秒后，才可将其直接写成秒。

## 5. 结果边界

当前本地可用的“最新场景预测”是 `static_scene_response_fusion_s1_cn0_only` 的静态分支预测。文档中提到的 `mixed_timeblock_outer_cv4_w5_v2` 冻结产物目前未出现在本地 checkout，因此本结果不能冒充 mixed 四折冻结分支的端到端结果。

此外，TTD 统计仍基于当前人工审核事件区间，anomaly 事件只有 4 个，属于开发性测量；模型规模和前向时间已经可以作为论文轻量化指标，TTD 还需在更多独立 Session 上复核。


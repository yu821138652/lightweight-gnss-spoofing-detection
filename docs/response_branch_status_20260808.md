# 响应分支当前方案与结果（2026-08-08）

## 1. 当前定位

响应分支不再把所有设备、所有频段压成互斥的 `normal / anomaly / direct` 三分类真值。当前采用“场景先确定目标频段，再对非目标频段判断是否存在关联异常”的语义：

```text
场景分支：normal / L1 / L5 / L1+L5
响应分支：normal / associated_anomaly / direct
```

其中：

- 场景目标频段直接输出 `direct`；
- 非目标且设备攻击前具备该频段能力的频段，判断 `normal` 或 `associated_anomaly`；
- 攻击前没有该频段能力的设备（例如没有 L5 的 Watch）不进入该频段的异常判断。

这种定义允许同一台双频设备在同一时刻同时具有“目标频段 direct”和“非目标频段 associated anomaly”。

## 2. 真实标签语义

当前响应张量使用独立的频段标签：

```text
y_target_l1 / y_target_l5
y_direct_l1 / y_direct_l5
y_associated_anomaly_l1 / y_associated_anomaly_l5
baseline_has_l1 / baseline_has_l5
has_l1 / has_l5
```

标签定义：

```text
direct = 攻击期间 + 当前频段是场景目标频段 + 当前频段可观测
associated anomaly = 攻击期间 + 当前频段是非目标频段
                    + 攻击前具备该频段能力 + 人工复核为异常
```

人工观测已确认的代表性事实：

- 新主楼、操场 `st_L5`：具备 L1 的设备均出现 L1 关联异常，包括没有 L5 的两台 Watch；
- 操场 `st_L1`：只有 Google Pixel6 和 XiaoMi MI8 的 L5 标为关联异常，其余具备 L5 的设备保持正常；
- Watch 的 L5 不能因为“没有观测”而标异常，因为其攻击前就没有 L5 能力。

完整标签说明见 `docs/scene_conditioned_band_response_label_spec_zh.md`，人工区间见 `docs/device_band_association_intervals.csv`。

## 3. 当前模型与规则

### 3.1 响应 backbone

```text
聚合：sparse_extreme
基线：攻击前初始 30 个窗口
特征：44 维 C/N0 extreme + 初始基线差分
模型：MLP-h16
输出：normal / anomaly / direct
```

### 3.2 direct 专家

使用独立的 2 类 MLP-h16 direct expert，对 direct 候选进行补充判断。全局阈值在 validation 上冻结后用于 outer-test，当前实验配置为 `t=0.20`。

### 3.3 统一关联异常规则 v2

L1、L5 使用同一套频段无关规则，不再分别写死不同阈值：

```text
associated_anomaly_evidence
    = individual_availability_loss OR cohort_consensus_evidence
```

- `individual_availability_loss`：攻击前可用率至少 90%，攻击期可用率不高于 50%，攻击后恢复到基线 10% 以内；
- `cohort_consensus_evidence`：同一 Session、同一非目标频段，至少 4 台设备具备基线能力，且至少 75% 的设备持续 30 个窗口出现同方向 C/N0 偏离；
- 设备数不足时，群体规则自动不适用，只保留个体严重可用性丢失证据；
- Watch 只可能通过 L1 关联异常通道被判为 anomaly，不会被写成 L5 direct。

实现与审计脚本：

```text
pipeline_total/75_summarize_scene_conditioned_band_response.py
```

## 4. 当前规则审核结果

在 6 个静态 outer-test Session 的人工审核覆盖集上：

| 对象 | 结果 |
|---|---:|
| 人工确认关联异常 | 13/13 检出 |
| 人工确认正常非目标频段 | 4/4 未误报 |
| 人工异常但 outer-test 中缺少设备流 | 1 条，无法评估 |

这组结果是“规则与人工审核集的一致性审计”，不是具有完整独立正负样本覆盖的通用分类器 Accuracy。

产物目录：

```text
output/analysis/scene_conditioned_band_response_static_v3_cohort/
```

## 5. 使用模型场景预测的严格端到端融合

当前本地可用的最新静态场景预测来自：

```text
output/training/static_scene_response_fusion_s1_cn0_only/band_scene/
```

使用 `pipeline_total/50_fuse_static_scene_response_predictions.py`，按 `(fold, endpoint_tow)` 对齐场景模型预测和响应模型预测，不使用人工场景真值补齐。

当前结果：

| 指标 | 数值 |
|---|---:|
| 响应窗口数 | 49,963 |
| 场景上下文覆盖率 | 99.8599% |
| 场景 Macro-F1 | 0.99399 |
| 设备异常 Recall | 0.93726 |
| 场景-响应联合 Recall | 0.91451 |

产物：

```text
output/hierarchical_event_v1/static_scene_response_fusion_s1_cn0_only_response44_t020/fusion_recheck/
```

## 6. 轻量化与检测延迟

测量环境为单线程 CPU、PyTorch 1.7.1+cpu，输入张量已预构建：

| 模块 | 参数量 | 单窗口 P50 | 单窗口 P95 |
|---|---:|---:|---:|
| 场景 TCN32 | 4,772 | 0.253 ms | 0.453 ms |
| 响应 backbone MLP-h16 | 771 | 0.035 ms | 0.048 ms |
| direct expert MLP-h16 | 754 | 0.042 ms | 0.054 ms |
| 合计 | 6,297 | 约 0.330 ms | 约 0.555 ms |

TTD 使用 `hold_windows=1`、`max_gap_tow=1.5`：

| 目标 | 检出率 | 中位 TTD | P90 TTD |
|---|---:|---:|---:|
| abnormal | 96.55% | 6 TOW | 11.3 TOW |
| anomaly | 75.00% | 6 TOW | 6 TOW |
| direct | 100.00% | 6 TOW | 42 TOW |

TTD 目前以 TOW 差值报告；只有确认采样间隔后，才能换算成秒。

详细测量记录见 `docs/end_to_end_scene_response_runtime_20260808.md`。

## 7. 论文可用性与边界

当前结果已经足够开始撰写论文的方法、标签重构、规则设计、静态端到端系统和轻量化实验章节。但定稿时必须说明：

1. 当前本地严格端到端结果使用的是静态 `s1_cn0_only` 场景预测；mixed 冻结场景产物尚未同步到本地；
2. `13/13、4/4` 是人工审核集一致性，不是通用分类器泛化准确率；
3. 动态场景尚缺少逐设备、逐频段人工响应标签，不能直接外推静态响应结论；
4. 新主楼 `st_L5` 的 Google Pixel6 L1 人工异常区间在当前 outer-test 中缺少设备流；
5. 场景分支适合广播式多星同频攻击，不能定位单星或选择性 PRN 欺骗。


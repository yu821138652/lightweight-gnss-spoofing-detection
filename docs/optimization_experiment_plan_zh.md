# 方案优化实验清单

> 目标：在论文正式定稿前，对场景分支、设备响应分支、后融合和轻量化能力进行受控优化，选出一个跨设备稳定、低误报、可解释且资源开销可报告的最终候选方案。
>
> 本清单强调“一次只改变一个主要因素”。每个实验都必须保存配置、特征名、fold 结果、逐设备结果和资源指标，不能只保留一个 pooled 分数。

## 1. 总体原则

### 1.1 先冻结协议，再优化模型

- 完整 Session/outer group 必须隔离 train、validation 和 test。
- scaler、初始基线、类别权重和阈值只使用 train/validation。
- 当前已经反复读取的 outer test 只能作为开发诊断，不能继续手调后包装成新的独立盲测。
- 如果没有新的完全未读数据，最终结果应明确标注为“开发性多折结果”；正式论文最好保留一个最后不读取的 Session/设备组作为确认集。

### 1.2 统一主指标

设备响应分支：

- abnormal recall；
- anomaly recall；
- direct recall；
- FAR；
- 每设备 recall、最差设备 recall；
- supported Macro-F1。

场景分支：

- Macro-F1；
- `normal/L1/L5/L1+L5` recall；
- 场景 FAR；
- 每设备、每 Session recall。

融合系统：

- joint recall；
- anomaly joint recall；
- direct joint recall；
- 场景上下文覆盖率；
- FAR。

轻量化与实时性：

- 输入维度；
- 可训练参数量；
- 权重文件大小；
- MACs/FLOPs；
- CPU 推理 p50/p95 延迟；
- 峰值推理内存；
- 特征构建耗时；
- 事件检测率、Median/P90 TTD；
- 初始化预热时间和算法窗口延迟。

### 1.3 候选方案接受标准

候选方案至少需要满足：

1. FAR 不超过预先设定的约束，建议开发阶段固定为不高于当前基线约 2% 的水平；
2. abnormal recall 不明显低于当前方案；
3. direct recall 不因优化 anomaly 而发生明显崩溃；
4. Huawei、RedMi、Pixel Watch 等关键设备不能出现新的系统性漏检；
5. 提升不能只来自某一个 Session 或某一个设备；
6. 多个随机种子或多数 fold 上方向一致；
7. 在性能接近时，优先选择输入更少、参数更少、延迟更低的方案。

## 2. 实验目录与命名

所有优化产物建议写入被 Git 忽略的本地目录：

```text
output/optimization/
  response_feature_ablation_v1/
  response_model_size_v1/
  scene_feature_ablation_v1/
  scene_model_compare_v1/
  fusion_compare_v1/
  realtime_metrics_v1/
```

每个实验目录至少保留：

```text
experiment_config.json
feature_names.json
train_metrics.json
val_metrics.json
test_metrics.json              # 仅在协议允许且明确标注开发性时生成
per_device_metrics.csv
per_fold_metrics.csv
resource_metrics.json
```

## 3. 阶段 0：基线复核与协议冻结

### O0：复核当前设备响应基线

**目的**：确认后续消融的参考点，不重新选择模型。

**参考配置**：

- `sparse_extreme` 聚合；
- `initial_baseline_delta_with_device`；
- 初始正常基线 30 个窗口；
- MLP hidden=32；
- 三分类 base + direct expert；
- 现有 validation FAR 约束和阈值校准规则。

**输出**：当前 6 个有效静态 fold 的统一表格、逐设备结果和资源基线。

**通过条件**：结果与已有记录一致；如果不一致，先查清协议或张量版本，禁止直接进入后续优化。

## 4. 阶段 1：设备响应分支特征消融

这一阶段固定 MLP hidden=32，只改变输入特征。三分类 base 和 direct expert 都要单独记录结果；不假设二者必须使用同一特征集。

### R1：去设备身份编码

```text
当前完整输入 → 去掉 device_is_* one-hot
```

**目的**：检查模型是否依赖设备身份捷径。重点看跨设备泛化、Watch、Huawei 和 RedMi。

### R2：去跨频段差值与耦合特征

```text
当前完整输入 → 去掉 l5_minus_l1_* 和 coupled_*
```

**目的**：验证跨频段耦合特征对 anomaly/direct 区分是否真正有效。

### R3：只保留初始基线差分

```text
initial_baseline_delta_only
```

**目的**：判断初始正常基线是否比绝对聚合量更具有跨设备泛化性。

### R4：初始基线差分 + 设备可用性信息

保留 L1/L5 是否存在、AGC 是否可用等能力掩码，但不直接输入设备身份 one-hot。

**目的**：区分“设备能力差异”与“设备身份记忆”。

### R5：紧凑核心特征

只保留：

- L1/L5 信号数或 presence；
- C/N0 水平；
- C/N0 斜率；
- 覆盖率/丢失率；
- 初始基线差分；
- 经过筛选的 L1/L5 耦合特征。

**目的**：构造低维、可解释的响应分支候选。

### R6：加入跟踪状态和物理动态特征

在 R5 基础上加入后续重建的：

- `PseudorangeRateMetersPerSecond` 的水平、斜率和异常比例；
- `State` / `AccumulatedDeltaRangeState` 的有效比例和切换次数；
- `AccumulatedDeltaRangeMeters` 的差分、跳变和重捕获比例。

**目的**：检验新增物理字段是否补足现有 C/N0 看不到的异常信息。

### 响应分支阶段 1 的选择规则

先按以下顺序筛选：

```text
FAR → abnormal recall → direct recall → anomaly recall → 最差设备 recall
```

若两个版本性能接近，选择特征维度更低、设备身份依赖更弱的版本。

## 5. 阶段 2：设备响应分支模型规模与结构

固定阶段 1 选出的前两种特征集，比较：

```text
Linear
MLP hidden=8
MLP hidden=16
MLP hidden=32
```

只有在单窗口输入无法解释时间变化时，再增加：

```text
GRU/TCN 小型时序模型
LightGBM 或其他浅层树模型
```

不建议一开始同时改变特征、模型和标签，否则无法判断提升来源。

### R7：直接欺骗专家单独消融

对 direct expert 单独比较：

- 完整特征；
- R1 去设备身份；
- R5 紧凑核心特征；
- R6 物理动态特征。

**目的**：避免 base model 的最优特征集被直接套给 direct expert。

## 6. 阶段 3：场景分支特征优化

场景分支固定窗口、fold 和 TCN 结构，先做逐组特征加入。

### S0：当前去 AGC 8 维基线

作为主参考版本：L1/L5 的 C/N0、接收机时间不确定度、伪距率不确定度和 presence。

### S1：C/N0-only

作为低维跨设备候选，必须在和融合方案完全相同的协议下复验，不能直接引用旧的非同协议结果。

### S2：加入伪距率动态特征

加入 `PseudorangeRateMetersPerSecond` 的频段聚合、斜率、稳健离散度和异常比例。

### S3：加入 State/ADR 连续性特征

加入跟踪状态比例、状态切换、ADR 跳变和重捕获比例，并保留缺失掩码。

### S4：加入伪距残差

使用因果 PV 解或 leave-one-satellite-out 残差，按 L1/L5 聚合：

- median；
- MAD；
- P95；
- 异常卫星比例；
- L1/L5 残差差值。

必须检查残差计算没有未来信息或跨 split 泄漏。

### S5：跨频段相对特征

加入：

- L5-L1 差值；
- L5 上升与 L1 下降的联合比例；
- L1/L5 可见性变化；
- 跨频段状态切换差异。

## 7. 阶段 4：场景分支模型比较

对阶段 3 最好的两个特征版本比较：

```text
TCN hidden=16
TCN hidden=32
GRU hidden=16
轻量 1D CNN
窗口统计量 + Linear/MLP
```

模型选择不以 validation 单点 Macro-F1 为唯一标准，必须同时查看跨 fold 场景 recall、最差设备 recall 和 FAR。

## 8. 阶段 5：后融合优化

固定最好的场景分支和响应分支，只比较融合方式：

### F1：简单平均

当前方案，多设备同一 TOW 的场景后验平均。

### F2：按训练/validation 可靠性加权

权重只能由 development 数据得到，不能按 outer test 手工设置。

### F3：按频段可用性加权

有 L5 的设备提供 L5 场景证据，无 L5 的 Watch 仍保留其 anomaly 响应。

### F4：校准概率融合

比较温度校准、等权概率和简单逻辑组合，不重新训练大型统一模型。

最终比较：

- joint recall；
- anomaly joint recall；
- direct joint recall；
- FAR；
- 场景上下文覆盖率；
- 各设备结果。

## 9. 阶段 6：实时性与轻量化实验

对最终候选和代表性对照统一测量：

### L1：模型资源

- 参数量；
- FP32 权重大小；
- INT8 权重大小；
- MACs/FLOPs；
- 峰值内存。

### L2：运行速度

- 单窗口 CPU 前向 p50/p95；
- 特征聚合 p50/p95；
- 完整端到端 p50/p95；
- 吞吐量（windows/s）。

### L3：检测时间

对每个攻击事件计算：

```text
TTD = first_valid_alarm_time - reviewed_attack_start_time
```

分别报告：

- abnormal TTD；
- anomaly TTD；
- direct TTD；
- Median TTD；
- P90 TTD；
- 1/3/5/10 秒内检测率；
- 漏检事件比例。

同时单独报告：

- 窗口长度；
- stride；
- 初始正常基线预热时间；
- 连续告警所需窗口数；
- 算法检测延迟和 CPU 推理延迟。

## 10. 最终方案确认

### C1：开发集冻结

冻结：

- 特征版本；
- 模型结构和 hidden size；
- 类别权重；
- 阈值；
- 融合方式；
- 连续告警规则；
- TTD 计算口径。

### C2：最终确认集

优先使用尚未读取的新 Session、新设备或新的 outer group。若无法获得新数据，则如实报告为开发性多折结果，并说明 test 已被用于迭代诊断。

### C3：论文最终表格

至少包含：

1. 特征消融表；
2. 模型规模对比表；
3. 场景分支结果；
4. 设备响应状态结果；
5. 后融合结果；
6. 逐设备结果；
7. TTD 与实时性表；
8. 适用范围和失败案例。

## 11. 建议的实际执行顺序

```text
O0 复核当前基线
↓
R1 去设备身份编码
↓
R2 去跨频段特征
↓
R3/R4 初始基线与能力掩码
↓
R5 紧凑核心特征
↓
R7 direct expert 单独消融
↓
R6 加入 State/伪距率/ADR
↓
响应分支 Linear/MLP-8/16/32
↓
场景 S1–S5 特征消融
↓
场景 TCN/GRU/轻量 CNN 比较
↓
融合 F1–F4
↓
TTD、CPU 延迟、内存和模型大小
↓
冻结最终候选并做一次确认测试
```

第一步应先完成 O0，不要直接从新特征或新模型开始。只有基线版本、张量版本和指标口径一致，后面的消融结果才有解释意义。


# 设备响应状态分层检测完整方案

本文是一份完整方案文档，用于说明当前 GNSS 欺骗检测项目中“设备响应状态分层检测”的研究动机、标签定义、数据处理、模型设计、实验结果、例外情况和后续计划。

## 1. 问题背景

项目原先主要使用二分类任务：

```text
normal / attack
```

这种标签方式默认认为：只要某个 Session 处于人工审核的攻击区间内，所有设备窗口都应被视为攻击正样本。

但在目前的数据中，尤其是 L5-only 攻击场景下，这种标签语义会出现明显问题。

以 Pixel Watch 为例：

- Pixel Watch1 和 Pixel Watch2 都没有 L5 频段；
- Pixel Watch1 没有 AGC 数据；
- 因此在 L5-only 攻击中，Watch 不可能表现为“L5 直接被欺骗”；
- 但人工审核图显示，Watch 的 L1 C/N0 会在 L5 攻击区间内明显下降，表现为攻击关联的 L1 压制异常。

这说明同一个全局攻击事件下，不同设备可能有不同响应：

```text
双频设备：可能出现目标频段直接欺骗；
L1-only Watch：不会出现 L5 直接欺骗，但可能出现 L1 攻击关联异常；
某些设备：可能没有明显可观测响应。
```

因此，单纯的 `normal / attack` 标签会把不同物理含义的现象混成同一类，导致模型既要学习“目标频段 C/N0 上升”，又要学习“非目标频段 C/N0 下降”，泛化容易变差。

## 2. 核心假设

当前方案的核心假设是：

```text
模型效果差的主要原因之一不是特征完全无效，而是标签语义过粗。
```

更合理的建模目标不是直接判断“全局是否处于攻击区间”，而是判断设备当前属于哪种响应状态：

```text
normal / no observable response
attack-associated anomaly
direct spoof
```

这种标签能够区分：

- 设备是否异常；
- 异常是否与攻击相关；
- 设备是否存在直接欺骗证据；
- Watch 这种无 L5 设备在 L5 攻击中的 L1 压制现象。

## 3. 标签体系

当前定义三类设备响应状态：

| 标签 ID | 标签名称 | 含义 |
|---:|---|---|
| 0 | normal / no observable response | 正常，或攻击区间内没有可观测异常响应 |
| 1 | attack-associated anomaly | 攻击关联异常，但不是直接欺骗 |
| 2 | direct spoof | 有直接欺骗证据 |

具体例子：

| 场景 | 设备/频段表现 | 响应状态 |
|---|---|---|
| st_L1 | L1 C/N0 上升 | direct spoof |
| st_L5 | 双频设备 L5 C/N0 上升 | direct spoof |
| st_L5 | Watch 无 L5，但 L1 C/N0 下降 | attack-associated anomaly |
| st_L5 | 某设备无明显变化 | normal / no observable response |
| st_L1+L5 | L1 或 L5 直接受骗 | direct spoof |

注意：本方案不修改原始 CSV 的观测值。人工响应区间写在：

```text
docs/device_response_intervals.csv
```

直接欺骗标签仍来自已有 target-band signal 标签。

## 4. 数据处理方案

当前使用设备窗口作为样本单位：

```text
one device x one endpoint time window = one sample
```

设备级特征由逐 signal 的 W5 统计聚合得到，当前主要使用：

```text
device_aggregate_profile = sparse_extreme
feature_set = initial_baseline_delta_with_device
initial_baseline_windows = 30
```

含义如下：

- `sparse_extreme`：保留稀疏但强烈的跨频段变化，例如极端 C/N0 上升/下降；
- `initial_baseline_delta_with_device`：每个设备流以前 30 个正常窗口作为初始基线，后续窗口使用相对基线差值；
- `device one-hot`：显式给模型设备身份信息，使其能够学习不同硬件响应差异。

选择初始基线的原因：

- rolling/causal baseline 容易被持续攻击污染；
- 固定初始正常基线更能表达“相对正常状态的变化”；
- 对 L5 攻击下 L1 压制这种现象更直接。

但该协议有一个输入条件：

```text
设备流开始阶段必须存在一段审核过的正常观测。
```

这也是 fold_3 目前不能直接评估的原因。

## 5. 模型结构

我们尝试过硬树结构：

```text
Stage 1: normal vs abnormal
Stage 2: anomaly vs direct
```

结果显示，硬树不是当前最优：

- fold_6 中 direct recall 有提升；
- 但 anomaly recall 被伤害；
- fold_7 中 direct recall 下降；
- 因此硬树过于依赖第一阶段的错误传播。

当前推荐结构是：

```text
flat three-class base model
+
binary direct expert override
```

即：

1. base model 直接输出三分类：

```text
normal / anomaly / direct
```

2. direct expert 训练为二分类：

```text
direct / non-direct
```

3. 推理时先使用 base model 输出初始类别，再使用 direct expert 判断是否覆盖为 direct：

```text
if direct_probability >= threshold:
    prediction = direct
else:
    prediction = base_prediction
```

当前 direct override 使用 validation 自动选择阈值，而不是看 test 手动调参。

阈值选择约束：

```text
max_val_far = 0.05
min_val_abnormal_recall = 0.90
```

在满足约束的候选阈值中，优先选择 Macro-F1、direct recall 和 abnormal recall 更好的阈值。

## 6. 指标定义

本方案不能只看 Watch L5 recall，也不能只看 Macro-F1。需要同时报告以下指标。

### 6.1 abnormal recall

衡量“有没有发现异常”：

```text
真实标签属于 {anomaly, direct} 的样本中，
有多少被预测为 {anomaly, direct}
```

如果真实是 direct，但预测成 anomaly：

```text
abnormal recall 算对
direct recall 算错
```

### 6.2 anomaly recall

衡量攻击关联异常是否被正确识别：

```text
真实标签为 anomaly 的样本中，
有多少被预测为 anomaly
```

Watch L5 中的 L1 压制异常主要看这个指标。

### 6.3 direct recall

衡量直接欺骗是否被正确识别：

```text
真实标签为 direct 的样本中，
有多少被预测为 direct
```

双频设备目标频段 C/N0 上升这类现象主要看这个指标。

### 6.4 FAR

衡量正常误报率：

```text
真实 normal 的样本中，
有多少被预测为 anomaly 或 direct
```

### 6.5 supported Macro-F1

有些静态 outer test 中只有 `normal/direct`，没有 `anomaly` 类。如果仍然把不存在的 anomaly 类按 0 分计入 Macro-F1，结果会被人为压低。

因此当前同时报告：

```text
raw Macro-F1
supported Macro-F1
```

其中 `supported Macro-F1` 只平均 test 中真实存在的类别。

## 7. 当前实验结果

当前响应状态流水已经在 6 个有效静态 fold 上得到结果：

```text
fold_1, fold_2, fold_4, fold_5, fold_6, fold_7
```

`fold_3` 暂不计入主结果，原因见第 8 节。

6 个有效 fold 的 validation-calibrated direct override 汇总：

```text
supported Macro-F1: 0.9485
raw Macro-F1: 0.7279
FAR: 1.28%
abnormal recall: 94.04%
direct recall: 93.39%
anomaly recall: 73.76%
```

其中 anomaly recall 只在有 anomaly support 的 fold 上统计。

逐 fold 结果如下：

| fold | outer 场景 | supported Macro-F1 | raw Macro-F1 | FAR | abnormal recall | anomaly recall | direct recall |
|---|---|---:|---:|---:|---:|---:|---:|
| fold_1 | st_L1 | 0.9906 | 0.6604 | 3.41% | 99.88% | n/a | 99.88% |
| fold_2 | st_L5 | 0.8670 | 0.8670 | 0.36% | 76.13% | 50.00% | 98.62% |
| fold_4 | st_L1+L5 | 0.9971 | 0.6648 | 0.50% | 100.00% | n/a | 100.00% |
| fold_5 | st_L1 | 0.9941 | 0.6627 | 0.31% | 99.00% | n/a | 99.00% |
| fold_6 | st_L5 | 0.8529 | 0.8529 | 2.85% | 91.73% | 97.52% | 65.36% |
| fold_7 | st_L1+L5 | 0.9890 | 0.6593 | 0.28% | 97.51% | n/a | 97.51% |

## 8. fold_3 的情况

fold_3 的 outer test 是：

```text
new_building / st_L5 / 2025.07.29.20.36_新主楼
```

这条记录从同一次 L5 攻击事件中途开始采集，不包含攻击前的正常起始段。

当前特征协议要求：

```text
前 30 个窗口必须是审核过的正常窗口
```

但 fold_3 的 test 流前 30 个窗口已经处于攻击区间内，因此不能作为正常基线。构建器按照 `initial_baseline_policy=exclude_stream` 排除了该 test stream，导致：

```text
fold_3 test windows = 0
```

因此 fold_3 当前不是模型失败，而是协议不适用。

后续可选处理方式：

1. 保守报告 6 个有效 fold，并明确 fold_3 是无初始正常基线例外；
2. 为 fold_3 单独设计无初始基线协议，例如使用原始聚合特征或 causal/rolling 特征；
3. 引入外部正常基线，但必须严格说明来源，避免跨 test 泄漏；
4. 将“需要攻击前正常观测”写成当前方法的部署前提。

目前建议先采用第 1 种，并把第 2 种作为后续鲁棒性实验。

## 9. 当前结论

当前结果支持以下判断：

1. 设备级异常检测已经较强：

```text
abnormal recall = 94.04%
FAR = 1.28%
```

2. 直接欺骗识别整体较好：

```text
direct recall = 93.39%
```

3. Watch L5 这类攻击关联异常能够被模型捕捉，但仍需补充和复核更多 anomaly 标注：

```text
anomaly recall = 73.76%
```

4. fold_2 的 anomaly recall 只有 50.00%，说明 new_building L5 Watch 关联异常标注或特征仍需要复核；

5. fold_6 的 direct recall 从原三分类的 47.06% 被 direct override 提升到 65.36%，但仍是主要薄弱点之一；

6. 该方案目前可以作为论文主线候选，但需要清楚说明 fold_3 的协议例外。

## 10. 复现命令

建议先设置 Python 路径：

```powershell
$PY = "H:\GNSS\program\Release_Package\Release_Package\venv\Scripts\python.exe"
```

单折完整流水：

```powershell
& $PY pipeline_total/43_run_static_response_state_fold.py `
  --fold fold_6 `
  --python-exe $PY `
  --overwrite-device-tensors
```

如果需要先构建 signal tensor，例如 fold_2：

```powershell
& $PY pipeline_total/20_build_static_timeblock_tensors.py `
  --outer-manifest output/protocols/static_time_block_outer_v2/fold_2/recording_split_manifest.csv `
  --block-manifest output/protocols/static_time_block_outer_v2/fold_2/epoch_split_manifest.csv `
  --output-dir output/tensors/static_timeblock_outer_v2/fold_2 `
  --time-steps 5 `
  --block-size 256
```

汇总有效 fold：

```powershell
$metrics = @(1,2,4,5,6,7) | ForEach-Object {
  "output/hierarchical_event_v1/static_response_state_v1/fold_$_/direct_override_mlp_h32_valcal_all/test_metrics_response_state_direct_override.json"
}

& $PY pipeline_total/40_summarize_response_state_metrics.py `
  --metrics $metrics `
  --output-csv output/hierarchical_event_v1/static_response_state_v1/static_response_state_direct_override_6fold_valid_summary.csv
```

## 11. 相关代码与文档

主要脚本：

```text
pipeline_total/36_build_device_attack_event_tensors.py
pipeline_total/37_train_device_attack_event.py
pipeline_total/40_summarize_response_state_metrics.py
pipeline_total/42_eval_response_state_direct_override.py
pipeline_total/43_run_static_response_state_fold.py
```

主要文档：

```text
docs/device_response_intervals.csv
docs/static_response_state_experiment.md
docs/response_state_team_sync_zh.md
docs/response_state_full_plan_zh.md
```

## 12. 论文叙事建议

论文中可以将该方案描述为对传统 spoofing 二分类标签的细化：

```text
由于不同接收设备的频段能力和芯片链路不同，同一攻击事件可能诱发不同响应。
因此，攻击检测不应只判断全局 spoof / normal，而应区分设备级 direct spoof 与 attack-associated anomaly。
```

特别是 L5-only 攻击下的 Watch 现象可以作为有价值的观察：

```text
Watch 不具备 L5 观测能力，因此不会表现为 L5 direct spoof；
但 L5 攻击可能通过跨频段链路或接收机内部机制影响 L1 搜星/跟踪，使 L1 C/N0 呈现压制异常。
```

这使本文方案具有两个贡献点：

1. 发现并解释了跨设备、跨频段的攻击关联异常；
2. 提出设备响应状态标签和 direct expert override，使模型既能发现异常，又能区分直接欺骗与关联异常。

## 13. 论文层面的推进与贡献

从论文角度看，这个方案的意义不只是“换了一组标签”或“提高了一些指标”，而是把原问题从粗粒度二分类推进到了更符合真实接收机响应机制的设备级诊断问题。它可以支撑论文中的问题定义、现象发现、方法设计、实验评价和局限讨论。

### 13.1 对问题定义的推进

原始任务通常被表述为：

```text
给定 GNSS 观测窗口，判断当前是否处于 spoofing attack。
```

这种定义隐含一个假设：

```text
同一攻击事件中的所有设备、所有频段都应呈现同一种 attack 响应。
```

但当前数据说明这个假设并不总成立。不同设备的硬件能力、频段支持、芯片链路和前端处理机制不同，同一攻击事件可能诱发不同响应。例如：

- 双频手机在 L5-only 攻击中可能直接表现为 L5 C/N0 上升；
- 不支持 L5 的 Watch 不可能出现 L5 direct spoof；
- 但 Watch 的 L1 可能因跨频段链路或接收机内部机制受到压制，表现为攻击关联异常；
- 也可能存在处于全局攻击区间但该设备无明显响应的窗口。

因此，本文可以将研究问题从：

```text
global spoofing detection
```

推进为：

```text
device-level response state diagnosis under GNSS spoofing events
```

即不仅判断“有没有攻击”，还判断设备当前对攻击的响应状态。

### 13.2 对实验现象的贡献

本方案背后最有论文价值的观察是：

```text
在 L5-only 欺骗环境下，不支持 L5 的设备仍可能在 L1 上出现显著异常响应。
```

这个现象不同于传统直觉中的“被欺骗频段 C/N0 上升”。它更像是攻击事件诱发的跨频段关联异常，可能与以下因素有关：

- 接收机不同频段并非完全独立工作；
- 某些芯片可能存在 L5 搜索/跟踪结果引导 L1 搜索/跟踪的机制；
- L5 受欺骗后，接收机内部状态或前端处理可能间接影响 L1；
- spoofing 信号或其伴随干扰可能导致非目标频段表现出压制特征。

这一点可以作为论文中的实测发现：

```text
GNSS spoofing does not only produce direct target-band enhancement; it may also induce non-target-band suppression-like anomalies on heterogeneous receivers.
```

对中文论文可以表述为：

```text
GNSS 欺骗并不只表现为目标频段载噪比升高；在异构设备上，还可能诱发非目标频段的压制型异常。
```

这个发现能解释为什么原来的 L5 和 L1+L5 场景模型效果差：模型在粗标签下混合学习了“目标频段增强”和“非目标频段压制”两种方向相反的响应。

### 13.3 对标签体系的贡献

传统二分类标签：

```text
normal / attack
```

无法区分以下情况：

```text
直接被欺骗；
攻击关联异常；
无明显响应但处于攻击区间。
```

本方案提出设备响应状态标签：

```text
normal / no observable response
attack-associated anomaly
direct spoof
```

它的贡献在于把“全局攻击事件标签”转换为“设备可观测响应标签”。这使标签更接近模型实际能从 C/N0、AGC、频段统计中学习到的东西。

论文中可以强调：

1. 该标签体系避免将物理含义不同的响应强行合并为 attack；
2. 它兼容无 L5、无 AGC 等硬件能力不完整的设备；
3. 它保留了安全任务最关心的异常发现能力；
4. 它进一步区分异常是 direct spoof 还是 attack-associated anomaly。

这可以形成一个清晰的方法贡献：

```text
We introduce a device-level response-state labeling scheme that separates direct spoofing evidence from attack-associated but non-direct anomalies.
```

### 13.4 对模型设计的贡献

当前最终推荐的模型不是简单三分类，也不是硬树，而是：

```text
flat three-class base model + binary direct expert override
```

这个结构的论文意义在于：

- base model 负责学习三类响应状态的整体分布；
- direct expert 专门强化 direct spoof 的识别；
- override 阈值只在 validation 上校准，避免 test 泄漏；
- 阈值选择同时约束 FAR 和 abnormal recall，符合实际检测系统需求。

与硬树结构相比，该方案避免了第一阶段错误传播：

```text
硬树：normal/abnormal 判断错后，后续 anomaly/direct 无法修复；
override：base 给出整体判断，direct expert 只在直接欺骗证据足够强时覆盖。
```

因此，这个结构可以被写成轻量级分层决策方法，而不是复杂深度模型堆叠。它符合项目主题中的 lightweight 要求。

### 13.5 对评价体系的贡献

本方案也推进了评价方式。原来只看整体 accuracy、Macro-F1 或 attack recall 不够，因为这些指标无法回答：

```text
是否发现了异常？
是否识别了直接欺骗？
是否把攻击关联异常错当成正常？
是否在正常窗口产生过多误报？
```

因此当前方案引入并同时报告：

- `abnormal recall`：衡量是否发现异常；
- `anomaly recall`：衡量是否识别攻击关联异常；
- `direct recall`：衡量是否识别直接欺骗；
- `FAR`：衡量正常误报率；
- `supported Macro-F1`：避免 test 中不存在类别时 raw Macro-F1 被人为压低。

这套指标更适合论文的原因是：它把检测任务拆成了实际系统关心的几个问题。

如果用于安全检测部署，最重要的是：

```text
低 FAR + 高 abnormal recall
```

如果用于欺骗类型解释，进一步需要：

```text
较高 direct recall 和 anomaly recall
```

因此当前结果可以更有层次地报告，而不是只说“模型效果提高了”。

### 13.6 对当前实验结果的论文支撑

在 6 个有效静态 fold 上，当前 validation-calibrated direct override 的汇总结果为：

```text
supported Macro-F1: 0.9485
FAR: 1.28%
abnormal recall: 94.04%
direct recall: 93.39%
anomaly recall: 73.76%
```

这些结果可以支撑三层结论：

第一，作为异常检测器：

```text
abnormal recall = 94.04%, FAR = 1.28%
```

说明该方法已经能够较可靠地区分正常与异常响应。

第二，作为直接欺骗识别器：

```text
direct recall = 93.39%
```

说明 direct expert override 能较好恢复目标频段直接欺骗证据。

第三，作为攻击关联异常识别器：

```text
anomaly recall = 73.76%
```

说明 Watch L5 这类非直接异常已经能被捕捉，但仍是后续需要增强标注和复核的部分。

因此论文中不建议只写“总体 Macro-F1 达到多少”，而应写成：

```text
The proposed response-state formulation achieves high abnormal-event detection performance while preserving the ability to distinguish direct spoofing from attack-associated anomalies.
```

中文可以写成：

```text
所提出的设备响应状态建模方法在保持较低误报率的同时，实现了较高的异常发现率，并进一步具备区分直接欺骗与攻击关联异常的能力。
```

### 13.7 对论文创新点的推荐表述

建议论文贡献可以整理成三点：

1. **异构设备下的跨频段异常发现**  
   本文观察到，在 L5-only 欺骗场景中，不支持 L5 的 Watch 设备仍可能在 L1 上出现明显压制型异常，说明 GNSS 欺骗在异构接收机上可能诱发非目标频段异常响应。

2. **设备级响应状态标签体系**  
   本文将传统 `normal/attack` 二分类扩展为 `normal / attack-associated anomaly / direct spoof` 三类设备响应状态，使标签语义更贴近设备实际可观测行为，缓解了 L5 和 L1+L5 场景中正负响应混杂导致的泛化问题。

3. **轻量级分层诊断模型与校准策略**  
   本文提出 `three-class base model + direct expert override` 的轻量级诊断框架，并在 validation 集上以 FAR 和 abnormal recall 为约束自动校准阈值，实现了低误报、高异常召回和较好的直接欺骗识别。

如果需要更强调工程实用性，可以加第四点：

4. **面向真实手机异构性的评估协议**  
   本文针对不同设备频段能力和传感器可用性差异，采用设备级聚合、初始正常基线差分和 supported Macro-F1 等评价方式，使实验更符合真实移动终端 GNSS 欺骗检测场景。

### 13.8 写论文时需要避免的过度声明

当前结果已经比较有说服力，但写论文时建议避免以下过度表述：

- 不要说“所有 L5 欺骗都会压制 L1”，应说“在部分异构设备上观察到”；
- 不要说“已经完全解决 L5 欺骗检测”，应说“显著缓解了 L5-only 场景下粗标签导致的泛化问题”；
- 不要把 fold_3 当作模型失败，应明确其不满足初始正常基线协议；
- 不要只报告 raw Macro-F1，因为不存在 anomaly 类的 fold 会被不公平压低；
- 不要把 attack-associated anomaly 直接等同于 direct spoof，它们在物理含义上不同。

推荐使用更稳妥的论文措辞：

```text
Our findings suggest that response-state modeling is a more appropriate formulation for heterogeneous mobile GNSS receivers than conventional binary attack labeling.
```

中文可以写成：

```text
实验结果表明，相比传统攻击二分类标签，设备响应状态建模更适合描述异构移动终端在 GNSS 欺骗事件中的复杂响应。
```

### 13.9 当前方案在论文中的位置

建议将该方案放在论文主线中，而不是作为附加消融实验。

可采用如下结构：

```text
Introduction:
  提出异构设备在欺骗事件中响应不一致的问题。

Observation / Motivation:
  展示 L5-only 场景下 Watch L1 压制异常。

Method:
  定义设备响应状态标签，提出 base + direct expert override。

Experiments:
  报告 supported Macro-F1、FAR、abnormal recall、direct recall、anomaly recall。

Discussion:
  解释 fold_3 无初始正常基线限制，以及 anomaly 标注仍需增强。
```

这样写的好处是：论文的贡献不是单纯追求一个更高分数，而是从真实现象出发，重新定义问题，再用一个轻量方案验证这个定义的有效性。

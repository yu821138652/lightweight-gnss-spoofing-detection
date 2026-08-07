# 场景分支交接说明（2026-08-08）

本文是当前场景分支的交接入口。它把原始四分类主分支、单频 L1 路由补充、冻结工件、与响应分支的接口以及结果边界放在同一处说明。

本文中的“冻结”表示：参数、特征、数据协议和本地产物已经固定，后续实验不得覆盖这些目录。冻结结果仍属于开发性多折结果，因为 outer-test 已在探索过程中被查看，不能包装为全新未读的盲测结果。

## 1. 先看结论

当前场景分支由两条互补路线组成：

| 路线 | 可回答的问题 | 输入窗口 | 当前状态 |
|---|---|---:|---|
| 四分类主分支 | 当前场景是 `normal / L1 / L5 / L1+L5` 中哪一种 | 同时有 L1、L5 的窗口 | 已冻结 |
| 单频 L1 路由专家 | 只有 L1 时，L1 是否是被攻击的目标频段 | 只有 L1 的窗口 | 已冻结 |

只有 L5 的窗口目前没有可用的正式专家。原因不是代码缺失，而是已有 train/validation 中无法同时提供可靠的二分类支持。

因此，当前系统不能在单频窗口上恢复完整四分类信息：

```text
只有 L1 -> 可以判断“L1 是否为目标”，不能判断 L5 是否同时被攻击
只有 L5 -> 当前没有正式专家，不能判断 L5 是否为目标
双频   -> 使用四分类主分支
```

四分类主分支的混合静态+动态冻结结果为：

```text
usable endpoints: 43672
Accuracy:         0.957524
Macro-F1:         0.924523
```

单频路由的 L1 目标二分类冻结结果为：

```text
all outer-test endpoints: 74408
supported endpoints:      74276
unsupported L5-only:      132
routing coverage:         99.8226%
Accuracy:                 98.3023%
Macro-F1:                 97.2889%
L1 positive recall:       95.8073%
```

## 2. 任务语义

场景分支的判别单元是：

```text
一个设备在一个时间窗口末端的频段场景
```

它不是逐 `signal_id` 的“这颗卫星是否被欺骗”，也不是设备响应分支的 `normal / anomaly / direct` 三分类。

场景类别定义为：

| 类别 | 含义 |
|---:|---|
| `0` | `normal`，末端时刻不在 reviewed 攻击区间 |
| `1` | `L1`，该 Session 的 reviewed 场景为 L1 欺骗 |
| `2` | `L5`，该 Session 的 reviewed 场景为 L5 欺骗 |
| `3` | `L1+L5`，该 Session 的 reviewed 场景为双频欺骗 |

场景标签来自 Session 的 reviewed 攻击区间和 `Scenario` 配置。它描述的是“当前全局攻击场景”，不是某个设备是否真正产生了 direct 响应。因此同一时刻不同设备共享同一个场景真值，但它们的设备响应真值可以不同。

## 3. 数据划分与防泄漏规则

### 3.1 四分类冻结协议

四分类主分支冻结快照使用：

```text
mixed_timeblock_outer_cv4_w5_v2
outer folds: 4
window length: 5 epochs
```

该协议将完整采集 Session/采集事件作为 outer 分组单位。同一采集事件不会跨 outer train/test。每个 outer fold 的 development 数据再按连续时间块拆成 train 和 validation。

W5 窗口不能跨越以下边界：

- train/validation/test 边界；
- guard 区间；
- segment 边界；
- 接收机时间间隔或断流边界。

scaler 和正常参考只使用该 fold 的 train 数据。checkpoint 按 validation Macro-F1 选择，之后才能读取 outer test。

### 3.2 单频路由协议

单频 L1 路由复用冻结的四分类 mixed 四折协议。双频窗口继续使用父四分类模型；只有 L1 的窗口进入 L1 专家；只有 L5 的窗口保留在 routing audit 中，但不进入 Accuracy/F1 的模型评分。

### 3.3 结果边界

当前 outer-test 已被多次用于探索、错误分析和模型选择。因此：

- 冻结结果是可复现的开发性多折结果；
- 不能把它描述为一次完全独立的最终测试；
- 后续如有新 Session，应优先作为确认集，而不是继续混入当前冻结结果。

## 4. 四分类主分支

### 4.1 处理流程

主分支的代码入口为：

| 脚本 | 作用 |
|---|---|
| `pipeline_total/45_build_band_mean_window_tensors.py` | 按设备和频段构造 band-mean 窗口张量 |
| `pipeline_total/46_train_band_mean_multiclass.py` | 训练四分类 TCN |
| `pipeline_total/47_aggregate_band_mean_cv.py` | 逐 fold 训练、test-only 推理和跨 fold 汇总 |
| `pipeline_total/62_freeze_scene_branch_baseline.py` | 创建或验证冻结快照 |

原始 signal 行先按设备和时间聚合，再按 L1/L5 分开求每个时刻的频段统计。连续 5 个满足同一 split、同一 segment 且无接收间隔的时刻组成一个窗口，窗口末端时刻决定场景标签。

### 4.2 输入特征

builder 原始上可以构造 11 维：

```text
L1: C/N0, AGC, 接收机时间不确定度, 伪距率不确定度
L5: C/N0, AGC, 接收机时间不确定度, 伪距率不确定度
L1 C/N0 - L5 C/N0
L1Present
L5Present
```

冻结主分支只保留 4 维：

```text
L1_Cn0DbHz
L5_Cn0DbHz
L1Present
L5Present
```

被删除的特征为：

```text
AgcDb
ReceivedSvTimeUncertaintyNanos
PseudorangeRateUncertaintyMetersPerSecond
Cn0DbHzL1MinusL5
```

C/N0 在进入模型前使用 train-only 的正常频段参考做 residual 化，再用 train-only scaler 标准化。冻结 metadata 和 normal reference 文件记录了每个 fold 的实际参考来源、设备覆盖和 fallback 情况，不能用其他 fold 的 scaler/reference 替代。

### 4.3 为什么删除这些特征

跨录制消融得到的主要结论是：

- C/N0 是最稳定、最能跨 Session 迁移的场景证据；
- AGC 在不同芯片上量纲和离散化方式不一致，RedMi K60 的 AGC 甚至近似只有少量离散值；
- 接收机时间不确定度和伪距率不确定度也更接近设备实现指纹；
- 显式 L1-L5 差值是两个 C/N0 的线性组合，加入后没有提供稳定增益。

这里的结论只适用于“band-mean 窗口级四分类场景任务”。不能把“AGC 是场景分支毒特征”直接推广到逐 signal 响应二分类。

### 4.4 模型结构

冻结模型为 `TCN32`：

```text
输入：[batch, 5, 4]
两层因果卷积：hidden=32，dilation=1/2
取窗口末端表示
四分类头：normal / L1 / L5 / L1+L5
dropout=0.1
```

每个 fold 最多训练 40 个 epoch，使用 seed `2026`，checkpoint 由 validation Macro-F1 选择。模型本身很小，符合轻量化定位。

### 4.5 single-band policy

窗口末端若只有 L1 或只有 L5，则设置 `single_band_mask=True`：

- 不进入四分类训练损失；
- 不进入四分类 Accuracy、Macro-F1 和混淆矩阵；
- 保留在张量和审计索引中，供单频路由使用。

这不是把单频窗口当作 `normal`，而是明确表示：原始四分类信息不足。

## 5. 单频 L1 路由专家

单频路由相关代码为：

| 脚本 | 作用 |
|---|---|
| `pipeline_total/70_build_single_band_scene_expert_tensors.py` | 构造 L1/L5 单频专家张量 |
| `pipeline_total/71_train_single_band_scene_experts.py` | 训练单频专家 |
| `pipeline_total/72_route_scene_single_band_experts.py` | 按频段能力路由输出 |
| `pipeline_total/73_freeze_scene_l1_binary_routed.py` | 冻结 L1 路由结果 |
| `pipeline_total/74_freeze_scene_route_aware_conditional_accuracy.py` | 冻结路由条件准确率结果 |

### 5.1 L1 专家任务

L1 专家不做四分类，而做一个可由单频输入回答的二分类：

```text
正类：reviewed 场景为 L1 或 L1+L5
负类：reviewed 场景为 normal 或 L5
```

该专家只使用只有 L1 的窗口训练和评估。它回答的是“L1 是否为攻击目标”，不声称知道 L5 是否同时受骗。

### 5.2 路由规则

```text
双频窗口 -> 原四分类 TCN
只有 L1 -> L1 专家
只有 L5 -> l5_unsupported
```

双频四分类输出投影到 L1 二分类时：

```text
预测 L1 或 L1+L5 -> L1 二分类正类
预测 normal 或 L5 -> L1 二分类负类
```

这里的“正确”必须按各路由对应的标签空间计算，不能把 L1 专家的二分类预测直接当成四分类预测。

### 5.3 L5-only 为什么没有专家

L5-only 窗口理论上可以训练一个“L5 是否为目标”的专家，但当前数据中对应 fold 的 train/validation 二分类支持不足，有的 fold 缺少某一类。若强行训练常数分类器，会把“没有可学习证据”伪装成模型结果，因此当前协议将其标为 `l5_unsupported`，只报告覆盖率，不纳入模型 Accuracy/F1。

## 6. 冻结工件位置

### 6.1 四分类主分支

```text
output/frozen/scene_branch_tcn32_mixed_cv4_w5_normalref_v1/
```

主要内容：

```text
README.md
freeze_manifest.json
protocol/
tensors/
training/
configuration/
code_snapshot/
```

其中：

- `protocol/` 保存四折 outer 和 inner 时间块清单；
- `tensors/` 保存每 fold 的 train/val/test 张量、scaler、normal reference 和 metadata；
- `training/` 保存 checkpoint、validation 指标、test predictions 和聚合指标；
- `code_snapshot/` 保存冻结时实际使用的源码；
- `freeze_manifest.json` 保存 SHA-256 和文件大小。

### 6.2 单频 L1 路由

```text
output/frozen/scene_branch_l1_binary_routed_mixed_cv4_w5_v1/
output/frozen/scene_branch_route_aware_conditional_accuracy_mixed_cv4_w5_v1/
```

前者保存 L1 二分类路由结果及完整 routing audit，后者保存路由条件准确率结果。两者都引用父四分类冻结快照，不应拆开替换其中某个 fold。

## 7. 与响应分支的接口

场景分支和响应分支是两个不同任务：

```text
场景分支：global scene = normal / L1 / L5 / L1+L5
响应分支：device response = normal / anomaly / direct
```

在联合响应链中，场景分支的作用是提供同一时刻的全局门控上下文：

1. 读取各设备的场景后验；
2. 按 `(fold, recording_id, TOW)` 跨设备平均概率；
3. 得到 `global_scene` 和 `global_scene_confidence`；
4. 只有 `global_scene=L5` 且置信度至少 `0.50` 时，才允许 Watch anomaly 或 L5 direct 自校准规则修正响应预测。

当前 `58_run_complete_scene_response_diagnosis_cv.py` 使用的是 `output/optimization/response_d4_matched_scene_fusion_20260806/scene_training/` 下与响应 fold 对齐的静态场景预测工件，而不是直接读取 mixed 四折冻结目录。这是因为联合响应实验当前只运行了静态 fold `2/3/6`。两者特征和 TCN 候选相同，但数据范围和工件目录不同，不能混称为同一份指标。

单频 L1 路由目前是独立冻结路线，尚未接入 `58` 的 Watch/L5 联合响应融合。后续若接入，必须重新定义：单频路由的输出如何转换成场景上下文，以及四分类和二分类指标如何分别报告。

## 8. 当前结果应该怎样引用

推荐引用方式：

```text
在 mixed_timeblock_outer_cv4_w5_v2 开发性四折协议下，
四维 C/N0 + presence 的 TCN32 场景分支在 43,672 个 usable endpoint
上取得 Macro-F1=0.9245、Accuracy=0.9575。
```

单频路线应单独写：

```text
对双频窗口沿用四分类场景判断，对只有 L1 的窗口使用 L1-target 二分类专家。
在 74,276 个 supported endpoints 上，路由 Accuracy=0.9830、Macro-F1=0.9729，
整体 routing coverage=99.8226%；132 个 L5-only endpoint 因缺少正式专家而不计入模型指标。
```

不要做以下比较：

- 不把 `0.9245` 四分类 Macro-F1 与逐 signal 二分类 Macro-F1 直接排序；
- 不把 L1 专家的 `0.9729` 当作四分类 Macro-F1；
- 不把 unsupported L5-only 当作模型正确或错误；
- 不把场景分支指标写成设备响应 `normal/anomaly/direct` 指标；
- 不把冻结开发结果写成未读独立 test 结果。

## 9. 已知限制与交接注意事项

1. 单频窗口的信息论上不足以恢复完整四分类。当前 L1 路由只能回答 L1 是否为目标，L5-only 仍无正式专家。
2. 每种静态攻击场景的 Session 数量较少；mixed 四折比 static 七折更适合作为当前主结果，但仍属于开发性结果。
3. 场景后验的全局平均依赖同一时刻存在多个设备。单设备孤立部署必须另行定义场景上下文来源。
4. 当前场景分支没有双频设备“另一频段关联异常”的正式标签。场景分类只能判断全局攻击场景，不能证明某个设备的非目标频段产生了 anomaly。
5. 场景分支只使用 C/N0 和 presence 的结论不能外推到响应分支；逐 signal 任务和 band-mean 四分类任务的最优特征不同。
6. 未来实验不得直接修改 `output/frozen/`。新特征、新窗口、新模型或新 split 必须使用新的 `output/optimization/` 或新冻结目录，并在文档中写明与本基线的差异。

## 10. 冻结完整性验证

从仓库根目录执行：

```powershell
python pipeline_total/62_freeze_scene_branch_baseline.py --verify
python pipeline_total/73_freeze_scene_l1_binary_routed.py --verify
python pipeline_total/74_freeze_scene_route_aware_conditional_accuracy.py --verify
```

验证脚本会检查冻结目录中的文件集合、文件大小、SHA-256、配置和冻结指标。若验证失败，不要手动替换单个 checkpoint 或 prediction CSV；应从对应协议重新构建一个新的实验目录。

## 11. 交接后的推荐入口

接手者应按以下顺序阅读：

1. 本文，确认任务和指标边界；
2. `output/frozen/scene_branch_tcn32_mixed_cv4_w5_normalref_v1/README.md`，确认四分类冻结基线；
3. `output/frozen/scene_branch_l1_binary_routed_mixed_cv4_w5_v1/README.md`，确认单频路由语义；
4. `pipeline_total/45–47`，理解主分支重建流程；
5. `pipeline_total/70–74`，理解单频专家和冻结验证流程；
6. `pipeline_total/54–58`，理解场景后验如何作为响应分支门控。

当前建议：把上述冻结工件作为只读基线，后续工作优先集中在动态单频覆盖、L5-only 专家可训练性和双频设备非目标频段关联异常标签设计，不再重新搜索已经冻结的四分类输入和 TCN 结构。

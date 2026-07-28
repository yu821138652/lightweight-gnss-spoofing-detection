# 静态+动态逐 `signal_id` 统一基线（2026-07-27）

## 结论先行

在模型、特征、窗口和标签语义不变，只把训练范围扩展为 reviewed 静态+动态 Session 后，统一模型可以稳定训练，但动态识别没有因数据量增加而出现质变：

- 24 个 Session 的 pooled Macro-F1 为 **0.8656**；该值由更长、表现更好的静态 Session 主导。
- 17 个动态 Session 的 pooled Macro-F1 为 **0.7693**，Session 等权为 **0.7024 +/- 0.1085**。
- `dy_L5` pooled Macro-F1 仅 **0.5647**，Precision **13.03%**、Recall **18.67%**；四个 Session 等权为 **0.5733 +/- 0.0427**。
- mixed 模型在同一批静态 test endpoint 上的 pooled Macro-F1 为 0.8782，高于纯静态对照的 0.8639；但 Session 等权从 0.8502 降至 0.8333，不能解释为稳定泛化提升。
- reviewed 区间内“所有频段均为正类”的标签敏感性实验也已完成：静态非目标频段可以学到一部分区间状态，但动态非目标频段合计 Recall 只有 **16.77%**；它没有解决动态识别，也不替代正式 target-band-only 基线。

因此，本轮建立的是当前标签口径下可审计的统一基线，并再次确认“加入更多动态样本”本身不足以解决动态 L5。它不支持继续围绕 overall 的零点零几做大规模扫参。

## 固定配置

| 项目 | 固定值 |
|---|---|
| 预测单位 | 每个 `signal_id` 的有效窗口末端 |
| 窗口 | `W=5`，严格因果 |
| raw 分支 | `Cn0DbHz`、`AgcDb`、接收机时间不确定度、伪距率不确定度、`FreqBand` |
| stats 分支 | compact11：C/N0 与 AGC 各 `Last/Mean/Std/Slope`，接收机时间不确定度 `Std`，两个 coverage ratio |
| 融合模型 | `TCN16 + stats MLP`，dropout=0.1，共 1,880 参数 |
| 优化 | AdamW，`lr=1e-3`，weight decay=1e-3，batch=256，最多 30 epochs，patience=6，seed=2026 |
| 决策 | 锁定 checkpoint 的二分类 argmax；test 不调阈值 |

raw 输入明确不使用中央 CSV 中已有的 `Cn0DbHz_dt` 和 `Cn0DbHz_std`，避免与逐 signal 的 W5 统计重复。

## 数据与可信协议

- 数据为 `LabelStatus=reviewed` 的 24 个完整 Session：7 静态、17 动态；中央 CSV 共 2,998,458 行、631,003 个正类行。
- 固定源清单为 [source_recording_manifest.csv](protocols/mixed_timeblock_outer_cv4_v2/source_recording_manifest.csv)。操场 `dy_L1/09.13-09.18` 的 reviewed 区间为空，因而是有意保留的全负 Session。
- outer 为 4-fold grouped CV，每折 6 个完整 Session test、18 个 development；每个 Session 恰好 test 一次。
- 新主楼两个重叠 `st_L5` 录制绑定为 `G08`，操场两个重叠 `dy_L_15` 录制绑定为 `G09`。同一采集事件不跨 train/test。
- 每折 test 覆盖静态/动态、两个环境和 L1/L5/L1+L5；development 保留六种 motion x band Scenario。
- 对当前 22 个不可拆采集组做精确集合覆盖审计后，正类行最大偏差的理论下限为 20.5833%。25% 上限是综合目标的 Pareto 膝点；最终各折偏差为 `+24.18% / -23.46% / +22.46% / -23.18%`。
- development 使用 64-epoch 连续块。选择 validation 时先匹配 20% 规模，再在 2 个百分点范围内优化类别覆盖和标签比例；W5 边界两侧各留 4 个 guard epoch。
- 四折 raw validation 占 development 的 `20.20%-20.78%`，guard 后为 `18.71%-19.21%`。每折六种 Scenario 均有正、负 validation 支持。
- scaler 仅用 train 拟合。四个 checkpoint 全部按 validation Macro-F1 锁定后，才首次依次读取 outer test。

此前 v1 试跑没有拆 Session，但 `G08/G09` 的同一采集事件跨 outer fold，且 256-epoch 分块使关键短动态 Session 无 validation。v1 结果只能用于发现问题，不能再称可信 mixed 基线。

## 四折结果

| Fold | Validation Macro-F1 | Test Macro-F1 | Precision | Recall | FAR |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.9026 | 0.8746 | 91.42% | 72.26% | 2.36% |
| 2 | 0.9091 | 0.8742 | 85.58% | 74.25% | 3.11% |
| 3 | 0.8828 | 0.8801 | 72.77% | 92.80% | 8.89% |
| 4 | 0.8966 | 0.8205 | 67.70% | 74.54% | 7.89% |
| 等权 mean +/- SD | **0.8978 +/- 0.0097** | **0.8624 +/- 0.0243** | 79.37% +/- 9.53% | 78.46% +/- 8.32% | 5.56% +/- 2.86% |

四折 test 混淆矩阵合计为 `TN/FP/FN/TP = 2,210,667 / 138,512 / 130,135 / 500,599`。

## Inner train/validation 划分消融：reviewed-state-stratified（2026-07-28）

为验证“在每个 development Session 中分别从 clean 和 attack 时段取约 80% train、20% validation”的设想，补做了一次严格 A/B 实验。中央 CSV、target-band-only 行级标签、outer 4-fold assignment、四折 test endpoint、W5、compact11、TCN16、优化参数和 seed 均保持不变；唯一改变的是 inner train/validation 的选取方式。本实验是协议消融，未替代上面的 strict-v2 基线。

具体规则与防边界泄漏约束如下：

- attack 时间状态由 `configs/preprocessing.yml` 中 reviewed Session 的闭区间 TOW 定义，与行级 `Label` 和 `FreqBand` 无关；它只用于分层，不改变模型学习的正式标签语义。
- clean 时段在每个 development Session 内约 80/20 划分；可安全拆分的长 attack run 也在本 Session 内约 80/20 划分。
- 短 attack run 若拆分后无法在 guard 外同时留下有效 W5，则作为不可拆 atom，在同 Scenario 的 development Sessions 之间分配给 train 或 validation。因此，并非每个短 Session 的 attack 时段都强制同时进入两边。
- 同一 canonical epoch 的全部设备、卫星和频段只能属于同一 split；每个 train/validation 边界两侧各留 4 个 guard epoch。

协议审计结果：strict-v2 与新协议的 `fold_assignment.csv` 及四折 `recording_split_manifest.csv` 逐字节一致；72/72 个 development Session×fold 的 clean train/validation 均保留有效 W5，clean raw validation 均值为 20.10%，范围为 17.92%-22.22%；39 个可拆长 attack run 与 33 个短 attack atom 全部通过窗口约束；每个 Scenario×fold 的 attack train/validation 均有有效 W5。旧 `recording-local --strict-validation` 路径的四折关键文件哈希也保持不变。

第一次生成的 `state_stratified_v1` 在训练后复现检查中发现：Fold 1 的 `dy_L1/09.18-09.22` clean validation 只有 `42/279=15.05%`。原因是长 attack run 的 prefix/suffix 方向先独立决定，随后限制了相邻 clean 可选区间。最终 v2 改为在 attack 规模误差并列时，联合选择长 attack 方向与可行 clean 区间，将该 Session 修正为 `56/279=20.07%`；协议元数据同时写入生成脚本 SHA256，且与当前源码一致。下表只报告修正后的 v2；v1 的训练结果不再作为本实验结论。

| 指标 | strict-v2 | state-stratified | 变化 |
|---|---:|---:|---:|
| Validation fold 等权 Macro-F1 | 0.8978 +/- 0.0097 | **0.9295 +/- 0.0141** | +0.0317 |
| Test fold 等权 Macro-F1 | **0.8624 +/- 0.0243** | 0.8605 +/- 0.0275 | -0.0018 |
| Overall pooled Macro-F1 | **0.8656** | 0.8637 | -0.0019 |
| Static pooled Macro-F1 | **0.8782** | 0.8756 | -0.0027 |
| Dynamic pooled Macro-F1 | **0.7693** | 0.7668 | -0.0025 |
| `dy_L5` pooled Macro-F1 | 0.5647 | **0.5817** | +0.0170 |
| Overall Session 等权 Macro-F1 | 0.7405 +/- 0.1210 | **0.7429 +/- 0.1157** | +0.0023 |
| Static Session 等权 Macro-F1 | 0.8333 +/- 0.0974 | **0.8337 +/- 0.0936** | +0.0004 |
| Dynamic Session 等权 Macro-F1 | 0.7024 +/- 0.1085 | **0.7055 +/- 0.1025** | +0.0031 |
| `dy_L5` Session 等权 Macro-F1 | 0.5733 +/- 0.0427 | **0.5897 +/- 0.0384** | +0.0164 |

新方案的 `dy_L5` Precision / Recall / FAR 为 `16.43% / 21.20% / 2.51%`，旧基线为 `13.03% / 18.67% / 2.90%`。四个 `dy_L5` Session 均改善，但单 Session Macro-F1 只增加 0.0036-0.0305，最低 Session 仍仅为 0.5422。放到全部 24 个 test Session 后，13 个改善、11 个退化；`dy_L1` pooled Macro-F1 从 0.7977 降至 0.7895，抵消了 `dy_L5` 和 `dy_L_15` 的局部收益。

新的 validation 明显变高而 outer test 基本持平，说明它更容易评估“同一 Session 分布内”的拟合情况，却没有增强对完整未见 Session 的选择能力。W5 guard 已排除相邻窗口跨边界重叠，但 train/validation 仍有意共享 Session、设备和环境分布；这种 domain overlap 不能当作跨 Session 泛化证据。`dy_L5` 的约 0.017 提升仍不属于质变，Recall 也只有 21.20%，动态整体还略有下降。因此继续保留 strict-v2 为统一 mixed 基线，state-stratified v2 只作为可复现的诊断协议。

四个 checkpoint 均在读取任何 test 前锁定：最晚 checkpoint 时间为 02:44:15.697，最早 test 结果时间为 02:45:04.385，间隔约 48.7 秒。本地产物为：

```text
output/protocols/mixed_timeblock_outer_cv4_w5_state_stratified_v2/
output/tensors/mixed_timeblock_outer_cv4_w5_state_stratified_v2/
output/training/mixed_timeblock_outer_cv4_w5_compact11_tcn16_d10_state_stratified_v2/
```

协议入口是在脚本 19 的原 mixed 命令上替换输出目录，并增加：

```text
--validation-mode reviewed-state-stratified
--label-config configs/preprocessing.yml
```

## 标签语义敏感性：reviewed 区间内全部信号为正类（2026-07-28）

本实验检验另一种任务定义：不再询问“这个 `signal_id` 是否属于被直接欺骗的目标频段”，而只询问“这个 endpoint 是否位于人工 reviewed 的欺骗时间区间”。具体策略为 `reviewed_interval_all_positive`：对每个 reviewed Session 严格令 `Label=1 iff TOW` 位于任一欺骗闭区间，不区分 L1/L5、设备、卫星或 `signal_id`；空区间 Session 保持全负。策略只在脚本 20 构建张量时生效，不修改中央 CSV 或 `preprocessing.yml`。

为避免把 inner split 与标签变化混在一起，本实验固定使用修正后的 state-stratified v2 协议、相同 outer test Session、W5、compact11、TCN16、优化参数和 seed。新旧四折的窗口、manifest、trace index、特征名和 train-only scaler 逐项一致，唯一实验变量是 train/validation/test 的标签语义。24 个 reviewed recording 中有 23 个非空区间、1 个显式空区间，共 24 段；中央行级正类由 631,003 增至 829,389（新增 198,386），进入 W5 endpoint 后正类由 630,734 增至 828,913（新增 198,179）。

新标签的四折最佳 Validation epoch 为 `27 / 20 / 30 / 28`。四个 checkpoint 全部锁定且记录 SHA-256 后才运行 `test-only`，测试与后续分组评估均未改变 checkpoint。

下表只是两个**不同预测任务**的标签敏感性对照，不能把差值表述成同一任务上的模型提升：

| 指标 | target-band-only | interval-all-positive | 变化 |
|---|---:|---:|---:|
| Validation fold 等权 Macro-F1 | 0.9295 +/- 0.0141 | 0.9229 +/- 0.0133 | -0.0066 |
| Test fold 等权 Macro-F1 | 0.8605 +/- 0.0275 | 0.8616 +/- 0.0255 | +0.0011 |
| Overall pooled Macro-F1 | 0.8637 | 0.8668 | +0.0031 |
| Static pooled Macro-F1 | 0.8756 | 0.8851 | +0.0096 |
| Dynamic pooled Macro-F1 | 0.7668 | 0.7351 | **-0.0317** |
| `dy_L1` pooled Macro-F1 | 0.7895 | 0.7464 | **-0.0431** |
| `dy_L5` pooled Macro-F1 | 0.5817 | 0.6394 | +0.0578 |
| Overall Session 等权 Macro-F1 | 0.7429 +/- 0.1157 | 0.7524 +/- 0.1128 | +0.0095 |
| Static Session 等权 Macro-F1 | 0.8337 +/- 0.0936 | 0.8717 +/- 0.0633 | +0.0380 |
| Dynamic Session 等权 Macro-F1 | 0.7055 +/- 0.1025 | 0.7033 +/- 0.0897 | -0.0022 |
| `dy_L5` Session 等权 Macro-F1 | 0.5897 +/- 0.0384 | 0.6382 +/- 0.0048 | +0.0485 |

Overall 的变化掩盖了任务内部的明显差异。新任务的 pooled Precision / Recall / FAR 为 `87.68% / 73.88% / 4.00%`，旧任务为 `78.01% / 79.09% / 5.98%`。Dynamic 的 Recall 则从 53.00% 降至 43.23%。尤其是 `dy_L5`：正类支持从 4,836 增至 21,975，Precision 从 16.43% 变为 62.93%，但 Recall 只从 21.20% 变为 22.31%；Session 等权 Recall 反而从 24.05% 降至 22.18%。因此 `dy_L5` Macro-F1 的上升主要来自标签定义改变后，大量原先算作 L1 假正的区间内预测被重新计为真阳性，而不是动态区间检出率发生质变。

对四类本次新增为正类的非目标频段单独统计：

| 场景中的新增正类频段 | Test 正类 endpoint | Macro-F1 | Precision | Recall | FAR |
|---|---:|---:|---:|---:|---:|
| `dy_L1` 的 L5 | 15,880 | 0.5446 | 61.57% | **11.20%** | 1.60% |
| `dy_L5` 的 L1 | 17,139 | 0.6361 | 61.20% | **21.93%** | 1.59% |
| `st_L1` 的 L5 | 26,228 | 0.8483 | 81.69% | 70.40% | 4.23% |
| `st_L5` 的 L1 | 138,932 | 0.7907 | 78.78% | 60.30% | 6.17% |

合并后，静态新增非目标频段的 Recall 为 **61.90%**，动态只有 **16.77%**。这说明“同一攻击区间可能影响非目标频段观测”在静态数据中存在一定可学习信号，但动态情况下这些信号依旧非常弱；把它们统一改成正类不会自动让现有逐 signal 特征获得时间区间检测能力。

逐完整 Session 配对也不支持稳定改善：24 个 Session 中 12 个 Macro-F1 上升、12 个下降；实际增加正类的 15 个 Session 中 7 个上升、8 个下降。六个含攻击正类的 `dy_L1` Session 全部下降；四个 `dy_L5` Session 中三个上升、一个下降；三个 `st_L5` Session 均上升。结论是：该语义可作为“攻击时间上下文/事件级异常”的独立任务继续研究，但不能替代当前卫星/目标频段直接欺骗标签，也没有推翻“动态特征可分性不足”的主要判断。

本地产物：

```text
output/tensors/mixed_timeblock_outer_cv4_w5_state_stratified_interval_all_positive_v1/
output/training/mixed_timeblock_outer_cv4_w5_compact11_tcn16_d10_state_stratified_interval_all_positive_v1/
```

## 静态与动态

| Test 子集 | endpoint 数 | Macro-F1 | Precision | Recall | FAR | Session 等权 Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 2,979,913 | **0.8656** | 78.33% | 79.37% | 5.90% | 0.7405 +/- 0.1210 |
| Static | 2,077,565 | **0.8782** | 80.73% | 83.46% | 6.99% | 0.8333 +/- 0.0974 |
| Dynamic | 902,348 | **0.7693** | 61.82% | 55.11% | 3.82% | 0.7024 +/- 0.1085 |

动态 pooled 只占总 endpoint 的 30.28%，动态正类只占全部正类的 14.45%，因此 overall 会明显稀释动态问题。只统计 16 个含正类动态 Session 时，Session 等权 Macro-F1 为 **0.7151 +/- 0.0988**、Recall 为 48.25%，判断不变。

全负 `dy_L1/09.13-09.18` 的 Macro-F1 为 0.4986，但实际为 `TN/FP=64,673/373`、FAR 0.57%；它没有可计算的正类 Recall，不能解释成攻击漏检。

## 分场景结果

| Scenario | endpoint 数 | 正类数 | Macro-F1 | Precision | Recall | FAR | Session 等权 Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `st_L1` | 752,236 | 229,987 | 0.9222 | 86.22% | 92.78% | 6.53% | 0.9229 +/- 0.0020 |
| `st_L5` | 676,177 | 66,635 | 0.7382 | 42.96% | 75.46% | 10.95% | 0.7262 +/- 0.0396 |
| `st_L_15` | 649,152 | 242,959 | 0.8914 | 96.57% | 76.84% | 1.63% | 0.9042 +/- 0.0249 |
| `dy_L1` | 383,126 | 60,866 | 0.7977 | 71.37% | 60.49% | 4.58% | 0.7402 +/- 0.1152 |
| `dy_L5` | 212,461 | 4,836 | **0.5647** | **13.03%** | **18.67%** | 2.90% | **0.5733 +/- 0.0427** |
| `dy_L_15` | 306,761 | 25,451 | 0.7392 | 55.03% | 49.18% | 3.64% | 0.7443 +/- 0.0462 |

四个 `dy_L5` Session 的 Macro-F1 分别为 0.5117、0.5682、0.5815、0.6318；Precision 为 4.67%-21.28%，Recall 为 6.88%-39.14%。失败不是由单个 Session 偶然造成。

按全部动态数据中的 L5-band endpoint 聚合，Pixel 6、Mate40、RedMi K60、MI8 的 Recall 分别为 21.83%、9.09%、29.01%、1.69%。不同设备程度不同，但没有一个设备已稳定解决动态 L5。

静态 L5 也仍明显弱于 L1/L1+L5：pooled Macro-F1 0.7382、FAR 10.95%。新主楼短 `st_L5/20.36` 的 Macro-F1 仅 0.6918、Recall 99.96%、FAR 13.09%，说明该难点表现为严重过报，而非统一方向的漏检。

## 与纯静态对照

下表使用同一批 7 个静态 Session、同样的 2,077,565 个 W5 endpoint；模型和特征不变。协议中的 development 构成和 outer fold 数不同，因此它不是“只增加动态样本”的严格单变量消融。

| 训练与协议 | Static pooled Macro-F1 | Precision | Recall | FAR | Static Session 等权 Macro-F1 |
|---|---:|---:|---:|---:|---:|
| 纯静态 7-fold 保留基线 | 0.8639 | 80.38% | 79.22% | 6.78% | **0.8502 +/- 0.0933** |
| mixed v2 的 static 子集 | **0.8782** | **80.73%** | **83.46%** | 6.99% | 0.8333 +/- 0.0974 |

逐 Session 并不一致。7 个静态 Session 中 5 个 Macro-F1 上升、2 个下降；新主楼短 `st_L5/20.36` 从 0.9118 降至 0.6918，抵消了其它 Session 的小幅收益。pooled 上升主要反映长 Session 权重，不能表述为静态泛化稳定提升。

## 当前判断

1. 统一静态+动态逐信号训练在工程和评估协议上已经跑通，v2 可作为后续 mixed 实验的固定对照。
2. 数据量增加没有带来动态 L5 的质变。当前瓶颈仍更像弱特征、标签语义和设备响应差异，而不是 TCN 容量不足。
3. validation 均值 0.8978、test fold 均值 0.8624 相差约 0.0354；完整未见采集事件的分布变化和折间组成差异均存在，不能仅凭差值称为传统样本级过拟合。
4. 下一步应优先做动态 L5 的数据、标签与可分性诊断，预先锁定判据；不值得继续为 overall 的零点零几扫参。

## 复现与本地产物

实现入口：

- `pipeline_total/38_generate_mixed_timeblock_protocol.py`：采集事件成组且带正类偏差上限的 outer assignment；
- `pipeline_total/19_generate_static_timeblock_protocol.py --strict-validation`：64-epoch validation 与 W5 guard；
- `pipeline_total/20_build_static_timeblock_tensors.py --data-scope mixed`：mixed raw/stats 张量；
- `pipeline_total/21_train_static_signal_fusion.py`：compact11 TCN16 训练与锁定后 test-only；
- `pipeline_total/39_evaluate_mixed_signal_groups.py`：motion、Scenario、Session 和 device x motion x band 指标；
- `pipeline_total/40_aggregate_mixed_signal_cv.py`：pooled、fold 等权和 Session 等权汇总。

本地生成物位于：

```text
output/protocols/mixed_timeblock_outer_cv4_v2/
output/protocols/mixed_timeblock_outer_cv4_w5_v2/
output/tensors/mixed_timeblock_outer_cv4_w5_v2/
output/training/mixed_timeblock_outer_cv4_w5_compact11_tcn16_d10_v2/
```

最终汇总文件为 `cv_test_pooled_group_metrics.csv`、`cv_test_fold_metrics.csv`、`cv_test_session_metrics.csv` 和 `cv_test_summary.json`。`output/` 被 Git 忽略；完整命令见 `pipeline_total/README.md`。

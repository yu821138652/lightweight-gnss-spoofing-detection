# 静态+动态逐 `signal_id` 统一基线（2026-07-27）

## 结论先行

在模型、特征、窗口和标签语义不变，只把训练范围扩展为 reviewed 静态+动态 Session 后，统一模型可以稳定训练，但动态识别没有因数据量增加而出现质变：

- 24 个 Session 的 pooled Macro-F1 为 **0.8656**；该值由更长、表现更好的静态 Session 主导。
- 17 个动态 Session 的 pooled Macro-F1 为 **0.7693**，Session 等权为 **0.7024 +/- 0.1085**。
- `dy_L5` pooled Macro-F1 仅 **0.5647**，Precision **13.03%**、Recall **18.67%**；四个 Session 等权为 **0.5733 +/- 0.0427**。
- mixed 模型在同一批静态 test endpoint 上的 pooled Macro-F1 为 0.8782，高于纯静态对照的 0.8639；但 Session 等权从 0.8502 降至 0.8333，不能解释为稳定泛化提升。

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

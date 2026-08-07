# 场景门控的设备响应联合诊断：历史开发链路

> 状态：历史开发性方案与结果记录。截至 2026-08-07，旧设备级 `normal/anomaly/direct` 三分类标签已被频段级标签体系取代；本文件第 1–6 节中的“完整系统”指标不能作为论文最终结果。当前可审计方案见 [场景条件化频段响应诊断：标签重构与六折覆盖审计](scene_conditioned_response_audit_20260807.md)。原始日志、张量、checkpoint 与预测文件均只保存在本地 `output/`，不纳入 Git。

## 1. 要解决的问题

逐卫星、按目标频段二分类的方案在异构消费级终端上存在两个典型失配。

1. **L5 攻击下的关联异常。** 直接受影响的 L5 信号数量有时较少，而更多 L1 观测会出现 C/N0 下降、可见星减少或跟踪受压制。若把所有变化都当成目标频段 direct，容易同时造成 L5 漏检和 L1 误报。两款 Watch 没有 L5；其中 Watch1 还没有 AGC，但在 L5 攻击时依然出现稳定的 L1 响应。
2. **设备间的 direct 响应不一致。** 固定的跨设备 L5 direct 分类器可以在某些手机上有效，却不能稳定迁移到 Pixel6 等设备；模型学习到的是设备响应模式，而不一定是“直接受骗”这一语义。

因此，最终输出不再只有“某条卫星信号是否受骗”，而是联合给出：

```text
攻击场景：normal / L1 / L5 / L1+L5
设备响应：normal / anomaly / direct
```

其中 `anomaly` 表示攻击相关、但不是本设备目标频段直接受骗的关联异常；例如 L5 场景下无 L5 能力 Watch 的 L1 压制响应。

标签复核表明，Watch1 的新主楼 `st_L5` 并非应删除的错误标签：攻击期间平均 C/N0 约下降 1.58 dB-Hz，跟踪信号数约从 11 降至 6，攻击结束后恢复。本文只将其解释为可重复观测到的跨频段/接收机关联异常，不对芯片内部因果机理作绝对断言。

## 2. 最终系统

```text
多设备 GNSS Raw
   ├─ 场景分支（全局频段聚合） ──> normal / L1 / L5 / L1+L5
   │                                │
   │                                └─ L5 且置信度 >= 0.50 时开放专项修复
   │
   └─ 设备响应 backbone（每设备） ──> normal / anomaly / direct
                                      │
                    ┌─────────────────┴──────────────────┐
                    ▼                                    ▼
             Watch 关联异常修复                   L5 direct 自校准备份
                    │                                    │
                    └──────────> 最终 normal/anomaly/direct ──> 联合诊断
```

### 2.1 场景分支：给出“发生了什么攻击”

- 输入：`L1_Cn0DbHz`、`L5_Cn0DbHz`、`L1Present`、`L5Present` 四维频段聚合特征。
- 时间窗口：`T=5`。
- 模型：TCN。
- 输出：`normal / L1 / L5 / L1+L5` 的场景后验；同一时刻对多设备预测取平均，形成全局场景上下文。
- 静态六折 pooled：Macro-F1 **0.9921**，Accuracy **0.9944**。

该分支仅回答攻击频段，不强行规定每一台设备必须出现哪一种响应。

### 2.2 响应 backbone：给出“设备表现为何种状态”

基础响应模型沿用已验证的轻量配置：`sparse_extreme` 聚合、初始 30 窗口基线差分、44 维 C/N0 extreme 特征、MLP-h16 三分类，并由 direct expert 覆盖直接受骗候选。其输出为 `normal / anomaly / direct`。全局 direct 覆盖阈值为 0.20。

它提供通用的设备级响应起点，但不单独承担所有罕见的单频 Watch 异常和跨设备 L5 direct 语义。

### 2.3 Watch 关联异常专家：修复 L5 场景的 L1 压制漏检

适用设备为 `Google_Pixel_Watch1` 与 `Google_Pixel_Watch2`，二者均没有 L5；Watch1 也没有 AGC。

- 特征：11 维 L1 C/N0 与可见/跟踪信号数的极值统计及其相对固定基线差分。
- 模型：MLP-h8，训练目标为 `anomaly_only`，即 direct 样本不参与该专家训练。
- 固定基线：先跳过每条流的前 300 个窗口，再以其后的 30 个窗口建立基线，避免起始不稳定段污染参考值。
- 门控：仅当全局场景为 `L5` 且置信度不低于 0.50、Watch 专家认为当前为压制异常、基础结果为 `normal` 时，执行 `normal -> anomaly`。

这个专家不把 Watch 判成 L5 direct；它只补回符合观察语义的关联异常。

### 2.4 L5 direct 自校准规则：修复跨设备 L5 direct 漏检

适用设备为具备 L5 的 `Google_Pixel6`、`HUAWEI_Mate40`、`RedMi_K60` 和 `XiaoMi_MI8`。

跨设备训练的 L5 direct MLP 曾作为备选，但在 Pixel6 上迁移不稳定，因此**不作为最终模块**。最终采用不依赖其它设备分布的每流自校准规则：

- 证据：该设备/源流的 `initial_baseline_delta_l5_cn0_last_q25`。
- 校准区间：第 30--60 个窗口；计算该段的低 10% 分位数作为该流下尾阈值。
- 门控：仅当全局场景为 `L5` 且置信度不低于 0.50，且当前证据低于该流自身阈值时，将 `normal/anomaly -> direct`。

这一步是“同设备自身参考”的 direct 证据补充，而非又训练一个可能携带设备捷径的跨设备分类器。

## 3. 静态六折结果

评估协议为 `static_time_block_outer_v2`，fold 为 `1/2/4/5/6/7`，总计 49,963 个设备窗口。结果路径：

```text
output/optimization/complete_scene_response_diagnosis_cv_v1/
  aggregate_complete_scene_response_metrics.json
```

### 3.1 基础响应与完整诊断对比

| 指标 | 基础响应 backbone | 完整场景门控联合诊断 |
|---|---:|---:|
| Accuracy | 96.57% | **99.07%** |
| Macro-F1 | 90.98% | **98.02%** |
| FAR | **0.851%** | 0.863% |
| abnormal recall | 93.73% | **99.05%** |
| anomaly recall | 76.35% | **98.76%** |
| direct recall | 94.68% | **98.98%** |

完整方案的 pooled 混淆矩阵（行是真实类别，列是预测类别；类别顺序 `normal/anomaly/direct`）为：

```text
              normal  anomaly  direct
normal        32039      202      77
anomaly          32     2702       2
direct          135       17   14757
```

FAR 从 0.851% 增至 0.863%，仅增加 0.012 个百分点；同时 anomaly recall 增加 22.40 个百分点，direct recall 增加 4.30 个百分点。

### 3.2 含 anomaly 样本的关键折

只有 fold_2 和 fold_6 含有 anomaly 真值样本，因此它们是检验关联异常诊断的关键折。

| Fold | Macro-F1 | FAR | anomaly recall | direct recall |
|---|---:|---:|---:|---:|
| fold_2 | **99.37%** | 0.356% | 98.80% | 99.72% |
| fold_6 | **98.00%** | 1.424% | 98.72% | 97.27% |

fold_1、4、5、7 的 anomaly support 为 0。三分类 Macro-F1 在这些折约为 0.66，是缺失类别的计算结果，**不能解释为该折的异常检测失败**；应同时报告每类 support、direct recall 与 pooled 结果。

### 3.3 设备 pooled 结果

| 设备 | 关键响应召回 | FAR |
|---|---:|---:|
| Google Pixel6 | direct 98.29% | 0.67% |
| Google Pixel Watch1 | anomaly 98.46%，direct 99.48% | 2.15% |
| Google Pixel Watch2 | anomaly 99.05%，direct 99.20% | 0.13% |
| HUAWEI Mate40 | direct 98.64% | 0.28% |
| RedMi K60 | direct 99.24% | 0.11% |
| XiaoMi MI8 | direct 98.83% | 1.23% |

由此可见，原先对 L5 场景最敏感的 Watch 关联异常和 Pixel6 L5 direct 漏检均已在完整链路中得到修复。Watch1 的 pooled FAR 仍为 2.15%，MI8 为 1.23%，属于后续应重点压低的设备级误报来源。

## 4. 复现顺序

以下命令在仓库根目录执行。默认解释器可以改成用户本地 venv 的 Python 路径。

### 4.1 先运行 Watch 专家六折链

```powershell
python pipeline_total/55_run_scene_gated_watch_anomaly_cv.py `
  --python-exe python
```

该脚本会构建 Watch 专用张量、训练/校准 MLP-h8，并输出场景门控后的 Watch 预测至：

```text
output/optimization/watch_suppression_warmup_cv_v1/
```

### 4.2 再汇总完整诊断

```powershell
python pipeline_total/58_run_complete_scene_response_diagnosis_cv.py `
  --python-exe python
```

它会先对每折执行 L5 自校准 direct 规则，再和 Watch 修复结果合并，最后生成：

```text
output/optimization/complete_scene_response_diagnosis_cv_v1/
  aggregate_complete_scene_response_metrics.json
  aggregate_complete_scene_response_predictions.csv
```

如已存在各折预测，只重新统计汇总，可使用：

```powershell
python pipeline_total/58_run_complete_scene_response_diagnosis_cv.py `
  --aggregate-only
```

## 5. 实现入口

| 脚本 | 作用 |
|---|---|
| `36_build_device_attack_event_tensors.py` | 设备响应张量；支持基线偏移和 Watch L1 特征集 |
| `37_train_device_attack_event.py` | 设备响应训练；支持 `anomaly_only` 与按场景限制 |
| `54_eval_scene_gated_watch_anomaly.py` | 对单折 Watch 专家加全局 L5 场景门控 |
| `55_run_scene_gated_watch_anomaly_cv.py` | Watch 专家的完整六折构建、训练、校准与评估 |
| `56_eval_scene_gated_l5_direct_expert.py` | 跨设备 L5 direct MLP 的对照/负结果评估入口，不是最终模块 |
| `57_eval_scene_gated_l5_self_calibrated.py` | 每流 L5 下尾自校准 direct 规则 |
| `58_run_complete_scene_response_diagnosis_cv.py` | 合并 Watch 与 L5 direct 修复，输出完整六折结果 |

`51`--`53` 为此前的场景条件响应探索工具，可用于对照，但不属于上述最终链路。

## 6. 适用范围与限制

- 当前结论仅针对静态六折开发性评估。2026-08-06 在 7 个静态与 17 个动态 Session 的 mixed 四折协议上进行的场景分支扩展表明，四维 C/N0 模型的纯动态子集 Macro-F1 只有 0.5412，L5 与 L1+L5 recall 分别为 25.89% 和 10.99%。追加因果 C/N0 斜率、波动和有效观测数后，动态 Macro-F1 升至 0.5819，L5/L1+L5 recall 升至 34.23%/23.63%，但总体和静态性能均略有下降，仍未达到可作为动态场景门控的标准。因此本文档的静态完整诊断指标不得外推到动态环境。
- 动态设备响应的 `normal/anomaly/direct` 区间尚未完成人工审查；在此之前不能把动态攻击区间机械地映射为每台设备的 `direct` 或 `anomaly`，也不能建立可信的端到端动态联合诊断指标。
- 为支持下一轮“更长因果上下文”对照，`47_aggregate_band_mean_cv.py` 已支持 `--time-steps`；T=15 的动态 C/N0 试验尚未运行，不应写成已有结果。
- TTD（从攻击开始到稳定告警的时延）以及端到端推理耗时/参数量仍需作为后续独立实验完成。
- 场景门控依赖同一时刻多设备的全局场景后验，适配多设备协同部署；单设备孤立部署需要另行定义场景上下文获取方式。
- 频段级场景识别适用于广播式、多星同频攻击，不能定位具体 PRN；单星或选择性 PRN 欺骗是后续卫星级精细化检测方向。
- L5 自校准假设启动后的 30--60 窗口包含可用参考。若设备启动时攻击已存在，应保守地禁用该规则，或未来加入基线稳定性/攻击先验检查。
- 本方案将 L5 场景下 Watch 的 L1 变化标记为 `anomaly`，是基于数据观测的诊断语义；其具体捕获、跟踪或资源竞争机理仍需要芯片级证据进一步验证。

## 7. 2026-08-07 标签体系修订：旧完整链路不再作为最终响应结果

后续人工复核确认：新主楼和操场的 `st_L5` 攻击区间内，全部具备 L1 观测能力的设备均出现 L1 关联异常（包含不具备 L5 的两只 Watch）；操场 `st_L1` 攻击区间内，`Google_Pixel6` 与 `XiaoMi_MI8` 出现 L5 关联异常，其余设备 L5 保持 normal。新主楼 `st_L1` 及两个 `st_L1+L5` 场景未观察到关联异常。

这暴露了本文档中旧设备级三分类标签的结构性限制：双频设备可以在同一时刻具有“目标 L5 direct + 非目标 L1 associated anomaly”，但旧 `normal/anomaly/direct` 互斥标签会将 direct 覆盖 anomaly，未显式标注的关联异常还会默认成为 normal。因此，第 3 节的旧完整链路结果只能作为旧标签体系下的开发性对照，不应再作为论文中跨频段关联异常联合诊断的最终主结果。

新的正式路线改为“场景分支 + 场景条件化频段响应二分类”：场景先确定目标频段；目标频段由此记为 direct，非目标且可观测频段才由响应分支判断 `normal / associated_anomaly`。新真值见 `docs/device_band_association_intervals.csv`，完整语义与重构要求见 `docs/scene_conditioned_band_response_label_spec_zh.md`。场景分支的攻击识别结果不受此修订影响。

# 项目交接状态（2026-07-26）

本文是当前工作区的唯一状态入口。它说明已经确认的数据事实、最近的探索结论、结果边界、可复现入口和下一步建议。历史 P0–P5 结果只用于追溯，不代表当前已经锁定的主线。

## 当前状态快照（2026-07-26）

> 本节是当前静态逐 `signal_id` 路线的权威结论；文中后续保留的
> `static_timeblock_outer_v2`、10 维联合消融等内容是其形成过程中的
> 基线或诊断记录。若数值冲突，以本节为准。

### 任务、数据和协议

- 当前可比较的主实验只使用 `LabelStatus=reviewed` 的**静态** `st_*` 数据；动态数据没有进入这轮训练。因此操场动态 L15 的标签区间修订 `[260990, 261020]` 是当前数据政策的一部分，但不会直接改变下述 7-fold 静态指标。
- 真正独立的测试单位只有 7 个完整静态 Session：新主楼 L1×1、L5×2、L1+L5×1；操场 L1、L5、L1+L5 各 1。每折以一个完整 Session 作 outer test，其余 6 个 Session 按连续 UTC 时间块作 inner train/validation。
- 每个样本是一个 `signal_id` 的 W5 因果窗口，标签为窗口末端历元；窗口不会跨 train/validation 边界、4 个历元 guard 或 2 秒以上的来源断档。连续特征只用训练块拟合、按设备标准化。
- 当前 7 个 outer test 都已被用于模型、特征和错误诊断决策。协议没有训练泄漏，但结果只能称为**迭代式交叉验证开发结果**，不再是新的独立盲测。

### 当前保留基线：`compact11 + TCN16 + dropout=0.1`

这是目前唯一在完整 7-fold 中同时取得最高 pooled 分数、最低参数量之一，并在操场 L5 不比其它完整候选更差的配置；它是**保留基线，不是最终模型**。

- 产物：`output/training/static_timeblock_outer_v2_explore_compact11_tcn16_d10/`。
- Raw TCN 输入 5 维：`Cn0DbHz`、`AgcDb`、`ReceivedSvTimeUncertaintyNanos`、`PseudorangeRateUncertaintyMetersPerSecond`、`FreqBand`。CSV 中已有的 `Cn0DbHz_dt`、`Cn0DbHz_std` 不进入训练。
- stats MLP 输入 11 维：C/N0 四统计（Last/Mean/Std/Slope）、AGC 四统计、接收机时间不确定度 Std、`SignalHistoryRatioW5`、`AgcObservedRatioW5`。
- TCN hidden=16、dropout=0.1、1,880 参数；AdamW `lr=1e-3`、weight decay=`1e-3`、class-balanced CE、batch=256、最多 30 epochs、patience=6、seed=2026、全局阈值 0.5。

| 聚合口径 | Macro-F1 | 正类 F1 | Precision | Recall | FAR | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| 2,077,565 个 signal endpoints pooled | **0.8639** | 0.7980 | 0.8038 | 0.7922 | 6.78% | 89.58% |
| 7 个 outer Session 等权 | **0.8502 ± 0.0933** | — | — | 81.63% ± 15.93% | 5.70% ± 4.55% | — |

与可比的原 19 维 stats TCN16（dropout=0.3、2,024 参数）相比，pooled Macro-F1 从 0.8546 升到 0.8639，Recall 从 77.68% 升到 79.22%，FAR 从 7.18% 降到 6.78%；但 Session 等权 Macro-F1 只从 0.8495 升到 0.8502。这是小幅工程改进，不足以说明跨场景问题已经解决。

| fold | outer test Session | Macro-F1 | Recall | FAR |
|---:|---|---:|---:|---:|
| 1 | 新主楼 `st_L1/19.22` | 0.9316 | 88.05% | 2.98% |
| 2 | 新主楼 `st_L5/20.16` | 0.7777 | 85.97% | 8.84% |
| 3 | 新主楼 `st_L5/20.36` | 0.9118 | 99.77% | 2.14% |
| 4 | 新主楼 `st_L_15/18.42` | 0.9000 | 81.20% | 3.21% |
| 5 | 操场 `st_L1/08.40–09.12` | 0.9113 | 95.53% | 9.33% |
| 6 | 操场 `st_L5/09.48–10.14` | **0.6761** | **55.86%** | **12.74%** |
| 7 | 操场 `st_L_15/07.30–08.01` | 0.8429 | 65.03% | 0.66% |

### `st_L5` 区间内同期 L1 改正类：标签敏感性实验（未采纳）

2026-07-25 使用完全相同的 7-fold 协议和保留模型，做了一次独立标签策略实验：保留正式标签，同时把 reviewed `st_L5` 欺骗区间内的同期 L1 endpoint 临时由 0 改为 1。实验只写入独立张量和训练目录，没有修改中央 CSV、`configs/preprocessing.yml`、原张量或原 checkpoint。

| 口径 | Pooled Macro-F1 | Precision | Recall | FAR | Session 等权 Macro-F1 |
|---|---:|---:|---:|---:|---:|
| 正式标签保留基线 | 0.8639 | 80.38% | 79.22% | 6.78% | 0.8502 ± 0.0933 |
| L5 区间内 L1 同标正类并重训 | 0.8655 | 88.07% | 75.43% | 4.96% | 0.8818 ± 0.0890 |

fold 2、3、6 的测试真值随策略改变，不能把表面 F1 上升直接解释为模型能力提升。固定旧 checkpoint、只按新标签重新评分后，再与新标签重训比较，fold 2/3/6 Macro-F1 分别由 `0.7648/0.8336/0.6428` 升至 `0.8450/0.9551/0.7122`，说明重训确实学到了一部分新增 L1 正类；但操场长 L5 仍没有质变。

最关键的反证是：操场长 L5 中，新标签下 L1 Recall 为 54.91%，但不同设备从 0.09% 到 97.80% 极不稳定；原本 L5 正类的 Recall 又从 55.86% 降至 37.06%。Huawei L5 Recall 从 81.50% 降至 66.53%，RedMi K60 L5 从 46.82% 降至 23.72%。因此该策略只保留为“广义受影响/伴随干扰”口径的敏感性对照，不替代正式二分类标签，也不改变当前保留基线。

### 未解决的核心问题

操场长 L5（fold 6）仍是当前主瓶颈。即使使用保留基线，Huawei Mate40 的 L1 FAR 为 39.40%，Huawei L5 的 Recall/FAR 为 81.50%/23.08%，RedMi K60 L5 的 Recall/FAR 为 46.82%/0%。这表明模型仍在不同设备间以相反方式利用 AGC 等特征：对 Huawei 容易误报，对 RedMi 容易漏报。

最近探索没有形成可推广的突破：

- `no_IsL5` 的 Session 等权 F1 虽为 0.8543，但操场 L5 降至 0.6119；不能替代当前基线。
- TCN32 pooled Macro-F1 为 0.8602、参数增至 6,296，操场 L5 为 0.6596；扩大容量没有价值。LSTM/GRU 也没有同时改善 Recall 与 FAR。
- W3/W5 pilot 分别为 0.6226/0.5847。该筛查使用 `time_steps=7 / guard=6` 的预生成协议，而主基线为 W5 / guard=4，不能作为严格窗口长度消融；但两者均未给出值得继续窗口搜索的正向证据。
- 删 raw+stats AGC 的 fold 6 可达 0.6998，AGC 同时刻同频段中位数残差可达 0.6889；两者都压低 Huawei 误报，却把 Huawei L5 Recall 压至约 54%，且跨折不稳定。因此它们只作为“AGC 共模/设备域偏移存在”的诊断证据，不是候选模型。
- 其余简单 stats 消融、模型结构和小幅超参数调整都没有达到“困难 L5 提升约 0.05 或错误结构同时改善”的门槛。

### 当前结论与建议

1. 将 `compact11 + TCN16 + dropout=0.1` 保留为后续对照基线，不再围绕模型容量、普通窗口长度或简单删特征进行完整 7-fold 搜索。
2. 下一阶段优先处理数据和评价：建立 `trusted / questionable / excluded` Session 清单，复核操场长 L5 的 Huawei/RedMi 原始曲线和标签语义，并取得真正未参与过调参的独立 Session。
3. 后续报告必须同时给出 pooled、Session 等权，以及 Session×Device×Band 的最坏组指标；不能用大 Session 的 pooled Accuracy 掩盖操场 L5 的失败。
4. 历史设备级 LightGBM 的约 0.916x 分数属于不同任务、旧数据/标签和已反复使用的测试协议，不能与本节的卫星级结果混称“当前最佳”。
5. `l5_l1_positive` 仅是显式可复现的实验策略；在老师确认目标语义前，正式标签仍保持原口径。

## 1. 一句话结论

项目目前仍处于“数据与评估协议收敛”阶段，尚未确定最终模型。当前静态逐 `signal_id` 的保留基线是 `compact11 + TCN16 + dropout=0.1`；它在完整 7-fold 的 pooled Macro-F1 为 0.8639，但 Session 等权均值仅为 `0.8502 ± 0.0933`，操场长 L5 仍只有 0.6761。跨 Session、设备和场景的波动仍很大；动态场景以及操场 L5/L15 的主要瓶颈更像是标签可信度、设备观测差异和特征域偏移，而不是模型容量不足。

因此，当前不应继续围绕某个模型反复调参。交接后的第一优先级应是建立可信 Session 清单、逐场景复核数据，并明确新的独立测试数据；模型比较应在这些前提固定后重跑。

## 2. Git 与工作区基线

- 当前分支：`main`。
- 本次同步前本地 `HEAD` 与 `origin/main` 均为 `86018cc`；后续状态以 Git 最新提交为准。
- 本次同步新增可选 `l5_l1_positive` 标签策略、复现说明和上述敏感性实验结论；正式标签配置及保留基线均未改变。
- 2026-07-23 重新生成的两套 label plots 位于被 Git 忽略的 `output/`，不会进入仓库。
- Git 历史中的 P0–P5 是设备级路线的探索记录；保留代码与实验台账用于追溯，但不再将 LightGBM、DLinear 或某个双分支网络描述为已锁定主模型。

## 3. 当前数据与标签状态

### 3.1 数据快照

- 当前正式原始日志：123 份，其中操场 89 份、新主楼 34 份。
- 已主动剔除 9 份操场日志：`dy_L5/2022.07.08` 3 份、`st_L5/2025.07.30.09.41_2025.07.30.09.45` 6 份；它们不参与当前重建和训练。
- `output/processed_gnss_data.csv` 已按上述 123 份日志重建为 2,998,458 行；正类 631,003、负类 2,367,455，全部为 `reviewed/session_config`。
- 数据政策说明：`docs/data_inventory.md`。
- 本地清单按需生成到 `output/data_manifest.csv` 和 `output/data_csv_session_manifest.csv`，不作为仓库文件维护。
- 权威标签配置：`configs/preprocessing.yml`。
- `output/` 不进入 Git；中央 CSV 保留在本地，是当前最值得保留的可重用缓存。

### 3.2 已确认的标签决定

- 操场 `dy_L_15/2025.07.30.08.26_2025.07.30.08.32（动态L1+L5）` 不再使用场景级旧区间 `[260970, 261040]`，改用经多人曲线复核后的 Session 级区间 `[260990, 261020]`。
- 该修订有明确边界：真实设备工作区间无法从隔壁组记录中精确还原，但原区间两端缺少可见欺骗响应，且数据提供方承认标注或干扰设备可能存在问题。因此当前区间是“人工审查后的可信修订”，不是物理真值。
- 同一 Session 中 RedMi K60 的 `gnss_log_2025_07_30_08_17_11` 暂时保持现状，不做额外剔除或重标。它在清单中继续作为 reviewed 数据存在。
- 操场与新主楼统一使用 `Environment -> Scenario -> Session -> {status, intervals}`；未显式列出的 Session 标为 `needs_review`，不得进入正式训练。

### 3.3 已知数据风险

- 操场动态 L15 即使收紧标签后，逐 signal LSTM validation Recall 仍只有 34.50%；操场动态 L5 Recall 仅 9.27%。这说明问题不只来自那一个过宽区间。
- 当前保留 7-fold 基线中，操场长 L5 Session 的 Macro-F1 为 0.6761、Recall 为 55.86%、FAR 为 12.74%，仍是最明显的瓶颈；操场 L15 Session 的 Macro-F1 为 0.8429、Recall 为 65.03%、FAR 为 0.66%，主要问题仍是漏检。这两个 Session 应继续优先做逐设备、逐频段和原始曲线复核。旧 full-19 TCN 基线对应的 0.6329/50.33%/15.50% 只作历史对照。
- 新主楼 L5 的正类数量偏少并不等于张量漏样本。当前标签语义是“目标 TOW 区间且 `FreqBand == 5` 才为正类”；同录制中的 L1 信号仍为负类。Pixel Watch 在这些录制中通常没有 L5 信号，因此会贡献大量负类而没有 L5 正类。
- Google Pixel Watch 1 的 `AgcDb` 全缺失。绝对 AGC 统计很容易携带设备身份，必须保留缺失标记，并在后续做 no-AGC 或相对化消融。
- `pipeline_total/02_batch_plot_feature_images.py` 只按显式 Session 级 reviewed 配置绘制阴影，不再使用场景级回退；PNG 仍只是复核辅助证据，标签判断以 `configs/preprocessing.yml` 和重建后的 CSV 为准。

## 4. 路线演进与当前边界

### 4.1 2026-07-18 前：P0–P5 设备级探索

P0–P5 覆盖逐信号聚合、设备级统计张量、LightGBM/DLinear 等模型、静态跨环境、静态 Session-CV 和静态+动态混合任务。其价值是暴露了环境、频段和 Session 划分对结果的强影响；详细数字保留在 `docs/experiment_registry.md`。

这些结果存在共同边界：部分 test 已被用于窗口、模型和错误诊断，不能再称为论文最终盲测；设备级 27 维统计与 LightGBM 也不是当前已经确认的论文主模型。

### 4.2 2026-07-22：静态逐 signal 探索

最近的独立路线把重点转回卫星/`signal_id` 级检测，并尝试逐 signal 窗口统计。当前保留窗口为 W5；操场 L5 的 W3/W5 筛查结果分别为 0.6226/0.5847，但它使用 `time_steps=7 / guard=6` 的预生成协议，而主基线为 W5 / guard=4，不能严格横向比较。它们仍未提供值得继续窗口搜索的正向证据，因此该方向不作为当前主线。

当前保留的双分支模型包含：

- raw 分支：5 个因果历元，实际使用 `Cn0DbHz`、`AgcDb`、`ReceivedSvTimeUncertaintyNanos`、`PseudorangeRateUncertaintyMetersPerSecond`、`FreqBand`；
- stats 分支：对每个 `signal_id` 在 W5 内保留 11 维统计特征：C/N0 四统计、AGC 四统计、接收机时间不确定度 Std、历史完整度和 AGC 可观测率；
- 明确排除 CSV 中已有的 `Cn0DbHz_dt` 和 `Cn0DbHz_std`，避免与重新计算的窗口统计重复；
- TCN + stats MLP 的当前保留配置为 hidden=16、dropout=0.1、weight decay=1e-3，共 1,880 个参数；原 19 维、dropout=0.3、2,024 参数的模型是历史完整基线。

这只是最近一次可复现实验配置，不是最终模型选择。

### 4.3 2026-07-23：清理数据后的 7-fold 静态重训

短时操场静态 L5 Session `2025.07.30.09.41_2025.07.30.09.45` 已随 6 份原始日志一起剔除。它全为负类，因此当前静态独立录制从 8 个变为 7 个，outer-session 协议也相应变为 7 folds。新 fold 6、7 分别对应旧 fold 7、8；比较结果时必须按 `Environment + Scenario + Session` 对齐，不能只看 fold 编号。

本轮仍然只训练静态数据。操场动态 L15 的区间修订不会直接进入本轮张量；真正改变训练的是上述全负静态 Session 不再出现在其余 folds 的开发集。剩余 7 个 outer test 的样本、正负类 support 与旧实验逐项一致，因此可以直接比较同一批 Session 上的新旧 checkpoint。

协议固定为 W5、raw TCN + stats MLP、hidden=16、AdamW `lr=1e-3`、weight decay=1e-3、batch=256、最多 30 epochs、patience=6、seed=2026、num_workers=0。原始完整基线使用 19 维 stats 与 dropout=0.3；当前保留基线使用 11 维 compact11 stats 与 dropout=0.1。每折只按 inner validation Macro-F1 锁定 checkpoint，7 个 checkpoint 全部锁定后才依次执行 `test-only`。

## 5. 关键实验摘要

| 实验 | 主要结果 | 能说明什么 | 不能说明什么 |
|---|---|---|---|
| 修订标签后的 mixed 逐 signal 基线 | LSTM validation Macro-F1 0.7947、Recall 0.5514、FAR 2.15% | 动态数据是主要漏检来源；短时序比 MLP 有帮助 | test 未读，不能代表泛化 |
| 单一静态固定划分 raw+stats | TCN validation 0.9906，test 仅 0.7015 | 单 Session validation 会造成严重选择过拟合 | 不能把 0.99 当作静态性能 |
| 静态 4-fold，W5 正则配置 | Macro-F1 0.8245 ± 0.0877，FAR 8.34% ± 5.33% | 跨 Session 波动显著，轻量正则略降误报 | 4 折样本仍只有 8 个独立录制 |
| W3/W5/W7 | 0.8154 / 0.8245 / 0.8151 | W5 略优，可作默认 | 差异不支持“W5 显著最好” |
| 6 train / 1 val / 1 test | pooled Macro-F1 0.8232、Recall 0.7487、FAR 9.57% | 增加 train Session 没有解决泛化 | 单 validation Session 仍不稳定 |
| 旧 8-fold Outer-Session / Inner-Time-Block | pooled Macro-F1 0.8386、Precision 0.7968、Recall 0.7168、FAR 6.19%；7 个有正类 Session 的 Macro-F1 0.8515 ± 0.1159 | 其余 Session 都可参与开发，误报较 6/1/1 少 | 包含现已剔除的短时全负操场 L5；不能直接按 fold 编号和新实验比较 |
| full-19 7-fold 基线（历史对照） | pooled Macro-F1 0.8546、Precision 0.7914、Recall 0.7768、FAR 7.18%；Session Macro-F1 0.8495 ± 0.1142 | 作为后续 compact11 的严格同协议对照 | 不是当前保留基线；不能把它误称为最新结果 |
| compact11 7-fold TCN16 d=.1（当前保留） | pooled Macro-F1 0.8639、Precision 0.8038、Recall 0.7922、FAR 6.78%；Session Macro-F1 0.8502 ± 0.0933 | 目前最好的完整 7-fold 结果，参数也从 2,024 降至 1,880 | pooled 仅 +0.0093、Session 均值仅 +0.0007；操场长 L5 仍未解决 |
| `st_L5` 区间同期 L1 改正类（未采纳） | pooled Macro-F1 0.8655、Precision 0.8807、Recall 0.7543、FAR 4.96%；操场 L5 原频段 Recall 降至 37.06% | 新标签重训能学习部分 L1 受影响模式 | 测试语义已改变且设备差异极大；不应替代正式标签或保留基线 |

full-19 7-fold 重训是这条路线的初始完整基线；其后 compact11 TCN16 d=.1 才是当前最新口径，详见文首“当前状态快照”。两者之间的 pooled 提升只有 0.0093，Session Macro-F1 均值仅提高 0.0007，且不同 Session 差异仍很大。所谓“样本量百万级”只是高度相关的 signal endpoints；真正独立的静态录制单元现在只有 7 个。

以下为 full-19 历史基线的逐 Session 结果；当前 compact11 结果、及其与该基线的对照见文首“当前状态快照”。

| 新 fold | Outer test Session | Macro-F1 | Recall | FAR |
|---:|---|---:|---:|---:|
| 1 | 新主楼 `st_L1/19.22` | 0.9382（0.9319） | 89.08% | 2.64% |
| 2 | 新主楼 `st_L5/20.16` | 0.7797（0.8306） | 83.26% | 8.20% |
| 3 | 新主楼 `st_L5/20.36` | 0.9393（0.9769） | 99.80% | 1.39% |
| 4 | 新主楼 `st_L_15/18.42` | 0.9325（0.9271） | 87.68% | 2.52% |
| 5 | 操场 `st_L1/08.40–09.12` | 0.9025（0.8922） | 94.33% | 9.92% |
| 6 | 操场 `st_L5/09.48–10.14` | 0.6329（0.6497） | 50.33% | 15.50% |
| 7 | 操场 `st_L_15/07.30–08.01` | 0.8214（0.7523） | 60.62% | 0.62% |

方向二与 6/1/1 也不是严格的单变量 split 消融：新的 builder 同时增加了断档窗口过滤和按 train 均值填充缺失值。因此两者的数值差不能全部归因于划分方式。

### 5.1 2026-07-24：10 维 stats 联合消融（已淘汰的诊断性结果）

为检查低重要性统计量是否造成设备捷径，训练器新增了可复现的
`--stats-feature-set cn0_agc_coverage`。它不重建张量、不改变 split、窗口、raw 分支或 train-only scaler，只在 stats MLP 输入端保留以下 10 项：C/N0 的 `Last/Mean/Std/Slope`、AGC 的 `Last/Mean/Std/Slope`、`SignalHistoryRatioW5`、`AgcObservedRatioW5`。它联合移除了 `IsL5`，以及伪距速率不确定度和接收机时间不确定度各自的 `Last/Mean/Std/Slope` 共 9 项。故这是**联合消融**，不能把结果单独归因于任意一个被删特征。

协议与基线完全一致：7-fold outer-session / inner-time-block、W5、TCN、hidden=16、dropout=0.3、AdamW `lr=1e-3`、weight decay=1e-3、batch=256、patience=6、seed=2026。参数量从 2,024 降至 1,862。

| 版本 | pooled Macro-F1 | Precision | Recall | FAR |
|---|---:|---:|---:|---:|
| full 19-d stats 基线 | 0.8546 | 0.7914 | 0.7768 | 7.18% |
| `cn0_agc_coverage` 10-d stats | 0.8587 | 0.8017 | 0.7782 | 6.75% |

7 个 Session 的 Macro-F1 均值也仅从 `0.8495 ± 0.1142` 升至 `0.8523 ± 0.1235`（4/7 folds 上升）；这一小幅总体变化不足以覆盖关键失败场景的退化。

Pooled 指标虽小幅上升，但目标难例明显恶化：操场长 L5（fold 6）Macro-F1 从 0.6329 降至 0.5988，Recall 从 50.33% 降至 39.04%，FAR 仅从 15.50% 降至 14.74%。其中 Huawei Mate40 的 L1 FP 从 30,817 降至 24,936（FAR 54.24%→43.89%），但其 L5 FAR 却从 13.35% 升至 25.15%；RedMi K60 的 L5 Recall 从 34.64% 降至 16.13%（FN 14,843→19,046）。

因此该联合消融**不应替代基线**：它降低了部分 Huawei L1 误报，却以更严重的 Redmi L5 漏检为代价。后续完整 7-fold 已验证 `no_IsL5` 不能改善操场 L5，且其它 AGC/uncertainty 简单消融也没有形成可接受的质变；不应再据此继续做同类小幅删特征搜索。每折的设备×频段 CSV 由 `23_evaluate_static_fusion_groups.py` 生成。

## 6. 整理后的代码定位

`pipeline_total/01–10` 是既有数据、画图、标注、基础张量、训练和错误分析链；02 已补充按当前 YAML 配置解析 Session 级标签阴影。

`pipeline_total/11–18` 保留为 P0–P5 历史设备级探索。它们可以复现旧实验，但不应作为新的默认入口。

当前逐 signal 静态实验收敛为四个脚本：

```text
19_generate_static_timeblock_protocol.py
20_build_static_timeblock_tensors.py
21_train_static_signal_fusion.py
23_evaluate_static_fusion_groups.py
```

原有 stats-only builder、6/1/1 生成器和两个重复编号的 time-block 22 脚本已被替代。当前新的 `22_generate_label_review_dashboards.py` 用于正式的 Session 级标签审查，与旧 22 无关。

## 7. 最小重建流程

标签或原始数据发生变化后，先重建中央 CSV 和审计文件：

```powershell
python scripts/build_mirrored_data_csv.py --config configs/preprocessing.yml --overwrite
python pipeline_total/04_build_labeled_processed_csv.py --mode full --config configs/preprocessing.yml
python pipeline_total/01_generate_plot_feature_csv.py --data-root data_raw --config configs/preprocessing.yml --overwrite
python scripts/build_data_manifest.py --output output/data_manifest.csv
python scripts/audit_extracted_csv.py --input-dir data_csv --output-json output/data_csv_audit.json
python scripts/build_csv_session_manifest.py --input-dir data_csv --output-csv output/data_csv_session_manifest.csv
```

重建当前 7-Session 静态 time-block 协议：

```powershell
python pipeline_total/19_generate_static_timeblock_protocol.py `
  --csv output/processed_gnss_data.csv `
  --source-recording-manifest output/protocols/static_time_block_outer_v2/source_recording_manifest.csv `
  --output-dir output/protocols/static_time_block_outer_v2 `
  --time-steps 5 `
  --block-epochs 256 `
  --val-fraction 0.20 `
  --segment-gap-seconds 2
```

每个 outer fold 分别构建张量；`epoch_split_manifest.csv` 是权威逐历元划分，不能误用仅作汇总的 `time_block_manifest.csv`：

```powershell
python pipeline_total/20_build_static_timeblock_tensors.py `
  --outer-manifest output/protocols/static_time_block_outer_v2/fold_1/recording_split_manifest.csv `
  --block-manifest output/protocols/static_time_block_outer_v2/fold_1/epoch_split_manifest.csv `
  --output-dir output/tensors/static_timeblock_outer_v2/fold_1 `
  --time-steps 5 `
  --block-size 256
```

训练与 test-only 的命令以 `pipeline_total/README.md` 为准。每折 scaler 只能用 train 拟合，validation 只用于早停；test checkpoint 锁定后再读。当前 7 个 outer tests 已经全部读取过，后续依据这些结果调参时必须记作迭代式 CV，不能再称完全盲测。

## 8. 本地生成物保留策略

本次清理后，`output/` 只保留：

- `processed_gnss_data.csv` 与缺失报告；
- `data_csv_audit.json`；
- `dynamic_labeling_review/`；
- `review/trusted_signal_baseline_v1/` 下的关键错分明细与汇总；
- `label_plots_20260723/new_building/`（238 张）和 `label_plots_20260723/playground/`（623 张），按统一 Session 级标签配置从 123 份现役日志干净重建；
- `protocols/static_time_block_outer_v2/`、`tensors/static_timeblock_outer_v2/` 和 `training/static_timeblock_outer_v2/`，作为本轮 7-fold 重训的协议、张量、checkpoint、日志和逐折指标；
- `training/static_timeblock_outer_v2_explore_compact11_tcn16_d10/`，作为当前保留静态逐 signal 基线的 7-fold checkpoint、逐折指标和 fold 6 设备×频段诊断 CSV；
- `training/static_timeblock_outer_v2_ablate_cn0_agc_coverage_v1/`，作为当前 10 维 stats 联合消融的 checkpoint、逐折指标和设备×频段诊断 CSV；
- `tensors/static_timeblock_outer_v2_l5_l1_positive/`、`training/static_timeblock_outer_v2_l5_l1_positive_compact11_tcn16_d10/` 和固定旧 checkpoint 的同标签评分目录，作为一次未采纳的标签敏感性实验；结论已写入本文，后续磁盘清理时可整体删除并按需重建；
- `output/README.md`。

张量、checkpoint 和训练日志仍属于可重建产物；当前 `static_timeblock_outer_v2` 只因本轮交接与新旧对照暂时保留，指标稳定写入文档后可再归档或删除。当前两套 plots 只是本轮标签复核需要。旧产物已集中迁入被 Git 忽略的 `output/_rebuildable_archive_20260722/`。此外，旧 `new_building_label_plots/` 与 `playground_label_plots/` 未自动删除，其中旧操场目录仍含已剔除 Session 的 63 张残留图，不能再作为当前数据口径使用。当前执行环境的递归删除审批服务异常，因此磁盘空间尚未真正释放；确认无需恢复后可人工删除旧目录和归档。历史指标已经压缩进本文和 P0–P5 台账。

## 9. 推荐交接顺序

1. 先按 Session 建立 `trusted / questionable / excluded` 数据清单，重点复核操场动态 L5、动态 L15、静态长 L5 和静态 L15；不要把“reviewed”简单等同于物理真值。
2. 明确论文主任务以卫星/逐 signal 检测为主，设备级聚合作为部署层或辅助实验；不要在两种评价单位之间混用指标。
3. 新增独立录制或保留真正未触碰的 Session。当前只有 7 个静态录制，再复杂的交叉验证也不能创造新的环境多样性。
4. 固定可信数据、特征和 split 后，再重跑 MLP、TCN/LSTM、raw+stats 等少量轻量基线；在此之前将 compact11 TCN16 d=.1 作为唯一对照。优先报告 Session×Device 宏平均、最差设备 FAR、攻击 Recall 和检测时延。
5. 在数据问题没有澄清前，不建议继续融合更多模型或扩大网络容量。若以后做融合，应先证明不同模型的错误具有互补性，而不是只比较 pooled Accuracy。

## 10. 文档入口

- 当前状态与交接：本文。
- 组会汇报提纲与讲稿：`docs/group_meeting_brief_20260725.md`。
- P0–P5 历史实验：`docs/experiment_registry.md`。
- 数据清单：`docs/data_inventory.md`；本地清单由上述脚本生成到 `output/`。
- 标签复核：`docs/dynamic_labeling_assistant.md`、`docs/dy_manual_label_intervals.csv`。
- 信号级数据构建：`docs/signal_level_feature_extraction.md`。
- 历史静态 4-fold 协议：`docs/static_session_cv_protocol.md`。
- 脚本索引：`pipeline_total/README.md`。

# 项目交接状态（2026-07-28）

本文是当前工作区的唯一状态入口。它说明已经确认的数据事实、最近的探索结论、结果边界和可复现入口。历史 P0–P5 结果只用于追溯，不代表当前已经锁定的主线。

## 当前状态快照（2026-07-28）

> 本节是当前逐 `signal_id` 路线的权威状态。后续各节保留此前 7-fold 静态基线、标签敏感性和历史实验的形成过程；若数值冲突，以本节、[静态+动态统一基线](static_dynamic_signal_cv4_20260727.md) 和 [Fold 6 诊断快照](static_signal_fold6_diagnostics_20260727.md) 为准。

### 任务、数据和结果边界

- 完整 7-fold `compact11 + TCN16 + dropout=0.1` 继续作为纯静态保留对照。它回答的是 7 个 reviewed 静态 Session 下的表现，不是最终模型声明。
- 2026-07-27 已用相同 W5、compact11 和 TCN16 配置完成统一静态+动态 4-fold outer-CV v2：24 个完整 Session（7 静态、17 动态）各 test 一次。同一采集事件通过 `outer_group` 保持完整，v2 是后续 mixed 路线的固定起点，不替代纯静态对照。
- 当前 7 个静态和 17 个动态 outer test 都已被读取；后续依据这些结果设计模型时均属于迭代式开发，不能再称新的独立盲测。
- 2026-07-26 至 27 的 E9-E11 仅围绕已反复读取的操场长 L5 Fold 6 做诊断。它们使用 all-development fixed-epoch refit，且 E10/E11 在测试时将非 L5 endpoint 强制为正常；这些数字只用于定位 L5 的错误结构，不能与通用 7-fold 基线的 Macro-F1 直接排序。

### 统一静态+动态基线已确认事实

| Test 口径 | Macro-F1 | Precision | Recall | FAR | Session 等权 Macro-F1 |
|---|---:|---:|---:|---:|---:|
| Overall | 0.8656 | 78.33% | 79.37% | 5.90% | 0.7405 ± 0.1210 |
| Static | 0.8782 | 80.73% | 83.46% | 6.99% | 0.8333 ± 0.0974 |
| Dynamic | 0.7693 | 61.82% | 55.11% | 3.82% | 0.7024 ± 0.1085 |

- Overall 被长静态 Session 明显主导，判断动态效果必须看 dynamic 子集和 Session 等权结果。
- `dy_L5` pooled Macro-F1 仅 0.5647、Precision 13.03%、Recall 18.67%；四个 `dy_L5` Session 等权为 `0.5733 ± 0.0427`，仍是最明确的瓶颈。
- mixed 模型在同一批静态 endpoint 上的 pooled Macro-F1 比纯静态基线高 0.0143，但静态 Session 等权低 0.0169；不同静态 Session 有升有降，不能称为稳定提升。
- v1 试跑曾让新主楼两个重叠 `st_L5` Session 和操场两个重叠 `dy_L_15` Session 跨 outer fold，且 256-epoch 内层块使短动态 Session 严重失衡或完全缺少 validation。v2 将两对采集事件绑定为 `G08/G09`，改用 64-epoch 严格 validation；四折 raw validation 为 20.20%-20.78%，六种 Scenario 均有正负支持。
- 2026-07-28 补做了“每个 development Session 的 clean 约 80/20、长 attack run 约 80/20、短 attack atom 按同 Scenario 跨 Session 分配”的 inner split 消融。修正版 v2 的 Validation 均值从 0.8978 升至 0.9295，但相同 outer test 的 Overall / Dynamic pooled Macro-F1 为 0.8637 / 0.7668，仍略低于 strict-v2；`dy_L5` 虽升至 0.5817，Recall 仍只有 21.20%，未形成质变，因此不替代 strict-v2。完整规则、v1 修正说明和配对结果见 [完整实验记录](static_dynamic_signal_cv4_20260727.md#inner-trainvalidation-划分消融reviewed-state-stratified2026-07-28)。
- 2026-07-28 又完成 reviewed 欺骗区间内“不区分频段、全部 `signal_id` 标为正类”的标签语义敏感性实验。相同 state-stratified v2 协议下 Overall pooled Macro-F1 为 0.8668，但 Dynamic 从旧语义的 0.7668 降至 0.7351；新增动态非目标频段的合计 Recall 仅 16.77%。`dy_L5` Macro-F1 虽升至 0.6394，Recall 仍只有 22.31%，主要变化是原 L1 假正被新语义重新计为真阳性，并非动态检测质变。该实验不修改中央 CSV/YAML、不替代 target-band-only 基线，详见 [完整实验记录](static_dynamic_signal_cv4_20260727.md#标签语义敏感性reviewed-区间内全部信号为正类2026-07-28)。
- 协议、逐场景结果、全负动态 Session 的解释和复现入口见 [完整实验记录](static_dynamic_signal_cv4_20260727.md)。

### Fold 6 已确认事实

- E9a 将此前落在 validation 的短新主楼 L5 攻击片段纳入 all-development refit 后，Recall 从 50.33% 升至 55.82%；E9b 的 Session 均匀采样只进一步升至 56.93%，说明样本量不是主因。
- E10 的 L5-only 专家把 L5 Recall 提升到 60.58%，E11 的共享编码器加设备条件头进一步升至 62.73%，但仍远低于 80% 目标，且 L5 FAR 分别为 11.46% 和 11.92%。
- 设备错误方向不一致：E11 中 Pixel 6 的 Recall 为 4.06%、FAR 为 0；Mate40 的 Recall 为 94.20%、FAR 为 50.51%；RedMi K60 的 Recall 为 50.80%、FAR 为 0.15%。这不是一个全局阈值或单纯样本失衡能够同时解决的问题。
- 只读阈值诊断表明 Pixel 6 的分数排序本身较好但默认阈值失配；Mate40、RedMi K60 与 MI8 在可接受 FAR 下的分数排序仍不足。详细数值、协议差异和解释见诊断快照。

### 当前问题陈述

当前证据支持以下问题描述，而非“某个模型已经优于其他模型”：在已审核的静态目标频段标签下，跨 Session 的绝对 C/N0/AGC 及其短历史统计会随设备和环境变化。模型对 Mate40 偏向过报、对 RedMi K60 和 Pixel 6 偏向漏报。已有频段路由、辅助任务、跨频段上下文、全开发集 refit、Session 均匀采样、L5-only 专家和设备条件头均未在同一 Fold 6 上同时获得高 Recall 与低 FAR。

### 本轮补充：跨频段耦合与分层事件检测（2026-07-27）

#### 现象、假设与论文价值

- 新发现不是简单的“L5 样本少”，而是同一设备的 L1/L5 观测可能并非独立：L5 受到欺骗时，L5 C/N0 常向上；未被直接欺骗的 L1 可能同时出现向下的压制/干扰响应。
- 一个待文献和原始曲线进一步验证的芯片实现假设是：部分接收机先捕获 L5，再利用该结果辅助或引导 L1 搜索/跟踪；L5 欺骗的突增因而可能改变 L1 的接收机行为。该假设可解释“L5 上升、L1 下降”的联合模式，但当前不能将它表述为已证实的因果机制。
- 该现象具有论文价值：不同手机芯片对跨频段耦合的响应不同，不能把所有非目标频段的变化直接并入“直接欺骗”标签。论文应报告这一罕见的设备异质性现象，并以相关文献和逐设备原始曲线补强解释。

#### 标签与决策层级

应明确区分两类输出，不能互相替代：

| 层级 | 预测问题 | 正类语义 | 对 L5 攻击期间 L1 下降的处理 |
|---|---|---|---|
| 卫星/`signal_id` 级 | 该信号是否被直接欺骗 | 保持当前 target-band-only 标签 | L1 是攻击关联异常，不自动记为 L1 直接欺骗 |
| 设备事件级 | 设备此刻是否处于攻击/异常事件 | 由人工审核的 Session 攻击 TOW 区间定义，与频段无关 | 可作为设备处于攻击事件的证据 |

因此，后续系统可同时输出“设备攻击事件”和“卫星直接欺骗/攻击关联异常”。设备级正类不能反向改写卫星级直接欺骗标签。

#### 为什么现有 L5 逐卫星模型会失败

- L1 欺骗场景中，受攻击的 L1 卫星数量多且 C/N0 普遍上升，因此模型较容易学习到主导模式。
- L5 欺骗场景中，真正上升的 L5 数量较少；更多的 L1 endpoint 则表现为下降/压制。若用目标频段直接欺骗标签训练，模型会同时面对少量“上升正类”和大量跨设备、跨 Session 的抑制型负类。
- 绝对 C/N0、AGC 及短历史统计还会携带设备与环境差异。已有 Fold 6 诊断显示 Mate40 倾向误报，RedMi K60 与 Pixel 6 倾向漏报，错误方向相反，说明不能靠一个全局阈值解决。

#### 已实现的两条诊断路线

1. **设备攻击事件基线**
   - 脚本：[36_build_device_attack_event_tensors.py](../pipeline_total/36_build_device_attack_event_tensors.py)、[37_train_device_attack_event.py](../pipeline_total/37_train_device_attack_event.py)。
   - 以同一 `device x endpoint time` 为一个样本，聚合 L1/L5 的 C/N0、AGC、接收机时间不确定度、可观测率，并显式加入 L5-L1 差值及“L5 上升 + L1 下降”耦合特征。
   - 事件标签来自审核后的 Session 区间，和目标频段无关。训练器将外层测试隔离为显式 `--test-only`，避免训练/早停阶段读取它。
   - 实验说明：[hierarchical_attack_event_experiment.md](hierarchical_attack_event_experiment.md)。

2. **E12a：动态 L5/L1+L5 增广静态 L5 专家**
   - 脚本：[36_build_static_dynamic_l5_augmentation_tensors.py](../pipeline_total/36_build_static_dynamic_l5_augmentation_tensors.py)、[37_train_static_dynamic_l5_augmentation.py](../pipeline_total/37_train_static_dynamic_l5_augmentation.py)。
   - 动态 `dy_L5`、`dy_L_15` Session 仅加入训练集，静态 Fold 6 外层测试保持为操场长 `st_L5`，不能混合后宣称为同一基准。
   - 每个动态窗口通过 `is_dynamic` 写入张量，确保其来源可审计；模型选择应只使用完整静态 development Session 的内层验证。
   - 协议和清单：[static_dynamic_l5_augment_v1/fold_6/README.md](protocols/static_dynamic_l5_augment_v1/fold_6/README.md)。

#### 设备事件基线：已完成的 Fold 6 诊断

本轮从 `output/tensors/static_timeblock_outer_v2/fold_6` 构建了设备事件张量；产物位于被 Git 忽略的 `output/tensors/hierarchical_event_v1/fold_6/device_tensors`。样本数如下：

| 划分 | 设备时间窗 | 正类 | 负类 |
|---|---:|---:|---:|
| train | 32,682 | 11,249 | 21,433 |
| val | 10,085 | 3,145 | 6,940 |
| outer test | 8,181 | 3,886 | 4,295 |

线性基线在 train/val 选择后得到 `val Macro-F1=0.9673`，但固定 checkpoint 的 outer test 为：

| 指标 | outer test |
|---|---:|
| Macro-F1 | 0.7461 |
| Precision | 0.9804 |
| Recall | 0.5162 |
| FAR | 0.93% |

逐设备 Recall 为 Pixel 6 `54.23%`、Pixel Watch 1 `40.13%`、Pixel Watch 2 `83.09%`、Mate40 `52.89%`、RedMi K60 `6.03%`、MI8 `96.91%`。Mate40 FAR 为 `4.82%`，其余设备接近 0。这再次证明单一 validation Session 的高分不能用于模型选择结论：模型总体上很保守，误报低但对 RedMi K60 等设备严重漏保。

该 Fold 6 test 已反复参与诊断，因此这些数字是迭代式开发诊断，不是新的独立盲测，也不能据此继续手调阈值后再报告为 test 性能。

#### 下一位同学的优先事项

1. 检索并阅读“GNSS 双频捕获/跟踪、辅助捕获、跨频段干扰或共享接收机资源”相关论文；把支持或反驳上述耦合假设的证据记录到论文素材中。
2. 从开发集实施留一完整静态 Session 的多折设备事件评估。选模型时同时报告各折中位数 Macro-F1、最差设备 Recall 和 Mate40 FAR；outer test 不参与参数、特征或阈值选择。
3. 重点检查 RedMi K60 在 L5 事件中的原始 L1/L5 曲线、活跃卫星数、缺失率和事件标签对齐，判断其 `6.03%` 设备事件召回是特征失配、设备窗口聚合失真，还是原始观测/标签问题。
4. 完成 E12a 的内层静态 Session 选择和固定 epoch refit，再以一次显式 `--test-only` 做诊断。不要把动态增广后的结果与纯静态 7-fold 基线直接排序。
5. 若设备级模型继续保守，优先评估设备条件化或校准方案；但必须同时保留卫星级“直接欺骗”和“攻击关联异常”的语义边界。

#### 本轮 Git 提交

- `da6a08a 修复(静态张量): 补全可选特征与标签接口`
- `989b2ba 新增(静态检测): 加入静态动态 L5 增广实验`
- `bb83338 新增(分层检测): 加入设备攻击事件基线`
- `67351eb 修复(分层检测): 固定默认标签配置路径`
## 1. 一句话结论

项目目前仍处于“数据与评估协议收敛”阶段，尚未确定最终模型。纯静态保留基线仍是 `compact11 + TCN16 + dropout=0.1`，7-fold pooled Macro-F1 为 0.8639、Session 等权为 `0.8502 ± 0.0933`；相同配置的静态+动态 4-fold v2 基线 overall 为 0.8656，但 dynamic 仅为 0.7693、动态 Session 等权仅为 `0.7024 ± 0.1085`，其中 `dy_L5` 只有 0.5647。跨 Session、设备和场景的波动仍很大；当前瓶颈更像是标签可信度、设备观测差异和特征域偏移，而不是模型容量不足。

因此，当前没有可作为最终部署结论的模型；任何结果引用都必须保留标签口径、Session 协议和 test 已被多次用于诊断的边界。

## 2. Git 与工作区基线

- 当前分支：`main`。
- 具体提交状态以 `git log --oneline` 和 `git status --short` 为准；本文不固定某一个会随同步变化的 HEAD。
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
- `protocols/mixed_timeblock_outer_cv4_v2/`、`protocols/mixed_timeblock_outer_cv4_w5_v2/`、`tensors/mixed_timeblock_outer_cv4_w5_v2/` 和 `training/mixed_timeblock_outer_cv4_w5_compact11_tcn16_d10_v2/`，作为可信统一静态+动态 4-fold 基线的可重建协议、张量、checkpoint 和汇总；v1 仅保留为发现协议问题的历史试跑，可在确认不需追溯后删除；
- `protocols/mixed_timeblock_outer_cv4_w5_state_stratified_v2/`、`tensors/mixed_timeblock_outer_cv4_w5_state_stratified_v2/` 和 `training/mixed_timeblock_outer_cv4_w5_compact11_tcn16_d10_state_stratified_v2/`，作为修正后但未采纳的 inner split 消融；v1 未严格满足逐 Session clean 80/20，仅保留为问题追溯，后续需要释放空间时可优先删除 v1 及 v2 张量并按需重建；
- `tensors/mixed_timeblock_outer_cv4_w5_state_stratified_interval_all_positive_v1/` 和 `training/mixed_timeblock_outer_cv4_w5_compact11_tcn16_d10_state_stratified_interval_all_positive_v1/`，作为未采纳的全区间正类标签敏感性实验；它与旧任务的 Test 标签定义不同，只能按文档中的场景/频段结果解释，后续可按需重建；
- `output/README.md`。

张量、checkpoint 和训练日志仍属于可重建产物；当前 `static_timeblock_outer_v2` 只因本轮交接与新旧对照暂时保留，指标稳定写入文档后可再归档或删除。当前两套 plots 只是本轮标签复核需要。旧产物已集中迁入被 Git 忽略的 `output/_rebuildable_archive_20260722/`。此外，旧 `new_building_label_plots/` 与 `playground_label_plots/` 未自动删除，其中旧操场目录仍含已剔除 Session 的 63 张残留图，不能再作为当前数据口径使用。当前执行环境的递归删除审批服务异常，因此磁盘空间尚未真正释放；确认无需恢复后可人工删除旧目录和归档。历史指标已经压缩进本文和 P0–P5 台账。

## 9. 交接阅读顺序

1. 先读本文的当前状态快照、`docs/static_dynamic_signal_cv4_20260727.md` 和 `docs/static_signal_fold6_diagnostics_20260727.md`，区分 mixed 4-fold 基线、静态 7-fold 对照与 Fold 6 迭代式诊断。
2. 再核对 `configs/preprocessing.yml`、`docs/data_inventory.md` 和 label review 产物，确认要引用的标签口径与数据快照。
3. 复现实验时仅使用 `pipeline_total/README.md` 中对应脚本、张量目录和 checkpoint 记录；训练输出仍写入被 Git 忽略的 `output/`。
4. 对外陈述结果时保留协议、标签语义和 test 已被使用的边界，不将单一折次或不同任务的指标混为“当前最佳”。

## 10. 文档入口

- 当前状态与交接：本文。
- 静态+动态逐 signal 4-fold 基线：`docs/static_dynamic_signal_cv4_20260727.md`。
- Fold 6 诊断结果快照：`docs/static_signal_fold6_diagnostics_20260727.md`。
- 组会汇报提纲与讲稿：`docs/group_meeting_brief_20260725.md`。
- P0–P5 历史实验：`docs/experiment_registry.md`。
- 数据清单：`docs/data_inventory.md`；本地清单由上述脚本生成到 `output/`。
- 标签复核：`docs/dynamic_labeling_assistant.md`、`docs/dy_manual_label_intervals.csv`。
- 信号级数据构建：`docs/signal_level_feature_extraction.md`。
- 历史静态 4-fold 协议：`docs/static_session_cv_protocol.md`。
- 脚本索引：`pipeline_total/README.md`。

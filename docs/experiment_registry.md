# P0–P5 历史实验台账

> 状态说明（2026-07-22）：本文记录 2026-07-18 前的设备级 P0–P5 探索，用于追溯实验演进，不代表当前已锁定模型或最新推进路线。当前数据问题、逐 signal 探索和交接结论请以 [handoff_status.md](handoff_status.md) 为准。

更新时间：2026-07-18

本文是当前实验的唯一查表入口。它记录每项实验究竟在回答什么问题、使用什么模型和窗口、数据如何按完整 Session 划分，以及结果能否与其他实验横向比较。

## 先读这一节

1. `static_cross_env_v1` 的 test 已被反复用于模型、窗口和错误分析。它现在只能称为**开发参考 test**，不能作为论文的最终独立测试结果。
2. "动态加入训练、静态测试"是在检验动态数据能否帮助**静态**检测；它不是同时检测静态和动态的模型。
3. "统一静态+动态检测"才是 train、val、test 都同时含静态和动态的任务。它和纯静态 test 的结果不能直接比较。
4. 逐信号模型经投票得到的设备告警，与直接设备级模型不是同一类输入和任务，不能只按 F1 排名。
5. 本文所有 `Recall` 均为欺骗类召回率，`FAR = FP / (TN + FP)`。默认分类阈值均为 `0.5`。

## 共同定义

### 数据和切分原则

- 数据：操场（`playground`）和新主楼（`new_building`）共 132 份 GNSS 原始日志，经审核后为 2,139,284 条信号级记录。
- 切分最小单位：一条完整真实录制 Session。一个 Session 的所有设备日志只能进入 train、val、test 中的一个，禁止跨集合。
- 场景：`st_*` 为静态，`dy_*` 为动态；`L1`、`L5`、`L_15` 分别代表 L1、L5、双频攻击场景。
- 标签：设备当前历元中任一有效信号为欺骗，则设备窗口标签为 1。
- 归一化：设备神经网络的标准化统计量仅在该协议的 train 中拟合；LightGBM 不依赖该标准化。

### 两条建模路线

| 路线 | 输入和输出 | 已完成实验 |
|---|---|---|
| 逐信号模型 + 设备聚合 | 每颗卫星/频段信号使用 7 维 Raw 特征的因果序列，逐信号预测后用投票规则产生设备告警 | SignalLSTM `L=5` + majority |
| 直接设备级模型 | 同一设备、同一历元的有效卫星聚合为 27 维统计特征，再对设备级因果序列直接分类 | MLP、GRU、LSTM、TCN、Depthwise CNN、NLinear、DLinear、TSMixer、LightGBM |

直接设备级特征固定为 27 维：6 个连续 Raw 特征各取 `median/std/P10/P90`（24 维），再加有效卫星数、L1 信号比例、L5 信号比例（3 维）。连续特征为 `Cn0DbHz`、`Cn0DbHz_dt`、`Cn0DbHz_std`、`AgcDb`、`ReceivedSvTimeUncertaintyNanos`、`PseudorangeRateUncertaintyMetersPerSecond`。窗口 `L` 表示包含当前时刻在内的因果设备历元数。

## 实验协议总览

| ID | 任务和目的 | train / val / test 的数据组成 | 固定模型和窗口 | 结果用途 |
|---|---|---|---|---|
| P0 | 逐信号到设备告警基线 | 新主楼静态 train；操场静态 val/test | SignalLSTM `L=5` + majority | 历史开发参考 |
| P1 | 静态跨环境模型/窗口扫描 | 新主楼静态 4 / 操场静态 2 / 操场静态 2 个 Session | 多个直接设备级模型，`L=5/15/30/64` | 历史开发参考，test 已用过 |
| P2a | 新主楼同环境静态消融 | 全部为新主楼静态；2 / 1 / 1 个 Session | LightGBM `L=30` | 单次静态 Session-holdout |
| P2b | 操场同环境静态消融 | 全部为操场静态；2 / 1 / 1 个 Session | LightGBM `L=30` | 单次静态 Session-holdout |
| P3 | 静态多环境 4 折 CV | 每折静态 train 4、val 2、test 2 个 Session，均含两环境 | LightGBM `L=30` | 当前较稳健的静态开发评估 |
| P4 | 动态训练增广的静态 4 折 CV | P3 的静态 split 不变；每折仅在 train 额外加入 10 个动态 Session | LightGBM `L=30` | 检验动态训练是否帮助静态 test |
| P5 | 统一静态+动态检测 | train、val、test 均同时含静态和动态 Session | LightGBM `L=30` | 当前统一任务首个固定划分基线 |

## P0：逐信号模型加设备聚合

| 数据划分 | 模型 / 输入 | 聚合规则 | test Macro-F1 | Recall | FAR |
|---|---|---|---:|---:|---:|
| 与 P1 相同的静态跨环境划分 | SignalLSTM，7 维逐信号特征，`L=5` | majority | 0.6754 | 32.81% | 0.7754% |

该实验说明直接做设备级建模比“逐卫星分类后多数投票”更适合当前设备告警目标。`any`、`k_of_n`、`ratio` 规则仅在 val 做过比较；P0 的 majority test 结果已经属于历史开发结果。

## P1：静态跨环境开发参考的模型和窗口扫描

### 精确 Session 划分

| split | 环境 | 静态 Session（场景） | Session 数 |
|---|---|---|---:|
| train | 新主楼 | `2025.07.29.19.22` (`st_L1`)、`2025.07.29.20.16` (`st_L5`)、`2025.07.29.20.36` (`st_L5`)、`2025.07.29.18.42` (`st_L_15`) | 4 |
| val | 操场 | `2025.07.30.08.40-09.12` (`st_L1`)、`2025.07.30.09.41-09.45` (`st_L5`) | 2 |
| test | 操场 | `2025.07.30.09.48-10.14` (`st_L5`)、`2025.07.30.07.30-08.01` (`st_L_15`) | 2 |

也就是说，P1 同时改变了环境、设备组成、频段攻击形式和录制时间段。它真实地暴露了迁移困难，但不能把分数单独归因于“环境”。完整机器可读清单位于本地 `output/tensors_static_cross_env/recording_split_manifest.csv`，不提交到仓库。

### 设备级模型结果

以下均使用同一组 27 维统计特征和阈值 0.5。不同 `L` 的序列前缀会被丢弃，因此样本数不同，这是正常现象。所有 test 均已参与模型筛选，只可作开发参考。

| 模型 | L | 参数量 / 文件大小 | test Macro-F1 | Precision | Recall | FAR |
|---|---:|---|---:|---:|---:|---:|
| DeviceLightGBM | 30 | 121.6 KB | **0.9161** | 99.22% | **79.08%** | 0.2912% |
| DeviceLightGBM | 64 | 171.3 KB | 0.9134 | 99.69% | 78.24% | 0.1168% |
| DeviceStatsDLinear | 30 | 1,540 parameters | 0.8342 | 99.29% | 60.68% | 0.2022% |
| DeviceStatsMLP | 5 | 3,584 parameters | 0.8249 | 99.97% | 58.34% | 0.0079% |
| DeviceStatsTCN | 5 | 3,818 parameters | 0.8218 | 99.64% | 57.86% | 0.0950% |
| DeviceStatsLSTM | 30 | 5,186 parameters | 0.8185 | 99.82% | 57.26% | 0.0485% |
| DeviceStatsMLP | 30 | 21,134 parameters | 0.7991 | 98.63% | 53.89% | 0.3478% |
| DeviceStatsLSTM | 5 | 5,186 parameters | 0.7887 | 99.76% | 51.34% | 0.0554% |
| DeviceStatsTCN | 30 | 3,818 parameters | 0.7871 | 99.73% | 51.20% | 0.0647% |
| DeviceStatsTCN | 15 | 3,818 parameters | 0.7869 | 99.33% | 51.20% | 0.1596% |
| DeviceStatsDLinear | 64 | 1,540 parameters | 0.7863 | 98.70% | 51.61% | 0.3253% |
| DeviceStatsDepthwiseCNN | 30 | 1,562 parameters | 0.7769 | 98.93% | 49.55% | 0.2507% |
| DeviceStatsTSMixer | 30 | 6,740 parameters | 0.7697 | 99.18% | 48.13% | 0.1860% |
| DeviceStatsDepthwiseCNN | 15 | 1,562 parameters | 0.7668 | 99.24% | 47.50% | 0.1676% |
| DeviceStatsMLP | 15 | 10,604 parameters | 0.7549 | 99.51% | 45.28% | 0.1038% |
| DeviceStatsGRU | 5 | 3,914 parameters | 0.7144 | 99.91% | 38.20% | 0.0158% |
| DeviceStatsNLinear | 30 | not recorded | 0.6172 | 49.99% | 42.53% | 19.8253% |

**P1 的正确结论：** 在当前 27 维统计特征下，LightGBM `L=30` 是最强的工程强基线；DLinear `L=30` 是已测神经网络中的最佳轻量候选。不能据此说它们已是最终模型，下一轮须使用 P3/P4 或新的未见协议复验。

## P2：同环境静态 Session-holdout

P2 的目标是隔离“同一环境中，动态训练数据是否会改善静态未见 Session”。每个环境有两组可直接比较的实验：static-only 与 dynamic-train-augmentation。两组的静态 val/test 完全相同，差别仅在 train 是否追加动态 Session。

| 环境 | 静态 train | 静态 val | 静态 test | 动态增广 train | static-only test F1 / Recall / FAR | 增广后 test F1 / Recall / FAR |
|---|---|---|---|---|---|---|
| 新主楼 | `st_L5 20.16` + `st_L_15 18.42` | `st_L5 20.36` | `st_L1 19.22` | `dy_L1 20.04` + `dy_L5 20.52` + `dy_L_15 18.33` | 0.9689 / 99.80% / 4.15% | **0.9838 / 99.88% / 2.13%** |
| 操场 | `st_L5 09.41` + `st_L_15 07.30` | `st_L5 09.48` | `st_L1 08.40` | 7 个操场动态 train Session，覆盖 `dy_L1/L5/L_15` | 0.7668 / 48.89% / 0.00% | **0.9944 / 98.87% / 0.17%** |

固定配置：DeviceLightGBM、27 维设备统计特征、`L=30`、阈值 0.5。协议清单分别见 `docs/protocols/static_within_*_v1/` 和 `docs/protocols/static_dynamic_train_*_v1/`。

操场 static-only 差，并不说明“纯静态一定不能训练”：其 train 中带正样本的场景只有双频 `st_L_15`，而 val 是 L5-only、test 是 L1-only；频段攻击覆盖不足使模型在 val 上选得非常保守。动态训练补足了 L1/L5/双频攻击形式，所以测试结果大幅上升。P2 是有价值的消融，但每个环境只有一次固定 holdout，不能替代 P3 的 4 折结果。

## P3/P4：多环境静态 4 折 Session-CV

### 固定划分

源数据为 8 个静态 Session：新主楼 4 个、操场 4 个。每一折均采用：

| split | 新主楼静态 Session | 操场静态 Session | 总数 |
|---|---:|---:|---:|
| train | 2 | 2 | 4 |
| val | 1 | 1 | 2 |
| test | 1 | 1 | 2 |

四折结束后，每一个完整静态 Session 恰好：1 次进入 test、1 次进入 val、2 次进入 train。P4 在每折 P3 的 train 中固定追加 10 个动态 Session；静态 val/test 一字不动。逐折精确清单位于 `docs/protocols/static_session_cv_4fold/fold_1..fold_4/recording_split_manifest.csv`；P4 清单位于 `docs/protocols/static_dynamic_train_cv_4fold/`。

### 结果：同一 LightGBM L=30 的可比静态消融

| 训练协议 | fold 1 | fold 2 | fold 3 | fold 4 | test Macro-F1 mean +/- SD | Recall mean +/- SD | FAR mean +/- SD |
|---|---:|---:|---:|---:|---:|---:|---:|
| P3: multi-env static-only | 0.8401 | 0.8371 | 0.9309 | 0.9772 | 0.8964 +/- 0.0693 | 78.71% +/- 21.92% | 2.37% +/- 4.05% |
| P4: static + dynamic train augmentation | 0.8907 | 0.8392 | 0.9478 | 0.9900 | **0.9169 +/- 0.0659** | **82.69% +/- 20.14%** | **1.96% +/- 2.93%** |

固定配置：DeviceLightGBM、27 维设备统计特征、`L=30`、阈值 0.5。P4 相对于 P3 的平均变化为 F1 `+0.0206`、Recall `+3.98` 个百分点、FAR `-0.41` 个百分点。结论是：在这 4 个静态 test 折上，动态数据作为**训练分布扩增**是有益的；它仍不能证明动态 test 的性能。

## P5：统一静态+动态检测

P5 才是“一个模型既识别静态欺骗、也识别动态欺骗”的当前基线。它使用既有 mixed Session manifest，静态和动态均进入每个 split，且完整 Session 严格隔离。

| split | 静态 Session | 动态 Session | 环境组成 | 总 Session |
|---|---:|---:|---|---:|
| train | 6 | 10 | 新主楼：3 static + 3 dynamic；操场：3 static + 7 dynamic | 16 |
| val | 1 | 4 | 新主楼：1 static + 1 dynamic；操场：3 dynamic | 5 |
| test | 1 | 4 | 新主楼：1 dynamic；操场：1 static + 3 dynamic | 5 |

固定配置：DeviceLightGBM、27 维设备统计特征、`L=30`、阈值 0.5。注意 P5 的静态 test 只含一个操场 `st_L5` Session，动态 test 包含新主楼和操场的动态 Session；因此它是**混合任务固定划分基线**，尚不是混合任务的多折稳健结论。

| P5 test 子集 | 样本窗口数 | Macro-F1 | Precision | Recall | FAR |
|---|---:|---:|---:|---:|---:|
| overall static + dynamic | 11,943 | 0.7944 | 97.30% | 51.33% | 0.51% |
| static subset | 8,159 | 0.8118 | 98.63% | 54.75% | 0.29% |
| dynamic subset | 3,784 | 0.7500 | 95.67% | 43.00% | 0.98% |

P5 的关键发现：动态 Recall 低于静态；且 L5-only 静态攻击在不同设备上差异大。当前 27 维特征是跨 L1/L5 的全局统计量，未受攻击的 L1 信号会稀释 L5-only 异常，是下一轮应优先改进的特征缺口。

## 当前可以怎么说，不能怎么说

| 可以确认的结论 | 不能作出的结论 |
|---|---|
| 当前 27 维设备统计输入下，LightGBM `L=30` 是最佳已测工程强基线。 | P1 的 0.9161 不是最终独立测试性能。 |
| DLinear `L=30` 是已测神经网络中 F1/Recall 最好的轻量候选。 | 不能因为 DLinear 参数少，就认定它已优于 LightGBM 或适合直接部署。 |
| P3/P4 表明，动态 Session 作为训练增广提高了静态 4 折平均 Recall 和 F1。 | P4 不等于模型能在动态 test 上达到相同性能。 |
| P5 已证明当前流程可用一个模型同时输出静态、动态设备告警。 | P5 的 0.7944 不能与 P3/P4 的纯静态 F1 做性能高低比较。 |
| 当前主要风险来自跨环境、设备、频段和 Session 时段的分布变化。 | 不能把所有 test 下降简单归因为“模型容量不够”。 |

## 当时拟议但未锁定的后续路线

以下内容保留用于理解 P0–P5 当时的推进思路，不是 2026-07-22 之后的当前计划；当前建议以 [handoff_status.md](handoff_status.md) 为准。

1. 将每个基础特征拆为 L1 与 L5 独立统计量，并加入相对滚动基线、短期变化量等因果特征，优先解决 L5-only 信号被 L1 稀释的问题。
2. 用**锁定的新特征版本**在 P3/P4 的四折清单上复验 LightGBM `L=30`、DLinear `L=30`、MLP `L=5`；只报告同一协议的均值和标准差。
3. 为 P5 构建静态与动态均进入 train/val/test 的多折 Session-CV，并分别报告 overall、static、dynamic 指标。
4. 在上述协议锁定后再做 TTD、每设备指标、模型大小、CPU/端侧推理时延与蒸馏实验。

## 相关文件

- [当前交接状态](handoff_status.md)：当前数据问题、逐 signal 探索与后续建议。
- [静态四折 Session-CV 规则](static_session_cv_protocol.md)：P3 的复现与防泄漏规则。
- `pipeline_total/16_collect_device_experiment_results.py`：从本地 `output/training/` 自动汇总已有指标，生成的 CSV 不提交。

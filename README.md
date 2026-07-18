# 面向真实多设备部署的轻量化 GNSS 导航欺骗检测方法研究

本仓库用于推进 **面向真实多设备部署的轻量化 GNSS 导航欺骗检测方法研究**。

项目目标不是追求复杂大模型，而是面向后续真实部署，构建一个只依赖手机、手表、u-blox 等真实设备可获得 GNSS Raw 少量特征的轻量化导航欺骗检测流程，并系统验证其跨环境、跨设备、跨频段和部署性能。

## 当前状态（2026-07-18）

本项目已从“数据整理与标签确认”进入“设备级模型开发与跨域泛化改进”阶段：

- 已统一处理操场与新主楼的 **132 份**原始 GNSS TXT 日志，生成 **2,139,284 条已审查的信号级记录**；
- 新主楼静态、动态场景已按独立 TOW 区间完成标签复核；删除了 1 份与 `st_L1` 重复的新主楼原始日志，并同步重建数据清单；
- 默认逐信号输入固定为 7 项真实设备可获得的 GNSS Raw 特征，窗口标签语义已修正为“当前端点历元”，避免攻击结束后的标签滞后；
- 已完成逐卫星模型、设备级聚合报警和直接设备级模型的第一轮比较；当前采用 27 维多卫星统计特征直接输出设备告警；
- 静态跨环境开发参考结果中，`LightGBM L=30` 当前最优（Macro-F1 `0.9161`、Recall `0.7908`、FAR `0.2912%`）；`DLinear L=30` 是当前最强轻量神经网络候选（1,540 参数、Macro-F1 `0.8342`、Recall `0.6068`）；
- 已完成 4 折静态多环境 Session-CV：在“动态仅加入训练、静态测试”的消融中，`LightGBM L=30` 从 Macro-F1 `0.8964 +/- 0.0693` 提升至 `0.9169 +/- 0.0659`；另已建立静态/动态均进入 train、val、test 的统一设备级检测基线，详细分组结果见实验进展文档。

> **结果边界：** 当前 `static_cross_env` 的操场 test 已用于模型、窗口和错误样本诊断，只能作为开发参考，不再是论文最终独立测试集。先查阅 [实验台账](docs/experiment_registry.md) 了解每项结果的模型、窗口、环境和 Session 划分；过程记录见 [docs/experiment_progress.md](docs/experiment_progress.md)。

## 项目主线

本研究的核心目标是：

> 利用操场与新主楼两套真实多设备导航欺骗数据，构建适合真实设备部署的轻量化 GNSS 导航欺骗检测模型。

重点关注：

- **真实设备少特征**：不依赖软件接收机内部富特征；
- **轻量化模型**：优先考虑低参数量、低 FLOPs、低延迟和小模型体积；
- **跨域验证**：重点验证跨环境、跨设备、跨频段泛化；
- **部署指标**：除 Accuracy/F1 外，报告 TTD、FAR、推理延迟、模型大小等指标。

## 仓库内容

```text
configs/          配置模板
data_raw/         本地原始 GNSS TXT（Git 忽略）
data_csv/         逐日志提取 CSV（Git 忽略）
pipeline_total/   数据处理、画图、标注、训练、推理全流程脚本
docs/             项目主线、数据清单、实验计划
README.md         项目说明
CONTRIBUTING.md   组内 GitHub 协作说明
.gitignore        忽略大数据和生成结果
```

当前仓库只保存代码、配置和文档，不保存大体量原始数据。

## 数据目录

主数据已统一放在本地：

```text
H:\GNSS\lightweight_gnss_spoofing_detection
```

推荐目录结构：

```text
lightweight_gnss_spoofing_detection
├─ data_raw
│  ├─ playground
│  │  ├─ st_L1
│  │  ├─ st_L5
│  │  ├─ st_L_15
│  │  ├─ dy_L1
│  │  ├─ dy_L5
│  │  └─ dy_L_15
│  └─ new_building
│     ├─ st_L1
│     ├─ st_L5
│     ├─ st_L_15
│     ├─ dy_L1
│     ├─ dy_L5
│     └─ dy_L_15
├─ pipeline_total
├─ configs
└─ output
```

数据使用策略：

| 数据角色 | 本地路径 | 用途 |
|---|---|---|
| 主数据 | `H:\GNSS\lightweight_gnss_spoofing_detection\data_raw\playground` | 操场导航欺骗数据，作为真实环境之一 |
| 主数据 | `H:\GNSS\lightweight_gnss_spoofing_detection\data_raw\new_building` | 新主楼导航欺骗数据，用于跨环境验证 |
| 辅助数据 | `H:\GNSS\Finland L1_E1 data\final_mat` | 富特征软件接收机数据，可用于强基线、Teacher 或对照分析 |
| 暂不纳入主线 | `H:\GNSS\Interference Data` | 标签可信度暂不确定，暂不进入主实验 |

> 注意：原始数据、处理后大 CSV、NPZ 张量、模型权重等不要直接提交到 GitHub。

## 特征提取约定

本项目的正式数据处理流程统一从原始 GNSS 日志 `.txt` 文件中重新解析所需字段，不直接依赖历史生成的 `raw.csv`、`raw_sort.csv`、`plot_features.csv` 或 `features_enhanced.csv`。这些历史 CSV 可以作为参考或检查用，但不作为正式实验数据来源。

统一处理流程如下：

```text
原始 GNSS txt 日志
  -> 解析 Raw 观测字段
  -> 计算 TOW / sv_id / SignalBand / signal_id
  -> 从目录结构补充 Environment / Scenario / DeviceName / SpoofingType
  -> 按独立 signal_id 计算 C/N0 差分和滑窗统计特征
  -> 根据人工确认的 TOW 欺骗区间生成 Label
  -> 输出统一 processed_gnss_data.csv
```

### 最终保留字段

正式输出的统一数据表建议保留以下字段：

```text
TimeNanos
TOW
utcTimeMillis
Environment
Scenario
Session
DeviceName
ConstellationType
Svid
sv_id
FreqBand
CarrierFrequencyHz
CodeType
SignalBand
signal_id
SignalEpochCount
SpoofingType
Label
LabelStatus
LabelSource
AgcDbMissing
Cn0DbHz
Cn0DbHz_dt
Cn0DbHz_std
AgcDb
ReceivedSvTimeUncertaintyNanos
PseudorangeRateUncertaintyMetersPerSecond
FreqBand
```

其中，真正作为模型输入的核心特征为：

```text
Cn0DbHz
Cn0DbHz_dt
Cn0DbHz_std
AgcDb
ReceivedSvTimeUncertaintyNanos
PseudorangeRateUncertaintyMetersPerSecond
FreqBand
```

其他字段用于排序、分组、标签生成、跨环境/跨设备实验划分和结果分析，不建议直接作为模型输入。

### 字段来源说明

| 字段 | 来源 | 说明 |
|---|---|---|
| `TimeNanos` | 原始 txt 中 Raw 行自带 | 接收机内部 GNSS 时钟时间，单位 ns，用于时间计算和排序 |
| `utcTimeMillis` | 原始 txt 中 Raw 行自带 | 手机记录该观测时的 UTC 时间，单位 ms，用于追溯和对齐 |
| `TOW` | 脚本计算 | 根据 `TimeNanos`、`FullBiasNanos`、`BiasNanos`、`ReceivedSvTimeNanos` 等字段计算 |
| `Environment` | 路径补充 | 由目录判断，例如 `playground` 或 `new_building` |
| `Scenario` | 路径补充 | 由目录判断，例如 `st_L1`、`dy_L5`、`st_L_15` |
| `DeviceName` | 日志头解析 + 路径兜底 | 优先从日志设备型号解析，解析失败时使用设备文件夹名 |
| `sv_id` | 脚本计算 | 由 `ConstellationType` 和 `Svid` 组合生成，例如 `G29`、`E12` |
| `SignalBand` | 脚本计算 | 由星座和载频标准化得到，例如 `BDS_B1I`、`GPS_L5`、`GAL_E1` |
| `signal_id` | 脚本计算 | `sv_id + SignalBand + CodeType`，表示独立时序信号，是差分、拆分和张量槽位主键 |
| `SignalEpochCount` | 脚本计算 | 同一 `signal_id + TimeNanos` 的原始观测数；大于 1 时预处理会先聚合，避免静默覆盖 |
| `FreqBand` | 脚本计算 | 由 `CarrierFrequencyHz` 判断 L1/L5 |
| `SpoofingType` | 路径/规则生成 | 第一版可直接设为 `Scenario` |
| `Label` | 标签配置生成 | 根据人工确认的欺骗 TOW 区间生成，正常为 0，欺骗为 1 |
| `LabelStatus` / `LabelSource` | 脚本配置生成 | 标识标签是否已审查及其来源；默认训练只使用 `reviewed` 行 |
| `Cn0DbHz` | 原始 txt 中 Raw 行自带 | 载噪比 C/N0，反映卫星信号质量 |
| `Cn0DbHz_dt` | 脚本计算 | 同一 `signal_id` 相邻时刻 C/N0 的差分，用于捕捉突变 |
| `Cn0DbHz_std` | 脚本计算 | 同一 `signal_id` 滑动窗口内 C/N0 标准差，用于描述短时波动 |
| `AgcDb` | 原始 txt 中 Raw 行通常自带 | 自动增益控制值，反映接收机前端对输入信号强弱的调节 |
| `ReceivedSvTimeUncertaintyNanos` | 原始 txt 中 Raw 行自带 | 卫星发射时间估计不确定度，单位 ns |
| `PseudorangeRateUncertaintyMetersPerSecond` | 原始 txt 中 Raw 行自带 | 伪距率不确定度，单位 m/s |
| `AccumulatedDeltaRangeUncertaintyMeters` | 原始 txt 中 Raw 行自带 | ADR 累计增量距离不确定度，单位 m |

### 时间字段使用原则

`TimeNanos`、`utcTimeMillis` 和 `TOW` 需要保留，但不建议直接作为模型输入特征。

它们主要用于：

```text
数据排序
时间对齐
TOW 标签区间匹配
滑窗切片
差分/统计特征计算
TTD 检测时间统计
实验结果追溯
```

如果直接把绝对时间字段作为模型输入，模型可能学到“某个时间段对应欺骗”，而不是学习 GNSS 信号本身的异常模式。这会削弱跨环境、跨设备和未来部署时的泛化能力。

### 标签生成原则

`Label` 不从原始 txt 直接读取，而是根据人工确认的欺骗 TOW 区间生成。

建议标签规则为：

```text
Label = 0：正常
Label = 1：导航欺骗
```

对于不同场景：

```text
st_L1 / dy_L1：只对 L1 频段观测打欺骗标签
st_L5 / dy_L5：只对 L5 频段观测打欺骗标签
st_L_15 / dy_L_15：L1 和 L5 均可打欺骗标签
```

标签区间必须经过可视化复核，不能只依赖旧配置或历史 CSV。

新主楼的 Session 尚未录入经人工确认的区间时会标记为 `needs_review`，默认不进入张量构建和正式训练。信号级数据模型、重建与拆分命令见 [docs/signal_level_feature_extraction.md](docs/signal_level_feature_extraction.md)。

### 缺失值注意事项

不同设备和日志版本的 Raw 字段可能存在差异。虽然核心字段在大多数日志中都能解析到，但仍需要统计缺失情况，尤其是：

```text
AgcDb
ReceivedSvTimeUncertaintyNanos
PseudorangeRateUncertaintyMetersPerSecond
AccumulatedDeltaRangeUncertaintyMeters
```

正式训练前应检查每个设备、每个环境、每个场景的字段缺失率。若某个字段在某类设备中大量缺失，需要统一处理策略，例如：

```text
剔除该设备/日志
使用缺失值标记
使用合理填充值
单独做无该字段的消融实验
```

### 第一版核心特征集

第一版模型输入固定为 7 个核心特征：

```text
Cn0DbHz
Cn0DbHz_dt
Cn0DbHz_std
AgcDb
ReceivedSvTimeUncertaintyNanos
PseudorangeRateUncertaintyMetersPerSecond
FreqBand
```

`AccumulatedDeltaRangeUncertaintyMeters` 保留在 CSV、审计和可视化中，但因人工标注阶段未表现出稳定判别价值，不进入默认模型输入。它可在后续消融实验中单独验证。其余特征数量少、来源清晰、真实设备可获得，符合轻量化部署主线。

## 生成数据清单

数据清单用于记录每个原始 GNSS 日志的环境、场景、设备、相对路径、当前 `data_csv` 提取状态和标签审查状态。生成命令：

```bash
python scripts/build_data_manifest.py
```

**何时运行：** 新增、删除或移动原始 TXT，或变更 `configs/preprocessing.yml` 中的标签审查状态后。

**为什么运行：** 清单从当前 `data_raw/` 和标签配置重建，路径使用相对 `data_raw/` 的形式，不依赖某台电脑的盘符；不要把旧清单当作当前数据状态。

默认会扫描：

```text
data_raw/
```

并输出：

```text
docs/data_manifest.csv
```

当前清单统计：

```text
raw logs: 132
playground: 98
new_building: 34
```

当前共 132 份原始日志。`label_status` 由 `configs/preprocessing.yml` 解析；当前纳入主实验的操场与新主楼 Session 已为 `reviewed`。以后新增或未复核的 Session 会自动标记为 `needs_review`，不会进入训练张量。

## CSV 训练准入审计

在构建训练窗口或训练模型前，必须先审计逐日志 CSV。审计只读取
`data_csv/`，不会修改原始 CSV：

```bash
python scripts/audit_extracted_csv.py \
  --input-dir data_csv \
  --output-json output/data_csv_audit.json
```

审计报告包含：

```text
CSV 文件数和观测总行数
字段完整性
Label、Environment、Scenario、DeviceName、FreqBand 分布
设备 x 标签、设备 x 频段覆盖
核心特征缺失率
```

随后生成 session 级清单。每个逐日志 CSV 是后续训练/验证/测试切分的
最小单位，不允许把同一录制的相邻数据切到不同集合：

```bash
python scripts/build_csv_session_manifest.py \
  --input-dir data_csv \
  --output-csv docs/data_csv_session_manifest.csv
```

**何时运行：** 镜像 CSV 重建、标签配置变更或删除某个原始日志后，并且在构建张量前。

**为什么运行：** 清单从 CSV 内的 `Session`、`LabelStatus`、`LabelSource` 读取元数据，排除 `_by_signal_id` 和 `_by_sv_id` 派生拆分文件；正常但已审查的会话会正确保留为 `reviewed`，不会因正样本数为 0 被误记为 `needs_review`。

当前 CSV 审计结论：

```text
new_building 的静态与动态 Session 均已按独立 TOW 区间完成人工复核；正式跨环境实验前仍应先运行一次全量 CSV 审计
Google_Pixel_Watch1 的 AgcDb 全缺失，后续需要保留缺失标记并做 no-AGC 消融
Cn0DbHz_dt / Cn0DbHz_std 已按独立 signal_id 重建，训练和绘图均应继续使用信号级 CSV
```

在完成全量 CSV 审计并确认训练/验证/测试按 Session 隔离前，不得开始正式跨环境模型训练。

## 当前推荐流程

### 已完成

1. 原始日志清单、信号级特征提取与 TOW 标签复核。
2. 统一 `processed_gnss_data.csv`、Session 隔离张量和逐卫星轻量 baseline。
3. 设备级真值语义、逐卫星聚合报警、错分样本回溯与可视化。
4. 直接设备级 27 维统计张量，以及 MLP/GRU/LSTM/TCN/Depthwise CNN/NLinear/DLinear/TSMixer/LightGBM 的第一轮窗口比较。

### 当前进行中

1. 基于 `LightGBM L=30` 导出特征贡献，识别对跨环境结果影响最大的统计量和历史时刻。
2. 构造相对历史基线、变化率和短期波动等因果特征，削弱 C/N0、AGC 的设备/环境绝对基线差异。
3. 在锁定的新特征协议上复验 4 折静态 Session-CV，并将 `DLinear L=30`、`MLP L=5` 与当前 LightGBM 基线并列报告。

### 后续评估

1. 跨环境、跨设备、跨频段与静态/动态迁移。
2. TTD、虚警频率、模型大小、CPU/端侧推理时延。
3. 仅当长窗口教师模型明显优于轻量候选时，再开展 RS-TimesNet/CWT-ConvNeXt 到神经网络学生的蒸馏。

## 重点实验

建议优先完成以下实验：

```text
同环境随机划分
playground 训练 -> new_building 测试
new_building 训练 -> playground 测试
leave-one-device-out
leave-one-frequency-out
static -> dynamic
dynamic -> static
特征消融
模型复杂度和推理延迟评估
TTD 检测时间评估
```

## 当前优先任务

当前优先解决跨环境泛化，而不是继续机械增加模型：先完成特征贡献分析与因果相对特征重建，再在既有 4 折 Session-CV 上复验 `DLinear L=30`、低虚警 `MLP L=5` 和新特征 LightGBM，并使用保留动态 Session 开展静态到动态迁移测试。

## 组内协作

如果要参与代码、配置、标签或实验结果整理，请先阅读：

- `CONTRIBUTING.md`：组内 GitHub 协作说明

## 文档入口

更详细的项目说明见：

- `docs/project_mainline.md`：项目主线与研究定位
- `docs/data_inventory.md`：数据来源与使用策略
- `docs/experiment_plan.md`：实验矩阵与评价指标
- `docs/experiment_progress.md`：当前模型结果、结论、已知边界与下一步
- `docs/dynamic_labeling_assistant.md`：动态场景短时欺骗候选区间的生成与人工复核
- `docs/model_training_framework.md`：张量接口、轻量 baseline、训练测试边界与模型扩展规范
- `docs/static_session_cv_protocol.md`：4 折静态 Session-CV 的锁定划分与复现实验规则
- `docs/experiment_registry.md`：当前所有已完成实验的任务边界、模型、窗口、Session 划分与结果总表
- `pipeline_total/README.md`：全流程脚本顺序说明

## 当前原则

本项目后续应始终围绕一句话展开：

> 不是做最大最复杂的 GNSS 欺骗检测模型，而是做一个能在真实多设备、多环境中稳定工作的轻量化、可部署 GNSS 导航欺骗检测框架。



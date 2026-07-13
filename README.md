# 面向真实多设备部署的轻量化 GNSS 导航欺骗检测方法研究

本仓库用于推进 **面向真实多设备部署的轻量化 GNSS 导航欺骗检测方法研究**。

项目目标不是追求复杂大模型，而是面向后续真实部署，构建一个只依赖手机、手表、u-blox 等真实设备可获得 GNSS Raw 少量特征的轻量化导航欺骗检测流程，并系统验证其跨环境、跨设备、跨频段和部署性能。

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
AccumulatedDeltaRangeUncertaintyMeters
```

其中，真正作为模型输入的核心特征为：

```text
Cn0DbHz
Cn0DbHz_dt
Cn0DbHz_std
AgcDb
ReceivedSvTimeUncertaintyNanos
PseudorangeRateUncertaintyMetersPerSecond
AccumulatedDeltaRangeUncertaintyMeters
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

第一版模型输入固定为 8 个核心特征：

```text
Cn0DbHz
Cn0DbHz_dt
Cn0DbHz_std
AgcDb
ReceivedSvTimeUncertaintyNanos
PseudorangeRateUncertaintyMetersPerSecond
AccumulatedDeltaRangeUncertaintyMeters
FreqBand
```

这组特征数量少、来源清晰、真实设备可获得，符合轻量化部署主线。后续如果需要扩展特征，应通过消融实验验证其贡献，避免盲目堆叠特征。

## 生成数据清单

数据清单用于记录每个原始 GNSS 日志的环境、场景、设备、文件路径和处理状态。生成命令：

```bash
python scripts/build_data_manifest.py
```

默认会扫描：

```text
H:\GNSS\lightweight_gnss_spoofing_detection\data_raw
```

并输出：

```text
docs/data_manifest.csv
```

当前清单统计：

```text
raw logs: 133
playground: 98
new_building: 35
```

`label_status` 初始均为 `unreviewed`，后续人工看图确认标签后再更新。

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

当前 CSV 审计结论：

```text
new_building 的 35 个 session 暂无欺骗正样本，不能参与正式跨环境检测率评估
Google_Pixel_Watch1 的 AgcDb 全缺失，后续需要保留缺失标记并做 no-AGC 消融
现有 Cn0DbHz_dt / Cn0DbHz_std 仍按 sv_id 计算，需先改为按独立 signal_id 重建
```

在完成新主楼标签补充和 signal_id 特征重建前，不得开始正式模型训练。

## 推荐流程

1. 检查仓库根目录下 `data_raw/` 与 `data_csv/` 的目录结构是否规范。
2. 建立或更新 `data_manifest.csv`，记录每个日志的环境、场景、设备、文件名和标注状态。
3. 使用 pipeline 生成每个日志对应的 `*-plot_features.csv`。
4. 可视化 `Cn0DbHz`、`AgcDb`、uncertainty 等特征。
5. 手工复核欺骗发生的 TOW 区间。
6. 生成统一的 `processed_gnss_data.csv`，并保留 `Environment` 字段。
7. 构建 train / validation / test 张量。
8. 先训练轻量 baseline。
9. 做跨环境、跨设备、跨频段实验。
10. 统计 Accuracy、Macro-F1、FAR、TTD、参数量、FLOPs、模型大小、推理延迟等指标。
11. 在主流程跑通后，再考虑知识蒸馏和部署优化。

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

近期请优先完成：

1. 整理仓库内 `data_raw/playground` 与 `data_raw/new_building` 的数据结构。
2. 生成 `data_manifest.csv`。
3. 生成两套数据的 `*-plot_features.csv`。
4. 看图复核 TOW 标签。
5. 形成统一标签配置。
6. 生成统一 `processed_gnss_data.csv`。

在数据和标签没有确认前，不建议急着训练模型。

## 组内协作

如果要参与代码、配置、标签或实验结果整理，请先阅读：

- `CONTRIBUTING.md`：组内 GitHub 协作说明

## 文档入口

更详细的项目说明见：

- `docs/project_mainline.md`：项目主线与研究定位
- `docs/data_inventory.md`：数据来源与使用策略
- `docs/experiment_plan.md`：实验矩阵与评价指标
- `pipeline_total/README.md`：全流程脚本顺序说明

## 当前原则

本项目后续应始终围绕一句话展开：

> 不是做最大最复杂的 GNSS 欺骗检测模型，而是做一个能在真实多设备、多环境中稳定工作的轻量化、可部署 GNSS 导航欺骗检测框架。



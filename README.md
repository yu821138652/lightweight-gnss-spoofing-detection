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
H:\GNSS\real_world_spoofing_dataset_pipeline
```

推荐目录结构：

```text
real_world_spoofing_dataset_pipeline
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
| 主数据 | `H:\GNSS\real_world_spoofing_dataset_pipeline\data_raw\playground` | 操场导航欺骗数据，作为真实环境之一 |
| 主数据 | `H:\GNSS\real_world_spoofing_dataset_pipeline\data_raw\new_building` | 新主楼导航欺骗数据，用于跨环境验证 |
| 辅助数据 | `H:\GNSS\Finland L1_E1 data\final_mat` | 富特征软件接收机数据，可用于强基线、Teacher 或对照分析 |
| 暂不纳入主线 | `H:\GNSS\Interference Data` | 标签可信度暂不确定，暂不进入主实验 |

> 注意：原始数据、处理后大 CSV、NPZ 张量、模型权重等不要直接提交到 GitHub。

## 核心特征

当前优先使用真实设备容易获得的 GNSS Raw 特征：

```text
Cn0DbHz
Cn0DbHz_dt
Cn0DbHz_std
AgcDb
ReceivedSvTimeUncertaintyNanos
PseudorangeRateUncertaintyMetersPerSecond
AccumulatedDeltaRangeUncertaintyMeters
FreqBand
sv_id
DeviceName
Environment
```

其中 `Environment` 用于区分：

```text
playground
new_building
```

后续跨环境实验必须依赖该字段。


## 生成数据清单

数据清单用于记录每个原始 GNSS 日志的环境、场景、设备、文件路径和处理状态。生成命令：

```bash
python scripts/build_data_manifest.py
```

默认会扫描：

```text
H:\GNSS\real_world_spoofing_dataset_pipeline\data_raw
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

## 推荐流程

1. 检查 `real_world_spoofing_dataset_pipeline` 的目录结构是否规范。
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

1. 整理 `real_world_spoofing_dataset_pipeline` 的 `playground/new_building` 数据结构。
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


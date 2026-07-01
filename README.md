# 面向真实多设备部署的轻量化 GNSS 导航欺骗检测方法研究

本仓库用于整理和推进 **面向真实多设备部署的轻量化 GNSS 导航欺骗检测方法研究**。

项目重点不是追求复杂大模型，而是围绕后续实际部署需求，构建一个只依赖真实设备可获得 GNSS Raw 少量特征的轻量化导航欺骗检测流程，并系统验证其跨环境、跨设备、跨频段和部署性能。

## 项目主线

本研究的核心目标是：

> 利用操场与新主楼两套真实多设备导航欺骗数据，构建适合手机、手表、u-blox 等真实设备部署的轻量化 GNSS 导航欺骗检测模型。

重点关注：

- 真实设备可获得特征，而不是软件接收机内部富特征；
- 轻量化模型，而不是单纯追求最高精度的大模型；
- 跨环境、跨设备、跨频段泛化；
- 参数量、FLOPs、模型大小、推理延迟、TTD 等部署指标。

## 仓库内容

```text
configs/          配置模板
pipeline_total/   数据处理、画图、标注、训练、推理全流程脚本
docs/             项目主线、数据清单、实验计划
README.md         项目说明
.gitignore        忽略大数据和生成结果
```

当前仓库只保存代码、配置和文档，不保存大体量原始数据。

## 数据来源

| 数据角色 | 本地路径 | 用途 |
|---|---|---|
| 主数据 1 | `H:\GNSS\data_raw` | 操场导航欺骗数据，作为主要真实环境之一 |
| 主数据 2 | `H:\GNSS\导航欺骗新主楼数据集及全流程处理脚本\导航欺骗新主楼数据集及全流程处理脚本\0729` | 新主楼导航欺骗数据，用于跨环境验证 |
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
```

这些特征兼顾了可部署性和检测有效性，适合用于轻量模型训练。

## 推荐流程

1. 统一操场和新主楼数据目录。
2. 使用 pipeline 生成每个日志对应的 `*-plot_features.csv`。
3. 可视化 `Cn0DbHz`、`AgcDb`、uncertainty 等特征。
4. 手工复核欺骗发生的 TOW 区间。
5. 生成统一的 `processed_gnss_data.csv`。
6. 构建 train / validation / test 张量。
7. 先训练轻量 baseline。
8. 做跨环境、跨设备、跨频段实验。
9. 统计 Accuracy、Macro-F1、FAR、TTD、参数量、FLOPs、模型大小、推理延迟等指标。
10. 在主流程跑通后，再考虑知识蒸馏和部署优化。

## 重点实验

建议优先完成以下实验：

```text
同环境随机划分
操场训练 -> 新主楼测试
新主楼训练 -> 操场测试
leave-one-device-out
leave-one-frequency-out
static -> dynamic
dynamic -> static
特征消融
模型复杂度和推理延迟评估
TTD 检测时间评估
```

## 文档入口

更详细的项目说明见：

- `docs/project_mainline.md`：项目主线与研究定位
- `docs/data_inventory.md`：数据来源与使用策略
- `docs/experiment_plan.md`：实验矩阵与评价指标
- `pipeline_total/README.md`：全流程脚本顺序说明

## 当前原则

本项目后续应始终围绕一句话展开：

> 不是做最大最复杂的 GNSS 欺骗检测模型，而是做一个能在真实多设备、多环境中稳定工作的轻量化、可部署 GNSS 导航欺骗检测框架。

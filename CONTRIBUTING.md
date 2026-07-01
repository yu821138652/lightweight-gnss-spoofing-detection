# 组内协作说明

本文档用于说明本项目在 GitHub 上的协作方式。请大家在提交代码、配置、实验结果或文档前先阅读本说明，避免误提交大数据、覆盖他人工作或造成实验结果难以复现。

## 1. 项目定位

本仓库服务于：

**面向真实多设备部署的轻量化 GNSS 导航欺骗检测方法研究**

当前主线聚焦：

- 真实多设备 GNSS 导航欺骗检测；
- 操场与新主楼两套真实环境数据；
- 轻量化模型与实际部署；
- 跨环境、跨设备、跨频段泛化；
- TTD、推理延迟、参数量、FLOPs、模型大小等部署指标。

暂不将 `Interference Data` 纳入主流程，避免标签不确定影响主实验结论。

## 2. 仓库里应该放什么

可以提交到 GitHub 的内容：

```text
代码脚本
配置模板
标签配置
小型统计结果
实验记录文档
README / Markdown 文档
数据清单 manifest
画图脚本
模型结构代码
```

不要提交到 GitHub 的内容：

```text
原始大数据
处理后大 CSV
NPZ / NPY 张量
模型权重 .pt / .pth / .onnx
压缩包
MAT 文件
临时输出 output/
个人环境文件 .env
虚拟环境 venv/
```

这些已经在 `.gitignore` 中尽量忽略。如果发现大文件即将被提交，请先停止并确认。

## 3. 数据放在哪里

大数据不放进 GitHub，统一放在本地或网盘中。

当前主数据建议结构：

```text
real_world_spoofing_dataset_pipeline
├─ data_raw
│  ├─ playground
│  └─ new_building
├─ pipeline_total
├─ configs
└─ output
```

本仓库只保存代码和说明。数据路径、样本数量、标签状态应记录在：

```text
docs/data_inventory.md
```

后续建议新增：

```text
docs/data_manifest.csv
```

用于记录每个日志文件的环境、场景、设备、是否已生成特征、是否已标注等状态。

## 4. 推荐协作流程

### 4.1 第一次参与项目

先克隆仓库：

```bash
git clone https://github.com/yu821138652/lightweight-gnss-spoofing-detection.git
cd lightweight-gnss-spoofing-detection
```

查看主线文档：

```text
README.md
docs/project_mainline.md
docs/experiment_plan.md
docs/data_inventory.md
```

### 4.2 开始一项新工作

不要直接在 `main` 分支上改。建议每个任务新建一个分支：

```bash
git checkout main
git pull
git checkout -b feature/your-task-name
```

分支命名建议：

```text
feature/data-manifest
feature/plot-features
feature/label-review
feature/baseline-models
feature/ttd-metrics
fix/preprocessing-path
 docs/update-experiment-plan
```

### 4.3 提交前检查

提交前先看状态：

```bash
git status
```

确认没有大数据和临时文件后再提交：

```bash
git add <需要提交的文件>
git commit -m "简短说明本次修改"
git push -u origin feature/your-task-name
```

然后在 GitHub 上创建 Pull Request，由组内其他同学检查后再合并。

## 5. Pull Request 要写清楚什么

每个 PR 建议说明：

```text
1. 这次改了什么
2. 为什么要改
3. 涉及哪些数据/场景/设备
4. 有没有改标签或配置
5. 是否跑过脚本或实验
6. 输出结果在哪里
7. 有什么还没完成
```

PR 标题建议示例：

```text
Add data manifest for playground and new-building logs
Update preprocessing config for new-building data
Add C/N0 and AGC plotting workflow
Add baseline model training script
```

## 6. 标签协作规范

标签是本项目最关键的部分，不能随意改。

如果修改欺骗区间，请说明：

```text
环境：playground / new_building
场景：st_L1 / st_L5 / st_L_15 / dy_L1 / dy_L5 / dy_L_15
设备：例如 HUAWEI、Xiaomi、Watch
依据：看了哪些 C/N0 / AGC / uncertainty 图
修改前区间
修改后区间
是否需要其他人复核
```

建议标签修改至少由另一位同学复核后再合并到 `main`。

## 7. 实验协作规范

实验结果需要可复现。提交实验相关修改时，请尽量记录：

```text
数据版本
配置文件
训练/测试划分方式
模型名称
特征组合
随机种子
主要指标
输出路径
```

建议把重要实验记录写入：

```text
docs/experiment_records/
```

后续可以按日期或实验编号记录，例如：

```text
docs/experiment_records/2026-07-xx_baseline_random_split.md
docs/experiment_records/2026-07-xx_cross_environment.md
```

## 8. 当前优先任务

近期建议优先完成：

1. 统一 `playground` 和 `new_building` 数据目录结构。
2. 生成或整理 `data_manifest.csv`。
3. 生成两套数据的 `*-plot_features.csv`。
4. 可视化 C/N0、AGC、uncertainty 特征。
5. 人工复核 TOW 标签。
6. 生成统一 `processed_gnss_data.csv`。
7. 跑轻量 baseline。
8. 做跨环境与跨设备实验。

## 9. 沟通建议

遇到以下情况请先在群里或 Issue 中说明，不要直接改主线：

- 想改标签区间；
- 想改变核心特征集合；
- 想纳入新的数据源；
- 想改变训练/测试划分协议；
- 发现数据路径、设备名、频段或标签有冲突；
- 准备提交较大文件或生成结果。

## 10. 一句话原则

本项目所有协作都围绕这一点：

> 做一个能在真实多设备、多环境中稳定工作的轻量化、可部署 GNSS 导航欺骗检测框架。

代码、标签、实验和文档都应服务于这个主线。

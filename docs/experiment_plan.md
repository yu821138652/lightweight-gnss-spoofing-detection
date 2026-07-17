# 面向真实多设备部署的轻量化 GNSS 导航欺骗检测方法研究：实验计划

## Main Task

面向真实多设备部署的轻量化 GNSS 导航欺骗检测方法研究

Primary labels:

```text
normal vs spoofing
```

## 当前阶段（2026-07-18）

数据整理、标签复核、逐卫星 baseline、设备级统计张量和第一轮设备级模型比较已完成。当前开发参考结果、模型筛选结论和 test 使用边界见 [experiment_progress.md](experiment_progress.md)。

下一轮实验不再以“继续增加模型数量”为优先，而是以降低跨环境设备指纹为核心：先构造因果相对特征，再用锁定的 4 折 Session-CV 比较 `LightGBM L=30`、`DLinear L=30` 和 `MLP L=5`。

Optional later labels:

```text
normal / L1 spoofing / L5 spoofing / L1+L5 spoofing
```

## Core Features

```text
Cn0DbHz
Cn0DbHz_dt
Cn0DbHz_std
AgcDb
ReceivedSvTimeUncertaintyNanos
PseudorangeRateUncertaintyMetersPerSecond
FreqBand
```

`sv_id`、`DeviceName` 用于信号组织和实验划分，不作为默认模型输入；ADR 不确定度保留用于消融，不进入首版模型。

## Required Experiments

1. Same-environment random split.
2. Playground train -> new-main-building test.
3. New-main-building train -> playground test.
4. Mixed train with environment-specific test reports.
5. Leave-one-device-out.
6. Leave-one-frequency-out.
7. Static -> dynamic and dynamic -> static.
8. Feature ablation.
9. Model complexity and deployment metrics.
10. TTD detection-time evaluation.

## Baseline Models

```text
Logistic Regression
Random Forest
XGBoost / LightGBM
Tiny-CNN
LSTM-small
DLinear
LightTS
PatchTST-small
```

已实现并完成第一轮直接设备级比较的模型为：

```text
DeviceStatsMLP / GRU / LSTM / TCN / DepthwiseCNN
DeviceStatsNLinear / DLinear / TSMixer
DeviceLightGBM
```

RS-TimesNet、CWT-ConvNeXt 等长窗口强模型暂作为教师或强基线候选，不作为当前首要部署模型。

## Main Model Direction

Main lightweight model:

```text
lightweight temporal model
+ real-device GNSS Raw features
+ device/frequency handling
+ sliding-window detection
+ deployment-oriented evaluation
```

## Metrics

Detection metrics:

```text
Accuracy
Macro-F1
Precision / Recall
FAR
Miss rate
TTD median
TTD 95th percentile
false alarms per minute
```

Deployment metrics:

```text
Params
FLOPs / MACs
Model size
CPU latency
Memory usage
ONNX latency if available
Android/edge latency if available
```


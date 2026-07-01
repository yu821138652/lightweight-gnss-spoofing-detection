# Experiment Plan

## Main Task

Real-device lightweight GNSS navigation spoofing detection.

Primary labels:

```text
normal vs spoofing
```

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
AccumulatedDeltaRangeUncertaintyMeters
FreqBand
sv_id
DeviceName
```

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

## Main Model Direction

`Lite-GNSSGuard`:

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

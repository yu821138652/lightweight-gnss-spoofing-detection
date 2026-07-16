# 当前开发基线协议

## 目的

本协议用于在不读取测试集的前提下，得到一个可复现的轻量化参考模型。它不是最终论文实验协议，也不是最终部署配置；它的作用是让不同模型只在时序编码器上存在差异，从而公平比较模型结构。

## 本阶段固定项

在第一轮模型比较完成前，以下内容不修改：

```text
张量目录：output/tensors_mixed
输入特征：7 项默认特征
窗口语义：连续 5 个历元的因果窗口，预测窗口末端当前历元
样本划分：Environment + Scenario + Session 分组划分
训练目标：逐 signal_id 的 normal / spoofing 二分类
模型选择：只依据 validation 集，不读取 test.npz
```

7 项默认特征：

```text
Cn0DbHz
Cn0DbHz_dt
Cn0DbHz_std
AgcDb
ReceivedSvTimeUncertaintyNanos
PseudorangeRateUncertaintyMetersPerSecond
FreqBand
```

`TOW`、时间戳、Environment、Scenario、Session、DeviceName 和标签来源只用于标注、追溯、数据划分与评价，不能作为当前检测模型输入，避免模型记住采集时段或场景。

## 比较模型

| 模型 | 作用 | 默认宽度 | 部署意义 |
|---|---|---:|---|
| `signal_mlp` | 将 5 x 7 窗口展平分类 | 32 | 最低复杂度参照 |
| `signal_gru` | 用 GRU 建模短时顺序 | 32 | 当前时序 baseline |
| `signal_tcn` | 两层因果 1D-CNN，膨胀卷积为 1、2 | 32 | 卷积式低延迟候选 |
| `signal_lstm` | 单层 LSTM | 32 | 循环网络对照 |
| `signal_transformer_tiny` | 一层、4 注意力头的因果 Transformer | 32 | 注意力机制的小型对照 |

所有模型输入均为 `[batch, signal, 5, 7]`，输出均为 `[batch, signal, 2]`。模型内部只使用截至当前历元的特征；TCN 和 Transformer 也保持因果性。

## 运行顺序

先对每个新模型执行干运行。**何时运行：** 第一次使用该模型、重建张量或更换 Python/PyTorch 环境后。**为什么运行：** 验证张量形状、掩码和前向传播，不更新权重、不保存 checkpoint，也不读取测试集。

```powershell
$PY = "C:\Users\Asus\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\python.exe"

& $PY pipeline_total\07_train_models.py `
  --data-dir output\tensors_mixed `
  --model signal_tcn `
  --dry-run
```

干运行通过后再训练。**何时运行：** 需要把模型纳入当前统一基线比较时。**为什么运行：** 仅用 train.npz 更新权重，并在 val.npz 上以相同规则保存最佳 checkpoint。请为每个模型使用独立输出目录，避免混放权重和指标。

```powershell
& $PY pipeline_total\07_train_models.py `
  --data-dir output\tensors_mixed `
  --output-dir output\training\signal_tcn_current_protocol `
  --model signal_tcn `
  --epochs 30 `
  --batch-size 256 `
  --patience 6 `
  --seed 2026
```

将上面的 `signal_tcn` 同时替换为 `signal_lstm`、`signal_transformer_tiny`，并相应替换输出目录。训练脚本会自动使用 CUDA；日志中的 `device=cuda` 表示 GPU 已生效。

## 当前轮次的选择规则

每次训练后记录对应目录中的 `val_metrics_<model>.json`，至少比较：

```text
validation Macro-F1
spoofing Precision
spoofing Recall
FAR
参数量
```

不要以单一 Macro-F1 选模型。对部署候选，优先排除 FAR 过高的模型，再比较 Recall、Precision、参数量与后续的推理延迟。此阶段不得因为验证集表现而重新划分 Session、修改 7 特征、调整窗口长度或读取测试集。

## 第一轮结束后

从验证集选出一个或两个轻量模型后，才进入协议优化：

1. 在相同的 train/val 划分下做逐特征移除消融。
2. 用验证集进行阈值和设备级多信号报警规则校准。
3. 锁定特征、窗口、模型和阈值后，对测试集进行一次最终评估。
4. 最后再开展跨环境、跨设备、静态/动态迁移和 EventGroup 级验证。

因此，第一轮的产物是“可比较的开发参考模型”，而不是最终部署结论。

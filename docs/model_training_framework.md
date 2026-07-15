# 模型训练框架

## 目的

本文件定义本项目从 GNSS 张量到模型结果的统一训练边界。目标是先建立可复现、可部署的轻量 baseline，再在相同数据协议下比较后续自定义模型或适配模型。

当前正式任务为逐信号二分类：

```text
0 = normal
1 = spoofing
```

## 数据层级

三种层级不能混淆：

```text
Environment + Scenario + Session
  -> 真实录制单元，只用于 train / val / test 划分

DeviceName + SourceRelativePath
  -> 单设备时序，只用于生成独立滑动窗口

signal_id
  -> 窗口中的信号槽位，每条信号有一个二分类标签
```

同一真实 Session 的所有设备日志必须进入同一个集合，防止同一次攻击、位置和时间条件泄漏到测试集。不同设备的同名 `signal_id` 不能合并为同一张量槽位。

## 当前张量接口

`output/tensors_mixed/*.npz` 中的关键数组为：

```text
x    [B, 128, 5, 7]
mask [B, 128]
y    [B, 128]
```

其中：

- `B`：窗口样本数；
- `128`：最多 128 条独立 `signal_id`；
- `5`：连续 5 个历史至当前时刻的观测；
- `7`：默认轻量特征；
- `mask`：`True` 表示该槽位是真实信号，`False` 是填充；
- `y`：每条有效信号的 `normal / spoofing` 标签。

默认 7 项输入特征：

```text
Cn0DbHz
Cn0DbHz_dt
Cn0DbHz_std
AgcDb
ReceivedSvTimeUncertaintyNanos
PseudorangeRateUncertaintyMetersPerSecond
FreqBand
```

ADR 不确定度仍保留在 CSV、审计和可视化中，但不进入首版模型输入。

## 模型接口

项目内逐信号模型必须满足：

```text
input:  x       [B, S, T, F]
output: logits  [B, S, 2]
```

训练器会使用 `mask` 过滤填充槽位，只对有效信号计算损失和指标。新模型只要满足该接口，即可复用现有的数据读取、类别加权、验证早停和指标计算。

## 当前 Baseline

| 模型 | 作用 | 默认参数量 | 适用问题 |
|---|---:|---:|---|
| `signal_mlp` | 展平 5 秒 x 7 特征后分类 | 1,288 | 简单特征组合是否已经足够 |
| `signal_gru` | 用小型 GRU 保留 5 秒顺序 | 4,066 | 短时变化顺序是否有额外价值 |

它们是本项目自有的最小可控 baseline，不依赖第一篇论文的 TimesNet，也不代表最终主模型。

## 训练顺序

### 1. 干运行

**何时运行：** 新建模型、重建张量、切换 VS Code Python 解释器后。

**为什么运行：** 确认张量形状、有效信号掩码和模型前向传播匹配。不会更新权重、保存 checkpoint 或读取测试集。

```powershell
$PY = "C:\Users\Asus\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\python.exe"

& $PY pipeline_total\07_train_models.py `
  --data-dir output\tensors_mixed `
  --model signal_mlp `
  --dry-run
```

### 2. 训练与验证

**何时运行：** 干运行通过后，比较 `signal_mlp` 和 `signal_gru` 或其他已接入模型时。

**为什么运行：** 参数更新只使用 `train.npz`；每个 epoch 在 `val.npz` 上计算 Macro-F1、Precision、Recall 和 FAR，用于早停和模型选择。此阶段不得读取 `test.npz`。

```powershell
& $PY pipeline_total\07_train_models.py `
  --data-dir output\tensors_mixed `
  --output-dir output\training\signal_mlp `
  --model signal_mlp `
  --epochs 30 `
  --batch-size 256 `
  --patience 6 `
  --seed 2026
```

训练完成后比较不同输出目录内的：

```text
val_metrics_signal_mlp.json
val_metrics_signal_gru.json
best_signal_mlp.pt / best_signal_gru.pt
```

模型选择应以验证集 Macro-F1 为主，同时检查 Recall、FAR 和参数量。不能根据测试集结果选择模型。

### 3. 最终测试

**何时运行：** 模型结构、7 项特征、窗口长度、超参数和阈值方案均已锁定后。

**为什么运行：** `test.npz` 是一次性最终评估数据。`--test-only` 只加载已有最佳 checkpoint，不会重新训练。

```powershell
& $PY pipeline_total\07_train_models.py `
  --data-dir output\tensors_mixed `
  --output-dir output\training\signal_mlp `
  --model signal_mlp `
  --test-only
```

## 设备级报警与 TTD

当前模型输出的是逐信号概率，而不是最终设备报警。后续部署层需要把同一时刻多条有效信号的结果聚合，例如：

```text
至少 2 条有效 signal_id 的 spoofing 概率超过阈值 -> 设备报警
```

聚合阈值和连续秒数必须只在验证集确定。随后在测试集按每个欺骗区间计算：

```text
TTD = 首次设备报警 TOW - 欺骗开始 TOW
```

由于动态场景可能只有数秒欺骗，不应在未验证前强制要求连续 3 秒报警。

现有 `pipeline_total/08_inference.py` 是历史模型接口，尚不能加载本文件定义的 baseline checkpoint。在设备级聚合规则确定前，不应使用它对 CSV 生成部署结论。

## Time-Series-Library 的使用边界

本地 `H:\GNSS\Time-Series-Library` 是外部模型与结构参考库，不是本项目正式数据和结果仓库。

TSLib 的现成分类接口通常为：

```text
input:  [B, T, C]
output: [B, 2]
```

它与本项目的逐信号标签不同，不能直接运行 UEA 分类脚本。合理使用方式包括：

1. **逐信号适配：** 将 `[B, S, T, F]` 重排为 `[B*S, T, F]`，模型逐信号输出后恢复为 `[B, S, 2]`。
2. **设备窗口适配：** 先定义窗口级标签与报警规则，再把多信号特征组织为 `[B, T, C]`。这种方式不能直接复用当前 `y[B, S]`。
3. **结构借鉴或自定义：** 参考 TSLib 的层、归一化和训练范式，在本项目 `models/` 下实现符合逐信号接口的模型。

当前窗口仅有 5 秒，TimesNet 的周期发现优势未必能发挥，因此它是后续可比较模型，不是默认主模型。

## 后续实验顺序

1. 完成 `signal_mlp`、`signal_gru` 的 mixed Session-grouped baseline 比较。
2. 锁定一个轻量模型方案。
3. 为同一方案分别重训并完成跨环境、静态到动态、动态到静态和跨设备实验。
4. 仅在上述 baseline 不足时，适配 TSLib 模型或开发多信号联合模型。
5. 在设备级聚合确定后报告 FAR、TTD、参数量、模型大小和 CPU 推理延迟。

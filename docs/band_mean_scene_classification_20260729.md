# 频段均值场景四分类：一条新路线的完整记录（2026-07-29）

## 结论先行

这条路线换了一个**判别单元**和**表征**：不再逐 `signal_id` 判"这个信号是否被欺骗"，而是把每个接收历元（TOW）的信号按频段（L1/L5）各自取特征均值，组成一个紧凑的窗口向量，让一个轻量时序模型直接输出**当前时刻属于哪种欺骗场景**——正常 / L1 / L5 / L1+L5 四分类。

在纯静态数据、7 折留一录制交叉验证、跨录制汇总的严格协议下：

- 初版（含全部特征）跨录制汇总 **Macro-F1 = 0.728**，准确率 87.4%。
- 通过诊断发现 **AGC 是一个"拐杖特征"**：它在跨设备时不可比（有的设备 AGC 退化成常数），模型学会依赖它反而损害泛化。
- **去掉 AGC 重新训练后 Macro-F1 = 0.948**。继续做全特征消融后发现：**只有 C/N0 能跨录制迁移**，AGC / 接收机时间不确定度 / 伪距率不确定度**三者都是毒特征**。
- **最终最佳配置只有 4 维**（L1 C/N0 均值 ‖ L5 C/N0 均值 ‖ 两个频段存在标志），静态跨录制 **Macro-F1 = 0.992**，准确率 99.4%，四类 recall 全部 ≥0.98。
- 逐设备看，AGC 退化最严重的 RedMi_K60 攻击窗口 recall 从 **0.228 → 0.996**。
- **扩到含动态的 mixed 范围**（4 折 grouped CV，43672 窗口）**Macro-F1 = 0.922**，四类 recall ≥0.877。主线长期塌陷的动态 L5 在这里没有重演。

模型是一个 **5348 参数的轻量 TCN 窗口分类器**（4 维输入版更小），延续项目一贯的轻量定位。

一条重要的**边界条件**：把同样的"只用 C/N0"搬到主线逐信号二分类上是**负结果**（pooled Macro-F1 0.8637 → 0.7827）。"AGC 是毒特征"只在本路线的 band-mean 窗口级表征下成立，**不能推广到主线二分类**。详见第五节。

> 本文既记录方法，也记录我们通过"看图 → 提出想法 → 交流 → 验证 → 证伪 → 修正 → 发现"的完整过程，供组内继续讨论。核心可信证据是**消融重训**；置换重要性是辅助佐证。所有数字均从各折 test 预测 CSV 源头重算核对过。

---

## 一、这条路线是怎么想出来的

出发点是一个观察：主线基线（逐 signal 二分类）里 `dy_L5` 长期塌陷（recall ~21%），而且我们不能再采集新数据，只能从现有信息里挖。

组内提出一个设想：**加一层场景分类** —— 先判断"当前是 L1 欺骗 / L5 欺骗 / L1+L5 欺骗 / 正常"哪一种。为了判断这个设想是否值得做，我们没有一上来就搭模型，而是**先画图看可分性**。

### 1.1 先画图：band-mean dashboard

仿照已有的逐信号标注 dashboard，做了一个新的可视化 [22b_plot_band_mean_dashboards.py](../pipeline_total/22b_plot_band_mean_dashboards.py)：对每个 TOW，把所有 L1 信号的特征取均值（蓝线）、所有 L5 信号取均值（红线），per-device 分列绘制。特征为四个基线量：C/N0、AGC、接收机时间不确定度、伪距率不确定度。

看图得到两个关键判断：

1. **L1 欺骗信号清晰**：C/N0 齐刷上冲，跨设备高度一致。
2. **L5 欺骗信号弱且不一致**：有的设备冲高、有的凹陷；无 L5 能力的设备（手表）完全看不到。这解释了主线里 `dy_L5` 塌陷的物理根源——不是模型没学好，是 L5 痕迹本身微弱。

尽管 L5 弱，但把 L1/L5 两条曲线**并排看**时，"L1 是否也在被打"是肉眼可辨的。这正是四分类里区分"纯 L5"和"L1+L5"的关键信息，而它恰恰是**逐信号视角看不到的跨信号属性**。于是判断：值得一试。

### 1.2 一个被否掉的中间设计

最初的设计想在逐信号模型上加窗口级池化头。但讨论后**否掉了**：单个信号只属于一个频段，L1+L5 场景里的一个 L5 信号，和纯 L5 场景里的 L5 信号在 per-signal 层面完全一样，逐信号头无法区分二者。

修正后的设计（即本路线）：**直接把信号维度压成 band-mean**，判别单元定为"窗口级时刻场景"。这样模型在一次判决里能同时看到 L1 和 L5 两路，跨信号信息被显式编码进输入。

---

## 二、方法

### 2.1 一条样本与窗口

| 项目 | 定义 |
|---|---|
| 判别单元 | 一个设备在某个窗口末端历元的**场景四分类** |
| 窗口 | `TIME_STEPS=5` 个连续同 split、同 segment、无接收间隔的历元，严格因果 |
| 每历元向量 | `[L1: C/N0, AGC, SvTimeUnc, PrRateUnc ‖ L5: 同四项 ‖ L1Present, L5Present]` = 8 均值 + 2 存在标志 = **10 维** |
| 四分类标签 | 末端历元 TOW 落在 reviewed 攻击区间内 → 按 Scenario 映射（L1→1, L5→2, L1+L5→3）；区间外 → 0（正常）|

### 2.2 单频段窗口的隔离

若末端历元**只观测到一个频段**（只有 L1 或只有 L5，例如手表无 L5），该窗口无法表达四分类场景，被标记 `single_band_mask=True`：标签强制置 0，且**训练损失与所有评估指标都排除它**。这些窗口有意保留在张量里，供后续单独的路径消费。

在静态数据上，51690 个 test 窗口中有 23473 个是 single-band，参与四分类任务的 usable 窗口为 28217 个。

### 2.3 缺失值处理

沿用主线的思路：band-mean 只在有限值上计算；缺失的频段均值为 NaN，经**per-device 训练集统计标准化**后填为中性 0；两个 presence 标志保留 [0,1] 物理含义不缩放。per-device 标准化本身也是消除硬件基线差异的手段。

### 2.4 模型

`BandMeanWindowClassifier`（[models/gnss_signal_baselines.py](../models/gnss_signal_baselines.py)）：

- 输入 `[B, 5, 10]` → 时序编码器（tcn / lstm / gru，本轮用 tcn）→ 取末端时刻 embedding → 4 类头。
- tcn 编码器：两层因果卷积 `CausalConv1d(10→32, k=3)` + `CausalConv1d(32→32, k=3, dilation=2)`，与主线基线复用同一组件。
- 分类头：`LayerNorm(32) → Linear(32→32) → GELU → Dropout(0.1) → Linear(32→4)`。
- `hidden_dim=32`，**参数量 5348**。

与主线基线的关系：同族（复用 CausalConv1d、tcn 结构、优化配方），但这是**单分支、窗口级输出、4 类头**，主线是双分支、逐信号、二分类、1880 参数。刻意保持可比与轻量定位。

### 2.5 训练与协议

- 优化：4 类反频率加权 CrossEntropy、AdamW（lr 1e-3, wd 1e-4）、batch 256、≤40 epoch、patience 8、seed 2026、按 val macro-F1 选点。
- 数据范围：纯 static，7 个录制。
- 协议：`static_time_block_outer_v2` 的 **7 折留一录制交叉验证**。每折留出 1 个完整录制做 test，其余做 development（内部再切时间块 train/val）。scaler 只用 train 拟合。

### 2.6 为什么必须跨折汇总看 test

静态留一录制交叉验证下，每折留出的录制只有一种攻击场景，所以**单折 test 不可能包含全部四类**：

| fold | held-out 录制 | test 可见类 |
|---|---|---|
| 1 | st_L1 | 0, 1 |
| 2/3 | st_L5 | 0, 2 |
| 4 | st_L_15 | 0, 3 |
| 5 | st_L1 | 0, 1 |
| 6 | st_L5 | 0, 2 |
| 7 | st_L_15 | 0, 3 |

因此真正的四分类评估**只能靠把 7 折的 test 预测汇总**成一张跨录制 4×4 混淆矩阵（[47_aggregate_band_mean_cv.py](../pipeline_total/47_aggregate_band_mean_cv.py)）。val 集（development 录制的时间块）四类齐全，可作快速信号，但它与 train 同源，会高估泛化——事实也确实如此（val Macro-F1 ~0.97，跨录制汇总只有 0.728）。

---

## 三、结果与关键发现

### 3.1 初版（含全部特征）跨录制汇总

Macro-F1 = **0.728**，准确率 87.4%。

| 真实\预测 | normal | L1 | L5 | L1+L5 | recall |
|---|---|---|---|---|---|
| normal | 19282 | 3 | 20 | 2 | 0.999 |
| L1 | 558 | 1140 | 498 | 203 | 0.475 |
| L5 | 673 | 0 | 2927 | 1 | 0.813 |
| L1+L5 | 568 | 9 | 1026 | 1307 | 0.449 |

好的方面：normal 近乎完美，L5 表现意外地好。问题：L1 与 L1+L5 recall 低，且 L1+L5 大量（1026 个）被误判成纯 L5。

### 3.2 组内提出的两个观察（驱动后续诊断）

1. **L1 常被误判成 normal，且概率咬得很近**（如 L1 0.3 / normal 0.65）——像是"看不到跳变"。
2. **RedMi vs MI8 在同一 st_L_15 场景下结果天差地别（recall 0.0 vs 0.989），但看 C/N0 图两者其实很相近**——不该差这么多。且物理上欺骗会抬高受扰信号的 C/N0。

这两个观察直接指向"某个特征在作怪"，而非模型能力问题。

### 3.3 根因诊断

**根因 A：AGC 设备退化。** 从原始 CSV 核实：RedMi 的 AGC 全程只有 **6 个离散值**（几乎常数），而 MI8 有 705 个；手表干脆无 AGC。RedMi 的 `AgcDbMissing=0`，即非缺失，是硬件本身不输出有效 AGC。而攻击时 RedMi 的 C/N0 明明涨得比 MI8 还猛（L5 +18.8 vs +11.9）。模型学到的捷径"L1 被打 ⟺ AGC 下掉"在 RedMi 上彻底失效，于是它把 L1+L5 误判成纯 L5。

**根因 B：窗口看不到跳变过程。** L1 recall 随末端离攻击起点越深越低（edge≤3s: 0.62 → deep>30s: 0.41）。5 秒窗口整个泡在攻击区内部时，C/N0 是一条抬高但平稳的线，与"高基线的正常"混淆。

### 3.4 先试类权重再平衡（负结果，排除一条路）

假设"模型太保守"，给 L1/L1+L5 加权 2.5×。结果 Macro-F1 **0.728 → 0.699**，L1 recall 反而 **0.475 → 0.384**。

这**证伪了"保守"假设**：L1 precision 高达 0.99，模型不是不敢判，而是"能可靠判成 L1 的窗口本就少"。加权只是让模型更用力拟合训练录制里的 L1 模式，换到测试录制上泛化更差。问题被彻底推向特征层面。类权重这条路关闭。

### 3.5 特征消融（核心发现）

去掉 AGC 重新训练全部 7 折：

| 指标 | 含 AGC | **去 AGC** | 变化 |
|---|---|---|---|
| Macro-F1 | 0.728 | **0.948** | **+0.220** |
| 准确率 | 0.874 | 0.971 | +0.097 |
| L1 recall | 0.475 | 0.713 | +0.238 |
| L5 recall | 0.813 | 0.984 | +0.171 |
| L1+L5 recall | 0.449 | **0.987** | +0.538 |

逐设备（攻击窗口 recall）：

| 设备 | 含 AGC | 去 AGC |
|---|---|---|
| **RedMi_K60** | **0.228** | **0.996** |
| Google_Pixel6 | 0.599 | 0.978 |
| HUAWEI_Mate40 | 0.667 | 0.881 |
| XiaoMi_MI8 | 0.938 | 0.986 |

组内基于看图的直觉被完全证实。这不是弱特征，是**毒特征**：AGC 在设备间不可比，模型在训练录制上学到的 AGC 捷径换录制即失效。去掉后模型被迫依赖跨设备可迁移的 C/N0，泛化立刻起来。

### 3.6 置换重要性（辅助佐证，含一次自我纠错）

在**含全部特征的冻结基线模型**上，对每个特征打乱其跨窗口取值、重测 macro-F1（[49_plot_band_mean_permutation_importance.py](../pipeline_total/49_plot_band_mean_permutation_importance.py)）。

一次踩坑：初版脚本每个特征只打乱**一次**，AGC 的贡献符号在两次运行间翻转（+0.09 / −0.06）。单次置换是高方差估计，不可用。**修正为每特征 20 次独立置换取均值±std**后稳定（std ≤ 0.004）：

| 特征 | Macro-F1 drop (n=20) | 含义 |
|---|---|---|
| Cn0DbHz | +0.120 | 模型最依赖，核心信号 |
| AgcDb | +0.090 | 模型**也依赖**它（正贡献）|
| ReceivedSvTimeUnc | −0.002 | ≈0，几乎没用 |
| PseudorangeRateUnc | −0.015 | 轻微有害 |

**关键澄清——置换与消融不矛盾：**

- 置换（AGC +0.09）说明冻结的基线模型**确实依赖 AGC**；
- 消融（去 AGC 涨到 0.948）说明**不学会依赖 AGC 反而更好**。

比喻：**AGC 是一根拐杖**。基线模型学会了拄着它走（置换一抽走就摔，故为正贡献），但这根拐杖在训练录制好用、换新录制/退化设备就坏。解法不是"测试时别用拐杖"，而是**一开始就别学会拄拐杖**——即消融。两图合起来，故事才完整。

> 方法论提醒：**消融重训是硬证据，置换重要性是辅助**。二者测的问题不同，不能互相替代。置换脚本务必多次重复取均值，否则符号会因采样噪声翻转。

### 3.7 完整单特征消融：不止 AGC 一个毒特征

既然 AGC 的发现如此显著，接着把四个特征逐个单独去掉，各跑一遍完整 7 折。结果推翻了"AGC 是唯一毒特征"的假设：

| 变体 | 维度 | Macro-F1 | normal R | L1 R | L5 R | L1+L5 R |
|---|---|---|---|---|---|---|
| full（基线） | 10 | 0.728 | 0.999 | 0.475 | 0.813 | 0.449 |
| drop AGC | 8 | **0.948** | 0.999 | 0.713 | 0.984 | 0.987 |
| drop C/N0 | 8 | 0.717 | 0.994 | 0.436 | 0.800 | 0.422 |
| drop 接收机时间不确定度 | 8 | 0.891 | 0.999 | 0.666 | 0.833 | 0.936 |
| drop 伪距率不确定度 | 8 | 0.857 | 0.998 | 0.452 | 0.822 | 0.982 |

读法很清楚：**只有去掉 C/N0 会变差**（0.728 → 0.717，唯一的负向），其余三个去掉都涨。也就是说——

> **只有 C/N0 能跨录制迁移；AGC、接收机时间不确定度、伪距率不确定度三者都在拖后腿。**

物理上讲得通：不确定度量高度依赖接收机芯片型号和解算实现，和 AGC 一样是**设备指纹**，不是攻击特征。四分类要判断"哪个频段被打"，依赖的是跨设备可比的相对关系，设备指纹类特征只会引入虚假捷径。

### 3.8 顺理成章的一步：只留 C/N0

上面的结论直接引出一个组合消融：把三个毒特征全砍掉，只留 C/N0 家族。

| 变体 | 维度 | Macro-F1 | acc | normal R | L1 R | L5 R | L1+L5 R |
|---|---|---|---|---|---|---|---|
| drop AGC | 8 | 0.948 | 0.971 | 0.999 | 0.713 | 0.984 | 0.987 |
| C/N0 + 差值 | 5 | 0.987 | 0.990 | 0.999 | 0.984 | 0.953 | 0.987 |
| **C/N0（无差值）** | **4** | **0.992** | **0.994** | 0.999 | 0.985 | 0.982 | 0.988 |

**最佳配置只有 4 维**：`L1_Cn0DbHz`、`L5_Cn0DbHz`、`L1Present`、`L5Present`。静态跨录制 Macro-F1 **0.992**，四类 recall 全部 ≥0.98。

一路追下来最想救的两个短板彻底解决：**L1 recall 0.475 → 0.985**，**L1+L5 recall 0.449 → 0.988**。

### 3.9 差值特征：一个诚实的负结果

组内提出加一维 `L1_Cn0 − L5_Cn0` 的显式差值，理由是让"L1 相对 L5 抬升了"直接可见。已实现在 builder 中（`Cn0DbHzL1MinusL5`，仅双频段可见时有值，否则 NaN → 标准化后落 0）。

净贡献用同协议 A/B 单独归因：

| 配置 | Macro-F1 | L5 R |
|---|---|---|
| C/N0 + 差值（5 维） | 0.987 | 0.953 |
| C/N0 无差值（4 维） | **0.992** | **0.982** |

**差值净贡献 −0.005**，主要来自 L5 recall 下降。诚实解读：差值是两维 C/N0 的**线性组合**，模型本来就能自己学出来；显式加入不增加信息，只多一个参数和一份标准化噪声。−0.005 属噪声量级，结论是"没必要"而非"有害"。

有意思的是，**在 full 配置下差值确实有用**（0.728 → 0.843，+0.115）：那时模型被毒特征干扰，差值提供了一个干净的相对信号。一旦毒特征被砍掉，这个作用就被 C/N0 本身覆盖了。这个对比本身说明：**辅助特征的价值取决于主特征集是否已经干净**。

差值列保留在 builder 里（默认生成），是否使用由 `--drop-features Cn0DbHzL1MinusL5` 控制。当前推荐配置**不使用**。

### 3.10 扩到动态：mixed 范围验证

前面全部在 static 上。接着把 scope 放开到含动态的 mixed 范围（builder 新增 `--scope static|dynamic|all`），协议用 `mixed_timeblock_outer_cv4_w5_v2`（4 折 grouped CV，每折 test 6 个录制且**四类齐全**，比 static 留一法更适合评估四分类）。

配置为当时的最佳（只留 C/N0 家族含差值，5 维），43672 个 usable 窗口：

| 真实\预测 | normal | L1 | L5 | L1+L5 | recall | F1 |
|---|---|---|---|---|---|---|
| normal | 31678 | 542 | 100 | 46 | 0.979 | 0.975 |
| L1 | 296 | 3724 | 0 | 75 | 0.909 | 0.868 |
| L5 | 484 | 0 | 3452 | 1 | 0.877 | 0.922 |
| L1+L5 | 139 | 221 | 0 | 2914 | 0.890 | 0.924 |

**Macro-F1 = 0.922，准确率 95.6%。**

- 相比 static-only 的 0.987 有回落，合理：动态录制中设备运动使 C/N0 均值波动更大，判别本就更难。
- **主线的老大难在这里没有重演**：主线逐信号二分类的 `dy_L5` recall 长期只有 ~18.7%，而这里 L5（含动态）recall 达 **0.877**、L1+L5 达 0.890。
- L5 与 L1+L5 之间**几乎不混**（L5→L1+L5 仅 1 个，L1+L5→L5 为 0）。当初最担心的问题在动态下也没出现。
- 主要错误仍是攻击↔normal 的边界漏判，符合物理直觉（区间边界处信号弱）。

---

## 四、边界条件：这个结论能推广到主线二分类吗？

不能。这一节很重要，避免组内误用。

"AGC 是毒特征"在本路线成立得如此彻底，自然想问：主线的逐信号二分类是不是也只用 C/N0 就更好？为此做了一次**严格复刻主线协议**的对照——同样的 `mixed_timeblock_outer_cv4_w5_state_stratified_v2` 协议、同样的 raw+stats 双分支 `SignalRawStatsFusion`、同样 TCN16 / dropout 0.1 / W5，**只把特征换成只用 C/N0**（新增档位 `--raw-feature-set cn0_only` + `--stats-feature-set cn0_coverage_rx_time_std`）。

| 指标 | 主线基线 compact11 | cn0_only | 变化 |
|---|---|---|---|
| pooled Macro-F1 | **0.8637** | 0.7827 | **↓ 0.081** |
| pooled Precision | 0.7801 | 0.6420 | ↓ |
| pooled Recall | 0.7909 | 0.6791 | ↓ |
| pooled FAR | 0.0598 | 0.1017 | ↑ 变差 |
| 参数量 | 1880 | 1664 | — |

四折 test：0.8304 / 0.7482 / 0.7766 / 0.7580，fold 等权 0.7783 ± 0.0367。

**明确的负结果：二分类任务上只用 C/N0 显著更差，AGC 等特征在那里是有用的。**

为什么两个任务的结论相反，这个反差本身有价值：

- **四分类（band-mean、窗口级）**要判断"**哪个**频段被打"，依赖**跨设备可比的相对关系**。AGC 的设备间不可比性（RedMi 卡在 6 个离散值、MI8 有 705 个）直接污染这个判断。
- **二分类（逐信号）**只判"**有没有**被打"，AGC 的**绝对下掉**是很强的本地证据。每个信号与自身历史比较，设备间不可比反而不构成问题。

> **结论：**"AGC/不确定度是毒特征"只在**判别单元为窗口级、表征为 band-mean、任务为频段归属**这三个条件同时成立时有效。搬到主线二分类会掉 0.08。

---

## 六、当前状态与局限

**已确立的成果**

- 静态跨录制四分类 Macro-F1 **0.992**（4 维只留 C/N0），四类 recall 全部 ≥0.98。
- 含动态的 mixed 范围 Macro-F1 **0.922**，L5 recall 0.877 —— 主线 `dy_L5` 的塌陷在本路线未重演。
- 模型极轻量（4 维输入的 TCN，量级同 5348 参数版本），符合项目一贯定位。
- 一个可推广的方法论发现：**特征是否有害取决于判别单元和任务**，设备指纹类特征在跨设备频段归属判断中是毒特征。

**局限与未验证项**

- **single-band 窗口尚未处理**：static 下 23473 个窗口因只观测到单一频段被隔离（标签强制为 0、不计入统计）。这批占比不小，是设计上留给后续独立路径的，当前结果**不覆盖**这些时刻。
- **每类攻击场景的 session 数很少**：static 仅 7 个录制（每类攻击 1–2 个 session）。0.992 这个数字建立在小样本上，需谨慎解读；mixed 的 4 折 grouped CV（24 session）更可靠。
- **未做多 encoder / 多 seed 稳健性检查**：全部结果均为 tcn + seed 2026 单次。差值的 −0.005 属噪声量级，正因如此未做重复实验确认。
- **不可与主线二分类数字直接比较**：四分类（0.992 / 0.922）与主线逐信号二分类（0.8637）是不同任务、不同判别单元、不同评估单位。
- **边界条件必须随结论一起引用**：见第四节。只用 C/N0 在主线二分类上会掉 0.08。

---

## 七、复现命令

```bash
# 1. 画 band-mean dashboard（可选，用于看可分性）
python pipeline_total/22b_plot_band_mean_dashboards.py

# 2. 当前最佳：static 4 维只留 C/N0（Macro-F1 0.992）
python pipeline_total/47_aggregate_band_mean_cv.py \
    --tensors-root output/tensors/band_mean_window_static_v2 \
    --drop-features AgcDb ReceivedSvTimeUncertaintyNanos \
        PseudorangeRateUncertaintyMetersPerSecond Cn0DbHzL1MinusL5 \
    --training-root output/training/band_mean_multiclass_cv_v2_cn0nodiff

# 3. 含动态的 mixed 范围（Macro-F1 0.922）
python pipeline_total/47_aggregate_band_mean_cv.py \
    --protocol-dir output/protocols/mixed_timeblock_outer_cv4_w5_v2 \
    --tensors-root output/tensors/band_mean_window_mixed_v2 \
    --scope all \
    --drop-features AgcDb ReceivedSvTimeUncertaintyNanos \
        PseudorangeRateUncertaintyMetersPerSecond \
    --training-root output/training/band_mean_multiclass_cv_mixed_cn0only

# 4. 含 AGC 的原始基线（0.728，用于对照）
python pipeline_total/47_aggregate_band_mean_cv.py \
    --training-root output/training/band_mean_multiclass_cv

# 5. 导出全体 test 窗口（含 single-band）供检视
python pipeline_total/48_export_band_mean_test_windows.py \
    --training-root output/training/band_mean_multiclass_cv_noagc \
    --output output/training/band_mean_multiclass_cv_noagc/all_test_windows.csv

# 6. 置换重要性（必须多次重复取均值，否则符号会翻转）
python pipeline_total/49_plot_band_mean_permutation_importance.py --repeats 20

# 7. 边界条件对照：主线二分类只用 C/N0（负结果，0.7827）
python pipeline_total/20_build_static_timeblock_tensors.py \
    --protocol-dir output/protocols/mixed_timeblock_outer_cv4_w5_state_stratified_v2/fold_1 \
    --output-dir output/tensors/mixed_timeblock_outer_cv4_w5_state_stratified_v2/fold_1 \
    --data-scope mixed
python pipeline_total/21_train_static_signal_fusion.py \
    --data-dir output/tensors/mixed_timeblock_outer_cv4_w5_state_stratified_v2/fold_1 \
    --output-dir output/training/mixed_ss_v2_cn0only_tcn16/fold_1/tcn \
    --encoder tcn --hidden-dim 16 --dropout 0.1 \
    --raw-feature-set cn0_only --stats-feature-set cn0_coverage_rx_time_std
```

## 八、相关文件

| 文件 | 作用 |
|---|---|
| [22b_plot_band_mean_dashboards.py](../pipeline_total/22b_plot_band_mean_dashboards.py) | band-mean 可视化，路线的起点 |
| [45_build_band_mean_window_tensors.py](../pipeline_total/45_build_band_mean_window_tensors.py) | 建 band-mean 窗口张量，含 single-band 标记、C/N0 差值列、`--scope static\|dynamic\|all` |
| [46_train_band_mean_multiclass.py](../pipeline_total/46_train_band_mean_multiclass.py) | 四分类训练/测试，支持 `--drop-features`、`--class-weight-mult` |
| [47_aggregate_band_mean_cv.py](../pipeline_total/47_aggregate_band_mean_cv.py) | 多折建张量+训练+跨折汇总混淆矩阵，透传 `--scope`/`--drop-features` |
| [48_export_band_mean_test_windows.py](../pipeline_total/48_export_band_mean_test_windows.py) | 导出全体 test 窗口（含 single-band）为 CSV |
| [49_plot_band_mean_permutation_importance.py](../pipeline_total/49_plot_band_mean_permutation_importance.py) | 置换重要性（多次重复取均值） |
| [21_train_static_signal_fusion.py](../pipeline_total/21_train_static_signal_fusion.py) | 主线二分类；本轮新增 `cn0_only` raw 档与 `cn0_coverage_rx_time_std` stats 档用于边界条件对照 |
| `models/gnss_signal_baselines.py::BandMeanWindowClassifier` | 窗口级四分类器（10 维输入时 5348 参数；4 维最佳配置更小） |

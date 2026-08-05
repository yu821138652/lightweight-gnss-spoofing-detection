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

---

## 九、设备专属训练统计量消融（2026-08-04）

### 9.1 问题与实验定义

当前场景分支**确实使用了该设备训练集统计量**。对连续特征，默认处理是：

```text
x_scaled = (x - mean_of_this_device_train) / std_of_this_device_train
```

这里有三个容易混淆的边界：

1. `mean/std` 只由本折 `train` 拟合，`val/test` 不参与，不存在直接的验证或测试泄漏。
2. 它不是“该设备干净区间均值”，而是该设备全部训练窗口（正常与攻击都包含）的整体统计量。
3. 拟合单位是重叠窗口中的 timestep，因此同一历元可能因窗口重叠被重复计入；更准确的名称是“该设备 train 窗口加权统计量”。

“不用该设备训练集均值”采用如下严格 A/B：

| 变体 | 连续特征标准化 | 部署所需统计量 |
|---|---|---|
| `per_device` | 每台设备使用本折 train-only `mean/std` | 每个已知设备一套 |
| `global` | 所有设备共享本折全体 train-only `mean/std` | 全局一套 |

没有把 raw/no-scale 混进主对照，因为完全不缩放会同时改变数值尺度，无法把变化只归因于去掉设备专属统计量。本实验切换的是一套完整、可部署的 scaler，所以均值和标准差一起从 per-device 改为 global；它回答“是否仍需要每台设备的训练统计量”，不是数学上只隔离减均值一项。

### 9.2 固定实验条件

| 项目 | 固定值 |
|---|---|
| 划分协议 | `mixed_timeblock_outer_cv4_w5_v2`，4 折 outer recording holdout |
| 纯静态 | 同一协议，建张量时 `--scope static` |
| 静态+动态 | 同一协议，建张量时 `--scope all` |
| train/val | development recording 内按时间块划分，边界留 4 个 epoch guard |
| 窗口 | `W=5`，严格因果窗口 |
| 输入 | `L1_Cn0DbHz`、`L5_Cn0DbHz`、`L1Present`、`L5Present`，共 4 维 |
| 模型 | TCN32，dropout 0.1，4 分类，4,772 参数 |
| 训练 | seed 2026，最多 40 epoch，patience 8，batch 256，AdamW，反频率加权 CE |
| 选点 | validation Macro-F1 |
| 评估 | 汇总四折完整 outer test 预测后计算 4 类 Macro-F1 |

两种 scaler 的标签、fold、窗口、设备、录制、时间戳及 `single_band_mask` 已逐项核对一致；只有连续输入值发生变化。纯静态汇总 28,217 个可用双频 test 窗口，静态+动态汇总 43,672 个。单频 endpoint 仍按既定规则排除在训练和指标之外。

### 9.3 纯静态结果

| 指标 | `per_device` | `global` | 去掉设备专属统计量后的变化 |
|---|---:|---:|---:|
| Test Macro-F1 | **0.991749** | 0.966601 | **-0.025148** |
| Test Accuracy | **99.4471%** | 97.4200% | **-2.0271 pp** |
| 错分窗口 | **156** | 728 | +572 |
| normal recall | **0.998498** | 0.996530 | -0.001968 |
| L1 recall | **0.983743** | 0.982076 | -0.001667 |
| L5 recall | **0.985282** | 0.839767 | **-0.145515** |
| L1+L5 recall | **0.987973** | 0.985911 | -0.002062 |

损失几乎全部集中在 L5。逐设备追查发现，静态 aggregate 的差距主要由 Pixel6 的 L5 样本驱动：Pixel6 L5 recall 从 `per_device=0.9814` 降到 `global=0.0760`，设备总体 accuracy 从 99.36% 降到 83.82%。这说明汇总差值很大，但并非所有设备都同幅受益；例如 MI8 使用 global 时反而小幅提高约 0.10 个百分点。

四折 validation Macro-F1 均值只从 `0.9752` 变为 `0.9739`，几乎没有发出警报；真正的差距出现在 outer test 的 Pixel6 L5 录制。这再次说明，同一 development recording 内切出的时间块 validation 适合选 checkpoint，但不能代替跨录制测试。

### 9.4 静态+动态结果

| 指标 | `per_device` | `global` | 去掉设备专属统计量后的变化 |
|---|---:|---:|---:|
| Test Macro-F1 | **0.925215** | 0.896820 | **-0.028395** |
| Test Accuracy | **95.8944%** | 93.7763% | **-2.1181 pp** |
| 错分窗口 | **1,793** | 2,718 | +925 |
| normal recall | **0.979701** | 0.964345 | -0.015356 |
| L1 recall | **0.901587** | 0.881074 | -0.020513 |
| L5 recall | **0.907290** | 0.824486 | **-0.082804** |
| L1+L5 recall | **0.887599** | 0.882101 | -0.005498 |

mixed 下不是只有一个类别受益，normal、L1、L5 recall 都有明显改善；24 个 held-out recording 中，`per_device` accuracy 提高 15 个、持平 1 个、下降 8 个。不过设备差异仍然很强：Pixel6 总体 accuracy 提高 17.88 pp，而 MI8 反而下降 1.42 pp。`global` 的 L1+L5 F1 还略高于 `per_device`（0.9349 对 0.9195），来自更高 precision，而不是更高 recall。

mixed 的四折 validation Macro-F1 均值为 `per_device=0.8633`、`global=0.8403`，与 outer test 的下降方向一致；因此 mixed 中去掉设备统计量造成的退化不只来自某一个测试折的偶发现象。

### 9.5 CPU 推理开销

测时环境为 Intel Core i7-14650HX、Python 3.8.10、PyTorch 2.4.1；固定 CPU 单线程。每个 checkpoint 预热 200 次，batch=1 测 1,000 次，batch=256 测 100 次。表中为四折结果的中位数：

| 数据范围 / scaler | batch=1 p50 | batch=1 p95 | batch=256 p50 | 按 batch p50 折算吞吐 |
|---|---:|---:|---:|---:|
| static / `per_device` | 0.172 ms | 0.266 ms | 0.559 ms | 458k window/s |
| static / `global` | 0.182 ms | 0.290 ms | 0.579 ms | 442k window/s |
| mixed / `per_device` | 0.171 ms | 0.263 ms | 0.554 ms | 462k window/s |
| mixed / `global` | 0.167 ms | 0.247 ms | 0.561 ms | 457k window/s |

四组模型结构完全相同，均为 4,772 参数、checkpoint 24,628 bytes，理论计算量也相同。表中的约 0.01 ms 波动没有一致方向，应视为桌面 CPU 调度噪声，不能解释为某种 scaler 让模型更快。

测时范围是**预构建并已标准化的 `[1, 5, 4]` 张量到四分类 logits 的模型前向**，不含原始日志解析、band mean 聚合和 scaler 应用。部署 4 维模型时，global 只需保存 4 个 C/N0 标量（两维各一个 mean/std）；per-device 则每台已知设备保存 4 个标量并做一次设备查表。两者相对模型前向都很小，但 global 的部署状态更简单。

### 9.6 结论与适用边界

在当前数据、当前 4 维 TCN32 和“测试设备型号在训练中出现过”的条件下，**不建议去掉设备专属训练统计量**：纯静态和 mixed 的 test Macro-F1 分别下降 0.0251 和 0.0284，已经不是零点零几以内的无意义波动；推理模型本体又没有获得可测的成本收益。

但这个结论不能扩张为“per-device 对任何接收机都更好”：

- 只有一个 seed，窗口高度重叠，28k/44k 不能当作独立同分布样本数。
- 当前 outer test 隔离 recording，但设备身份在 train/test 重复；它评估的是**已知设备校准**，不是 unseen-device 泛化。
- 未见设备在实现中会回退到 global scaler，因此当前收益不会自动迁移到新型号。
- 静态总差距受 Pixel6 L5 强烈主导；mixed 更广泛受益，但 MI8 存在反例。

所以工程决策是：场景分支当前继续保留 `per_device`；若目标改为“新设备无需校准即可部署”，应另做 leave-one-device-out 协议，而不是用本实验的 recording-CV 数字替代。

### 9.7 实验产物与复现入口

```text
output/tensors/scene_scaler_ablation_v1/
  static_per_device/fold_1..4
  static_global/fold_1..4
  mixed_per_device/fold_1..4
  mixed_global/fold_1..4

output/training/scene_scaler_ablation_v1/
  <上述四个变体>/fold_1..4/
    best_band_mean_window_tcn.pt
    val_metrics_band_mean_window_tcn.json
    test_predictions_band_mean_window_tcn.csv
    runtime_cpu_1thread.json
  <上述四个变体>/
    aggregate_test_metrics.json
    aggregate_test_predictions.csv
```

新增或扩展的入口：

- [45_build_band_mean_window_tensors.py](../pipeline_total/45_build_band_mean_window_tensors.py)：新增 `--scaler-mode per_device|global`，并把 scaler fit 范围写入 metadata。
- [46_train_band_mean_multiclass.py](../pipeline_total/46_train_band_mean_multiclass.py)：checkpoint 记录 scaler mode，test-only 校验 checkpoint 与 tensor 一致。
- [47_aggregate_band_mean_cv.py](../pipeline_total/47_aggregate_band_mean_cv.py)：透传 `--scaler-mode`。
- [56_measure_band_mean_runtime.py](../pipeline_total/56_measure_band_mean_runtime.py)：单线程 CPU 模型前向测时。

等价的一体化复现命令如下。该入口会顺序执行各折；本次为缩短墙钟时间，将同一组参数拆成四个独立 fold 并行执行，结果汇总逻辑不变：

```bash
python pipeline_total/47_aggregate_band_mean_cv.py \
    --protocol-dir output/protocols/mixed_timeblock_outer_cv4_w5_v2 \
    --tensors-root output/tensors/scene_scaler_ablation_v1/<variant> \
    --training-root output/training/scene_scaler_ablation_v1/<variant> \
    --scope <static|all> \
    --scaler-mode <per_device|global> \
    --drop-features AgcDb ReceivedSvTimeUncertaintyNanos \
        PseudorangeRateUncertaintyMetersPerSecond Cn0DbHzL1MinusL5 \
    --encoder tcn --hidden-dim 32 --dropout 0.1 --epochs 40 --seed 2026
```

---

## 十、因果在线 C/N0 基线实验（2026-08-04）

### 10.1 目的与实验边界

第九节说明 `per_device` scaler 在当前 recording-CV 中有效，但它依赖“测试时已经拥有该型号设备的训练统计量”，不能直接证明对未见设备的泛化能力。为避免把固定的设备训练均值当作部署先验，本轮实现了两种只依赖当前数据流过去信息的在线基线：

| 路线 | 基线更新规则 | 是否需要设备训练统计量 |
|---|---|---|
| `relative_ema` | 每个有效历元都用慢 EMA 更新 | 否 |
| `relative_gated` | 仅 gate 高置信预测正常时更新 | 否，但需要额外 gate 模型 |

对频段 $b\in\{L1,L5\}$，记当前 band-mean C/N0 为 $k_b(t)$、更新前在线基线为 $x_b(t^-)$。输入残差和更新为：

```text
d_b(t) = k_b(t) - x_b(t^-)
alpha  = exp(-ln(2) * delta_t / 60s)
x_b(t) = alpha * x_b(t^-) + (1 - alpha) * k_b(t)
```

特征严格先计算、后更新，因此时刻 `t` 的 gate 决策只影响未来时刻。最终每个历元固定为 6 维：

```text
L1_Cn0Relative, L5_Cn0Relative,
L1_Cn0AbsRelative, L5_Cn0AbsRelative,
L1Present, L5Present
```

这里没有保留绝对 C/N0，只保留带符号残差、绝对残差和频段存在标记。连续四维仍用 outer-train 的 global scaler 标准化；scaler 只在训练集唯一窗口 endpoint 上拟合，避免同一历元因 W5 重叠而被重复计权。

这不是第九节 4 维模型的“只替换 scaler”消融。`per_device/global` 使用 `L1_Cn0DbHz`、`L5_Cn0DbHz`、`L1Present`、`L5Present`，本轮两条路线则以 6 维因果残差替换绝对 C/N0。后续总表比较的是完整可部署路线，不能把差值只归因于 EMA 或 scaler 中的某一个因素。

### 10.2 因果边界和 gated 防泄漏流程

固定实验条件与第九节一致：`mixed_timeblock_outer_cv4_w5_v2` 四折 outer recording holdout、W5、TCN32、dropout 0.1、seed 2026、最多 40 epoch、反频率加权 CE；纯静态用 `--scope static`，静态+动态用 `--scope all`。

在线状态在 recording、原始 source 文件、device、train/val/test split、连续 segment 或超过 2 秒的 receiver gap 处重置。L1/L5 分别初始化和更新，某个频段缺失不会推动另一个频段的基线。这样保证 val/test 不继承 train 状态，也不会通过重叠窗口或后续标签回流；代价是每个新流都存在冷启动。

`relative_gated` 的 gate 也是同规格四分类 TCN32，使用 `relative_ema` 六维输入，并取四分类 softmax 的 `P(normal)`：

```text
前 W-1=4 个历元：因 gate 尚无完整窗口，允许 warmup 更新
后续历元：P(normal) >= 0.8 才更新，否则冻结
```

gate 训练采用 recording-grouped cross-fitting：

1. outer-train 的每条录制只接收未见过该录制的 OOF gate 预测。
2. outer-val 和 outer-test 的 gate 预测来自只用 outer-train 拟合的 full gate。
3. `(source_id, device_id, window_time_nanos)` 键、概率范围、split 覆盖率和 source mapping 均写入审计 manifest；四折 gate 与最终分类器 endpoint 覆盖率均为 100%。

因此本轮不存在“gate 先见到同一 outer-test 录制或其标签”的捷径。需要注意，gate 仍是反频率加权的四分类器，其 softmax 分数没有做概率校准；`0.8` 只是预先固定的工作阈值，不代表统计意义上校准后的 80% 正常概率。

### 10.3 四折 outer-test 主结果

| 数据范围 | 路线 | 输入 | Test Macro-F1 | Test Accuracy | Pixel6 L5 recall |
|---|---|---|---:|---:|---:|
| 纯静态 | `per_device` | 4 维绝对 C/N0 | **0.991749** | **0.994471** | **0.981419** |
| 纯静态 | `global` | 4 维绝对 C/N0 | 0.966601 | 0.974200 | 0.076014 |
| 纯静态 | `relative_ema` | 6 维因果残差 | 0.308087 | 0.356133 | 0.408784 |
| 纯静态 | `relative_gated` | 6 维 gated 残差 | 0.436958 | 0.445334 | 0.013514 |
| 静态+动态 | `per_device` | 4 维绝对 C/N0 | **0.925215** | **0.958944** | **0.888554** |
| 静态+动态 | `global` | 4 维绝对 C/N0 | 0.896820 | 0.937763 | 0.308735 |
| 静态+动态 | `relative_ema` | 6 维因果残差 | 0.330834 | 0.453929 | 0.259036 |
| 静态+动态 | `relative_gated` | 6 维 gated 残差 | 0.467123 | 0.503091 | 0.064759 |

`relative_gated` 相比持续 EMA 有明显改善，但仍与 `global` 相差约 0.43 Macro-F1，更远低于 `per_device`。这已经不是调整 dropout、hidden dim 或训练 epoch 能解释的零点零几差距。

gated 的逐类 recall 为：

| 数据范围 | normal | L1 | L5 | L1+L5 |
|---|---:|---:|---:|---:|
| 纯静态 | 0.393329 | 0.261776 | 0.522910 | 0.845704 |
| 静态+动态 | 0.452110 | 0.689621 | 0.514351 | 0.760232 |

它不是简单地“全部预测攻击”：normal recall 只有 0.39/0.45，同时 L1、L5 的 precision 也很低。纯静态的 L1 precision 仅 0.1045、L5 precision 仅 0.2483；mixed 分别为 0.2625、0.1934。六维残差表示没有形成稳定的正常/攻击分界。

Pixel6 L5 仍是关键反例。纯静态同一组 592 个 L5 窗口，`relative_ema` 识别 242 个，gated 仅识别 8 个；mixed 中 Pixel6 的 664 个 L5 窗口，分别识别 172 个和 43 个。门控更新没有解决此前全局标准化暴露出的 Pixel6 L5 问题，反而使其进一步恶化。

### 10.4 Gate 更新审计

| 数据范围 | gate 行数 | `P(normal)>=0.8` | 高置信比例 | 实际允许更新率（含 warmup） | 冻结率 |
|---|---:|---:|---:|---:|---:|
| 纯静态 | 110,534 | 9,169 | 8.295% | 5.696% | 94.304% |
| 静态+动态 | 168,595 | 6,503 | 3.857% | 4.210% | 95.790% |

“高置信比例”以有 gate 预测的 W5 endpoint 为分母；“更新/冻结率”以全部 eligible epoch 为分母，缺 gate 的非 warmup 历元也会冻结，所以两列不能直接相加。

折间差异非常大：static 四折高置信比例依次为 `0.11% / 10.99% / 6.58% / 15.51%`，mixed 为 `0.46% / 1.41% / 1.06% / 12.51%`。这说明反频率加权四分类 softmax 的 `P(normal)` 没有跨折可比的概率含义，固定 `0.8` 门槛导致大部分时间冻结，并且不同折的更新行为完全不同。

### 10.5 CPU 推理开销

同第九节测时环境：CPU 单线程、预热 200 次、batch=1 测 1,000 次、batch=256 测 100 次。表中为四折中位数：

| 数据范围 | 路线 | batch=1 p50 | batch=1 p95 | batch=256 p50 | 总参数量 |
|---|---|---:|---:|---:|---:|
| 纯静态 | `relative_ema` | 0.1576 ms | 0.1952 ms | 0.4991 ms | 4,964 |
| 纯静态 | `relative_gated` | 0.3158 ms | 0.4885 ms | 1.0237 ms | 9,928 |
| 静态+动态 | `relative_ema` | 0.1571 ms | 0.1970 ms | 0.5032 ms | 4,964 |
| 静态+动态 | `relative_gated` | 0.3177 ms | 0.4910 ms | 1.0264 ms | 9,928 |

gated 数据是 gate+classifier 两次顺序前向的实测，不是单模型耗时乘二的估算。它仍不含日志解析、band mean 聚合、global scaler 和 EMA 标量更新。两次小 TCN 前向在桌面 CPU 上仍低于 0.5 ms p95，但当前精度不足，因此低延迟不能构成采用理由。

### 10.6 失败原因与工程结论

本轮结果支持以下判断：

1. **持续 EMA 会吸收持续攻击。** 一旦异常持续时间接近或超过 60 秒半衰期，攻击后的 C/N0 会逐渐成为新基线，残差随之消失。
2. **gated 缓解漂移，但 gate 本身不可靠。** class-weighted 四分类 softmax 未校准，固定阈值折间极不稳定；错误冻结和错误更新又会递归改变后续输入。
3. **没有可信冷启动。** 每个 split/segment 的首个观测直接初始化基线；若新流从攻击区间开始，初值本身就是受污染的。这个在线初值不等价于老师所说的“已采集正常开阔环境基线”。
4. **表示丢失了绝对信息。** 只有相对值和绝对残差，无法保留设备当前 C/N0 绝对水平、L1/L5 绝对关系以及老师建议的多维正常偏差。gated 无法恢复构造特征时已经丢掉的信息。
5. **频繁重置是必要约束，也是现实代价。** source/split/segment/gap 重置避免状态泄漏，却使离线结果诚实地暴露出部署时每次启动都需要校准的问题。

因此，`causal_relative_v1` 应记录为**已完整验证的负结果**，不进入当前场景分支主线。继续盲调 TCN 宽度、dropout、EMA 半衰期或 `0.8` 阈值不值得；当前差距首先是基线可辨识性、概率校准和输入信息量问题。

当前工程选择仍是：已知设备、允许预先校准时保留 `per_device`；若研究目标明确要求未见设备部署，则必须单独建立 leave-one-device-out 评估，不能用 recording-CV 的 0.99/0.93 结果代替。

下一轮更有信息量的路线按优先级为：

1. 明确部署协议，给每台新设备一段**确认正常的开阔场景校准期**；从该段拟合 C/N0、AGC 等多维正常基线，再做跨设备 held-out 测试。
2. 保留 global 标准化下的绝对 C/N0，同时追加相对残差，做 `absolute + relative` 混合输入消融，验证本轮失败是否主要来自绝对信息丢失。
3. 只有在必须无校准在线启动时，才继续 gated：先训练独立的 `normal vs attack` 二分类 gate，在 outer-train OOF 预测上做概率/阈值校准，再加入滞回、最短正常持续时间和单步更新限幅。所有阈值仍只能由 outer-train/val 决定。

### 10.7 产物与复现入口

```text
output/tensors/causal_relative_v1/
  static_ema/ mixed_ema/ static_gated/ mixed_gated/

output/training/causal_relative_v1/
  <四个变体>/fold_1..4/
  <四个变体>/aggregate_test_metrics.json
  <四个变体>/aggregate_test_predictions.csv
  <四个变体>/prediction_summary/

output/causal_gate_crossfit/
  static_gated/ mixed_gated/
```

相关入口：

- [45_build_band_mean_window_tensors.py](../pipeline_total/45_build_band_mean_window_tensors.py)：构造 EMA/gated 因果特征与唯一 endpoint scaler。
- [46_train_band_mean_multiclass.py](../pipeline_total/46_train_band_mean_multiclass.py)：训练、按 split 导出预测并校验 checkpoint 元数据。
- [47_aggregate_band_mean_cv.py](../pipeline_total/47_aggregate_band_mean_cv.py)：四折聚合与因果参数透传。
- [56_measure_band_mean_runtime.py](../pipeline_total/56_measure_band_mean_runtime.py)：单模型或 gate+classifier 双模型 CPU 测时。
- [57_run_causal_gated_cv.py](../pipeline_total/57_run_causal_gated_cv.py)：防泄漏 OOF gate 编排和审计 manifest。
- [58_summarize_band_mean_predictions.py](../pipeline_total/58_summarize_band_mean_predictions.py)：overall、逐类、逐设备、逐录制汇总。
- [test_causal_band_mean_features.py](../tests/test_causal_band_mean_features.py)：因果性、重置、缺频段、gate 键和 scaler 范围测试。
- [test_band_mean_pipeline_contracts.py](../tests/test_band_mean_pipeline_contracts.py)：checkpoint/tensor 绑定、fold/endpoint 唯一性和录制覆盖完整性测试。

---

## 十一、绝对 C/N0 + 模型自更新在线基线（2026-08-05）

### 11.1 这次实现的协议

第十节的 `relative_ema/gated` 是“仅保留残差”的负向消融，不能回答“保留当前绝对 C/N0、再把在线基线作为额外输入”这一设想。本节单独实现该设想，输入固定为：

```text
[L1_Cn0DbHz(t), L5_Cn0DbHz(t), L1Present(t), L5Present(t),
 x_L1(t-),   x_L5(t-)]
```

其中前四维是原有的绝对观测，后两维是当前时刻推理前可用的在线基线。对每个频段的有效观测 `k_b(t)`，状态机严格按下列顺序执行：

```text
features(t) 使用 x_b(t-)
pred(t) = argmax(classifier(W5 ending at t))
若 pred(t) == normal (class 0):
    x_b(t) = 0.98 * x_b(t-) + 0.02 * k_b(t)   # 仅更新当前存在的频段
否则:
    x_b(t) = x_b(t-)
```

因此当前预测只会影响未来输入。训练、验证和测试均使用模型自身此前的离散预测更新状态；标签只进入当前端点的交叉熵和最终指标，绝不参与更新。没有外部 gate、置信度阈值或 teacher forcing。

所有端点（含单频端点）均进行前向推理，单频端点只从四分类损失、指标和预测 CSV 中排除；若其被预测为正常，只更新其实际存在的频段。这保证“模型预测正常才更新”的规则没有在缺频段处被悄悄改写。状态会在 recording、source、train/val/test split、manifest segment 和超过 2 秒的 receiver gap 处重置；每个新流以该流首个可用频段 C/N0 冷启动。

为避免重新引入设备泄漏，连续特征仍仅由 outer-train 拟合一套 `global` mean/std，未使用该设备的训练均值或标准差。在线基线在标准化空间中更新，与先在原始 C/N0 空间 EMA 再应用同一线性 global scaler 等价。

模型保持 W5、TCN32、dropout 0.1、4,964 参数。先训练四维绝对 C/N0 TCN，再把其输入投影扩展为六维，新增两列权重初始化为 0。这个 warm start 的“epoch 0”与四维模型完全等价，故也纳入 validation checkpoint 选择；若后续在线微调没有改善验证集，就保留 epoch 0，而不是人为接受退化。

### 11.2 四折 outer-test 结果

协议仍为 `mixed_timeblock_outer_cv4_w5_v2`、四折 recording holdout、seed 2026。四分类指标只统计双频端点，故与第九、十节的主表口径一致。

| 数据范围 | 路线 | Test Macro-F1 | Test Accuracy | 相对 4 维 global 的 Macro-F1 变化 |
|---|---|---:|---:|---:|
| 纯静态 | 4 维绝对 C/N0 / `global` | 0.966601 | 0.974200 | - |
| 纯静态 | 绝对 C/N0 + 在线 `x_L1/x_L5` | 0.966254 | 0.974058 | -0.000347 |
| 静态+动态 | 4 维绝对 C/N0 / `global` | 0.896820 | 0.937763 | - |
| 静态+动态 | 绝对 C/N0 + 在线 `x_L1/x_L5` | 0.894044 | 0.936458 | -0.002776 |

纯静态基本持平，mixed 下降约 0.28 个百分点，均没有出现可称为质变的提升。mixed 的逐类 F1 为 normal `0.9611`、L1 `0.8096`、L5 `0.8710`、L1+L5 `0.9345`；相对四维 global 的对应值均略低或近似持平。

这说明两件事：

1. 这条路线避免了第十节残差-only 表征丢弃绝对信息造成的崩溃，且在已知设备 recording-CV 下稳定可运行。
2. 在当前数据、固定 `alpha=0.98` 和当前 TCN 容量下，模型没有学到能稳定利用这两条在线基线通道的额外判别信息。静态四折的选择轮次为 `0/1/0/0`，mixed 为 `0/1/0/4`；多个 outer fold 的最优 checkpoint 正是 epoch 0 warm start，说明微调常常没有带来有效增益。

因此本结果应作为“直接实现后未发现收益”的严谨消融，而不是把它与 `relative_ema/gated` 的失败混为一谈。它也不能证明所有在线基线都无效：可信正常冷启动、不同的更新动力学、绝对值与显式差值的联合输入，以及 leave-one-device-out 协议仍是独立问题；但在这些条件改变前，不应把它当作当前场景分支的主模型替换方案。

### 11.3 完整在线推理开销

新增测时器把基线特征构造、TCN 前向、`argmax` 和条件状态更新作为一个完整端点决策计时；不包括文件读取、原始日志解析、band-mean 聚合、global scaler、指标和 CSV 写出。CPU 单线程、每折预热 100 次，以下为四折中位数，单位均为毫秒/端点：

| 数据范围 | 单流完整循环平均 | 单流 p50 | 四折中最大 p95 |
|---|---:|---:|---:|
| 纯静态 | 0.2325 | 0.2143 | 0.3497 |
| 静态+动态 | 0.2548 | 0.2194 | 0.4534 |

跨 stream 的 round-robin batch=256 补充测量中，static fold 1 为 `11,864 endpoint/s`（`0.0843 ms/endpoint`），mixed fold 1 为 `15,304 endpoint/s`（`0.0653 ms/endpoint`）。这两个吞吐数受各折 stream 长度和组成影响，只用于说明批处理能力，不能当作单设备实时延迟。

对 static fold 1 和 mixed fold 1，单流与 batch rollout 导出的可评分端点预测 checksum 完全一致，确认计时器没有改变状态更新语义。

### 11.4 产物与复现入口

```text
output/tensors/online_cn0_baseline_v2/
  static_global_alpha0p9800/fold_1..4/
  mixed_global_alpha0p9800/fold_1..4/

output/training/online_cn0_baseline_v2/
  <scope>_global_alpha0p9800_absolute_warm_start/fold_1..4/
  <scope>_global_alpha0p9800/fold_1..4/
    best_online_cn0_baseline_tcn.pt
    val_metrics_online_cn0_baseline_tcn.json
    test_predictions_band_mean_window_tcn.csv
    runtime_online_rollout_single.json
  <scope>_global_alpha0p9800/
    aggregate_test_metrics.json
    aggregate_test_predictions.csv
```

主入口：

- [59_train_online_cn0_baseline.py](../pipeline_total/59_train_online_cn0_baseline.py)：严格时序 rollout、epoch-0 warm-start 选择、checkpoint/tensor 合约检查。
- [60_run_online_cn0_baseline_cv.py](../pipeline_total/60_run_online_cn0_baseline_cv.py)：每折构建 global 绝对张量、四维 warm start、在线训练/测试和四折聚合编排。
- [61_measure_online_cn0_rollout_runtime.py](../pipeline_total/61_measure_online_cn0_rollout_runtime.py)：包含状态机的真实在线 CPU 测时。
- [test_online_cn0_baseline_rollout.py](../tests/test_online_cn0_baseline_rollout.py)：预测只影响未来、异常冻结、标签隔离、单频更新、segment 重置和 rollout 策略测试。

例如，完整静态或 mixed 复现可运行：

```bash
python pipeline_total/60_run_online_cn0_baseline_cv.py --scope static
python pipeline_total/60_run_online_cn0_baseline_cv.py --scope all
```

并行折运行时加入 `--folds <N> --no-aggregate`，待四个 fold 完成后再执行：

```bash
python pipeline_total/47_aggregate_band_mean_cv.py \
  --protocol-dir output/protocols/mixed_timeblock_outer_cv4_w5_v2 \
  --training-root output/training/online_cn0_baseline_v2/<scope>_global_alpha0p9800 \
  --encoder tcn --aggregate-only --folds 1 2 3 4
```

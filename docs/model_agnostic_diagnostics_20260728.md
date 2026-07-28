# 模型无关诊断快照（2026-07-28）

本文记录一组**模型无关**的诊断，目的不是提出新模型，而是先回答一个此前从未被直接测量的问题：当前 `dy_L5` 等难例的失败，究竟是**信息不存在**（compact11 + W5 特征里没有足够区分欺骗的信息），还是**信息存在但逐信号架构取不到**。

四项诊断都只读 inner validation 或中央 CSV，**从不读取 outer test**，因此不构成新的盲测，也不参与模型选择。脚本：

- `pipeline_total/41_diagnose_information_ceiling.py`（诊断 1、2）
- `pipeline_total/42_audit_raw_field_coverage.py`（诊断 4）
- `pipeline_total/43_diagnose_device_separability.py`（诊断 3）

产物在被 Git 忽略的 `output/diagnostics/`。数据源为保留的 `mixed_timeblock_outer_cv4_w5_v2` 四折张量与 `output/processed_gnss_data.csv`。

## 关键口径提醒

诊断 1/2 的强模型（sklearn `HistGradientBoostingClassifier`，直方图梯度提升，同类 LightGBM）在每折的 **inner validation** 上评估。validation 时间块与 train 来自同一批 development Session，因此是**乐观估计**（同 Session 泄漏），不能与 TCN16 的 **outer-test**（完全留出 Session）数字直接排序。它回答的是上限方向，不是泛化性能。

---

## 诊断 1：信息上限探测（强模型，固定 compact11+W5 特征）

在与 TCN16 完全相同的 split、特征、标签下，换用一个不受参数量约束的强模型。4 折 inner-val 均值：

| 子集 | Macro-F1 | Recall | FAR |
|---|---:|---:|---:|
| overall | 0.9137 | 84.59% | 3.12% |
| static | 0.9544 | 92.10% | 1.94% |
| dynamic | 0.7663 | 56.39% | 4.92% |

逐场景 `dy_L5`（4 折）：

| fold | Macro-F1 | Recall | FAR | 正类 support |
|---:|---:|---:|---:|---:|
| 1 | 0.6506 | 38.69% | 3.82% | 672 |
| 2 | 0.5945 | 24.23% | 3.89% | 1395 |
| 3 | 0.5822 | 26.51% | 6.06% | 2067 |
| 4 | 0.5831 | 21.48% | 4.83% | 2067 |
| 均值 | ~0.60 | 27.73% | 4.65% | — |

**结论**：即使在乐观的同 Session validation 上、即使换用不受参数量限制的强模型，`dy_L5` Recall 也只能到约 28%（低 FAR 下）。TCN16 在更难的 outer test 上是约 19%。两者差距远小于"信息充足但模型太弱"应有的差距。这强烈指向：**在 compact11 + W5 特征张成的空间里，`dy_L5` 已接近信息上限，瓶颈不是 TCN16 的容量。** 继续扩大模型或换时序结构，预期不会救回 `dy_L5`。

## 诊断 2：捷径探测（仅设备 / 场景 / 频段元数据）

完全不给任何信号特征，只用 `device + scenario + band` 元数据训分类器。4 折 inner-val Macro-F1：0.6042 / 0.6405 / 0.5855 / 0.6021，**均值 0.6081**。

**结论**：仅凭"这是哪台设备、哪个场景、哪个频段"就能拿到 0.61 Macro-F1。这是**捷径下界**——因为 scenario 强编码了攻击类型（`st_L1` 场景几乎都是 L1 攻击等），组身份与标签高度相关。这解释了为什么 state-stratified validation 能把 validation 从 0.898 拉到 0.930 而 outer test 不动：模型在学"组→标签"的相关性。完整模型（val 0.91）减去捷径下界（0.61）才是真正来自信号特征的贡献，评估时必须警惕这条捷径。

## 诊断 3：逐设备可分性（中央 CSV 原始未标准化 C/N0、AGC）

用未经 per-device 标准化的原始值，量化每台设备 clean vs attack 分布。张量是 per-device 标准化过的，会先天抹掉这里的跨设备差异，所以必须回到中央 CSV。

### 发现 A：L5 攻击下 C/N0 变化方向**因设备而相反**

L5 场景各设备 C/N0 中位数（clean → attack）：

| 设备 | clean p50 | attack p50 | 方向 |
|---|---:|---:|---|
| HUAWEI Mate40 | 31.0 | 40.0 | **上升** |
| RedMi K60 | 28.6 | 42.6 | **上升** |
| XiaoMi MI8 | 35.3 | 42.3 | **上升** |
| Google Pixel6 | 36.1 | 26.6 | **下降** |

Mate40/K60/MI8 在 L5 欺骗下 C/N0 **升高**，而 Pixel6 反而 **降低**。这直接解释了 Fold 6 里"设备错误方向相反"的现象：任何键控"C/N0 升高 = 攻击"的模型都会在 Pixel6 上系统性漏检（对应 Fold 6 诊断中 Pixel6 recall 仅约 4%）。**单一全局阈值在物理上就不可能同时覆盖这两个方向。**

### 发现 B：绝对 AGC 是纯设备身份

各设备 clean AGC 中位数：Pixel6 ≈ 29.7、Watch2 ≈ −56.6、Mate40 ≈ −2、K60 ≈ 6.0（std 0.4，几乎恒定）、MI8 ≈ 48.7。跨设备 AGC 分布重叠接近 0（不同设备 AGC 量程完全不重叠）。RedMi K60 的 AGC 基本量化在 6.0 常量，**几乎不携带信息**。Pixel Watch1/2 无 AgcDb。

**结论**：绝对 AGC 编码的是"哪台设备"而非"是否被攻击"。这是设备域偏移的直接来源，也印证了"因果相对基线（相对每 Session 起始清洁段的 robust z-score）"从"值得试"上升为"几乎必须做"。

### 发现 C：单特征 clean/attack 重叠普遍很高

L5 场景同设备 clean vs attack 的 C/N0 分布重叠：Mate40 0.62、K60 0.50、MI8 0.43、Pixel6 0.39。即便在攻击方向一致的设备上，仅凭单点 C/N0 也难以把 clean 与 attack 分开——这与诊断 1 的信息上限一致。

## 诊断 4：原始日志字段覆盖率审计

`04_build_labeled_processed_csv.py` 解析时读入全部 38 列 Android 原始字段，但只把其中约 10 列写进 processed CSV，丢弃了 28 列。审计 123 份日志、7 个设备目录（含新旧 Pixel6 命名）后，被丢弃字段的跨设备可用性（`devs_usable` / 存在率范围）：

**广泛可用（值得优先考虑重建）：**

| 字段 | 可用设备 | 说明 |
|---|---|---|
| `PseudorangeRateMetersPerSecond` | 7/7 | **实际伪距率**。当前只保留了它的 uncertainty，没保留值本身；欺骗会操纵伪距率，这是最值得补的字段 |
| `State` | 7/7 | tracking state 位掩码，反映锁定/跟踪质量 |
| `FullBiasNanos` / 时钟字段 | 7/7 | 接收机时钟状态 |
| `AccumulatedDeltaRangeMeters` / `State` | 6/7（MI8 缺） | 载波积累距离，欺骗切换时相位不连续会跳变 |
| `DriftNanosPerSecond` | 6/7 | 时钟漂移 |

**设备受限（不建议投入）：**

| 字段 | 可用设备 | 原因 |
|---|---|---|
| `CarrierPhase` / `CarrierCycles` | 0/7 | 全部设备都无有效载波相位 |
| `SnrInDb` | 1/7 | 仅 Pixel6 |
| `BasebandCn0DbHz` | 5/7 | 手表和 MI8 缺失，与 AGC 缺失设备重叠 |

**结论**：确实存在被丢弃、但**广泛可用**的物理字段，其中 `PseudorangeRateMetersPerSecond`（实际伪距率）、`AccumulatedDeltaRangeMeters`（载波积累距离）和 `State`（跟踪状态）与欺骗机制直接相关，且不依赖易缺失的 AGC/基带通道。这些是重建中央 CSV 时投入产出比最高的候选。载波相位、SnrInDb 因设备覆盖太差不值得投入。

---

## 综合判断

1. **`dy_L5` 是信息问题，不是容量问题**（诊断 1）。在现有 compact11+W5 特征下，最强模型在最乐观的评估上也到不了 30% recall。应停止"在固定特征上换更大/更复杂时序模型"这类尝试。

2. **两条可能真正扩大信息量的路，方向一致**：
   - **重组特征**：因果相对基线消除绝对 AGC 的设备身份（诊断 3B）；跨卫星/跨频段共模特征捕捉逐信号结构看不到的联合响应；补入 `PseudorangeRateMetersPerSecond`、`AccumulatedDeltaRangeMeters`、`State` 等被丢弃的物理字段（诊断 4）。
   - **设备条件化**：C/N0 在 L5 下变化方向因设备而相反（诊断 3A），单一阈值物理上无解，必须做设备条件化或相对化，而非再调全局模型。

3. **评估必须换口径**：捷径下界 0.61（诊断 2）说明 Overall/pooled 指标被组身份和长静态 Session 污染。主指标应改为 Dynamic Session 等权、`dy_L5` 在固定 FAR 下的 Recall、最差设备 Recall。

4. **数据集已固定**，无法再采集。因此上述"重组特征 + 设备条件化"是在现有数据上唯一还可能扩大有效信息的方向；任何依赖新采集的设想都不在当前可行域内。

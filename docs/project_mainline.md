# 面向真实多设备部署的轻量化 GNSS 导航欺骗检测方法研究

## 1. 最新主线定位

当前数据状态、设备级模型结果和开发参考 test 的使用边界见 [experiment_progress.md](experiment_progress.md)。本文件聚焦研究定位与长期路线，不重复维护逐项实验数值。

本项目后续主线建议确定为：

**面向真实多设备部署的轻量化 GNSS 导航欺骗检测方法研究。**

一句话概括：

> 利用操场与新主楼两套真实多设备导航欺骗数据，构建只依赖手机、手表、u-blox 等真实设备可获得少量 GNSS Raw 特征的轻量化检测模型，并重点验证其跨设备、跨环境、跨频段和实际部署能力。

这篇论文不建议继续扩展成“欺骗与干扰联合检测”。`Interference Data` 暂时不进入主线，只作为后续备选探索数据。

---

## 2. 为什么这样定主线

已有 FGI-GSRx / Finland 数据实验表明，富特征条件下可以实现高精度欺骗与干扰检测；这类软件接收机特征较丰富，例如 I/Q、相关能量、跟踪环误差、FLL/PLL/DLL 等。

如果继续在类似富特征数据上追求更高精度，研究增量不够明显。更有价值的方向是转向真实部署场景：

- 真实手机、手表、u-blox 接收机；
- 可获得特征更少；
- 设备差异明显；
- 环境变化明显；
- 模型需要轻量化，后续可能实际部署。

因此，本研究的核心不应是“提出一个更复杂的新模型”，而应是：

**用轻量模型，在真实多设备、多环境、少特征条件下实现可靠的导航欺骗检测。**

---

## 3. 数据使用规划

### 3.1 主数据一：操场数据

路径：

```text
H:\GNSS\lightweight_gnss_spoofing_detection\data_raw\playground
```

这是之前已有的操场数据，包含：

```text
st_L1
st_L5
st_L_15
dy_L1
dy_L5
dy_L_15
```

主要价值：

- 已有处理基础；
- 覆盖静态/动态欺骗；
- 覆盖 L1、L5、L1+L5；
- 包含多设备数据；
- 可作为第一套真实环境数据。

### 3.2 主数据二：新主楼数据

路径：

```text
H:\GNSS\lightweight_gnss_spoofing_detection\data_raw\new_building
```

主要价值：

- 与操场数据形成跨环境对照；
- 同样包含静态/动态、L1/L5/L1+L5；
- 包含多设备，如 HUAWEI Mate40、Xiaomi MI8、RedMi K60、Pixel Watch 等；
- 附带完整 pipeline，可用于生成特征、画图、人工标注、训练和推理。

### 3.3 辅助数据：Finland / FGI-GSRx 数据

路径示例：

```text
H:\GNSS\Finland L1_E1 data\final_mat
```

用途：

- 作为富特征软件接收机数据；
- 可用于说明富特征模型虽然精度高，但难以直接部署到真实手机/手表；
- 可作为 Teacher 模型或强基线的来源；
- 不作为真实部署主数据。

### 3.4 暂不纳入主线：Interference Data

路径：

```text
H:\GNSS\Interference Data
```

暂不纳入主线的原因：

- 标签可信度暂不确定；
- 不清楚其中到底是干扰、欺骗，还是混合异常；
- 如果贸然纳入训练，会削弱论文标签可靠性；
- 容易让论文主线从“轻量化导航欺骗检测”发散成“复杂异常综合检测”。

建议处理方式：

```text
保留数据，不删除；
暂不进入主实验；
后续若标签确认，可作为扩展实验或后续第三篇方向。
```

---

## 4. 研究目标

### 4.1 真实设备少特征检测

主模型只使用真实设备可获得的少量 GNSS Raw 特征，不依赖软件接收机内部富特征。

推荐核心特征：

```text
Cn0DbHz
Cn0DbHz_dt
Cn0DbHz_std
AgcDb
ReceivedSvTimeUncertaintyNanos
PseudorangeRateUncertaintyMetersPerSecond
FreqBand
```

其中，重点特征是：

- C/N0 及其变化；
- AGC；
- 卫星时间不确定度；
- 伪距率不确定度；
- 频段与设备信息。

### 4.2 轻量化模型部署

本项目重点不是追求最大模型精度，而是实现适合真实设备部署的模型。

需要重点关注：

```text
低参数量
低 FLOPs / MACs
低推理延迟
低内存占用
小模型体积
可 ONNX / Android / 边缘端部署
```

### 4.3 跨域鲁棒性

必须验证模型不是只在随机划分下有效，而是在真实部署的 domain shift 下仍然可靠。

重点实验：

```text
操场训练 -> 新主楼测试
新主楼训练 -> 操场测试
leave-one-device-out
leave-one-frequency-out
static -> dynamic
dynamic -> static
```

### 4.4 检测时效性

后续实际部署时，不能只看 Accuracy，还要看攻击发生后多久报警。

建议增加 TTD 指标：

```text
median TTD
95% TTD
false alarms per minute
```

---

## 5. 推荐论文题目方向

英文题目可考虑：

```text
面向真实多设备部署的轻量化 GNSS 导航欺骗检测方法研究
```

中文定位可写成：

```text
面向真实多设备部署的轻量化 GNSS 导航欺骗检测方法研究
```

也可以简化为：

```text
面向真实多设备部署的轻量化 GNSS 导航欺骗检测方法研究
```

---

## 6. 模型路线

### 6.1 第一阶段：轻量 baseline 跑通

优先跑稳定、容易解释的 baseline：

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

目标不是一开始就追求最复杂结构，而是先证明少特征轻量模型能在真实数据上工作。

### 6.2 第二阶段：形成主模型

主模型可以设计为：

```text
轻量时序模型
+ 少特征输入
+ 设备归一化
+ 频段嵌入或频段条件化
+ 滑窗检测
+ 早停报警策略
```

模型应突出：

- 参数少；
- 速度快；
- 可部署；
- 跨设备稳定；
- 跨环境稳定。

### 6.3 第三阶段：可选知识蒸馏

如果时间允许，可以加入知识蒸馏：

```text
RS-TimesNet Teacher -> Lightweight Student
```

作用：

- 利用富特征模型的经验；
- 提升轻量模型性能；
- 形成富特征检测能力向轻量化真实设备检测能力的自然迁移。

但注意：知识蒸馏不是第一优先级。第一优先级是先把真实多环境轻量检测跑通。

---

## 7. 实验矩阵

### 7.1 基础实验

```text
同环境随机划分
操场训练，新主楼测试
新主楼训练，操场测试
操场 + 新主楼混合训练，分环境测试
```

### 7.2 跨设备实验

```text
leave-one-device-out
```

例如：

```text
留出 HUAWEI 测试
留出 Xiaomi 测试
留出 Watch 测试
留出 Pixel / RedMi 测试
```

### 7.3 跨频段实验

```text
L1 训练 -> L5 测试
L5 训练 -> L1 测试
L1 + L5 训练 -> L1/L5 分别测试
```

### 7.4 静态/动态实验

```text
static -> dynamic
dynamic -> static
static 和 dynamic 分开报告
```

### 7.5 特征消融实验

建议至少做：

```text
C/N0 only
C/N0 + AGC
C/N0 + uncertainty
C/N0 + AGC + uncertainty
去掉 DeviceName
去掉 FreqBand
全特征
```

### 7.6 模型复杂度与部署实验

必须报告：

```text
Params
FLOPs / MACs
Model size
CPU latency
Memory usage
Accuracy
Macro-F1
FAR
TTD
```

如果条件允许，进一步报告：

```text
ONNX latency
Android device latency
edge CPU latency
```

---

## 8. 论文贡献设计

建议写成四点贡献。

### 贡献一：真实多设备、多环境导航欺骗数据流程

整合操场和新主楼两套真实采集数据，统一处理、标注和划分协议，为真实部署场景下的 GNSS 导航欺骗检测提供实验基础。

### 贡献二：少特征轻量化检测框架

提出仅依赖真实设备可获得 GNSS Raw 特征的轻量化检测框架，避免依赖软件接收机内部富特征，提高实际部署可行性。

### 贡献三：跨环境、跨设备、跨频段系统验证

不仅进行随机划分，还重点验证模型在跨环境、跨设备、跨频段下的泛化能力，更贴近真实部署需求。

### 贡献四：部署导向评估

除 Accuracy 和 F1 外，同时报告虚警率、检测时间、参数量、FLOPs、推理延迟和模型大小，证明模型适合实际部署。

---

## 9. 后续执行顺序

建议按以下顺序推进：

1. 统一操场和新主楼的数据目录结构。
2. 用 pipeline 生成两套数据的 `plot_features.csv`。
3. 可视化 C/N0、AGC、uncertainty 等特征。
4. 手工复核欺骗区间标签。
5. 生成统一的 `processed_gnss_data.csv`。
6. 构建训练、验证、测试张量。
7. 先跑轻量 baseline。
8. 做跨环境测试。
9. 做跨设备、跨频段测试。
10. 做特征消融。
11. 统计参数量、FLOPs、延迟和模型大小。
12. 增加 TTD 检测时间评估。
13. 再考虑知识蒸馏和部署优化。

---

## 10. 当前最终判断

本项目后续不建议走“大模型刷精度”路线，也不建议现在扩展成标签不稳定的欺骗/干扰联合检测。

最稳、最有论文价值、也最符合后续实际部署的主线是：

> 不是做最大最复杂的 GNSS 欺骗检测模型，而是做一个能在真实多设备、多环境中稳定工作的轻量化、可部署 GNSS 导航欺骗检测框架。

这条主线聚焦、数据可信、工程价值明确，也更适合围绕后续实际部署持续推进。



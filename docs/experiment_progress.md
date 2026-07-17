# 当前实验进展与结果边界

更新时间：2026-07-18

## 1. 已完成的数据与标签工作

- 操场与新主楼共保留 132 份原始 GNSS TXT 日志，其中操场 98 份、新主楼 34 份；
- 已删除 1 份与 `new_building/st_L1` 重复的原始日志，并重建数据清单；
- 统一 CSV 包含 2,139,284 条 `reviewed` 信号级记录；
- 新主楼静态与动态 Session 均已按独立 TOW 区间人工复核；
- 使用 `signal_id = sv_id + SignalBand + CodeType` 组织独立信号时序；
- 默认逐信号模型输入为 7 项特征：`Cn0DbHz`、`Cn0DbHz_dt`、`Cn0DbHz_std`、`AgcDb`、`ReceivedSvTimeUncertaintyNanos`、`PseudorangeRateUncertaintyMetersPerSecond`、`FreqBand`；ADR 不进入默认输入；
- 窗口标签固定为末端当前历元的标签，而不是窗口内标签的最大值。

## 2. 建模路线演进

第一轮逐卫星模型将每条 `signal_id` 独立分类。错误诊断表明：同一设备同一时刻的多数卫星预测正常时，少数卫星误报不应直接等同于设备告警错误；L1/L5 单频欺骗时，设备内出现真实标签 0/1 混合也属于预期现象。

因此新增两条设备级路线：

1. **逐卫星预测后聚合：** 使用 `any`、`majority`、`k_of_n`、`ratio` 等规则形成设备告警。
2. **直接设备级建模：** 同一设备当前历元聚合有效卫星的统计状态，直接预测设备是否受欺骗。

直接设备级输入为 27 维：6 个连续特征各取 `median/std/P10/P90`，加可见卫星数、L1 卫星比例和 L5 卫星比例；使用仅含历史至当前时刻的因果窗口。

## 3. 当前静态跨环境开发协议

`static_cross_env_v1` 的录制划分为：

```text
train: 新主楼静态 4 个 Session
val:   操场静态 2 个 Session
test:  操场静态 2 个 Session
```

该协议用于识别新主楼到操场的环境迁移困难。所有设备日志随完整真实 Session 划分，不允许同一录制跨 train/val/test。

**重要：** 该 test 已被用于模型、窗口和错分诊断，后续只能称为“开发参考 test”，不能作为论文最终独立测试。新一轮开发比较应使用 [static_session_cv_protocol.md](static_session_cv_protocol.md) 中的 4 折 Session-CV。

## 4. 设备级模型结果

下表均为 `static_cross_env_v1` 的开发参考 test，阈值为默认分类阈值 0.5；不能据此宣称最终论文性能。

| 模型 | 窗口 | 参数量/模型大小 | Macro-F1 | Recall | FAR |
|---|---:|---:|---:|---:|---:|
| 逐卫星 LSTM + 多数投票 | 5 | 5,378 参数 | 0.6754 | 0.3281 | 0.7754% |
| DeviceStatsMLP | 5 | 3,584 参数 | 0.8249 | 0.5834 | 0.0079% |
| DeviceStatsTCN | 5 | 3,818 参数 | 0.8218 | 0.5786 | 0.0950% |
| DeviceStatsLSTM | 30 | 5,186 参数 | 0.8185 | 0.5726 | 0.0485% |
| DeviceStatsDLinear | 30 | 1,540 参数 | 0.8342 | 0.6068 | 0.2022% |
| DeviceLightGBM | 30 | 121 KB | **0.9161** | **0.7908** | 0.2912% |
| DeviceLightGBM | 64 | 171 KB | 0.9134 | 0.7824 | 0.1168% |

已淘汰或不再扩展的分支：

- `NLinear L=30`：FAR 19.8253%，不具备可用性；
- `TSMixer L=30`、Depthwise CNN、TCN 的开发参考 Recall 未超过主要候选；
- `DLinear L=64` 相比 L=30 的 Recall 从 0.6068 降至 0.5161；
- 原始 27 维特征下，MLP/TCN/Depthwise CNN 的长窗口整体未优于其短窗口版本。

## 5. 当前结论

1. **性能强基线：** `LightGBM L=30` 当前 F1、Recall 最好，说明现有设备统计窗口中存在较强的结构化阈值和特征交互信息。
2. **轻量神经网络候选：** `DLinear L=30` 在已测神经网络中 Recall/F1 最好，参数量仅 1,540，适合作为后续蒸馏和 ONNX 端侧模型的学生候选。
3. **低虚警对照：** `MLP L=5` FAR 极低，但 Recall 明显低于 LightGBM，适合作为高保守报警的参考，而不是唯一部署方案。
4. **主要瓶颈：** 多数模型 validation 高、跨环境开发参考 test 下降，表明 C/N0、AGC 等绝对统计值包含设备与环境指纹；当前问题不是简单增加模型复杂度。

## 6. 同环境静态测试：static-only 与 static+dynamic 训练对照

为回答“加入动态训练数据是否改善静态检测”，在新主楼和操场分别固定同一批静态 train/val/test Session。两组均使用 27 维设备统计特征、因果窗口 `L=30`、LightGBM 和阈值 0.5；训练、验证、测试均按完整 Session 隔离。

`static-only` 只使用静态训练 Session；`static+dynamic` 完全复用同一批静态 train/val/test，仅额外向 train 加入同一环境的动态 Session。因此，下面同一环境内的两行可直接比较，衡量的是动态训练对未见静态 Session 的影响，不是动态 test 性能。

| 环境 | 训练组成 | 静态 test Macro-F1 | Recall | FAR | 相对 static-only 的结论 |
|---|---|---:|---:|---:|---|
| 新主楼 | [static-only](protocols/static_within_new_building_v1/README.md) | 0.9689 | 99.80% | 4.15% | 基线已很高；验证集只有 1 个设备。 |
| 新主楼 | [static+dynamic](protocols/static_dynamic_train_new_building_v1/README.md) | **0.9838** | **99.88%** | **2.13%** | 动态训练小幅提高 F1/Recall，并将 FAR 降低 2.02 个百分点。 |
| 操场 | [static-only](protocols/static_within_playground_v1/README.md) | 0.7668 | 48.89% | **0.00%** | L5 验证集 Recall 为 0，模型过于保守。 |
| 操场 | [static+dynamic](protocols/static_dynamic_train_playground_v1/README.md) | **0.9944** | **98.87%** | 0.17% | 动态训练显著改善静态 L1 test 的 Recall，F1 提升 0.2276。 |

初步结论：在这两组固定 Session 对照中，加入同环境动态训练没有损害静态 test，且操场上明显改善了静态召回。该结果支持“动态数据可作为训练期分布扩增”的假设，但样本数仍有限，频段、设备和 Session 时段同时变化；必须使用多折 Session-CV 及独立动态 test 进一步验证。

## 7. 下一阶段

1. 导出 `LightGBM L=30` 的特征贡献，分析特征组和历史时刻贡献。
2. 构造因果相对特征：相对滚动基线、变化率、短期波动、分位差等。
3. 以 `LightGBM L=30`、`DLinear L=30` 和 `MLP L=5` 为固定参考，在新特征协议上开展同环境与跨环境比较。
4. 使用 4 折静态 Session-CV，并分别报告环境内、跨环境、跨设备和静态到动态迁移结果。
5. 增加 TTD、虚警频率、模型大小、CPU/端侧时延；教师模型和知识蒸馏仅在上述基础完成后进行。

# 同星双频伪距率残差受控实验（2026-08-05）

## 结论

在当前无新增 Session 的数据条件下，同星同历元 L1/L5 `PseudorangeRateMetersPerSecond`
差分不能提升正式 mixed 场景分支。它使四折 pooled Macro-F1 从 **0.9245** 降至
**0.9060**，Accuracy 从 **0.9575** 降至 **0.9458**。该特征不进入正式 C/N0-only
场景基线，也不再在当前数据上围绕它调阈值或模型容量。

这不是对历史“按频段聚合伪距率”S2 的重复：S2 已在静态协议下失败；本实验改为同一
卫星、同一历元的 L1/L5 差分，以消除大部分卫星几何距离率和共同接收机时钟项，仍然
得到负结果。

## 数据与协议

- 原始日志由 `04_build_labeled_processed_csv.py --include-dynamic-raw-features` 重建为独立的
  增强 CSV；不覆盖 `output/processed_gnss_data.csv`。
- 外层协议保持 `output/protocols/mixed_timeblock_outer_cv4_w5_v2`：24 个 reviewed recordings、
  4 折 acquisition-group holdout、W=5。
- 模型与正式基线相同：TCN32、dropout=0.1、最多 40 epoch、seed=2026、全局标准化、
  train-only frozen normal C/N0 reference；仍只在双频端点计算四分类指标。
- 首折逐项校验：增强 CSV 构建出的窗口键、标签、单频掩码和原 11 维张量与正式基线
  **逐值一致**；新增实验只多出下面四项，不存在数据重建或切分变化造成的对照偏差。

## 特征定义与防泄漏措施

对每个 `(TimeNanos, ConstellationType, Svid)`，先分别对同频段重复 CodeType 的伪距率取
中位数，再计算：

```text
d(t, sv) = PseudorangeRate_L1(t, sv) - PseudorangeRate_L5(t, sv)
```

每一 outer fold 仅从 train split 中、且 L1/L5 均标为 normal 的配对观测，拟合
`device x constellation` 的 `d` 中位数；测试和验证仅应用冻结引用，不参与拟合。若未来
出现未知或样本不足的 device/constellation，设计上依次回退到该折的 constellation 全局引用和
全局总引用。

对每个接收机历元，模型输入为残差的 `median(d)`、`median(abs(d))`、MAD，外加
`PrrPairAvailable`。缺配对不会填成物理零，而是保持缺失并由单独的可用性标志表示。

在可评估双频端点中，至少一颗同星配对的覆盖率为：normal 99.94%、L1 99.83%、L5 91.07%、
L1+L5 99.73%。因此负结果不是由大范围缺失导致。

## 四折结果

| 指标 | 正式 C/N0 基线 | C/N0 + 同星双频伪距率残差 | 变化 |
|---|---:|---:|---:|
| Macro-F1 | **0.9245** | 0.9060 | -0.0185 |
| Accuracy | **0.9575** | 0.9458 | -0.0117 |
| normal F1 / recall | **0.9758 / 0.9757** | 0.9690 / 0.9611 | -0.0067 / -0.0147 |
| L1 F1 / recall | **0.8716 / 0.9072** | 0.8288 / 0.8828 | -0.0427 / -0.0244 |
| L5 F1 / recall | **0.9283 / 0.9174** | 0.9140 / 0.9241 | -0.0143 / +0.0066 |
| L1+L5 F1 / recall | **0.9225 / 0.8885** | 0.9122 / 0.9001 | -0.0102 / +0.0116 |

两版预测在全部 43,672 个 outer-test endpoint 上一一对齐。新增特征使原基线正确、而新模型
错误的窗口为 907 个；反向修正为 396 个，净少 511 个正确窗口。损失主要来自正常类被推成
L1/L5：正常类净增加 475 个错误；L1 也净增加 100 个错误。操场静态 L1、操场早期动态 L1
和华为正常段是净退化最大的来源。

新增特征在“基线正确而新模型错误”的端点上呈现明显重尾，而其可用率仍约 99.7%。这说明
残余中仍包含设备/通道异常，模型会把这类正常波动误当成攻击证据；冻结
device x constellation 中位基线不足以将其变成跨 Session 稳定判据。

## 复现与处置

实现位于：

- `pipeline_total/45_build_band_mean_window_tensors.py --include-paired-pseudorange-rate`
- `pipeline_total/47_aggregate_band_mean_cv.py --include-paired-pseudorange-rate`
- `tests/test_paired_pseudorange_rate.py`

输出位于被 Git 忽略的 `output/diagnostics/raw_tracking_candidate_v1/`、
`output/tensors/paired_prr_cv4_v1/` 和 `output/training/paired_prr_cv4_v1/`。

当前可证伪的是这个“同星双频伪距率残差 + 当前混合四折协议”的输入方案，而不是声称
伪距率在任何新采集条件下都没有价值。没有新的跨设备/跨环境 Session 时，不应继续对该特征
做同数据集调参；若以后新增采集，再以冻结本实验参数的方式重新验证即可。

# 静态逐 Signal Fold 6 诊断快照（2026-07-27）

## 用途与边界

本文固化 2026-07-26 至 27 在静态操场长 L5 Session 上完成的 E9-E11 结果、产物路径与解释边界。它是仓库内可追溯的结果快照，不包含张量、checkpoint 或原始 GNSS 数据；这些大文件留在本地 `output/` 并可按脚本重建。

该 outer test 已被多次用于设计和错误分析。因此本文所有 Fold 6 数值仅为**迭代式开发诊断**，不能表述为新的独立盲测结论，也不能替代完整 7-fold `compact11 + TCN16 + dropout=0.1` 保留对照基线。

## 固定任务与数据

- 数据：`LabelStatus=reviewed` 的静态 `st_*` Session。
- 正式标签：target-band-only 二分类。非目标频段在单频攻击期间仍为 0；“受压制或受干扰”不等同于已被欺骗。
- Fold 6 outer test：操场 `st_L5/2025.07.30.09.48_2025.07.30.10.14`。
- 测试张量：`output/tensors/static_timeblock_outer_v2_e9a_refit_all_dev/fold_6`。
- E9-E11 均为 TCN、hidden=16、dropout=0.3、raw `full` 和 stats `full`。E9 参数量为 2,024，E11 为 2,160。

## 结果概览

| 实验 | 训练或决策变化 | 指标口径 | Recall | Precision | FAR | Macro-F1 | 结论 |
|---|---|---|---:|---:|---:|---:|---|
| E0 | 原 time-block 训练，对照 checkpoint | 全部 endpoint | 50.33% | 30.84% | 15.50% | 0.6329 | Fold 6 可复现对照 |
| E9a | all-development fixed-epoch refit | 全部 endpoint | 55.82% | 34.11% | 14.81% | 0.6572 | 补回短 L5 Session 训练覆盖后小幅改善 |
| E9b | E9a + Session 均匀采样 | 全部 endpoint | 56.93% | 34.40% | 14.91% | 0.6599 | 仅小幅改善；短 Session 欠采样不是主因 |
| E10 | L5-only 损失，非 L5 强制正常 | 仅 L5 endpoint | 60.58% | 82.45% | 11.46% | 0.7453 | 减少 L1 影响后 Recall 上升，但仍不足且误报增加 |
| E11 | E10 + 设备条件 L5 头 | 仅 L5 endpoint | 62.73% | 82.38% | 11.92% | 0.7543 | 最好 L5 诊断 Recall，但设备失配仍显著 |

E10/E11 的“全部 endpoint”指标会因 Fold 6 没有 L1 正类且脚本强制 L1 为正常而显著变好，不应与 E0/E9 的全部 endpoint 指标直接排名。L5-only 行才是对 E10/E11 的可比口径。

## E11 的设备级错误结构

| 设备 | L5 正样本数 | Recall | FAR | Precision | 解释 |
|---|---:|---:|---:|---:|---|
| Google Pixel 6 | 2,288 | 4.06% | 0.00% | 100.00% | 极端保守，主要为漏检 |
| HUAWEI Mate40 | 12,780 | 94.20% | 50.51% | 72.25% | 极端激进，主要为误报 |
| RedMi K60 | 22,709 | 50.80% | 0.15% | 99.78% | 极端保守，且占主要正样本量 |
| XiaoMi MI8 | 249 | 75.10% | 11.31% | 29.26% | 样本量较小且误报偏高 |

同一 L5 攻击下，模型在 Mate40 上几乎把大量正常样本判为正，在 RedMi K60 上又几乎不报正。该相反错误方向说明总体均值掩盖了设备条件失配。

## 只读阈值可分性诊断

下面的数值由 E11 checkpoint 在同一 Fold 6 test 上离线扫描得出，仅用于理解分数排序，不可用于反向选择最终阈值。

| 设备 | Recall@FAR<=5% | Recall@FAR<=10% | Recall@FAR<=20% | 达到 Recall=80% 所需 FAR |
|---|---:|---:|---:|---:|
| Google Pixel 6 | 89.03% | 89.73% | 90.73% | 0.03% |
| HUAWEI Mate40 | 30.74% | 46.54% | 66.98% | 30.62% |
| RedMi K60 | 65.15% | 67.90% | 70.14% | 44.27% |
| XiaoMi MI8 | 56.63% | 73.09% | 77.11% | 48.91% |

Pixel 6 的问题主要是默认阈值校准；Mate40、RedMi K60 和 MI8 的问题则是在可接受 FAR 区间内分数排序不足。因而单一全局阈值、单纯 Session 重采样或单纯设备头均不能同时解决这些设备的错误。

## 产物与复现定位

| 项目 | 本地路径 | 仓库入口 |
|---|---|---|
| E9a 模型与 test 指标 | `output/training/static_timeblock_outer_v2_e9a_refit_all_dev/fold_6/tcn/` | `pipeline_total/33_refit_static_signal_fusion.py` |
| E9b 模型与分组指标 | `output/training/static_timeblock_outer_v2_e9b_session_uniform/fold_6/tcn/` | `pipeline_total/33_refit_static_signal_fusion.py` |
| E10 模型与 JSON 指标 | `output/training/static_timeblock_outer_v2_e10_l5_expert/fold_6/tcn/` | `pipeline_total/34_refit_static_l5_expert.py` |
| E11 模型与 JSON 指标 | `output/training/static_timeblock_outer_v2_e11_l5_device_heads/fold_6/tcn/` | `pipeline_total/35_refit_static_l5_device_heads.py` |
| 标签与数据政策 | `configs/preprocessing.yml`、本地中央 CSV | `docs/data_inventory.md`、`docs/handoff_status.md` |

## 当前问题

在当前标签语义和已审查数据下，静态操场长 L5 Session 的主要困难不是模型参数量，也不是短新主楼 L5 Session 的样本数量。现有证据指向跨 Session 的设备/环境条件变化：绝对 C/N0、AGC 和其短时间统计在不同接收机上的判别方向不一致。任何后续结果都应同时报告目标频段 Recall、FAR、Session 和设备分组，并保留本文件说明的 test 已被用于诊断这一边界。
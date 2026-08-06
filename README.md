# 面向真实多设备部署的轻量化 GNSS 导航欺骗检测

本仓库研究如何利用手机、手表等真实设备能够获得的少量 GNSS Raw 特征，进行轻量化导航欺骗检测。当前阶段的核心问题不是继续扩大模型，而是确认标签、Session 质量和跨环境评估协议。

## 当前状态

截至 2026-08-06：

- 正式数据口径为 123 份原始日志：操场 89 份、新主楼 34 份；大文件、中央 CSV、张量、checkpoint 和图像均保留在本地 `output/`，不提交 Git。
- 当前最终候选是“**频段场景识别 + 设备响应联合诊断**”：场景分支输出 `normal/L1/L5/L1+L5`，设备分支输出 `normal/anomaly/direct`，以区分直接受骗与攻击关联异常。
- 完整 7-fold `compact11 + TCN16` 是当前保留对照基线；其结果与近期 Fold 6 诊断实验使用的特征集、refit 协议和评价口径不同，不能只按单个 pooled 分数排序。
- Fold 6 的 outer test 已被多次读取以诊断错误结构。E9-E11 因而只能作为迭代式开发诊断，而非新的独立盲测结果。
- 已完成静态六折完整诊断链：四维 C/N0-only + TCN 场景分支（pooled Macro-F1=0.9921，Accuracy=0.9944），结合 44 维 C/N0 设备响应 backbone、Watch L1 压制异常专家和 L5 自校准 direct 证据。完整系统 pooled Macro-F1=0.9802、FAR=0.863%、abnormal recall=99.05%、anomaly recall=98.76%、direct recall=98.98%。
- 该方案针对并缓解了 L5 场景下 Watch 无 L5 观测但 L1 受压制的关联异常漏检，以及 Pixel6 等异构设备的 L5 direct 响应迁移失配；Watch 的 L1 压制输出为 `anomaly`，而不错误标为 L5 `direct`。
- 方案优化阶段 3 已完成：场景分支经过 S1–S5 六折特征消融后，确定四维 C/N0-only + TCN 为当前正式候选，pooled Macro-F1=0.9921、Accuracy=0.9944。
- 当前结果仍是静态协议下的开发性六折结果。动态泛化、TTD（检测时延）、端到端模型大小/推理耗时，以及更严格的独立盲测仍是下一阶段工作。

请依次阅读：[当前交接状态](docs/handoff_status.md)、[Fold 6 诊断快照](docs/static_signal_fold6_diagnostics_20260727.md)、[数据清单](docs/data_inventory.md) 和 [脚本索引](pipeline_total/README.md)。

## 仓库结构

```text
configs/          预处理和标签配置
data_raw/         本地原始 GNSS TXT，Git 忽略
data_csv/         逐日志 CSV，Git 忽略
docs/             数据、标签、协议和交接文档
models/           逐 signal 与设备级轻量模型
pipeline_total/   数据处理、审查、训练和评估脚本
scripts/          清单、审计等辅助脚本
output/           本地缓存和生成结果，Git 忽略
```

仓库只同步代码、配置和必要文档，不提交原始数据、中央 CSV、NPZ 张量、checkpoint 或批量图像。

## 数据与标签权威来源

按以下优先级判断当前状态：

1. `configs/preprocessing.yml`：标签区间与 reviewed 状态；
2. `docs/data_inventory.md`：数据来源和使用政策；
3. `output/data_manifest.csv`、`output/data_csv_session_manifest.csv`：按需生成的本地清单；
4. `output/processed_gnss_data.csv`：当前本地统一处理缓存。

PNG 只是标签复核辅助图；02 与 22 均只读取配置中的显式 Session 级标签，不再使用场景级兜底。标签或原始数据发生变化后，必须重建镜像 CSV、中央 CSV、审计和训练张量，旧 checkpoint 不应继续混用。

## 当前代码路线

`pipeline_total/01–10` 是既有数据处理、绘图、标注、基础训练和错误分析链。

`pipeline_total/11–18` 是 P0–P5 设备级历史探索，代码保留用于追溯。

场景分支的静态逐 signal 实验入口为：

```text
19_generate_static_timeblock_protocol.py
20_build_static_timeblock_tensors.py
21_train_static_signal_fusion.py
```

当前正式的 Session 级标签一致性审查入口为 `22_generate_label_review_dashboards.py`；02 继续用于逐设备、逐特征单图复核。

最终设备联合诊断的完整配置、结果与复现顺序见 [场景门控的设备响应联合诊断](docs/complete_scene_response_diagnosis_20260806.md)；其脚本索引和完整参数见 [pipeline_total/README.md](pipeline_total/README.md)。

## 基础重建

从仓库根目录执行：

```powershell
python scripts/build_mirrored_data_csv.py --config configs/preprocessing.yml --overwrite
python pipeline_total/04_build_labeled_processed_csv.py --mode full --config configs/preprocessing.yml
python pipeline_total/01_generate_plot_feature_csv.py --data-root data_raw --config configs/preprocessing.yml --overwrite
python scripts/build_data_manifest.py --output output/data_manifest.csv
python scripts/audit_extracted_csv.py --input-dir data_csv --output-json output/data_csv_audit.json
python scripts/build_csv_session_manifest.py --input-dir data_csv --output-csv output/data_csv_session_manifest.csv
```

最近的静态 time-block 协议从以下命令开始：

```powershell
python pipeline_total/19_generate_static_timeblock_protocol.py
```

生成的协议、张量和训练结果全部写入 `output/`，需要时重建，不作为仓库资产。

## 当前研究原则

- 场景分支以频段聚合识别广播式攻击环境；设备响应分支作为部署诊断层，区分 `normal/anomaly/direct`；
- 数据切分至少以完整录制 Session 为外层隔离单位；
- signal endpoints 高度相关，不能把百万条端点当作百万个独立样本；
- validation 只用于选模和早停，已使用过的 test 不再包装成独立盲测；
- 除 Accuracy/Macro-F1 外，同时报告攻击 Recall、FAR、逐 Session/设备波动和检测时延；
- 在数据质量未收敛前，不以扩大模型或模型融合替代标签与协议审查。

## 文档入口

- [当前交接状态](docs/handoff_status.md)：当前唯一状态入口；
- [论文主线大纲](docs/paper_mainline_outline_zh.md)：论文题目、问题定义、方法、实验组织和结果表述边界；
- [方案优化实验清单](docs/optimization_experiment_plan_zh.md)：特征消融、模型比较、融合、轻量化和检测时延实验顺序；
- [方案优化结果记录](docs/optimization_experiment_results_zh.md)：O0 基线及逐项优化实验的配置、指标和决策；
- [场景门控的设备响应联合诊断](docs/complete_scene_response_diagnosis_20260806.md)：当前静态六折完整候选方案、关键结果、复现顺序和适用边界；
- [P0–P5 历史实验台账](docs/experiment_registry.md)：旧设备级路线与结果边界；
- [信号级特征提取](docs/signal_level_feature_extraction.md)：统一 CSV 与逐 signal 数据语义；
- [动态标签辅助](docs/dynamic_labeling_assistant.md)：新主楼动态场景复核流程；
- [数据清单](docs/data_inventory.md)：数据来源和使用政策；
- [协作说明](CONTRIBUTING.md)：GitHub 协作约定。

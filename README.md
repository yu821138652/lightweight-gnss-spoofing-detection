# 面向真实多设备部署的轻量化 GNSS 导航欺骗检测

本仓库研究如何利用手机、手表等真实设备能够获得的少量 GNSS Raw 特征，进行轻量化导航欺骗检测。当前阶段的核心问题不是继续扩大模型，而是确认标签、Session 质量和跨环境评估协议。

## 当前状态

截至 2026-07-27：

- 正式数据口径为 123 份原始日志：操场 89 份、新主楼 34 份；大文件、中央 CSV、张量、checkpoint 和图像均保留在本地 `output/`，不提交 Git。
- 当前主研究对象是静态逐 `signal_id` 的 target-band-only 欺骗检测。动态数据、历史设备级 P0-P5 和静态/动态混合任务均保留为独立历史路线，不能与当前主任务混合比较。
- 完整 7-fold `compact11 + TCN16` 是当前保留对照基线；其结果与近期 Fold 6 诊断实验使用的特征集、refit 协议和评价口径不同，不能只按单个 pooled 分数排序。
- Fold 6 的 outer test 已被多次读取以诊断错误结构。E9-E11 因而只能作为迭代式开发诊断，而非新的独立盲测结果。
- 截至目前没有确定最终部署模型。当前最明确的问题是静态操场长 L5 Session 中存在严重的设备条件失配：不同设备会出现相反的漏检和误报模式。

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

当前静态逐 signal 实验入口为：

```text
19_generate_static_timeblock_protocol.py
20_build_static_timeblock_tensors.py
21_train_static_signal_fusion.py
```

当前正式的 Session 级标签一致性审查入口为 `22_generate_label_review_dashboards.py`；02 继续用于逐设备、逐特征单图复核。

完整参数与运行顺序见 [pipeline_total/README.md](pipeline_total/README.md)。

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

- 以卫星/`signal_id` 级检测为主要研究对象，设备级聚合作为部署层或辅助实验；
- 数据切分至少以完整录制 Session 为外层隔离单位；
- signal endpoints 高度相关，不能把百万条端点当作百万个独立样本；
- validation 只用于选模和早停，已使用过的 test 不再包装成独立盲测；
- 除 Accuracy/Macro-F1 外，同时报告攻击 Recall、FAR、逐 Session/设备波动和检测时延；
- 在数据质量未收敛前，不以扩大模型或模型融合替代标签与协议审查。

## 文档入口

- [当前交接状态](docs/handoff_status.md)：当前唯一状态入口；
- [P0–P5 历史实验台账](docs/experiment_registry.md)：旧设备级路线与结果边界；
- [信号级特征提取](docs/signal_level_feature_extraction.md)：统一 CSV 与逐 signal 数据语义；
- [动态标签辅助](docs/dynamic_labeling_assistant.md)：新主楼动态场景复核流程；
- [数据清单](docs/data_inventory.md)：数据来源和使用政策；
- [协作说明](CONTRIBUTING.md)：GitHub 协作约定。

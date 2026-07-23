# 项目交接状态（2026-07-23）

本文是当前工作区的唯一状态入口。它说明已经确认的数据事实、最近的探索结论、结果边界、可复现入口和下一步建议。历史 P0–P5 结果只用于追溯，不代表当前已经锁定的主线。

## 1. 一句话结论

项目目前仍处于“数据与评估协议收敛”阶段，尚未确定最终模型。2026-07-23 的 7-fold 静态逐 `signal_id` 重训说明：部分 Session 可以取得较好结果，但跨 Session、设备和场景的波动仍很大；动态场景以及操场 L5/L15 的主要瓶颈更像是标签可信度、设备观测差异和特征域偏移，而不是模型容量不足。

因此，当前不应继续围绕某个模型反复调参。交接后的第一优先级应是建立可信 Session 清单、逐场景复核数据，并明确新的独立测试数据；模型比较应在这些前提固定后重跑。

## 2. Git 与工作区基线

- 当前分支：`main`。
- 当前本地 `HEAD` 为 `a19b2f7`，`origin/main` 为 `f434dd0`，本地分支落后 1 个提交。
- `9477462` 已同步本地整理、标签修订和逐 signal/time-block 探索；`a19b2f7` 已加入 Session 级标签审查面板；`f434dd0` 的按 Session TOW 范围裁切逻辑已手工并入当前未提交工作区。
- 2026-07-23 重新生成的两套 label plots 位于被 Git 忽略的 `output/`，不会进入仓库。
- Git 历史中的 P0–P5 是设备级路线的探索记录；保留代码与实验台账用于追溯，但不再将 LightGBM、DLinear 或某个双分支网络描述为已锁定主模型。

## 3. 当前数据与标签状态

### 3.1 数据快照

- 当前正式原始日志：123 份，其中操场 89 份、新主楼 34 份。
- 已主动剔除 9 份操场日志：`dy_L5/2022.07.08` 3 份、`st_L5/2025.07.30.09.41_2025.07.30.09.45` 6 份；它们不参与当前重建和训练。
- `output/processed_gnss_data.csv` 已按上述 123 份日志重建为 2,998,458 行；正类 631,003、负类 2,367,455，全部为 `reviewed/session_config`。
- 数据政策说明：`docs/data_inventory.md`。
- 本地清单按需生成到 `output/data_manifest.csv` 和 `output/data_csv_session_manifest.csv`，不作为仓库文件维护。
- 权威标签配置：`configs/preprocessing.yml`。
- `output/` 不进入 Git；中央 CSV 保留在本地，是当前最值得保留的可重用缓存。

### 3.2 已确认的标签决定

- 操场 `dy_L_15/2025.07.30.08.26_2025.07.30.08.32（动态L1+L5）` 不再使用场景级旧区间 `[260970, 261040]`，改用经多人曲线复核后的 Session 级区间 `[260990, 261020]`。
- 该修订有明确边界：真实设备工作区间无法从隔壁组记录中精确还原，但原区间两端缺少可见欺骗响应，且数据提供方承认标注或干扰设备可能存在问题。因此当前区间是“人工审查后的可信修订”，不是物理真值。
- 同一 Session 中 RedMi K60 的 `gnss_log_2025_07_30_08_17_11` 暂时保持现状，不做额外剔除或重标。它在清单中继续作为 reviewed 数据存在。
- 操场与新主楼统一使用 `Environment -> Scenario -> Session -> {status, intervals}`；未显式列出的 Session 标为 `needs_review`，不得进入正式训练。

### 3.3 已知数据风险

- 操场动态 L15 即使收紧标签后，逐 signal LSTM validation Recall 仍只有 34.50%；操场动态 L5 Recall 仅 9.27%。这说明问题不只来自那一个过宽区间。
- 最新 7-fold 静态 outer-session 重训中，操场长 L5 Session 的 Macro-F1 为 0.6329、Recall 为 0.5033、FAR 为 15.50%，仍是最明显的瓶颈；操场 L15 Session 的 Macro-F1 为 0.8214、Recall 为 0.6062、FAR 为 0.62%，主要问题仍是漏检。这两个 Session 应继续优先做逐设备、逐频段和原始曲线复核。
- 新主楼 L5 的正类数量偏少并不等于张量漏样本。当前标签语义是“目标 TOW 区间且 `FreqBand == 5` 才为正类”；同录制中的 L1 信号仍为负类。Pixel Watch 在这些录制中通常没有 L5 信号，因此会贡献大量负类而没有 L5 正类。
- Google Pixel Watch 1 的 `AgcDb` 全缺失。绝对 AGC 统计很容易携带设备身份，必须保留缺失标记，并在后续做 no-AGC 或相对化消融。
- `pipeline_total/02_batch_plot_feature_images.py` 只按显式 Session 级 reviewed 配置绘制阴影，不再使用场景级回退；PNG 仍只是复核辅助证据，标签判断以 `configs/preprocessing.yml` 和重建后的 CSV 为准。

## 4. 路线演进与当前边界

### 4.1 2026-07-18 前：P0–P5 设备级探索

P0–P5 覆盖逐信号聚合、设备级统计张量、LightGBM/DLinear 等模型、静态跨环境、静态 Session-CV 和静态+动态混合任务。其价值是暴露了环境、频段和 Session 划分对结果的强影响；详细数字保留在 `docs/experiment_registry.md`。

这些结果存在共同边界：部分 test 已被用于窗口、模型和错误诊断，不能再称为论文最终盲测；设备级 27 维统计与 LightGBM 也不是当前已经确认的论文主模型。

### 4.2 2026-07-22：静态逐 signal 探索

最近的独立路线把重点转回卫星/`signal_id` 级检测，并尝试逐 signal 窗口统计。当前默认窗口为 W5，但 W3/W5/W7 差异不大；W5 只是略优的工程默认值。

最新双分支模型包含：

- raw 分支：5 个因果历元，实际使用 `Cn0DbHz`、`AgcDb`、`ReceivedSvTimeUncertaintyNanos`、`PseudorangeRateUncertaintyMetersPerSecond`、`FreqBand`；
- stats 分支：对每个 `signal_id` 在 W5 内计算 19 维 `Last/Mean/Std/Slope` 与观测完整度特征；
- 明确排除 CSV 中已有的 `Cn0DbHz_dt` 和 `Cn0DbHz_std`，避免与重新计算的窗口统计重复；
- TCN + stats MLP 的当前正则配置为 hidden=16、dropout=0.3、weight decay=1e-3，共 2,024 个参数。

这只是最近一次可复现实验配置，不是最终模型选择。

### 4.3 2026-07-23：清理数据后的 7-fold 静态重训

短时操场静态 L5 Session `2025.07.30.09.41_2025.07.30.09.45` 已随 6 份原始日志一起剔除。它全为负类，因此当前静态独立录制从 8 个变为 7 个，outer-session 协议也相应变为 7 folds。新 fold 6、7 分别对应旧 fold 7、8；比较结果时必须按 `Environment + Scenario + Session` 对齐，不能只看 fold 编号。

本轮仍然只训练静态数据。操场动态 L15 的区间修订不会直接进入本轮张量；真正改变训练的是上述全负静态 Session 不再出现在其余 folds 的开发集。剩余 7 个 outer test 的样本、正负类 support 与旧实验逐项一致，因此可以直接比较同一批 Session 上的新旧 checkpoint。

训练严格沿用上一轮配置：W5、raw TCN + stats MLP、hidden=16、dropout=0.3、AdamW `lr=1e-3`、weight decay=1e-3、batch=256、最多 30 epochs、patience=6、seed=2026、num_workers=0。每折只按 inner validation Macro-F1 锁定 checkpoint，7 个 checkpoint 全部锁定后才依次执行 `test-only`。

## 5. 关键实验摘要

| 实验 | 主要结果 | 能说明什么 | 不能说明什么 |
|---|---|---|---|
| 修订标签后的 mixed 逐 signal 基线 | LSTM validation Macro-F1 0.7947、Recall 0.5514、FAR 2.15% | 动态数据是主要漏检来源；短时序比 MLP 有帮助 | test 未读，不能代表泛化 |
| 单一静态固定划分 raw+stats | TCN validation 0.9906，test 仅 0.7015 | 单 Session validation 会造成严重选择过拟合 | 不能把 0.99 当作静态性能 |
| 静态 4-fold，W5 正则配置 | Macro-F1 0.8245 ± 0.0877，FAR 8.34% ± 5.33% | 跨 Session 波动显著，轻量正则略降误报 | 4 折样本仍只有 8 个独立录制 |
| W3/W5/W7 | 0.8154 / 0.8245 / 0.8151 | W5 略优，可作默认 | 差异不支持“W5 显著最好” |
| 6 train / 1 val / 1 test | pooled Macro-F1 0.8232、Recall 0.7487、FAR 9.57% | 增加 train Session 没有解决泛化 | 单 validation Session 仍不稳定 |
| 旧 8-fold Outer-Session / Inner-Time-Block | pooled Macro-F1 0.8386、Precision 0.7968、Recall 0.7168、FAR 6.19%；7 个有正类 Session 的 Macro-F1 0.8515 ± 0.1159 | 其余 Session 都可参与开发，误报较 6/1/1 少 | 包含现已剔除的短时全负操场 L5；不能直接按 fold 编号和新实验比较 |
| 当前 7-fold 重训 | pooled Macro-F1 0.8546、Precision 0.7914、Recall 0.7768、FAR 7.18%；Session Macro-F1 0.8495 ± 0.1142 | 同 7 个测试 Session 上 pooled Recall 提高 6.00 个百分点，操场 L15 改善明显 | Session 均值没有改善，Precision 降低 1.63 个百分点、FAR 增加 1.20 个百分点；不是全面泛化提升 |

当前 7-fold 重训是最新口径，但仍不能锁定模型：inner validation Macro-F1 为 0.9388 ± 0.0121，outer test 的 Session 均值只有 0.8495 ± 0.1142，且不同 Session 差异很大。与旧模型在同 7 个测试 Session 上比较，pooled Macro-F1 从 0.8408 升至 0.8546，但 Session Macro-F1 均值从 0.8515 略降至 0.8495；提升主要来自大 Session 上更多检出正类，同时付出了更多误报。所谓“样本量百万级”只是高度相关的 signal endpoints；真正独立的静态录制单元现在只有 7 个。

逐 Session 新结果如下；括号内为同一 Session 的旧 Macro-F1：

| 新 fold | Outer test Session | Macro-F1 | Recall | FAR |
|---:|---|---:|---:|---:|
| 1 | 新主楼 `st_L1/19.22` | 0.9382（0.9319） | 89.08% | 2.64% |
| 2 | 新主楼 `st_L5/20.16` | 0.7797（0.8306） | 83.26% | 8.20% |
| 3 | 新主楼 `st_L5/20.36` | 0.9393（0.9769） | 99.80% | 1.39% |
| 4 | 新主楼 `st_L_15/18.42` | 0.9325（0.9271） | 87.68% | 2.52% |
| 5 | 操场 `st_L1/08.40–09.12` | 0.9025（0.8922） | 94.33% | 9.92% |
| 6 | 操场 `st_L5/09.48–10.14` | 0.6329（0.6497） | 50.33% | 15.50% |
| 7 | 操场 `st_L_15/07.30–08.01` | 0.8214（0.7523） | 60.62% | 0.62% |

方向二与 6/1/1 也不是严格的单变量 split 消融：新的 builder 同时增加了断档窗口过滤和按 train 均值填充缺失值。因此两者的数值差不能全部归因于划分方式。

## 6. 整理后的代码定位

`pipeline_total/01–10` 是既有数据、画图、标注、基础张量、训练和错误分析链；02 已补充按当前 YAML 配置解析 Session 级标签阴影。

`pipeline_total/11–18` 保留为 P0–P5 历史设备级探索。它们可以复现旧实验，但不应作为新的默认入口。

当前逐 signal 静态实验收敛为三个脚本：

```text
19_generate_static_timeblock_protocol.py
20_build_static_timeblock_tensors.py
21_train_static_signal_fusion.py
```

原有 stats-only builder、6/1/1 生成器和两个重复编号的 time-block 22 脚本已被替代。当前新的 `22_generate_label_review_dashboards.py` 用于正式的 Session 级标签审查，与旧 22 无关。

## 7. 最小重建流程

标签或原始数据发生变化后，先重建中央 CSV 和审计文件：

```powershell
python scripts/build_mirrored_data_csv.py --config configs/preprocessing.yml --overwrite
python pipeline_total/04_build_labeled_processed_csv.py --mode full --config configs/preprocessing.yml
python pipeline_total/01_generate_plot_feature_csv.py --data-root data_raw --config configs/preprocessing.yml --overwrite
python scripts/build_data_manifest.py --output output/data_manifest.csv
python scripts/audit_extracted_csv.py --input-dir data_csv --output-json output/data_csv_audit.json
python scripts/build_csv_session_manifest.py --input-dir data_csv --output-csv output/data_csv_session_manifest.csv
```

重建当前 7-Session 静态 time-block 协议：

```powershell
python pipeline_total/19_generate_static_timeblock_protocol.py `
  --csv output/processed_gnss_data.csv `
  --source-recording-manifest output/protocols/static_time_block_outer_v2/source_recording_manifest.csv `
  --output-dir output/protocols/static_time_block_outer_v2 `
  --time-steps 5 `
  --block-epochs 256 `
  --val-fraction 0.20 `
  --segment-gap-seconds 2
```

每个 outer fold 分别构建张量；`epoch_split_manifest.csv` 是权威逐历元划分，不能误用仅作汇总的 `time_block_manifest.csv`：

```powershell
python pipeline_total/20_build_static_timeblock_tensors.py `
  --outer-manifest output/protocols/static_time_block_outer_v2/fold_1/recording_split_manifest.csv `
  --block-manifest output/protocols/static_time_block_outer_v2/fold_1/epoch_split_manifest.csv `
  --output-dir output/tensors/static_timeblock_outer_v2/fold_1 `
  --time-steps 5 `
  --block-size 256
```

训练与 test-only 的命令以 `pipeline_total/README.md` 为准。每折 scaler 只能用 train 拟合，validation 只用于早停；test checkpoint 锁定后再读。当前 7 个 outer tests 已经全部读取过，后续依据这些结果调参时必须记作迭代式 CV，不能再称完全盲测。

## 8. 本地生成物保留策略

本次清理后，`output/` 只保留：

- `processed_gnss_data.csv` 与缺失报告；
- `data_csv_audit.json`；
- `dynamic_labeling_review/`；
- `review/trusted_signal_baseline_v1/` 下的关键错分明细与汇总；
- `label_plots_20260723/new_building/`（238 张）和 `label_plots_20260723/playground/`（623 张），按统一 Session 级标签配置从 123 份现役日志干净重建；
- `protocols/static_time_block_outer_v2/`、`tensors/static_timeblock_outer_v2/` 和 `training/static_timeblock_outer_v2/`，作为本轮 7-fold 重训的协议、张量、checkpoint、日志和逐折指标；
- `output/README.md`。

张量、checkpoint 和训练日志仍属于可重建产物；当前 `static_timeblock_outer_v2` 只因本轮交接与新旧对照暂时保留，指标稳定写入文档后可再归档或删除。当前两套 plots 只是本轮标签复核需要。旧产物已集中迁入被 Git 忽略的 `output/_rebuildable_archive_20260722/`。此外，旧 `new_building_label_plots/` 与 `playground_label_plots/` 未自动删除，其中旧操场目录仍含已剔除 Session 的 63 张残留图，不能再作为当前数据口径使用。当前执行环境的递归删除审批服务异常，因此磁盘空间尚未真正释放；确认无需恢复后可人工删除旧目录和归档。历史指标已经压缩进本文和 P0–P5 台账。

## 9. 推荐交接顺序

1. 先按 Session 建立 `trusted / questionable / excluded` 数据清单，重点复核操场动态 L5、动态 L15、静态长 L5 和静态 L15；不要把“reviewed”简单等同于物理真值。
2. 明确论文主任务以卫星/逐 signal 检测为主，设备级聚合作为部署层或辅助实验；不要在两种评价单位之间混用指标。
3. 新增独立录制或保留真正未触碰的 Session。当前只有 7 个静态录制，再复杂的交叉验证也不能创造新的环境多样性。
4. 固定可信数据、特征和 split 后，再重跑 MLP、TCN/LSTM、raw+stats 等少量轻量基线；优先报告 Session×Device 宏平均、最差设备 FAR、攻击 Recall 和检测时延。
5. 在数据问题没有澄清前，不建议继续融合更多模型或扩大网络容量。若以后做融合，应先证明不同模型的错误具有互补性，而不是只比较 pooled Accuracy。

## 10. 文档入口

- 当前状态与交接：本文。
- P0–P5 历史实验：`docs/experiment_registry.md`。
- 数据清单：`docs/data_inventory.md`；本地清单由上述脚本生成到 `output/`。
- 标签复核：`docs/dynamic_labeling_assistant.md`、`docs/dy_manual_label_intervals.csv`。
- 信号级数据构建：`docs/signal_level_feature_extraction.md`。
- 历史静态 4-fold 协议：`docs/static_session_cv_protocol.md`。
- 脚本索引：`pipeline_total/README.md`。

# 项目交接状态（2026-07-22）

本文是当前工作区的唯一状态入口。它说明已经确认的数据事实、最近的探索结论、结果边界、可复现入口和下一步建议。历史 P0–P5 结果只用于追溯，不代表当前已经锁定的主线。

## 1. 一句话结论

项目目前仍处于“数据与评估协议收敛”阶段，尚未确定最终模型。最近一轮实验说明：静态逐 `signal_id` 检测可以在部分 Session 上取得较好结果，但跨 Session、设备和场景的波动很大；动态场景以及操场 L5/L15 的主要瓶颈更像是标签可信度、设备观测差异和特征域偏移，而不是模型容量不足。

因此，当前不应继续围绕某个模型反复调参。交接后的第一优先级应是建立可信 Session 清单、逐场景复核数据，并明确新的独立测试数据；模型比较应在这些前提固定后重跑。

## 2. Git 与工作区基线

- 当前分支：`main`。
- 当前 `HEAD` 与 `origin/main` 均为 `73252e0`，最后一次同步状态截至 2026-07-18。
- 2026-07-18 之后的标签修订、逐 signal 静态实验和 time-block 实验尚未提交。
- 本次整理只修改本地文件，没有创建 commit，也没有 push。
- Git 历史中的 P0–P5 是设备级路线的探索记录；保留代码与实验台账用于追溯，但不再将 LightGBM、DLinear 或某个双分支网络描述为已锁定主模型。

## 3. 当前数据与标签状态

### 3.1 数据快照

- 原始日志：132 份，其中操场 98 份、新主楼 34 份。
- 当前统一处理表：`output/processed_gnss_data.csv`，约 213.9 万条信号级记录。
- 权威数据清单：`docs/data_manifest.csv`。
- 权威逐日志 CSV 审计清单：`docs/data_csv_session_manifest.csv`。
- 权威标签配置：`configs/preprocessing.yml`。
- `output/` 不进入 Git；中央 CSV 保留在本地，是当前最值得保留的可重用缓存。

### 3.2 已确认的标签决定

- 操场 `dy_L_15/2025.07.30.08.26_2025.07.30.08.32（动态L1+L5）` 不再使用场景级旧区间 `[260970, 261040]`，改用经多人曲线复核后的 Session 级区间 `[260990, 261020]`。
- 该修订有明确边界：真实设备工作区间无法从隔壁组记录中精确还原，但原区间两端缺少可见欺骗响应，且数据提供方承认标注或干扰设备可能存在问题。因此当前区间是“人工审查后的可信修订”，不是物理真值。
- 同一 Session 中 RedMi K60 的 `gnss_log_2025_07_30_08_17_11` 暂时保持现状，不做额外剔除或重标。它在清单中继续作为 reviewed 数据存在。
- 新主楼静态与动态 Session 使用各自的 Session 级区间；未列出的新主楼 Session 应继续标为 `needs_review`，不得进入正式训练。

### 3.3 已知数据风险

- 操场动态 L15 即使收紧标签后，逐 signal LSTM validation Recall 仍只有 34.50%；操场动态 L5 Recall 仅 9.27%。这说明问题不只来自那一个过宽区间。
- 最新静态 outer-session 实验中，操场长 L5 Session 的 Macro-F1 为 0.6497、Recall 为 0.4754、FAR 为 12.38%；操场 L15 Session 的 Recall 为 0.4786。这两个 Session 应优先做逐设备、逐频段和原始曲线复核。
- 新主楼 L5 的正类数量偏少并不等于张量漏样本。当前标签语义是“目标 TOW 区间且 `FreqBand == 5` 才为正类”；同录制中的 L1 信号仍为负类。Pixel Watch 在这些录制中通常没有 L5 信号，因此会贡献大量负类而没有 L5 正类。
- Google Pixel Watch 1 的 `AgcDb` 全缺失。绝对 AGC 统计很容易携带设备身份，必须保留缺失标记，并在后续做 no-AGC 或相对化消融。
- `pipeline_total/02_batch_plot_feature_images.py` 的阴影仍基于场景级固定区间，不能把历史 PNG 当作当前标签真值；标签判断以 `configs/preprocessing.yml` 和重建后的 CSV 为准。

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

## 5. 关键实验摘要

| 实验 | 主要结果 | 能说明什么 | 不能说明什么 |
|---|---|---|---|
| 修订标签后的 mixed 逐 signal 基线 | LSTM validation Macro-F1 0.7947、Recall 0.5514、FAR 2.15% | 动态数据是主要漏检来源；短时序比 MLP 有帮助 | test 未读，不能代表泛化 |
| 单一静态固定划分 raw+stats | TCN validation 0.9906，test 仅 0.7015 | 单 Session validation 会造成严重选择过拟合 | 不能把 0.99 当作静态性能 |
| 静态 4-fold，W5 正则配置 | Macro-F1 0.8245 ± 0.0877，FAR 8.34% ± 5.33% | 跨 Session 波动显著，轻量正则略降误报 | 4 折样本仍只有 8 个独立录制 |
| W3/W5/W7 | 0.8154 / 0.8245 / 0.8151 | W5 略优，可作默认 | 差异不支持“W5 显著最好” |
| 6 train / 1 val / 1 test | pooled Macro-F1 0.8232、Recall 0.7487、FAR 9.57% | 增加 train Session 没有解决泛化 | 单 validation Session 仍不稳定 |
| Outer-Session / Inner-Time-Block | pooled Macro-F1 0.8386、Precision 0.7968、Recall 0.7168、FAR 6.19%；7 个有正类 Session 的 Macro-F1 0.8515 ± 0.1159 | 其余 Session 都可参与开发，误报较 6/1/1 少 | inner validation 仍是同 Session 时间块；outer test 已全部读取 |

最后一项是当前最有交接价值的结果，但仍不能锁定模型：inner validation Macro-F1 为 0.9382 ± 0.0112，outer test 的有正类 Session 均值只有 0.8515 ± 0.1159，且不同 Session 差异很大。所谓“样本量百万级”只是高度相关的 signal endpoints；真正独立的静态录制单元只有 8 个。

方向二与 6/1/1 也不是严格的单变量 split 消融：新的 builder 同时增加了断档窗口过滤和按 train 均值填充缺失值。因此两者的数值差不能全部归因于划分方式。

## 6. 整理后的代码定位

`pipeline_total/01–10` 是既有数据、画图、标注、基础张量、训练和错误分析链，本次不改动其结构。

`pipeline_total/11–18` 保留为 P0–P5 历史设备级探索。它们可以复现旧实验，但不应作为新的默认入口。

当前逐 signal 静态实验收敛为三个脚本：

```text
19_generate_static_timeblock_protocol.py
20_build_static_timeblock_tensors.py
21_train_static_signal_fusion.py
```

原有 stats-only builder、6/1/1 生成器和重复编号的两个 22 脚本已被替代。

## 7. 最小重建流程

标签或原始数据发生变化后，先重建中央 CSV 和审计文件：

```powershell
python pipeline_total/04_build_labeled_processed_csv.py --mode full --config configs/preprocessing.yml
python scripts/build_data_manifest.py
python scripts/audit_extracted_csv.py --input-dir data_csv --output-json output/data_csv_audit.json
python scripts/build_csv_session_manifest.py --input-dir data_csv --output-csv docs/data_csv_session_manifest.csv
```

重建最近的静态 time-block 协议：

```powershell
python pipeline_total/19_generate_static_timeblock_protocol.py
```

每个 outer fold 分别构建张量；`epoch_split_manifest.csv` 是权威逐历元划分，不能误用仅作汇总的 `time_block_manifest.csv`：

```powershell
python pipeline_total/20_build_static_timeblock_tensors.py `
  --outer-manifest output/protocols/static_time_block_outer_v1/fold_1/recording_split_manifest.csv `
  --block-manifest output/protocols/static_time_block_outer_v1/fold_1/epoch_split_manifest.csv `
  --output-dir output/tensors/static_timeblock_outer_v1/fold_1 `
  --time-steps 5
```

训练与 test-only 的命令以 `pipeline_total/README.md` 为准。每折 scaler 只能用 train 拟合，validation 只用于早停；test checkpoint 锁定后再读。当前 8 个 outer tests 已经全部读取过，后续依据这些结果调参时必须记作迭代式 CV，不能再称完全盲测。

## 8. 本地生成物保留策略

本次清理后，`output/` 只保留：

- `processed_gnss_data.csv` 与缺失报告；
- `data_csv_audit.json`；
- `dynamic_labeling_review/`；
- `review/trusted_signal_baseline_v1/` 下的关键错分明细与汇总；
- `output/README.md`。

张量、checkpoint、训练日志、旧 plots、smoke 目录和压缩副本全部视为可重建产物，不再长期保存。它们已集中迁入被 Git 忽略的 `output/_rebuildable_archive_20260722/`；当前执行环境的递归删除审批服务异常，因此磁盘空间尚未真正释放。确认无需恢复后，人工删除这一个归档目录即可。历史指标已经压缩进本文和 P0–P5 台账。

## 9. 推荐交接顺序

1. 先按 Session 建立 `trusted / questionable / excluded` 数据清单，重点复核操场动态 L5、动态 L15、静态长 L5 和静态 L15；不要把“reviewed”简单等同于物理真值。
2. 明确论文主任务以卫星/逐 signal 检测为主，设备级聚合作为部署层或辅助实验；不要在两种评价单位之间混用指标。
3. 新增独立录制或保留真正未触碰的 Session。只有 8 个静态录制时，再复杂的交叉验证也不能创造新的环境多样性。
4. 固定可信数据、特征和 split 后，再重跑 MLP、TCN/LSTM、raw+stats 等少量轻量基线；优先报告 Session×Device 宏平均、最差设备 FAR、攻击 Recall 和检测时延。
5. 在数据问题没有澄清前，不建议继续融合更多模型或扩大网络容量。若以后做融合，应先证明不同模型的错误具有互补性，而不是只比较 pooled Accuracy。

## 10. 文档入口

- 当前状态与交接：本文。
- P0–P5 历史实验：`docs/experiment_registry.md`。
- 数据清单：`docs/data_inventory.md`、`docs/data_manifest.csv`、`docs/data_csv_session_manifest.csv`。
- 标签复核：`docs/dynamic_labeling_assistant.md`、`docs/dy_manual_label_intervals.csv`。
- 信号级数据构建：`docs/signal_level_feature_extraction.md`。
- 历史静态 4-fold 协议：`docs/static_session_cv_protocol.md`。
- 脚本索引：`pipeline_total/README.md`。

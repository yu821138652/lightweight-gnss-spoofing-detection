# pipeline_total 脚本索引

当前状态以 `docs/handoff_status.md` 为准。本目录分为三段：01–10 是既有数据与基础实验链；11–18 是 P0–P5 历史设备级探索；19–21、23 是最近的静态逐 signal 实验入口。22 是当前推荐的 Session 级标签审查工具。

## 01–10：既有数据与诊断链

这些脚本保留既有结构；02 已补充按当前 YAML 配置解析 Session 级标签阴影。

| 编号 | 脚本 | 作用 |
|---:|---|---|
| 01 | `01_generate_plot_feature_csv.py` | 从原始日志生成逐日志 plot feature CSV |
| 02 | `02_batch_plot_feature_images.py` | 批量生成特征 PNG |
| 03 | `03_interactive_labeling_helper.py` | 交互式列出和复核候选日志/标签 |
| 04 | `04_build_labeled_processed_csv.py` | 从原始日志重建统一带标签 CSV |
| 05 | `05_build_train_val_test_tensors.py` | 构建旧逐 signal 张量与 recording split |
| 06 | `06_verify_tensor_splits.py` | 检查张量划分和泄漏 |
| 07 | `07_train_models.py` | 训练旧逐 signal 基线 |
| 08 | `08_inference.py` | 推理与指标输出 |
| 09 | `09_export_validation_misclassifications.py` | 导出 validation 错分 |
| 10 | `10_plot_validation_error_review.py` | 生成错分复核图 |
| 22 | `22_generate_label_review_dashboards.py` | 按完整 Session 整合全设备、全特征的正式标签审查面板 |
| 23 | `23_evaluate_static_fusion_groups.py` | 用锁定的双分支 checkpoint 输出设备×频段 test 指标 |

注意：

- 02 只按 `Environment + Scenario + Session` 读取 `configs/preprocessing.yml` 中的 reviewed 区间；缺少显式 Session 配置时不画阴影，也不会再回退到脚本内或场景级常量。操场动态 L15 当前权威区间是 `[260990, 261020]`；PNG 仍是复核辅助证据，不替代配置和 CSV 标签。
- 22 读取 `data_csv/` 的每日志镜像 CSV，并以 `configs/preprocessing.yml` 的当前 Session 级配置为阴影来源；同时检查镜像 CSV 的 `Label` 是否已随配置重建。人工全量审查优先使用 22 的面板，而非 02 的历史单图。
- 04 和配置文件是标签变化后的正式重建入口。
- 05 的基础接口保留给旧路线；最近的 time-block 实验使用 20。

标签相关数据重建：

```powershell
python scripts/build_mirrored_data_csv.py --config configs/preprocessing.yml --overwrite
python pipeline_total/04_build_labeled_processed_csv.py --mode full --config configs/preprocessing.yml
python pipeline_total/01_generate_plot_feature_csv.py --data-root data_raw --config configs/preprocessing.yml --overwrite
```

重建两套逐设备、逐特征 label plots：

```powershell
python pipeline_total/02_batch_plot_feature_images.py `
  --input-base data_raw/new_building `
  --output-base output/label_plots_20260723/new_building `
  --config configs/preprocessing.yml

python pipeline_total/02_batch_plot_feature_images.py `
  --input-base data_raw/playground `
  --output-base output/label_plots_20260723/playground `
  --config configs/preprocessing.yml
```

生成所有 Session 的标签审查包：

```powershell
python pipeline_total/22_generate_label_review_dashboards.py `
  --input-dir data_csv `
  --output-dir output/label_review_dashboards
```

输出根目录中的 `index.html` 用于浏览；每个 Session 另有 `dashboard.png`（全设备、标签时间轴和 7 项特征）与 `signals.csv`（逐 `signal_id` 清单）。`session_review_index.csv` 中的 `label_mismatch_rows > 0` 表示镜像 CSV 与正式配置不一致，应先重建该 Session 再训练。

## 11–18：P0–P5 历史设备级探索

这些脚本保留原位以便追溯，不是当前默认主链。

| 编号 | 脚本 | 历史用途 |
|---:|---|---|
| 11 | `11_evaluate_device_aggregation.py` | 将逐 signal 预测聚合为设备告警 |
| 12 | `12_generate_static_session_cv_manifests.py` | 生成静态 4-fold Session-CV 清单 |
| 13 | `13_build_device_stats_tensors.py` | 构建设备级 27 维统计张量 |
| 14 | `14_train_device_models.py` | 训练设备级 MLP/TCN/RNN/Linear/TSMixer |
| 15 | `15_train_device_lightgbm.py` | 训练设备级 LightGBM |
| 16 | `16_collect_device_experiment_results.py` | 汇总历史设备级实验 |
| 17 | `17_generate_static_dynamic_cv_manifests.py` | 向静态 CV train 加入动态 Session |
| 18 | `18_evaluate_device_motion_subgroups.py` | 按静态/动态子组评估设备模型 |

结果与边界见 `docs/experiment_registry.md`。15/18 需要可选的 LightGBM 依赖；未安装时不应把它们当成基础环境自检入口。

## 19–21、23：当前静态逐 signal 探索

这条链复现当前的 7-Session outer-session / inner-time-block W5 实验。当前保留开发基线是 `compact11 + TCN16 + dropout=.1`，产物位于 `output/training/static_timeblock_outer_v2_explore_compact11_tcn16_d10/`；它仍是探索协议，不是最终模型。完整结果、难例和可信度边界见 `docs/handoff_status.md`。

### 19_generate_static_timeblock_protocol.py

输入当前中央 CSV 和静态 recording 清单。对每个静态 recording 生成一个 outer fold：完整 recording 作为 test，其余 recording 在连续 canonical UTC 时间块内划分 train/validation，并在边界加入 W-1 guard。

当前保留基线使用的 7-Session 源清单位于 `output/protocols/static_time_block_outer_v2/source_recording_manifest.csv`。它只包含当前中央 CSV 中仍存在的 reviewed 静态 Session；不要继续使用包含已剔除短时操场 L5 的历史 4-fold 清单。以下命令刻意写入独立的 `*_repro_v1` 目录；**不要在保留目录 `static_timeblock_outer_v2_explore_compact11_tcn16_d10/` 中执行训练命令**，以免覆盖已锁定的 checkpoint 和诊断结果。

```powershell
python pipeline_total/19_generate_static_timeblock_protocol.py `
  --csv output/processed_gnss_data.csv `
  --source-recording-manifest output/protocols/static_time_block_outer_v2/source_recording_manifest.csv `
  --output-dir output/protocols/static_time_block_outer_v2_repro_v1 `
  --time-steps 5 `
  --block-epochs 256 `
  --val-fraction 0.20 `
  --segment-gap-seconds 2
```

上述独立复现命令的输出结构：

```text
output/protocols/static_time_block_outer_v2_repro_v1/
  fold_assignment.csv
  fold_summary.csv
  protocol_metadata.json
  fold_N/
    recording_split_manifest.csv
    time_block_manifest.csv
    epoch_split_manifest.csv
    recording_summary.csv
```

`epoch_split_manifest.csv` 是权威逐历元划分。生成器允许任意不少于 2 个 reviewed 静态 recording，不把 Session 数写死；本轮输入清单为 7 个 Session。

### 20_build_static_timeblock_tensors.py

按单个 outer fold 构建配对 raw/stats 张量。窗口不会跨 split、guard、segment 或 source 内大于 2 秒的断档；scaler 只用 train 拟合。

```powershell
python pipeline_total/20_build_static_timeblock_tensors.py `
  --outer-manifest output/protocols/static_time_block_outer_v2_repro_v1/fold_1/recording_split_manifest.csv `
  --block-manifest output/protocols/static_time_block_outer_v2_repro_v1/fold_1/epoch_split_manifest.csv `
  --output-dir output/tensors/static_timeblock_outer_v2_repro_v1/fold_1 `
  --time-steps 5 `
  --block-size 256
```

输出结构：

```text
fold_1/
  raw/{train,val,test}.npz
  raw/feature_names.json
  stats/{train,val,test}.npz
  stats/feature_names.json
```

raw 张量为兼容 builder 仍保存 7 列；训练器按 `feature_names.json` 只选择 5 列，排除 `Cn0DbHz_dt` 和 `Cn0DbHz_std`。stats 为逐 `signal_id` 的 19 维窗口统计。

默认 `--agc-common-mode none` 保持绝对 AGC。`--agc-common-mode same_time_band_median` 会在同一 source、同一时刻、同频段内把 AGC 改为相对中位数残差，并重算 AGC 统计；它仅用于 fold-6 诊断，当前不是保留基线。

标签敏感性实验可显式使用 `--label-policy l5_l1_positive --label-config configs/preprocessing.yml`：它保留原标签，并把配置中 reviewed `st_L5` 欺骗区间内的同期 L1 endpoint 临时改为正类。运行时必须指定独立的 `--output-dir`，不要覆盖正式张量；`tensor_metadata.json` 会记录策略、配置 SHA-256、实际区间快照、改标行数和受影响 Session。该策略不修改中央 CSV 或 YAML，也不代表已确认这类 L1 必然属于欺骗；若没有解析到 reviewed 区间、没有匹配行或没有实际改标，构建器会直接报错。示例：

```powershell
python pipeline_total/20_build_static_timeblock_tensors.py `
  --outer-manifest output/protocols/static_time_block_outer_v2/fold_1/recording_split_manifest.csv `
  --block-manifest output/protocols/static_time_block_outer_v2/fold_1/epoch_split_manifest.csv `
  --output-dir output/tensors/static_timeblock_outer_v2_l5_l1_positive/fold_1 `
  --time-steps 5 `
  --block-size 256 `
  --label-policy l5_l1_positive `
  --label-config configs/preprocessing.yml
```

### 21_train_static_signal_fusion.py

训练 raw 因果 TCN/LSTM + stats MLP 双分支。脚本会校验 raw/stats 的特征名、shape、mask、标签和设备元数据是否一致。

`--stats-feature-set full` 是 19 维历史基线。当前默认复现实验使用 `cn0_agc_coverage_rx_time_std`（compact11）：C/N0、AGC 各四项 `Last/Mean/Std/Slope`，接收机时间不确定度的 `Std`，以及两个 coverage ratio。`cn0_agc_coverage` 是 10 维联合消融，不是候选默认值。各 profile 都只在训练读取时按名称选列，所以无需重建张量，且不得覆盖其它 profile 的输出目录。`--test-only` 必须指定与 checkpoint 相同的 feature set，脚本会严格校验其记录的特征名。

先做轻量检查：

```powershell
python pipeline_total/21_train_static_signal_fusion.py `
  --data-dir output/tensors/static_timeblock_outer_v2_repro_v1/fold_1 `
  --output-dir output/training/static_timeblock_outer_v2_compact11_repro_v1/fold_1/tcn `
  --encoder tcn `
  --hidden-dim 16 `
  --dropout 0.1 `
  --lr 0.001 `
  --weight-decay 0.001 `
  --raw-feature-set full `
  --stats-feature-set cn0_agc_coverage_rx_time_std `
  --batch-size 256 `
  --num-workers 0 `
  --dry-run
```

正式训练：

```powershell
python pipeline_total/21_train_static_signal_fusion.py `
  --data-dir output/tensors/static_timeblock_outer_v2_repro_v1/fold_1 `
  --output-dir output/training/static_timeblock_outer_v2_compact11_repro_v1/fold_1/tcn `
  --encoder tcn `
  --hidden-dim 16 `
  --dropout 0.1 `
  --lr 0.001 `
  --weight-decay 0.001 `
  --raw-feature-set full `
  --stats-feature-set cn0_agc_coverage_rx_time_std `
  --epochs 30 `
  --batch-size 256 `
  --patience 6 `
  --seed 2026 `
  --num-workers 0
```

checkpoint 锁定后再读取 test：

```powershell
python pipeline_total/21_train_static_signal_fusion.py `
  --data-dir output/tensors/static_timeblock_outer_v2_repro_v1/fold_1 `
  --output-dir output/training/static_timeblock_outer_v2_compact11_repro_v1/fold_1/tcn `
  --encoder tcn `
  --raw-feature-set full `
  --stats-feature-set cn0_agc_coverage_rx_time_std `
  --batch-size 256 `
  --num-workers 0 `
  --test-only
```

`--test-only` 会从 checkpoint 恢复 encoder、hidden、dropout 和输入维度，并校验当前张量特征；不会信任不一致的命令行网络参数。

### 23_evaluate_static_fusion_groups.py

只读取已锁定 checkpoint，不参与训练或早停。即使 `IsL5` 被模型消融，脚本仍从完整 stats 张量中把它作为只读评估 sidecar，输出 `DeviceName × {L1,L5}` 的 support、TN/FP/FN/TP、Precision、Recall、FAR 和 Macro-F1。

```powershell
python pipeline_total/23_evaluate_static_fusion_groups.py `
  --data-dir output/tensors/static_timeblock_outer_v2_repro_v1/fold_6 `
  --checkpoint output/training/static_timeblock_outer_v2_compact11_repro_v1/fold_6/tcn/best_signal_tcn_stats_mlp_fusion.pt `
  --output-dir output/training/static_timeblock_outer_v2_compact11_repro_v1/fold_6/tcn `
  --batch-size 256
```

## 生成物策略

协议 CSV、NPZ、checkpoint、metrics、plots 和 smoke 目录都写入 `output/`，默认可重建且不提交 Git。当前只长期保留中央 CSV、审计、标签复核证据和必要错分明细；详情见 `output/README.md`。

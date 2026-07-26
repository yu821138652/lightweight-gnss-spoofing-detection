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

### 25_train_static_band_experts.py

`25_train_static_band_experts.py` 是针对 L5-only 难例的**独立探索实验**。它不改变标签、窗口、outer-session / inner-time-block 划分或 compact11 特征，而是使用 stats 张量中未缩放、在线可得的 `IsL5` sidecar 将每条卫星信号路由到独立的 L1 或 L5 `TCN + stats MLP` 专家。专家内部不再输入常量 `FreqBand`；它不是场景、攻击类型或标签的代理，不能据此预知攻击频段。

先以 fold 6 验证结构本身。不要覆盖 `static_timeblock_outer_v2_explore_compact11_tcn16_d10/` 的保留基线：

```powershell
python pipeline_total/25_train_static_band_experts.py `
  --data-dir output/tensors/static_timeblock_outer_v2/fold_6 `
  --output-dir output/training/static_timeblock_outer_v2_band_experts_v1/fold_6/tcn `
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

训练只读取 train/val。锁定 checkpoint 后才运行测试；测试会同时输出总体、L1/L5 和 device x band 指标：

```powershell
python pipeline_total/25_train_static_band_experts.py `
  --data-dir output/tensors/static_timeblock_outer_v2/fold_6 `
  --output-dir output/training/static_timeblock_outer_v2_band_experts_v1/fold_6/tcn `
  --encoder tcn `
  --raw-feature-set full `
  --stats-feature-set cn0_agc_coverage_rx_time_std `
  --batch-size 256 `
  --num-workers 0 `
  --test-only
```

### 26_train_static_conditional_heads.py (E1)

E1 保留当前正式的 `target-band-only` 标签和 compact11 输入，但使用一个共享 raw+stats 编码器与两个由未缩放 `IsL5` 选择的 L1/L5 分类头。训练时以单个有效卫星窗口为采样单位，每个 batch 严格包含相同数量的 `L1-`、`L1+`、`L5-`、`L5+`。这与 25 的两个独立专家不同；它只检验“共享表征 + 频段条件化头 + frequency x class 均衡”这一单独假设。

先在 fold 6 执行，使用独立输出目录：

```powershell
python pipeline_total/26_train_static_conditional_heads.py `
  --data-dir output/tensors/static_timeblock_outer_v2/fold_6 `
  --output-dir output/training/static_timeblock_outer_v2_e1_conditional_heads_v2/fold_6/tcn `
  --encoder tcn `
  --hidden-dim 16 `
  --dropout 0.1 `
  --lr 0.001 `
  --weight-decay 0.001 `
  --raw-feature-set full `
  --stats-feature-set cn0_agc_coverage_rx_time_std `
  --epochs 30 `
  --batch-size 8192 `
  --eval-batch-size 256 `
  --steps-per-epoch 128 `
  --patience 6 `
  --seed 2026 `
  --num-workers 0
```

锁定 checkpoint 后再测试：

```powershell
python pipeline_total/26_train_static_conditional_heads.py `
  --data-dir output/tensors/static_timeblock_outer_v2/fold_6 `
  --output-dir output/training/static_timeblock_outer_v2_e1_conditional_heads_v2/fold_6/tcn `
  --encoder tcn `
  --raw-feature-set full `
  --stats-feature-set cn0_agc_coverage_rx_time_std `
  --eval-batch-size 256 `
  --num-workers 0 `
  --test-only
```

### 27_train_static_auxiliary_state.py (E3)

E3 keeps the formal binary target-band spoofing label as the only deployable
output and the only primary metric.  It adds a training-only three-class head:
`normal`, `target_spoofed`, and `non_target_single_band_attack`.  The third
state is derived from the reviewed `st_L1`/`st_L5` TOW intervals in
`configs/preprocessing.yml`, using the endpoint TOW and recording trace stored
in the existing tensors.  It is an auxiliary operational context label, not a
claim that a non-target signal is spoofed.  E3 therefore differs from the E4
label-sensitivity control: it never changes the original `y` labels and needs
no CSV or tensor rebuild.

Start with the fixed fold-6 diagnostic and retain the same W5, compact11,
TCN16, train/validation protocol as the current baseline.  The checkpoint is
selected only by the original binary validation Macro-F1; auxiliary metrics
are reported separately.

```powershell
python pipeline_total/27_train_static_auxiliary_state.py `
  --data-dir output/tensors/static_timeblock_outer_v2/fold_6 `
  --output-dir output/training/static_timeblock_outer_v2_e3_auxiliary_state_v1/fold_6/tcn `
  --label-config configs/preprocessing.yml `
  --encoder tcn `
  --hidden-dim 16 `
  --dropout 0.1 `
  --lr 0.001 `
  --weight-decay 0.001 `
  --raw-feature-set full `
  --stats-feature-set cn0_agc_coverage_rx_time_std `
  --aux-loss-weight 0.25 `
  --epochs 30 `
  --batch-size 256 `
  --patience 6 `
  --seed 2026 `
  --num-workers 0
```

After the validation-selected checkpoint is locked, evaluate the original
binary detector and the auxiliary state report on fold 6:

```powershell
python pipeline_total/27_train_static_auxiliary_state.py `
  --data-dir output/tensors/static_timeblock_outer_v2/fold_6 `
  --output-dir output/training/static_timeblock_outer_v2_e3_auxiliary_state_v1/fold_6/tcn `
  --label-config configs/preprocessing.yml `
  --encoder tcn `
  --raw-feature-set full `
  --stats-feature-set cn0_agc_coverage_rx_time_std `
  --batch-size 256 `
  --num-workers 0 `
  --test-only
```

### 28_train_static_cross_band_context.py (E5a)

E5a keeps the formal binary target-band spoofing task unchanged and introduces
no interval-derived training target.  It conditions every signal prediction
on causal, same-device/source endpoint context: L1/L5 visible counts, C/N0 and
AGC means, and their changes from the preceding W5 history.  Context is
computed from the current tensor batch only; it never uses labels, TOW,
scenario, or future observations.  The initial diagnostic is fold 6, with
the same compact11 TCN16 configuration as E0.

```powershell
python pipeline_total/28_train_static_cross_band_context.py `
  --data-dir output/tensors/static_timeblock_outer_v2/fold_6 `
  --output-dir output/training/static_timeblock_outer_v2_e5a_cross_band_context_v1/fold_6/tcn `
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

After checkpoint selection, test writes both the primary binary metrics and
`test_metrics_by_device_band.csv` for the fold-6 Huawei-L1 and L5 diagnostics.

```powershell
python pipeline_total/28_train_static_cross_band_context.py `
  --data-dir output/tensors/static_timeblock_outer_v2/fold_6 `
  --output-dir output/training/static_timeblock_outer_v2_e5a_cross_band_context_v1/fold_6/tcn `
  --encoder tcn `
  --raw-feature-set full `
  --stats-feature-set cn0_agc_coverage_rx_time_std `
  --batch-size 256 `
  --num-workers 0 `
  --test-only
```

### 29_train_static_context_attack_aux.py (E5b)

E5b extends E5a's causal same-endpoint L1/L5 context with a second,
training-only `attack_associated` head.  The primary output and all selection
metrics remain the unchanged formal target-band-only binary label.  The
auxiliary target is 1 for every active signal in a reviewed static attack TOW
interval from `configs/preprocessing.yml`, and 0 outside it.  It is never used
as an input feature or deployment output.  The initial auxiliary loss weight
is deliberately small (`0.05`) because E3 showed that a stronger interval
supervision can overwhelm the primary signal task.

```powershell
python pipeline_total/29_train_static_context_attack_aux.py `
  --data-dir output/tensors/static_timeblock_outer_v2/fold_6 `
  --output-dir output/training/static_timeblock_outer_v2_e5b_context_attack_aux_v1/fold_6/tcn `
  --label-config configs/preprocessing.yml `
  --encoder tcn `
  --hidden-dim 16 `
  --dropout 0.1 `
  --lr 0.001 `
  --weight-decay 0.001 `
  --raw-feature-set full `
  --stats-feature-set cn0_agc_coverage_rx_time_std `
  --aux-loss-weight 0.05 `
  --epochs 30 `
  --batch-size 256 `
  --patience 6 `
  --seed 2026 `
  --num-workers 0
```

After checkpoint selection, E5b writes formal primary metrics, the auxiliary
attack-associated metrics, and the usual device-by-band primary CSV:

```powershell
python pipeline_total/29_train_static_context_attack_aux.py `
  --data-dir output/tensors/static_timeblock_outer_v2/fold_6 `
  --output-dir output/training/static_timeblock_outer_v2_e5b_context_attack_aux_v1/fold_6/tcn `
  --label-config configs/preprocessing.yml `
  --encoder tcn `
  --raw-feature-set full `
  --stats-feature-set cn0_agc_coverage_rx_time_std `
  --batch-size 256 `
  --num-workers 0 `
  --test-only
```

### 30_train_static_attack_associated.py (E6)

E6 is a separate direct task, not an improvement claim over the formal
target-band detector.  Every active signal inside a reviewed static attack TOW
interval becomes positive, independent of its frequency band; every active
signal outside is negative.  The target is derived from the tensor endpoint
trace plus `configs/preprocessing.yml` at load time, leaving the central CSV
and formal tensor `y` unchanged.  Its metrics must be reported as
`attack-associated anomaly detection`, separately from E0/E1/E2/E3/E5.

```powershell
python pipeline_total/30_train_static_attack_associated.py `
  --data-dir output/tensors/static_timeblock_outer_v2/fold_6 `
  --output-dir output/training/static_timeblock_outer_v2_e6_attack_associated_v1/fold_6/tcn `
  --label-config configs/preprocessing.yml `
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

### 31_train_static_direct_state.py (E7)

E7 makes the three-way state the primary prediction: `normal`, formal
`target_spoofed`, and `non_target_single_band_attack`.  The latter is only an
attack-associated context label for an active non-target-band signal within a
reviewed single-band attack interval; it does not claim that signal is
spoofed.  The report includes both direct three-class metrics and two clearly
marked binary projections, including the original E0 formal target-band
endpoint definition.  It is a separate task and must not be described as an
E0 improvement without comparing that projection.

```powershell
python pipeline_total/31_train_static_direct_state.py `
  --data-dir output/tensors/static_timeblock_outer_v2/fold_6 `
  --output-dir output/training/static_timeblock_outer_v2_e7_direct_state_v1/fold_6/tcn `
  --label-config configs/preprocessing.yml `
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

### 32_generate_static_inner_session_manifests.py

The current time-block validation is not representative of a complete unseen
Session.  This generator turns one fixed outer-test fold into leave-one-
development-Session-out inner folds.  Each generated block manifest assigns a
whole development Session to validation and every other development Session to
training; it removes the old within-Session validation split and its guard
gaps.  The outer test Session stays excluded from both inner train and val.

```powershell
python pipeline_total/32_generate_static_inner_session_manifests.py `
  --outer-manifest output/tensors/static_timeblock_outer_v2/fold_6/outer_recording_manifest.csv `
  --block-manifest output/tensors/static_timeblock_outer_v2/fold_6/block_manifest.csv `
  --output-dir output/protocols/static_inner_session_outer_v1/fold_6
```

Add `--write-refit-all-development` when a final outer-development refit is
needed after all model choices are locked.  It writes
`refit_all_development/block_manifest.csv`, assigning every development epoch
to train while retaining the complete outer test Session.

### E8: Causal Session Reference Features

`20_build_static_timeblock_tensors.py` can add per-source, per-signal C/N0
and AGC reference features with `--causal-reference-epochs`.  At each epoch,
the reference uses only preceding finite observations.  After the requested
number of observations, its median and MAD scale are frozen, preventing a
sustained attack from being treated as the new normal.  A `CausalReferenceReady`
feature distinguishes the initial uncalibrated period.  The option is disabled
by default and does not change historical tensor contracts.

Fold 6 has approximately 590 seconds before the reviewed L5 attack begins,
so use 120 source/signal epochs as the initial online calibration window:

```powershell
python pipeline_total/20_build_static_timeblock_tensors.py `
  --csv output/processed_gnss_data.csv `
  --outer-manifest output/tensors/static_timeblock_outer_v2/fold_6/outer_recording_manifest.csv `
  --block-manifest output/tensors/static_timeblock_outer_v2/fold_6/block_manifest.csv `
  --output-dir output/tensors/static_timeblock_outer_v2_e8_causalref120/fold_6 `
  --time-steps 5 `
  --causal-reference-epochs 120
```

Train with the historical E0 TCN hyperparameters and the dedicated
`causal_reference` raw profile.  This is the direct E8 comparison: five E0
raw features plus five causal-reference features, with unchanged labels and
stats branch.

```powershell
python pipeline_total/21_train_static_signal_fusion.py `
  --data-dir output/tensors/static_timeblock_outer_v2_e8_causalref120/fold_6 `
  --output-dir output/training/static_timeblock_outer_v2_e8_causalref120/fold_6/tcn `
  --encoder tcn `
  --hidden-dim 16 `
  --dropout 0.3 `
  --lr 0.001 `
  --weight-decay 0.001 `
  --raw-feature-set causal_reference `
  --stats-feature-set full `
  --epochs 30 `
  --batch-size 256 `
  --patience 6 `
  --seed 2026 `
  --num-workers 0
```

### 33_refit_static_signal_fusion.py (E9a / E9b)

E9a is a fixed-epoch final refit, not a validation experiment.  It trains on
all outer-development Sessions and never reads the outer test tensor.  Use it
only with an epoch count chosen before the refit.  The resulting checkpoint is
compatible with `21_train_static_signal_fusion.py --test-only --checkpoint`.

E9b keeps the same refit model, labels, and number of sampled windows per
epoch, but adds `--sampling session_uniform`.  It draws each window with an
inverse-session-size probability, so every development Session has roughly
equal expected representation in an epoch.  This is intended to test whether a
short L5 attack Session is being drowned out by longer Sessions; it is not a
change to the formal target-band label definition.

```powershell
python pipeline_total/33_refit_static_signal_fusion.py `
  --data-dir output/tensors/static_timeblock_outer_v2_e9a_refit_all_dev/fold_6 `
  --output-dir output/training/static_timeblock_outer_v2_e9b_session_uniform/fold_6/tcn `
  --encoder tcn `
  --epochs 10 `
  --hidden-dim 16 `
  --dropout 0.3 `
  --lr 0.001 `
  --weight-decay 0.001 `
  --raw-feature-set full `
  --stats-feature-set full `
  --sampling session_uniform `
  --batch-size 256 `
  --seed 2026 `
  --num-workers 0
```


## 生成物策略

协议 CSV、NPZ、checkpoint、metrics、plots 和 smoke 目录都写入 `output/`，默认可重建且不提交 Git。当前只长期保留中央 CSV、审计、标签复核证据和必要错分明细；详情见 `output/README.md`。

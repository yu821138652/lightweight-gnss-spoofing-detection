# Watch 专家评估链修复（2026-08-07）

## 1. 修复原因

原静态六折完整链条的 Watch 异常专家在 `fold_2` 存在无效阈值校准：其内层验证集经过 `anomaly_only` 过滤后只有 normal 样本，没有 anomaly 正样本。该情况下仍从验证集选择阈值，只能约束误报率，不能证明异常检测能力。

因此，旧完整链条结果（包括 pooled Macro-F1 `0.9802`、anomaly recall `0.9876`）仅保留为**开发性/探索性结果**，不能直接作为论文最终主结果。

## 2. 代码级防线

- `pipeline_total/37_train_device_attack_event.py`
  - 阈值校准现在要求校准集同时包含负类和正类；任一支持数为 0 时直接报错。
  - 校准 checkpoint 与 `val_threshold_calibration.json` 记录 `negative_support`、`positive_support` 和 `calibration_valid=true`。
  - 可用 `--calibration-data-dir` 指定独立的合法校准张量目录。
- `pipeline_total/55_run_scene_gated_watch_anomaly_cv.py`
  - 每折训练/校准前审计 Watch 的实际 `anomaly_only` 验证支持数，并输出 `watch_calibration_audit.json`。
  - 默认 `--invalid-calibration-policy error`：校准无效即停止，避免静默产生伪校准结果。
  - 仅在明确指定 `--invalid-calibration-policy disable` 时，才禁用该折 Watch 修复并导出未修复的基础预测；汇总结果会保留该状态。
- `pipeline_total/58_run_complete_scene_response_diagnosis_cv.py`
  - 最终融合与 `--aggregate-only` 都要求每折存在 Watch 校准审计文件；不会再把旧的、未经审计的 Watch 预测静默汇总为“完整系统”结果。
- `pipeline_total/63_build_watch_attack_aware_inner_split.py`
  - 为类似 fold_2 的情况构造合法内层划分。
  - 只从 outer-development 的 `train.npz` 与 `val.npz` 选择数据；outer-test 仅原样复制，**不参与样本选择、阈值或参数确定**。
  - 对有 anomaly 的 Watch 源流，保留末段 anomaly 和攻击后 normal 作为内层验证，并在训练侧留出时间 guard band，降低相邻滑窗泄漏。

## 3. fold_2 的已核验事实

旧标准张量的 Watch 验证支持为：

| split | normal | anomaly | direct（被 anomaly_only 排除） |
|---|---:|---:|---:|
| 原 inner val | 1380 | 0 | 620 |

但 outer-development 的原 train 中存在两个 Watch 的 `st_L5` 关联异常源流。以预注册的固定配置：每源保留 200 个末段 anomaly 窗口、100 个攻击后 normal 窗口，并在训练侧保留 30 窗口时间 guard，构建后得到：

| split | Watch normal | Watch anomaly |
|---|---:|---:|
| 新 inner train | — | 673 |
| 新 inner val | 200 | 400 |

因此，该折可以在完全不访问 outer-test 标签的情况下训练、模型选择和阈值校准。

## 4. 重新评估命令

以下命令在仓库根目录执行。训练由人工启动；输出使用新的目录，不覆盖旧开发性结果。

```powershell
$py = 'H:\GNSS\program\Release_Package\Release_Package\venv\Scripts\python.exe'

# 1) 为 fold_2 建立 attack-aware、且含两类 Watch 样本的 inner split。
& $py pipeline_total/63_build_watch_attack_aware_inner_split.py `
  --source-data-dir output/optimization/watch_suppression_warmup_cv_v1/fold_2/device_tensors_l1_suppression `
  --output-dir output/optimization/watch_suppression_audited_cv_v2/fold_2/device_tensors_l1_suppression `
  --positive-windows-per-source 200 `
  --post-attack-normal-windows-per-source 100 `
  --guard-windows 30 `
  --overwrite

# 2) 重跑六折 Watch 分支。fold_2 会复用上一步张量；其余折按原协议构建。
& $py pipeline_total/55_run_scene_gated_watch_anomaly_cv.py `
  --python-exe $py `
  --output-root output/optimization/watch_suppression_audited_cv_v2 `
  --folds 1 2 4 5 6 7 `
  --skip-existing

# 3) 重新生成完整静态联合诊断结果。
& $py pipeline_total/58_run_complete_scene_response_diagnosis_cv.py `
  --python-exe $py `
  --watch-output-root output/optimization/watch_suppression_audited_cv_v2 `
  --self-calibrated-output-root output/optimization/l5_self_calibrated_audited_cv_v2 `
  --output-root output/optimization/complete_scene_response_diagnosis_audited_cv_v2 `
  --folds 1 2 4 5 6 7
```

第 2 步如果再次出现单类验证集，会直接报错；此时不能通过调低阈值规避，必须修复其 inner split，或显式禁用该折 Watch 专家后再报告边界。

## 5. 新结果的报告要求

新结果产生后，论文和总结中需要同时保留：

1. 每折的 Watch `negative_support / positive_support` 与 `calibration_valid`；
2. `watch_calibration_audit.json` 中的 split 来源与时间 guard 配置；
3. 每折 Watch 修正数量；
4. pooled 与按设备结果；
5. 旧 `0.9802` 结果仅作为“修复前开发性结果”，不与新的可审计正式结果混用。

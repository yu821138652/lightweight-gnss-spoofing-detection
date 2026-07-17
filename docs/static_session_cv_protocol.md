# 静态数据 Session 级交叉验证协议

## 目的

当前 `static_cross_env_v1` 已完成一次固定的“新主楼训练、操场测试”评估。其测试集已经用于正式结果解释，后续不能继续以该测试集选择特征、模型或设备告警规则。

本协议用于下一轮开发比较。它以完整真实录制为最小分组单位，构建 4 个静态 Session 折次；同一次实验的所有设备日志只能出现在 train、val、test 的其中一个集合，不能跨集合。

每折包含：

- train：4 条真实录制；
- val：1 条新主楼录制和 1 条操场录制；
- test：1 条新主楼录制和 1 条操场录制。

这是一套 Session 泛化开发协议，不等同于纯跨环境部署测试。正式论文应报告 4 折 test 的均值和标准差，并另外保留既有 `static_cross_env_v1` 作为跨环境参考结果。

## 第一步：生成锁定清单

在首次创建折次，或静态录制清单发生真实变化时运行：

```powershell
& $PY pipeline_total\12_generate_static_session_cv_manifests.py
```

它会生成：

```text
docs/protocols/static_session_cv_4fold/
  fold_assignment.csv
  fold_1/recording_split_manifest.csv
  fold_2/recording_split_manifest.csv
  fold_3/recording_split_manifest.csv
  fold_4/recording_split_manifest.csv
```

先检查 `fold_assignment.csv`，确认每条录制在每个折次的角色符合预期。锁定后不应手动改动清单。

## 第二步：先审查单折划分

在实际构建张量前运行。此命令读取统一 CSV、应用锁定清单，但不生成 NPZ；用于确认行数、正样本和录制分配正确：

```powershell
& $PY pipeline_total\05_build_train_val_test_tensors.py `
  --csv output\processed_gnss_data.csv `
  --scenario static `
  --split-manifest docs\protocols\static_session_cv_4fold\fold_1\recording_split_manifest.csv `
  --output_dir output\tensors_static_session_cv\fold_1 `
  --split-only
```

## 第三步：构建单折张量

仅当第二步输出正确时运行。每折都使用同一 7 特征、5 历元因果窗口和训练集统计量归一化：

```powershell
& $PY pipeline_total\05_build_train_val_test_tensors.py `
  --csv output\processed_gnss_data.csv `
  --scenario static `
  --split-manifest docs\protocols\static_session_cv_4fold\fold_1\recording_split_manifest.csv `
  --output_dir output\tensors_static_session_cv\fold_1
```

将 `fold_1` 依次替换为 `fold_2`、`fold_3`、`fold_4`。不要将不同折次的张量、checkpoint 或指标写入同一目录。

## 第四步：训练和设备级评估

每个候选模型均在每折独立训练；训练只读 train，早停和规则选择只读 val。锁定该折的模型和设备聚合规则后，才读取该折 test：

```powershell
& $PY pipeline_total\07_train_models.py `
  --data-dir output\tensors_static_session_cv\fold_1 `
  --output-dir output\training\static_session_cv\fold_1\signal_lstm `
  --model signal_lstm `
  --epochs 30 `
  --batch-size 256 `
  --patience 6 `
  --seed 2026
```

设备级 test 评估：

```powershell
& $PY pipeline_total\11_evaluate_device_aggregation.py `
  --data-dir output\tensors_static_session_cv\fold_1 `
  --csv output\processed_gnss_data.csv `
  --model-dir output\training\static_session_cv\fold_1\signal_lstm `
  --model signal_lstm `
  --rule majority `
  --split test
```

## 结果记录

每折记录设备级 Macro-F1、Precision、Recall、FAR、参数量和推理时延。对同一模型报告 4 折均值和标准差，而不是只报告最好的一折。

特征消融和模型比较必须使用完全相同的 4 份清单、随机种子与设备级规则。若修改了窗口、标签语义、归一化方式或特征集合，则应作为新的实验配置单独记录。

# pipeline_total: GNSS 数据处理全流程脚本顺序

这个目录把本项目里与数据处理、画图标注、建模直接相关的脚本按真实实验顺序集中起来。

真实流程是：

```text
原始 GNSS TXT
  -> 生成逐日志信号级特征 CSV (SignalBand / signal_id)
  -> 画时序图
  -> 人眼判断欺骗 TOW 区间
  -> 更新标注配置
  -> 生成总 processed_gnss_data.csv
  -> 构建 train/val/test NPZ
  -> 检查张量
  -> 训练/推理
```

信号级数据规范、验证结果和常用命令见 `docs/signal_level_feature_extraction.md`。

## 00_preprocessing_config.yml

来源：`configs/preprocessing_template.yml`。完成新主楼标签配置后，复制为
`configs/preprocessing.yml` 供正式预处理使用。

这是预处理和打标签的主配置。重点维护：

- `paths.input_dir`
- `paths.output_csv`
- `device_model_map`
- `labeling.spoofing_tow_intervals`
- `labeling.spoofing_type_to_label`
- `final_columns`

人工看图标注后，主要就是把欺骗发生的 GPS TOW 区间写到 `labeling.spoofing_tow_intervals`。

正式运行时建议同步改项目根目录的：

```text
configs/preprocessing.yml
```

因为现有脚本默认读的是根目录配置。

## 01_generate_plot_feature_csv.py

来源：`scripts/generate_plot_features.py`

作用：扫描 `data_raw/` 下的原始 GNSS 日志，给每个 TXT 生成一个对应的 `*-plot_features.csv`，并优先按独立 `SignalID` 绘图。

这些 CSV 是后续画图和人工标注的中间文件。

常用命令：

```bash
python pipeline_total/01_generate_plot_feature_csv.py
```

只处理某个场景：

```bash
python pipeline_total/01_generate_plot_feature_csv.py --scenario st_L1
```

覆盖已有中间 CSV：

```bash
python pipeline_total/01_generate_plot_feature_csv.py --overwrite
```

注意：当前主数据统一位于仓库根目录 `data_raw/`。

## 02_batch_plot_feature_images.py

来源：`scripts/batch_plot_features.py`

作用：读取 `*-plot_features.csv`，批量画出各特征的信号时序图，输出到 `output_plots/`；旧 CSV 会回退为卫星级绘图。

常用命令：

```bash
python pipeline_total/02_batch_plot_feature_images.py
```

当前脚本里默认只处理 `st_L1` 和 `dy_L1`，如果要画全部场景，需要在 `main()` 里补齐：

```python
for scenario in ["st_L1", "st_L5", "st_L_15", "dy_L1", "dy_L5", "dy_L_15"]:
    process_scenario(scenario)
```

## 03_interactive_labeling_helper.py

来源：`labeling/run_labeling.py`

作用：交互式画 C/N0 和 AGC，辅助人工记录欺骗开始/结束 TOW。

示例：

```bash
python -m labeling.run_labeling --spoof_type dy_L1 --folder 2022.07.08semicircle
```

注意：原脚本仍带有旧路径配置。继续使用前，需将 `config.ROOT_DATA_DIR`
改为仓库根目录的 `data_raw/`，或者直接用前两步生成的 PNG 进行人工标注。

## 人工步骤：更新欺骗区间

看完图后，把区间写回：

```text
configs/preprocessing.yml
```

例如：

```yaml
labeling:
  spoofing_tow_intervals:
    dy_L1:
      - [263995, 264050]
      - [264690, 264740]
```

这一步是整个流程中最关键的人工环节。

## 04_build_labeled_processed_csv.py

来源：`pipeline/01_preprocess.py`

作用：从原始 TXT 重新解析、过滤、计算特征，并根据配置中的 TOW 区间打标签，生成总 CSV。

常用命令：

```bash
python pipeline_total/04_build_labeled_processed_csv.py --mode full --config configs/preprocessing.yml
```

默认输出：

```text
output/processed_gnss_data.csv
```

若需要自定义输出位置，可以复制配置并修改：

```yaml
paths:
  input_dir: './data_raw'
  output_csv: './output/processed_gnss_data.csv'
```

然后运行：

```bash
python pipeline_total/04_build_labeled_processed_csv.py --mode full --config your_config.yml
```

## 05_build_train_val_test_tensors.py

来源：`pipeline/02_build_tensors.py`

作用：把 `processed_gnss_data.csv` 构造成训练用 NPZ 张量，默认使用 `signal_id` 作为空间槽位并排除未审查标签。每个输入由截至当前时刻的 5 个历元组成，目标是窗口末端当前历元的标签，只有末端出现的信号参与损失和指标。训练/验证/测试以 `Environment + Scenario + Session` 为不可拆分的真实录制单元，同一场实验的多设备数据不会落入不同集合；每个设备日志仍生成独立张量，避免不同接收机的同名 `signal_id` 相互覆盖。划分会在数据允许时保证每个集合都含静态和动态录制。每次运行会输出 `recording_split_manifest.csv` 供复现与审计。

常用命令：

```bash
python pipeline_total/05_build_train_val_test_tensors.py --csv output/processed_gnss_data.csv --output_dir output/tensors_mixed --scenario mixed --max-signals 128
```

在首次构建张量前，先只生成并检查真实录制级划分清单。这样做是为了确认同一场实验的多设备数据没有跨集合泄漏：

```bash
python pipeline_total/05_build_train_val_test_tensors.py --csv output/processed_gnss_data.csv --output_dir output/tensors_mixed --scenario mixed --split-only
```

检查 `output/tensors_mixed/recording_split_manifest.csv` 后，再去掉 `--split-only` 生成 NPZ。

静态/动态分别构建：

```bash
python pipeline_total/05_build_train_val_test_tensors.py --csv output/processed_gnss_data.csv --output_dir output/tensors_static --scenario static
python pipeline_total/05_build_train_val_test_tensors.py --csv output/processed_gnss_data.csv --output_dir output/tensors_dynamic --scenario dynamic
```

## 06_verify_tensor_splits.py

来源：`scripts/verify_data.py`

作用：检查 `train.npz / val.npz / test.npz` 的张量形状、二分类标签分布，并读取 `recording_split_manifest.csv` 验证真实录制单元没有重复分配到不同集合。

常用命令：

```bash
python pipeline_total/06_verify_tensor_splits.py --npz_dir output/tensors_mixed
```

## 07_train_models.py

来源：`pipeline/03_train.py`

作用：训练项目自有的逐信号轻量 baseline。每条有效 `signal_id` 的 5 秒、7 特征窗口独立输出正常/欺骗 logits；同一设备窗口中的填充槽位会由 `mask` 排除。

首轮可选模型：

```text
signal_mlp：5 秒 x 7 特征直接展平，作为最低复杂度参考
signal_gru：保留 5 秒时间顺序的轻量 GRU，作为时序参考
```

先在未训练状态做干运行。此命令只读取 train/val，并验证形状、掩码与前向传播，不更新权重、不生成 checkpoint、也不读取 test：

```bash
C:\Users\Asus\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\python.exe pipeline_total/07_train_models.py --data-dir output/tensors_mixed --model signal_mlp --dry-run
```

模型结构和超参数确定后，才运行正式训练。训练过程只使用 train，依据 val 的 Macro-F1 早停并保存最佳权重：

常用命令：

```bash
C:\Users\Asus\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\python.exe pipeline_total/07_train_models.py --data-dir output/tensors_mixed --output-dir output/training/signal_mlp --model signal_mlp --epochs 30
```

只有在模型和超参数均已锁定后，才显式读取 test 并记录最终指标：

```bash
C:\Users\Asus\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\python.exe pipeline_total/07_train_models.py --data-dir output/tensors_mixed --output-dir output/training/signal_mlp --model signal_mlp --test-only
```

张量接口、模型扩展方式、测试集使用边界和 TSLib 适配原则见 `docs/model_training_framework.md`。

张量以连续 5 个历元构成因果输入窗口，标签固定为窗口末端当前历元的信号级 `Label`；窗口内先前历元只作为历史上下文。重建张量后，旧 checkpoint 及其指标不能与新标签语义下的结果混用。

## 08_inference.py

来源：`pipeline/04_inference.py`

作用：历史模型接口的 CSV 推理脚本。

**当前状态：** 该脚本尚未适配 `signal_mlp / signal_gru` checkpoint，不可用于本次新 baseline。原因是新模型输出逐信号概率，而部署推理还需先在验证集确定多信号设备级报警聚合规则。完成 baseline 选择与报警规则设计后，再单独适配该脚本；在此之前不得基于它形成部署性能结论。

示例：

```bash
python pipeline_total/08_inference.py --model_path output/best_model.pth --csv your_data.csv --output_csv predictions.csv
```

## 09_export_validation_misclassifications.py

来源：`pipeline_total/09_export_validation_misclassifications.py`

**何时运行：** 当前开发协议下，某个模型完成训练并保存最佳 validation checkpoint 后，且在修改特征、窗口长度、划分或开始任何 test 评估之前。

**为什么运行：** 导出 validation 集的逐信号错分样本，优先检查错分是否集中在少数 Session、设备、TOW 区间或某类信号。脚本会依据锁定的 `recording_split_manifest.csv` 重建窗口与信号槽位，并严格核对重建的 `mask` 和 `Label` 是否与 `val.npz` 一致；不一致时会拒绝生成 CSV，防止预测结果与原始数据行错位。该脚本不会读取 `test.npz`。

当前 Tiny Transformer 的示例：

```powershell
$PY = "C:\Users\Asus\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\python.exe"

& $PY pipeline_total\09_export_validation_misclassifications.py `
  --data-dir output\tensors_mixed `
  --csv output\processed_gnss_data.csv `
  --model-dir output\training\signal_transformer_tiny_current_protocol `
  --model signal_transformer_tiny
```

输出写入对应模型目录：

```text
validation_misclassifications_<model>.csv
validation_misclassifications_<model>_summary.csv
validation_misclassifications_<model>_by_recording.csv
validation_misclassifications_<model>_by_source_log.csv
validation_misclassifications_<model>_by_signal_band.csv
validation_misclassifications_<model>_by_tow.csv
```

主 CSV 仅包含 false positive 和 false negative，字段包括窗口起止时间、当前 TOW、录制环境、Session、设备、来源日志、`signal_id`、真实/预测标签、欺骗概率及 7 项当前历元特征。

四个汇总 CSV 分别用于定位错误集中在哪个录制单元、哪个设备源日志、哪个信号频段、哪个当前 TOW。它们基于完整 validation 预测计算 TP、TN、FP、FN、Recall、漏检率、FAR 与总体错误率，不能只按错分主 CSV 的行数推断比例。

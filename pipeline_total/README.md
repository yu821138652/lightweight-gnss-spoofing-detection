# pipeline_total: GNSS 数据处理全流程脚本顺序

这个目录把本项目里与数据处理、画图标注、建模直接相关的脚本按真实实验顺序集中起来。

真实流程是：

```text
原始 GNSS TXT
  -> 生成逐日志特征 CSV
  -> 画时序图
  -> 人眼判断欺骗 TOW 区间
  -> 更新标注配置
  -> 生成总 processed_gnss_data.csv
  -> 构建 train/val/test NPZ
  -> 检查张量
  -> 训练/推理
```

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

作用：扫描 `data_raw/` 下的原始 GNSS 日志，给每个 TXT 生成一个对应的 `*-plot_features.csv`。

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

作用：读取 `*-plot_features.csv`，批量画出各特征的卫星时序图，输出到 `output_plots/`。

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

作用：把 `processed_gnss_data.csv` 构造成训练用 NPZ 张量。

常用命令：

```bash
python pipeline_total/05_build_train_val_test_tensors.py --csv output/processed_gnss_data.csv --output_dir output/tensors_mixed --scenario mixed
```

静态/动态分别构建：

```bash
python pipeline_total/05_build_train_val_test_tensors.py --csv output/processed_gnss_data.csv --output_dir output/tensors_static --scenario static
python pipeline_total/05_build_train_val_test_tensors.py --csv output/processed_gnss_data.csv --output_dir output/tensors_dynamic --scenario dynamic
```

## 06_verify_tensor_splits.py

来源：`scripts/verify_data.py`

作用：检查 `train.npz / val.npz / test.npz` 内部张量形状、标签分布、划分是否健康。

常用命令：

```bash
python pipeline_total/06_verify_tensor_splits.py --npz_dir output/tensors_mixed --csv output/processed_gnss_data.csv
```

## 07_train_models.py

来源：`pipeline/03_train.py`

作用：训练模型。

常用命令：

```bash
python pipeline_total/07_train_models.py --data_dir output/tensors_mixed --model st_mamba --epochs 100
```

支持模型包括：

```text
lstm
mamba
st_mamba
st_mamba_gated
st_mamba_core_residual
st_mamba_adaptive_core_residual
transformer
cnn
```

## 08_inference.py

来源：`pipeline/04_inference.py`

作用：加载训练好的模型，对 CSV 做推理并输出预测结果。

示例：

```bash
python pipeline_total/08_inference.py --model_path output/best_model.pth --csv your_data.csv --output_csv predictions.csv
```

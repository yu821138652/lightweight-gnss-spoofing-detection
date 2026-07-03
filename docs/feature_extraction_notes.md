# 特征提取与 data_csv 生成说明

本文档记录当前真实多设备 GNSS 导航欺骗数据的特征提取流程，方便组内协作时快速了解本次工作内容、运行方式和注意事项。

## 当前数据结构

当前原始数据目录为：

```text
data_raw/
  playground/
  new_building/
```

每个环境下包含 6 类场景：

```text
st_L1
st_L5
st_L_15
dy_L1
dy_L5
dy_L_15
```

当前 `docs/data_manifest.csv` 记录了 133 个原始 GNSS txt 日志：

```text
playground: 98
new_building: 35
```

## 本次新增和修改内容

### `configs/preprocessing.yml`

新增本地预处理配置文件：

- 默认输入目录为 `./data_raw`；
- 记录设备名称映射；
- 记录当前使用的 TOW 欺骗标签区间；
- 定义最终 CSV 输出字段。

### `scripts/build_data_manifest.py`

更新数据清单生成脚本：

- 默认扫描当前仓库下的 `data_raw`；
- 支持 `data_raw/playground/...` 和 `data_raw/new_building/...` 两种环境结构；
- 输出 `docs/data_manifest.csv`。

### `scripts/build_mirrored_data_csv.py`

新增逐日志镜像 CSV 生成脚本：

- 扫描 `data_raw` 下的原始 GNSS txt；
- 每个 txt 生成一个对应 CSV；
- 输出目录为 `data_csv`；
- `data_csv` 的目录结构与 `data_raw` 保持一致。

例如：

```text
data_raw/new_building/dy_L1/.../log_mimir_20250729200504.txt
data_csv/new_building/dy_L1/.../log_mimir_20250729200504.csv
```

### `pipeline_total/04_build_labeled_processed_csv.py`

更新统一预处理脚本：

- 作为共享解析和特征工程逻辑；
- 从路径中推断 `Environment`、`Scenario`、`Session` 和 `DeviceName`；
- 支持生成单个总表 `processed_gnss_data.csv`；
- 支持输出缺失率报告。

### `pipeline_total/01_generate_plot_feature_csv.py`

更新逐日志画图特征 CSV 生成脚本：

- 修复原先引用不存在的 `pipeline/01_preprocess.py` 的问题；
- 复用统一预处理逻辑；
- 支持 `--data-root` 和 `--limit`，便于指定数据根目录和小规模测试。

### `pipeline_total/02_batch_plot_feature_images.py`

更新批量画图脚本：

- 支持 `--input-base`；
- 支持 `--output-base`；
- 支持 `--scenario`；
- 默认覆盖全部 6 类场景。

## 正式特征来源

本项目正式特征提取以原始 GNSS txt 日志为唯一正式数据源。

历史生成的 CSV 文件，例如：

```text
raw.csv
raw_sort.csv
plot_features.csv
features_enhanced.csv
```

可以作为参考或检查用，但不作为正式训练数据来源。

当前正式流程只解析原始 txt 中的 `Raw,` 行，并从中提取或计算轻量化检测所需特征。

## 当前核心特征

当前第一版模型输入特征为：

```text
Cn0DbHz
Cn0DbHz_dt
Cn0DbHz_std
AgcDb
ReceivedSvTimeUncertaintyNanos
PseudorangeRateUncertaintyMetersPerSecond
AccumulatedDeltaRangeUncertaintyMeters
FreqBand
```

其中：

- `Cn0DbHz_dt` 为同一颗卫星相邻时刻 C/N0 差分；
- `Cn0DbHz_std` 为同一颗卫星滑动窗口内 C/N0 标准差；
- `FreqBand` 由 `CarrierFrequencyHz` 判断得到。

除核心输入特征外，每个输出 CSV 还保留以下字段，用于排序、标注、分组和实验划分：

```text
TimeNanos
TOW
utcTimeMillis
Environment
Scenario
DeviceName
sv_id
SpoofingType
Label
```

注意：`TimeNanos`、`TOW` 和 `utcTimeMillis` 不建议直接作为模型输入特征。

## 常用命令

生成或更新数据清单：

```bash
python scripts/build_data_manifest.py
```

生成镜像逐日志 CSV：

```bash
python scripts/build_mirrored_data_csv.py --overwrite
```

只生成新主楼数据：

```bash
python scripts/build_mirrored_data_csv.py --environment new_building --overwrite
```

只生成某个场景：

```bash
python scripts/build_mirrored_data_csv.py --scenario dy_L1 --overwrite
```

如需生成单个集合版大 CSV：

```bash
python pipeline_total/04_build_labeled_processed_csv.py --mode full --config configs/preprocessing.yml
```

## 当前生成结果

最近一次全量镜像导出结果为：

```text
原始 txt 日志数量: 133
镜像 CSV 文件数量: 133
总行数: 3,175,866
输出目录: data_csv/
```

按环境划分：

```text
playground: 98
new_building: 35
```

## 不应提交到 Git 的内容

以下目录包含原始数据、本地生成数据或探索性数据，不应提交到 Git：

```text
data_raw/
data_csv/
output/
local/
```

这些目录已加入 `.gitignore`。

## 标签注意事项

当前 `Label` 由 `configs/preprocessing.yml` 中的 TOW 区间生成。

这些标签适合第一版流程跑通和数据检查，但正式实验前仍需要人工看图复核欺骗发生区间。复核后应更新：

```text
configs/preprocessing.yml
```

中的：

```text
labeling.spoofing_tow_intervals
```

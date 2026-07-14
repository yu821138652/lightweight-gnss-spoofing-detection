# 信号级 GNSS 特征提取流程

## 方案结论

项目当前规范化处理数据目录为 `data_csv/`。数据以**独立信号**为基本单位，
而不是仅以卫星为单位：`sv_id` 保留用于卫星级分析，`signal_id` 用于区分真正
独立的 GNSS 信号时间序列。

```text
signal_id = sv_id | SignalBand | CodeType
```

例如，同一颗北斗卫星可以在同一接收历元产生：

```text
C20|BDS_B1I|I
C20|BDS_B1C|Q
```

两者不能共用 C/N0 差分、拆分 CSV 文件或张量槽位。

## 数据约束

- `Cn0DbHz_dt` 和 `Cn0DbHz_std` 在同一 `signal_id` 内按 `TimeNanos` 计算。
- 对重复的 `signal_id + TimeNanos` 观测，预处理先执行确定性聚合；
  `SignalEpochCount` 保留聚合前的观测数量。
- `SignalBand` 由星座类型与载频在限定容差内映射得到；无法识别的信号保留为
  `UNKNOWN_*`，不会被静默归入错误频段。
- 未录入人工确认 Session 区间的新主楼数据使用 `LabelStatus=needs_review`，
  张量构建器默认排除这些数据。
- 张量默认最多使用 128 个独立信号槽位。超过容量时脚本会报错，绝不静默截断；
  可通过 `--max-signals` 开展 64/96/128 槽位的部署开销消融。

## 常用命令

全量重建逐日志信号级 CSV：

```powershell
python scripts/build_mirrored_data_csv.py --overwrite
```

按独立信号拆分，供可视化和标签复核使用：

```powershell
python scripts/split_csv_by_sv_id.py --group-column signal_id --sort-columns TOW TimeNanos --overwrite
```

仅使用已审查标签构建训练张量：

```powershell
python pipeline_total/05_build_train_val_test_tensors.py --csv output/processed_gnss_data.csv --output_dir output/tensor_data --max-signals 128
```

张量构建器只会在旧 CSV 缺少 `signal_id` 时回退到 `sv_id`。该回退模式仅用于
复现旧基线，不可用于正式实验结果。

## 当前验证结果

本次全量重建得到：

```text
源 CSV：133 个
源数据行：3,175,866 行
信号级拆分文件：7,044 个
拆分后总行数：3,175,866 行
```

审计结果：

```text
必需字段缺失文件：0
未知 SignalBand 行：0
重复信号历元行：0
包含多个 signal_id 的拆分文件：0
```

新主楼的欺骗 TOW 区间仍需按 `Environment + Scenario + Session` 人工复核后写入
`configs/preprocessing.yml` 的 `session_spoofing_tow_intervals.new_building`。在此之前，
这些数据保持 `needs_review`，不进入正式训练。

# 信号级 GNSS 特征提取流程

> 当前标签、数据快照和实验边界以 [handoff_status.md](handoff_status.md) 为准。本文只描述可重建的数据语义和脚本接口。

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

**何时运行：** 修改了解析、特征、标签配置，或所有新主楼 Session 都完成复核后。

**为什么运行：** 让全部 `data_csv/` 与当前代码和标签配置一致；它会覆盖派生 CSV，但不会修改原始 TXT。

全量重建逐日志信号级 CSV：

```powershell
python scripts/build_mirrored_data_csv.py --overwrite
```

**何时运行：** 全量 CSV 重建完成后，或需要按独立信号查看时序细节时。

**为什么运行：** 每个拆分文件只保留一个 `signal_id`，避免同一卫星的不同频率、不同 CodeType 在图中混成一条曲线。

按独立信号拆分，供可视化和标签复核使用：

```powershell
python scripts/split_csv_by_sv_id.py --group-column signal_id --sort-columns TOW TimeNanos --overwrite
```

**何时运行：** 标签审查完成、集合版 `processed_gnss_data.csv` 已生成，且准备开始训练或槽位容量消融时。

**为什么运行：** 把不定长的信号观测组织为模型输入；默认排除 `needs_review`，并确保超过槽位容量时显式报错而非丢失信号。

仅使用已审查标签构建训练张量：

```powershell
python pipeline_total/05_build_train_val_test_tensors.py --csv output/processed_gnss_data.csv --output-dir output/tensor_data --max-signals 128
```

张量构建器只会在旧 CSV 缺少 `signal_id` 时回退到 `sv_id`。该回退模式仅用于
复现旧基线，不可用于正式实验结果。

## 当前验证结果

本次全量重建得到：

```text
正式源日志 / 镜像 CSV / plot-feature CSV：均为 123 个（操场 89、新主楼 34）
镜像与中央 CSV：均为 2,998,458 行
标签分布：正类 631,003、负类 2,367,455
标签状态与来源：2,998,458 行全部为 `reviewed/session_config`
```

审计结果：

```text
必需字段缺失文件：0
未知 SignalBand 行：0
重复信号历元行：0
包含多个 signal_id 的拆分文件：0
```

两套环境均按 `Environment + Scenario + Session` 人工复核后写入
`configs/preprocessing.yml` 的 `session_spoofing_tow_intervals`。未显式配置的 Session
保持 `needs_review`，不进入正式训练。

## 新主楼协作标注流程

每名成员负责一个完整的 `Scenario + Session`，不要按设备拆分任务。欺骗标签应由
同一 Session 的多台设备共同确认，最终只写入一次 Session 级区间。

1. **何时运行：** 第一次接手新主楼标注任务，或预处理逻辑更新后。

   **为什么运行：** 从原始 TXT 生成包含 `SignalID`、C/N0、AGC 与不确定度的绘图中间表；没有这一步无法可靠地按独立信号查看异常。

   生成待标注 Session 的信号级绘图 CSV：

   ```powershell
   python pipeline_total/01_generate_plot_feature_csv.py --data-root data_raw/new_building --overwrite
   ```

2. **何时运行：** 需要对全部 Session 做一次系统人工标签审查，或修改标签配置后需要核对镜像 CSV 是否已经重建时。

   **为什么运行：** 每个完整 Session 生成一张统一面板：设备按列、标签时间轴和 7 个特征按行，红色阴影来自当前 Session 级正式标签；同时导出 HTML 索引和 CSV/config 一致性检查，避免动态图继续被旧的静态或场景级标签图误导。

   生成所有 Session 的审查包：

   ```powershell
   python pipeline_total/22_generate_label_review_dashboards.py `
     --input-dir data_csv `
     --output-dir output/label_review_dashboards
   ```

   浏览 `output/label_review_dashboards/index.html`。优先审查其中 `label_mismatch_rows > 0` 的 Session；图上红色背景是正式攻击时间，红色条表示受影响频段，黑色刻度是当前 CSV 已标正的观测。

3. **何时运行：** 已生成绘图 CSV，且需要放大检查某一个设备、特征或候选边界时。

   **为什么运行：** 将多设备、多信号的特征变化转为可比对的 PNG，作为人工 TOW 标签的直接证据。

   绘制该场景的信号时序图：

   ```powershell
   python pipeline_total/02_batch_plot_feature_images.py --input-base data_raw/new_building --output-base output/label_plots_20260723/new_building --scenario st_L1 --config configs/preprocessing.yml
   ```

4. 优先查看至少两台设备的 `Cn0DbHz`，并使用可用设备的 `AgcDb`、时间不确定度和
   伪距率不确定度交叉验证。Pixel Watch1 的 AGC 全缺失，不能单独作为 AGC 依据。

5. 记录所有设备共同出现的异常开始/结束秒。区间两端均为闭区间，格式为
   `[start_tow, end_tow]`。仅有单设备异常、日志中断或无法确定的片段不写入正式标签。

6. **何时运行：** 至少两台设备的变化共同支持同一候选区间，并完成第二人复核后。

   **为什么运行：** 标签配置是从原始 TXT 重新生成 `Label` 的唯一依据；只修改 CSV 会在下一次重建时丢失。

   将已确认结果写入 `configs/preprocessing.yml`：

   ```yaml
   labeling:
     session_spoofing_tow_intervals:
       new_building:
         st_L1:
           "2025.07.29.19.22_新主楼":
             status: reviewed
             intervals:
               - [start_tow, end_tow]
   ```

   已确认全程正常的 Session 使用 `status: reviewed` 与空 `intervals: []`。未完成复核的
   Session 不应写入配置，会自动保留为 `needs_review`。

7. **何时运行：** 第 6 步写入或修改该 Session 的正式区间后。

   **为什么运行：** 将新标签真正写入该 Session 全部设备的派生 CSV，并通过审计确认字段和标签状态正确；不必每次重跑全部 123 个日志。

   仅重建本人负责的 Session，并重新运行 CSV 审计和该 Session 面板：

   ```powershell
   python scripts/build_mirrored_data_csv.py --environment new_building --scenario st_L1 --session 2025.07.29.19.22_新主楼 --overwrite
   python scripts/audit_extracted_csv.py --input-dir data_csv --output-json output/data_csv_audit.json
   python pipeline_total/22_generate_label_review_dashboards.py --environment new_building --scenario st_L1 --session 2025.07.29.19.22_新主楼 --overwrite
   ```

8. 在协作记录中提交 `Scenario`、`Session`、候选区间、交叉验证设备、图像路径和复核人。
   未经第二人复核的区间只作为候选，不进入 `reviewed` 配置。

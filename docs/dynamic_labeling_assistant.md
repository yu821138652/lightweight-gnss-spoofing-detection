# 动态场景短时欺骗标签辅助

## 目的

新主楼动态场景中的欺骗片段可能只有数秒到几十秒。逐颗卫星曲线容易被运动、遮挡和单设备掉星淹没，因此不能只依赖单条 `Cn0DbHz` 曲线确定 TOW。本工具把同一秒、同一设备、同一频段的多条信号聚合，再寻找至少两个设备共同出现的异常。

它只生成**候选区间**，不自动修改 `configs/preprocessing.yml`，更不会把候选直接写入训练标签。

## 何时运行

完成 `data_csv/new_building/` 的信号级 CSV 重建、但某个 `dy_*` Session 的原始特征图看不出明确欺骗边界时运行。建议每位标注人员只处理自己负责的场景或 Session。

从仓库根目录运行全部动态 Session：

```powershell
python scripts/build_dynamic_labeling_assistant.py
```

运行 `dy_L1` 场景：

```powershell
python scripts/build_dynamic_labeling_assistant.py --scenario dy_L1
```

仅重新检查一个 Session：

```powershell
python scripts/build_dynamic_labeling_assistant.py --scenario dy_L_15 --session "2025.07.29.18.27_新主楼"
```

## 为什么这样做

- 每秒对每个 `DeviceName + FreqBand` 聚合，保留 C/N0 中位数、C/N0 变化率、滚动波动、AGC、三类 uncertainty 和可见 `signal_id` 数量。
- 每条设备频段序列用 31 秒滚动中位数与 MAD 做稳健标准化，并以该序列全局波动和各特征物理量级作为最小尺度，突出数秒级局部突变，同时避免量化字段的单次离散跳变被夸大。
- 只把至少两个独立设备同时有两类以上异常特征的秒标为候选。单设备变化更可能来自运动、遮挡或接收机自身问题。
- 默认最短候选区间为 3 秒，允许保留动态场景中的短时事件；相隔 1 秒以内的候选会被合并，方便人工查看。

默认阈值为 `2.5`，代表设备频段序列相对其局部稳健基线的异常程度。阈值不是欺骗判据，不能为了得到更多候选而直接降低后当作标签。

## 输出与判读

输出目录为 `output/dynamic_labeling_review/<Scenario>/<Session>/`：

- `overview.png`：四联总览图。红色阴影是候选段；第一图是每设备频段异常分数，第二图是同时存在强证据的设备数，后两图分别给出 C/N0 和可见信号数的局部标准化变化。
- `per_second_device_band_evidence.csv`：每秒、每设备、每频段的聚合数据和各特征局部 z 分数，用于追溯具体异常来源。
- `per_second_consensus.csv`：跨设备每秒共识结果。
- `candidate_intervals.csv`：当前 Session 的候选 TOW 区间。
- `output/dynamic_labeling_review/candidate_interval_summary.csv`：所有本次处理 Session 的候选汇总。

人工确认时先看 `overview.png`，再回到原始 7 项特征图检查候选段是否在目标频段出现跨设备同步变化。对于 `dy_L1` 优先核对 L1，`dy_L5` 优先核对 L5，`dy_L_15` 要检查双频共同证据。仅在多设备、目标频段和时间连续性都合理时，才把最终 TOW 区间填入 `configs/preprocessing.yml` 并重建对应 CSV。

## 常用参数

短事件被遗漏时，优先检查原始图和 `per_second_*` CSV，而不是立即调低阈值。确认候选确实受阈值影响后，才可以用较低阈值重新生成供人工查看：

```powershell
python scripts/build_dynamic_labeling_assistant.py --scenario dy_L5 --threshold 2.0
```

此命令的作用是增加待审查候选，原因是降低单设备局部异常门槛；它不提高标签可信度。正式标签仍必须由人工确认。

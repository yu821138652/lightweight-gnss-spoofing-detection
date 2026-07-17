# 操场静态+动态训练、静态测试 v1

该协议与 `static_within_playground_v1` 使用完全相同的静态 train/val/test Session：静态 test 保持为 `st_L1 / 2025.07.30.08.40_2025.07.30.09.12`。唯一改动是在训练集中加入 7 个操场动态 Session，覆盖 `dy_L1`、`dy_L5` 与 `dy_L_15`。

因此 `static-only` 与 `static+dynamic` 两组使用相同的设备级特征、窗口、模型、静态验证集、静态测试集和阈值；静态 test 指标差异可归因于动态训练数据的加入，但不应外推为动态 test 性能。

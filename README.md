# 面向真实多设备部署的轻量化 GNSS 导航欺骗检测方法研究

This repository is the clean project workspace for this GNSS research project.

## Project Mainline

**面向真实多设备部署的轻量化 GNSS 导航欺骗检测方法研究**

The project focuses on lightweight, deployable GNSS navigation spoofing detection using only GNSS Raw features available from real phones, watches, and u-blox receivers. The main experiments should validate cross-environment, cross-device, cross-frequency, and deployment-oriented performance.

## What Belongs Here

This repository should contain:

- pipeline scripts for preprocessing, plotting, labeling, tensor building, training, and inference;
- configuration templates;
- project planning documents;
- data inventory and labeling notes;
- small examples only, if needed later.

Large raw datasets are intentionally **not** copied into this repository.

## Main Data Sources

| Role | Local path | Use |
|---|---|---|
| Main data 1 | `H:\GNSS\data_raw` | Playground/campus spoofing data |
| Main data 2 | `H:\GNSS\导航欺骗新主楼数据集及全流程处理脚本\导航欺骗新主楼数据集及全流程处理脚本\0729` | New-main-building spoofing data |
| Auxiliary | `H:\GNSS\Finland L1_E1 data\final_mat` | Rich-feature software-receiver data / teacher / reference |
| Excluded for now | `H:\GNSS\Interference Data` | Label reliability uncertain; not part of the main paper line |

## Recommended Workflow

1. Generate `*-plot_features.csv` for both playground and new-main-building data.
2. Visualize `Cn0DbHz`, `AgcDb`, and uncertainty features.
3. Manually verify spoofing TOW intervals.
4. Build unified `processed_gnss_data.csv`.
5. Build train/validation/test tensors.
6. Train lightweight baselines first.
7. Run cross-environment, leave-one-device-out, and leave-one-frequency-out experiments.
8. Report deployment metrics: params, FLOPs/MACs, model size, CPU latency, memory, FAR, and TTD.

See `docs/project_mainline.md` for the detailed project plan.


# Data Inventory

This project uses large local datasets stored outside this repository.

## 1. Playground Spoofing Data

Path:

```text
H:\GNSS\data_raw
```

Summary:

- scenarios: `st_L1`, `st_L5`, `st_L_15`, `dy_L1`, `dy_L5`, `dy_L_15`
- raw TXT files: 98
- processed/intermediate CSV files already present: 768
- devices include HUAWEI, Xiaomi MI8, Redmi/K60-class Xiaomi, Pixel/Google, watch1, watch2, and u-blox-related files

Role:

- main dataset for real-device navigation spoofing detection;
- one side of cross-environment evaluation.

## 2. New Main Building Spoofing Data

Path:

```text
H:\GNSS\导航欺骗新主楼数据集及全流程处理脚本\导航欺骗新主楼数据集及全流程处理脚本\0729
```

Summary:

- scenarios: `st_L1`, `st_L5`, `st_L_15`, `dy_L1`, `dy_L5`, `dy_L_15`
- raw TXT files: 35
- CSV files: 45
- devices include HUAWEI Mate40, XiaoMi MI8, RedMi K60, Google Pixel Watch 1, and Google Pixel Watch 2

Role:

- second real environment;
- enables playground -> new-building and new-building -> playground tests.

## 3. Finland / FGI-GSRx Data

Path:

```text
H:\GNSS\Finland L1_E1 data\final_mat
```

Role:

- rich-feature software receiver data from prior work;
- useful as a strong baseline or teacher source;
- not the main deployment dataset.

## 4. Interference Data

Path:

```text
H:\GNSS\Interference Data
```

Current decision:

- not included in the main research line;
- labels are not trusted enough for main experiments;
- keep as a future extension or exploratory source only.

## Data Policy

Do not commit raw large data files to GitHub. Keep raw data local or upload it to a dedicated data storage service if sharing is required.


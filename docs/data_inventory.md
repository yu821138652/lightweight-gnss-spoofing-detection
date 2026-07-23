# Data Inventory

This project keeps large local datasets under the repository root. They are
excluded from GitHub through `.gitignore`.

## 1. Playground Spoofing Data

Path:

```text
H:\GNSS\lightweight_gnss_spoofing_detection\data_raw\playground
```

Summary:

- scenarios: `st_L1`, `st_L5`, `st_L_15`, `dy_L1`, `dy_L5`, `dy_L_15`
- active raw TXT files: 89
- extracted per-log CSV files are stored under `data_csv/playground`
- devices include HUAWEI, Xiaomi MI8, Redmi/K60-class Xiaomi, Pixel/Google, watch1, watch2, and u-blox-related files

Role:

- main dataset for real-device navigation spoofing detection;
- one side of cross-environment evaluation.

## 2. New Main Building Spoofing Data

Path:

```text
H:\GNSS\lightweight_gnss_spoofing_detection\data_raw\new_building
```

Summary:

- scenarios: `st_L1`, `st_L5`, `st_L_15`, `dy_L1`, `dy_L5`, `dy_L_15`
- active raw TXT files: 34
- extracted per-log CSV files are stored under `data_csv/new_building`
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

The current active corpus contains 123 raw logs. Nine playground logs were
deliberately removed from the active `data_raw` tree and are excluded from all
current rebuilds: three `dy_L5/2022.07.08` logs and six
`st_L5/2025.07.30.09.41_2025.07.30.09.45` logs. Historical experiments that
report 132 logs used the earlier corpus and are not directly comparable.

Do not commit raw large data files to GitHub. Keep raw data local or upload it to a dedicated data storage service if sharing is required.


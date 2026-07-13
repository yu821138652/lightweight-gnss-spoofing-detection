# Signal-Level Feature Extraction

## Decision

The canonical processed dataset is `data_csv/`. It is signal-level, not
satellite-level: `sv_id` remains available for satellite analyses, while
`signal_id` identifies an independently tracked GNSS signal.

```text
signal_id = sv_id | SignalBand | CodeType
```

For example, one BDS satellite can yield `C20|BDS_B1I|I` and
`C20|BDS_B1C|Q` at the same receiver epoch. They must not share a C/N0
derivative, a split CSV file, or a tensor slot.

## Data Invariants

- `Cn0DbHz_dt` and `Cn0DbHz_std` are computed within one `signal_id`, ordered
  by `TimeNanos`.
- Repeated `signal_id + TimeNanos` observations are collapsed before temporal
  features; `SignalEpochCount` preserves the original multiplicity.
- `SignalBand` is derived from constellation type and carrier frequency with a
  bounded frequency tolerance. Unknown signals remain visible as `UNKNOWN_*`.
- New-building rows without reviewed session intervals use
  `LabelStatus=needs_review` and are excluded by default from tensor builds.
- Signal tensors use up to 128 slots by default. Overflow raises an error
  instead of silently truncating observations. Use `--max-signals` for
  deployment-capacity ablations.

## Commands

Rebuild the mirrored signal-level CSV files:

```powershell
python scripts/build_mirrored_data_csv.py --overwrite
```

Split each source file by independent signal for plotting and label review:

```powershell
python scripts/split_csv_by_sv_id.py --group-column signal_id --sort-columns TOW TimeNanos --overwrite
```

Build training tensors from reviewed labels only:

```powershell
python pipeline_total/05_build_train_val_test_tensors.py --csv output/processed_gnss_data.csv --output_dir output/tensor_data --max-signals 128
```

The tensor builder falls back to `sv_id` only for an explicit legacy CSV that
does not contain `signal_id`. Do not use that fallback for formal results.

## Validation Snapshot

The current full rebuild contains 133 source CSV files and 3,175,866 rows.
It produced 7,044 `_by_signal_id` files with the same total row count. The
audit found no missing required fields, no unknown signal bands, and no
duplicate signal epochs in the current data.

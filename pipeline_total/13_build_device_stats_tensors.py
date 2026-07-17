"""Build causal device-level GNSS statistic windows from the processed CSV.

One row in the output represents one device at one current epoch. The device
label is positive when any valid signal at that epoch is labelled positive.
Split assignments must come from an explicit recording-level manifest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RECORDING_COLUMNS = ["Environment", "Scenario", "Session"]
CONTINUOUS_FEATURES = [
    "Cn0DbHz",
    "Cn0DbHz_dt",
    "Cn0DbHz_std",
    "AgcDb",
    "ReceivedSvTimeUncertaintyNanos",
    "PseudorangeRateUncertaintyMetersPerSecond",
]
STATISTICS = ["median", "std", "p10", "p90"]


def load_manifest(manifest_path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path, encoding="utf-8-sig")
    required = {*RECORDING_COLUMNS, "split"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"Split manifest is missing columns: {sorted(missing)}")
    if manifest.duplicated(RECORDING_COLUMNS).any():
        raise ValueError("Split manifest contains duplicate recording identities.")
    allowed_splits = {"train", "val", "test"}
    unknown_splits = set(manifest["split"].astype(str)).difference(allowed_splits)
    if unknown_splits:
        raise ValueError(f"Split manifest contains unsupported split names: {sorted(unknown_splits)}")
    missing_splits = allowed_splits.difference(set(manifest["split"].astype(str)))
    if missing_splits:
        raise ValueError(f"Split manifest must include train, val and test: missing {sorted(missing_splits)}")
    if "recording_id" not in manifest.columns:
        manifest["recording_id"] = pd.factorize(
            pd.MultiIndex.from_frame(manifest[RECORDING_COLUMNS].astype(str)), sort=True
        )[0].astype(np.int32)
    return manifest[["recording_id", *RECORDING_COLUMNS, "split"]].copy()


def aggregate_device_epochs(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    required = {"TimeNanos", "DeviceName", "FreqBand", "Label", *CONTINUOUS_FEATURES}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Processed CSV is missing device-statistics columns: {sorted(missing)}")

    if "SourceRelativePath" in frame.columns and frame["SourceRelativePath"].notna().all():
        frame["sequence_source"] = frame["SourceRelativePath"].astype(str)
    elif "SourceFile" in frame.columns:
        frame["sequence_source"] = frame["DeviceName"].astype(str) + "|" + frame["SourceFile"].astype(str)
    else:
        frame["sequence_source"] = frame["DeviceName"].astype(str)

    group_columns = ["recording_id", "split", "sequence_source", "TimeNanos"]
    frame["is_l1"] = np.isclose(pd.to_numeric(frame["FreqBand"], errors="coerce"), 1.0).astype(np.float32)
    frame["is_l5"] = np.isclose(pd.to_numeric(frame["FreqBand"], errors="coerce"), 5.0).astype(np.float32)
    grouped = frame.groupby(group_columns, sort=False, dropna=False)

    result = grouped.agg(
        device_label=("Label", "max"),
        valid_signal_count=("Label", "size"),
        l1_signal_ratio=("is_l1", "mean"),
        l5_signal_ratio=("is_l5", "mean"),
        Environment=("Environment", "first"),
        Scenario=("Scenario", "first"),
        Session=("Session", "first"),
        DeviceName=("DeviceName", "first"),
    ).reset_index()
    if "SourceRelativePath" in frame.columns:
        source_lookup = grouped["SourceRelativePath"].first().rename("SourceRelativePath").reset_index()
        result = result.merge(source_lookup, on=group_columns, how="left", validate="one_to_one")
    if "TOW" in frame.columns:
        tow_lookup = grouped["TOW"].median().rename("endpoint_TOW").reset_index()
        result = result.merge(tow_lookup, on=group_columns, how="left", validate="one_to_one")

    feature_columns: list[str] = []
    for feature in CONTINUOUS_FEATURES:
        values = grouped[feature]
        feature_frame = pd.DataFrame(
            {
                f"{feature}_median": values.median(),
                f"{feature}_std": values.std().fillna(0.0),
                f"{feature}_p10": values.quantile(0.10),
                f"{feature}_p90": values.quantile(0.90),
            }
        ).reset_index()
        result = result.merge(feature_frame, on=group_columns, how="left", validate="one_to_one")
        feature_columns.extend(f"{feature}_{statistic}" for statistic in STATISTICS)

    feature_columns.extend(["valid_signal_count", "l1_signal_ratio", "l5_signal_ratio"])
    result[feature_columns] = result[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    result["device_label"] = result["device_label"].astype(np.int64)
    result["sequence_id"] = pd.factorize(
        pd.MultiIndex.from_arrays([result["recording_id"], result["sequence_source"]]), sort=True
    )[0].astype(np.int32)
    return result, feature_columns


def normalize_from_train(frame: pd.DataFrame, feature_columns: list[str], output_dir: Path) -> pd.DataFrame:
    train = frame.loc[frame["split"] == "train", feature_columns]
    if train.empty:
        raise ValueError("No train device epochs are available for normalization.")
    means = train.mean()
    stds = train.std().replace(0.0, 1.0).fillna(1.0)
    normalized = frame.copy()
    normalized[feature_columns] = (normalized[feature_columns] - means) / stds
    normalized[feature_columns] = normalized[feature_columns].replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(np.float32)
    scaler = {
        "normalization": "global_train_device_epoch_statistics",
        "features": {column: {"mean": float(means[column]), "std": float(stds[column])} for column in feature_columns},
    }
    (output_dir / "device_stats_scaler.json").write_text(json.dumps(scaler, indent=2, ensure_ascii=False), encoding="utf-8")
    return normalized


def build_split_windows(
    frame: pd.DataFrame, split: str, feature_columns: list[str], time_steps: int, output_dir: Path
) -> dict[str, int]:
    subset = frame.loc[frame["split"] == split].copy()
    windows: list[np.ndarray] = []
    labels: list[int] = []
    records: list[dict] = []
    sample_index = 0
    metadata_columns = [
        column
        for column in [
            "Environment",
            "Scenario",
            "Session",
            "DeviceName",
            "SourceRelativePath",
            "sequence_source",
            "endpoint_TOW",
            "TimeNanos",
        ]
        if column in subset.columns
    ]
    for _, sequence in subset.groupby("sequence_id", sort=True):
        sequence = sequence.sort_values("TimeNanos", kind="mergesort")
        if len(sequence) < time_steps:
            continue
        values = sequence[feature_columns].to_numpy(dtype=np.float32)
        labels_array = sequence["device_label"].to_numpy(dtype=np.int64)
        for endpoint_index in range(time_steps - 1, len(sequence)):
            windows.append(values[endpoint_index - time_steps + 1 : endpoint_index + 1])
            labels.append(int(labels_array[endpoint_index]))
            endpoint = sequence.iloc[endpoint_index]
            record = {"sample_index": sample_index, "device_true_label": int(labels_array[endpoint_index])}
            for column in metadata_columns:
                if column == "TimeNanos":
                    record["endpoint_TimeNanos"] = int(endpoint[column])
                else:
                    record[column] = endpoint[column]
            records.append(record)
            sample_index += 1

    x = np.stack(windows).astype(np.float32) if windows else np.empty((0, time_steps, len(feature_columns)), dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    np.savez_compressed(output_dir / f"{split}.npz", x=x, y=y)
    pd.DataFrame.from_records(records).to_csv(output_dir / f"{split}_metadata.csv", index=False, encoding="utf-8-sig")
    return {"windows": int(len(x)), "positive_windows": int(y.sum()) if len(y) else 0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=PROJECT_ROOT / "output" / "processed_gnss_data.csv")
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scenario", choices=["static", "dynamic", "mixed"], default="static")
    parser.add_argument("--time-steps", type=int, default=5)
    args = parser.parse_args()
    if args.time_steps < 1:
        parser.error("--time-steps must be positive")
    for path in (args.csv, args.split_manifest):
        if not path.exists():
            raise FileNotFoundError(path)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(args.split_manifest)
    frame = pd.read_csv(args.csv)
    if "Label" not in frame.columns:
        raise ValueError("Processed CSV has no Label column.")
    frame.loc[frame["Label"] > 0, "Label"] = 1
    if "LabelStatus" in frame.columns:
        frame = frame.loc[frame["LabelStatus"].astype(str) == "reviewed"].copy()
    if args.scenario == "static":
        frame = frame.loc[frame["Scenario"].astype(str).str.startswith("st_")].copy()
    elif args.scenario == "dynamic":
        frame = frame.loc[frame["Scenario"].astype(str).str.startswith("dy_")].copy()

    frame = frame.merge(manifest, on=RECORDING_COLUMNS, how="inner", validate="many_to_one")
    matched = frame[RECORDING_COLUMNS].drop_duplicates()
    if len(matched) != len(manifest):
        missing = manifest.merge(matched, on=RECORDING_COLUMNS, how="left", indicator=True)
        preview = missing.loc[missing["_merge"] == "left_only", RECORDING_COLUMNS].head(5).to_dict("records")
        raise ValueError(f"Manifest recordings absent after review/scenario filtering: {preview}")

    print(f"Aggregating {len(frame):,} reviewed signal rows into device epochs...", flush=True)
    device_epochs, feature_columns = aggregate_device_epochs(frame)
    normalized = normalize_from_train(device_epochs, feature_columns, args.output_dir)
    manifest.to_csv(args.output_dir / "recording_split_manifest.csv", index=False, encoding="utf-8-sig")
    (args.output_dir / "device_feature_columns.json").write_text(
        json.dumps({"features": feature_columns, "time_steps": args.time_steps}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    summary = {"device_epoch_count": int(len(normalized)), "feature_count": len(feature_columns), "time_steps": args.time_steps}
    for split in ("train", "val", "test"):
        summary[split] = build_split_windows(normalized, split, feature_columns, args.time_steps, args.output_dir)
    (args.output_dir / "device_tensor_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

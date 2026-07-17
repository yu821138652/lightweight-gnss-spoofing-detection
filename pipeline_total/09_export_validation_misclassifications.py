"""Export validation-set signal-level misclassifications with source CSV context.

The training NPZ deliberately stores only tensors, labels and masks. This tool
recreates the recorded validation-window order from the processed CSV and the
locked recording manifest, verifies the reconstructed masks and labels against
the NPZ, then exports false positives and false negatives with their source
recording context. It never reads ``test.npz``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TIME_STEPS = 5
MAX_SIGNALS = 128


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def prepare_validation_frame(
    csv_path: Path,
    manifest_path: Path,
    tensor_builder,
    include_features: bool = True,
    split: str = "val",
):
    manifest = pd.read_csv(manifest_path)
    required_manifest_columns = {"recording_id", "Environment", "Scenario", "Session", "split"}
    missing_manifest_columns = required_manifest_columns.difference(manifest.columns)
    if missing_manifest_columns:
        raise ValueError(f"Split manifest is missing columns: {sorted(missing_manifest_columns)}")
    join_columns = ["Environment", "Scenario", "Session"]

    # The unified CSV can exceed available RAM. This function only rebuilds
    # validation metadata, so retain matching validation recordings while
    # streaming the source file instead of loading the full dataset first.
    target_manifest = manifest.loc[manifest["split"] == split, ["recording_id", *join_columns, "split"]].copy()
    if target_manifest.empty:
        raise ValueError(f"Split manifest has no {split!r} recordings.")
    target_keys = {
        tuple(row)
        for row in target_manifest[join_columns].astype(str).itertuples(index=False, name=None)
    }
    available_columns = pd.read_csv(csv_path, nrows=0).columns.tolist()
    requested_columns = {
        *join_columns,
        "Label",
        "LabelStatus",
        "TimeNanos",
        "DeviceName",
        "SourceRelativePath",
        "SourceFile",
        "signal_id",
        "sv_id",
        "TOW",
        "utcTimeMillis",
        "SpoofingType",
        "SignalBand",
        "ConstellationType",
        "Svid",
    }
    if include_features:
        requested_columns.update(tensor_builder.FEATURE_COLS)
    use_columns = [column for column in available_columns if column in requested_columns]
    required_raw_columns = {"Label", "TimeNanos", "DeviceName", *join_columns}
    missing_raw_columns = required_raw_columns.difference(use_columns)
    if missing_raw_columns:
        raise ValueError(f"Processed CSV is missing required columns: {sorted(missing_raw_columns)}")
    validation_chunks: list[pd.DataFrame] = []
    for chunk_index, chunk in enumerate(pd.read_csv(csv_path, usecols=use_columns, chunksize=20_000), start=1):
        chunk_keys = pd.MultiIndex.from_frame(chunk[join_columns].astype(str))
        matched = np.fromiter((key in target_keys for key in chunk_keys), dtype=bool, count=len(chunk))
        if matched.any():
            validation_chunks.append(chunk.loc[matched].copy())
        if chunk_index % 20 == 0:
            print(f"  scanned {chunk_index * 20_000:,} CSV rows...", flush=True)
    if not validation_chunks:
        raise ValueError(f"No processed CSV rows match {split!r} recordings in the locked split manifest.")
    frame = pd.concat(validation_chunks, ignore_index=True)
    if "Label" not in frame.columns:
        raise ValueError("Processed CSV has no Label column")
    frame.loc[frame["Label"] > 0, "Label"] = 1
    if "LabelStatus" in frame.columns:
        frame = frame[frame["LabelStatus"].astype(str) == "reviewed"].copy()
    if include_features:
        frame = tensor_builder.preprocess_features(frame)
    identity_column = tensor_builder.resolve_identity_column(frame)
    frame = frame.merge(
        target_manifest,
        on=join_columns,
        how="inner",
        validate="many_to_one",
    )
    if frame.empty:
        raise ValueError("No processed CSV rows match the locked split manifest.")

    if "SourceRelativePath" in frame.columns and frame["SourceRelativePath"].notna().all():
        sequence_source = frame["SourceRelativePath"].astype(str)
    elif "SourceFile" in frame.columns:
        sequence_source = frame["DeviceName"].astype(str) + "|" + frame["SourceFile"].astype(str)
    else:
        sequence_source = frame["DeviceName"].astype(str)
    sequence_index = pd.MultiIndex.from_arrays([frame["recording_id"], sequence_source])
    session_id, _ = pd.factorize(sequence_index, sort=True)
    frame["session_id"] = session_id.astype(np.int32)
    return frame, identity_column


def reconstruct_validation_metadata(
    frame: pd.DataFrame, identity_column: str, feature_columns: list[str], split: str = "val"
):
    metadata_columns = [
        "Environment",
        "Scenario",
        "Session",
        "DeviceName",
        "SourceRelativePath",
        "TOW",
        "utcTimeMillis",
        "SpoofingType",
        "SignalBand",
        "ConstellationType",
        "Svid",
        "sv_id",
    ]
    metadata_columns = [column for column in metadata_columns if column in frame.columns and column not in feature_columns]
    available_feature_columns = [column for column in feature_columns if column in frame.columns]
    keep_columns = ["session_id", "TimeNanos", identity_column, "Label", *available_feature_columns, *metadata_columns]
    validation = frame.loc[frame["split"] == split, keep_columns].copy()
    aggregation = {"Label": "max", **{column: "median" for column in available_feature_columns}}
    aggregation.update({column: "first" for column in metadata_columns})
    validation = (
        validation.groupby(["session_id", "TimeNanos", identity_column], as_index=False, sort=False)
        .agg(aggregation)
        .sort_values(["session_id", "TimeNanos", identity_column], kind="mergesort")
    )

    expected_masks: list[np.ndarray] = []
    expected_labels: list[np.ndarray] = []
    records: list[dict] = []
    sample_index = 0
    for session_id, session in validation.groupby("session_id", sort=True):
        session = session.sort_values(["TimeNanos", identity_column], kind="mergesort")
        unique_times = session["TimeNanos"].unique()
        if len(unique_times) < TIME_STEPS:
            continue
        time_to_tow = session.groupby("TimeNanos", sort=False)["TOW"].first().to_dict() if "TOW" in session else {}
        for start_index in range(len(unique_times) - TIME_STEPS + 1):
            window_times = unique_times[start_index : start_index + TIME_STEPS]
            endpoint_time = window_times[-1]
            endpoint = session.loc[session["TimeNanos"] == endpoint_time].sort_values(identity_column, kind="mergesort")
            identities = endpoint[identity_column].to_numpy()
            if len(identities) > MAX_SIGNALS:
                raise ValueError(f"Sample {sample_index} contains {len(identities)} signals, exceeding {MAX_SIGNALS}")

            mask = np.zeros(MAX_SIGNALS, dtype=bool)
            labels = np.full(MAX_SIGNALS, -100, dtype=np.int64)
            for signal_slot, (_, row) in enumerate(endpoint.iterrows()):
                mask[signal_slot] = True
                labels[signal_slot] = int(row["Label"])
                record = {
                    "sample_index": sample_index,
                    "window_start_TimeNanos": int(window_times[0]),
                    "endpoint_TimeNanos": int(endpoint_time),
                    "window_start_TOW": time_to_tow.get(window_times[0]),
                    "endpoint_TOW": row.get("TOW"),
                    "signal_id": row[identity_column],
                    "true_label": int(row["Label"]),
                }
                for column in metadata_columns:
                    if column != "TOW":
                        record[column] = row.get(column)
                for column in available_feature_columns:
                    record[column] = row[column]
                records.append(record)
            expected_masks.append(mask)
            expected_labels.append(labels)
            sample_index += 1

    if not expected_masks:
        raise ValueError("No validation windows could be reconstructed")
    return np.stack(expected_masks), np.stack(expected_labels), pd.DataFrame.from_records(records)


def predict_validation(training_module, checkpoint_path: Path, npz_path: Path, batch_size: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = training_module.build_model(
        checkpoint["model"],
        int(checkpoint["input_dim"]),
        int(checkpoint["time_steps"]),
        int(checkpoint["hidden_dim"]),
        float(checkpoint["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    dataset = training_module.GNSSWindowDataset(npz_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    probabilities: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for x, mask, _ in loader:
            logits = model(x.to(device))
            spoofing_probability = torch.softmax(logits, dim=-1)[..., 1].cpu().numpy()
            predicted_label = logits.argmax(dim=-1).cpu().numpy()
            valid = mask.numpy()
            probabilities.append(spoofing_probability[valid])
            predictions.append(predicted_label[valid])
    return checkpoint, np.concatenate(probabilities), np.concatenate(predictions), dataset


def build_error_breakdown(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    """Summarize signal-level confusion counts and rates for one review dimension."""
    grouped = (
        frame.groupby(group_columns, dropna=False)
        .agg(
            valid_signal_count=("true_label", "size"),
            positive_count=("true_label", "sum"),
            predicted_positive_count=("predicted_label", "sum"),
            true_positive=("is_true_positive", "sum"),
            true_negative=("is_true_negative", "sum"),
            false_positive=("is_false_positive", "sum"),
            false_negative=("is_false_negative", "sum"),
        )
        .reset_index()
    )
    grouped["negative_count"] = grouped["valid_signal_count"] - grouped["positive_count"]
    grouped["recall"] = grouped["true_positive"] / grouped["positive_count"].replace(0, np.nan)
    grouped["miss_rate"] = grouped["false_negative"] / grouped["positive_count"].replace(0, np.nan)
    grouped["far"] = grouped["false_positive"] / grouped["negative_count"].replace(0, np.nan)
    grouped["error_rate"] = (grouped["false_positive"] + grouped["false_negative"]) / grouped["valid_signal_count"]
    return grouped.sort_values(["false_negative", "false_positive"], ascending=False, kind="mergesort")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "output" / "tensors_mixed")
    parser.add_argument("--csv", type=Path, default=PROJECT_ROOT / "output" / "processed_gnss_data.csv")
    parser.add_argument("--model-dir", type=Path, required=True, help="Directory containing best_<model>.pt")
    parser.add_argument("--model", required=True, help="Checkpoint model name, for example signal_lstm")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    checkpoint_path = args.checkpoint or args.model_dir / f"best_{args.model}.pt"
    npz_path = args.data_dir / f"{args.split}.npz"
    manifest_path = args.data_dir / "recording_split_manifest.csv"
    for path in (args.csv, checkpoint_path, npz_path, manifest_path):
        if not path.exists():
            raise FileNotFoundError(path)
    output_csv = args.output_csv or args.model_dir / f"{args.split}_misclassifications_{args.model}.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    tensor_builder = load_module("tensor_builder_for_error_analysis", PROJECT_ROOT / "pipeline_total" / "05_build_train_val_test_tensors.py")
    training_module = load_module("training_module_for_error_analysis", PROJECT_ROOT / "pipeline_total" / "07_train_models.py")
    frame, identity_column = prepare_validation_frame(
        args.csv, manifest_path, tensor_builder, split=args.split
    )
    expected_mask, expected_y, metadata = reconstruct_validation_metadata(
        frame, identity_column, tensor_builder.FEATURE_COLS, split=args.split
    )
    checkpoint, probabilities, predicted_labels, dataset = predict_validation(
        training_module, checkpoint_path, npz_path, args.batch_size
    )

    if expected_mask.shape != tuple(dataset.mask.shape) or not np.array_equal(expected_mask, dataset.mask.numpy()):
        raise RuntimeError(f"Reconstructed {args.split} mask differs from {npz_path.name}; refusing to export misaligned rows.")
    if expected_y.shape != tuple(dataset.y.shape) or not np.array_equal(expected_y, dataset.y.numpy()):
        raise RuntimeError(f"Reconstructed {args.split} labels differ from {npz_path.name}; refusing to export misaligned rows.")
    if len(metadata) != len(probabilities):
        raise RuntimeError(f"Metadata rows ({len(metadata)}) do not match valid predictions ({len(probabilities)}).")

    metadata["predicted_label"] = predicted_labels.astype(int)
    metadata["spoofing_probability"] = probabilities
    metadata["is_true_positive"] = ((metadata["true_label"] == 1) & (metadata["predicted_label"] == 1)).astype(int)
    metadata["is_true_negative"] = ((metadata["true_label"] == 0) & (metadata["predicted_label"] == 0)).astype(int)
    metadata["is_false_positive"] = ((metadata["true_label"] == 0) & (metadata["predicted_label"] == 1)).astype(int)
    metadata["is_false_negative"] = ((metadata["true_label"] == 1) & (metadata["predicted_label"] == 0)).astype(int)
    metadata["error_type"] = np.where(
        metadata["is_false_negative"].astype(bool),
        "false_negative",
        "false_positive",
    )
    errors = metadata.loc[metadata["true_label"] != metadata["predicted_label"]].copy()
    errors = errors.sort_values(["error_type", "Environment", "Scenario", "Session", "DeviceName", "endpoint_TimeNanos", "signal_id"])
    errors.to_csv(output_csv, index=False, encoding="utf-8-sig")

    summary = pd.DataFrame(
        [
            {"item": "checkpoint", "value": str(checkpoint_path)},
            {"item": "model", "value": checkpoint["model"]},
            {"item": "split", "value": args.split},
            {"item": "valid_signal_predictions", "value": len(metadata)},
            {"item": "false_positive", "value": int((errors["error_type"] == "false_positive").sum())},
            {"item": "false_negative", "value": int((errors["error_type"] == "false_negative").sum())},
            {"item": "misclassification_total", "value": len(errors)},
        ]
    )
    summary_path = output_csv.with_name(f"{output_csv.stem}_summary.csv")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    recording_columns = [column for column in ["Environment", "Scenario", "Session"] if column in metadata.columns]
    recording_breakdown = build_error_breakdown(metadata, recording_columns)
    recording_path = output_csv.with_name(f"{output_csv.stem}_by_recording.csv")
    recording_breakdown.to_csv(recording_path, index=False, encoding="utf-8-sig")

    source_columns = [
        column
        for column in ["Environment", "Scenario", "Session", "DeviceName", "SourceRelativePath"]
        if column in metadata.columns
    ]
    source_breakdown = build_error_breakdown(metadata, source_columns)
    source_path = output_csv.with_name(f"{output_csv.stem}_by_source_log.csv")
    source_breakdown.to_csv(source_path, index=False, encoding="utf-8-sig")

    band_columns = [column for column in ["Scenario", "SignalBand"] if column in metadata.columns]
    band_breakdown = build_error_breakdown(metadata, band_columns)
    band_path = output_csv.with_name(f"{output_csv.stem}_by_signal_band.csv")
    band_breakdown.to_csv(band_path, index=False, encoding="utf-8-sig")

    metadata["endpoint_TOW_second"] = pd.to_numeric(metadata["endpoint_TOW"], errors="coerce").round().astype("Int64")
    tow_columns = [
        column
        for column in ["Environment", "Scenario", "Session", "DeviceName", "endpoint_TOW_second"]
        if column in metadata.columns
    ]
    tow_breakdown = build_error_breakdown(metadata, tow_columns)
    tow_path = output_csv.with_name(f"{output_csv.stem}_by_tow.csv")
    tow_breakdown.to_csv(tow_path, index=False, encoding="utf-8-sig")
    print(
        json.dumps(
            {
                "output_csv": str(output_csv),
                "summary_csv": str(summary_path),
                "recording_breakdown_csv": str(recording_path),
                "source_log_breakdown_csv": str(source_path),
                "signal_band_breakdown_csv": str(band_path),
                "tow_breakdown_csv": str(tow_path),
                **dict(summary.values),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

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


def prepare_validation_frame(csv_path: Path, manifest_path: Path, tensor_builder):
    frame = pd.read_csv(csv_path)
    if "Label" not in frame.columns:
        raise ValueError("Processed CSV has no Label column")
    frame.loc[frame["Label"] > 0, "Label"] = 1
    if "LabelStatus" in frame.columns:
        frame = frame[frame["LabelStatus"].astype(str) == "reviewed"].copy()
    frame = tensor_builder.preprocess_features(frame)
    identity_column = tensor_builder.resolve_identity_column(frame)

    manifest = pd.read_csv(manifest_path)
    required_manifest_columns = {"recording_id", "Environment", "Scenario", "Session", "split"}
    missing_manifest_columns = required_manifest_columns.difference(manifest.columns)
    if missing_manifest_columns:
        raise ValueError(f"Split manifest is missing columns: {sorted(missing_manifest_columns)}")
    join_columns = ["Environment", "Scenario", "Session"]
    frame = frame.merge(
        manifest[["recording_id", *join_columns, "split"]],
        on=join_columns,
        how="left",
        validate="many_to_one",
    )
    if frame["split"].isna().any():
        examples = frame.loc[frame["split"].isna(), join_columns].drop_duplicates().head(5)
        raise ValueError(f"Rows are absent from the locked split manifest: {examples.to_dict('records')}")

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


def reconstruct_validation_metadata(frame: pd.DataFrame, identity_column: str, feature_columns: list[str]):
    metadata_columns = [
        "Environment",
        "Scenario",
        "Session",
        "DeviceName",
        "SourceRelativePath",
        "SourceFile",
        "TOW",
        "utcTimeMillis",
        "SpoofingType",
        "LabelStatus",
        "LabelSource",
        "SignalBand",
        "ConstellationType",
        "Svid",
        "sv_id",
    ]
    metadata_columns = [column for column in metadata_columns if column in frame.columns and column not in feature_columns]
    keep_columns = ["session_id", "TimeNanos", identity_column, "Label", *feature_columns, *metadata_columns]
    validation = frame.loc[frame["split"] == "val", keep_columns].copy()
    aggregation = {"Label": "max", **{column: "median" for column in feature_columns}}
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
                    "signal_slot": signal_slot,
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
                for column in feature_columns:
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "output" / "tensors_mixed")
    parser.add_argument("--csv", type=Path, default=PROJECT_ROOT / "output" / "processed_gnss_data.csv")
    parser.add_argument("--model-dir", type=Path, required=True, help="Directory containing best_<model>.pt")
    parser.add_argument("--model", required=True, help="Checkpoint model name, for example signal_lstm")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    checkpoint_path = args.checkpoint or args.model_dir / f"best_{args.model}.pt"
    npz_path = args.data_dir / "val.npz"
    manifest_path = args.data_dir / "recording_split_manifest.csv"
    for path in (args.csv, checkpoint_path, npz_path, manifest_path):
        if not path.exists():
            raise FileNotFoundError(path)
    output_csv = args.output_csv or args.model_dir / f"validation_misclassifications_{args.model}.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    tensor_builder = load_module("tensor_builder_for_error_analysis", PROJECT_ROOT / "pipeline_total" / "05_build_train_val_test_tensors.py")
    training_module = load_module("training_module_for_error_analysis", PROJECT_ROOT / "pipeline_total" / "07_train_models.py")
    frame, identity_column = prepare_validation_frame(args.csv, manifest_path, tensor_builder)
    expected_mask, expected_y, metadata = reconstruct_validation_metadata(
        frame, identity_column, tensor_builder.FEATURE_COLS
    )
    checkpoint, probabilities, predicted_labels, dataset = predict_validation(
        training_module, checkpoint_path, npz_path, args.batch_size
    )

    if expected_mask.shape != tuple(dataset.mask.shape) or not np.array_equal(expected_mask, dataset.mask.numpy()):
        raise RuntimeError("Reconstructed validation mask differs from val.npz; refusing to export misaligned rows.")
    if expected_y.shape != tuple(dataset.y.shape) or not np.array_equal(expected_y, dataset.y.numpy()):
        raise RuntimeError("Reconstructed validation labels differ from val.npz; refusing to export misaligned rows.")
    if len(metadata) != len(probabilities):
        raise RuntimeError(f"Metadata rows ({len(metadata)}) do not match valid predictions ({len(probabilities)}).")

    metadata["predicted_label"] = predicted_labels.astype(int)
    metadata["spoofing_probability"] = probabilities
    metadata["error_type"] = np.where(
        (metadata["true_label"] == 1) & (metadata["predicted_label"] == 0),
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
            {"item": "valid_signal_predictions", "value": len(metadata)},
            {"item": "false_positive", "value": int((errors["error_type"] == "false_positive").sum())},
            {"item": "false_negative", "value": int((errors["error_type"] == "false_negative").sum())},
            {"item": "misclassification_total", "value": len(errors)},
        ]
    )
    summary_path = output_csv.with_name(f"{output_csv.stem}_summary.csv")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(json.dumps({"output_csv": str(output_csv), "summary_csv": str(summary_path), **dict(summary.values)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

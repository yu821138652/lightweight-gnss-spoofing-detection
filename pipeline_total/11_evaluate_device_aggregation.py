"""Evaluate device-level alarms by aggregating validation signal predictions.

This script converts per-signal predictions at one current epoch into one
device alarm. For a single-band spoofing experiment, signals in unaffected
bands correctly retain label 0. Therefore device truth is positive when any
valid signal at the current epoch has true label 1. It defaults to validation
and does not read test.npz.
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
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


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
            probabilities.append(torch.softmax(logits, dim=-1)[..., 1].cpu().numpy()[mask.numpy()])
            predictions.append(logits.argmax(dim=-1).cpu().numpy()[mask.numpy()])
    return checkpoint, np.concatenate(probabilities), np.concatenate(predictions), dataset


def aggregate_device_predictions(frame: pd.DataFrame, rule: str, min_positive_signals: int, positive_ratio: float):
    group_columns = [
        "Environment",
        "Scenario",
        "Session",
        "DeviceName",
        "SourceRelativePath",
        "endpoint_TimeNanos",
        "endpoint_TOW",
    ]
    grouped = (
        frame.groupby(group_columns, dropna=False)
        .agg(
            true_label=("true_label", "max"),
            label_value_count=("true_label", "nunique"),
            valid_signal_count=("signal_id", "size"),
            true_positive_signal_count=("true_label", "sum"),
            predicted_positive_signal_count=("predicted_signal_label", "sum"),
            mean_spoofing_probability=("spoofing_probability", "mean"),
            max_spoofing_probability=("spoofing_probability", "max"),
        )
        .reset_index()
    )
    # A mixed true label is expected for st_L1/st_L5 attacks: unaffected bands
    # remain normal while the device is still in a spoofing event.
    grouped["mixed_signal_truth"] = (grouped["label_value_count"] > 1).astype(int)
    grouped["true_positive_signal_ratio"] = (
        grouped["true_positive_signal_count"] / grouped["valid_signal_count"]
    )
    grouped["predicted_positive_ratio"] = grouped["predicted_positive_signal_count"] / grouped["valid_signal_count"]
    if rule == "majority":
        grouped["device_predicted_label"] = (
            grouped["predicted_positive_signal_count"] > grouped["valid_signal_count"] / 2
        ).astype(int)
    elif rule == "any":
        grouped["device_predicted_label"] = (grouped["predicted_positive_signal_count"] >= 1).astype(int)
    elif rule == "k_of_n":
        grouped["device_predicted_label"] = (grouped["predicted_positive_signal_count"] >= min_positive_signals).astype(int)
    elif rule == "ratio":
        grouped["device_predicted_label"] = (grouped["predicted_positive_ratio"] >= positive_ratio).astype(int)
    else:
        raise ValueError(f"Unknown rule: {rule}")
    return grouped


def compute_metrics(device_epochs: pd.DataFrame) -> dict[str, float | int]:
    y_true = device_epochs["true_label"].to_numpy()
    y_pred = device_epochs["device_predicted_label"].to_numpy()
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return {
        "device_epoch_count": int(len(device_epochs)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "mixed_signal_truth_device_epochs": int(device_epochs["mixed_signal_truth"].sum()),
        "device_truth_rule": "any_true_positive_signal",
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "far": float(fp / (fp + tn)) if fp + tn else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "output" / "tensors_mixed")
    parser.add_argument("--csv", type=Path, default=PROJECT_ROOT / "output" / "processed_gnss_data.csv")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--rule", choices=["majority", "any", "k_of_n", "ratio"], default="majority")
    parser.add_argument("--min-positive-signals", type=int, default=2)
    parser.add_argument("--positive-ratio", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.min_positive_signals < 1:
        parser.error("--min-positive-signals must be at least 1")
    if not 0 < args.positive_ratio <= 1:
        parser.error("--positive-ratio must be in (0, 1]")

    checkpoint_path = args.model_dir / f"best_{args.model}.pt"
    validation_npz = args.data_dir / "val.npz"
    manifest_path = args.data_dir / "recording_split_manifest.csv"
    for path in (args.csv, checkpoint_path, validation_npz, manifest_path):
        if not path.exists():
            raise FileNotFoundError(path)

    error_export = load_module(
        "validation_error_export_for_device_metrics",
        PROJECT_ROOT / "pipeline_total" / "09_export_validation_misclassifications.py",
    )
    tensor_builder = load_module(
        "tensor_builder_for_device_metrics",
        PROJECT_ROOT / "pipeline_total" / "05_build_train_val_test_tensors.py",
    )
    training_module = load_module(
        "training_module_for_device_metrics",
        PROJECT_ROOT / "pipeline_total" / "07_train_models.py",
    )
    print("[1/5] Rebuilding validation metadata from processed CSV...", flush=True)
    frame, identity_column = error_export.prepare_validation_frame(
        args.csv, manifest_path, tensor_builder, include_features=False
    )
    print(f"[2/5] Filtered validation rows: {len(frame):,}; rebuilding window metadata...", flush=True)
    expected_mask, expected_y, metadata = error_export.reconstruct_validation_metadata(
        frame, identity_column, []
    )
    print(f"[3/5] Metadata ready: {len(metadata):,} valid signal windows; loading model...", flush=True)
    checkpoint, probabilities, predicted_labels, dataset = predict_validation(
        training_module, checkpoint_path, validation_npz, args.batch_size
    )
    if expected_mask.shape != tuple(dataset.mask.shape) or not np.array_equal(expected_mask, dataset.mask.numpy()):
        raise RuntimeError("Reconstructed validation mask differs from val.npz; refusing to aggregate misaligned rows.")
    if expected_y.shape != tuple(dataset.y.shape) or not np.array_equal(expected_y, dataset.y.numpy()):
        raise RuntimeError("Reconstructed validation labels differ from val.npz; refusing to aggregate misaligned rows.")
    if len(metadata) != len(probabilities):
        raise RuntimeError("Metadata rows do not match valid validation predictions.")

    print("[4/5] Aggregating signal predictions to device epochs...", flush=True)
    metadata["predicted_signal_label"] = predicted_labels.astype(int)
    metadata["spoofing_probability"] = probabilities
    device_epochs = aggregate_device_predictions(metadata, args.rule, args.min_positive_signals, args.positive_ratio)
    metrics = compute_metrics(device_epochs)
    metrics.update(
        {
            "model": checkpoint["model"],
            "rule": args.rule,
            "min_positive_signals": args.min_positive_signals,
            "positive_ratio": args.positive_ratio,
        }
    )

    output_dir = args.output_dir or args.model_dir / "device_level_validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{args.model}_{args.rule}"
    predictions_path = output_dir / f"device_epoch_predictions_{suffix}.csv"
    metrics_path = output_dir / f"device_metrics_{suffix}.json"
    device_epochs.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[5/5] Device-level files written.", flush=True)
    print(json.dumps({"predictions_csv": str(predictions_path), "metrics_json": str(metrics_path), **metrics}, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Evaluate a flat response-state model with a direct-spoof override expert."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "pipeline_total" / "37_train_device_attack_event.py"
STATE_NAMES = {0: "normal", 1: "anomaly", 2: "direct"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--flat-checkpoint", type=Path, required=True)
    parser.add_argument("--direct-checkpoint", type=Path, required=True)
    parser.add_argument("--direct-threshold", type=float, default=0.5)
    parser.add_argument("--calibrate-threshold-on-val", action="store_true")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    parser.add_argument("--max-val-far", type=float, default=0.05)
    parser.add_argument("--min-val-abnormal-recall", type=float, default=0.0)
    parser.add_argument("--override-scope", choices=("abnormal", "all"), default="abnormal")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--fold", type=str, default="", help="Optional outer-fold tag written to the prediction CSV.")
    parser.add_argument(
        "--predictions-csv", type=Path,
        help="Optional per-window export for downstream scene/response fusion.",
    )
    return parser.parse_args()


def load_train_module() -> Any:
    spec = importlib.util.spec_from_file_location("device_event_train", TRAIN_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {TRAIN_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def predict_override(
    flat_probability: np.ndarray,
    direct_probability: np.ndarray,
    direct_threshold: float,
    override_scope: str,
) -> tuple[np.ndarray, np.ndarray]:
    predicted = flat_probability.argmax(axis=1).astype(np.int64)
    override = direct_probability[:, 1] >= direct_threshold
    if override_scope == "abnormal":
        override &= predicted != 0
    predicted[override] = 2
    return predicted, override


def evaluate_override(
    train_mod: Any,
    data: Any,
    flat_probability: np.ndarray,
    direct_probability: np.ndarray,
    direct_threshold: float,
    override_scope: str,
    device_names: dict[int, str],
    metadata: dict[str, Any],
    split: str,
    flat_checkpoint: Path,
    direct_checkpoint: Path,
) -> dict[str, Any]:
    predicted, override = predict_override(flat_probability, direct_probability, direct_threshold, override_scope)
    labels = data.y.numpy()
    result: dict[str, Any] = {
        "split": split,
        "direct_threshold": direct_threshold,
        "override_scope": override_scope,
        "flat_checkpoint": str(flat_checkpoint),
        "direct_checkpoint": str(direct_checkpoint),
        "override_count": int(override.sum()),
        "overall": train_mod.multiclass_metrics(labels, predicted, 3),
    }
    device_ids = data.device_id.numpy()
    recording_ids = data.recording_id.numpy()
    scenario_key, scenario_names = train_mod.scenario_ids(metadata, recording_ids)
    result["by_device"] = train_mod.group_metrics(labels, predicted, device_ids, device_names, 3)
    result["by_scenario"] = train_mod.group_metrics(labels, predicted, scenario_key, scenario_names, 3)
    result["by_recording"] = train_mod.group_metrics(labels, predicted, recording_ids, train_mod.recording_names(metadata), 3)
    return result


def score_for_selection(result: dict[str, Any]) -> tuple[float, float, float]:
    overall = result["overall"]
    return (
        float(overall["macro_f1"]),
        float(overall["per_class"]["2"]["recall"]),
        float(overall["abnormal_recall"]),
    )


def export_predictions(
    output_path: Path,
    data: Any,
    flat_probability: np.ndarray,
    direct_probability: np.ndarray,
    predicted: np.ndarray,
    override: np.ndarray,
    device_names: dict[int, str],
    metadata: dict[str, Any],
    fold: str,
) -> None:
    """Write traceable response-state predictions for the late-fusion evaluator."""
    if len(data) != len(predicted):
        raise ValueError("Prediction count differs from response-state data")
    recording_names = train_mod_recording_names(metadata)
    fields = [
        "fold", "recording_id", "recording_name", "device_id", "device_name", "source_id", "endpoint_tow",
        "true_state", "true_state_name", "flat_pred_state", "flat_pred_state_name",
        "pred_state", "pred_state_name", "direct_probability", "direct_override",
        *[f"flat_prob_{STATE_NAMES[c]}" for c in range(3)],
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = data.y.numpy()
    device_ids = data.device_id.numpy()
    recording_ids = data.recording_id.numpy()
    source_ids = data.source_id.numpy()
    endpoint_tow = data.endpoint_tow.numpy()
    flat_predicted = flat_probability.argmax(axis=1).astype(np.int64)
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(len(labels)):
            device_id = int(device_ids[index])
            recording_id = int(recording_ids[index])
            true_state = int(labels[index])
            flat_state = int(flat_predicted[index])
            final_state = int(predicted[index])
            row = {
                "fold": fold,
                "recording_id": recording_id,
                "recording_name": recording_names.get(recording_id, str(recording_id)),
                "device_id": device_id,
                "device_name": device_names.get(device_id, str(device_id)),
                "source_id": int(source_ids[index]),
                "endpoint_tow": float(endpoint_tow[index]),
                "true_state": true_state,
                "true_state_name": STATE_NAMES[true_state],
                "flat_pred_state": flat_state,
                "flat_pred_state_name": STATE_NAMES[flat_state],
                "pred_state": final_state,
                "pred_state_name": STATE_NAMES[final_state],
                "direct_probability": float(direct_probability[index, 1]),
                "direct_override": bool(override[index]),
            }
            row.update({f"flat_prob_{STATE_NAMES[c]}": float(flat_probability[index, c]) for c in range(3)})
            writer.writerow(row)


def train_mod_recording_names(metadata: dict[str, Any]) -> dict[int, str]:
    """Local counterpart of the train helper, kept explicit for CSV provenance."""
    rows = metadata.get("recordings", [])
    if not isinstance(rows, list):
        return {}
    return {
        index: "/".join(str(row.get(key, "")) for key in ("Environment", "Scenario", "Session"))
        for index, row in enumerate(rows)
        if isinstance(row, dict)
    }


def main() -> None:
    args = parse_args()
    if not 0.0 < args.direct_threshold < 1.0:
        raise ValueError("direct-threshold must be strictly between zero and one")
    if any(not 0.0 < threshold < 1.0 for threshold in args.thresholds):
        raise ValueError("thresholds must be strictly between zero and one")
    if not 0.0 <= args.max_val_far <= 1.0:
        raise ValueError("max-val-far must be in [0, 1]")
    if not 0.0 <= args.min_val_abnormal_recall <= 1.0:
        raise ValueError("min-val-abnormal-recall must be in [0, 1]")
    train_mod = load_train_module()
    metadata = json.loads((args.data_dir / "metadata.json").read_text(encoding="utf-8"))
    device_names = {int(value): str(key) for key, value in metadata.get("device_mapping", {}).items()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = train_mod.EventDataset(args.data_dir / f"{args.split}.npz", "y_response_state", None, "raw")
    flat_checkpoint, flat_model = train_mod.load_checkpoint_model(args.flat_checkpoint, device, data.x.shape[1])
    direct_checkpoint, direct_model = train_mod.load_checkpoint_model(args.direct_checkpoint, device, data.x.shape[1])
    if int(flat_checkpoint.get("num_classes", 3)) != 3:
        raise ValueError("Flat checkpoint must be a three-class response-state model")
    if int(direct_checkpoint.get("num_classes", 2)) != 2:
        raise ValueError("Direct checkpoint must be binary")
    selected_threshold = args.direct_threshold
    calibration: dict[str, Any] | None = None
    if args.calibrate_threshold_on_val:
        val_data = train_mod.EventDataset(args.data_dir / "val.npz", "y_response_state", None, "raw")
        val_flat_probability = train_mod.probabilities(flat_model, val_data, device, args.batch_size)
        val_direct_probability = train_mod.probabilities(direct_model, val_data, device, args.batch_size)
        candidates = [
            evaluate_override(
                train_mod, val_data, val_flat_probability, val_direct_probability, threshold, args.override_scope,
                device_names, metadata, "val", args.flat_checkpoint, args.direct_checkpoint,
            )
            for threshold in args.thresholds
        ]
        eligible = [
            candidate for candidate in candidates
            if candidate["overall"]["far"] <= args.max_val_far
            and candidate["overall"]["abnormal_recall"] >= args.min_val_abnormal_recall
        ]
        selected = max(eligible or candidates, key=score_for_selection)
        selected_threshold = float(selected["direct_threshold"])
        calibration = {
            "split": "val",
            "max_val_far": args.max_val_far,
            "min_val_abnormal_recall": args.min_val_abnormal_recall,
            "selected_threshold": selected_threshold,
            "selected": selected,
            "candidates": candidates,
        }
    flat_probability = train_mod.probabilities(flat_model, data, device, args.batch_size)
    direct_probability = train_mod.probabilities(direct_model, data, device, args.batch_size)
    predicted, override = predict_override(
        flat_probability, direct_probability, selected_threshold, args.override_scope,
    )
    result = evaluate_override(
        train_mod, data, flat_probability, direct_probability, selected_threshold, args.override_scope,
        device_names, metadata, args.split, args.flat_checkpoint, args.direct_checkpoint,
    )
    if calibration is not None:
        result["calibration"] = {
            key: value for key, value in calibration.items()
            if key not in ("selected", "candidates")
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if calibration is not None:
        (args.output_dir / "direct_override_threshold_calibration.json").write_text(
            json.dumps(calibration, indent=2), encoding="utf-8"
        )
    if args.predictions_csv is not None:
        export_predictions(
            args.predictions_csv, data, flat_probability, direct_probability, predicted, override,
            device_names, metadata, args.fold,
        )
        result["predictions_csv"] = str(args.predictions_csv)
    output_path = args.output_dir / f"{args.split}_metrics_response_state_direct_override.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "metrics": str(output_path),
        "direct_threshold": result["direct_threshold"],
        "override_count": result["override_count"],
        "macro_f1": result["overall"]["macro_f1"],
        "far": result["overall"]["far"],
        "abnormal_recall": result["overall"]["abnormal_recall"],
        "anomaly_recall": result["overall"]["per_class"]["1"]["recall"],
        "direct_recall": result["overall"]["per_class"]["2"]["recall"],
    }, indent=2))


if __name__ == "__main__":
    main()

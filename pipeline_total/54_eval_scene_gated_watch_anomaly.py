#!/usr/bin/env python3
"""Apply a scene-gated single-frequency Watch anomaly expert.

The response backbone remains responsible for normal/direct/anomaly output.
This evaluator only repairs ``normal -> anomaly`` for named single-frequency
Watches when both conditions hold:

1. a *global* scene context, pooled from capable receivers at the same time,
   confidently indicates an L5 attack; and
2. a Watch-only suppression expert sees an L1 C/N0 / tracked-signal loss.

The L5 gate keeps an L1 direct spoof response from being re-labelled as an
indirect anomaly.  Threshold selection belongs to the expert's validation
split (``37_train_device_attack_event.py --calibrate-only``); this script
never consults outer-test labels to choose a threshold.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "pipeline_total" / "37_train_device_attack_event.py"
STATE_NAMES = {0: "normal", 1: "anomaly", 2: "direct"}
SCENE_NAMES = {0: "normal", 1: "L1", 2: "L5", 3: "L1+L5"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch-data-dir", type=Path, required=True)
    parser.add_argument("--expert-checkpoint", type=Path, required=True)
    parser.add_argument("--base-predictions", type=Path, required=True)
    parser.add_argument("--scene-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--watch-devices", nargs="+",
        default=["Google_Pixel_Watch1", "Google_Pixel_Watch2"],
        help="devices eligible for the single-frequency suppression expert",
    )
    parser.add_argument(
        "--scene-classes", type=int, nargs="+", choices=(1, 2, 3), default=[2],
        help="global scene classes that permit normal-to-anomaly correction; default is L5 only",
    )
    parser.add_argument("--min-scene-confidence", type=float, default=0.50)
    parser.add_argument("--threshold", type=float, help="override the calibrated expert alarm threshold")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--tow-decimals", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.min_scene_confidence <= 1:
        parser.error("--min-scene-confidence must be in [0, 1]")
    if args.threshold is not None and not 0 < args.threshold < 1:
        parser.error("--threshold must be strictly between zero and one")
    if args.batch_size < 1 or not 0 <= args.tow_decimals <= 9:
        parser.error("--batch-size must be positive and --tow-decimals must be in [0, 9]")
    return args


def load_train_module() -> Any:
    spec = importlib.util.spec_from_file_location("device_event_train", TRAIN_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {TRAIN_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metric_bundle(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    labels = [0, 1, 2]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0,
    )
    normal = y_true == 0
    abnormal = y_true > 0
    return {
        "samples": int(len(y_true)),
        "accuracy": float((y_true == y_pred).mean()) if len(y_true) else 0.0,
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "far": float((y_pred[normal] > 0).mean()) if normal.any() else 0.0,
        "abnormal_recall": float((y_pred[abnormal] > 0).mean()) if abnormal.any() else 0.0,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).astype(int).tolist(),
        "per_class": {
            STATE_NAMES[label]: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(labels)
        },
    }


def key_columns(frame: pd.DataFrame, decimals: int) -> pd.DataFrame:
    result = frame.copy()
    result["fold"] = result["fold"].astype(int)
    result["recording_id"] = result["recording_id"].astype(int)
    result["device_id"] = result["device_id"].astype(int)
    result["tow_key"] = result["endpoint_tow"].astype(float).round(decimals)
    return result


def require_columns(frame: pd.DataFrame, columns: set[str], path: Path) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns {missing}")


def global_scene_context(scene: pd.DataFrame, decimals: int) -> pd.DataFrame:
    """Average capable-device scene posteriors into recording-time context."""
    required = {"fold", "recording_id", "endpoint_tow", "prob_normal", "prob_L1", "prob_L5", "prob_L1+L5"}
    require_columns(scene, required, Path("scene predictions"))
    frame = scene.copy()
    frame["fold"] = frame["fold"].astype(int)
    frame["recording_id"] = frame["recording_id"].astype(int)
    frame["tow_key"] = frame["endpoint_tow"].astype(float).round(decimals)
    probability_columns = ["prob_normal", "prob_L1", "prob_L5", "prob_L1+L5"]
    context = frame.groupby(["fold", "recording_id", "tow_key"], as_index=False)[probability_columns].mean()
    probabilities = context[probability_columns].to_numpy(dtype=np.float64)
    context["global_scene_pred"] = probabilities.argmax(axis=1).astype(np.int64)
    context["global_scene_confidence"] = probabilities.max(axis=1)
    context["global_scene_device_count"] = frame.groupby(["fold", "recording_id", "tow_key"])["device_id"].nunique().to_numpy()
    return context


def expert_predictions(
    train_mod: Any,
    data_dir: Path,
    checkpoint_path: Path,
    watch_device_ids: set[int],
    device: torch.device,
    batch_size: int,
) -> tuple[pd.DataFrame, float]:
    data = train_mod.EventDataset(data_dir / "test.npz", "y_response_state", watch_device_ids, "anomaly_only")
    checkpoint, model = train_mod.load_checkpoint_model(checkpoint_path, device, data.x.shape[1])
    if int(checkpoint.get("num_classes", 2)) != 2:
        raise ValueError("Watch anomaly expert checkpoint must be binary")
    if checkpoint.get("label_transform") != "anomaly_only":
        raise ValueError("Watch anomaly expert must be trained with --label-transform anomaly_only")
    threshold = float(checkpoint.get("alarm_threshold", 0.5))
    probability = train_mod.probabilities(model, data, device, batch_size)[:, 1]
    return pd.DataFrame({
        "device_id": data.device_id.numpy().astype(int),
        "source_id": data.source_id.numpy().astype(int),
        "recording_id": data.recording_id.numpy().astype(int),
        "endpoint_tow": data.endpoint_tow.numpy().astype(float),
        "watch_anomaly_probability": probability.astype(float),
    }), threshold


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}; use --overwrite")
    for path in (args.watch_data_dir, args.expert_checkpoint, args.base_predictions, args.scene_predictions):
        if not path.exists():
            raise FileNotFoundError(path)

    metadata = json.loads((args.watch_data_dir / "metadata.json").read_text(encoding="utf-8"))
    device_mapping = {str(name): int(value) for name, value in metadata.get("device_mapping", {}).items()}
    unknown = sorted(set(args.watch_devices).difference(device_mapping))
    if unknown:
        raise ValueError(f"Unknown watch devices {unknown}; available={sorted(device_mapping)}")
    watch_ids = {device_mapping[name] for name in args.watch_devices}
    threshold_override = args.threshold

    base = pd.read_csv(args.base_predictions, encoding="utf-8-sig")
    scene = pd.read_csv(args.scene_predictions, encoding="utf-8-sig")
    require_columns(base, {"fold", "recording_id", "device_id", "source_id", "endpoint_tow", "device_name", "true_state", "pred_state"}, args.base_predictions)
    base = key_columns(base, args.tow_decimals)
    scene_context = global_scene_context(scene, args.tow_decimals)

    train_mod = load_train_module()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    expert, calibrated_threshold = expert_predictions(
        train_mod, args.watch_data_dir, args.expert_checkpoint, watch_ids, device, args.batch_size,
    )
    expert["tow_key"] = expert["endpoint_tow"].round(args.tow_decimals)
    # A source is the unambiguous device stream within a recording.  Direct
    # rows are excluded from expert training/prediction and remain untouched.
    expert_keys = ["recording_id", "device_id", "source_id", "tow_key"]
    if expert.duplicated(expert_keys).any():
        raise ValueError("Watch expert produced duplicate alignment keys")
    merged = base.merge(
        expert[expert_keys + ["watch_anomaly_probability"]], how="left", on=expert_keys, validate="one_to_one",
    )
    scene_keys = ["fold", "recording_id", "tow_key"]
    merged = merged.merge(scene_context, how="left", on=scene_keys, validate="many_to_one")
    threshold = threshold_override if threshold_override is not None else calibrated_threshold

    merged["global_scene_available"] = merged["global_scene_pred"].notna()
    merged["global_scene_pred_name"] = merged["global_scene_pred"].map(SCENE_NAMES).fillna("unknown")
    merged["watch_anomaly_override"] = False
    eligible = (
        merged["device_id"].isin(watch_ids)
        & (merged["pred_state"].astype(int) == 0)
        & merged["watch_anomaly_probability"].notna()
        & (merged["watch_anomaly_probability"] >= threshold)
        & merged["global_scene_available"]
        & (merged["global_scene_confidence"] >= args.min_scene_confidence)
        & merged["global_scene_pred"].astype("Int64").isin(args.scene_classes)
    )
    merged.loc[eligible, "watch_anomaly_override"] = True
    merged["scene_gated_pred_state"] = merged["pred_state"].astype(int)
    merged.loc[eligible, "scene_gated_pred_state"] = 1
    merged["scene_gated_pred_state_name"] = merged["scene_gated_pred_state"].map(STATE_NAMES)

    y_true = merged["true_state"].astype(int).to_numpy()
    base_pred = merged["pred_state"].astype(int).to_numpy()
    guided_pred = merged["scene_gated_pred_state"].astype(int).to_numpy()
    result: dict[str, Any] = {
        "protocol": "scene-gated single-frequency Watch suppression expert",
        "threshold": float(threshold),
        "threshold_source": "argument" if threshold_override is not None else "expert_validation_checkpoint",
        "watch_devices": args.watch_devices,
        "scene_classes": [int(value) for value in args.scene_classes],
        "min_scene_confidence": float(args.min_scene_confidence),
        "global_scene_context_coverage": float(merged["global_scene_available"].mean()),
        "eligible_override_count": int(eligible.sum()),
        "base_metrics": metric_bundle(y_true, base_pred),
        "scene_gated_metrics": metric_bundle(y_true, guided_pred),
        "by_device": {},
    }
    for name, group in merged.groupby("device_name", sort=True):
        result["by_device"][str(name)] = {
            "samples": int(len(group)),
            "eligible_override_count": int(group["watch_anomaly_override"].sum()),
            "base": metric_bundle(group["true_state"].astype(int).to_numpy(), group["pred_state"].astype(int).to_numpy()),
            "scene_gated": metric_bundle(group["true_state"].astype(int).to_numpy(), group["scene_gated_pred_state"].astype(int).to_numpy()),
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output_dir / "test_scene_gated_watch_anomaly_predictions.csv"
    metrics_path = args.output_dir / "test_scene_gated_watch_anomaly_metrics.json"
    merged.to_csv(prediction_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    metrics_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics = result["scene_gated_metrics"]
    print(json.dumps({
        "predictions": str(prediction_path),
        "metrics": str(metrics_path),
        "threshold": threshold,
        "eligible_override_count": result["eligible_override_count"],
        "macro_f1": metrics["macro_f1"],
        "far": metrics["far"],
        "abnormal_recall": metrics["abnormal_recall"],
        "anomaly_recall": metrics["per_class"]["anomaly"]["recall"],
        "direct_recall": metrics["per_class"]["direct"]["recall"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

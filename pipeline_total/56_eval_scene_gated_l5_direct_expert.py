#!/usr/bin/env python3
"""Apply an L5-specific direct-spoof expert under a global L5 scene gate.

The direct label is window-specific: a receiver can be inside an L5 attack
interval but have no observable target-band signal in a particular window.
Therefore this evaluator never uses device capability as a direct-label rule.
Instead, a binary expert trained on L5-capable devices and ``st_L5`` records
supplies the local target-band evidence; a global L5 scene posterior only
selects when that expert is allowed to override the base response state.
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
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--expert-checkpoint", type=Path, required=True)
    parser.add_argument("--base-predictions", type=Path, required=True)
    parser.add_argument("--scene-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-devices", nargs="+", required=True)
    parser.add_argument("--min-scene-confidence", type=float, default=0.50)
    parser.add_argument("--threshold", type=float, help="override calibrated direct-expert threshold")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--tow-decimals", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.min_scene_confidence <= 1:
        parser.error("--min-scene-confidence must be in [0, 1]")
    if args.threshold is not None and not 0 < args.threshold < 1:
        parser.error("--threshold must be strictly between zero and one")
    return args


def load_train_module() -> Any:
    spec = importlib.util.spec_from_file_location("device_event_train", TRAIN_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {TRAIN_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require_columns(frame: pd.DataFrame, columns: set[str], path: Path) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns {missing}")


def metric_bundle(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    labels = [0, 1, 2]
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    normal, abnormal = y_true == 0, y_true > 0
    return {
        "samples": int(len(y_true)), "accuracy": float((y_true == y_pred).mean()),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "far": float((y_pred[normal] > 0).mean()) if normal.any() else 0.0,
        "abnormal_recall": float((y_pred[abnormal] > 0).mean()) if abnormal.any() else 0.0,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).astype(int).tolist(),
        "per_class": {STATE_NAMES[label]: {"precision": float(precision[index]), "recall": float(recall[index]), "f1": float(f1[index]), "support": int(support[index])} for index, label in enumerate(labels)},
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}; use --overwrite")
    metadata = json.loads((args.data_dir / "metadata.json").read_text(encoding="utf-8"))
    mapping = {str(name): int(value) for name, value in metadata.get("device_mapping", {}).items()}
    unknown = sorted(set(args.include_devices).difference(mapping))
    if unknown:
        raise ValueError(f"Unknown devices {unknown}; available={sorted(mapping)}")
    included = {mapping[name] for name in args.include_devices}
    base = pd.read_csv(args.base_predictions, encoding="utf-8-sig")
    scene = pd.read_csv(args.scene_predictions, encoding="utf-8-sig")
    require_columns(base, {"fold", "recording_id", "device_id", "source_id", "endpoint_tow", "device_name", "true_state", "pred_state"}, args.base_predictions)
    require_columns(scene, {"fold", "recording_id", "endpoint_tow", "prob_normal", "prob_L1", "prob_L5", "prob_L1+L5"}, args.scene_predictions)
    base["tow_key"] = base["endpoint_tow"].astype(float).round(args.tow_decimals)
    scene["tow_key"] = scene["endpoint_tow"].astype(float).round(args.tow_decimals)
    prob_cols = ["prob_normal", "prob_L1", "prob_L5", "prob_L1+L5"]
    context = scene.groupby(["fold", "recording_id", "tow_key"], as_index=False)[prob_cols].mean()
    probs = context[prob_cols].to_numpy(dtype=np.float64)
    context["global_scene_pred"] = probs.argmax(axis=1).astype(np.int64)
    context["global_scene_confidence"] = probs.max(axis=1)

    train_mod = load_train_module()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test = train_mod.EventDataset(args.data_dir / "test.npz", "y_response_state", included, "direct")
    checkpoint, model = train_mod.load_checkpoint_model(args.expert_checkpoint, device, test.x.shape[1])
    if int(checkpoint.get("num_classes", 2)) != 2 or checkpoint.get("label_transform") != "direct":
        raise ValueError("L5 direct expert must be a binary checkpoint trained with --label-transform direct")
    threshold = float(args.threshold if args.threshold is not None else checkpoint.get("alarm_threshold", 0.5))
    direct_probability = train_mod.probabilities(model, test, device, args.batch_size)[:, 1]
    expert = pd.DataFrame({
        "device_id": test.device_id.numpy().astype(int), "source_id": test.source_id.numpy().astype(int),
        "recording_id": test.recording_id.numpy().astype(int), "endpoint_tow": test.endpoint_tow.numpy().astype(float),
        "l5_direct_probability": direct_probability.astype(float),
    })
    expert["tow_key"] = expert["endpoint_tow"].round(args.tow_decimals)
    key = ["recording_id", "device_id", "source_id", "tow_key"]
    if expert.duplicated(key).any():
        raise ValueError("L5 direct expert generated duplicate alignment keys")
    merged = base.merge(expert[key + ["l5_direct_probability"]], how="left", on=key, validate="one_to_one")
    merged = merged.merge(context[["fold", "recording_id", "tow_key", "global_scene_pred", "global_scene_confidence"]], how="left", on=["fold", "recording_id", "tow_key"], validate="many_to_one")
    eligible = (
        merged["l5_direct_probability"].notna()
        & (merged["l5_direct_probability"] >= threshold)
        & (merged["global_scene_pred"] == 2)
        & (merged["global_scene_confidence"] >= args.min_scene_confidence)
    )
    merged["l5_direct_override"] = eligible
    merged["scene_gated_l5_direct_pred_state"] = merged["pred_state"].astype(int)
    merged.loc[eligible, "scene_gated_l5_direct_pred_state"] = 2
    merged["scene_gated_l5_direct_pred_state_name"] = merged["scene_gated_l5_direct_pred_state"].map(STATE_NAMES)
    merged["global_scene_pred_name"] = merged["global_scene_pred"].map(SCENE_NAMES).fillna("unknown")
    y = merged["true_state"].astype(int).to_numpy()
    base_pred = merged["pred_state"].astype(int).to_numpy()
    result: dict[str, Any] = {
        "protocol": "L5-specific direct expert with global L5 scene gate", "threshold": threshold,
        "threshold_source": "argument" if args.threshold is not None else "expert_validation_checkpoint",
        "min_scene_confidence": args.min_scene_confidence, "eligible_override_count": int(eligible.sum()),
        "base_metrics": metric_bundle(y, base_pred), "scene_gated_l5_direct_metrics": metric_bundle(y, merged["scene_gated_l5_direct_pred_state"].astype(int).to_numpy()), "by_device": {},
    }
    for name, group in merged.groupby("device_name", sort=True):
        result["by_device"][str(name)] = {
            "samples": int(len(group)), "eligible_override_count": int(group["l5_direct_override"].sum()),
            "base": metric_bundle(group["true_state"].astype(int).to_numpy(), group["pred_state"].astype(int).to_numpy()),
            "scene_gated_l5_direct": metric_bundle(group["true_state"].astype(int).to_numpy(), group["scene_gated_l5_direct_pred_state"].astype(int).to_numpy()),
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output_dir / "test_scene_gated_l5_direct_predictions.csv"
    metrics_path = args.output_dir / "test_scene_gated_l5_direct_metrics.json"
    merged.to_csv(prediction_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    metrics_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics = result["scene_gated_l5_direct_metrics"]
    print(json.dumps({"predictions": str(prediction_path), "metrics": str(metrics_path), "threshold": threshold, "eligible_override_count": result["eligible_override_count"], "macro_f1": metrics["macro_f1"], "far": metrics["far"], "abnormal_recall": metrics["abnormal_recall"], "anomaly_recall": metrics["per_class"]["anomaly"]["recall"], "direct_recall": metrics["per_class"]["direct"]["recall"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

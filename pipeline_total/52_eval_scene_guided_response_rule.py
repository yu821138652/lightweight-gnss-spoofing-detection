#!/usr/bin/env python3
"""Evaluate a scene-guided direct/anomaly rule without retraining.

The rule keeps the response model's normal/abnormal decision, but uses a
matched dual-band scene prediction to resolve an abnormal state as ``direct``
when the scene indicates an attack on L1, L5, or both bands.  Rows without a
usable scene context (for example single-band watches) are left unchanged.

This is an exploratory diagnostic, not a replacement for a learned
scene-conditioned response model.  It tests whether the observed fold-6
failure is mainly a target-band attribution problem.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support


ROOT = Path(__file__).resolve().parents[1]
SCENE_NAMES = {0: "normal", 1: "L1", 2: "L5", 3: "L1+L5"}
STATE_NAMES = {0: "normal", 1: "anomaly", 2: "direct"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-predictions", type=Path, required=True)
    parser.add_argument("--response-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tow-decimals", type=int, default=3)
    parser.add_argument("--min-scene-confidence", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.tow_decimals <= 9:
        parser.error("--tow-decimals must be in [0, 9]")
    if not 0 <= args.min_scene_confidence <= 1:
        parser.error("--min-scene-confidence must be in [0, 1]")
    return args


def require_columns(frame: pd.DataFrame, columns: set[str], path: Path) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns {missing}")


def metric_bundle(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    labels = [0, 1, 2]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0,
    )
    normal = y_true == 0
    abnormal = y_true > 0
    return {
        "samples": int(len(y_true)),
        "accuracy": float((y_true == y_pred).mean()),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "far": float((y_pred[normal] > 0).mean()) if normal.any() else 0.0,
        "abnormal_recall": float((y_pred[abnormal] > 0).mean()) if abnormal.any() else 0.0,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
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


def add_keys(frame: pd.DataFrame, decimals: int) -> pd.DataFrame:
    result = frame.copy()
    result["fold"] = result["fold"].astype(int)
    result["recording_id"] = result["recording_id"].astype(int)
    result["device_id"] = result["device_id"].astype(int)
    result["tow_key"] = result["endpoint_tow"].astype(float).round(decimals)
    return result


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}; use --overwrite")
    scene = pd.read_csv(args.scene_predictions, encoding="utf-8-sig")
    response = pd.read_csv(args.response_predictions, encoding="utf-8-sig")
    require_columns(
        scene,
        {"fold", "recording_id", "device_id", "endpoint_tow", "pred_class", "prob_normal", "prob_L1", "prob_L5", "prob_L1+L5"},
        args.scene_predictions,
    )
    require_columns(
        response,
        {"fold", "recording_id", "device_id", "endpoint_tow", "device_name", "true_state", "pred_state"},
        args.response_predictions,
    )
    scene = add_keys(scene, args.tow_decimals)
    response = add_keys(response, args.tow_decimals)
    scene_key = ["fold", "recording_id", "device_id", "tow_key"]
    if scene.duplicated(scene_key).any():
        raise ValueError("Scene predictions contain duplicate alignment keys")
    scene_fields = scene[scene_key + ["pred_class", "prob_normal", "prob_L1", "prob_L5", "prob_L1+L5"]]
    merged = response.merge(scene_fields, how="left", on=scene_key, validate="many_to_one")
    prob_columns = ["prob_normal", "prob_L1", "prob_L5", "prob_L1+L5"]
    merged["scene_context_available"] = merged["pred_class"].notna()
    merged["scene_confidence"] = merged[prob_columns].max(axis=1).fillna(0.0)
    merged["guided_pred_state"] = merged["pred_state"].astype(int)
    eligible = (
        merged["scene_context_available"]
        & (merged["scene_confidence"] >= args.min_scene_confidence)
        & merged["pred_class"].isin([1, 2, 3])
        & (merged["pred_state"].astype(int) == 1)
    )
    merged.loc[eligible, "guided_pred_state"] = 2
    merged["guided_pred_state_name"] = merged["guided_pred_state"].map(STATE_NAMES)
    merged["scene_pred_name"] = merged["pred_class"].map(SCENE_NAMES).fillna("unknown")

    y_true = merged["true_state"].astype(int).to_numpy()
    base_pred = merged["pred_state"].astype(int).to_numpy()
    guided_pred = merged["guided_pred_state"].astype(int).to_numpy()
    result: dict[str, Any] = {
        "rule": "scene_attack_context + response_abnormal -> direct",
        "scene_predictions": str(args.scene_predictions),
        "response_predictions": str(args.response_predictions),
        "tow_decimals": args.tow_decimals,
        "min_scene_confidence": args.min_scene_confidence,
        "scene_context_coverage": float(merged["scene_context_available"].mean()),
        "eligible_override_count": int(eligible.sum()),
        "base_metrics": metric_bundle(y_true, base_pred),
        "guided_metrics": metric_bundle(y_true, guided_pred),
        "by_device": {},
    }
    for device_name, group in merged.groupby("device_name", sort=True):
        gy = group["true_state"].astype(int).to_numpy()
        gb = group["pred_state"].astype(int).to_numpy()
        gg = group["guided_pred_state"].astype(int).to_numpy()
        result["by_device"][str(device_name)] = {
            "samples": int(len(group)),
            "scene_context_coverage": float(group["scene_context_available"].mean()),
            "base": metric_bundle(gy, gb),
            "guided": metric_bundle(gy, gg),
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "test_scene_guided_response_predictions.csv"
    metrics_path = args.output_dir / "test_scene_guided_response_metrics.json"
    merged.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    metrics_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "predictions": str(predictions_path),
        "metrics": str(metrics_path),
        "scene_context_coverage": result["scene_context_coverage"],
        "eligible_override_count": result["eligible_override_count"],
        "base": result["base_metrics"],
        "guided": result["guided_metrics"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

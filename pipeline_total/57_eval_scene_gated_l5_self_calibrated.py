#!/usr/bin/env python3
"""Evaluate a self-calibrated L5 direct-evidence rule.

For each L5-capable device/source stream, estimate a normal L5 C/N0 lower
tail from an early post-start calibration slice.  Under a confident global L5
scene, a later window below that stream-specific threshold is assigned direct
spoofing.  No response labels are used to form the threshold; the early slice
is only a receiver self-calibration period.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support


STATE_NAMES = {0: "normal", 1: "anomaly", 2: "direct"}
SCENE_NAMES = {0: "normal", 1: "L1", 2: "L5", 3: "L1+L5"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--base-predictions", type=Path, required=True)
    parser.add_argument("--scene-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-devices", nargs="+", required=True)
    parser.add_argument("--feature-name", default="initial_baseline_delta_l5_cn0_last_q25")
    parser.add_argument("--calibration-offset-windows", type=int, default=30)
    parser.add_argument("--calibration-windows", type=int, default=30)
    parser.add_argument("--lower-quantile", type=float, default=0.10)
    parser.add_argument("--min-scene-confidence", type=float, default=0.50)
    parser.add_argument("--tow-decimals", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.calibration_offset_windows < 0 or args.calibration_windows < 1:
        parser.error("calibration windows must be non-negative/positive")
    if not 0 < args.lower_quantile < 1 or not 0 <= args.min_scene_confidence <= 1:
        parser.error("quantile/confidence values are out of range")
    return args


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
    names = list(metadata.get("feature_names", []))
    if args.feature_name not in names:
        raise ValueError(f"Feature {args.feature_name!r} not found in {args.data_dir / 'metadata.json'}")
    feature_index = names.index(args.feature_name)
    mapping = {str(name): int(value) for name, value in metadata.get("device_mapping", {}).items()}
    unknown = sorted(set(args.include_devices).difference(mapping))
    if unknown:
        raise ValueError(f"Unknown devices {unknown}; available={sorted(mapping)}")
    include_ids = set(mapping[name] for name in args.include_devices)
    with np.load(args.data_dir / "test.npz", allow_pickle=False) as data:
        device_id = data["device_id"].astype(int)
        recording_id = data["recording_id"].astype(int)
        source_id = data["source_id"].astype(int)
        endpoint_tow = data["endpoint_tow"].astype(float)
        endpoint_time = data["endpoint_utc_millis"].astype(float)
        feature = data["x"][:, feature_index].astype(float)
    threshold = np.full(len(feature), np.nan, dtype=float)
    for recording, source in np.unique(np.column_stack((recording_id, source_id)), axis=0):
        group = np.flatnonzero((recording_id == recording) & (source_id == source))
        ordered = group[np.argsort(endpoint_time[group], kind="mergesort")]
        start = args.calibration_offset_windows
        stop = start + args.calibration_windows
        calibration = feature[ordered[start:stop]]
        calibration = calibration[np.isfinite(calibration)]
        if len(calibration) < args.calibration_windows:
            continue
        threshold[ordered] = float(np.quantile(calibration, args.lower_quantile))
    base = pd.read_csv(args.base_predictions, encoding="utf-8-sig")
    scene = pd.read_csv(args.scene_predictions, encoding="utf-8-sig")
    require_columns(base, {"fold", "recording_id", "device_id", "source_id", "endpoint_tow", "device_name", "true_state", "pred_state"}, args.base_predictions)
    require_columns(scene, {"fold", "recording_id", "endpoint_tow", "prob_normal", "prob_L1", "prob_L5", "prob_L1+L5"}, args.scene_predictions)
    base["tow_key"] = base["endpoint_tow"].astype(float).round(args.tow_decimals)
    scene["tow_key"] = scene["endpoint_tow"].astype(float).round(args.tow_decimals)
    probability_columns = ["prob_normal", "prob_L1", "prob_L5", "prob_L1+L5"]
    context = scene.groupby(["fold", "recording_id", "tow_key"], as_index=False)[probability_columns].mean()
    probabilities = context[probability_columns].to_numpy(dtype=float)
    context["global_scene_pred"] = probabilities.argmax(axis=1).astype(int)
    context["global_scene_confidence"] = probabilities.max(axis=1)
    feature_frame = pd.DataFrame({
        "recording_id": recording_id, "device_id": device_id, "source_id": source_id,
        "endpoint_tow": endpoint_tow, "self_l5_score": feature, "self_l5_threshold": threshold,
    })
    feature_frame["tow_key"] = feature_frame["endpoint_tow"].round(args.tow_decimals)
    key = ["recording_id", "device_id", "source_id", "tow_key"]
    if feature_frame.duplicated(key).any():
        raise ValueError("Device tensor contains duplicate alignment keys")
    merged = base.merge(feature_frame[key + ["self_l5_score", "self_l5_threshold"]], how="left", on=key, validate="one_to_one")
    merged = merged.merge(context[["fold", "recording_id", "tow_key", "global_scene_pred", "global_scene_confidence"]], how="left", on=["fold", "recording_id", "tow_key"], validate="many_to_one")
    eligible = (
        merged["device_name"].isin(args.include_devices)
        & merged["self_l5_score"].notna() & merged["self_l5_threshold"].notna()
        & (merged["self_l5_score"] <= merged["self_l5_threshold"])
        & (merged["global_scene_pred"] == 2)
        & (merged["global_scene_confidence"] >= args.min_scene_confidence)
    )
    merged["self_calibrated_l5_direct_override"] = eligible
    merged["self_calibrated_l5_direct_pred_state"] = merged["pred_state"].astype(int)
    merged.loc[eligible, "self_calibrated_l5_direct_pred_state"] = 2
    merged["self_calibrated_l5_direct_pred_state_name"] = merged["self_calibrated_l5_direct_pred_state"].map(STATE_NAMES)
    merged["global_scene_pred_name"] = merged["global_scene_pred"].map(SCENE_NAMES).fillna("unknown")
    y = merged["true_state"].astype(int).to_numpy()
    base_pred = merged["pred_state"].astype(int).to_numpy()
    calibrated_pred = merged["self_calibrated_l5_direct_pred_state"].astype(int).to_numpy()
    result: dict[str, Any] = {
        "protocol": "self-calibrated L5 direct evidence with global L5 scene gate",
        "feature_name": args.feature_name, "calibration_offset_windows": args.calibration_offset_windows,
        "calibration_windows": args.calibration_windows, "lower_quantile": args.lower_quantile,
        "min_scene_confidence": args.min_scene_confidence, "include_devices": args.include_devices,
        "eligible_override_count": int(eligible.sum()), "base_metrics": metric_bundle(y, base_pred),
        "self_calibrated_metrics": metric_bundle(y, calibrated_pred), "by_device": {},
    }
    for name, group in merged.groupby("device_name", sort=True):
        result["by_device"][str(name)] = {
            "samples": int(len(group)), "eligible_override_count": int(group["self_calibrated_l5_direct_override"].sum()),
            "base": metric_bundle(group["true_state"].astype(int).to_numpy(), group["pred_state"].astype(int).to_numpy()),
            "self_calibrated": metric_bundle(group["true_state"].astype(int).to_numpy(), group["self_calibrated_l5_direct_pred_state"].astype(int).to_numpy()),
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output_dir / "test_self_calibrated_l5_direct_predictions.csv"
    metrics_path = args.output_dir / "test_self_calibrated_l5_direct_metrics.json"
    merged.to_csv(prediction_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    metrics_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics = result["self_calibrated_metrics"]
    print(json.dumps({"predictions": str(prediction_path), "metrics": str(metrics_path), "eligible_override_count": result["eligible_override_count"], "macro_f1": metrics["macro_f1"], "far": metrics["far"], "abnormal_recall": metrics["abnormal_recall"], "anomaly_recall": metrics["per_class"]["anomaly"]["recall"], "direct_recall": metrics["per_class"]["direct"]["recall"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

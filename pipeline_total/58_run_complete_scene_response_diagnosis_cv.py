#!/usr/bin/env python3
"""Assemble the complete six-fold scene-response diagnosis result.

This runner combines two independently validated, non-overlapping repairs on
top of the existing response backbone:

* single-frequency Watch: a warm-up-baseline suppression expert repairs
  ``normal -> anomaly`` only for the two Watches under a global L5 scene;
* L5-capable phones: a self-calibrated L5 C/N0 lower-tail rule repairs
  ``normal/anomaly -> direct`` only under a global L5 scene.

The Watch branch must be generated first by
``55_run_scene_gated_watch_anomaly_cv.py`` because it builds tensors and
trains the Watch experts.  This script runs only the lightweight direct-rule
evaluation and the deterministic final fusion.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "pipeline_total"
STATE_NAMES = {0: "normal", 1: "anomaly", 2: "direct"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-exe", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--response-tensors-root", type=Path,
        default=ROOT / "output" / "optimization" / "response_extreme_cv_v1",
    )
    parser.add_argument("--response-tensor-dirname", default="device_tensors_sparse_initial30_cn0_extreme")
    parser.add_argument("--base-response-name", default="direct_override_global_t020_mlp_h16")
    parser.add_argument(
        "--scene-training-root", type=Path,
        default=ROOT / "output" / "training" / "static_scene_response_fusion_s1_cn0_only" / "band_scene",
    )
    parser.add_argument(
        "--watch-output-root", type=Path,
        default=ROOT / "output" / "optimization" / "watch_suppression_warmup_cv_v1",
        help="output root produced by 55_run_scene_gated_watch_anomaly_cv.py",
    )
    parser.add_argument(
        "--self-calibrated-output-root", type=Path,
        default=ROOT / "output" / "optimization" / "l5_self_calibrated_cv_v1",
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=ROOT / "output" / "optimization" / "complete_scene_response_diagnosis_cv_v1",
    )
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 4, 5, 6, 7])
    parser.add_argument(
        "--l5-capable-devices", nargs="+",
        default=["Google_Pixel6", "HUAWEI_Mate40", "RedMi_K60", "XiaoMi_MI8"],
    )
    parser.add_argument("--calibration-offset-windows", type=int, default=30)
    parser.add_argument("--calibration-windows", type=int, default=30)
    parser.add_argument("--lower-quantile", type=float, default=0.10)
    parser.add_argument("--min-scene-confidence", type=float, default=0.50)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    if args.calibration_offset_windows < 0 or args.calibration_windows < 1:
        parser.error("calibration windows are invalid")
    if not 0 < args.lower_quantile < 1 or not 0 <= args.min_scene_confidence <= 1:
        parser.error("quantile/confidence values are invalid")
    return args


def run(command: list[str]) -> None:
    print("run:", " ".join(command), flush=True)
    result = subprocess.run(command, cwd=str(ROOT))
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(command)}")


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
        "per_class": {
            STATE_NAMES[label]: {
                "precision": float(precision[index]), "recall": float(recall[index]),
                "f1": float(f1[index]), "support": int(support[index]),
            }
            for index, label in enumerate(labels)
        },
    }


def paths_for_fold(args: argparse.Namespace, fold: int) -> dict[str, Path]:
    response_root = args.response_tensors_root / f"fold_{fold}"
    return {
        "tensor": response_root / args.response_tensor_dirname,
        "base": response_root / args.base_response_name / "test_response_state_predictions.csv",
        "scene": args.scene_training_root / f"fold_{fold}" / "test_predictions_band_mean_window_tcn.csv",
        "watch": args.watch_output_root / f"fold_{fold}" / "scene_gated_eval" / "test_scene_gated_watch_anomaly_predictions.csv",
        "self_root": args.self_calibrated_output_root / f"fold_{fold}",
        "self": args.self_calibrated_output_root / f"fold_{fold}" / "test_self_calibrated_l5_direct_predictions.csv",
        "final_root": args.output_root / f"fold_{fold}",
        "final": args.output_root / f"fold_{fold}" / "test_complete_scene_response_predictions.csv",
    }


def fuse_fold(args: argparse.Namespace, fold: int) -> Path:
    paths = paths_for_fold(args, fold)
    if args.skip_existing and paths["final"].is_file():
        print(f"fold {fold}: reusing {paths['final']}", flush=True)
        return paths["final"]
    for key in ("tensor", "base", "scene", "watch"):
        if not paths[key].exists():
            raise FileNotFoundError(f"fold {fold}: missing {key} path {paths[key]}")
    if not (args.skip_existing and paths["self"].is_file()):
        run([
            str(args.python_exe), str(PIPELINE / "57_eval_scene_gated_l5_self_calibrated.py"),
            "--data-dir", str(paths["tensor"]), "--base-predictions", str(paths["base"]),
            "--scene-predictions", str(paths["scene"]), "--output-dir", str(paths["self_root"]),
            "--include-devices", *args.l5_capable_devices,
            "--calibration-offset-windows", str(args.calibration_offset_windows),
            "--calibration-windows", str(args.calibration_windows),
            "--lower-quantile", str(args.lower_quantile),
            "--min-scene-confidence", str(args.min_scene_confidence), "--overwrite",
        ])
    watch = pd.read_csv(paths["watch"], encoding="utf-8-sig")
    direct = pd.read_csv(paths["self"], encoding="utf-8-sig")
    keys = ["fold", "recording_id", "device_id", "source_id", "tow_key"]
    watch_required = set(keys + ["true_state", "pred_state", "watch_anomaly_override"])
    direct_required = set(keys + ["true_state", "self_calibrated_l5_direct_pred_state"])
    missing_watch = sorted(watch_required.difference(watch.columns))
    missing_direct = sorted(direct_required.difference(direct.columns))
    if missing_watch or missing_direct:
        raise ValueError(f"fold {fold}: incompatible prediction exports; watch={missing_watch}, direct={missing_direct}")
    merged = direct[keys + ["true_state", "self_calibrated_l5_direct_pred_state"]].merge(
        watch[keys + ["pred_state", "watch_anomaly_override"]], on=keys, how="inner", validate="one_to_one",
    )
    if len(merged) != len(direct) or len(merged) != len(watch):
        raise ValueError(f"fold {fold}: incomplete Watch/direct prediction alignment")
    if not np.array_equal(merged["true_state"].to_numpy(), watch.sort_values(keys)["true_state"].to_numpy()):
        # The merge preserves the direct-row ordering rather than a generic
        # sort, so compare through a key-based merge instead of row positions.
        truth = watch[keys + ["true_state"]].merge(direct[keys + ["true_state"]], on=keys, validate="one_to_one")
        if not np.array_equal(truth["true_state_x"].to_numpy(), truth["true_state_y"].to_numpy()):
            raise ValueError(f"fold {fold}: Watch/direct truth labels differ")
    merged["complete_pred_state"] = merged["self_calibrated_l5_direct_pred_state"].astype(int)
    watch_repair = merged["watch_anomaly_override"].astype(bool) & (merged["complete_pred_state"] == 0)
    merged.loc[watch_repair, "complete_pred_state"] = 1
    merged["complete_pred_state_name"] = merged["complete_pred_state"].map(STATE_NAMES)
    merged["watch_anomaly_repair_applied"] = watch_repair
    y = merged["true_state"].astype(int).to_numpy()
    base = merged["pred_state"].astype(int).to_numpy()
    final = merged["complete_pred_state"].astype(int).to_numpy()
    result: dict[str, Any] = {
        "fold": int(fold), "watch_anomaly_repair_count": int(watch_repair.sum()),
        "l5_direct_repair_count": int((merged["self_calibrated_l5_direct_pred_state"].astype(int) != base).sum()),
        "base_metrics": metric_bundle(y, base), "complete_metrics": metric_bundle(y, final), "by_device": {},
    }
    device_names = watch[keys + ["device_name"]]
    merged = merged.merge(device_names, on=keys, how="left", validate="one_to_one")
    for name, group in merged.groupby("device_name", sort=True):
        result["by_device"][str(name)] = {
            "samples": int(len(group)), "watch_anomaly_repair_count": int(group["watch_anomaly_repair_applied"].sum()),
            "complete": metric_bundle(group["true_state"].astype(int).to_numpy(), group["complete_pred_state"].astype(int).to_numpy()),
        }
    paths["final_root"].mkdir(parents=True, exist_ok=True)
    merged.to_csv(paths["final"], index=False, encoding="utf-8-sig")
    (paths["final_root"] / "test_complete_scene_response_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return paths["final"]


def aggregate(args: argparse.Namespace, prediction_paths: list[Path]) -> dict[str, Any]:
    combined = pd.concat([pd.read_csv(path, encoding="utf-8-sig") for path in prediction_paths], ignore_index=True)
    required = {"fold", "device_name", "true_state", "pred_state", "complete_pred_state", "watch_anomaly_repair_applied"}
    missing = sorted(required.difference(combined.columns))
    if missing:
        raise ValueError(f"Final prediction exports missing columns {missing}")
    y = combined["true_state"].astype(int).to_numpy()
    base = combined["pred_state"].astype(int).to_numpy()
    final = combined["complete_pred_state"].astype(int).to_numpy()
    result: dict[str, Any] = {
        "protocol": "static_time_block_outer_v2",
        "configuration": {
            "watch_anomaly": "MLP-h8, 11-D L1 suppression features, 30-window baseline after 300-window warm-up",
            "l5_direct": "per-stream self-calibrated L5 C/N0 q25, windows 30:60, lower quantile %.3f" % args.lower_quantile,
            "scene_gate": "global pooled L5 posterior confidence >= %.3f" % args.min_scene_confidence,
        },
        "folds": [int(fold) for fold in args.folds],
        "watch_anomaly_repair_count": int(combined["watch_anomaly_repair_applied"].astype(bool).sum()),
        "base_metrics": metric_bundle(y, base), "complete_metrics": metric_bundle(y, final),
        "by_fold": {}, "by_device": {},
    }
    for fold, group in combined.groupby("fold", sort=True):
        result["by_fold"][str(int(fold))] = {
            "watch_anomaly_repair_count": int(group["watch_anomaly_repair_applied"].astype(bool).sum()),
            "complete": metric_bundle(group["true_state"].astype(int).to_numpy(), group["complete_pred_state"].astype(int).to_numpy()),
        }
    for name, group in combined.groupby("device_name", sort=True):
        result["by_device"][str(name)] = {
            "samples": int(len(group)), "watch_anomaly_repair_count": int(group["watch_anomaly_repair_applied"].astype(bool).sum()),
            "complete": metric_bundle(group["true_state"].astype(int).to_numpy(), group["complete_pred_state"].astype(int).to_numpy()),
        }
    args.output_root.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output_root / "aggregate_complete_scene_response_predictions.csv", index=False, encoding="utf-8-sig")
    (args.output_root / "aggregate_complete_scene_response_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return result


def main() -> None:
    args = parse_args()
    prediction_paths: list[Path] = []
    for fold in args.folds:
        paths = paths_for_fold(args, fold)
        if args.aggregate_only:
            if not paths["final"].is_file():
                raise FileNotFoundError(paths["final"])
            prediction_paths.append(paths["final"])
        else:
            prediction_paths.append(fuse_fold(args, fold))
    result = aggregate(args, prediction_paths)
    metrics = result["complete_metrics"]
    print(json.dumps({
        "metrics": str(args.output_root / "aggregate_complete_scene_response_metrics.json"),
        "watch_anomaly_repair_count": result["watch_anomaly_repair_count"],
        "macro_f1": metrics["macro_f1"], "far": metrics["far"],
        "abnormal_recall": metrics["abnormal_recall"],
        "anomaly_recall": metrics["per_class"]["anomaly"]["recall"],
        "direct_recall": metrics["per_class"]["direct"]["recall"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run six-fold validation for the scene-gated single-frequency Watch expert.

Each outer fold uses a post-warm-up fixed L1 baseline for Watch1/Watch2,
trains a binary ``normal/anomaly`` suppression expert without direct-spoof
rows, calibrates it only on the inner validation split, and applies it to the
existing response predictions only under a globally pooled L5 scene context.

The script writes no data outside its output root.  Tensor/model artifacts are
kept under ``output/`` and are intentionally not repository source files.
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
        "--signal-tensors-root", type=Path,
        default=ROOT / "output" / "tensors" / "static_timeblock_outer_v2",
    )
    parser.add_argument(
        "--base-response-root", type=Path,
        default=ROOT / "output" / "optimization" / "response_extreme_cv_v1",
    )
    parser.add_argument(
        "--base-response-name", type=str, default="direct_override_global_t020_mlp_h16",
    )
    parser.add_argument(
        "--scene-training-root", type=Path,
        default=ROOT / "output" / "training" / "static_scene_response_fusion_s1_cn0_only" / "band_scene",
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=ROOT / "output" / "optimization" / "watch_suppression_warmup_cv_v1",
    )
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 4, 5, 6, 7])
    parser.add_argument("--hidden-dim", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--baseline-windows", type=int, default=30)
    parser.add_argument("--baseline-offset-windows", type=int, default=300)
    parser.add_argument("--max-val-far", type=float, default=0.05)
    parser.add_argument("--min-scene-confidence", type=float, default=0.50)
    parser.add_argument(
        "--invalid-calibration-policy", choices=("error", "disable"), default="error",
        help=(
            "how to handle a fold whose Watch inner validation split lacks normal or anomaly support; "
            "error (default) stops, disable writes an explicitly unaided base-response fold"
        ),
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    if min(args.hidden_dim, args.epochs, args.batch_size, args.patience, args.baseline_windows) < 1:
        parser.error("model/training and baseline window counts must be positive")
    if args.baseline_offset_windows < 0:
        parser.error("--baseline-offset-windows must be non-negative")
    if not 0 <= args.max_val_far <= 1 or not 0 <= args.min_scene_confidence <= 1:
        parser.error("FAR/confidence constraints must be in [0, 1]")
    return args


def run(command: list[str]) -> None:
    print("run:", " ".join(command), flush=True)
    result = subprocess.run(command, cwd=str(ROOT))
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(command)}")


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


def fold_paths(args: argparse.Namespace, fold: int) -> dict[str, Path]:
    fold_root = args.output_root / f"fold_{fold}"
    return {
        "signal": args.signal_tensors_root / f"fold_{fold}",
        "base": args.base_response_root / f"fold_{fold}" / args.base_response_name / "test_response_state_predictions.csv",
        "scene": args.scene_training_root / f"fold_{fold}" / "test_predictions_band_mean_window_tcn.csv",
        "tensor": fold_root / "device_tensors_l1_suppression",
        "expert": fold_root / f"watch_anomaly_mlp_h{args.hidden_dim}",
        "eval": fold_root / "scene_gated_eval",
        "predictions": fold_root / "scene_gated_eval" / "test_scene_gated_watch_anomaly_predictions.csv",
        "audit": fold_root / "watch_calibration_audit.json",
    }


def watch_validation_audit(tensor_dir: Path) -> dict[str, Any]:
    """Audit the exact filtered validation support used by the Watch expert."""
    metadata = json.loads((tensor_dir / "metadata.json").read_text(encoding="utf-8"))
    mapping = {str(name): int(value) for name, value in metadata.get("device_mapping", {}).items()}
    names = ("Google_Pixel_Watch1", "Google_Pixel_Watch2")
    missing = sorted(set(names).difference(mapping))
    if missing:
        raise ValueError(f"Watch tensor metadata is missing devices {missing}")
    watch_ids = np.asarray([mapping[name] for name in names], dtype=np.int64)
    with np.load(tensor_dir / "val.npz", allow_pickle=False) as data:
        required = {"device_id", "y_response_state"}
        if missing := required.difference(data.files):
            raise ValueError(f"{tensor_dir / 'val.npz'} is missing {sorted(missing)}")
        raw_state = data["y_response_state"].astype(np.int64)
        is_watch = np.isin(data["device_id"].astype(np.int64), watch_ids)
    # anomaly_only maps state 1 -> positive, 0 -> negative, and drops direct state 2.
    included = is_watch & (raw_state != 2)
    negative_support = int((raw_state[included] == 0).sum())
    positive_support = int((raw_state[included] == 1).sum())
    audit: dict[str, Any] = {
        "split": "val",
        "watch_devices": list(names),
        "label_transform": "anomaly_only",
        "negative_support": negative_support,
        "positive_support": positive_support,
        "excluded_direct_support": int((is_watch & (raw_state == 2)).sum()),
        "calibration_valid": bool(negative_support > 0 and positive_support > 0),
    }
    inner_split_audit = tensor_dir / "attack_aware_inner_split_audit.json"
    if inner_split_audit.is_file():
        audit["inner_split_audit"] = json.loads(inner_split_audit.read_text(encoding="utf-8"))
    return audit


def write_disabled_fold(args: argparse.Namespace, fold: int, paths: dict[str, Path], audit: dict[str, Any]) -> Path:
    """Export an explicit no-Watch-correction fold when calibration is invalid."""
    base = pd.read_csv(paths["base"], encoding="utf-8-sig")
    required = {
        "fold", "recording_id", "device_id", "source_id", "endpoint_tow",
        "device_name", "true_state", "pred_state",
    }
    missing = sorted(required.difference(base.columns))
    if missing:
        raise ValueError(f"{paths['base']} is missing columns {missing}")
    base = base.copy()
    base["watch_anomaly_probability"] = np.nan
    base["watch_anomaly_override"] = False
    base["scene_gated_pred_state"] = base["pred_state"].astype(int)
    base["scene_gated_pred_state_name"] = base["scene_gated_pred_state"].map(STATE_NAMES)
    base["tow_key"] = base["endpoint_tow"].astype(float).round(3)
    base["watch_expert_status"] = "disabled_invalid_calibration"
    paths["eval"].mkdir(parents=True, exist_ok=True)
    base.to_csv(paths["predictions"], index=False, encoding="utf-8-sig")
    y_true = base["true_state"].astype(int).to_numpy()
    y_pred = base["pred_state"].astype(int).to_numpy()
    result = {
        "protocol": "scene-gated single-frequency Watch suppression expert",
        "fold": int(fold),
        "watch_expert_enabled": False,
        "reason": "invalid_inner_validation_calibration",
        "calibration_audit": audit,
        "eligible_override_count": 0,
        "base_metrics": metric_bundle(y_true, y_pred),
        "scene_gated_metrics": metric_bundle(y_true, y_pred),
    }
    (paths["eval"] / "test_scene_gated_watch_anomaly_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(
        f"fold {fold}: Watch expert disabled because its inner validation split is invalid; "
        f"negative_support={audit['negative_support']} positive_support={audit['positive_support']}",
        flush=True,
    )
    return paths["predictions"]


def run_fold(args: argparse.Namespace, fold: int) -> Path:
    paths = fold_paths(args, fold)
    for key in ("signal", "base", "scene"):
        if not paths[key].exists():
            raise FileNotFoundError(paths[key])
    python = str(args.python_exe)
    if not (args.skip_existing and (paths["tensor"] / "train.npz").is_file()):
        run([
            python, str(PIPELINE / "36_build_device_attack_event_tensors.py"),
            "--signal-data-dir", str(paths["signal"]), "--output-dir", str(paths["tensor"]),
            "--feature-set", "initial_baseline_delta_l1_cn0_extreme",
            "--device-aggregate-profile", "sparse_extreme",
            "--initial-baseline-windows", str(args.baseline_windows),
            "--initial-baseline-offset-windows", str(args.baseline_offset_windows),
            "--initial-baseline-policy", "exclude_stream", "--overwrite",
        ])
    audit = watch_validation_audit(paths["tensor"])
    audit["fold"] = int(fold)
    audit["invalid_calibration_policy"] = args.invalid_calibration_policy
    if not audit["calibration_valid"]:
        audit["watch_expert_enabled"] = False
        audit["status"] = "disabled_invalid_calibration" if args.invalid_calibration_policy == "disable" else "blocked_invalid_calibration"
        paths["audit"].write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.invalid_calibration_policy == "disable":
            return write_disabled_fold(args, fold, paths, audit)
        raise RuntimeError(
            f"fold {fold}: Watch threshold calibration is invalid because its inner validation split has "
            f"negative_support={audit['negative_support']} and positive_support={audit['positive_support']}. "
            "Use a leakage-free calibration split containing both classes, or explicitly rerun with "
            "--invalid-calibration-policy disable to exclude the Watch expert from this fold."
        )
    audit["watch_expert_enabled"] = True
    audit["status"] = "valid_inner_calibration"
    paths["audit"].write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.skip_existing and paths["predictions"].is_file():
        print(f"fold {fold}: reusing {paths['predictions']} after validation-support audit", flush=True)
        return paths["predictions"]
    checkpoint = paths["expert"] / "best_device_event_mlp.pt"
    if not (args.skip_existing and checkpoint.is_file()):
        run([
            python, str(PIPELINE / "37_train_device_attack_event.py"),
            "--data-dir", str(paths["tensor"]), "--output-dir", str(paths["expert"]),
            "--include-devices", "Google_Pixel_Watch1", "Google_Pixel_Watch2",
            "--label-key", "y_response_state", "--label-transform", "anomaly_only", "--num-classes", "2",
            "--model", "mlp", "--hidden-dim", str(args.hidden_dim), "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size), "--patience", str(args.patience),
        ])
    calibrated = paths["expert"] / f"calibrated_{checkpoint.name}"
    if not (args.skip_existing and calibrated.is_file()):
        run([
            python, str(PIPELINE / "37_train_device_attack_event.py"),
            "--data-dir", str(paths["tensor"]), "--output-dir", str(paths["expert"]),
            "--include-devices", "Google_Pixel_Watch1", "Google_Pixel_Watch2",
            "--label-key", "y_response_state", "--label-transform", "anomaly_only",
            "--checkpoint", str(checkpoint), "--calibrate-only", "--max-val-far", str(args.max_val_far),
        ])
    run([
        python, str(PIPELINE / "54_eval_scene_gated_watch_anomaly.py"),
        "--watch-data-dir", str(paths["tensor"]), "--expert-checkpoint", str(calibrated),
        "--base-predictions", str(paths["base"]), "--scene-predictions", str(paths["scene"]),
        "--output-dir", str(paths["eval"]), "--min-scene-confidence", str(args.min_scene_confidence),
        "--overwrite",
    ])
    return paths["predictions"]


def aggregate(args: argparse.Namespace, prediction_paths: list[Path]) -> dict[str, Any]:
    combined = pd.concat([pd.read_csv(path, encoding="utf-8-sig") for path in prediction_paths], ignore_index=True)
    required = {"fold", "device_name", "true_state", "pred_state", "scene_gated_pred_state", "watch_anomaly_override"}
    missing = sorted(required.difference(combined.columns))
    if missing:
        raise ValueError(f"Prediction exports missing columns {missing}")
    y_true = combined["true_state"].astype(int).to_numpy()
    base_pred = combined["pred_state"].astype(int).to_numpy()
    guided_pred = combined["scene_gated_pred_state"].astype(int).to_numpy()
    calibration_audit = {}
    for fold in args.folds:
        audit_path = fold_paths(args, fold)["audit"]
        if not audit_path.is_file():
            raise FileNotFoundError(
                f"Missing calibration audit for fold {fold}: {audit_path}. "
                "Refusing to aggregate unaudited Watch predictions."
            )
        calibration_audit[str(int(fold))] = json.loads(audit_path.read_text(encoding="utf-8"))
    result: dict[str, Any] = {
        "protocol": "static_time_block_outer_v2",
        "configuration": {
            "watch_features": "11-D L1 C/N0 and signal-count deltas from a 30-window baseline after 300-window warm-up",
            "watch_model": f"MLP-h{args.hidden_dim}, anomaly_only",
            "scene_gate": "global pooled scene posterior = L5 with confidence >= %.3f" % args.min_scene_confidence,
            "threshold_selection": "inner validation, max FAR %.3f" % args.max_val_far,
            "invalid_calibration_policy": args.invalid_calibration_policy,
        },
        "folds": [int(fold) for fold in args.folds],
        "watch_calibration_audit": calibration_audit,
        "watch_anomaly_override_count": int(combined["watch_anomaly_override"].astype(bool).sum()),
        "base_metrics": metric_bundle(y_true, base_pred),
        "scene_gated_watch_metrics": metric_bundle(y_true, guided_pred),
        "by_fold": {},
        "by_device": {},
    }
    for fold, group in combined.groupby("fold", sort=True):
        result["by_fold"][str(int(fold))] = {
            "watch_calibration": calibration_audit[str(int(fold))],
            "watch_anomaly_override_count": int(group["watch_anomaly_override"].astype(bool).sum()),
            "base": metric_bundle(group["true_state"].astype(int).to_numpy(), group["pred_state"].astype(int).to_numpy()),
            "scene_gated": metric_bundle(group["true_state"].astype(int).to_numpy(), group["scene_gated_pred_state"].astype(int).to_numpy()),
        }
    for name, group in combined.groupby("device_name", sort=True):
        result["by_device"][str(name)] = {
            "samples": int(len(group)),
            "watch_anomaly_override_count": int(group["watch_anomaly_override"].astype(bool).sum()),
            "scene_gated": metric_bundle(group["true_state"].astype(int).to_numpy(), group["scene_gated_pred_state"].astype(int).to_numpy()),
        }
    args.output_root.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output_root / "aggregate_scene_gated_watch_predictions.csv", index=False, encoding="utf-8-sig")
    (args.output_root / "aggregate_scene_gated_watch_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return result


def main() -> None:
    args = parse_args()
    prediction_paths: list[Path] = []
    for fold in args.folds:
        paths = fold_paths(args, fold)
        if args.aggregate_only:
            if not paths["predictions"].is_file():
                raise FileNotFoundError(paths["predictions"])
            if not paths["audit"].is_file():
                raise FileNotFoundError(
                    f"Missing calibration audit for fold {fold}: {paths['audit']}. "
                    "Re-run this fold to audit its Watch calibration before aggregation."
                )
            prediction_paths.append(paths["predictions"])
        else:
            prediction_paths.append(run_fold(args, fold))
    result = aggregate(args, prediction_paths)
    metrics = result["scene_gated_watch_metrics"]
    print(json.dumps({
        "metrics": str(args.output_root / "aggregate_scene_gated_watch_metrics.json"),
        "watch_anomaly_override_count": result["watch_anomaly_override_count"],
        "macro_f1": metrics["macro_f1"],
        "far": metrics["far"],
        "abnormal_recall": metrics["abnormal_recall"],
        "anomaly_recall": metrics["per_class"]["anomaly"]["recall"],
        "direct_recall": metrics["per_class"]["direct"]["recall"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

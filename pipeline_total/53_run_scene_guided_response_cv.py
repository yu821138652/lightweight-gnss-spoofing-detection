#!/usr/bin/env python3
"""Run and aggregate the exploratory scene-guided response diagnosis CV.

For every requested outer fold this driver:

1. appends the frozen TCN scene posterior to the 44-D response tensor;
2. trains a 3-class conditioned response MLP and a binary direct expert;
3. calibrates/evaluates the response model on the held-out outer test split;
4. applies the scene-guided ``anomaly -> direct`` attribution rule; and
5. pools all outer-test rows into one cross-fold report.

The builder currently uses same-fold scene predictions for response-train rows,
so this runner is explicitly an exploratory six-fold validation.  A final paper
claim requires cross-fitted scene context for the response-training split.
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
    parser.add_argument(
        "--response-feature-dirname", type=str,
        default="device_tensors_sparse_initial30_cn0_extreme",
    )
    parser.add_argument(
        "--scene-tensors-root", type=Path,
        default=ROOT / "output" / "tensors" / "static_scene_response_fusion_s1_cn0_only" / "band_scene",
    )
    parser.add_argument(
        "--scene-training-root", type=Path,
        default=ROOT / "output" / "training" / "static_scene_response_fusion_s1_cn0_only" / "band_scene",
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=ROOT / "output" / "optimization" / "response_scene_guided_cv_v1",
    )
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 4, 5, 6, 7])
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--scene-encoder", choices=("tcn", "gru", "lstm"), default="tcn")
    parser.add_argument("--scene-confidence", type=float, default=0.5)
    parser.add_argument("--max-val-far", type=float, default=0.05)
    parser.add_argument("--min-val-abnormal-recall", type=float, default=0.90)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    if args.hidden_dim <= 0 or args.batch_size <= 0 or args.epochs <= 0 or args.patience < 0:
        parser.error("model/training sizes must be positive (patience may be zero)")
    if not 0 <= args.dropout < 1:
        parser.error("--dropout must be in [0, 1)")
    if not 0 <= args.scene_confidence <= 1:
        parser.error("--scene-confidence must be in [0, 1]")
    if not 0 <= args.max_val_far <= 1 or not 0 <= args.min_val_abnormal_recall <= 1:
        parser.error("validation constraints must be in [0, 1]")
    return args


def run(command: list[str]) -> None:
    print("run:", " ".join(command), flush=True)
    result = subprocess.run(command, cwd=str(ROOT))
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(command)}")


def metric_bundle(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    labels = [0, 1, 2]
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
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


def run_fold(args: argparse.Namespace, fold: int) -> Path:
    base_response = args.response_tensors_root / f"fold_{fold}" / args.response_feature_dirname
    scene_tensor = args.scene_tensors_root / f"fold_{fold}"
    scene_dir = args.scene_training_root / f"fold_{fold}"
    scene_checkpoint = scene_dir / f"best_band_mean_window_{args.scene_encoder}.pt"
    scene_predictions = scene_dir / f"test_predictions_band_mean_window_{args.scene_encoder}.csv"
    fold_root = args.output_root / f"fold_{fold}"
    tensor_dir = fold_root / "device_tensors_sceneprob"
    flat_dir = fold_root / f"mlp_sceneprob_h{args.hidden_dim}"
    direct_dir = fold_root / f"direct_expert_sceneprob_h{args.hidden_dim}"
    response_dir = fold_root / f"direct_override_sceneprob_h{args.hidden_dim}"
    rule_dir = fold_root / f"scene_guided_rule_t{int(args.scene_confidence * 100):02d}"
    rule_predictions = rule_dir / "test_scene_guided_response_predictions.csv"

    for path in (base_response, scene_tensor, scene_checkpoint, scene_predictions):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.skip_existing and rule_predictions.is_file():
        print(f"fold {fold}: reusing {rule_predictions}", flush=True)
        return rule_predictions
    python = str(args.python_exe)
    if not (args.skip_existing and (tensor_dir / "train.npz").is_file()):
        run([
            python, str(PIPELINE / "51_build_conditioned_response_tensors.py"),
            "--response-data-dir", str(base_response),
            "--scene-data-dir", str(scene_tensor),
            "--scene-checkpoint", str(scene_checkpoint),
            "--output-dir", str(tensor_dir), "--overwrite",
        ])
    flat_checkpoint = flat_dir / "best_device_event_mlp.pt"
    if not (args.skip_existing and flat_checkpoint.is_file()):
        run([
            python, str(PIPELINE / "37_train_device_attack_event.py"),
            "--data-dir", str(tensor_dir), "--output-dir", str(flat_dir),
            "--label-key", "y_response_state", "--num-classes", "3",
            "--model", "mlp", "--hidden-dim", str(args.hidden_dim),
            "--dropout", str(args.dropout), "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size), "--patience", str(args.patience),
        ])
    direct_checkpoint = direct_dir / "best_device_event_mlp.pt"
    if not (args.skip_existing and direct_checkpoint.is_file()):
        run([
            python, str(PIPELINE / "37_train_device_attack_event.py"),
            "--data-dir", str(tensor_dir), "--output-dir", str(direct_dir),
            "--label-key", "y_response_state", "--label-transform", "direct", "--num-classes", "2",
            "--model", "mlp", "--hidden-dim", str(args.hidden_dim),
            "--dropout", str(args.dropout), "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size), "--patience", str(args.patience),
        ])
    response_predictions = response_dir / "test_response_state_predictions.csv"
    if not (args.skip_existing and response_predictions.is_file()):
        run([
            python, str(PIPELINE / "42_eval_response_state_direct_override.py"),
            "--data-dir", str(tensor_dir), "--output-dir", str(response_dir), "--split", "test",
            "--flat-checkpoint", str(flat_checkpoint), "--direct-checkpoint", str(direct_checkpoint),
            "--calibrate-threshold-on-val",
            "--thresholds", "0.05", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "0.95",
            "--max-val-far", str(args.max_val_far),
            "--min-val-abnormal-recall", str(args.min_val_abnormal_recall),
            "--override-scope", "all", "--fold", str(fold),
            "--predictions-csv", str(response_predictions),
        ])
    run([
        python, str(PIPELINE / "52_eval_scene_guided_response_rule.py"),
        "--scene-predictions", str(scene_predictions),
        "--response-predictions", str(response_predictions),
        "--output-dir", str(rule_dir),
        "--min-scene-confidence", str(args.scene_confidence), "--overwrite",
    ])
    return rule_predictions


def aggregate(args: argparse.Namespace, prediction_paths: list[Path]) -> dict[str, Any]:
    frames = [pd.read_csv(path, encoding="utf-8-sig") for path in prediction_paths]
    combined = pd.concat(frames, ignore_index=True)
    required = {"fold", "device_name", "true_state", "pred_state", "guided_pred_state", "scene_context_available"}
    missing = sorted(required.difference(combined.columns))
    if missing:
        raise ValueError(f"Rule predictions missing columns {missing}")
    y_true = combined["true_state"].astype(int).to_numpy()
    base = combined["pred_state"].astype(int).to_numpy()
    guided = combined["guided_pred_state"].astype(int).to_numpy()
    result: dict[str, Any] = {
        "protocol": "static_time_block_outer_v2",
        "status": "exploratory; response-train scene context is not cross-fitted",
        "folds": [int(fold) for fold in args.folds],
        "configuration": {
            "response_features": "44-D C/N0-extreme + four scene posterior features + availability flag",
            "response_model": f"MLP-h{args.hidden_dim}",
            "scene_model": f"{args.scene_encoder} scene branch",
            "rule": "scene attack context + response anomaly -> direct",
            "min_scene_confidence": args.scene_confidence,
        },
        "scene_context_coverage": float(combined["scene_context_available"].astype(bool).mean()),
        "base_response_metrics": metric_bundle(y_true, base),
        "scene_guided_metrics": metric_bundle(y_true, guided),
        "by_fold": {},
        "by_device": {},
    }
    for fold, group in combined.groupby("fold", sort=True):
        result["by_fold"][str(int(fold))] = {
            "scene_context_coverage": float(group["scene_context_available"].astype(bool).mean()),
            "base": metric_bundle(group["true_state"].astype(int).to_numpy(), group["pred_state"].astype(int).to_numpy()),
            "guided": metric_bundle(group["true_state"].astype(int).to_numpy(), group["guided_pred_state"].astype(int).to_numpy()),
        }
    for device_name, group in combined.groupby("device_name", sort=True):
        result["by_device"][str(device_name)] = {
            "samples": int(len(group)),
            "scene_context_coverage": float(group["scene_context_available"].astype(bool).mean()),
            "guided": metric_bundle(group["true_state"].astype(int).to_numpy(), group["guided_pred_state"].astype(int).to_numpy()),
        }
    args.output_root.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output_root / "aggregate_scene_guided_response_predictions.csv", index=False, encoding="utf-8-sig")
    (args.output_root / "aggregate_scene_guided_response_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    args = parse_args()
    prediction_paths: list[Path] = []
    for fold in args.folds:
        path = args.output_root / f"fold_{fold}" / f"scene_guided_rule_t{int(args.scene_confidence * 100):02d}" / "test_scene_guided_response_predictions.csv"
        if args.aggregate_only:
            if not path.is_file():
                raise FileNotFoundError(path)
            prediction_paths.append(path)
        else:
            prediction_paths.append(run_fold(args, fold))
    result = aggregate(args, prediction_paths)
    guided = result["scene_guided_metrics"]
    print(json.dumps({
        "metrics": str(args.output_root / "aggregate_scene_guided_response_metrics.json"),
        "samples": guided["samples"],
        "scene_context_coverage": result["scene_context_coverage"],
        "macro_f1": guided["macro_f1"],
        "far": guided["far"],
        "abnormal_recall": guided["abnormal_recall"],
        "anomaly_recall": guided["per_class"]["anomaly"]["recall"],
        "direct_recall": guided["per_class"]["direct"]["recall"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

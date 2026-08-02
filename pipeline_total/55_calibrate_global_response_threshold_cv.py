"""Select one global direct-override threshold from pooled validation data."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "pipeline_total" / "37_train_device_attack_event.py"


def load_train_module() -> Any:
    spec = importlib.util.spec_from_file_location("device_event_train_global_threshold", TRAIN_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {TRAIN_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--feature-tag", required=True)
    parser.add_argument("--model", choices=("linear", "mlp"), default="mlp")
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 4, 5, 6, 7])
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    parser.add_argument("--max-val-far", type=float, default=0.01)
    parser.add_argument("--min-val-abnormal-recall", type=float, default=0.90)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def override(flat_probability: np.ndarray, direct_probability: np.ndarray, threshold: float) -> np.ndarray:
    predicted = flat_probability.argmax(axis=1).astype(np.int64)
    predicted[direct_probability[:, 1] >= threshold] = 2
    return predicted


def pooled_metrics(train_mod: Any, labels: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    result = train_mod.multiclass_metrics(labels, predicted, 3)
    result["anomaly_recall"] = result["per_class"]["1"]["recall"]
    result["direct_recall"] = result["per_class"]["2"]["recall"]
    return result


def main() -> None:
    args = parse_args()
    train_mod = load_train_module()
    device = __import__("torch").device("cuda" if __import__("torch").cuda.is_available() else "cpu")
    pooled: dict[str, dict[str, list[np.ndarray]]] = {
        "val": {"labels": [], "flat": [], "direct": []},
        "test": {"labels": [], "flat": [], "direct": []},
    }

    for fold in args.folds:
        fold_root = args.experiment_root / f"fold_{fold}"
        data_dir = fold_root / f"device_tensors_{args.feature_tag}"
        flat_path = fold_root / f"{args.model}_{args.feature_tag}_h{args.hidden_dim}" / f"best_device_event_{args.model}.pt"
        direct_path = fold_root / f"direct_expert_{args.feature_tag}_{args.model}_h{args.hidden_dim}" / f"best_device_event_{args.model}.pt"
        metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
        for split in ("val", "test"):
            data = train_mod.EventDataset(data_dir / f"{split}.npz", "y_response_state", None, "raw")
            _, flat_model = train_mod.load_checkpoint_model(flat_path, device, data.x.shape[1])
            _, direct_model = train_mod.load_checkpoint_model(direct_path, device, data.x.shape[1])
            pooled[split]["labels"].append(data.y.numpy())
            pooled[split]["flat"].append(train_mod.probabilities(flat_model, data, device, args.batch_size))
            pooled[split]["direct"].append(train_mod.probabilities(direct_model, data, device, args.batch_size))

    val_labels = np.concatenate(pooled["val"]["labels"])
    val_flat = np.concatenate(pooled["val"]["flat"])
    val_direct = np.concatenate(pooled["val"]["direct"])
    test_labels = np.concatenate(pooled["test"]["labels"])
    test_flat = np.concatenate(pooled["test"]["flat"])
    test_direct = np.concatenate(pooled["test"]["direct"])

    candidates = []
    for threshold in args.thresholds:
        val_pred = override(val_flat, val_direct, threshold)
        metrics = pooled_metrics(train_mod, val_labels, val_pred)
        candidates.append({"threshold": float(threshold), **metrics})
    eligible = [row for row in candidates if row["far"] <= args.max_val_far and row["abnormal_recall"] >= args.min_val_abnormal_recall]
    selected = max(eligible or candidates, key=lambda row: (row["macro_f1"], row["direct_recall"], row["abnormal_recall"]))
    test_pred = override(test_flat, test_direct, selected["threshold"])
    result = {
        "folds": args.folds,
        "feature_tag": args.feature_tag,
        "selected_threshold": selected["threshold"],
        "max_val_far": args.max_val_far,
        "min_val_abnormal_recall": args.min_val_abnormal_recall,
        "validation_selected": selected,
        "validation_candidates": candidates,
        "pooled_test": pooled_metrics(train_mod, test_labels, test_pred),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "global_threshold_cv_metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

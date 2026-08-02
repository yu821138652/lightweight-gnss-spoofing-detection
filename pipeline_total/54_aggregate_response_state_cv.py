"""Aggregate response-state prediction CSVs into one pooled CV report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames = [pd.read_csv(path, encoding="utf-8-sig") for path in args.predictions]
    if not frames:
        raise ValueError("No prediction CSVs supplied")
    data = pd.concat(frames, ignore_index=True)
    required = {"true_state", "pred_state"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Prediction CSVs are missing columns: {sorted(missing)}")

    y_true = data["true_state"].to_numpy(dtype=np.int64)
    y_pred = data["pred_state"].to_numpy(dtype=np.int64)
    labels = [0, 1, 2]
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    normal = y_true == 0
    abnormal = ~normal
    predicted_abnormal = y_pred != 0
    result = {
        "windows": int(len(data)),
        "folds": sorted(data["fold"].astype(str).unique().tolist()) if "fold" in data else [],
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "accuracy": float((y_true == y_pred).mean()),
        "far": float(((normal) & predicted_abnormal).sum() / max(int(normal.sum()), 1)),
        "abnormal_recall": float(((abnormal) & predicted_abnormal).sum() / max(int(abnormal.sum()), 1)),
        "anomaly_recall": float(recall[1]),
        "direct_recall": float(recall[2]),
        "confusion_matrix": matrix.tolist(),
        "per_class": {
            str(label): {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(labels)
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output_dir / "pooled_response_state_predictions.csv", index=False, encoding="utf-8-sig")
    (args.output_dir / "pooled_response_state_metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

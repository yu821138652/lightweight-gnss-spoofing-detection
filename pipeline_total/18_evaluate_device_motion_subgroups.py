"""评估设备级 LightGBM 在静态、动态测试窗口上的分组指标。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


def load_test_split(data_dir: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    with np.load(data_dir / "test.npz") as data:
        x = data["x"].astype(np.float32)
        y = data["y"].astype(np.int64)
    metadata = pd.read_csv(data_dir / "test_metadata.csv", encoding="utf-8-sig")
    if x.ndim != 3 or y.ndim != 1 or len(x) != len(y) or len(y) != len(metadata):
        raise ValueError(f"测试张量和元数据不一致：x={x.shape}, y={y.shape}, metadata={len(metadata)}")
    return x.reshape(len(x), -1), y, metadata


def evaluate(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float | int]:
    y_pred = (probabilities >= 0.5).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "far": float(fp / (fp + tn)) if fp + tn else 0.0,
        "samples": int(len(y_true)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    model_path = args.model_dir / "best_device_lightgbm.txt"
    for path in (args.data_dir / "test.npz", args.data_dir / "test_metadata.csv", model_path):
        if not path.exists():
            raise FileNotFoundError(path)
    x_test, y_test, metadata = load_test_split(args.data_dir)
    probabilities = lgb.Booster(model_file=str(model_path)).predict(x_test)
    is_dynamic = metadata["Scenario"].astype(str).str.startswith("dy_").to_numpy()

    result: dict[str, dict[str, float | int]] = {"overall": evaluate(y_test, probabilities)}
    for name, mask in (("static", ~is_dynamic), ("dynamic", is_dynamic)):
        if not mask.any():
            raise ValueError(f"测试集中没有 {name} 窗口。")
        result[name] = evaluate(y_test[mask], probabilities[mask])
    output_path = args.output or args.model_dir / "test_metrics_device_lightgbm_by_motion.json"
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

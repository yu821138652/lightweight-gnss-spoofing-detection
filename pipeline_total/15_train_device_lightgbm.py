"""Train a constrained LightGBM device-level spoofing detection baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

try:
    import lightgbm as lgb
except ImportError as error:  # pragma: no cover - dependency error is user-facing
    raise SystemExit("LightGBM is required. Install it with: python -m pip install lightgbm") from error


def load_split(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        x = data["x"].astype(np.float32)
        y = data["y"].astype(np.int64)
    if x.ndim != 3 or y.ndim != 1 or len(x) != len(y):
        raise ValueError(f"Invalid device tensor in {path}: x={x.shape}, y={y.shape}")
    return x.reshape(len(x), -1), y


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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-leaves", type=int, default=15)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--early-stopping-rounds", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--test-only", action="store_true")
    args = parser.parse_args()
    if args.dry_run and args.test_only:
        parser.error("--dry-run cannot be combined with --test-only")

    train_path = args.data_dir / "train.npz"
    val_path = args.data_dir / "val.npz"
    for path in (train_path, val_path):
        if not path.exists():
            raise FileNotFoundError(path)
    x_train, y_train = load_split(train_path)
    x_val, y_val = load_split(val_path)
    if x_train.shape[1] != x_val.shape[1]:
        raise ValueError("Train and validation feature dimensions differ.")
    if args.dry_run:
        print(json.dumps({
            "model": "lightgbm",
            "input_features": int(x_train.shape[1]),
            "train_samples": int(len(y_train)),
            "val_samples": int(len(y_val)),
            "train_positive": int(y_train.sum()),
            "val_positive": int(y_val.sum()),
            "test_read": False,
        }, ensure_ascii=False))
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "best_device_lightgbm.txt"
    if args.test_only:
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        test_path = args.data_dir / "test.npz"
        if not test_path.exists():
            raise FileNotFoundError(test_path)
        x_test, y_test = load_split(test_path)
        booster = lgb.Booster(model_file=str(model_path))
        metrics = evaluate(y_test, booster.predict(x_test))
        metrics["model_size_bytes"] = model_path.stat().st_size
        (args.output_dir / "test_metrics_device_lightgbm.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        print(json.dumps({"split": "test", **metrics}, ensure_ascii=False))
        return

    positive_count = max(int(y_train.sum()), 1)
    negative_count = max(int(len(y_train) - y_train.sum()), 1)
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        max_depth=args.max_depth,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        scale_pos_weight=negative_count / positive_count,
        random_state=args.seed,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(args.early_stopping_rounds, verbose=False)],
    )
    model.booster_.save_model(str(model_path), num_iteration=model.best_iteration_)
    metrics = evaluate(y_val, model.predict_proba(x_val)[:, 1])
    metrics.update({
        "best_iteration": int(model.best_iteration_ or args.n_estimators),
        "model_size_bytes": model_path.stat().st_size,
        "input_features": int(x_train.shape[1]),
    })
    (args.output_dir / "val_metrics_device_lightgbm.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps({"split": "val", **metrics}, ensure_ascii=False))


if __name__ == "__main__":
    main()

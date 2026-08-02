"""Aggregate band-mean four-way scene classification across static CV folds.

The static leave-one-recording-out folds each hold out a single scene type, so
no individual fold's test split contains all four classes.  This driver builds
per-fold band-mean tensors, trains one model per fold, exports each fold's
per-window test predictions, and finally stacks every fold's held-out test
predictions into one true cross-recording 4x4 confusion matrix.

For each fold it runs, in order:

    1. ``45_build_band_mean_window_tensors.py``  -> per-fold tensors
    2. ``46_train_band_mean_multiclass.py``      -> trained checkpoint
    3. ``46_train_band_mean_multiclass.py --test-only`` -> test predictions CSV

then concatenates the exported ``test_predictions_*`` rows and reports the
aggregate confusion matrix plus per-class precision/recall/F1.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "pipeline_total"
NUM_CLASSES = 4
CLASS_NAMES = {0: "normal", 1: "L1", 2: "L5", 3: "L1+L5"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
LOG = logging.getLogger(__name__)


def run(cmd: list[str]) -> None:
    LOG.info("run: %s", " ".join(str(part) for part in cmd))
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(str(p) for p in cmd)}")


def fold_ids(protocol_dir: Path) -> list[int]:
    ids = sorted(
        int(path.name.split("_")[1])
        for path in protocol_dir.glob("fold_*")
        if path.is_dir() and path.name.split("_")[1].isdigit()
    )
    if not ids:
        raise ValueError(f"No fold_* directories under {protocol_dir}")
    return ids


def build_and_train_fold(
    fold: int,
    protocol_dir: Path,
    tensors_root: Path,
    training_root: Path,
    csv_path: Path,
    config_path: Path,
    encoder: str,
    hidden_dim: int,
    dropout: float,
    epochs: int,
    seed: int,
    skip_existing: bool,
    class_weight_mult: list[float] | None = None,
    drop_features: list[str] | None = None,
    scope: str = "static",
    include_pseudorange_rate: bool = False,
    include_state_adr: bool = False,
    include_pseudorange_residual: bool = False,
    include_cross_band: bool = False,
) -> Path:
    fold_protocol = protocol_dir / f"fold_{fold}"
    epoch_manifest = fold_protocol / "epoch_split_manifest.csv"
    outer_manifest = fold_protocol / "recording_split_manifest.csv"
    for path in (epoch_manifest, outer_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    tensor_dir = tensors_root / f"fold_{fold}"
    output_dir = training_root / f"fold_{fold}"
    predictions_path = output_dir / f"test_predictions_band_mean_window_{encoder}.csv"

    if skip_existing and predictions_path.is_file():
        LOG.info("fold %d: reusing existing %s", fold, predictions_path)
        return predictions_path

    if not (skip_existing and (tensor_dir / "train.npz").is_file()):
        build_cmd = [
            sys.executable, str(PIPELINE / "45_build_band_mean_window_tensors.py"),
            "--csv", str(csv_path),
            "--epoch-manifest", str(epoch_manifest),
            "--outer-manifest", str(outer_manifest),
            "--config", str(config_path),
            "--output-dir", str(tensor_dir),
            "--scope", scope,
        ]
        if include_pseudorange_rate:
            build_cmd.append("--include-pseudorange-rate")
        if include_state_adr:
            build_cmd.append("--include-state-adr")
        if include_pseudorange_residual:
            build_cmd.append("--include-pseudorange-residual")
        if include_cross_band:
            build_cmd.append("--include-cross-band")
        run(build_cmd)

    checkpoint = output_dir / f"best_band_mean_window_{encoder}.pt"
    if not (skip_existing and checkpoint.is_file()):
        train_cmd = [
            sys.executable, str(PIPELINE / "46_train_band_mean_multiclass.py"),
            "--data-dir", str(tensor_dir),
            "--output-dir", str(output_dir),
            "--encoder", encoder,
            "--hidden-dim", str(hidden_dim),
            "--dropout", str(dropout),
            "--epochs", str(epochs),
            "--seed", str(seed),
        ]
        if class_weight_mult is not None:
            train_cmd += ["--class-weight-mult", *[str(v) for v in class_weight_mult]]
        if drop_features:
            train_cmd += ["--drop-features", *drop_features]
        run(train_cmd)

    run([
        sys.executable, str(PIPELINE / "46_train_band_mean_multiclass.py"),
        "--data-dir", str(tensor_dir),
        "--output-dir", str(output_dir),
        "--encoder", encoder,
        "--test-only",
        "--fold", str(fold),
    ])
    if not predictions_path.is_file():
        raise FileNotFoundError(f"fold {fold} did not produce {predictions_path}")
    return predictions_path


def aggregate(prediction_paths: list[Path], output_dir: Path) -> dict:
    frames = [pd.read_csv(path, encoding="utf-8-sig") for path in prediction_paths]
    combined = pd.concat(frames, ignore_index=True)
    if combined.empty:
        raise ValueError("No test predictions to aggregate")
    y_true = combined["true_class"].to_numpy(dtype=np.int64)
    y_pred = combined["pred_class"].to_numpy(dtype=np.int64)
    labels = list(range(NUM_CLASSES))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = {
        CLASS_NAMES[c]: {
            "precision": float(precision[c]),
            "recall": float(recall[c]),
            "f1": float(f1[c]),
            "support": int(support[c]),
        }
        for c in labels
    }
    per_fold = {
        str(int(fold)): {
            CLASS_NAMES[c]: int((group["true_class"].to_numpy() == c).sum())
            for c in labels
        }
        for fold, group in combined.groupby("fold")
    }
    result = {
        "windows": int(len(combined)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "accuracy": float((y_true == y_pred).mean()),
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_labels": [CLASS_NAMES[c] for c in labels],
        "per_class": per_class,
        "class_support": {CLASS_NAMES[c]: int(support[c]) for c in labels},
        "per_fold_true_class_counts": per_fold,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_dir / "aggregate_test_predictions.csv", index=False, encoding="utf-8-sig")
    (output_dir / "aggregate_test_metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol-dir", type=Path,
        default=ROOT / "output" / "protocols" / "static_time_block_outer_v2",
    )
    parser.add_argument(
        "--tensors-root", type=Path,
        default=ROOT / "output" / "tensors" / "band_mean_window_static_v1",
    )
    parser.add_argument(
        "--training-root", type=Path,
        default=ROOT / "output" / "training" / "band_mean_multiclass_cv",
    )
    parser.add_argument("--csv", type=Path, default=ROOT / "output" / "processed_gnss_data.csv")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "preprocessing.yml")
    parser.add_argument("--encoder", choices=("lstm", "gru", "tcn"), default="tcn")
    parser.add_argument("--hidden-dim", type=int, default=32,
                        help="Hidden/channel dimension passed to the trainer.")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout passed to the trainer (must be in [0, 1)).")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--folds", type=int, nargs="*", default=None, help="Subset of fold ids; default all.")
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Reuse tensors/checkpoints/predictions already present instead of recomputing.",
    )
    parser.add_argument(
        "--aggregate-only", action="store_true",
        help="Skip build/train; only aggregate existing per-fold test-prediction CSVs.",
    )
    parser.add_argument(
        "--class-weight-mult", type=float, nargs=NUM_CLASSES, default=None,
        help=f"Per-class loss weight multipliers ({NUM_CLASSES} values: normal L1 L5 L1+L5), passed to the trainer.",
    )
    parser.add_argument(
        "--drop-features", type=str, nargs="+", default=None,
        help="Feature ablation names passed through to the trainer (e.g. AgcDb).",
    )
    parser.add_argument(
        "--scope", choices=("static", "dynamic", "all"), default="static",
        help="Recording scope passed to the tensor builder: static (st_*), dynamic (dy_*), or all.",
    )
    parser.add_argument(
        "--include-pseudorange-rate", action="store_true",
        help="Pass --include-pseudorange-rate to the band-mean tensor builder.",
    )
    parser.add_argument(
        "--include-state-adr", action="store_true",
        help="Pass --include-state-adr to the band-mean tensor builder.",
    )
    parser.add_argument(
        "--include-pseudorange-residual", action="store_true",
        help="Pass --include-pseudorange-residual to the band-mean tensor builder.",
    )
    parser.add_argument(
        "--include-cross-band", action="store_true",
        help="Pass --include-cross-band to the band-mean tensor builder.",
    )
    args = parser.parse_args()
    if args.hidden_dim <= 0:
        parser.error("--hidden-dim must be positive")
    if not 0.0 <= args.dropout < 1.0:
        parser.error("--dropout must be in [0, 1)")
    return args


def main() -> None:
    args = parse_args()
    folds = args.folds or fold_ids(args.protocol_dir)
    LOG.info("folds=%s encoder=%s hidden_dim=%d dropout=%.3f",
             folds, args.encoder, args.hidden_dim, args.dropout)

    prediction_paths: list[Path] = []
    for fold in folds:
        predictions_path = args.training_root / f"fold_{fold}" / f"test_predictions_band_mean_window_{args.encoder}.csv"
        if args.aggregate_only:
            if not predictions_path.is_file():
                raise FileNotFoundError(f"--aggregate-only but missing {predictions_path}")
            prediction_paths.append(predictions_path)
            continue
        prediction_paths.append(
            build_and_train_fold(
                fold, args.protocol_dir, args.tensors_root, args.training_root,
                args.csv, args.config, args.encoder, args.hidden_dim, args.dropout,
                args.epochs, args.seed,
                args.skip_existing, args.class_weight_mult, args.drop_features,
                args.scope, args.include_pseudorange_rate, args.include_state_adr,
                args.include_pseudorange_residual,
                args.include_cross_band,
            )
        )

    result = aggregate(prediction_paths, args.training_root)
    LOG.info("aggregate windows=%d macro_f1=%.4f accuracy=%.4f",
             result["windows"], result["macro_f1"], result["accuracy"])
    LOG.info("aggregate test confusion matrix (rows=true, cols=pred) labels=%s",
             result["confusion_matrix_labels"])
    for label, row in zip(result["confusion_matrix_labels"], result["confusion_matrix"]):
        LOG.info("  %-7s %s", label, row)
    for name, stats in result["per_class"].items():
        LOG.info("  %-7s P=%.3f R=%.3f F1=%.3f support=%d",
                 name, stats["precision"], stats["recall"], stats["f1"], stats["support"])


if __name__ == "__main__":
    main()

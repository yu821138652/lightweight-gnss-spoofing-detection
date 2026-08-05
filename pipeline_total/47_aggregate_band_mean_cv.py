"""Aggregate band-mean four-way scene classification across recording CV folds.

An individual recording-held-out fold can omit one or more scene classes.  This
driver builds per-fold band-mean tensors, trains one model per fold, exports
each fold's per-window test predictions, and finally stacks the selected held-
out predictions into one fixed-label 4x4 confusion matrix.

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
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "pipeline_total"
NUM_CLASSES = 4
CLASS_NAMES = {0: "normal", 1: "L1", 2: "L5", 3: "L1+L5"}
STAGE_MARKER_PREFIX = "aggregate_cv_stage"

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


def file_fingerprint(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def stage_matches(marker: Path, config: dict, required: list[Path]) -> bool:
    if not marker.is_file() or any(not path.is_file() for path in required):
        return False
    try:
        stored = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if stored.get("config") != config:
        return False
    expected_artifacts = stored.get("artifact_fingerprints")
    if not isinstance(expected_artifacts, dict):
        return False
    actual_artifacts = {
        str(path.resolve()): file_fingerprint(path)
        for path in required
    }
    return expected_artifacts == actual_artifacts


def write_stage_marker(
    marker: Path, config: dict, command: list[str], artifacts: list[Path]
) -> None:
    missing = [str(path) for path in artifacts if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Stage did not create required artifacts: {missing}")
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps(
            {
                "config": config,
                "command": [str(part) for part in command],
                "artifacts": [str(path) for path in artifacts],
                "artifact_fingerprints": {
                    str(path.resolve()): file_fingerprint(path)
                    for path in artifacts
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def integer_series(values: pd.Series, context: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="raise")
    array = numeric.to_numpy(dtype=np.float64)
    if not np.isfinite(array).all() or not np.equal(array, np.floor(array)).all():
        raise ValueError(f"{context} must contain finite integers")
    return numeric.astype(np.int64)


def load_fold_predictions(path: Path, expected_fold: int) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = {"fold", "true_class", "pred_class"}
    if missing := required.difference(frame.columns):
        raise ValueError(f"Prediction CSV {path} missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"Prediction CSV {path} is empty")
    for column in ("fold", "true_class", "pred_class"):
        frame[column] = integer_series(frame[column], f"{path} column {column}")
    observed_folds = sorted(int(value) for value in frame["fold"].unique())
    if observed_folds != [expected_fold]:
        raise ValueError(
            f"Prediction CSV {path} contains folds {observed_folds}, expected [{expected_fold}]"
        )
    for column in ("true_class", "pred_class"):
        invalid = sorted(set(frame[column].unique()).difference(range(NUM_CLASSES)))
        if invalid:
            raise ValueError(f"Prediction CSV {path} has invalid {column}: {invalid}")

    source_columns = {"source_id", "window_time_nanos"}
    present_source = source_columns.intersection(frame.columns)
    if present_source and present_source != source_columns:
        raise ValueError(
            f"Prediction CSV {path} must contain both source_id and window_time_nanos"
        )
    if present_source:
        if "device_id" not in frame.columns:
            raise ValueError(f"Prediction CSV {path} missing device_id for strict endpoint keys")
        for column in ("source_id", "device_id", "window_time_nanos"):
            frame[column] = integer_series(frame[column], f"{path} column {column}")
        key = ["fold", "source_id", "device_id", "window_time_nanos"]
        duplicate = frame.duplicated(key, keep=False)
        if duplicate.any():
            sample = frame.loc[duplicate, key].head(5).to_dict("records")
            raise ValueError(f"Prediction CSV {path} has duplicate endpoint keys: {sample}")
        frame.attrs["strict_prediction_key"] = key
    else:
        frame.attrs["strict_prediction_key"] = None
        LOG.warning(
            "legacy predictions lack source_id/window_time_nanos; endpoint uniqueness "
            "cannot be audited: %s",
            path,
        )
    return frame


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
    scaler_mode: str = "per_device",
    include_pseudorange_rate: bool = False,
    include_state_adr: bool = False,
    include_pseudorange_residual: bool = False,
    include_cross_band: bool = False,
    include_cn0_dynamics: bool = False,
    include_paired_pseudorange_rate: bool = False,
    paired_pseudorange_rate_reference_min_pairs: int = 256,
    causal_baseline_mode: str = "none",
    causal_half_life_seconds: float = 60.0,
    normal_reference_mode: str = "none",
    normal_reference_minimum_epochs: int = 1,
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
    builder_script = PIPELINE / "45_build_band_mean_window_tensors.py"
    trainer_script = PIPELINE / "46_train_band_mean_multiclass.py"
    model_script = ROOT / "models" / "gnss_signal_baselines.py"
    build_cmd = [
        sys.executable, str(builder_script),
        "--csv", str(csv_path),
        "--epoch-manifest", str(epoch_manifest),
        "--outer-manifest", str(outer_manifest),
        "--config", str(config_path),
        "--output-dir", str(tensor_dir),
        "--scope", scope,
        "--scaler-mode", scaler_mode,
        "--causal-baseline-mode", causal_baseline_mode,
        "--causal-half-life-seconds", str(causal_half_life_seconds),
        "--normal-reference-mode", normal_reference_mode,
        "--normal-reference-min-epochs", str(normal_reference_minimum_epochs),
    ]
    if include_pseudorange_rate:
        build_cmd.append("--include-pseudorange-rate")
    if include_state_adr:
        build_cmd.append("--include-state-adr")
    if include_pseudorange_residual:
        build_cmd.append("--include-pseudorange-residual")
    if include_cross_band:
        build_cmd.append("--include-cross-band")
    if include_cn0_dynamics:
        build_cmd.append("--include-cn0-dynamics")
    if include_paired_pseudorange_rate:
        build_cmd.extend([
            "--include-paired-pseudorange-rate",
            "--paired-pseudorange-rate-reference-min-pairs",
            str(paired_pseudorange_rate_reference_min_pairs),
        ])
    build_required = [
        tensor_dir / "train.npz",
        tensor_dir / "val.npz",
        tensor_dir / "test.npz",
        tensor_dir / "feature_names.json",
        tensor_dir / "scaler.json",
        tensor_dir / "normal_reference.json",
        tensor_dir / "tensor_metadata.json",
        tensor_dir / "device_mapping.json",
        tensor_dir / "source_mapping.json",
    ]
    if include_paired_pseudorange_rate:
        build_required.append(tensor_dir / "paired_pseudorange_rate_reference.json")
    build_config = {
        "fold": fold,
        "inputs": {
            "csv": file_fingerprint(csv_path),
            "config": file_fingerprint(config_path),
            "epoch_manifest": file_fingerprint(epoch_manifest),
            "outer_manifest": file_fingerprint(outer_manifest),
            "builder_script": file_fingerprint(builder_script),
        },
        "scope": scope,
        "scaler_mode": scaler_mode,
        "include_pseudorange_rate": include_pseudorange_rate,
        "include_state_adr": include_state_adr,
        "include_pseudorange_residual": include_pseudorange_residual,
        "include_cross_band": include_cross_band,
        "include_cn0_dynamics": include_cn0_dynamics,
        "include_paired_pseudorange_rate": include_paired_pseudorange_rate,
        "paired_pseudorange_rate_reference_min_pairs": int(
            paired_pseudorange_rate_reference_min_pairs
        ),
        "causal_baseline_mode": causal_baseline_mode,
        "causal_half_life_seconds": float(causal_half_life_seconds),
        "normal_reference_mode": normal_reference_mode,
        "normal_reference_minimum_epochs": int(normal_reference_minimum_epochs),
    }
    build_marker = tensor_dir / f"{STAGE_MARKER_PREFIX}_build.json"
    if skip_existing and stage_matches(build_marker, build_config, build_required):
        LOG.info("fold %d: reusing config-matched tensors", fold)
    else:
        run(build_cmd)
        write_stage_marker(build_marker, build_config, build_cmd, build_required)

    checkpoint = output_dir / f"best_band_mean_window_{encoder}.pt"
    metrics_path = output_dir / f"val_metrics_band_mean_window_{encoder}.json"
    train_cmd = [
        sys.executable, str(trainer_script),
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
    train_required = [checkpoint, metrics_path]
    train_config = {
        "fold": fold,
        "tensor_artifacts": {
            path.name: file_fingerprint(path)
            for path in build_required
        },
        "trainer_script": file_fingerprint(trainer_script),
        "model_script": file_fingerprint(model_script),
        "encoder": encoder,
        "hidden_dim": hidden_dim,
        "dropout": float(dropout),
        "epochs": epochs,
        "seed": seed,
        "class_weight_mult": list(class_weight_mult) if class_weight_mult is not None else None,
        "drop_features": sorted(drop_features or []),
    }
    train_marker = output_dir / f"{STAGE_MARKER_PREFIX}_train.json"
    if skip_existing and stage_matches(train_marker, train_config, train_required):
        LOG.info("fold %d: reusing config-matched checkpoint", fold)
    else:
        run(train_cmd)

        write_stage_marker(train_marker, train_config, train_cmd, train_required)

    predict_cmd = [
        sys.executable, str(trainer_script),
        "--data-dir", str(tensor_dir),
        "--output-dir", str(output_dir),
        "--encoder", encoder,
        "--test-only",
        "--fold", str(fold),
    ]
    predict_required = [predictions_path]
    predict_config = {
        "fold": fold,
        "checkpoint": file_fingerprint(checkpoint),
        "test_tensor": file_fingerprint(tensor_dir / "test.npz"),
        "feature_names": file_fingerprint(tensor_dir / "feature_names.json"),
        "tensor_metadata": file_fingerprint(tensor_dir / "tensor_metadata.json"),
        "trainer_script": file_fingerprint(trainer_script),
        "model_script": file_fingerprint(model_script),
    }
    predict_marker = output_dir / f"{STAGE_MARKER_PREFIX}_predict.json"
    if skip_existing and stage_matches(predict_marker, predict_config, predict_required):
        LOG.info("fold %d: reusing config-matched predictions", fold)
    else:
        run(predict_cmd)
        write_stage_marker(predict_marker, predict_config, predict_cmd, predict_required)
    return predictions_path


def aggregate(
    prediction_paths: list[Path], output_dir: Path, expected_folds: list[int]
) -> dict:
    if len(prediction_paths) != len(expected_folds):
        raise ValueError(
            f"Prediction paths/folds length mismatch: {len(prediction_paths)} vs {len(expected_folds)}"
        )
    frames = [
        load_fold_predictions(path, fold)
        for path, fold in zip(prediction_paths, expected_folds)
    ]
    combined = pd.concat(frames, ignore_index=True)
    if combined.empty:
        raise ValueError("No test predictions to aggregate")
    if all(frame.attrs.get("strict_prediction_key") for frame in frames):
        key = ["fold", "source_id", "device_id", "window_time_nanos"]
        duplicate = combined.duplicated(key, keep=False)
        if duplicate.any():
            sample = combined.loc[duplicate, key].head(5).to_dict("records")
            raise ValueError(f"Cross-fold predictions contain duplicate endpoint keys: {sample}")
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
        "macro_f1": float(np.mean(f1)),
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
    parser.add_argument(
        "--aggregate-output-dir", type=Path, default=None,
        help=(
            "Directory for aggregate CSV/JSON. Full-fold runs default to "
            "--training-root; partial-fold runs default to a _subset_folds_* child."
        ),
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
        help="Reuse only artifacts whose stage marker and exact configuration match.",
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
        "--scaler-mode", choices=("per_device", "global"), default="per_device",
        help="Train-only standardization passed to the tensor builder.",
    )
    parser.add_argument(
        "--normal-reference-mode",
        choices=("none", "train_normal_band_mean"),
        default="none",
        help=(
            "Frozen train-only normal C/N0 mean per known device/band, with "
            "fold-global fallback; requires --scaler-mode global."
        ),
    )
    parser.add_argument(
        "--normal-reference-min-epochs",
        type=int,
        default=1,
        help="Minimum train-only normal epochs before a device/band uses the global fallback.",
    )
    parser.add_argument(
        "--causal-baseline-mode", choices=("none", "ema"), default="none",
        help="Optional label-independent causal C/N0 reference passed to the builder.",
    )
    parser.add_argument(
        "--causal-half-life-seconds", type=float, default=60.0,
        help="EMA half-life passed to the causal tensor builder.",
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
    parser.add_argument(
        "--include-cn0-dynamics", action="store_true",
        help="Pass --include-cn0-dynamics to the band-mean tensor builder.",
    )
    parser.add_argument(
        "--include-paired-pseudorange-rate", action="store_true",
        help="Pass same-satellite L1/L5 pseudorange-rate residual features to the tensor builder.",
    )
    parser.add_argument(
        "--paired-pseudorange-rate-reference-min-pairs",
        type=int,
        default=256,
        help="Minimum train-normal pair count for a device/constellation reference.",
    )
    args = parser.parse_args()
    if args.hidden_dim <= 0:
        parser.error("--hidden-dim must be positive")
    if not 0.0 <= args.dropout < 1.0:
        parser.error("--dropout must be in [0, 1)")
    if args.causal_half_life_seconds <= 0:
        parser.error("--causal-half-life-seconds must be positive")
    if args.causal_baseline_mode != "none" and args.scaler_mode != "global":
        parser.error("causal baseline modes require --scaler-mode global")
    if args.normal_reference_min_epochs < 1:
        parser.error("--normal-reference-min-epochs must be at least 1")
    if args.paired_pseudorange_rate_reference_min_pairs < 1:
        parser.error("--paired-pseudorange-rate-reference-min-pairs must be at least 1")
    if args.normal_reference_mode != "none" and args.scaler_mode != "global":
        parser.error("normal reference mode requires --scaler-mode global")
    if args.normal_reference_mode != "none" and args.causal_baseline_mode != "none":
        parser.error("normal reference and causal baseline modes are mutually exclusive")
    return args


def main() -> None:
    args = parse_args()
    available_folds = fold_ids(args.protocol_dir)
    folds = args.folds or available_folds
    if len(folds) != len(set(folds)):
        raise ValueError(f"Duplicate fold ids are not allowed: {folds}")
    invalid = sorted(set(folds).difference(available_folds))
    if invalid:
        raise ValueError(
            f"Requested folds are absent from {args.protocol_dir}: {invalid}"
        )
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
                args.scope, args.scaler_mode,
                args.include_pseudorange_rate, args.include_state_adr,
                args.include_pseudorange_residual,
                args.include_cross_band,
                args.include_cn0_dynamics,
                args.include_paired_pseudorange_rate,
                args.paired_pseudorange_rate_reference_min_pairs,
                args.causal_baseline_mode,
                args.causal_half_life_seconds,
                args.normal_reference_mode,
                args.normal_reference_min_epochs,
            )
        )

    if args.aggregate_output_dir is not None:
        aggregate_output_dir = args.aggregate_output_dir
    elif set(folds) == set(available_folds):
        aggregate_output_dir = args.training_root
    else:
        fold_tag = "_".join(str(fold) for fold in folds)
        aggregate_output_dir = args.training_root / f"_subset_folds_{fold_tag}"
    result = aggregate(prediction_paths, aggregate_output_dir, folds)
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

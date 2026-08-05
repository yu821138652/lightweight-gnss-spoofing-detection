"""Run the exact self-updating C/N0-baseline scene-classification protocol.

For every outer fold this driver builds fresh global-scaled absolute tensors,
trains a four-feature absolute-C/N0 warm-start model, then trains and evaluates
``59_train_online_cn0_baseline.py``.  The online model alone controls its
L1/L5 baseline updates; no labels or external gate predictions enter the state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "pipeline_total"
NUM_CLASSES = 4
ABSOLUTE_DROP_FEATURES = [
    "AgcDb",
    "ReceivedSvTimeUncertaintyNanos",
    "PseudorangeRateUncertaintyMetersPerSecond",
    "Cn0DbHzL1MinusL5",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
LOG = logging.getLogger(__name__)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def stage_matches(marker: Path, config: dict, artifacts: list[Path]) -> bool:
    if not marker.is_file() or not all(path.is_file() for path in artifacts):
        return False
    try:
        saved = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if saved.get("config") != config:
        return False
    expected = saved.get("artifacts")
    if not isinstance(expected, dict):
        return False
    return all(expected.get(str(path.resolve())) == sha256_file(path) for path in artifacts)


def write_stage(marker: Path, config: dict, command: list[str], artifacts: list[Path]) -> None:
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "config": config,
                "command": command,
                "artifacts": {str(path.resolve()): sha256_file(path) for path in artifacts},
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def run(command: list[str], dry_run: bool) -> None:
    LOG.info("$ %s", subprocess.list2cmdline(command))
    if not dry_run:
        subprocess.run(command, check=True)


def scope_tag(scope: str) -> str:
    return "mixed" if scope == "all" else scope


def fold_ids(protocol_dir: Path) -> list[int]:
    result = []
    for directory in protocol_dir.glob("fold_*"):
        try:
            fold = int(directory.name[len("fold_"):])
        except ValueError:
            continue
        if (directory / "epoch_split_manifest.csv").is_file() and (directory / "recording_split_manifest.csv").is_file():
            result.append(fold)
    if not result:
        raise ValueError(f"No fold_N protocol directories in {protocol_dir}")
    return sorted(result)


def builder_command(args: argparse.Namespace, fold: int, tensor_dir: Path) -> list[str]:
    protocol = args.protocol_dir / f"fold_{fold}"
    return [
        sys.executable, str(PIPELINE / "45_build_band_mean_window_tensors.py"),
        "--csv", str(args.csv),
        "--epoch-manifest", str(protocol / "epoch_split_manifest.csv"),
        "--outer-manifest", str(protocol / "recording_split_manifest.csv"),
        "--config", str(args.config),
        "--output-dir", str(tensor_dir),
        "--scope", args.scope,
        "--scaler-mode", "global",
        "--causal-baseline-mode", "none",
    ]


def absolute_train_command(args: argparse.Namespace, tensor_dir: Path, output_dir: Path) -> list[str]:
    command = [
        sys.executable, str(PIPELINE / "46_train_band_mean_multiclass.py"),
        "--data-dir", str(tensor_dir),
        "--output-dir", str(output_dir),
        "--encoder", args.encoder,
        "--hidden-dim", str(args.hidden_dim),
        "--dropout", str(args.dropout),
        "--epochs", str(args.base_epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--weight-decay", str(args.weight_decay),
        "--patience", str(args.patience),
        "--seed", str(args.seed),
        "--drop-features", *ABSOLUTE_DROP_FEATURES,
    ]
    if args.class_weight_mult is not None:
        command += ["--class-weight-mult", *[str(value) for value in args.class_weight_mult]]
    return command


def online_train_command(
    args: argparse.Namespace, tensor_dir: Path, output_dir: Path, warm_start: Path | None
) -> list[str]:
    command = [
        sys.executable, str(PIPELINE / "59_train_online_cn0_baseline.py"),
        "--data-dir", str(tensor_dir),
        "--output-dir", str(output_dir),
        "--encoder", args.encoder,
        "--hidden-dim", str(args.hidden_dim),
        "--dropout", str(args.dropout),
        "--epochs", str(args.epochs),
        "--lr", str(args.online_lr),
        "--weight-decay", str(args.weight_decay),
        "--patience", str(args.patience),
        "--seed", str(args.seed),
        "--alpha", str(args.alpha),
        "--rollout-batch-size", str(args.rollout_batch_size),
        "--optimizer-step-interval", str(args.optimizer_step_interval),
    ]
    if warm_start is not None:
        command += ["--init-checkpoint", str(warm_start)]
    if args.class_weight_mult is not None:
        command += ["--class-weight-mult", *[str(value) for value in args.class_weight_mult]]
    return command


def online_predict_command(
    args: argparse.Namespace, fold: int, tensor_dir: Path, output_dir: Path, checkpoint: Path
) -> list[str]:
    return [
        sys.executable, str(PIPELINE / "59_train_online_cn0_baseline.py"),
        "--data-dir", str(tensor_dir),
        "--output-dir", str(output_dir),
        "--encoder", args.encoder,
        "--test-only",
        "--checkpoint", str(checkpoint),
        "--predict-split", "test",
        "--fold", str(fold),
    ]


def stage(
    marker: Path,
    config: dict,
    command: list[str],
    artifacts: list[Path],
    resume: bool,
    dry_run: bool,
) -> None:
    if resume and stage_matches(marker, config, artifacts):
        LOG.info("reusing config-matched stage: %s", marker.parent)
        return
    run(command, dry_run)
    if not dry_run:
        missing = [path for path in artifacts if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Command completed but did not create {missing}")
        write_stage(marker, config, command, artifacts)


def run_fold(args: argparse.Namespace, fold: int) -> Path:
    protocol = args.protocol_dir / f"fold_{fold}"
    tensor_dir = args.tensors_root / f"fold_{fold}"
    base_dir = args.base_training_root / f"fold_{fold}"
    online_dir = args.training_root / f"fold_{fold}"
    builder = PIPELINE / "45_build_band_mean_window_tensors.py"
    absolute_trainer = PIPELINE / "46_train_band_mean_multiclass.py"
    online_trainer = PIPELINE / "59_train_online_cn0_baseline.py"
    model = ROOT / "models" / "gnss_signal_baselines.py"
    tensor_artifacts = [
        tensor_dir / "train.npz", tensor_dir / "val.npz", tensor_dir / "test.npz",
        tensor_dir / "feature_names.json", tensor_dir / "scaler.json", tensor_dir / "tensor_metadata.json",
        tensor_dir / "device_mapping.json", tensor_dir / "source_mapping.json",
    ]
    build_config = {
        "stage": "build_absolute_global",
        "fold": fold,
        "scope": args.scope,
        "scaler_mode": "global",
        "causal_baseline_mode": "none",
        "inputs": {
            "csv": fingerprint(args.csv),
            "config": fingerprint(args.config),
            "epoch_manifest": fingerprint(protocol / "epoch_split_manifest.csv"),
            "outer_manifest": fingerprint(protocol / "recording_split_manifest.csv"),
            "builder": fingerprint(builder),
        },
    }
    stage(
        tensor_dir / "online_baseline_build_stage.json", build_config,
        builder_command(args, fold, tensor_dir), tensor_artifacts, args.resume, args.dry_run,
    )

    absolute_checkpoint = base_dir / f"best_band_mean_window_{args.encoder}.pt"
    absolute_metrics = base_dir / f"val_metrics_band_mean_window_{args.encoder}.json"
    warm_start = absolute_checkpoint if args.warm_start else None
    online_checkpoint = online_dir / f"best_online_cn0_baseline_{args.encoder}.pt"
    prediction_path = online_dir / f"test_predictions_band_mean_window_{args.encoder}.csv"
    if args.dry_run:
        if args.warm_start:
            run(absolute_train_command(args, tensor_dir, base_dir), dry_run=True)
        run(online_train_command(args, tensor_dir, online_dir, warm_start), dry_run=True)
        run(
            online_predict_command(args, fold, tensor_dir, online_dir, online_checkpoint),
            dry_run=True,
        )
        return prediction_path
    if args.warm_start:
        base_config = {
            "stage": "absolute_warm_start",
            "fold": fold,
            "tensor_sha256": {path.name: sha256_file(path) for path in tensor_artifacts},
            "trainer": fingerprint(absolute_trainer),
            "model": fingerprint(model),
            "encoder": args.encoder,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "epochs": args.base_epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
            "seed": args.seed,
            "drop_features": ABSOLUTE_DROP_FEATURES,
            "class_weight_mult": args.class_weight_mult,
        }
        stage(
            base_dir / "online_baseline_absolute_stage.json", base_config,
            absolute_train_command(args, tensor_dir, base_dir), [absolute_checkpoint, absolute_metrics],
            args.resume, args.dry_run,
        )

    online_metrics = online_dir / f"val_metrics_online_cn0_baseline_{args.encoder}.json"
    online_config = {
        "stage": "online_self_updating_baseline",
        "fold": fold,
        "tensor_sha256": {path.name: sha256_file(path) for path in tensor_artifacts},
        "trainer": fingerprint(online_trainer),
        "model": fingerprint(model),
        "encoder": args.encoder,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "epochs": args.epochs,
        "lr": args.online_lr,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "seed": args.seed,
        "alpha": args.alpha,
        "rollout_batch_size": args.rollout_batch_size,
        "optimizer_step_interval": args.optimizer_step_interval,
        "class_weight_mult": args.class_weight_mult,
        "warm_start": fingerprint(absolute_checkpoint) if args.warm_start else None,
    }
    stage(
        online_dir / "online_baseline_train_stage.json", online_config,
        online_train_command(args, tensor_dir, online_dir, warm_start), [online_checkpoint, online_metrics],
        args.resume, args.dry_run,
    )

    predict_config = {
        "stage": "online_test_rollout",
        "fold": fold,
        "checkpoint": fingerprint(online_checkpoint),
        "test_tensor": fingerprint(tensor_dir / "test.npz"),
        "feature_names": fingerprint(tensor_dir / "feature_names.json"),
        "metadata": fingerprint(tensor_dir / "tensor_metadata.json"),
        "trainer": fingerprint(online_trainer),
    }
    stage(
        online_dir / "online_baseline_predict_stage.json", predict_config,
        online_predict_command(args, fold, tensor_dir, online_dir, online_checkpoint), [prediction_path],
        args.resume, args.dry_run,
    )
    return prediction_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol-dir", type=Path,
        default=ROOT / "output" / "protocols" / "mixed_timeblock_outer_cv4_w5_v2",
    )
    parser.add_argument("--csv", type=Path, default=ROOT / "output" / "processed_gnss_data.csv")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "preprocessing.yml")
    parser.add_argument("--scope", choices=("static", "all"), default="static")
    parser.add_argument(
        "--experiment-name",
        default="online_cn0_baseline_v2",
        help="Output namespace; v2 is the strict all-endpoint online-update protocol.",
    )
    parser.add_argument("--alpha", type=float, default=0.98)
    parser.add_argument("--encoder", choices=("lstm", "gru", "tcn"), default="tcn")
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--base-epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--rollout-batch-size", type=int, default=256)
    parser.add_argument("--optimizer-step-interval", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--online-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--folds", type=int, nargs="*", default=None)
    parser.add_argument("--class-weight-mult", type=float, nargs=NUM_CLASSES, default=None)
    parser.add_argument("--no-warm-start", dest="warm_start", action="store_false")
    parser.set_defaults(warm_start=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--no-aggregate", action="store_true",
        help="Run fold-local stages only; use this when several folds run in parallel.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 0.0 <= args.alpha < 1.0:
        parser.error("--alpha must be in [0, 1)")
    if args.hidden_dim < 1 or args.epochs < 1 or args.base_epochs < 1:
        parser.error("hidden-dim, epochs, and base-epochs must be positive")
    if args.batch_size < 1 or args.rollout_batch_size < 1 or args.optimizer_step_interval < 1 or args.patience < 1:
        parser.error("batch sizes, optimizer-step-interval, and patience must be positive")
    tag = f"{scope_tag(args.scope)}_global_alpha{args.alpha:.4f}".replace(".", "p")
    args.tensors_root = ROOT / "output" / "tensors" / args.experiment_name / tag
    args.base_training_root = ROOT / "output" / "training" / args.experiment_name / f"{tag}_absolute_warm_start"
    args.training_root = ROOT / "output" / "training" / args.experiment_name / tag
    return args


def main() -> None:
    args = parse_args()
    for path in (args.protocol_dir, args.csv, args.config):
        if not path.exists():
            raise FileNotFoundError(path)
    available = fold_ids(args.protocol_dir)
    folds = args.folds or available
    if len(folds) != len(set(folds)):
        raise ValueError(f"Duplicate fold IDs are not allowed: {folds}")
    invalid = sorted(set(folds).difference(available))
    if invalid:
        raise ValueError(f"Unknown folds {invalid}; available={available}")
    LOG.info(
        "online C/N0 baseline scope=%s folds=%s alpha=%.6f warm_start=%s",
        args.scope, folds, args.alpha, args.warm_start,
    )
    predictions = [run_fold(args, fold) for fold in folds]
    if args.dry_run or args.no_aggregate:
        return

    aggregate_command = [
        sys.executable, str(PIPELINE / "47_aggregate_band_mean_cv.py"),
        "--protocol-dir", str(args.protocol_dir),
        "--training-root", str(args.training_root),
        "--encoder", args.encoder,
        "--aggregate-only",
        "--folds", *[str(fold) for fold in folds],
    ]
    aggregate_artifacts = [
        args.training_root / "aggregate_test_metrics.json",
        args.training_root / "aggregate_test_predictions.csv",
    ]
    if set(folds) != set(available):
        tag = "_".join(str(fold) for fold in folds)
        aggregate_artifacts = [
            args.training_root / f"_subset_folds_{tag}" / "aggregate_test_metrics.json",
            args.training_root / f"_subset_folds_{tag}" / "aggregate_test_predictions.csv",
        ]
    aggregate_config = {
        "stage": "aggregate_online_test_predictions",
        "folds": folds,
        "predictions": {str(path.resolve()): sha256_file(path) for path in predictions},
        "aggregator": fingerprint(PIPELINE / "47_aggregate_band_mean_cv.py"),
    }
    stage(
        args.training_root / "online_baseline_aggregate_stage.json", aggregate_config,
        aggregate_command, aggregate_artifacts, args.resume, args.dry_run,
    )
    LOG.info("complete: %s", args.training_root)


if __name__ == "__main__":
    main()

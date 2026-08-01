"""Run one static response-state fold end to end.

This is a small orchestration wrapper around the existing pipeline scripts.  It
does not build the signal-level W5 tensors; it expects
output/tensors/static_timeblock_outer_v2/fold_<n> to already exist.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", required=True, help="fold name, e.g. fold_6")
    parser.add_argument("--python-exe", type=Path, default=Path(sys.executable))
    parser.add_argument("--signal-root", type=Path, default=ROOT / "output" / "tensors" / "static_timeblock_outer_v2")
    parser.add_argument("--device-root", type=Path, default=ROOT / "output" / "tensors" / "static_response_state_v1")
    parser.add_argument("--training-root", type=Path, default=ROOT / "output" / "hierarchical_event_v1" / "static_response_state_v1")
    parser.add_argument("--device-aggregate-profile", choices=("robust", "sparse_extreme"), default="sparse_extreme")
    parser.add_argument("--initial-baseline-windows", type=int, default=30)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--max-val-far", type=float, default=0.05)
    parser.add_argument("--min-val-abnormal-recall", type=float, default=0.90)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    parser.add_argument("--overwrite-device-tensors", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="print commands without executing them")
    args = parser.parse_args()
    if not args.fold.startswith("fold_"):
        parser.error("--fold must look like fold_<n>")
    if min(args.initial_baseline_windows, args.hidden_dim, args.epochs, args.batch_size, args.patience) < 1:
        parser.error("window/training counts must be positive")
    return args


def command_json(command: list[str]) -> str:
    return json.dumps(command, ensure_ascii=False)


def run(command: list[str], dry_run: bool) -> None:
    print(command_json(command))
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    args = parse_args()
    signal_dir = args.signal_root / args.fold
    if not signal_dir.exists():
        raise FileNotFoundError(f"Missing signal tensor directory: {signal_dir}")
    feature_tag = f"{args.device_aggregate_profile.replace('_extreme', '')}_initial{args.initial_baseline_windows}_device"
    device_dir = args.device_root / args.fold / f"device_tensors_{feature_tag}"
    fold_train_root = args.training_root / args.fold
    flat_dir = fold_train_root / f"mlp_{feature_tag}_h{args.hidden_dim}"
    direct_dir = fold_train_root / f"direct_expert_mlp_h{args.hidden_dim}"
    override_dir = fold_train_root / f"direct_override_mlp_h{args.hidden_dim}_valcal_all"
    python_exe = str(args.python_exe)

    build_cmd = [
        python_exe, str(ROOT / "pipeline_total" / "36_build_device_attack_event_tensors.py"),
        "--signal-data-dir", str(signal_dir),
        "--output-dir", str(device_dir),
        "--feature-set", "initial_baseline_delta_with_device",
        "--device-aggregate-profile", args.device_aggregate_profile,
        "--initial-baseline-windows", str(args.initial_baseline_windows),
        "--initial-baseline-policy", "exclude_stream",
    ]
    if args.overwrite_device_tensors:
        build_cmd.append("--overwrite")
    run(build_cmd, args.dry_run)

    train_common = [
        "--model", "mlp",
        "--hidden-dim", str(args.hidden_dim),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--patience", str(args.patience),
    ]
    run([
        python_exe, str(ROOT / "pipeline_total" / "37_train_device_attack_event.py"),
        "--data-dir", str(device_dir),
        "--output-dir", str(flat_dir),
        "--label-key", "y_response_state",
        "--num-classes", "3",
        *train_common,
    ], args.dry_run)
    flat_checkpoint = flat_dir / "best_device_event_mlp.pt"
    run([
        python_exe, str(ROOT / "pipeline_total" / "37_train_device_attack_event.py"),
        "--data-dir", str(device_dir),
        "--output-dir", str(flat_dir),
        "--label-key", "y_response_state",
        "--checkpoint", str(flat_checkpoint),
        "--test-only",
    ], args.dry_run)

    run([
        python_exe, str(ROOT / "pipeline_total" / "37_train_device_attack_event.py"),
        "--data-dir", str(device_dir),
        "--output-dir", str(direct_dir),
        "--label-key", "y_response_state",
        "--label-transform", "direct",
        "--num-classes", "2",
        *train_common,
    ], args.dry_run)
    direct_checkpoint = direct_dir / "best_device_event_mlp.pt"
    run([
        python_exe, str(ROOT / "pipeline_total" / "42_eval_response_state_direct_override.py"),
        "--data-dir", str(device_dir),
        "--output-dir", str(override_dir),
        "--split", "test",
        "--flat-checkpoint", str(flat_checkpoint),
        "--direct-checkpoint", str(direct_checkpoint),
        "--calibrate-threshold-on-val",
        "--thresholds", *[str(threshold) for threshold in args.thresholds],
        "--max-val-far", str(args.max_val_far),
        "--min-val-abnormal-recall", str(args.min_val_abnormal_recall),
        "--override-scope", "all",
        "--fold", args.fold[len("fold_"):],
        "--predictions-csv", str(override_dir / "test_response_state_predictions.csv"),
    ], args.dry_run)


if __name__ == "__main__":
    main()

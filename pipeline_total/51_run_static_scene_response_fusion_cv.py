#!/usr/bin/env python3
"""Run the static late-fusion scene + response-state experiment end to end.

The pipeline has two separately supervised branches under the *same* outer
static recording-holdout folds:

1. a no-AGC dual-band scene classifier (normal / L1 / L5 / L1+L5), and
2. an all-device response-state classifier (normal / anomaly / direct).

After their outer-test predictions are exported, script 50 aligns the endpoint
TOWs and attaches the dual-band scene consensus to every device state.  This is
not a pre-fusion standalone benchmark: its final product is the joint diagnosis
table and its state/scene/joint metrics.

``fold_3`` is excluded by default.  Its held-out recording starts during an
attack and has no initial reviewed-normal segment, whereas the response-state
branch explicitly needs an initial baseline.  The six default folds are the
common valid evaluation support for both branches.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FOLDS = (1, 2, 4, 5, 6, 7)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-exe", type=Path, default=Path(sys.executable))
    parser.add_argument("--folds", type=int, nargs="+", default=list(DEFAULT_FOLDS))
    parser.add_argument(
        "--band-tensors-root", type=Path,
        default=ROOT / "output" / "tensors" / "static_scene_response_fusion_v1" / "band_scene",
    )
    parser.add_argument(
        "--band-training-root", type=Path,
        default=ROOT / "output" / "training" / "static_scene_response_fusion_v1" / "band_scene",
    )
    parser.add_argument(
        "--response-device-root", type=Path,
        default=ROOT / "output" / "tensors" / "static_scene_response_fusion_v1" / "response_state",
    )
    parser.add_argument(
        "--response-training-root", type=Path,
        default=ROOT / "output" / "hierarchical_event_v1" / "static_scene_response_fusion_v1" / "response_state",
    )
    parser.add_argument(
        "--fusion-output-dir", type=Path,
        default=ROOT / "output" / "hierarchical_event_v1" / "static_scene_response_fusion_v1" / "fusion",
    )
    parser.add_argument("--band-epochs", type=int, default=40)
    parser.add_argument("--response-epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--skip-existing-band", action="store_true")
    parser.add_argument("--overwrite-response-tensors", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.folds or any(fold < 1 for fold in args.folds):
        parser.error("--folds must contain positive fold identifiers")
    if len(set(args.folds)) != len(args.folds):
        parser.error("--folds cannot contain duplicates")
    if min(args.band_epochs, args.response_epochs) < 1:
        parser.error("epoch counts must be positive")
    return args


def run(command: list[str], dry_run: bool) -> None:
    print(json.dumps(command, ensure_ascii=False))
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    args = parse_args()
    python = str(args.python_exe)
    folds = [int(fold) for fold in args.folds]

    band_command = [
        python, str(ROOT / "pipeline_total" / "47_aggregate_band_mean_cv.py"),
        "--folds", *[str(fold) for fold in folds],
        "--tensors-root", str(args.band_tensors_root),
        "--training-root", str(args.band_training_root),
        "--drop-features", "AgcDb",
        "--epochs", str(args.band_epochs),
        "--seed", str(args.seed),
    ]
    if args.skip_existing_band:
        band_command.append("--skip-existing")
    run(band_command, args.dry_run)

    for fold in folds:
        response_command = [
            python, str(ROOT / "pipeline_total" / "43_run_static_response_state_fold.py"),
            "--fold", f"fold_{fold}",
            "--python-exe", python,
            "--device-root", str(args.response_device_root),
            "--training-root", str(args.response_training_root),
            "--epochs", str(args.response_epochs),
        ]
        if args.overwrite_response_tensors:
            response_command.append("--overwrite-device-tensors")
        run(response_command, args.dry_run)

    run([
        python, str(ROOT / "pipeline_total" / "50_fuse_static_scene_response_predictions.py"),
        "--band-training-root", str(args.band_training_root),
        "--response-training-root", str(args.response_training_root),
        "--output-dir", str(args.fusion_output_dir),
        "--folds", *[str(fold) for fold in folds],
    ], args.dry_run)


if __name__ == "__main__":
    main()

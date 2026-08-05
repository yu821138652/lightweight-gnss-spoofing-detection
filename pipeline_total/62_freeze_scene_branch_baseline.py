"""Create or verify the immutable snapshot of the formal scene-branch baseline.

The snapshot deliberately contains the compact tensors rather than the 1.2 GB
processed CSV.  The CSV is recorded as an external, SHA-256-pinned dependency.
This keeps the frozen baseline self-contained for evaluation and audit while
avoiding an unnecessary duplicate of the derived source data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASELINE_ID = "scene_branch_tcn32_mixed_cv4_w5_normalref_v1"
DEFAULT_SNAPSHOT_DIR = ROOT / "output" / "frozen" / BASELINE_ID
PROTOCOL_DIR = ROOT / "output" / "protocols" / "mixed_timeblock_outer_cv4_w5_v2"
TENSORS_DIR = ROOT / "output" / "tensors" / "normal_reference_v1" / "mixed"
TRAINING_DIR = ROOT / "output" / "training" / "normal_reference_v1_rebuilt" / "mixed"
PROCESSED_CSV = ROOT / "output" / "processed_gnss_data.csv"
CONFIG_PATH = ROOT / "configs" / "preprocessing.yml"
FOLDS = (1, 2, 3, 4)
EXPECTED_METRICS = {
    "windows": 43672,
    "macro_f1": 0.9245234304140474,
    "accuracy": 0.9575242718446602,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def external_file_record(path: Path) -> dict[str, Any]:
    return {
        "source_path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def require_dir(path: Path) -> None:
    if not path.is_dir():
        raise FileNotFoundError(path)


def git_output(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def installed_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def source_layout() -> list[tuple[Path, Path]]:
    """Return source/destination pairs for all internal snapshot artifacts."""
    pairs: list[tuple[Path, Path]] = [
        (PROTOCOL_DIR, Path("protocol")),
        (CONFIG_PATH, Path("configuration") / CONFIG_PATH.name),
        (
            ROOT / "pipeline_total" / "45_build_band_mean_window_tensors.py",
            Path("code_snapshot") / "pipeline_total" / "45_build_band_mean_window_tensors.py",
        ),
        (
            ROOT / "pipeline_total" / "46_train_band_mean_multiclass.py",
            Path("code_snapshot") / "pipeline_total" / "46_train_band_mean_multiclass.py",
        ),
        (
            ROOT / "pipeline_total" / "47_aggregate_band_mean_cv.py",
            Path("code_snapshot") / "pipeline_total" / "47_aggregate_band_mean_cv.py",
        ),
        (
            ROOT / "models" / "__init__.py",
            Path("code_snapshot") / "models" / "__init__.py",
        ),
        (
            ROOT / "models" / "gnss_signal_baselines.py",
            Path("code_snapshot") / "models" / "gnss_signal_baselines.py",
        ),
        (
            TRAINING_DIR / "aggregate_test_metrics.json",
            Path("training") / "aggregate_test_metrics.json",
        ),
        (
            TRAINING_DIR / "aggregate_test_predictions.csv",
            Path("training") / "aggregate_test_predictions.csv",
        ),
    ]
    for fold in FOLDS:
        pairs.extend(
            [
                (TENSORS_DIR / f"fold_{fold}", Path("tensors") / f"fold_{fold}"),
                (TRAINING_DIR / f"fold_{fold}", Path("training") / f"fold_{fold}"),
            ]
        )
    return pairs


def validate_source() -> dict[str, Any]:
    require_dir(PROTOCOL_DIR)
    require_dir(TENSORS_DIR)
    require_dir(TRAINING_DIR)
    require_file(PROCESSED_CSV)
    require_file(CONFIG_PATH)
    for source, _ in source_layout():
        if source.is_dir():
            require_dir(source)
        else:
            require_file(source)

    metrics_path = TRAINING_DIR / "aggregate_test_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not isinstance(metrics, dict):
        raise ValueError(f"Expected an object in {metrics_path}")
    for key, expected in EXPECTED_METRICS.items():
        actual = metrics.get(key)
        if isinstance(expected, float):
            if not isinstance(actual, (int, float)) or abs(float(actual) - expected) > 1e-12:
                raise ValueError(f"Baseline metric {key}={actual!r}, expected {expected!r}")
        elif actual != expected:
            raise ValueError(f"Baseline metric {key}={actual!r}, expected {expected!r}")
    return metrics


def copy_source_layout(destination: Path) -> None:
    for source, relative_destination in source_layout():
        target = destination / relative_destination
        if source.is_dir():
            shutil.copytree(source, target, copy_function=shutil.copy2)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def baseline_description(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "baseline_id": BASELINE_ID,
        "task": "four-way scene branch classification",
        "classes": {"0": "normal", "1": "L1", "2": "L5", "3": "L1+L5"},
        "protocol": "mixed_timeblock_outer_cv4_w5_v2",
        "outer_folds": 4,
        "window_epochs": 5,
        "encoder": "tcn",
        "hidden_dim": 32,
        "dropout": 0.1,
        "epochs": 40,
        "seed": 2026,
        "feature_names": ["L1_Cn0DbHz", "L5_Cn0DbHz", "L1Present", "L5Present"],
        "dropped_features": [
            "AgcDb",
            "ReceivedSvTimeUncertaintyNanos",
            "PseudorangeRateUncertaintyMetersPerSecond",
            "Cn0DbHzL1MinusL5",
        ],
        "scaler_mode": "global",
        "normal_reference_mode": "train_normal_band_mean",
        "normal_reference_minimum_epochs": 256,
        "single_band_policy": "excluded from optimization and reported metrics",
        "aggregate_metrics": {
            key: metrics[key] for key in ("windows", "macro_f1", "accuracy", "confusion_matrix", "per_class")
        },
        "interpretation_limit": (
            "This is the frozen development baseline. Its outer-test outputs have been "
            "examined during model exploration and must not be presented as a newly untouched "
            "independent final test."
        ),
    }


def render_readme(metrics: dict[str, Any]) -> str:
    return f"""# {BASELINE_ID}

This directory is the frozen local snapshot of the formal mixed scene-branch
baseline. Do not use it as an output directory for new experiments.

## Frozen result

- Model: TCN32, two causal convolution layers (dilation 1 and 2)
- Input: five epochs of L1/L5 train-normal C/N0 residuals plus L1Present and
  L5Present; four input features per epoch
- Protocol: mixed_timeblock_outer_cv4_w5_v2, four outer recording folds
- Aggregate usable endpoints: {metrics['windows']}
- Macro-F1: {metrics['macro_f1']:.12f}
- Accuracy: {metrics['accuracy']:.12f}

## Contents

- `protocol/`: four-fold recording and time-block split manifests.
- `tensors/`: every fold's frozen train/validation/test tensors, scaler,
  normal reference, and mappings.
- `training/`: four checkpoints, per-fold validation metrics and predictions,
  plus the aggregate prediction CSV and metrics.
- `configuration/`: preprocessing configuration as it stood at freeze time.
- `code_snapshot/`: exact source files used for the tensor/training/aggregation
  workflow and model definition.
- `freeze_manifest.json`: SHA-256 and size for every internal artifact.

The raw `processed_gnss_data.csv` is intentionally not duplicated here. Its
absolute source path, size, and SHA-256 are recorded as an external dependency
in `freeze_manifest.json`. The copied tensors are sufficient to audit the
frozen checkpoints and predictions without that CSV.

## Verify

Run this command from the repository root:

```powershell
python pipeline_total\\62_freeze_scene_branch_baseline.py --verify
```

Verification is strict: it checks the expected artifact set, file sizes,
SHA-256 hashes, and the frozen aggregate metrics. New work must write to a new
output directory rather than changing this snapshot.

## Result boundary

This is a frozen development baseline. The outer-test predictions were examined
during model exploration, so they are not a newly untouched independent final
test result.
"""


def build_manifest(snapshot_root: Path, metrics: dict[str, Any]) -> dict[str, Any]:
    files = [
        file_record(path, snapshot_root)
        for path in sorted(snapshot_root.rglob("*"))
        if path.is_file() and path.name != "freeze_manifest.json"
    ]
    return {
        "schema_version": 1,
        "baseline": baseline_description(metrics),
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "repository": {
            "root": str(ROOT.resolve()),
            "git_head": git_output("rev-parse", "HEAD"),
            "git_branch": git_output("branch", "--show-current"),
            "git_status_porcelain": (git_output("status", "--porcelain") or "").splitlines(),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": installed_version("numpy"),
            "pandas": installed_version("pandas"),
            "scikit_learn": installed_version("scikit-learn"),
            "torch": installed_version("torch"),
        },
        "external_dependencies": {
            "processed_gnss_data_csv": external_file_record(PROCESSED_CSV),
        },
        "artifacts": files,
    }


def verify_snapshot(snapshot_root: Path) -> list[str]:
    manifest_path = snapshot_root / "freeze_manifest.json"
    require_file(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        return ["freeze_manifest.json is not a JSON object"]
    if manifest.get("schema_version") != 1:
        return [f"Unsupported manifest schema: {manifest.get('schema_version')!r}"]
    baseline = manifest.get("baseline")
    if not isinstance(baseline, dict) or baseline.get("baseline_id") != BASELINE_ID:
        return ["Manifest does not identify the expected baseline"]
    expected = manifest.get("artifacts")
    if not isinstance(expected, list):
        return ["Manifest artifacts must be a list"]

    errors: list[str] = []
    expected_by_path: dict[str, dict[str, Any]] = {}
    for entry in expected:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            errors.append(f"Invalid manifest artifact entry: {entry!r}")
            continue
        expected_by_path[entry["path"]] = entry
    actual_paths = {
        path.relative_to(snapshot_root).as_posix()
        for path in snapshot_root.rglob("*")
        if path.is_file() and path.name != "freeze_manifest.json"
    }
    missing = sorted(set(expected_by_path).difference(actual_paths))
    extra = sorted(actual_paths.difference(expected_by_path))
    errors.extend(f"Missing artifact: {path}" for path in missing)
    errors.extend(f"Unexpected artifact: {path}" for path in extra)
    for relative_path, entry in sorted(expected_by_path.items()):
        path = snapshot_root / relative_path
        if not path.is_file():
            continue
        actual_size = path.stat().st_size
        if actual_size != entry.get("size_bytes"):
            errors.append(
                f"Size mismatch for {relative_path}: {actual_size} != {entry.get('size_bytes')}"
            )
            continue
        actual_hash = sha256_file(path)
        if actual_hash != entry.get("sha256"):
            errors.append(f"SHA-256 mismatch for {relative_path}")

    metrics_path = snapshot_root / "training" / "aggregate_test_metrics.json"
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        frozen_metrics = baseline.get("aggregate_metrics", {})
        for key in ("windows", "macro_f1", "accuracy"):
            actual = metrics.get(key)
            expected_value = frozen_metrics.get(key)
            if isinstance(expected_value, float):
                if not isinstance(actual, (int, float)) or abs(float(actual) - expected_value) > 1e-12:
                    errors.append(f"Frozen metric mismatch for {key}: {actual!r} != {expected_value!r}")
            elif actual != expected_value:
                errors.append(f"Frozen metric mismatch for {key}: {actual!r} != {expected_value!r}")
    else:
        errors.append("Missing training/aggregate_test_metrics.json")
    return errors


def create_snapshot(snapshot_root: Path) -> None:
    metrics = validate_source()
    if snapshot_root.exists():
        raise FileExistsError(
            f"Snapshot already exists: {snapshot_root}. Verify it with --verify; do not overwrite it."
        )
    snapshot_root.parent.mkdir(parents=True, exist_ok=True)
    staging = snapshot_root.parent / f".{snapshot_root.name}.staging-{uuid.uuid4().hex}"
    try:
        copy_source_layout(staging)
        (staging / "README.md").write_text(render_readme(metrics), encoding="utf-8")
        manifest = build_manifest(staging, metrics)
        (staging / "freeze_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        errors = verify_snapshot(staging)
        if errors:
            raise RuntimeError("Snapshot verification failed:\n" + "\n".join(errors))
        os.replace(staging, snapshot_root)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT_DIR)
    parser.add_argument("--verify", action="store_true", help="Verify an existing frozen snapshot.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snapshot_root = args.snapshot_dir.resolve()
    if args.verify:
        errors = verify_snapshot(snapshot_root)
        if errors:
            for error in errors:
                print(f"ERROR: {error}", file=sys.stderr)
            raise SystemExit(1)
        print(f"Verified frozen baseline: {snapshot_root}")
        return
    create_snapshot(snapshot_root)
    print(f"Created frozen baseline: {snapshot_root}")


if __name__ == "__main__":
    main()

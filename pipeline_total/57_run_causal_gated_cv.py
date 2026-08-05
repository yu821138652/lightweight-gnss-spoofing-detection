"""Run leakage-controlled causal-gated scene-classification cross-validation.

The gate is itself the four-way band-mean classifier, trained on the six
device-independent EMA features::

    L1_Cn0Relative, L5_Cn0Relative,
    L1_Cn0AbsRelative, L5_Cn0AbsRelative,
    L1Present, L5Present

For each outer fold this driver creates two kinds of temporary protocols:

* grouped cross-fit protocols produce out-of-fold gate probabilities for every
  outer-train endpoint.  A recording is never used for fitting or early-stop
  selection by the model that predicts that recording;
* one full-gate protocol fits only outer-train epochs (with one whole
  outer-train recording reserved for early stopping) and predicts the original
  outer validation and test epochs.

The predictions are merged by ``source_id + device_id + window_time_nanos``
and passed to ``45_build_band_mean_window_tensors.py`` in ``gated`` mode.  No
ground-truth label is consulted when the causal baseline is updated.  Missing
gate scores (notably the four pre-window warm-up epochs and single-band
endpoints excluded by the trainer) retain script 45's freeze semantics.

This is deliberately an orchestration layer: tensor construction and all model
code remain in scripts 45 and 46, while final aggregation is delegated to 47.
Every reusable stage gets a configuration marker, and every outer fold gets an
audit manifest with commands, group assignments, probability checks, and exact
split coverage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "pipeline_total"
KEYS = ["Environment", "Scenario", "Session"]
PREDICTION_KEYS = ["source_id", "device_id", "window_time_nanos"]
ENCODERS = ("lstm", "gru", "tcn")
SIX_FEATURE_NAMES = [
    "L1_Cn0Relative",
    "L5_Cn0Relative",
    "L1_Cn0AbsRelative",
    "L5_Cn0AbsRelative",
    "L1Present",
    "L5Present",
]
# Script 46 matches both exact names and band-agnostic suffixes.  These tokens
# drop all legacy columns while leaving the four causal residuals and two flags.
SIX_FEATURE_DROP = [
    "Cn0DbHz",
    "AgcDb",
    "ReceivedSvTimeUncertaintyNanos",
    "PseudorangeRateUncertaintyMetersPerSecond",
    "Cn0DbHzL1MinusL5",
]
STAGE_MARKER = "orchestration_stage.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelConfig:
    encoder: str
    hidden_dim: int
    dropout: float
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    patience: int
    seed: int
    class_weight_mult: tuple[float, ...] | None


class CommandRunner:
    def __init__(self, dry_run: bool) -> None:
        self.dry_run = dry_run
        self.commands: list[list[str]] = []
        self._lock = threading.Lock()

    def run(self, command: list[str]) -> None:
        rendered = [str(part) for part in command]
        with self._lock:
            self.commands.append(rendered)
        LOG.info("%s: %s", "plan" if self.dry_run else "run", subprocess.list2cmdline(rendered))
        if self.dry_run:
            return
        result = subprocess.run(rendered, cwd=str(ROOT))
        if result.returncode != 0:
            raise RuntimeError(
                f"Command failed ({result.returncode}): {subprocess.list2cmdline(rendered)}"
            )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(jsonable(value), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fold_ids(protocol_dir: Path) -> list[int]:
    ids = sorted(
        int(path.name.split("_", 1)[1])
        for path in protocol_dir.glob("fold_*")
        if path.is_dir() and path.name.split("_", 1)[1].isdigit()
    )
    if not ids:
        raise ValueError(f"No fold_* directories under {protocol_dir}")
    return ids


def scope_mask(frame: pd.DataFrame, scope: str) -> pd.Series:
    scenario = frame["Scenario"].astype(str)
    if scope == "static":
        return scenario.str.startswith("st_")
    if scope == "dynamic":
        return scenario.str.startswith("dy_")
    if scope == "all":
        return scenario.str.startswith("st_") | scenario.str.startswith("dy_")
    raise ValueError(f"Unsupported scope: {scope}")


def load_outer_protocol(fold_protocol: Path, scope: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    recording_path = fold_protocol / "recording_split_manifest.csv"
    epoch_path = fold_protocol / "epoch_split_manifest.csv"
    for path in (recording_path, epoch_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    recordings = pd.read_csv(recording_path, encoding="utf-8-sig")
    epochs = pd.read_csv(epoch_path, encoding="utf-8-sig")
    recording_required = {*KEYS, "recording_id", "split"}
    epoch_required = {*KEYS, "recording_id", "canonical_epoch_ms", "split"}
    if missing := recording_required.difference(recordings.columns):
        raise ValueError(f"{recording_path} missing columns: {sorted(missing)}")
    if missing := epoch_required.difference(epochs.columns):
        raise ValueError(f"{epoch_path} missing columns: {sorted(missing)}")
    if recordings["recording_id"].duplicated().any():
        raise ValueError(f"Duplicate recording_id values in {recording_path}")
    for frame in (recordings, epochs):
        frame["recording_id"] = pd.to_numeric(frame["recording_id"], errors="raise").astype(int)
        for column in KEYS:
            frame[column] = frame[column].astype(str)
    scoped_ids = set(recordings.loc[scope_mask(recordings, scope), "recording_id"].tolist())
    if not scoped_ids:
        raise ValueError(f"No recordings in scope={scope!r} under {fold_protocol}")
    epoch_ids = set(epochs["recording_id"].unique().tolist())
    if not scoped_ids.issubset(epoch_ids):
        raise ValueError(f"Recordings without epoch rows: {sorted(scoped_ids - epoch_ids)}")
    return recordings, epochs


def outer_train_recordings(
    recordings: pd.DataFrame, epochs: pd.DataFrame, scope: str
) -> pd.DataFrame:
    train_epochs = epochs.loc[
        (epochs["split"].astype(str) == "train") & scope_mask(epochs, scope)
    ].copy()
    train_epochs["_positive_epoch"] = (
        pd.to_numeric(train_epochs["positive_epoch"], errors="coerce").fillna(0)
        if "positive_epoch" in train_epochs.columns
        else 0
    )
    train_counts = (
        train_epochs.groupby("recording_id")
        .agg(
            train_epoch_count=("recording_id", "size"),
            train_positive_epoch_count=("_positive_epoch", "sum"),
        )
    )
    result = recordings.loc[
        scope_mask(recordings, scope) & recordings["recording_id"].isin(train_counts.index)
    ].copy()
    result = result.merge(train_counts, on="recording_id", how="left")
    result["train_epoch_count"] = result["train_epoch_count"].astype(int)
    result["train_positive_epoch_count"] = result["train_positive_epoch_count"].astype(int)
    if len(result) < 3:
        raise ValueError(
            "Grouped OOF plus a recording-level early-stop set requires at least "
            f"three outer-train recordings; found {len(result)}"
        )
    return result.sort_values("recording_id", kind="mergesort").reset_index(drop=True)


def assign_group_folds(
    train_recordings: pd.DataFrame, requested_folds: int, seed: int
) -> list[list[int]]:
    count = len(train_recordings)
    folds = min(requested_folds, count)
    if folds < 2:
        raise ValueError("--inner-folds must produce at least two grouped folds")
    rng = random.Random(seed)
    tie_order = list(range(count))
    rng.shuffle(tie_order)
    tie_rank = {index: rank for rank, index in enumerate(tie_order)}
    rows = train_recordings.reset_index(drop=True).copy()
    rows["_tie"] = [tie_rank[index] for index in range(count)]
    rows = rows.sort_values(
        ["train_epoch_count", "Scenario", "_tie"],
        ascending=[False, True, True],
        kind="mergesort",
    )
    assignments: list[list[int]] = [[] for _ in range(folds)]
    totals = [0 for _ in range(folds)]
    scenario_counts: list[dict[str, int]] = [{} for _ in range(folds)]
    for row in rows.itertuples(index=False):
        scenario = str(row.Scenario)
        candidates = list(range(folds))
        rng.shuffle(candidates)
        chosen = min(
            candidates,
            key=lambda index: (
                scenario_counts[index].get(scenario, 0),
                totals[index],
                len(assignments[index]),
                index,
            ),
        )
        recording_id = int(row.recording_id)
        assignments[chosen].append(recording_id)
        totals[chosen] += int(row.train_epoch_count)
        scenario_counts[chosen][scenario] = scenario_counts[chosen].get(scenario, 0) + 1
    if any(not group for group in assignments):
        raise AssertionError(f"Internal error: empty cross-fit group in {assignments}")
    flat = [recording_id for group in assignments for recording_id in group]
    expected = train_recordings["recording_id"].astype(int).tolist()
    if sorted(flat) != sorted(expected) or len(flat) != len(set(flat)):
        raise AssertionError("Cross-fit assignment is not a recording partition")
    return [sorted(group) for group in assignments]


def choose_calibration_recording(
    train_recordings: pd.DataFrame, excluded_ids: set[int]
) -> int:
    available = train_recordings.loc[
        ~train_recordings["recording_id"].isin(excluded_ids)
    ].copy()
    if len(available) < 2:
        raise ValueError(
            f"Need at least two fitting recordings after excluding {sorted(excluded_ids)}"
        )
    rows = []
    for candidate in available.itertuples(index=False):
        remaining = available.loc[available["recording_id"] != int(candidate.recording_id)]
        scenario_counts = remaining.groupby("Scenario").size()
        positive = int(candidate.train_positive_epoch_count)
        normal = int(candidate.train_epoch_count) - positive
        rows.append(
            (
                -int(len(scenario_counts)),
                -int(scenario_counts.min()) if len(scenario_counts) else 0,
                -int(positive > 0 and normal > 0),
                -min(positive, normal),
                -positive,
                int(candidate.train_epoch_count),
                int(candidate.recording_id),
            )
        )
    return min(rows)[-1]


def write_inner_protocol(
    output_dir: Path,
    recordings: pd.DataFrame,
    epochs: pd.DataFrame,
    mode: str,
    train_ids: set[int],
    calibration_id: int,
    held_ids: set[int] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    held_ids = held_ids or set()
    if calibration_id in held_ids:
        raise ValueError("Calibration recording cannot also be an OOF target")
    inner = epochs.copy()
    original_split = inner["split"].astype(str)
    inner["outer_split"] = original_split
    inner["split"] = "guard"
    is_outer_train = original_split == "train"
    if mode == "oof":
        inner.loc[is_outer_train & inner["recording_id"].isin(train_ids), "split"] = "train"
        inner.loc[is_outer_train & (inner["recording_id"] == calibration_id), "split"] = "val"
        inner.loc[is_outer_train & inner["recording_id"].isin(held_ids), "split"] = "test"
    elif mode == "full":
        inner.loc[is_outer_train & inner["recording_id"].isin(train_ids), "split"] = "train"
        inner.loc[is_outer_train & (inner["recording_id"] == calibration_id), "split"] = "val"
        inner.loc[original_split.isin(["val", "test"]), "split"] = "test"
    else:
        raise ValueError(f"Unsupported inner protocol mode: {mode}")
    if "raw_split" in inner.columns:
        inner["outer_raw_split"] = inner["raw_split"].astype(str)
        inner["raw_split"] = inner["split"]
    if "is_guard" in inner.columns:
        inner["is_guard"] = inner["split"] == "guard"
    if "guard_reason" in inner.columns:
        converted = original_split != inner["split"]
        inner.loc[inner["split"] == "guard", "guard_reason"] = "causal_gate_inner_excluded"
        inner.loc[converted & (inner["split"] != "guard"), "guard_reason"] = ""
    counts = {str(key): int(value) for key, value in inner["split"].value_counts().items()}
    for required in ("train", "val", "test"):
        if counts.get(required, 0) == 0:
            raise ValueError(f"Inner {mode} protocol has no {required} epochs")
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        recordings.to_csv(
            output_dir / "recording_split_manifest.csv", index=False, encoding="utf-8-sig"
        )
        inner.to_csv(output_dir / "epoch_split_manifest.csv", index=False, encoding="utf-8-sig")
    return {
        "mode": mode,
        "train_recording_ids": sorted(train_ids),
        "calibration_recording_id": int(calibration_id),
        "held_recording_ids": sorted(held_ids),
        "epoch_split_counts": counts,
    }


def builder_command(
    args: argparse.Namespace,
    protocol_dir: Path,
    output_dir: Path,
    causal_mode: str,
    gate_predictions: Path | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(PIPELINE / "45_build_band_mean_window_tensors.py"),
        "--csv",
        str(args.csv),
        "--epoch-manifest",
        str(protocol_dir / "epoch_split_manifest.csv"),
        "--outer-manifest",
        str(protocol_dir / "recording_split_manifest.csv"),
        "--config",
        str(args.config),
        "--output-dir",
        str(output_dir),
        "--scope",
        args.scope,
        "--scaler-mode",
        "global",
        "--causal-baseline-mode",
        causal_mode,
        "--causal-half-life-seconds",
        str(args.causal_half_life_seconds),
        "--causal-normal-threshold",
        str(args.causal_normal_threshold),
    ]
    if gate_predictions is not None:
        command += ["--gate-predictions", str(gate_predictions)]
    return command


def trainer_command(
    data_dir: Path, output_dir: Path, config: ModelConfig
) -> list[str]:
    command = [
        sys.executable,
        str(PIPELINE / "46_train_band_mean_multiclass.py"),
        "--data-dir",
        str(data_dir),
        "--output-dir",
        str(output_dir),
        "--encoder",
        config.encoder,
        "--hidden-dim",
        str(config.hidden_dim),
        "--dropout",
        str(config.dropout),
        "--epochs",
        str(config.epochs),
        "--batch-size",
        str(config.batch_size),
        "--lr",
        str(config.learning_rate),
        "--weight-decay",
        str(config.weight_decay),
        "--patience",
        str(config.patience),
        "--seed",
        str(config.seed),
        "--drop-features",
        *SIX_FEATURE_DROP,
    ]
    if config.class_weight_mult is not None:
        command += ["--class-weight-mult", *[str(value) for value in config.class_weight_mult]]
    return command


def prediction_command(
    data_dir: Path,
    output_dir: Path,
    config: ModelConfig,
    split: str,
    fold: int,
) -> list[str]:
    return [
        sys.executable,
        str(PIPELINE / "46_train_band_mean_multiclass.py"),
        "--data-dir",
        str(data_dir),
        "--output-dir",
        str(output_dir),
        "--encoder",
        config.encoder,
        "--test-only",
        "--predict-split",
        split,
        "--fold",
        str(fold),
    ]


def stage_matches(marker: Path, config: dict[str, Any], required: list[Path]) -> bool:
    if not marker.is_file() or any(not path.is_file() for path in required):
        return False
    try:
        stored = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if stored.get("config") != jsonable(config):
        return False
    expected_hashes = stored.get("artifact_sha256")
    if not isinstance(expected_hashes, dict):
        return False
    actual_hashes = {str(path): sha256_file(path) for path in required}
    return expected_hashes == actual_hashes


def run_stage(
    runner: CommandRunner,
    marker: Path,
    config: dict[str, Any],
    required: list[Path],
    command: list[str],
    skip_existing: bool,
) -> str:
    if skip_existing and stage_matches(marker, config, required):
        LOG.info("reuse stage: %s", marker.parent)
        return "reused"
    runner.run(command)
    if not runner.dry_run:
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Stage did not create required artifacts: {missing}")
        write_json(
            marker,
            {
                "created_utc": utc_now(),
                "config": config,
                "command": command,
                "artifacts": [str(path) for path in required],
                "artifact_sha256": {
                    str(path): sha256_file(path)
                    for path in required
                },
            },
        )
    return "planned" if runner.dry_run else "ran"


def validate_six_features(tensor_dir: Path) -> None:
    feature_path = tensor_dir / "feature_names.json"
    names = json.loads(feature_path.read_text(encoding="utf-8"))
    kept = []
    for name in names:
        suffix = name.split("_", 1)[1] if "_" in name else name
        if name in ("L1Present", "L5Present") or not (
            name in SIX_FEATURE_DROP or suffix in SIX_FEATURE_DROP
        ):
            kept.append(name)
    if kept != SIX_FEATURE_NAMES:
        raise ValueError(
            f"Six-feature contract failed for {tensor_dir}: expected {SIX_FEATURE_NAMES}, got {kept}"
        )


def expected_keys(npz_path: Path) -> set[tuple[int, int, int]]:
    with np.load(npz_path, allow_pickle=False) as data:
        required = {*PREDICTION_KEYS, "single_band_mask"}
        if missing := required.difference(data.files):
            raise ValueError(f"{npz_path} missing trace arrays: {sorted(missing)}")
        usable = ~np.asarray(data["single_band_mask"], dtype=bool)
        arrays = [np.asarray(data[column])[usable] for column in PREDICTION_KEYS]
    keys = {
        (int(source_id), int(device_id), int(window_time))
        for source_id, device_id, window_time in zip(*arrays)
    }
    if len(keys) != int(usable.sum()):
        raise ValueError(f"Usable endpoint keys are not unique in {npz_path}")
    return keys


def read_gate_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = {*PREDICTION_KEYS, "prob_normal"}
    if missing := required.difference(frame.columns):
        raise ValueError(f"{path} missing prediction columns: {sorted(missing)}")
    for column in PREDICTION_KEYS:
        numeric = pd.to_numeric(frame[column], errors="raise")
        values = numeric.to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            raise ValueError(f"{path}: {column} must contain finite integers")
        frame[column] = values.astype(np.int64)
    frame["prob_normal"] = pd.to_numeric(frame["prob_normal"], errors="raise")
    probabilities = frame["prob_normal"].to_numpy(dtype=np.float64)
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError(f"{path}: prob_normal must be finite and in [0, 1]")
    duplicate = frame.duplicated(PREDICTION_KEYS, keep=False)
    if duplicate.any():
        sample = frame.loc[duplicate, PREDICTION_KEYS].head(5).to_dict("records")
        raise ValueError(f"{path}: duplicate gate keys, sample={sample}")
    return frame


def key_set(frame: pd.DataFrame) -> set[tuple[int, int, int]]:
    return set(
        zip(
            frame["source_id"].astype(int),
            frame["device_id"].astype(int),
            frame["window_time_nanos"].astype(np.int64),
        )
    )


def probability_stats(frame: pd.DataFrame, threshold: float) -> dict[str, Any]:
    values = frame["prob_normal"].to_numpy(dtype=np.float64)
    return {
        "rows": int(len(frame)),
        "minimum": float(values.min()) if len(values) else None,
        "maximum": float(values.max()) if len(values) else None,
        "mean": float(values.mean()) if len(values) else None,
        "normal_confident_rows": int((values >= threshold).sum()),
        "normal_confident_rate": float((values >= threshold).mean()) if len(values) else None,
    }


def validate_prediction_coverage(
    frame: pd.DataFrame, tensor_split: Path, label: str
) -> dict[str, Any]:
    actual = key_set(frame)
    expected = expected_keys(tensor_split)
    missing = expected - actual
    extra = actual - expected
    result = {
        "label": label,
        "expected_usable_endpoints": len(expected),
        "prediction_rows": len(actual),
        "covered": len(expected & actual),
        "coverage": float(len(expected & actual) / len(expected)) if expected else 1.0,
        "missing": len(missing),
        "extra": len(extra),
        "missing_sample": [list(key) for key in sorted(missing)[:5]],
        "extra_sample": [list(key) for key in sorted(extra)[:5]],
    }
    if missing or extra:
        raise ValueError(f"Gate prediction coverage mismatch for {label}: {result}")
    return result


def validate_source_mapping(reference: Path | None, candidate: Path) -> tuple[Path, str]:
    candidate_hash = sha256_file(candidate)
    if reference is not None and sha256_file(reference) != candidate_hash:
        raise ValueError(
            "source_id mapping changed between inner gate builds; all temporary outer "
            f"manifests must retain the complete recording set: {reference} vs {candidate}"
        )
    return candidate, candidate_hash


def build_train_predict_gate(
    args: argparse.Namespace,
    runner: CommandRunner,
    protocol_dir: Path,
    tensor_dir: Path,
    training_dir: Path,
    gate_config: ModelConfig,
    outer_fold: int,
    stage_tag: str,
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    tensor_required = [
        tensor_dir / "train.npz",
        tensor_dir / "val.npz",
        tensor_dir / "test.npz",
        tensor_dir / "feature_names.json",
        tensor_dir / "scaler.json",
        tensor_dir / "device_mapping.json",
        tensor_dir / "source_mapping.json",
        tensor_dir / "tensor_metadata.json",
    ]
    build_config = {
        "stage": stage_tag,
        "mode": "ema",
        "scope": args.scope,
        "csv": str(args.csv),
        "csv_sha256": args.csv_sha256,
        "config": str(args.config),
        "config_sha256": args.config_sha256,
        "epoch_manifest_sha256": (
            sha256_file(protocol_dir / "epoch_split_manifest.csv") if not runner.dry_run else None
        ),
        "outer_manifest_sha256": (
            sha256_file(protocol_dir / "recording_split_manifest.csv") if not runner.dry_run else None
        ),
        "builder_code_sha256": args.builder_code_sha256,
        "half_life_seconds": args.causal_half_life_seconds,
        "scaler_mode": "global",
    }
    build_status = run_stage(
        runner,
        tensor_dir / STAGE_MARKER,
        build_config,
        tensor_required,
        builder_command(args, protocol_dir, tensor_dir, "ema"),
        args.skip_existing,
    )
    if not runner.dry_run:
        validate_six_features(tensor_dir)

    checkpoint = training_dir / f"best_band_mean_window_{gate_config.encoder}.pt"
    metrics = training_dir / f"val_metrics_band_mean_window_{gate_config.encoder}.json"
    train_stage_config = {
        "stage": stage_tag,
        "tensor_metadata_sha256": (
            sha256_file(tensor_dir / "tensor_metadata.json") if not runner.dry_run else None
        ),
        "tensor_train_sha256": (
            sha256_file(tensor_dir / "train.npz") if not runner.dry_run else None
        ),
        "tensor_val_sha256": (
            sha256_file(tensor_dir / "val.npz") if not runner.dry_run else None
        ),
        "model": asdict(gate_config),
        "trainer_code_sha256": args.trainer_code_sha256,
        "model_code_sha256": args.model_code_sha256,
        "kept_features": SIX_FEATURE_NAMES,
        "drop_features": SIX_FEATURE_DROP,
    }
    train_status = run_stage(
        runner,
        training_dir / f"train_{STAGE_MARKER}",
        train_stage_config,
        [checkpoint, metrics],
        trainer_command(tensor_dir, training_dir, gate_config),
        args.skip_existing,
    )

    prediction = training_dir / f"test_predictions_band_mean_window_{gate_config.encoder}.csv"
    predict_stage_config = {
        "stage": stage_tag,
        "checkpoint_sha256": sha256_file(checkpoint) if not runner.dry_run else None,
        "tensor_test_sha256": sha256_file(tensor_dir / "test.npz") if not runner.dry_run else None,
        "predict_split": "test",
        "outer_fold": outer_fold,
        "trainer_code_sha256": args.trainer_code_sha256,
        "model_code_sha256": args.model_code_sha256,
    }
    predict_status = run_stage(
        runner,
        training_dir / f"predict_{STAGE_MARKER}",
        predict_stage_config,
        [prediction],
        prediction_command(tensor_dir, training_dir, gate_config, "test", outer_fold),
        args.skip_existing,
    )
    audit: dict[str, Any] = {
        "stage": stage_tag,
        "paths": {
            "protocol": protocol_dir,
            "tensors": tensor_dir,
            "training": training_dir,
            "prediction": prediction,
        },
        "statuses": {
            "build": build_status,
            "train": train_status,
            "predict": predict_status,
        },
    }
    if runner.dry_run:
        return None, audit
    frame = read_gate_predictions(prediction)
    audit["coverage"] = validate_prediction_coverage(frame, tensor_dir / "test.npz", stage_tag)
    audit["probabilities"] = probability_stats(frame, args.causal_normal_threshold)
    audit["source_mapping_sha256"] = sha256_file(tensor_dir / "source_mapping.json")
    return frame, audit


def process_outer_fold(
    args: argparse.Namespace,
    runner: CommandRunner,
    fold: int,
    gate_config: ModelConfig,
    classifier_config: ModelConfig,
) -> dict[str, Any]:
    fold_protocol = args.protocol_dir / f"fold_{fold}"
    recordings, epochs = load_outer_protocol(fold_protocol, args.scope)
    train_recordings = outer_train_recordings(recordings, epochs, args.scope)
    groups = assign_group_folds(train_recordings, args.inner_folds, args.split_seed + fold)
    outer_train_ids = set(train_recordings["recording_id"].astype(int).tolist())
    fold_work = args.work_root / f"fold_{fold}"
    assignment_rows = []
    for group_index, held_ids in enumerate(groups, start=1):
        for recording_id in held_ids:
            row = train_recordings.loc[train_recordings["recording_id"] == recording_id].iloc[0]
            assignment_rows.append(
                {
                    "inner_fold": group_index,
                    "recording_id": int(recording_id),
                    "Environment": str(row["Environment"]),
                    "Scenario": str(row["Scenario"]),
                    "Session": str(row["Session"]),
                    "train_epoch_count": int(row["train_epoch_count"]),
                    "train_positive_epoch_count": int(row["train_positive_epoch_count"]),
                }
            )
    if not runner.dry_run:
        fold_work.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(assignment_rows).to_csv(
            fold_work / "crossfit_recording_assignments.csv",
            index=False,
            encoding="utf-8-sig",
        )

    inner_specs: list[dict[str, Any]] = []
    for inner_fold, held_group in enumerate(groups, start=1):
        held_ids = set(held_group)
        calibration_id = choose_calibration_recording(train_recordings, held_ids)
        fit_ids = outer_train_ids - held_ids - {calibration_id}
        base = fold_work / "crossfit" / f"inner_{inner_fold}"
        protocol = base / "protocol"
        protocol_summary = write_inner_protocol(
            protocol,
            recordings,
            epochs,
            "oof",
            fit_ids,
            calibration_id,
            held_ids,
            runner.dry_run,
        )
        inner_specs.append(
            {
                "inner_fold": inner_fold,
                "held_ids": sorted(held_ids),
                "protocol": protocol,
                "tensor": base / "tensors_ema",
                "training": base / "training_gate",
                "protocol_summary": protocol_summary,
            }
        )

    def execute_inner(spec: dict[str, Any]) -> tuple[int, pd.DataFrame | None, dict[str, Any]]:
        frame, audit = build_train_predict_gate(
            args,
            runner,
            spec["protocol"],
            spec["tensor"],
            spec["training"],
            gate_config,
            fold,
            f"outer_{fold}_oof_{spec['inner_fold']}",
        )
        audit["inner_fold"] = spec["inner_fold"]
        audit["protocol_assignment"] = spec["protocol_summary"]
        return int(spec["inner_fold"]), frame, audit

    oof_results: list[tuple[int, pd.DataFrame | None, dict[str, Any]]] = []
    if args.jobs == 1 or len(inner_specs) == 1:
        oof_results = [execute_inner(spec) for spec in inner_specs]
    else:
        with ThreadPoolExecutor(max_workers=min(args.jobs, len(inner_specs))) as executor:
            futures = {executor.submit(execute_inner, spec): spec for spec in inner_specs}
            for future in as_completed(futures):
                oof_results.append(future.result())
    oof_results.sort(key=lambda item: item[0])

    full_calibration_id = choose_calibration_recording(train_recordings, set())
    full_fit_ids = outer_train_ids - {full_calibration_id}
    full_base = fold_work / "full_gate"
    full_protocol_summary = write_inner_protocol(
        full_base / "protocol",
        recordings,
        epochs,
        "full",
        full_fit_ids,
        full_calibration_id,
        dry_run=runner.dry_run,
    )
    full_frame, full_audit = build_train_predict_gate(
        args,
        runner,
        full_base / "protocol",
        full_base / "tensors_ema",
        full_base / "training_gate",
        gate_config,
        fold,
        f"outer_{fold}_full_gate",
    )
    full_audit["protocol_assignment"] = full_protocol_summary

    fold_audit: dict[str, Any] = {
        "outer_fold": fold,
        "created_utc": utc_now(),
        "outer_protocol": fold_protocol,
        "outer_manifest_hashes": {
            "recording": sha256_file(fold_protocol / "recording_split_manifest.csv"),
            "epoch": sha256_file(fold_protocol / "epoch_split_manifest.csv"),
        },
        "outer_train_recordings": assignment_rows,
        "crossfit_groups": groups,
        "crossfit": [item[2] for item in oof_results],
        "full_gate": full_audit,
        "gate_model": asdict(gate_config),
        "classifier_model": asdict(classifier_config),
        "feature_contract": SIX_FEATURE_NAMES,
        "causal_half_life_seconds": args.causal_half_life_seconds,
        "normal_threshold": args.causal_normal_threshold,
    }
    if runner.dry_run:
        gate_csv = fold_work / "gate_predictions.csv"
        final_tensor = args.tensors_root / f"fold_{fold}"
        final_training = args.training_root / f"fold_{fold}"
        runner.run(builder_command(args, fold_protocol, final_tensor, "gated", gate_csv))
        runner.run(trainer_command(final_tensor, final_training, classifier_config))
        runner.run(
            prediction_command(
                final_tensor, final_training, classifier_config, "test", fold
            )
        )
        fold_audit["merged_gate"] = {"path": gate_csv, "status": "planned"}
        fold_audit["final_classifier"] = {
            "tensor_dir": final_tensor,
            "training_dir": final_training,
            "status": "planned",
        }
        return fold_audit

    reference_mapping: Path | None = None
    reference_hash: str | None = None
    frames: list[pd.DataFrame] = []
    for inner_fold, frame, audit in oof_results:
        assert frame is not None
        mapping = Path(audit["paths"]["tensors"]) / "source_mapping.json"
        reference_mapping, reference_hash = validate_source_mapping(reference_mapping, mapping)
        tagged = frame[[*PREDICTION_KEYS, "prob_normal"]].copy()
        tagged["gate_source"] = f"oof_inner_{inner_fold}"
        frames.append(tagged)
    assert full_frame is not None
    full_mapping = Path(full_audit["paths"]["tensors"]) / "source_mapping.json"
    reference_mapping, reference_hash = validate_source_mapping(reference_mapping, full_mapping)
    tagged_full = full_frame[[*PREDICTION_KEYS, "prob_normal"]].copy()
    tagged_full["gate_source"] = "outer_train_full_gate"
    frames.append(tagged_full)
    merged = pd.concat(frames, ignore_index=True)
    duplicate = merged.duplicated(PREDICTION_KEYS, keep=False)
    if duplicate.any():
        sample = merged.loc[duplicate, [*PREDICTION_KEYS, "gate_source"]].head(10)
        raise ValueError(f"Cross-fit/full gate predictions overlap:\n{sample.to_string(index=False)}")
    probabilities = merged["prob_normal"].to_numpy(dtype=np.float64)
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Merged gate probabilities must be finite and in [0, 1]")
    gate_csv = fold_work / "gate_predictions.csv"
    merged.to_csv(gate_csv, index=False, encoding="utf-8-sig")

    final_tensor = args.tensors_root / f"fold_{fold}"
    final_tensor_required = [
        final_tensor / "train.npz",
        final_tensor / "val.npz",
        final_tensor / "test.npz",
        final_tensor / "feature_names.json",
        final_tensor / "scaler.json",
        final_tensor / "device_mapping.json",
        final_tensor / "source_mapping.json",
        final_tensor / "tensor_metadata.json",
    ]
    final_build_config = {
        "stage": f"outer_{fold}_gated_classifier_tensors",
        "mode": "gated",
        "scope": args.scope,
        "csv_sha256": args.csv_sha256,
        "config_sha256": args.config_sha256,
        "outer_epoch_manifest_sha256": sha256_file(
            fold_protocol / "epoch_split_manifest.csv"
        ),
        "outer_recording_manifest_sha256": sha256_file(
            fold_protocol / "recording_split_manifest.csv"
        ),
        "builder_code_sha256": args.builder_code_sha256,
        "gate_predictions_sha256": sha256_file(gate_csv),
        "half_life_seconds": args.causal_half_life_seconds,
        "normal_threshold": args.causal_normal_threshold,
        "scaler_mode": "global",
    }
    final_build_status = run_stage(
        runner,
        final_tensor / STAGE_MARKER,
        final_build_config,
        final_tensor_required,
        builder_command(args, fold_protocol, final_tensor, "gated", gate_csv),
        args.skip_existing,
    )
    validate_six_features(final_tensor)
    _, final_mapping_hash = validate_source_mapping(reference_mapping, final_tensor / "source_mapping.json")

    expected_by_split = {
        split: expected_keys(final_tensor / f"{split}.npz")
        for split in ("train", "val", "test")
    }
    seen_expected: set[tuple[int, int, int]] = set()
    for split, keys in expected_by_split.items():
        overlap = seen_expected & keys
        if overlap:
            raise ValueError(f"Endpoint keys overlap outer splits ({split}), sample={list(overlap)[:5]}")
        seen_expected.update(keys)
    actual_keys = key_set(merged)
    missing = seen_expected - actual_keys
    extra = actual_keys - seen_expected
    split_coverage = {
        split: {
            "expected_usable_endpoints": len(keys),
            "covered": len(keys & actual_keys),
            "coverage": float(len(keys & actual_keys) / len(keys)) if keys else 1.0,
            "missing": len(keys - actual_keys),
        }
        for split, keys in expected_by_split.items()
    }
    if missing or extra:
        raise ValueError(
            "Merged gate coverage does not exactly match final usable endpoints: "
            f"missing={len(missing)} sample={sorted(missing)[:5]}, "
            f"extra={len(extra)} sample={sorted(extra)[:5]}"
        )
    split_lookup = {
        key: split for split, keys in expected_by_split.items() for key in keys
    }
    audited_gate_csv = fold_work / "gate_predictions_with_split.csv"
    audited_merged = merged.copy()
    audited_merged["outer_split"] = [
        split_lookup[key] for key in key_set_rows(audited_merged)
    ]
    audited_merged.to_csv(audited_gate_csv, index=False, encoding="utf-8-sig")

    final_training = args.training_root / f"fold_{fold}"
    classifier_checkpoint = (
        final_training / f"best_band_mean_window_{classifier_config.encoder}.pt"
    )
    classifier_metrics = (
        final_training / f"val_metrics_band_mean_window_{classifier_config.encoder}.json"
    )
    classifier_train_config = {
        "stage": f"outer_{fold}_gated_classifier_train",
        "tensor_metadata_sha256": sha256_file(final_tensor / "tensor_metadata.json"),
        "tensor_train_sha256": sha256_file(final_tensor / "train.npz"),
        "tensor_val_sha256": sha256_file(final_tensor / "val.npz"),
        "model": asdict(classifier_config),
        "trainer_code_sha256": args.trainer_code_sha256,
        "model_code_sha256": args.model_code_sha256,
        "kept_features": SIX_FEATURE_NAMES,
        "drop_features": SIX_FEATURE_DROP,
    }
    classifier_train_status = run_stage(
        runner,
        final_training / f"train_{STAGE_MARKER}",
        classifier_train_config,
        [classifier_checkpoint, classifier_metrics],
        trainer_command(final_tensor, final_training, classifier_config),
        args.skip_existing,
    )
    final_prediction = (
        final_training
        / f"test_predictions_band_mean_window_{classifier_config.encoder}.csv"
    )
    classifier_predict_config = {
        "stage": f"outer_{fold}_gated_classifier_predict",
        "checkpoint_sha256": sha256_file(classifier_checkpoint),
        "tensor_test_sha256": sha256_file(final_tensor / "test.npz"),
        "predict_split": "test",
        "outer_fold": fold,
        "trainer_code_sha256": args.trainer_code_sha256,
        "model_code_sha256": args.model_code_sha256,
    }
    classifier_predict_status = run_stage(
        runner,
        final_training / f"predict_{STAGE_MARKER}",
        classifier_predict_config,
        [final_prediction],
        prediction_command(
            final_tensor, final_training, classifier_config, "test", fold
        ),
        args.skip_existing,
    )
    final_predictions = read_gate_predictions(final_prediction)
    classifier_coverage = validate_prediction_coverage(
        final_predictions, final_tensor / "test.npz", f"outer_{fold}_classifier_test"
    )
    tensor_metadata = json.loads(
        (final_tensor / "tensor_metadata.json").read_text(encoding="utf-8")
    )
    fold_audit.update(
        {
            "source_mapping_sha256": final_mapping_hash or reference_hash,
            "merged_gate": {
                "path": gate_csv,
                "sha256": sha256_file(gate_csv),
                "annotated_path": audited_gate_csv,
                "annotated_sha256": sha256_file(audited_gate_csv),
                "probabilities": probability_stats(merged, args.causal_normal_threshold),
                "split_coverage": split_coverage,
                "missing": len(missing),
                "extra": len(extra),
            },
            "final_classifier": {
                "tensor_dir": final_tensor,
                "training_dir": final_training,
                "prediction": final_prediction,
                "statuses": {
                    "build": final_build_status,
                    "train": classifier_train_status,
                    "predict": classifier_predict_status,
                },
                "test_coverage": classifier_coverage,
                "causal_construction": tensor_metadata.get("causal_baseline", {}),
                "split_stats": tensor_metadata.get("split_stats", {}),
            },
        }
    )
    write_json(fold_work / "gate_audit_manifest.json", fold_audit)
    return fold_audit


def key_set_rows(frame: pd.DataFrame) -> list[tuple[int, int, int]]:
    return [
        (int(row.source_id), int(row.device_id), int(row.window_time_nanos))
        for row in frame.itertuples(index=False)
    ]


def model_config_from_args(args: argparse.Namespace, prefix: str) -> ModelConfig:
    values = getattr(args, f"{prefix}_class_weight_mult")
    return ModelConfig(
        encoder=getattr(args, f"{prefix}_encoder"),
        hidden_dim=getattr(args, f"{prefix}_hidden_dim"),
        dropout=getattr(args, f"{prefix}_dropout"),
        epochs=getattr(args, f"{prefix}_epochs"),
        batch_size=getattr(args, f"{prefix}_batch_size"),
        learning_rate=getattr(args, f"{prefix}_lr"),
        weight_decay=getattr(args, f"{prefix}_weight_decay"),
        patience=getattr(args, f"{prefix}_patience"),
        seed=getattr(args, f"{prefix}_seed"),
        class_weight_mult=tuple(values) if values is not None else None,
    )


def add_model_arguments(parser: argparse.ArgumentParser, prefix: str, title: str) -> None:
    group = parser.add_argument_group(title)
    option = prefix.replace("_", "-")
    group.add_argument(f"--{option}-encoder", choices=ENCODERS, default="tcn")
    group.add_argument(f"--{option}-hidden-dim", type=int, default=32)
    group.add_argument(f"--{option}-dropout", type=float, default=0.1)
    group.add_argument(f"--{option}-epochs", type=int, default=40)
    group.add_argument(f"--{option}-batch-size", type=int, default=256)
    group.add_argument(f"--{option}-lr", type=float, default=1e-3)
    group.add_argument(f"--{option}-weight-decay", type=float, default=1e-4)
    group.add_argument(f"--{option}-patience", type=int, default=8)
    group.add_argument(f"--{option}-seed", type=int, default=2026)
    group.add_argument(
        f"--{option}-class-weight-mult",
        type=float,
        nargs=4,
        default=None,
        metavar=("NORMAL", "L1", "L5", "L1L5"),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-dir", type=Path, default=None)
    parser.add_argument("--work-root", type=Path, default=None)
    parser.add_argument("--tensors-root", type=Path, default=None)
    parser.add_argument("--training-root", type=Path, default=None)
    parser.add_argument("--csv", type=Path, default=ROOT / "output" / "processed_gnss_data.csv")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "preprocessing.yml")
    parser.add_argument("--scope", choices=("static", "dynamic", "all"), default="static")
    parser.add_argument("--folds", type=int, nargs="*", default=None)
    parser.add_argument(
        "--inner-folds",
        type=int,
        default=4,
        help="Number of recording-grouped OOF gate folds (capped by recording count).",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help=(
            "Concurrent inner gate folds per outer fold. Parallel GPU training can "
            "increase memory use; the default is conservative."
        ),
    )
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--causal-half-life-seconds", type=float, default=60.0)
    parser.add_argument("--causal-normal-threshold", type=float, default=0.8)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse only stages whose required artifacts and exact config marker match.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate protocols and print commands without writing or executing stages.",
    )
    parser.add_argument(
        "--no-aggregate",
        action="store_true",
        help="Do not call script 47 in aggregate-only mode after the selected folds finish.",
    )
    add_model_arguments(parser, "gate", "Gate model")
    add_model_arguments(parser, "classifier", "Final gated classifier")
    args = parser.parse_args()

    # Static and mixed experiments intentionally share the same outer folds;
    # ``--scope`` only filters recordings within that fixed W5 protocol.
    default_protocol = (
        ROOT / "output" / "protocols" / "mixed_timeblock_outer_cv4_w5_v2"
    )
    tag = "mixed_gated" if args.scope == "all" else f"{args.scope}_gated"
    args.protocol_dir = args.protocol_dir or default_protocol
    args.work_root = args.work_root or ROOT / "output" / "causal_gate_crossfit" / tag
    args.tensors_root = args.tensors_root or ROOT / "output" / "tensors" / "causal_relative_v1" / tag
    args.training_root = (
        args.training_root or ROOT / "output" / "training" / "causal_relative_v1" / tag
    )
    for path_name in ("protocol_dir", "csv", "config"):
        path = getattr(args, path_name)
        if not path.exists():
            parser.error(f"--{path_name.replace('_', '-')} does not exist: {path}")
    if args.inner_folds < 2:
        parser.error("--inner-folds must be at least 2")
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.causal_half_life_seconds <= 0:
        parser.error("--causal-half-life-seconds must be positive")
    if not 0 <= args.causal_normal_threshold <= 1:
        parser.error("--causal-normal-threshold must be in [0, 1]")
    for prefix in ("gate", "classifier"):
        if getattr(args, f"{prefix}_hidden_dim") <= 0:
            parser.error(f"--{prefix}-hidden-dim must be positive")
        if not 0 <= getattr(args, f"{prefix}_dropout") < 1:
            parser.error(f"--{prefix}-dropout must be in [0, 1)")
        for suffix in ("epochs", "batch_size", "patience"):
            if getattr(args, f"{prefix}_{suffix}") < 1:
                parser.error(f"--{prefix}-{suffix.replace('_', '-')} must be positive")
        if getattr(args, f"{prefix}_lr") <= 0:
            parser.error(f"--{prefix}-lr must be positive")
        if getattr(args, f"{prefix}_weight_decay") < 0:
            parser.error(f"--{prefix}-weight-decay cannot be negative")
    return args


def main() -> None:
    args = parse_args()
    args.builder_code_sha256 = sha256_file(
        PIPELINE / "45_build_band_mean_window_tensors.py"
    )
    args.trainer_code_sha256 = sha256_file(
        PIPELINE / "46_train_band_mean_multiclass.py"
    )
    args.model_code_sha256 = sha256_file(
        ROOT / "models" / "gnss_signal_baselines.py"
    )
    # The processed table is currently over 1 GB.  Hash it once per invocation,
    # not once per parallel inner fold.
    args.csv_sha256 = None if args.dry_run else sha256_file(args.csv)
    args.config_sha256 = None if args.dry_run else sha256_file(args.config)
    available_fold_ids = fold_ids(args.protocol_dir)
    folds = sorted(args.folds or available_fold_ids)
    available = set(available_fold_ids)
    invalid = sorted(set(folds) - available)
    if invalid:
        raise ValueError(f"Requested folds not found under {args.protocol_dir}: {invalid}")
    if len(folds) != len(set(folds)):
        raise ValueError(f"Duplicate --folds values: {folds}")
    complete_fold_set = set(folds) == available
    aggregate_output_dir = (
        args.training_root
        if complete_fold_set
        else args.training_root
        / f"_subset_folds_{'_'.join(str(fold) for fold in folds)}"
    )
    gate_config = model_config_from_args(args, "gate")
    classifier_config = model_config_from_args(args, "classifier")
    runner = CommandRunner(args.dry_run)
    LOG.info(
        "causal gated CV scope=%s outer_folds=%s inner_folds=%d jobs=%d features=%s",
        args.scope,
        folds,
        args.inner_folds,
        args.jobs,
        SIX_FEATURE_NAMES,
    )
    fold_audits = [
        process_outer_fold(args, runner, fold, gate_config, classifier_config)
        for fold in folds
    ]

    if not args.no_aggregate:
        aggregate_command = [
            sys.executable,
            str(PIPELINE / "47_aggregate_band_mean_cv.py"),
            "--protocol-dir",
            str(args.protocol_dir),
            "--training-root",
            str(args.training_root),
            "--aggregate-output-dir",
            str(aggregate_output_dir),
            "--encoder",
            classifier_config.encoder,
            "--folds",
            *[str(fold) for fold in folds],
            "--aggregate-only",
        ]
        runner.run(aggregate_command)

    run_manifest = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "dry_run": args.dry_run,
        "scope": args.scope,
        "folds": folds,
        "protocol_dir": args.protocol_dir,
        "work_root": args.work_root,
        "tensors_root": args.tensors_root,
        "training_root": args.training_root,
        "aggregate_output_dir": aggregate_output_dir,
        "csv": args.csv,
        "config": args.config,
        "inner_folds": args.inner_folds,
        "jobs": args.jobs,
        "split_seed": args.split_seed,
        "causal_half_life_seconds": args.causal_half_life_seconds,
        "causal_normal_threshold": args.causal_normal_threshold,
        "gate_model": asdict(gate_config),
        "classifier_model": asdict(classifier_config),
        "feature_contract": SIX_FEATURE_NAMES,
        "drop_features": SIX_FEATURE_DROP,
        "code_sha256": {
            "builder": args.builder_code_sha256,
            "trainer": args.trainer_code_sha256,
            "model": args.model_code_sha256,
        },
        "commands": runner.commands,
        "outer_fold_audits": fold_audits,
        "complete_fold_set": complete_fold_set,
    }
    if args.dry_run:
        print(json.dumps(jsonable(run_manifest), indent=2, ensure_ascii=False))
    else:
        manifest_name = (
            "run_manifest.json"
            if complete_fold_set
            else f"run_manifest_folds_{'_'.join(str(fold) for fold in folds)}.json"
        )
        manifest_path = args.work_root / manifest_name
        write_json(manifest_path, run_manifest)
        LOG.info("audit manifest: %s", manifest_path)


if __name__ == "__main__":
    main()

"""Evaluate a locked mixed signal-fusion checkpoint on auditable test groups.

This script is evaluation-only.  It loads one already selected checkpoint and
the paired ``test`` raw/stats tensors, predicts with the model's unchanged
argmax decision rule, and reports pooled metrics for:

* the complete test set;
* static and dynamic motion subsets;
* each Scenario and complete recording Session; and
* receiver device x motion x signal-band groups.

The recording metadata comes from ``window_trace_index.json``.  A separate
JSON report summarizes the per-Session rows with equal Session weight so that
long static recordings cannot dominate short dynamic recordings.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import SignalRawStatsFusion


def _load_static_group_evaluator():
    """Reuse the feature-contract and metric implementation from pipeline 23."""

    path = Path(__file__).with_name("23_evaluate_static_fusion_groups.py")
    spec = importlib.util.spec_from_file_location("_static_fusion_group_evaluator", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = _load_static_group_evaluator()
IGNORE_INDEX = base.IGNORE_INDEX
REQUIRED_ARRAYS = base.REQUIRED_ARRAYS
PAIRED_METADATA = REQUIRED_ARRAYS - {"x"}
TRACE_ARRAYS = {"recording_id"}
GROUP_METRICS = ("macro_f1", "precision", "recall", "far", "specificity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    return args


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def _validate_checkpoint(checkpoint: Any, path: Path) -> dict[str, Any]:
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(f"Invalid fusion checkpoint: {path}")
    required = {
        "encoder",
        "raw_time_steps",
        "raw_feature_names",
        "stats_input_dim",
        "stats_feature_names",
        "hidden_dim",
        "dropout",
    }
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise ValueError(f"Checkpoint missing metadata: {missing}")
    return checkpoint


def _fold_component(path: Path) -> int | None:
    folds = {
        int(match.group(1))
        for part in path.resolve().parts
        for match in [re.fullmatch(r"fold_(\d+)", part)]
        if match is not None
    }
    if len(folds) > 1:
        raise ValueError(f"Path contains conflicting fold components: {path}")
    return next(iter(folds), None)


def _validate_checkpoint_data_binding(
    checkpoint: dict[str, Any], checkpoint_path: Path, data_dir: Path, raw_dir: Path
) -> None:
    """Reject a checkpoint selected on another fold before opening test data."""

    checkpoint_fold = _fold_component(checkpoint_path)
    data_fold = _fold_component(data_dir)
    if checkpoint_fold is not None or data_fold is not None:
        if checkpoint_fold != data_fold:
            raise ValueError(
                f"Checkpoint fold {checkpoint_fold} differs from tensor fold {data_fold}"
            )

    recorded = checkpoint.get("val_metrics")
    required = {"samples", "negative_support", "positive_support"}
    if not isinstance(recorded, dict) or not required.issubset(recorded):
        raise ValueError(
            "Checkpoint lacks validation support metadata needed to bind it to this tensor fold"
        )
    val_path = raw_dir / "val.npz"
    with np.load(val_path, allow_pickle=False) as val:
        missing = {"mask", "y"}.difference(val.files)
        if missing:
            raise ValueError(f"{val_path} lacks validation binding arrays: {sorted(missing)}")
        mask = np.asarray(val["mask"]).astype(bool)
        labels = np.asarray(val["y"])
    if mask.shape != labels.shape:
        raise ValueError(f"Validation mask/label shapes differ in {val_path}")
    active_labels = labels[mask & (labels != IGNORE_INDEX)]
    unexpected = sorted(set(np.unique(active_labels).tolist()).difference({0, 1}))
    if unexpected:
        raise ValueError(f"Validation labels must be binary; found {unexpected}")
    observed = {
        "samples": int(active_labels.size),
        "negative_support": int((active_labels == 0).sum()),
        "positive_support": int((active_labels == 1).sum()),
    }
    expected = {key: int(recorded[key]) for key in required}
    if observed != expected:
        raise ValueError(
            "Checkpoint validation support differs from the selected tensor fold: "
            f"checkpoint={expected}, tensors={observed}"
        )


def _broadcast_window_metadata(values: np.ndarray, shape: tuple[int, int], name: str) -> np.ndarray:
    if values.ndim == 1 and values.shape[0] == shape[0]:
        return np.broadcast_to(values[:, None], shape)
    if values.shape == shape:
        return values
    raise ValueError(f"Unexpected {name} shape {values.shape}; labels={shape}")


def _load_trace_recordings(path: Path) -> list[dict[str, Any]]:
    trace = _load_json(path)
    if not isinstance(trace, dict) or not isinstance(trace.get("recordings"), list):
        raise ValueError(f"{path} must contain a recordings list")
    recordings = trace["recordings"]
    required = {"Environment", "Scenario", "Session"}
    normalized: list[dict[str, Any]] = []
    identities: set[tuple[str, str, str]] = set()
    for index, recording in enumerate(recordings):
        if not isinstance(recording, dict):
            raise ValueError(f"Recording {index} in {path} is not an object")
        missing = required.difference(recording)
        if missing:
            raise ValueError(f"Recording {index} in {path} is missing {sorted(missing)}")
        item = dict(recording)
        for key in required:
            item[key] = str(item[key])
        identity = tuple(item[key] for key in ("Environment", "Scenario", "Session"))
        if identity in identities:
            raise ValueError(f"Duplicate recording identity in {path}: {identity}")
        identities.add(identity)
        normalized.append(item)
    if not normalized:
        raise ValueError(f"{path} contains no recordings")
    return normalized


def _predict(
    model: torch.nn.Module,
    raw_x: np.ndarray,
    stats_x: np.ndarray,
    raw_indices: list[int],
    stats_indices: list[int],
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    predictions = np.zeros(raw_x.shape[:2], dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(raw_x), batch_size):
            stop = min(start + batch_size, len(raw_x))
            raw_batch = torch.from_numpy(raw_x[start:stop, ..., raw_indices].copy()).float().to(device)
            stats_batch = torch.from_numpy(stats_x[start:stop, ..., stats_indices].copy()).float().to(device)
            predictions[start:stop] = model(raw_batch, stats_batch).argmax(-1).cpu().numpy()
    return predictions


def _group_row(
    level: str,
    selector: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    recording_ids: np.ndarray,
    raw_feature_set: str,
    stats_feature_set: str,
    **dimensions: object,
) -> dict[str, object]:
    row: dict[str, object] = {
        "group_level": level,
        "Environment": "ALL",
        "Scenario": "ALL",
        "Session": "ALL",
        "motion": "ALL",
        "device_id": "ALL",
        "device_name": "ALL",
        "band": "ALL",
        "session_count": int(np.unique(recording_ids[selector]).size),
        "raw_feature_set": raw_feature_set,
        "stats_feature_set": stats_feature_set,
    }
    row.update(dimensions)
    row.update(base.metrics(y_true[selector], y_pred[selector]))
    return row


def _summary_statistics(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array):
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _equal_session_summary(session_rows: list[dict[str, object]]) -> dict[str, Any]:
    def summarize(rows: list[dict[str, object]]) -> dict[str, Any]:
        return {
            "session_count": len(rows),
            "positive_session_count": sum(int(row["positive_support"]) > 0 for row in rows),
            "negative_session_count": sum(int(row["negative_support"]) > 0 for row in rows),
            "metrics": {
                metric: _summary_statistics(float(row[metric]) for row in rows)
                for metric in GROUP_METRICS
            },
        }

    result: dict[str, Any] = {
        "aggregation": "unweighted arithmetic summary of complete-Session metric rows",
        "zero_division": "0, matching pooled group metrics",
        "overall": summarize(session_rows),
        "by_motion": {},
    }
    for motion in ("static", "dynamic"):
        rows = [row for row in session_rows if row["motion"] == motion]
        result["by_motion"][motion] = summarize(rows)
    return result


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = _validate_checkpoint(
        torch.load(args.checkpoint, map_location=device, weights_only=False), args.checkpoint
    )
    raw_dir = args.data_dir / "raw"
    stats_dir = args.data_dir / "stats"
    raw_names = base.load_feature_names(raw_dir / "feature_names.json")
    stats_names = base.load_feature_names(stats_dir / "feature_names.json")
    raw_indices = base.select_indices(raw_names, checkpoint["raw_feature_names"], "raw")
    stats_indices = base.select_indices(stats_names, checkpoint["stats_feature_names"], "stats")
    raw_input_dim = int(checkpoint.get("raw_input_dim", len(raw_indices)))
    if raw_input_dim != len(raw_indices):
        raise ValueError("Checkpoint raw dimension differs from its recorded feature names")
    if int(checkpoint["stats_input_dim"]) != len(stats_indices):
        raise ValueError("Checkpoint stats dimension differs from its recorded feature names")
    if "IsL5" not in stats_names:
        raise ValueError("Full stats tensor lacks IsL5 sidecar needed for band stratification")
    is_l5_index = stats_names.index("IsL5")
    _validate_checkpoint_data_binding(
        checkpoint, args.checkpoint, args.data_dir, raw_dir
    )

    raw_path = raw_dir / "test.npz"
    stats_path = stats_dir / "test.npz"
    with np.load(raw_path, allow_pickle=False) as raw, np.load(stats_path, allow_pickle=False) as stats:
        raw_missing = (REQUIRED_ARRAYS | TRACE_ARRAYS).difference(raw.files)
        stats_missing = REQUIRED_ARRAYS.difference(stats.files)
        if raw_missing or stats_missing:
            raise ValueError(f"Missing test arrays: raw={sorted(raw_missing)}, stats={sorted(stats_missing)}")
        for name in PAIRED_METADATA:
            if not np.array_equal(raw[name], stats[name]):
                raise ValueError(f"Raw/stats {name} metadata mismatch")
        raw_x = np.asarray(raw["x"])
        stats_x = np.asarray(stats["x"])
        mask = np.asarray(raw["mask"]).astype(bool)
        labels = np.asarray(raw["y"])
        is_dynamic_values = np.asarray(raw["is_dynamic"]).astype(bool)
        device_ids = np.asarray(raw["device_id"])
        window_recording_ids = np.asarray(raw["recording_id"])

    if raw_x.ndim != 4 or stats_x.ndim != 4 or stats_x.shape[-2] != 1:
        raise ValueError(f"Unexpected test shapes raw={raw_x.shape}, stats={stats_x.shape}")
    if raw_x.shape[:2] != stats_x.shape[:2] or mask.shape != raw_x.shape[:2] or labels.shape != raw_x.shape[:2]:
        raise ValueError("Test tensors and label metadata have incompatible shapes")
    if raw_x.shape[-1] != len(raw_names) or stats_x.shape[-1] != len(stats_names):
        raise ValueError("Test feature tensor widths differ from feature metadata")
    if raw_x.shape[-2] != int(checkpoint["raw_time_steps"]):
        raise ValueError("Checkpoint time-step count differs from test tensor")
    if window_recording_ids.ndim != 1 or window_recording_ids.shape[0] != raw_x.shape[0]:
        raise ValueError(
            f"recording_id must have shape [B]={raw_x.shape[0]}, got {window_recording_ids.shape}"
        )
    if not np.issubdtype(window_recording_ids.dtype, np.integer):
        raise ValueError("recording_id must use an integer dtype")

    recordings = _load_trace_recordings(args.data_dir / "window_trace_index.json")
    if window_recording_ids.size and (
        int(window_recording_ids.min()) < 0 or int(window_recording_ids.max()) >= len(recordings)
    ):
        raise ValueError("test recording_id contains an index absent from window_trace_index.json")
    scenario_by_window = np.asarray(
        [recordings[int(index)]["Scenario"] for index in window_recording_ids], dtype=object
    )
    scenario_text = scenario_by_window.astype(str)
    expected_dynamic = np.char.startswith(scenario_text, "dy_")
    expected_static = np.char.startswith(scenario_text, "st_")
    if not np.all(expected_dynamic | expected_static):
        unknown = sorted(set(scenario_text[~(expected_dynamic | expected_static)].tolist()))
        raise ValueError(f"Trace index contains Scenario values outside st_/dy_: {unknown}")
    dynamic_grid = _broadcast_window_metadata(is_dynamic_values, labels.shape, "is_dynamic")
    expected_dynamic_grid = np.broadcast_to(expected_dynamic[:, None], labels.shape)
    if not np.array_equal(dynamic_grid, expected_dynamic_grid):
        raise ValueError("is_dynamic sidecar disagrees with Scenario in window_trace_index.json")

    model = SignalRawStatsFusion(
        raw_input_dim=len(raw_indices),
        stats_input_dim=len(stats_indices),
        encoder=str(checkpoint["encoder"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        dropout=float(checkpoint["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    prediction_grid = _predict(
        model, raw_x, stats_x, raw_indices, stats_indices, args.batch_size, device
    )

    device_grid = _broadcast_window_metadata(device_ids, labels.shape, "device_id")
    recording_grid = np.broadcast_to(window_recording_ids[:, None], labels.shape)
    active = mask & (labels != IGNORE_INDEX)
    if not np.any(active):
        raise ValueError("Test split has no active labels")
    unexpected_labels = sorted(set(np.unique(labels[active]).tolist()).difference({0, 1}))
    if unexpected_labels:
        raise ValueError(f"Active test labels must be binary; found {unexpected_labels}")
    band_values = stats_x[..., 0, is_l5_index]
    valid_band_values = np.isfinite(band_values) & (
        np.isclose(band_values, 0.0) | np.isclose(band_values, 1.0)
    )
    if not np.all(valid_band_values[active]):
        raise ValueError("Active IsL5 sidecar values must be finite physical 0/1 values")
    is_l5 = band_values >= 0.5

    mapping = _load_json(args.data_dir / "device_mapping.json")
    if not isinstance(mapping, dict):
        raise ValueError("device_mapping.json must contain an object")
    inverse_mapping = {int(value): str(name) for name, value in mapping.items()}
    if len(inverse_mapping) != len(mapping):
        raise ValueError("device_mapping.json assigns the same device id more than once")
    active_device_ids = device_grid[active].astype(np.int64)
    unknown_devices = sorted(set(active_device_ids.tolist()).difference(inverse_mapping))
    if unknown_devices:
        raise ValueError(f"Test tensors contain device ids absent from device_mapping.json: {unknown_devices}")

    y_true = labels[active].astype(np.int64)
    y_pred = prediction_grid[active].astype(np.int64)
    active_recordings = recording_grid[active].astype(np.int64)
    active_dynamic = dynamic_grid[active].astype(bool)
    active_band = np.where(is_l5[active], "L5", "L1")
    active_scenario = np.asarray(
        [recordings[index]["Scenario"] for index in active_recordings], dtype=object
    )
    raw_feature_set = str(checkpoint.get("raw_feature_set", "legacy_or_unspecified"))
    stats_feature_set = str(checkpoint.get("stats_feature_set", "legacy_or_unspecified"))
    all_rows = np.ones(len(y_true), dtype=bool)
    rows: list[dict[str, object]] = [
        _group_row(
            "overall", all_rows, y_true, y_pred, active_recordings,
            raw_feature_set, stats_feature_set,
        )
    ]

    for dynamic, motion in ((False, "static"), (True, "dynamic")):
        selector = active_dynamic == dynamic
        if selector.any():
            rows.append(_group_row(
                "motion", selector, y_true, y_pred, active_recordings,
                raw_feature_set, stats_feature_set, motion=motion,
            ))

    for scenario in sorted(np.unique(active_scenario).tolist()):
        selector = active_scenario == scenario
        motion = "dynamic" if str(scenario).startswith("dy_") else "static"
        rows.append(_group_row(
            "scenario", selector, y_true, y_pred, active_recordings,
            raw_feature_set, stats_feature_set, Scenario=str(scenario), motion=motion,
        ))

    session_rows: list[dict[str, object]] = []
    for recording_id in sorted(np.unique(active_recordings).tolist()):
        selector = active_recordings == recording_id
        recording = recordings[int(recording_id)]
        motion = "dynamic" if str(recording["Scenario"]).startswith("dy_") else "static"
        row = _group_row(
            "session", selector, y_true, y_pred, active_recordings,
            raw_feature_set, stats_feature_set,
            Environment=recording["Environment"], Scenario=recording["Scenario"],
            Session=recording["Session"], motion=motion,
        )
        rows.append(row)
        session_rows.append(row)

    for device_id in sorted(np.unique(active_device_ids).tolist()):
        device_name = inverse_mapping[int(device_id)]
        device_selector = active_device_ids == device_id
        for dynamic, motion in ((False, "static"), (True, "dynamic")):
            for band in ("L1", "L5"):
                selector = device_selector & (active_dynamic == dynamic) & (active_band == band)
                if not selector.any():
                    continue
                rows.append(_group_row(
                    "device_motion_band", selector, y_true, y_pred, active_recordings,
                    raw_feature_set, stats_feature_set, device_id=device_id,
                    device_name=device_name, motion=motion, band=band,
                ))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    group_path = args.output_dir / "test_metrics_mixed_groups.csv"
    pd.DataFrame(rows).to_csv(group_path, index=False, encoding="utf-8-sig")
    session_summary = _equal_session_summary(session_rows)
    session_summary.update({
        "checkpoint": str(args.checkpoint),
        "decision_rule": "argmax of the locked two-class checkpoint; no threshold tuning",
        "raw_feature_names": [str(name) for name in checkpoint["raw_feature_names"]],
        "stats_feature_names": [str(name) for name in checkpoint["stats_feature_names"]],
    })
    summary_path = args.output_dir / "test_metrics_session_equal_weight.json"
    summary_path.write_text(json.dumps(session_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"wrote {group_path} ({len(rows)} rows)")
    print(f"wrote {summary_path} ({len(session_rows)} Sessions)")
    print(json.dumps(rows[0], ensure_ascii=False))


if __name__ == "__main__":
    main()

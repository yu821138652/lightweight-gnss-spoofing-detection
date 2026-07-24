"""Evaluate a locked static raw-plus-stats fusion checkpoint by device and band.

This is intentionally separate from training.  It reads the complete test
tensor metadata as a sidecar, while loading only the raw/stats feature names
recorded in the checkpoint.  Consequently a feature removed from the model
(for example ``IsL5``) can still be used safely to stratify evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import SignalRawStatsFusion


IGNORE_INDEX = -100
REQUIRED_ARRAYS = {"x", "mask", "y", "is_dynamic", "device_id"}


def load_feature_names(path: Path) -> list[str]:
    try:
        names = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON feature metadata: {path}") from exc
    if not isinstance(names, list) or not names or not all(isinstance(name, str) for name in names):
        raise ValueError(f"{path} must contain a non-empty list of feature names")
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate feature names in {path}")
    return names


def select_indices(available: list[str], selected: Any, kind: str) -> list[int]:
    if not isinstance(selected, list) or not selected or not all(isinstance(name, str) for name in selected):
        raise ValueError(f"Checkpoint {kind}_feature_names must be a non-empty name list")
    missing = [name for name in selected if name not in available]
    if missing:
        raise ValueError(f"Checkpoint {kind} feature(s) absent from current tensors: {missing}")
    return [available.index(name) for name in selected]


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    if y_true.size == 0:
        return {
            "samples": 0, "negative_support": 0, "positive_support": 0,
            "tn": 0, "fp": 0, "fn": 0, "tp": 0,
            "precision": 0.0, "recall": 0.0, "far": 0.0, "macro_f1": 0.0,
        }
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    negative_support = tn + fp
    positive_support = fn + tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / positive_support if positive_support else 0.0
    negative_recall = tn / negative_support if negative_support else 0.0
    positive_f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    negative_f1 = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.0
    return {
        "samples": int(y_true.size),
        "negative_support": negative_support,
        "positive_support": positive_support,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "precision": precision,
        "recall": recall,
        "far": fp / negative_support if negative_support else 0.0,
        "macro_f1": (positive_f1 + negative_f1) / 2,
        "specificity": negative_recall,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(f"Invalid fusion checkpoint: {args.checkpoint}")
    required_checkpoint = {
        "encoder", "raw_time_steps", "raw_feature_names", "stats_input_dim",
        "stats_feature_names", "hidden_dim", "dropout",
    }
    missing_checkpoint = sorted(required_checkpoint.difference(checkpoint))
    if missing_checkpoint:
        raise ValueError(f"Checkpoint missing metadata: {missing_checkpoint}")

    raw_dir = args.data_dir / "raw"
    stats_dir = args.data_dir / "stats"
    raw_names = load_feature_names(raw_dir / "feature_names.json")
    stats_names = load_feature_names(stats_dir / "feature_names.json")
    raw_indices = select_indices(raw_names, checkpoint["raw_feature_names"], "raw")
    stats_indices = select_indices(stats_names, checkpoint["stats_feature_names"], "stats")
    if len(stats_indices) != int(checkpoint["stats_input_dim"]):
        raise ValueError("Checkpoint stats dimension differs from its recorded feature names")
    if "IsL5" not in stats_names:
        raise ValueError("Full stats tensor lacks IsL5 sidecar needed for band stratification")
    is_l5_index = stats_names.index("IsL5")

    raw_path = raw_dir / "test.npz"
    stats_path = stats_dir / "test.npz"
    with np.load(raw_path, allow_pickle=False) as raw, np.load(stats_path, allow_pickle=False) as stats:
        raw_missing = REQUIRED_ARRAYS.difference(raw.files)
        stats_missing = REQUIRED_ARRAYS.difference(stats.files)
        if raw_missing or stats_missing:
            raise ValueError(f"Missing test arrays: raw={sorted(raw_missing)}, stats={sorted(stats_missing)}")
        for name in ("mask", "y", "is_dynamic", "device_id"):
            if not np.array_equal(raw[name], stats[name]):
                raise ValueError(f"Raw/stats {name} metadata mismatch")
        raw_x = np.asarray(raw["x"])
        stats_x = np.asarray(stats["x"])
        mask = np.asarray(raw["mask"]).astype(bool)
        labels = np.asarray(raw["y"])
        device_ids = np.asarray(raw["device_id"])

    if raw_x.ndim != 4 or stats_x.ndim != 4 or stats_x.shape[-2] != 1:
        raise ValueError(f"Unexpected test shapes raw={raw_x.shape}, stats={stats_x.shape}")
    if raw_x.shape[:2] != stats_x.shape[:2] or mask.shape != raw_x.shape[:2] or labels.shape != raw_x.shape[:2]:
        raise ValueError("Test tensors and label metadata have incompatible shapes")
    if raw_x.shape[-1] != len(raw_names) or stats_x.shape[-1] != len(stats_names):
        raise ValueError("Test feature tensor widths differ from feature metadata")
    if raw_x.shape[-2] != int(checkpoint["raw_time_steps"]):
        raise ValueError("Checkpoint time-step count differs from test tensor")

    model = SignalRawStatsFusion(
        raw_input_dim=len(raw_indices),
        stats_input_dim=len(stats_indices),
        encoder=str(checkpoint["encoder"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        dropout=float(checkpoint["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    prediction_grid = np.zeros(labels.shape, dtype=np.int64)
    with torch.no_grad():
        for start in range(0, len(raw_x), args.batch_size):
            stop = min(start + args.batch_size, len(raw_x))
            raw_batch = torch.from_numpy(raw_x[start:stop, ..., raw_indices].copy()).float().to(device)
            stats_batch = torch.from_numpy(stats_x[start:stop, ..., stats_indices].copy()).float().to(device)
            prediction_grid[start:stop] = model(raw_batch, stats_batch).argmax(-1).cpu().numpy()

    if device_ids.ndim == 1 and device_ids.shape[0] == labels.shape[0]:
        device_grid = np.broadcast_to(device_ids[:, None], labels.shape)
    elif device_ids.shape == labels.shape:
        device_grid = device_ids
    else:
        raise ValueError(f"Unexpected device_id shape {device_ids.shape}; labels={labels.shape}")
    is_l5 = stats_x[..., 0, is_l5_index] >= 0.5
    active = mask & (labels != IGNORE_INDEX)
    if not np.any(active):
        raise ValueError("Test split has no active labels")

    mapping_path = args.data_dir / "device_mapping.json"
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    if not isinstance(mapping, dict):
        raise ValueError(f"Invalid device mapping: {mapping_path}")
    inverse_mapping = {int(value): str(name) for name, value in mapping.items()}

    records: list[dict[str, object]] = []
    device_active = device_grid[active]
    band_active = np.where(is_l5[active], "L5", "L1")
    y_true = labels[active].astype(np.int64)
    y_pred = prediction_grid[active]
    raw_feature_set = str(checkpoint.get("raw_feature_set", "full"))
    stats_feature_set = str(checkpoint.get("stats_feature_set", "legacy_or_unspecified"))
    for device_id in sorted(np.unique(device_active).tolist()):
        device_mask = device_active == device_id
        for band in ("L1", "L5"):
            group_mask = device_mask & (band_active == band)
            if not np.any(group_mask):
                continue
            row: dict[str, object] = {
                "device_id": int(device_id),
                "device_name": inverse_mapping.get(int(device_id), f"unknown_{device_id}"),
                "band": band,
                "raw_feature_set": raw_feature_set,
                "stats_feature_set": stats_feature_set,
            }
            row.update(metrics(y_true[group_mask], y_pred[group_mask]))
            records.append(row)
    total: dict[str, object] = {
        "device_id": "ALL", "device_name": "ALL", "band": "ALL",
        "raw_feature_set": raw_feature_set,
        "stats_feature_set": stats_feature_set,
    }
    total.update(metrics(y_true, y_pred))
    records.append(total)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "test_metrics_by_device_band.csv"
    pd.DataFrame(records).to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"wrote {out_path} ({len(records)} rows)")
    print(json.dumps(total, ensure_ascii=False))


if __name__ == "__main__":
    main()

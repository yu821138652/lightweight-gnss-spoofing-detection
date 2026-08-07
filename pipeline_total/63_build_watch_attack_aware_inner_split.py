#!/usr/bin/env python3
"""Build a leakage-free, attack-aware inner split for the Watch expert.

This utility repairs an outer fold whose standard inner validation partition
contains no Watch ``anomaly`` samples.  It uses *only* the outer-development
``train.npz`` and ``val.npz`` tensors: a contiguous late portion of each
eligible Watch source's anomaly interval plus a post-attack normal portion is
reserved for the new inner validation split.  A configurable temporal guard
band is removed from training around that held-out interval.  The outer
``test.npz`` is copied unchanged and is never inspected to choose a split.

The resulting directory has the regular train/val/test tensor layout and can
be passed directly to 37_train_device_attack_event.py and the Watch CV runner.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np


WATCH_NAMES = ("Google_Pixel_Watch1", "Google_Pixel_Watch2")
REQUIRED = {
    "x", "y_event", "y_response_state", "device_id", "recording_id", "source_id", "endpoint_tow",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--watch-devices", nargs="+", default=list(WATCH_NAMES))
    parser.add_argument(
        "--positive-windows-per-source", type=int, default=200,
        help="late anomaly windows held out from every eligible Watch source",
    )
    parser.add_argument(
        "--post-attack-normal-windows-per-source", type=int, default=100,
        help="normal windows after the final anomaly window held out from every eligible Watch source",
    )
    parser.add_argument(
        "--guard-windows", type=int, default=30,
        help="development-training windows excluded on either side of each held-out interval",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if min(args.positive_windows_per_source, args.post_attack_normal_windows_per_source) < 1:
        parser.error("per-source held-out window counts must be positive")
    if args.guard_windows < 0:
        parser.error("--guard-windows must be non-negative")
    return args


def read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        missing = REQUIRED.difference(data.files)
        if missing:
            raise ValueError(f"{path} is missing {sorted(missing)}")
        return {key: data[key].copy() for key in data.files}


def concat_rows(first: dict[str, np.ndarray], second: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if set(first) != set(second):
        raise ValueError("Development train/val NPZ keys differ")
    result: dict[str, np.ndarray] = {}
    for key in first:
        if first[key].ndim == 0 or second[key].ndim == 0:
            raise ValueError(f"Unsupported scalar tensor key: {key}")
        if first[key].shape[1:] != second[key].shape[1:]:
            raise ValueError(f"Incompatible tensor shape for {key}")
        result[key] = np.concatenate([first[key], second[key]], axis=0)
    return result


def subset_rows(data: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    if mask.dtype != bool or mask.ndim != 1 or len(mask) != len(data["y_response_state"]):
        raise ValueError("Invalid row-selection mask")
    return {key: values[mask] for key, values in data.items()}


def select_source_holdout(
    data: dict[str, np.ndarray],
    source_mask: np.ndarray,
    positive_windows: int,
    post_attack_normal_windows: int,
    guard_windows: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Select late anomaly plus post-attack normal blocks from one source."""
    source_indices = np.flatnonzero(source_mask)
    order = source_indices[np.argsort(data["endpoint_tow"][source_indices], kind="stable")]
    state = data["y_response_state"][order].astype(np.int64)
    positive_positions = np.flatnonzero(state == 1)
    if len(positive_positions) < positive_windows:
        raise ValueError(
            f"source_id={int(data['source_id'][order[0]])} has only {len(positive_positions)} anomaly windows; "
            f"need {positive_windows}"
        )
    last_positive = int(positive_positions[-1])
    post_normal_positions = np.flatnonzero((state == 0) & (np.arange(len(state)) > last_positive))
    if len(post_normal_positions) < post_attack_normal_windows:
        raise ValueError(
            f"source_id={int(data['source_id'][order[0]])} has only {len(post_normal_positions)} post-attack normal windows; "
            f"need {post_attack_normal_windows}"
        )
    selected_positions = np.concatenate([
        positive_positions[-positive_windows:],
        post_normal_positions[-post_attack_normal_windows:],
    ])
    selected_positions.sort()
    selected_rows = order[selected_positions]
    heldout = np.zeros(len(data["y_response_state"]), dtype=bool)
    heldout[selected_rows] = True

    # Exclude source-local windows directly adjacent to the held-out temporal
    # block(s), so a nearly identical sliding window cannot occur in train.
    guarded = np.zeros_like(heldout)
    first = max(0, int(selected_positions[0]) - guard_windows)
    last = min(len(order), int(selected_positions[-1]) + guard_windows + 1)
    guarded[order[first:last]] = True
    train_excluded = guarded
    selected_tow = data["endpoint_tow"][selected_rows]
    return heldout, train_excluded, {
        "recording_id": int(data["recording_id"][order[0]]),
        "device_id": int(data["device_id"][order[0]]),
        "source_id": int(data["source_id"][order[0]]),
        "calibration_positive_windows": int((data["y_response_state"][selected_rows] == 1).sum()),
        "calibration_post_attack_normal_windows": int((data["y_response_state"][selected_rows] == 0).sum()),
        "calibration_tow_start": float(selected_tow.min()),
        "calibration_tow_end": float(selected_tow.max()),
        "training_guard_excluded_windows": int(guarded.sum()),
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Output directory is not empty: {args.output_dir}; use --overwrite")
        shutil.rmtree(args.output_dir)
    for name in ("train.npz", "val.npz", "test.npz", "metadata.json"):
        path = args.source_data_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)

    metadata = json.loads((args.source_data_dir / "metadata.json").read_text(encoding="utf-8"))
    device_mapping = {str(name): int(value) for name, value in metadata.get("device_mapping", {}).items()}
    unknown = sorted(set(args.watch_devices).difference(device_mapping))
    if unknown:
        raise ValueError(f"Unknown Watch devices {unknown}; available={sorted(device_mapping)}")
    watch_ids = np.asarray([device_mapping[name] for name in args.watch_devices], dtype=np.int64)

    original_train = read_npz(args.source_data_dir / "train.npz")
    original_val = read_npz(args.source_data_dir / "val.npz")
    development = concat_rows(original_train, original_val)
    raw_state = development["y_response_state"].astype(np.int64)
    is_watch = np.isin(development["device_id"].astype(np.int64), watch_ids)
    candidate = is_watch & np.isin(raw_state, (0, 1))
    if not candidate.any():
        raise ValueError("Outer development contains no Watch normal/anomaly windows")

    holdout = np.zeros(len(raw_state), dtype=bool)
    train_excluded = np.zeros(len(raw_state), dtype=bool)
    source_audit: list[dict[str, Any]] = []
    source_keys = np.unique(
        np.column_stack([
            development["recording_id"][candidate],
            development["device_id"][candidate],
            development["source_id"][candidate],
        ]), axis=0,
    )
    for recording_id, device_id, source_id in source_keys:
        source_mask = (
            candidate
            & (development["recording_id"] == recording_id)
            & (development["device_id"] == device_id)
            & (development["source_id"] == source_id)
        )
        if not np.any(raw_state[source_mask] == 1):
            continue
        selected, guarded, audit = select_source_holdout(
            development, source_mask, args.positive_windows_per_source,
            args.post_attack_normal_windows_per_source, args.guard_windows,
        )
        holdout |= selected
        train_excluded |= guarded
        source_audit.append(audit)
    if not source_audit:
        raise ValueError("No outer-development Watch source contains anomaly windows for calibration")

    train_mask = ~train_excluded
    val_mask = holdout
    train = subset_rows(development, train_mask)
    val = subset_rows(development, val_mask)
    train_watch = np.isin(train["device_id"], watch_ids) & np.isin(train["y_response_state"], (0, 1))
    val_watch = np.isin(val["device_id"], watch_ids) & np.isin(val["y_response_state"], (0, 1))
    train_positive = int((train["y_response_state"][train_watch] == 1).sum())
    val_positive = int((val["y_response_state"][val_watch] == 1).sum())
    val_negative = int((val["y_response_state"][val_watch] == 0).sum())
    if train_positive == 0 or val_positive == 0 or val_negative == 0:
        raise RuntimeError(
            "Attack-aware split lacks required Watch support: "
            f"train_positive={train_positive} val_negative={val_negative} val_positive={val_positive}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "train.npz", **train)
    np.savez_compressed(args.output_dir / "val.npz", **val)
    shutil.copy2(args.source_data_dir / "test.npz", args.output_dir / "test.npz")
    metadata = dict(metadata)
    metadata["inner_split_protocol"] = "attack_aware_watch_time_block_v1"
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    audit = {
        "protocol": "attack_aware_watch_time_block_v1",
        "source_data_dir": str(args.source_data_dir),
        "outer_test_used_for_selection": False,
        "watch_devices": list(args.watch_devices),
        "positive_windows_per_source": int(args.positive_windows_per_source),
        "post_attack_normal_windows_per_source": int(args.post_attack_normal_windows_per_source),
        "guard_windows": int(args.guard_windows),
        "source_blocks": source_audit,
        "support": {
            "train_watch_anomaly": train_positive,
            "val_watch_normal": val_negative,
            "val_watch_anomaly": val_positive,
        },
        "rows": {
            "train": int(len(train["y_response_state"])),
            "val": int(len(val["y_response_state"])),
            "outer_test": "copied_unchanged_not_used_for_selection",
        },
    }
    (args.output_dir / "attack_aware_inner_split_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(args.output_dir), "support": audit["support"], "source_blocks": source_audit}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

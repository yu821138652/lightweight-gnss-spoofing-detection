"""Create a diagnostic response-state split from an existing tensor directory.

The current fold-6 inner split keeps the only manually reviewable L5 Watch
response in validation, while the playground L5 session is the outer test.
That is correct for event-model selection, but it leaves the response-state
classifier without any anomaly examples in train.  This helper is explicitly a
feasibility diagnostic: it moves a stratified part of the original validation
windows into train and keeps the untouched test split for a first external
check.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label-key", default="y_response_state")
    parser.add_argument("--promote-val-fraction", type=float, default=0.70)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 0.0 < args.promote_val_fraction < 1.0:
        parser.error("promote-val-fraction must be between zero and one")
    return args


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def save_npz(path: Path, data: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **data)


def concatenate(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = set(parts[0])
    if any(set(part) != keys for part in parts):
        raise ValueError("Cannot concatenate tensor splits with different keys")
    return {key: np.concatenate([part[key] for part in parts], axis=0) for key in sorted(keys)}


def take(data: dict[str, np.ndarray], selected: np.ndarray) -> dict[str, np.ndarray]:
    return {key: value[selected] for key, value in data.items()}


def stratified_val_masks(data: dict[str, np.ndarray], label_key: str, train_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if label_key not in data:
        raise ValueError(f"Input validation split has no {label_key!r}")
    rng = np.random.default_rng(seed)
    labels = data[label_key].astype(np.int64)
    source_ids = data["source_id"].astype(np.int64)
    promote = np.zeros(len(labels), dtype=bool)
    holdout = np.zeros(len(labels), dtype=bool)
    for source_id in np.unique(source_ids):
        for label in np.unique(labels[source_ids == source_id]):
            group = np.flatnonzero((source_ids == source_id) & (labels == label))
            if len(group) == 0:
                continue
            shuffled = group.copy()
            rng.shuffle(shuffled)
            cut = int(round(len(shuffled) * train_fraction))
            if len(shuffled) > 1:
                cut = min(max(cut, 1), len(shuffled) - 1)
            promote[shuffled[:cut]] = True
            holdout[shuffled[cut:]] = True
    return promote, holdout


def counts(data: dict[str, np.ndarray], label_key: str) -> dict[str, int]:
    labels = data[label_key].astype(np.int64)
    return {str(label): int((labels == label).sum()) for label in sorted(np.unique(labels))}


def copy_sidecars(input_dir: Path, output_dir: Path) -> None:
    for name in ("feature_names.json", "scaler.json"):
        source = input_dir / name
        if source.exists():
            (output_dir / name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    train = load_npz(args.input_dir / "train.npz")
    original_val = load_npz(args.input_dir / "val.npz")
    test = load_npz(args.input_dir / "test.npz")
    promote, holdout = stratified_val_masks(original_val, args.label_key, args.promote_val_fraction, args.seed)
    diagnostic_train = concatenate([train, take(original_val, promote)])
    diagnostic_val = take(original_val, holdout)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_npz(args.output_dir / "train.npz", diagnostic_train)
    save_npz(args.output_dir / "val.npz", diagnostic_val)
    save_npz(args.output_dir / "test.npz", test)
    copy_sidecars(args.input_dir, args.output_dir)
    metadata: dict[str, Any] = json.loads((args.input_dir / "metadata.json").read_text(encoding="utf-8"))
    metadata["diagnostic_split"] = {
        "source": str(args.input_dir),
        "warning": "original val windows are stratified into train/val for response-state feasibility only",
        "label_key": args.label_key,
        "promote_val_fraction": args.promote_val_fraction,
        "seed": args.seed,
        "promoted_original_val_windows": int(promote.sum()),
        "heldout_original_val_windows": int(holdout.sum()),
    }
    metadata["splits"] = {
        "train": {"windows": int(len(diagnostic_train["x"])), "response_state_counts": counts(diagnostic_train, args.label_key)},
        "val": {"windows": int(len(diagnostic_val["x"])), "response_state_counts": counts(diagnostic_val, args.label_key)},
        "test": {"windows": int(len(test["x"])), "response_state_counts": counts(test, args.label_key)},
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metadata["splits"], indent=2))


if __name__ == "__main__":
    main()

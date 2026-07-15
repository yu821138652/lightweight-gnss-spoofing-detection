"""Verify binary GNSS tensors and their recording-level split manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


IGNORE_INDEX = -100


def section(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def check_tensor_distribution(npz_path: Path, split_name: str) -> bool:
    if not npz_path.exists():
        print(f"ERROR: missing {split_name} tensor: {npz_path}")
        return False

    with np.load(npz_path) as data:
        required = {'x', 'mask', 'y'}
        missing = required.difference(data.files)
        if missing:
            print(f"ERROR: {npz_path.name} missing keys: {sorted(missing)}")
            return False
        x = data['x']
        mask = data['mask'].astype(bool)
        y = data['y']
        is_dynamic = data['is_dynamic'] if 'is_dynamic' in data.files else None

    if x.ndim != 4 or mask.shape != y.shape or x.shape[:2] != y.shape:
        print(f"ERROR: incompatible tensor shapes in {npz_path.name}: x={x.shape}, mask={mask.shape}, y={y.shape}")
        return False

    labels = y[mask]
    invalid = labels[(labels != 0) & (labels != 1)]
    values, counts = np.unique(labels, return_counts=True)
    distribution = dict(zip(values.tolist(), counts.tolist()))
    active_ratio = float(mask.mean()) if mask.size else 0.0

    print(f"{split_name}: x={x.shape}, active_slots={active_ratio:.2%}, labels={distribution}")
    if x.shape[-1] != 7:
        print(f"WARNING: expected 7 core features, found {x.shape[-1]}")
    if invalid.size:
        print(f"ERROR: {split_name} contains invalid active labels: {np.unique(invalid).tolist()}")
        return False
    if distribution.get(0, 0) == 0 or distribution.get(1, 0) == 0:
        print(f"WARNING: {split_name} is missing one binary class.")
    if is_dynamic is not None:
        print(f"  dynamic_windows={float(np.mean(is_dynamic)):.2%}")
    return True


def check_recording_manifest(manifest_path: Path) -> bool:
    if not manifest_path.exists():
        print(f"ERROR: recording split manifest not found: {manifest_path}")
        return False

    manifest = pd.read_csv(manifest_path, encoding='utf-8-sig')
    required = {
        'recording_id', 'Environment', 'Scenario', 'Session', 'rows',
        'positive_rows', 'device_count', 'sequence_count', 'split',
    }
    missing = required.difference(manifest.columns)
    if missing:
        print(f"ERROR: manifest missing columns: {sorted(missing)}")
        return False

    duplicate_recordings = manifest.duplicated(['Environment', 'Scenario', 'Session'], keep=False)
    if duplicate_recordings.any():
        print("ERROR: a recording appears more than once in the split manifest.")
        return False

    print(f"recordings={len(manifest)}, device_logs={int(manifest['sequence_count'].sum())}")
    all_ok = True
    for split_name in ('train', 'val', 'test'):
        group = manifest[manifest['split'] == split_name]
        static_count = int(group['Scenario'].astype(str).str.startswith('st_').sum())
        dynamic_count = int(group['Scenario'].astype(str).str.startswith('dy_').sum())
        print(
            f"{split_name}: recordings={len(group)}, rows={int(group['rows'].sum())}, "
            f"positive_rows={int(group['positive_rows'].sum())}, static={static_count}, dynamic={dynamic_count}"
        )
        if group.empty:
            print(f"ERROR: {split_name} has no recording.")
            all_ok = False
        elif static_count == 0 or dynamic_count == 0:
            print(f"WARNING: {split_name} lacks static or dynamic coverage.")
    return all_ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--npz_dir', default='output/tensor_data', help='Directory containing train.npz/val.npz/test.npz.')
    parser.add_argument('--manifest', default=None, help='Recording-level split manifest. Defaults to <npz_dir>/recording_split_manifest.csv.')
    args = parser.parse_args()

    npz_dir = Path(args.npz_dir)
    manifest_path = Path(args.manifest) if args.manifest else npz_dir / 'recording_split_manifest.csv'

    section('Tensor Files')
    tensor_ok = all(
        check_tensor_distribution(npz_dir / f'{split_name}.npz', split_name)
        for split_name in ('train', 'val', 'test')
    )

    section('Recording Split Manifest')
    manifest_ok = check_recording_manifest(manifest_path)

    if tensor_ok and manifest_ok:
        print('\nPASS: tensor files and recording-level split manifest are consistent.')
    else:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

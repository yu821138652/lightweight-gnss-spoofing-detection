#!/usr/bin/env python3
"""Audit independent per-band direct/association labels before response training.

The legacy ``y_response_state`` is mutually exclusive.  This checker reports
the new independent labels emitted by 36_build_device_attack_event_tensors.py
by split, attack scenario, receiver and response band, so an experiment is not
started when an intended binary subtask has only one class.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


SPLITS = ("train", "val", "test")
BANDS = (
    ("L1", "has_l1", "baseline_has_l1", "y_target_l1", "y_direct_l1", "y_associated_anomaly_l1"),
    ("L5", "has_l5", "baseline_has_l5", "y_target_l5", "y_direct_l5", "y_associated_anomaly_l5"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def counts_for_group(
    data: dict[str, np.ndarray], mask: np.ndarray, has_key: str, baseline_key: str, target_key: str, direct_key: str, assoc_key: str,
) -> dict[str, int]:
    event = mask & (data["y_event"] == 1)
    observable = event & (data[has_key] == 1)
    baseline_capable = event & (data[baseline_key] == 1)
    target = observable & (data[target_key] == 1)
    non_target = baseline_capable & (data[target_key] == 0)
    associated = event & (data[assoc_key] == 1)
    return {
        "event_windows": int(event.sum()),
        "currently_observable_band_windows": int(observable.sum()),
        "baseline_capable_band_windows": int(baseline_capable.sum()),
        "target_band_windows": int(target.sum()),
        "direct_observed_windows": int((observable & (data[direct_key] == 1)).sum()),
        "non_target_windows": int(non_target.sum()),
        "associated_anomaly_positive": int((non_target & (data[assoc_key] == 1)).sum()),
        "associated_anomaly_negative": int((non_target & (data[assoc_key] == 0)).sum()),
        "association_positive_unobservable": int((associated & (data[baseline_key] == 1) & (data[has_key] == 0)).sum()),
        "association_positive_on_target_band": int((associated & (data[target_key] == 1)).sum()),
        "association_positive_outside_eligible_scope": int((associated & ~non_target).sum()),
    }


def main() -> None:
    args = parse_args()
    metadata_path = args.data_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    recordings = metadata.get("recordings", [])
    mapping = {int(value): str(name) for name, value in metadata.get("device_mapping", {}).items()}
    result: dict[str, Any] = {"data_dir": str(args.data_dir), "splits": {}}
    for split in SPLITS:
        path = args.data_dir / f"{split}.npz"
        with np.load(path, allow_pickle=False) as loaded:
            required = {"y_event", "recording_id", "device_id"}
            for _, has_key, baseline_key, target_key, direct_key, assoc_key in BANDS:
                required.update((has_key, baseline_key, target_key, direct_key, assoc_key))
            if missing := required.difference(loaded.files):
                raise ValueError(f"{path} is missing {sorted(missing)}")
            data = {key: loaded[key].copy() for key in required}
        split_result: dict[str, Any] = {"by_scenario": {}, "by_device": {}, "by_scenario_and_device": {}}
        scenario_names = {
            int(recording_id): str(row.get("Scenario", "unknown"))
            for recording_id, row in enumerate(recordings) if isinstance(row, dict)
        }
        for scenario in sorted(set(scenario_names.get(int(value), "unknown") for value in data["recording_id"])):
            scenario_ids = {recording_id for recording_id, name in scenario_names.items() if name == scenario}
            group = np.isin(data["recording_id"], list(scenario_ids))
            split_result["by_scenario"][scenario] = {
                band: counts_for_group(data, group, has, baseline, target, direct, assoc)
                for band, has, baseline, target, direct, assoc in BANDS
            }
            split_result["by_scenario_and_device"][scenario] = {}
            for device_id in sorted(set(int(value) for value in data["device_id"][group])):
                device_group = group & (data["device_id"] == device_id)
                split_result["by_scenario_and_device"][scenario][mapping.get(device_id, str(device_id))] = {
                    band: counts_for_group(data, device_group, has, baseline, target, direct, assoc)
                    for band, has, baseline, target, direct, assoc in BANDS
                }
        for device_id in sorted(set(int(value) for value in data["device_id"])):
            group = data["device_id"] == device_id
            split_result["by_device"][mapping.get(device_id, str(device_id))] = {
                band: counts_for_group(data, group, has, baseline, target, direct, assoc)
                for band, has, baseline, target, direct, assoc in BANDS
            }
        result["splits"][split] = split_result
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

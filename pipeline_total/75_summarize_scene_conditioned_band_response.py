#!/usr/bin/env python3
"""Summarize reviewed-scene-conditioned device-by-band response evidence.

This is an evidence report, not a response classifier.  It reads the repaired
per-band labels emitted by ``36_build_device_attack_event_tensors.py`` and
reports, for each outer-test recording/device/band:

* attack-preceding baseline, attack-period, and post-attack summary values;
* C/N0 and signal-count changes relative to the pre-attack baseline;
* frequency availability during the attack and after the attack;
* the independent target/direct and associated-anomaly label semantics.

The report deliberately conditions on *reviewed* scene labels contained in the
response tensors.  It must not be presented as end-to-end performance of the
scene classifier.  Its purpose is to audit and present the observed response
phenomena before an online scene-to-response interface is defined.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


BANDS = (
    ("L1", "has_l1", "baseline_has_l1", "y_target_l1", "y_direct_l1", "y_associated_anomaly_l1"),
    ("L5", "has_l5", "baseline_has_l5", "y_target_l5", "y_direct_l5", "y_associated_anomaly_l5"),
)
BASE_FEATURES = (
    "log_signal_count",
    "cn0_last_median",
    "cn0_last_q25",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path, default=Path("output/label_repair_v1"),
        help="directory containing fold_<id>/device_tensors_* directories",
    )
    parser.add_argument("--folds", type=int, nargs="+", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument(
        "--tensor-dir-name", default="device_tensors_scene_conditioned_cn0_extreme",
        help="child directory under each fold containing repaired device tensors",
    )
    parser.add_argument(
        "--association-labels", type=Path,
        default=Path("docs/device_band_association_intervals.csv"),
        help="manual associated-anomaly intervals used for input-coverage audit",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--overwrite", action="store_true",
        help="allow replacing existing CSV/JSON report files",
    )
    return parser.parse_args()


def finite_median(values: np.ndarray) -> float | None:
    values = values[np.isfinite(values)]
    return None if len(values) == 0 else float(np.median(values))


def finite_mean(values: np.ndarray) -> float | None:
    values = values[np.isfinite(values)]
    return None if len(values) == 0 else float(np.mean(values))


def as_number(value: float | None) -> float | str:
    return "" if value is None else value


def scalar_bool(values: np.ndarray) -> int:
    return int(bool(np.any(values == 1)))


def phase_masks(event: np.ndarray, tow: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    attack = event == 1
    if not np.any(attack):
        return np.zeros_like(attack), attack, np.zeros_like(attack)
    first_attack_tow = float(np.min(tow[attack]))
    last_attack_tow = float(np.max(tow[attack]))
    pre = (event == 0) & (tow < first_attack_tow)
    post = (event == 0) & (tow > last_attack_tow)
    return pre, attack, post


def feature_index(feature_names: list[str], band: str) -> dict[str, int]:
    prefix = band.lower() + "_"
    result: dict[str, int] = {}
    for name in BASE_FEATURES:
        feature_name = prefix + name
        delta_name = "initial_baseline_delta_" + feature_name
        if feature_name not in feature_names or delta_name not in feature_names:
            raise ValueError(f"Required feature missing: {feature_name} or {delta_name}")
        result[feature_name] = feature_names.index(feature_name)
        result[delta_name] = feature_names.index(delta_name)
    return result


def value_by_phase(x: np.ndarray, mask: np.ndarray, index: int) -> float | None:
    return finite_median(x[mask, index])


def response_scope(
    attack_windows: int,
    target: int,
    direct_windows: int,
    associated: int,
    baseline_capable: int,
) -> str:
    if attack_windows == 0:
        return "no_reviewed_attack"
    if target:
        return "target_band_direct_observed" if direct_windows else "target_band_unobserved"
    if not baseline_capable:
        return "non_target_not_applicable_no_baseline_capability"
    return "non_target_associated_anomaly" if associated else "non_target_no_associated_anomaly_label"


def read_fold(
    fold: int, tensor_dir: Path, split: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any], list[str]]:
    metadata_path = tensor_dir / "metadata.json"
    feature_path = tensor_dir / "feature_names.json"
    tensor_path = tensor_dir / f"{split}.npz"
    if not tensor_path.exists():
        raise FileNotFoundError(tensor_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    feature_names = json.loads(feature_path.read_text(encoding="utf-8"))
    required = {
        "x", "y_event", "recording_id", "device_id", "endpoint_tow",
        "baseline_has_l1", "baseline_has_l5",
    }
    for _, has, baseline, target, direct, associated in BANDS:
        required.update((has, baseline, target, direct, associated))
    with np.load(tensor_path, allow_pickle=False) as loaded:
        missing = required.difference(loaded.files)
        if missing:
            raise ValueError(f"{tensor_path} is missing {sorted(missing)}")
        data = {key: loaded[key].copy() for key in required}
    return data, metadata, feature_names


def read_association_labels(path: Path) -> list[dict[str, str | float]]:
    required = {
        "Environment", "Scenario", "Session", "DeviceName", "response_band", "start_tow", "end_tow",
    }
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return []
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"{path} is missing {sorted(missing)}")
        return [
            {
                "Environment": str(row["Environment"]),
                "Scenario": str(row["Scenario"]),
                "Session": str(row["Session"]),
                "DeviceName": str(row["DeviceName"]),
                "response_band": str(row["response_band"]).upper(),
                "start_tow": float(row["start_tow"]),
                "end_tow": float(row["end_tow"]),
            }
            for row in reader
        ]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "device_band_response_summary.csv"
    windows_path = args.output_dir / "device_band_response_windows.csv"
    coverage_path = args.output_dir / "manual_association_label_coverage.csv"
    report_path = args.output_dir / "response_evidence_report.json"
    outputs = (summary_path, windows_path, coverage_path, report_path)
    if not args.overwrite:
        existing = [str(path) for path in outputs if path.exists()]
        if existing:
            raise FileExistsError(f"Output already exists; use --overwrite: {existing}")

    summary_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    seen_recordings: set[tuple[str, str, str]] = set()
    observed_band_windows: dict[tuple[str, str, str, str, str], dict[str, np.ndarray]] = {}
    label_counts: Counter[str] = Counter()
    fold_sources: list[dict[str, Any]] = []

    for fold in args.folds:
        tensor_dir = args.data_root / f"fold_{fold}" / args.tensor_dir_name
        data, metadata, feature_names = read_fold(fold, tensor_dir, args.split)
        device_names = {int(value): str(name) for name, value in metadata["device_mapping"].items()}
        recordings = metadata.get("recordings", [])
        feature_indices = {band: feature_index(feature_names, band) for band, *_ in BANDS}
        fold_sources.append({"fold": fold, "tensor_dir": str(tensor_dir), "samples": int(len(data["y_event"]))})

        for recording_id in sorted(set(int(value) for value in data["recording_id"])):
            if recording_id >= len(recordings):
                raise ValueError(f"fold {fold}: recording_id {recording_id} is not in metadata")
            recording = recordings[recording_id]
            recording_key = (
                str(recording.get("Environment", "unknown")),
                str(recording.get("Scenario", "unknown")),
                str(recording.get("Session", recording_id)),
            )
            if recording_key in seen_recordings:
                raise ValueError(
                    "A recording appears in multiple requested fold/split inputs; "
                    f"deduplicate before reporting: {recording_key}"
                )
            seen_recordings.add(recording_key)
            recording_mask = data["recording_id"] == recording_id

            for device_id in sorted(set(int(value) for value in data["device_id"][recording_mask])):
                group = recording_mask & (data["device_id"] == device_id)
                order = np.argsort(data["endpoint_tow"][group])
                selected = {key: values[group][order] for key, values in data.items()}
                pre, attack, post = phase_masks(selected["y_event"], selected["endpoint_tow"])
                device_name = device_names.get(device_id, str(device_id))

                for band, has_key, baseline_key, target_key, direct_key, associated_key in BANDS:
                    indices = feature_indices[band]
                    has_values = selected[has_key].astype(float)
                    attack_windows = int(attack.sum())
                    direct_windows = int(np.sum(attack & (selected[direct_key] == 1)))
                    associated_windows = int(np.sum(attack & (selected[associated_key] == 1)))
                    baseline_capable = scalar_bool(selected[baseline_key])
                    target = scalar_bool(selected[target_key][attack]) if attack_windows else 0
                    associated = scalar_bool(selected[associated_key][attack]) if attack_windows else 0
                    attack_end_tow = float(np.max(selected["endpoint_tow"][attack])) if attack_windows else None
                    first_post_available_delay = None
                    if attack_end_tow is not None:
                        available_post = post & (selected[has_key] == 1)
                        if np.any(available_post):
                            first_post_available_delay = float(
                                np.min(selected["endpoint_tow"][available_post]) - attack_end_tow
                            )

                    row: dict[str, Any] = {
                        "fold": fold,
                        "Environment": recording_key[0],
                        "Scenario": recording_key[1],
                        "Session": recording_key[2],
                        "DeviceName": device_name,
                        "response_band": band,
                        "diagnosis_scope": response_scope(
                            attack_windows, target, direct_windows, associated, baseline_capable,
                        ),
                        "baseline_capable": baseline_capable,
                        "reviewed_target_band": target,
                        "reviewed_direct_observed": int(direct_windows > 0),
                        "reviewed_associated_anomaly": associated,
                        "pre_windows": int(pre.sum()),
                        "attack_windows": attack_windows,
                        "post_windows": int(post.sum()),
                        "attack_start_tow": as_number(float(np.min(selected["endpoint_tow"][attack])) if attack_windows else None),
                        "attack_end_tow": as_number(attack_end_tow),
                        "pre_present_rate": as_number(finite_mean(has_values[pre])),
                        "attack_present_rate": as_number(finite_mean(has_values[attack])),
                        "post_present_rate": as_number(finite_mean(has_values[post])),
                        "first_post_available_delay_s": as_number(first_post_available_delay),
                    }
                    for name in BASE_FEATURES:
                        base_name = f"{band.lower()}_{name}"
                        delta_name = "initial_baseline_delta_" + base_name
                        pre_value = value_by_phase(selected["x"], pre, indices[base_name])
                        attack_value = value_by_phase(selected["x"], attack, indices[base_name])
                        post_value = value_by_phase(selected["x"], post, indices[base_name])
                        attack_delta = None if pre_value is None or attack_value is None else attack_value - pre_value
                        post_delta = None if pre_value is None or post_value is None else post_value - pre_value
                        row[f"pre_{name}_median"] = as_number(pre_value)
                        row[f"attack_{name}_median"] = as_number(attack_value)
                        row[f"post_{name}_median"] = as_number(post_value)
                        row[f"attack_minus_pre_{name}"] = as_number(attack_delta)
                        row[f"post_minus_pre_{name}"] = as_number(post_delta)
                        row[f"attack_initial_baseline_delta_{name}_median"] = as_number(
                            value_by_phase(selected["x"], attack, indices[delta_name])
                        )
                    summary_rows.append(row)
                    label_counts[row["diagnosis_scope"]] += 1
                    observed_band_windows[
                        (recording_key[0], recording_key[1], recording_key[2], device_name, band)
                    ] = {
                        "tow": selected["endpoint_tow"].copy(),
                        "associated": selected[associated_key].copy(),
                    }

                    for index in range(len(selected["endpoint_tow"])):
                        window_row: dict[str, Any] = {
                            "fold": fold,
                            "Environment": recording_key[0],
                            "Scenario": recording_key[1],
                            "Session": recording_key[2],
                            "DeviceName": device_name,
                            "response_band": band,
                            "endpoint_tow": float(selected["endpoint_tow"][index]),
                            "phase": "attack" if attack[index] else ("pre_attack" if pre[index] else "post_attack"),
                            "event_active": int(selected["y_event"][index]),
                            "baseline_capable": int(selected[baseline_key][index]),
                            "band_present": int(selected[has_key][index]),
                            "reviewed_target_band": int(selected[target_key][index]),
                            "reviewed_direct_observed": int(selected[direct_key][index]),
                            "reviewed_associated_anomaly": int(selected[associated_key][index]),
                        }
                        for name in BASE_FEATURES:
                            base_name = f"{band.lower()}_{name}"
                            delta_name = "initial_baseline_delta_" + base_name
                            window_row[name] = float(selected["x"][index, indices[base_name]])
                            window_row[f"initial_baseline_delta_{name}"] = float(
                                selected["x"][index, indices[delta_name]]
                            )
                        window_rows.append(window_row)

    summary_fields = list(summary_rows[0]) if summary_rows else []
    window_fields = list(window_rows[0]) if window_rows else []
    for path, rows, fields in ((summary_path, summary_rows, summary_fields), (windows_path, window_rows, window_fields)):
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    coverage_rows: list[dict[str, Any]] = []
    for label in read_association_labels(args.association_labels):
        key = (
            str(label["Environment"]), str(label["Scenario"]), str(label["Session"]),
            str(label["DeviceName"]), str(label["response_band"]),
        )
        observed = observed_band_windows.get(key)
        if observed is None:
            coverage_rows.append({
                **label,
                "outer_test_stream_present": 0,
                "windows_in_manual_interval": 0,
                "positive_label_windows_in_interval": 0,
                "coverage_status": "stream_not_present_in_requested_outer_tests",
            })
            continue
        interval = (observed["tow"] >= float(label["start_tow"])) & (observed["tow"] <= float(label["end_tow"]))
        windows = int(interval.sum())
        positive = int(np.sum(interval & (observed["associated"] == 1)))
        coverage_rows.append({
            **label,
            "outer_test_stream_present": 1,
            "windows_in_manual_interval": windows,
            "positive_label_windows_in_interval": positive,
            "coverage_status": "covered" if windows and positive == windows else (
                "interval_has_no_tensor_window" if not windows else "partial_or_missing_positive_label"
            ),
        })
    coverage_fields = list(coverage_rows[0]) if coverage_rows else []
    with coverage_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=coverage_fields)
        writer.writeheader()
        writer.writerows(coverage_rows)

    report = {
        "task": "reviewed_scene_conditioned_device_band_response_evidence",
        "not_an_end_to_end_scene_response_evaluation": True,
        "scene_conditioning": (
            "reviewed target-band labels embedded in repaired response tensors; "
            "no scene-model prediction is consumed by this report"
        ),
        "split": args.split,
        "fold_sources": fold_sources,
        "recordings": len(seen_recordings),
        "device_band_summaries": len(summary_rows),
        "device_band_window_rows": len(window_rows),
        "diagnosis_scope_counts": dict(sorted(label_counts.items())),
        "manual_association_label_coverage": dict(sorted(Counter(
            str(row["coverage_status"]) for row in coverage_rows
        ).items())),
        "outputs": {
            "summary_csv": str(summary_path),
            "windows_csv": str(windows_path),
            "manual_association_label_coverage_csv": str(coverage_path),
        },
        "field_interpretation": {
            "attack_minus_pre_*": "attack-period median minus pre-attack median; descriptive effect size",
            "attack_initial_baseline_delta_*": "median of the fixed initial-normal-baseline delta feature",
            "attack_present_rate": "fraction of attack windows where this band remains observable",
            "first_post_available_delay_s": "delay from the final reviewed attack endpoint to first observable post-attack window",
        },
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

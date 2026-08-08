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
    parser.add_argument("--z-threshold", type=float, default=3.0)
    parser.add_argument("--min-modalities", type=int, default=2, choices=(1, 2, 3))
    parser.add_argument("--min-persistent-windows", type=int, default=3)
    parser.add_argument("--baseline-min-presence", type=float, default=0.90)
    parser.add_argument(
        "--cohort-min-devices", type=int, default=4,
        help="minimum baseline-capable peers before cohort-consensus evidence is allowed",
    )
    parser.add_argument(
        "--cohort-min-fraction", type=float, default=0.75,
        help="same-direction quality-support fraction required for cohort-consensus evidence",
    )
    parser.add_argument(
        "--cohort-min-persistent-windows", type=int, default=30,
        help="consecutive quality-deviation windows required for one device to support a cohort",
    )
    parser.add_argument(
        "--individual-max-attack-presence", type=float, default=0.50,
        help="maximum attack-period presence rate for severe individual availability-loss evidence",
    )
    parser.add_argument(
        "--recovery-presence-tolerance", type=float, default=0.10,
        help="permitted post-attack presence-rate deficit from the pre-attack baseline",
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


def finite_mad(values: np.ndarray) -> float | None:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return None
    median = np.median(values)
    return float(np.median(np.abs(values - median)))


def robust_z(value: float, median: float | None, mad: float | None, floor: float = 1e-3) -> float | None:
    if not np.isfinite(value) or median is None or mad is None:
        return None
    scale = max(1.4826 * mad, floor)
    return float((value - median) / scale)


def phase_value(values: np.ndarray, mask: np.ndarray) -> float | None:
    return finite_median(values[mask])


def phase_mad(values: np.ndarray, mask: np.ndarray) -> float | None:
    return finite_mad(values[mask])


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


def initial_window_diagnosis(event: bool, direct: bool, target: bool, baseline_capable: bool) -> str:
    if not event:
        return "outside_attack"
    if direct:
        return "direct"
    if target:
        return "target_band_unobserved"
    if not baseline_capable:
        return "not_applicable"
    return "no_associated_anomaly_evidence"


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
    rule_summary_lookup: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
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
                    quality_mask = pre & (has_values == 1)
                    count_values = selected["x"][:, indices[f"{band.lower()}_log_signal_count"]]
                    cn0_median_values = selected["x"][:, indices[f"{band.lower()}_cn0_last_median"]]
                    cn0_q25_values = selected["x"][:, indices[f"{band.lower()}_cn0_last_q25"]]
                    baseline_presence_rate = finite_mean(has_values[pre])
                    baseline_count_median = phase_value(count_values, pre)
                    baseline_count_mad = phase_mad(count_values, pre)
                    baseline_cn0_median = phase_value(cn0_median_values, quality_mask)
                    baseline_cn0_mad = phase_mad(cn0_median_values, quality_mask)
                    baseline_cn0_q25 = phase_value(cn0_q25_values, quality_mask)
                    baseline_cn0_q25_mad = phase_mad(cn0_q25_values, quality_mask)

                    quality_z = np.full(len(selected["endpoint_tow"]), np.nan, dtype=float)
                    count_z = np.full(len(selected["endpoint_tow"]), np.nan, dtype=float)
                    for index in np.flatnonzero(has_values == 1):
                        z_median = robust_z(
                            float(cn0_median_values[index]), baseline_cn0_median, baseline_cn0_mad, floor=0.1,
                        )
                        z_q25 = robust_z(
                            float(cn0_q25_values[index]), baseline_cn0_q25, baseline_cn0_q25_mad, floor=0.1,
                        )
                        finite_quality_z = [abs(value) for value in (z_median, z_q25) if value is not None]
                        if finite_quality_z:
                            quality_z[index] = max(finite_quality_z)
                    for index in range(len(selected["endpoint_tow"])):
                        z_count = robust_z(
                            float(count_values[index]), baseline_count_median, baseline_count_mad, floor=0.1,
                        )
                        if z_count is not None:
                            count_z[index] = abs(z_count)
                    quality_flag = np.isfinite(quality_z) & (quality_z >= args.z_threshold)
                    count_flag = np.isfinite(count_z) & (count_z >= args.z_threshold)
                    availability_flag = (
                        (baseline_presence_rate is not None)
                        & (baseline_presence_rate >= args.baseline_min_presence)
                        & (has_values == 0)
                    )
                    evidence_score = quality_flag.astype(int) + count_flag.astype(int) + availability_flag.astype(int)
                    evidence_candidate = attack & (selected[target_key] == 0) & (selected[baseline_key] == 1)
                    evidence_raw = evidence_candidate & (evidence_score >= args.min_modalities)
                    persistent_run = np.zeros(len(evidence_raw), dtype=int)
                    current_run = 0
                    for index, positive in enumerate(evidence_raw):
                        if positive:
                            current_run += 1
                        else:
                            current_run = 0
                        persistent_run[index] = current_run
                    raw_rule_evidence = evidence_candidate & (persistent_run >= args.min_persistent_windows)
                    quality_support_raw = evidence_candidate & quality_flag
                    quality_support_run = np.zeros(len(quality_support_raw), dtype=int)
                    current_quality_run = 0
                    for index, positive in enumerate(quality_support_raw):
                        if positive:
                            current_quality_run += 1
                        else:
                            current_quality_run = 0
                        quality_support_run[index] = current_quality_run
                    quality_support = bool(np.max(quality_support_run) >= args.cohort_min_persistent_windows)
                    attack_quality_median = finite_median(cn0_median_values[attack & (has_values == 1)])
                    quality_shift = (
                        None if attack_quality_median is None or baseline_cn0_median is None
                        else attack_quality_median - baseline_cn0_median
                    )
                    quality_direction = 0 if quality_shift is None or quality_shift == 0 else int(np.sign(quality_shift))
                    attack_end_tow = float(np.max(selected["endpoint_tow"][attack])) if attack_windows else None
                    first_post_available_delay = None
                    if attack_end_tow is not None:
                        available_post = post & (selected[has_key] == 1)
                        if np.any(available_post):
                            first_post_available_delay = float(
                                np.min(selected["endpoint_tow"][available_post]) - attack_end_tow
                            )
                    attack_presence_rate = finite_mean(has_values[attack])
                    post_presence_rate = finite_mean(has_values[post])
                    individual_availability_loss = bool(
                        baseline_capable
                        and baseline_presence_rate is not None
                        and attack_presence_rate is not None
                        and post_presence_rate is not None
                        and baseline_presence_rate >= args.baseline_min_presence
                        and attack_presence_rate <= args.individual_max_attack_presence
                        and post_presence_rate >= baseline_presence_rate - args.recovery_presence_tolerance
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
                        "attack_present_rate": as_number(attack_presence_rate),
                        "post_present_rate": as_number(post_presence_rate),
                        "first_post_available_delay_s": as_number(first_post_available_delay),
                        "rule_z_threshold": args.z_threshold,
                        "rule_min_modalities": args.min_modalities,
                        "rule_min_persistent_windows": args.min_persistent_windows,
                        "baseline_presence_rate": as_number(baseline_presence_rate),
                        "baseline_count_median": as_number(baseline_count_median),
                        "baseline_count_mad": as_number(baseline_count_mad),
                        "baseline_cn0_median": as_number(baseline_cn0_median),
                        "baseline_cn0_mad": as_number(baseline_cn0_mad),
                        "baseline_cn0_q25": as_number(baseline_cn0_q25),
                        "baseline_cn0_q25_mad": as_number(baseline_cn0_q25_mad),
                        "raw_rule_alert_windows": int(np.sum(raw_rule_evidence)),
                        "raw_rule_alert_rate": as_number(
                            float(np.mean(raw_rule_evidence[attack])) if attack_windows else None
                        ),
                        "raw_rule_max_persistent_run": int(np.max(persistent_run)) if len(persistent_run) else 0,
                        "cohort_quality_support": int(quality_support),
                        "cohort_quality_direction": quality_direction,
                        "cohort_quality_max_run": int(np.max(quality_support_run)) if len(quality_support_run) else 0,
                        "individual_availability_loss": int(individual_availability_loss),
                        "cohort_consensus_evidence": 0,
                        "rule_associated_evidence": 0,
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
                    rule_summary_lookup[
                        (recording_key[0], recording_key[1], recording_key[2], device_name, band)
                    ] = row
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
                            "quality_deviation_z": as_number(float(quality_z[index]) if np.isfinite(quality_z[index]) else None),
                            "signal_count_deviation_z": as_number(float(count_z[index]) if np.isfinite(count_z[index]) else None),
                            "availability_loss_flag": int(availability_flag[index]),
                            "evidence_modalities": int(evidence_score[index]),
                            "raw_evidence_candidate": int(evidence_raw[index]),
                            "persistent_evidence_run": int(persistent_run[index]),
                            "raw_rule_associated_anomaly_evidence": int(raw_rule_evidence[index]),
                            "cohort_quality_support": int(quality_support),
                            "cohort_quality_direction": quality_direction,
                            "individual_availability_loss": int(individual_availability_loss),
                            "rule_associated_anomaly_evidence": 0,
                            "rule_diagnosis": initial_window_diagnosis(
                                bool(attack[index]),
                                bool(selected[direct_key][index]),
                                bool(selected[target_key][index]),
                                bool(selected[baseline_key][index]),
                            ),
                        }
                        for name in BASE_FEATURES:
                            base_name = f"{band.lower()}_{name}"
                            delta_name = "initial_baseline_delta_" + base_name
                            window_row[name] = float(selected["x"][index, indices[base_name]])
                            window_row[f"initial_baseline_delta_{name}"] = float(
                                selected["x"][index, indices[delta_name]]
                            )
                        window_rows.append(window_row)

    cohort_groups: dict[tuple[int, str, str, str, str], list[dict[str, Any]]] = {}
    for row in summary_rows:
        eligible = (
            row["attack_windows"] > 0
            and row["reviewed_target_band"] == 0
            and row["baseline_capable"] == 1
        )
        if not eligible:
            continue
        key = (row["fold"], row["Environment"], row["Scenario"], row["Session"], row["response_band"])
        cohort_groups.setdefault(key, []).append(row)
    for rows in cohort_groups.values():
        if len(rows) < args.cohort_min_devices:
            continue
        direction_support = Counter(
            int(row["cohort_quality_direction"])
            for row in rows
            if row["cohort_quality_support"] and row["cohort_quality_direction"] != 0
        )
        if not direction_support:
            continue
        dominant_direction, support_count = direction_support.most_common(1)[0]
        required_support = int(np.ceil(args.cohort_min_fraction * len(rows)))
        if support_count < required_support:
            continue
        for row in rows:
            row["cohort_consensus_evidence"] = 1
            row["cohort_consensus_direction"] = dominant_direction
            row["cohort_peer_count"] = len(rows)
            row["cohort_support_count"] = support_count
    for row in summary_rows:
        row.setdefault("cohort_consensus_direction", 0)
        row.setdefault("cohort_peer_count", 0)
        row.setdefault("cohort_support_count", 0)
        eligible = row["reviewed_target_band"] == 0 and row["baseline_capable"] == 1
        row["rule_associated_evidence"] = int(
            eligible and (row["individual_availability_loss"] or row["cohort_consensus_evidence"])
        )

    final_rule_lookup = {
        (row["fold"], row["Environment"], row["Scenario"], row["Session"], row["DeviceName"], row["response_band"]): row
        for row in summary_rows
    }
    for row in window_rows:
        key = (row["fold"], row["Environment"], row["Scenario"], row["Session"], row["DeviceName"], row["response_band"])
        summary = final_rule_lookup[key]
        row["cohort_consensus_evidence"] = summary["cohort_consensus_evidence"]
        row["cohort_consensus_direction"] = summary["cohort_consensus_direction"]
        row["cohort_peer_count"] = summary["cohort_peer_count"]
        row["cohort_support_count"] = summary["cohort_support_count"]
        if row["event_active"] and not row["reviewed_target_band"] and row["baseline_capable"]:
            row["rule_associated_anomaly_evidence"] = summary["rule_associated_evidence"]
            row["rule_diagnosis"] = (
                "associated_anomaly_evidence" if summary["rule_associated_evidence"] else "no_associated_anomaly_evidence"
            )

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
                "rule_associated_evidence": 0,
                "coverage_status": "stream_not_present_in_requested_outer_tests",
            })
            continue
        interval = (observed["tow"] >= float(label["start_tow"])) & (observed["tow"] <= float(label["end_tow"]))
        windows = int(interval.sum())
        positive = int(np.sum(interval & (observed["associated"] == 1)))
        summary = rule_summary_lookup.get(key)
        coverage_rows.append({
            **label,
            "outer_test_stream_present": 1,
            "windows_in_manual_interval": windows,
            "positive_label_windows_in_interval": positive,
            "rule_associated_evidence": int(summary["rule_associated_evidence"]) if summary else 0,
            "coverage_status": "covered" if windows and positive == windows else (
                "interval_has_no_tensor_window" if not windows else "partial_or_missing_positive_label"
            ),
        })
    coverage_fields = list(coverage_rows[0]) if coverage_rows else []
    with coverage_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=coverage_fields)
        writer.writeheader()
        writer.writerows(coverage_rows)

    manual_normal_controls = [
        row for row in summary_rows
        if row["diagnosis_scope"] == "non_target_no_associated_anomaly_label"
    ]

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
        "manual_association_rule_detection": dict(sorted(Counter(
            "stream_missing" if not row["outer_test_stream_present"] else (
                "rule_positive" if row["rule_associated_evidence"] else "rule_negative"
            )
            for row in coverage_rows
        ).items())),
        "manual_normal_control_rule_detection": {
            "controls": len(manual_normal_controls),
            "rule_positive": int(sum(row["rule_associated_evidence"] for row in manual_normal_controls)),
            "rule_negative": int(sum(not row["rule_associated_evidence"] for row in manual_normal_controls)),
        },
        "rule_v2_configuration": {
            "cohort_min_devices": args.cohort_min_devices,
            "cohort_min_fraction": args.cohort_min_fraction,
            "cohort_min_persistent_windows": args.cohort_min_persistent_windows,
            "quality_z_threshold": args.z_threshold,
            "individual_max_attack_presence": args.individual_max_attack_presence,
            "baseline_min_presence": args.baseline_min_presence,
            "recovery_presence_tolerance": args.recovery_presence_tolerance,
        },
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
            "cohort_consensus_evidence": (
                "at least the configured fraction of baseline-capable peers has a persistent, "
                "same-direction C/N0 deviation; applied to every eligible peer"
            ),
            "individual_availability_loss": (
                "baseline availability is high, attack-period availability is severely reduced, "
                "and post-attack availability returns close to the baseline"
            ),
        },
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Measure event-level detection delay from response-state prediction CSVs.

The prediction CSVs are produced by 42_eval_response_state_direct_override.py.
This script is read-only with respect to model artifacts and reports event
detection rate plus Median/P90 time-to-detect (TTD) in TOW units.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


TARGETS = {
    "abnormal": lambda true_state: true_state > 0,
    "anomaly": lambda true_state: true_state == 1,
    "direct": lambda true_state: true_state == 2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-csv", type=Path, nargs="+", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--label-config", type=Path, default=Path("configs/preprocessing.yml"))
    parser.add_argument("--response-labels", type=Path, default=Path("docs/device_response_intervals.csv"))
    parser.add_argument("--hold-windows", type=int, default=1)
    parser.add_argument("--max-gap-tow", type=float, default=1.5)
    args = parser.parse_args()
    if args.hold_windows < 1:
        parser.error("--hold-windows must be positive")
    if args.max_gap_tow <= 0:
        parser.error("--max-gap-tow must be positive")
    return args


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"fold", "recording_id", "recording_name", "source_id", "device_name", "endpoint_tow", "true_state", "pred_state"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        rows = []
        for row in reader:
            rows.append(
                {
                    "fold": row["fold"],
                    "recording_id": row["recording_id"],
                    "recording_name": row["recording_name"],
                    "source_id": row["source_id"],
                    "device_name": row["device_name"],
                    "endpoint_tow": float(row["endpoint_tow"]),
                    "true_state": int(row["true_state"]),
                    "pred_state": int(row["pred_state"]),
                }
            )
    return rows


def grouped_rows(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["fold"], row["recording_id"], row["source_id"], row["device_name"])
        groups.setdefault(key, []).append(row)
    return [sorted(group, key=lambda item: item["endpoint_tow"]) for group in groups.values()]


def load_reviewed_intervals(path: Path) -> dict[tuple[str, str, str], list[tuple[float, float]]]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    entries = config.get("labeling", {}).get("session_spoofing_tow_intervals", {})
    result: dict[tuple[str, str, str], list[tuple[float, float]]] = {}
    for environment, scenarios in entries.items():
        for scenario, sessions in scenarios.items():
            for session, entry in sessions.items():
                if entry.get("status") != "reviewed":
                    continue
                result[(str(environment), str(scenario), str(session))] = [
                    (float(start), float(end)) for start, end in entry.get("intervals", [])
                ]
    return result


def load_response_intervals(path: Path) -> dict[tuple[str, str, str, str], list[tuple[float, float]]]:
    result: dict[tuple[str, str, str, str], list[tuple[float, float]]] = {}
    if not path.exists():
        return result
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("response_state") != "attack_associated_anomaly":
                continue
            key = (row["Environment"], row["Scenario"], row["Session"], row["DeviceName"])
            result.setdefault(key, []).append((float(row["start_tow"]), float(row["end_tow"])))
    return result


def recording_key(recording_name: str) -> tuple[str, str, str]:
    parts = recording_name.split("/", 2)
    if len(parts) != 3:
        raise ValueError(f"Cannot parse recording_name: {recording_name!r}")
    return parts[0], parts[1], parts[2]


def interval_events(
    group: list[dict[str, Any]],
    target: str,
    session_intervals: dict[tuple[str, str, str], list[tuple[float, float]]],
    response_intervals: dict[tuple[str, str, str, str], list[tuple[float, float]]],
) -> list[tuple[float, float, list[dict[str, Any]]]]:
    session_key = recording_key(group[0]["recording_name"])
    device_key = (*session_key, group[0]["device_name"])
    if target == "anomaly":
        intervals = response_intervals.get(device_key, [])
    else:
        intervals = session_intervals.get(session_key, [])
    result = []
    for start, end in intervals:
        overlap = [row for row in group if start <= row["endpoint_tow"] <= end]
        if not overlap:
            continue
        if target == "anomaly" and not any(row["true_state"] == 1 for row in overlap):
            continue
        if target == "direct" and not any(row["true_state"] == 2 for row in overlap):
            continue
        result.append((start, end, overlap))
    return result


def first_alarm_delay(
    event_start: float,
    event_end: float,
    rows: list[dict[str, Any]],
    target: str,
    hold_windows: int,
    max_gap_tow: float,
) -> float | None:
    def alarm(row: dict[str, Any]) -> bool:
        if target == "abnormal":
            return row["pred_state"] > 0
        if target == "anomaly":
            return row["pred_state"] == 1
        return row["pred_state"] == 2

    for start in range(0, len(rows) - hold_windows + 1):
        candidate = rows[start : start + hold_windows]
        if not all(alarm(row) for row in candidate):
            continue
        if any(
            candidate[index]["endpoint_tow"] - candidate[index - 1]["endpoint_tow"] > max_gap_tow
            for index in range(1, len(candidate))
        ):
            continue
        if candidate[0]["endpoint_tow"] < event_start or candidate[0]["endpoint_tow"] > event_end:
            continue
        return candidate[0]["endpoint_tow"] - event_start
    return None


def summarize(delays: list[float], events: int) -> dict[str, float | int | None]:
    detected = len(delays)
    values = np.asarray(delays, dtype=np.float64)
    return {
        "events": int(events),
        "detected_events": int(detected),
        "detection_rate": float(detected / events) if events else None,
        "missed_events": int(events - detected),
        "median_ttd_tow": float(np.median(values)) if detected else None,
        "p90_ttd_tow": float(np.percentile(values, 90)) if detected else None,
        "mean_ttd_tow": float(np.mean(values)) if detected else None,
        "max_ttd_tow": float(np.max(values)) if detected else None,
    }


def measure_file(
    path: Path,
    hold_windows: int,
    max_gap_tow: float,
    session_intervals: dict[tuple[str, str, str], list[tuple[float, float]]],
    response_intervals: dict[tuple[str, str, str, str], list[tuple[float, float]]],
) -> dict[str, Any]:
    rows = read_rows(path)
    groups = grouped_rows(rows)
    result: dict[str, Any] = {"predictions_csv": str(path), "groups": len(groups), "targets": {}}
    for target, predicate in TARGETS.items():
        delays: list[float] = []
        events = 0
        for group in groups:
            # Event starts come from reviewed intervals, not the first positive
            # row visible in the outer-test CSV. This avoids artificial TTD=0
            # when an outer test begins after an attack has already started.
            events_for_target = interval_events(group, target, session_intervals, response_intervals)
            events += len(events_for_target)
            for event_start, event_end, event_rows in events_for_target:
                delay = first_alarm_delay(
                    event_start, event_end, event_rows, target, hold_windows, max_gap_tow,
                )
                if delay is not None:
                    delays.append(delay)
        result["targets"][target] = summarize(delays, events)
    return result


def main() -> None:
    args = parse_args()
    files = [path.resolve() for path in args.predictions_csv]
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing prediction CSVs: {missing}")
    session_intervals = load_reviewed_intervals(args.label_config)
    response_intervals = load_response_intervals(args.response_labels)
    by_file = [
        measure_file(path, args.hold_windows, args.max_gap_tow, session_intervals, response_intervals)
        for path in files
    ]
    output = {
        "hold_windows": args.hold_windows,
        "max_gap_tow": args.max_gap_tow,
        "files": by_file,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

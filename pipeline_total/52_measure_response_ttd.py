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


TARGETS = {
    "abnormal": lambda true_state: true_state > 0,
    "anomaly": lambda true_state: true_state == 1,
    "direct": lambda true_state: true_state == 2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-csv", type=Path, nargs="+", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
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
        required = {"fold", "recording_id", "source_id", "device_name", "endpoint_tow", "true_state", "pred_state"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        rows = []
        for row in reader:
            rows.append(
                {
                    "fold": row["fold"],
                    "recording_id": row["recording_id"],
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


def event_segments(
    rows: list[dict[str, Any]], predicate: Any, max_gap_tow: float
) -> list[list[dict[str, Any]]]:
    segments: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for row in rows:
        if not predicate(row["true_state"]):
            if current:
                segments.append(current)
                current = []
            continue
        if current and row["endpoint_tow"] - current[-1]["endpoint_tow"] > max_gap_tow:
            segments.append(current)
            current = []
        current.append(row)
    if current:
        segments.append(current)
    return segments


def first_alarm_delay(
    segment: list[dict[str, Any]], target: str, hold_windows: int, max_gap_tow: float
) -> float | None:
    def alarm(row: dict[str, Any]) -> bool:
        if target == "abnormal":
            return row["pred_state"] > 0
        if target == "anomaly":
            return row["pred_state"] == 1
        return row["pred_state"] == 2

    for start in range(0, len(segment) - hold_windows + 1):
        candidate = segment[start : start + hold_windows]
        if not all(alarm(row) for row in candidate):
            continue
        if any(
            candidate[index]["endpoint_tow"] - candidate[index - 1]["endpoint_tow"] > max_gap_tow
            for index in range(1, len(candidate))
        ):
            continue
        return candidate[0]["endpoint_tow"] - segment[0]["endpoint_tow"]
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


def measure_file(path: Path, hold_windows: int, max_gap_tow: float) -> dict[str, Any]:
    rows = read_rows(path)
    groups = grouped_rows(rows)
    result: dict[str, Any] = {"predictions_csv": str(path), "groups": len(groups), "targets": {}}
    for target, predicate in TARGETS.items():
        delays: list[float] = []
        events = 0
        for group in groups:
            segments = event_segments(group, predicate, max_gap_tow)
            events += len(segments)
            for segment in segments:
                delay = first_alarm_delay(segment, target, hold_windows, max_gap_tow)
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
    by_file = [measure_file(path, args.hold_windows, args.max_gap_tow) for path in files]
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

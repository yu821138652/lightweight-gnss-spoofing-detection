#!/usr/bin/env python3
"""Audit extracted GNSS CSV files before windowing or model training."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


REQUIRED_COLUMNS = [
    "TimeNanos",
    "TOW",
    "utcTimeMillis",
    "Environment",
    "Scenario",
    "DeviceName",
    "sv_id",
    "FreqBand",
    "SpoofingType",
    "Label",
    "Cn0DbHz",
    "Cn0DbHz_dt",
    "Cn0DbHz_std",
    "AgcDb",
    "ReceivedSvTimeUncertaintyNanos",
    "PseudorangeRateUncertaintyMetersPerSecond",
    "AccumulatedDeltaRangeUncertaintyMeters",
]


def ratio(count: int, total: int) -> float:
    return round(count / total, 6) if total else 0.0


def audit(input_dir: Path) -> dict[str, Any]:
    files = sorted(input_dir.rglob("*.csv"))
    rows = 0
    label_counts: Counter[str] = Counter()
    environment_rows: Counter[str] = Counter()
    scenario_rows: Counter[str] = Counter()
    device_rows: Counter[str] = Counter()
    freq_band_rows: Counter[str] = Counter()
    environment_label_counts: Counter[str] = Counter()
    scenario_label_counts: Counter[str] = Counter()
    device_label_counts: Counter[str] = Counter()
    device_freq_band_counts: Counter[str] = Counter()
    missing_counts: Counter[str] = Counter()
    schema_errors: list[dict[str, Any]] = []
    files_by_environment: Counter[str] = Counter()

    for csv_path in files:
        relative_parts = csv_path.relative_to(input_dir).parts
        if relative_parts:
            files_by_environment[relative_parts[0]] += 1

        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            missing_columns = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
            if missing_columns:
                schema_errors.append(
                    {
                        "file": str(csv_path.relative_to(input_dir)),
                        "missing_columns": missing_columns,
                    }
                )
                continue

            for row in reader:
                rows += 1
                label_counts[row["Label"].strip()] += 1
                environment_rows[row["Environment"].strip()] += 1
                scenario_rows[row["Scenario"].strip()] += 1
                device_rows[row["DeviceName"].strip()] += 1
                freq_band_rows[row["FreqBand"].strip()] += 1
                environment_label_counts[
                    f"{row['Environment'].strip()}|{row['Label'].strip()}"
                ] += 1
                scenario_label_counts[
                    f"{row['Scenario'].strip()}|{row['Label'].strip()}"
                ] += 1
                device_label_counts[
                    f"{row['DeviceName'].strip()}|{row['Label'].strip()}"
                ] += 1
                device_freq_band_counts[
                    f"{row['DeviceName'].strip()}|{row['FreqBand'].strip()}"
                ] += 1
                for column in REQUIRED_COLUMNS:
                    if not row[column].strip():
                        missing_counts[column] += 1

    return {
        "input_dir": str(input_dir),
        "csv_files": len(files),
        "rows": rows,
        "files_by_environment": dict(sorted(files_by_environment.items())),
        "label_counts": dict(sorted(label_counts.items())),
        "environment_rows": dict(sorted(environment_rows.items())),
        "scenario_rows": dict(sorted(scenario_rows.items())),
        "device_rows": dict(sorted(device_rows.items())),
        "freq_band_rows": dict(sorted(freq_band_rows.items())),
        "environment_label_counts": dict(sorted(environment_label_counts.items())),
        "scenario_label_counts": dict(sorted(scenario_label_counts.items())),
        "device_label_counts": dict(sorted(device_label_counts.items())),
        "device_freq_band_counts": dict(sorted(device_freq_band_counts.items())),
        "missing_counts": dict(sorted(missing_counts.items())),
        "missing_rates": {
            column: ratio(missing_counts[column], rows)
            for column in REQUIRED_COLUMNS
            if missing_counts[column]
        },
        "schema_errors": schema_errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    report = audit(args.input_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()

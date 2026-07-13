#!/usr/bin/env python3
"""Build a session-level manifest from extracted GNSS CSV files.

The manifest preserves the original CSV as the split unit. This prevents
adjacent samples from one recording leaking across train, validation and test.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


REQUIRED_COLUMNS = {
    "Environment",
    "Scenario",
    "DeviceName",
    "Label",
    "AgcDb",
}


def summarize_csv(csv_path: Path, root: Path) -> dict[str, object]:
    row_count = 0
    positive_count = 0
    agc_missing_count = 0
    environment = scenario = device = ""

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - fields
        if missing:
            raise ValueError(f"{csv_path}: missing columns {sorted(missing)}")

        for row in reader:
            row_count += 1
            environment = environment or row["Environment"].strip()
            scenario = scenario or row["Scenario"].strip()
            device = device or row["DeviceName"].strip()
            positive_count += row["Label"].strip() == "1"
            agc_missing_count += not row["AgcDb"].strip()

    relative = csv_path.relative_to(root)
    session = relative.parts[2] if len(relative.parts) >= 4 else csv_path.stem
    return {
        "source_csv": relative.as_posix(),
        "session_id": relative.as_posix(),
        "environment": environment,
        "scenario": scenario,
        "session": session,
        "device": device,
        "row_count": row_count,
        "positive_count": positive_count,
        "positive_ratio": f"{positive_count / row_count:.6f}" if row_count else "0.000000",
        "agc_missing_count": agc_missing_count,
        "agc_missing_ratio": f"{agc_missing_count / row_count:.6f}" if row_count else "0.000000",
        "label_status": "needs_review" if row_count and positive_count == 0 else "has_positive_samples",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    files = sorted(args.input_dir.rglob("*.csv"))
    rows = [summarize_csv(path, args.input_dir) for path in files]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "source_csv",
        "session_id",
        "environment",
        "scenario",
        "session",
        "device",
        "row_count",
        "positive_count",
        "positive_ratio",
        "agc_missing_count",
        "agc_missing_ratio",
        "label_status",
    ]
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    reviewed = sum(row["label_status"] == "has_positive_samples" for row in rows)
    print(f"Wrote {len(rows)} sessions to {args.output_csv}")
    print(f"Sessions with positive samples: {reviewed}")
    print(f"Sessions needing label review: {len(rows) - reviewed}")


if __name__ == "__main__":
    main()

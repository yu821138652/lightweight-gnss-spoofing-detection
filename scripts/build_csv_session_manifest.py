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
    "Session",
    "DeviceName",
    "Label",
    "LabelStatus",
    "LabelSource",
    "AgcDb",
}


def summarize_csv(csv_path: Path, root: Path) -> dict[str, object]:
    row_count = 0
    positive_count = 0
    agc_missing_count = 0
    environment = scenario = session = device = ""
    label_status = label_source = ""

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
            session = session or row["Session"].strip()
            device = device or row["DeviceName"].strip()
            label_status = label_status or row["LabelStatus"].strip()
            label_source = label_source or row["LabelSource"].strip()
            positive_count += row["Label"].strip() == "1"
            agc_missing_count += not row["AgcDb"].strip()

    relative = csv_path.relative_to(root)
    return {
        "source_csv": relative.as_posix(),
        "session_id": "/".join((environment, scenario, session, device)),
        "environment": environment,
        "scenario": scenario,
        "session": session,
        "device": device,
        "row_count": row_count,
        "positive_count": positive_count,
        "positive_ratio": f"{positive_count / row_count:.6f}" if row_count else "0.000000",
        "agc_missing_count": agc_missing_count,
        "agc_missing_ratio": f"{agc_missing_count / row_count:.6f}" if row_count else "0.000000",
        "label_status": label_status or "unknown",
        "label_source": label_source or "unknown",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    files = sorted(
        path
        for path in args.input_dir.rglob("*.csv")
        if not any(part.endswith(("_by_signal_id", "_by_sv_id")) for part in path.parts)
    )
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
        "label_source",
    ]
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    reviewed = sum(row["label_status"] == "reviewed" for row in rows)
    print(f"Wrote {len(rows)} sessions to {args.output_csv}")
    print(f"Reviewed source logs: {reviewed}")
    print(f"Source logs needing label review: {len(rows) - reviewed}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build a manifest for real-world GNSS spoofing raw logs.

The manifest is a project management table, not a model input. It records
which raw logs exist, where they are, and whether their current project CSV
counterparts have been generated.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import yaml

RAW_PATTERNS = ("gnss_log_*.txt", "log_mimir_*.txt")
ENVIRONMENTS = ("playground", "new_building")
SCENARIOS = ("st_L1", "st_L5", "st_L_15", "dy_L1", "dy_L5", "dy_L_15")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data_raw"
DEFAULT_DATA_CSV_ROOT = PROJECT_ROOT / "data_csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "docs" / "data_manifest.csv"


@dataclass
class ManifestRow:
    environment: str
    scenario: str
    session: str
    device: str
    sub_path: str
    raw_file: str
    raw_relative_path: str
    extracted_csv_relative_path: str
    has_extracted_csv: str
    label_status: str
    label_source: str
    notes: str


def yes_no(value: bool) -> str:
    return "yes" if value else "no"


def find_raw_logs(root: Path) -> Iterable[Path]:
    seen: set[Path] = set()
    for pattern in RAW_PATTERNS:
        for path in root.rglob(pattern):
            if path.is_file() and path not in seen:
                seen.add(path)
                yield path


def parse_log_path(data_root: Path, raw_path: Path) -> tuple[str, str, str, str, str]:
    rel = raw_path.relative_to(data_root)
    parts = rel.parts

    if len(parts) > 1 and parts[0] in ENVIRONMENTS:
        environment = parts[0]
        scenario = parts[1]
        offset = 2
    elif len(parts) > 0 and parts[0] in SCENARIOS:
        environment = "playground"
        scenario = parts[0]
        offset = 1
    else:
        environment = parts[0] if len(parts) > 0 else "unknown"
        scenario = parts[1] if len(parts) > 1 else "unknown"
        offset = 2

    session = parts[offset] if len(parts) > offset else "unknown"
    device = parts[offset + 1] if len(parts) > offset + 1 else "unknown"

    if len(parts) > offset + 3:
        sub_path = str(Path(*parts[offset + 2:-1]))
    else:
        sub_path = ""

    return environment, scenario, session, device, sub_path


def resolve_label_metadata(
    environment: str, scenario: str, session: str, config: dict
) -> tuple[str, str]:
    """Mirror preprocessing label provenance without parsing raw samples."""
    labeling = config.get("labeling", {})
    session_entry = (
        labeling.get("session_spoofing_tow_intervals", {})
        .get(environment, {})
        .get(scenario, {})
        .get(session)
    )
    if session_entry is None:
        return "needs_review", "missing_session_config"
    if not isinstance(session_entry, dict):
        raise ValueError(
            "Session label entries must be mappings with status and intervals: "
            f"{environment}/{scenario}/{session}"
        )
    return str(session_entry.get("status", "needs_review")), "session_config"


def build_rows(data_root: Path, data_csv_root: Path, config: dict) -> list[ManifestRow]:
    rows: list[ManifestRow] = []

    for raw_path in sorted(find_raw_logs(data_root), key=lambda p: str(p).lower()):
        environment, scenario, session, device, sub_path = parse_log_path(data_root, raw_path)
        raw_relative_path = raw_path.relative_to(data_root)
        extracted_csv_path = (data_csv_root / raw_relative_path).with_suffix(".csv")
        label_status, label_source = resolve_label_metadata(
            environment, scenario, session, config
        )
        rows.append(
            ManifestRow(
                environment=environment,
                scenario=scenario,
                session=session,
                device=device,
                sub_path=sub_path,
                raw_file=raw_path.name,
                raw_relative_path=raw_relative_path.as_posix(),
                extracted_csv_relative_path=(
                    raw_relative_path.with_suffix(".csv").as_posix()
                    if extracted_csv_path.is_file()
                    else ""
                ),
                has_extracted_csv=yes_no(extracted_csv_path.is_file()),
                label_status=label_status,
                label_source=label_source,
                notes="",
            )
        )

    return rows


def write_manifest(rows: list[ManifestRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(rows[0]).keys()) if rows else list(ManifestRow.__dataclass_fields__.keys())
    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def print_summary(rows: list[ManifestRow]) -> None:
    print(f"Total raw logs: {len(rows)}")

    by_env: dict[str, int] = {}
    by_scenario: dict[tuple[str, str], int] = {}
    missing_extracted_csv = 0
    for row in rows:
        by_env[row.environment] = by_env.get(row.environment, 0) + 1
        key = (row.environment, row.scenario)
        by_scenario[key] = by_scenario.get(key, 0) + 1
        if row.has_extracted_csv == "no":
            missing_extracted_csv += 1

    print("By environment:")
    for env, count in sorted(by_env.items()):
        print(f"  {env}: {count}")

    print("By environment/scenario:")
    for (env, scenario), count in sorted(by_scenario.items()):
        print(f"  {env}/{scenario}: {count}")

    print(f"Missing extracted CSV: {missing_extracted_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build data_manifest.csv for GNSS spoofing raw logs.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Path to data_raw containing playground/new_building.")
    parser.add_argument(
        "--data-csv-root",
        type=Path,
        default=DEFAULT_DATA_CSV_ROOT,
        help="Path to current extracted CSV files mirroring data_raw.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output manifest CSV path.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "preprocessing.yml",
        help="Preprocessing YAML used to resolve label review status.",
    )
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"data root not found: {data_root}")
    data_csv_root = args.data_csv_root.resolve()

    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    rows = build_rows(data_root, data_csv_root, config)
    write_manifest(rows, args.output.resolve())
    print(f"Wrote manifest: {args.output.resolve()}")
    print_summary(rows)


if __name__ == "__main__":
    main()

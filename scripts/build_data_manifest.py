#!/usr/bin/env python3
"""Build a manifest for real-world GNSS spoofing raw logs.

The manifest is a project management table, not a model input. It records
which raw logs exist, where they are, and whether common intermediate files
have already been generated.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

RAW_PATTERNS = ("gnss_log_*.txt", "log_mimir_*.txt")
ENVIRONMENTS = ("playground", "new_building")
DEFAULT_DATA_ROOT = Path(r"H:\GNSS\real_world_spoofing_dataset_pipeline\data_raw")
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "docs" / "data_manifest.csv"


@dataclass
class ManifestRow:
    environment: str
    scenario: str
    session: str
    device: str
    sub_path: str
    raw_file: str
    raw_file_path: str
    has_raw_csv: str
    has_raw_sort_csv: str
    has_plot_features_csv: str
    has_features_enhanced_csv: str
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


def related_exists(raw_path: Path, suffixes: tuple[str, ...]) -> bool:
    stem = raw_path.with_suffix("")
    return any(stem.with_name(stem.name + suffix).exists() for suffix in suffixes)


def parse_log_path(data_root: Path, raw_path: Path) -> tuple[str, str, str, str, str]:
    rel = raw_path.relative_to(data_root)
    parts = rel.parts

    environment = parts[0] if len(parts) > 0 else "unknown"
    scenario = parts[1] if len(parts) > 1 else "unknown"
    session = parts[2] if len(parts) > 2 else "unknown"
    device = parts[3] if len(parts) > 3 else "unknown"

    if len(parts) > 5:
        sub_path = str(Path(*parts[4:-1]))
    else:
        sub_path = ""

    return environment, scenario, session, device, sub_path


def build_rows(data_root: Path) -> list[ManifestRow]:
    rows: list[ManifestRow] = []

    for raw_path in sorted(find_raw_logs(data_root), key=lambda p: str(p).lower()):
        environment, scenario, session, device, sub_path = parse_log_path(data_root, raw_path)
        rows.append(
            ManifestRow(
                environment=environment,
                scenario=scenario,
                session=session,
                device=device,
                sub_path=sub_path,
                raw_file=raw_path.name,
                raw_file_path=str(raw_path),
                has_raw_csv=yes_no(related_exists(raw_path, ("-raw.csv",))),
                has_raw_sort_csv=yes_no(related_exists(raw_path, ("-raw_sort3.csv", "-raw_sort.csv"))),
                has_plot_features_csv=yes_no(related_exists(raw_path, ("-plot_features.csv",))),
                has_features_enhanced_csv=yes_no(related_exists(raw_path, ("-features_enhanced.csv",))),
                label_status="unreviewed",
                label_source="",
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
    missing_plot = 0
    for row in rows:
        by_env[row.environment] = by_env.get(row.environment, 0) + 1
        key = (row.environment, row.scenario)
        by_scenario[key] = by_scenario.get(key, 0) + 1
        if row.has_plot_features_csv == "no":
            missing_plot += 1

    print("By environment:")
    for env, count in sorted(by_env.items()):
        print(f"  {env}: {count}")

    print("By environment/scenario:")
    for (env, scenario), count in sorted(by_scenario.items()):
        print(f"  {env}/{scenario}: {count}")

    print(f"Missing plot feature CSV: {missing_plot}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build data_manifest.csv for GNSS spoofing raw logs.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Path to data_raw containing playground/new_building.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output manifest CSV path.")
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"data root not found: {data_root}")

    rows = build_rows(data_root)
    write_manifest(rows, args.output.resolve())
    print(f"Wrote manifest: {args.output.resolve()}")
    print_summary(rows)


if __name__ == "__main__":
    main()

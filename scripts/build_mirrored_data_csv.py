#!/usr/bin/env python3
"""Build per-log processed CSV files under a mirrored data_csv tree.

The script scans raw GNSS txt logs under data_raw and writes one processed CSV
for each txt file under data_csv, preserving the directory structure.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
from pathlib import Path
from typing import Iterable

import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "preprocessing.yml"
PREPROCESS_PATH = PROJECT_ROOT / "pipeline_total" / "04_build_labeled_processed_csv.py"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")


def load_preprocess_module():
    spec = importlib.util.spec_from_file_location("preprocess", PREPROCESS_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load preprocessing module: {PREPROCESS_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


preprocess = load_preprocess_module()
process_single_file = preprocess.process_single_file
get_spoofing_type_from_path = preprocess.get_spoofing_type_from_path


def resolve_path(path_value: str | Path, base: Path = PROJECT_ROOT) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def find_dataset_root(scan_root: Path) -> Path:
    """Return the root used for mirror-relative paths.

    If scan_root points at data_raw/new_building or data_raw/playground, mirror
    relative to data_raw so output stays data_csv/new_building/... .
    """
    if scan_root.name in preprocess.ENVIRONMENTS and scan_root.parent.name == "data_raw":
        return scan_root.parent
    return scan_root


def find_raw_logs(scan_root: Path, patterns: Iterable[str]) -> list[Path]:
    seen: set[Path] = set()
    files: list[Path] = []
    for pattern in patterns:
        for path in scan_root.rglob(pattern):
            if path.is_file() and path not in seen:
                seen.add(path)
                files.append(path)
    return sorted(files, key=lambda p: str(p).lower())


def output_path_for(txt_path: Path, mirror_root: Path, output_root: Path, suffix: str) -> Path:
    rel_path = txt_path.relative_to(mirror_root)
    return (output_root / rel_path).with_suffix(suffix)


def build_one_csv(txt_path: Path, output_path: Path, mirror_root: Path, config: dict) -> tuple[bool, int]:
    known_types = list(config.get("labeling", {}).get("spoofing_type_to_label", {}).keys())
    spoofing_type = get_spoofing_type_from_path(txt_path, known_types)
    df, _ = process_single_file(txt_path, spoofing_type, config, data_root=mirror_root)

    if df.empty:
        logging.warning("No valid data after parsing/filtering: %s", txt_path)
        return False, 0

    df['SourceFile'] = txt_path.name
    df['SourceRelativePath'] = txt_path.relative_to(mirror_root).as_posix()

    final_columns = config.get(
        "final_columns",
        [
            "TimeNanos",
            "TOW",
            "utcTimeMillis",
            "Environment",
            "Scenario",
            "Session",
            "DeviceName",
            "ConstellationType",
            "Svid",
            "sv_id",
            "FreqBand",
            "CarrierFrequencyHz",
            "CarrierFrequencyHzRounded",
            "CodeType",
            "SignalBand",
            "signal_id",
            "SignalEpochCount",
            "SpoofingType",
            "Label",
            "LabelStatus",
            "LabelSource",
            "AgcDbMissing",
            "SourceFile",
            "SourceRelativePath",
        ]
        + preprocess.FEATURE_COLS,
    )
    available_cols = [col for col in final_columns if col in df.columns]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df[available_cols].to_csv(output_path, index=False)
    return True, len(df)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mirror data_raw txt logs into processed per-log CSV files.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Preprocessing YAML config.")
    parser.add_argument("--data-root", type=Path, default=None, help="Root to scan. Defaults to paths.input_dir in config.")
    parser.add_argument("--output-root", type=Path, default=None, help="Output root. Defaults to sibling data_csv.")
    parser.add_argument("--environment", choices=sorted(preprocess.ENVIRONMENTS), default=None, help="Only process one environment.")
    parser.add_argument("--scenario", choices=sorted(preprocess.SCENARIOS), default=None, help="Only process one scenario.")
    parser.add_argument(
        "--session",
        default=None,
        help="Only process one session directory, for example 2025.07.29.19.22_新主楼.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing CSV files.")
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip the first N sorted raw logs before applying --limit.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N matched files.")
    parser.add_argument("--suffix", default=".csv", help="Output file suffix. Default: .csv")
    args = parser.parse_args()

    config_path = resolve_path(args.config)
    config = load_config(config_path)

    configured_root = config.get("paths", {}).get("input_dir", "data_raw")
    scan_root = resolve_path(args.data_root or configured_root)
    mirror_root = find_dataset_root(scan_root)
    output_root = resolve_path(args.output_root) if args.output_root else mirror_root.parent / "data_csv"

    patterns = config.get("file_patterns", ["gnss_log_*.txt", "log_mimir_*.txt"])
    raw_files = find_raw_logs(scan_root, patterns)

    if args.environment:
        raw_files = [p for p in raw_files if args.environment in p.relative_to(mirror_root).parts]
    if args.scenario:
        raw_files = [p for p in raw_files if args.scenario in p.relative_to(mirror_root).parts]
    if args.session:
        raw_files = [p for p in raw_files if args.session in p.relative_to(mirror_root).parts]
    if args.offset < 0:
        parser.error("--offset must be non-negative")
    if args.offset:
        raw_files = raw_files[args.offset:]
    if args.limit is not None:
        raw_files = raw_files[: args.limit]

    logging.info("Config: %s", config_path)
    logging.info("Scan root: %s", scan_root)
    logging.info("Mirror root: %s", mirror_root)
    logging.info("Output root: %s", output_root)
    logging.info("Matched raw txt logs: %d", len(raw_files))

    if not raw_files:
        return

    generated = 0
    skipped = 0
    failed = 0
    total_rows = 0

    for txt_path in tqdm(raw_files, desc="Building mirrored CSV"):
        out_path = output_path_for(txt_path, mirror_root, output_root, args.suffix)
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        ok, rows = build_one_csv(txt_path, out_path, mirror_root, config)
        if ok:
            generated += 1
            total_rows += rows
        else:
            failed += 1

    logging.info("Generated: %d", generated)
    logging.info("Skipped: %d", skipped)
    logging.info("Failed: %d", failed)
    logging.info("Total rows written: %s", f"{total_rows:,}")


if __name__ == "__main__":
    main()

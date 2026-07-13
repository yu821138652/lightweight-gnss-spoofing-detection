#!/usr/bin/env python3
"""Split processed GNSS CSV files by a signal-aware identity column."""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data_csv"
OUTPUT_DIR_PREFIX = "_by_"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def safe_filename(value: object) -> str:
    """Convert a group identifier to a filesystem-safe CSV filename stem."""
    text = str(value).strip()
    text = re.sub(r'[<>:"/\\|?*]+', "_", text)
    return text or "unknown_sv"


def split_one_csv(
    csv_path: Path,
    group_column: str,
    sort_columns: list[str],
    overwrite: bool = False,
) -> tuple[int, int, Path]:
    """Split one CSV by an identity column with deterministic temporal ordering."""
    df = pd.read_csv(csv_path)
    required_columns = {group_column, *sort_columns}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"missing required columns {sorted(missing_columns)}: {csv_path}")

    output_dir = csv_path.with_name(csv_path.stem + OUTPUT_DIR_PREFIX + group_column)
    output_dir.mkdir(parents=True, exist_ok=True)

    valid_df = df[df[group_column].notna()].copy()
    valid_df[group_column] = valid_df[group_column].astype(str).str.strip()
    valid_df = valid_df[valid_df[group_column] != ""]
    helper_columns = []
    for column in sort_columns:
        helper = f"__sort_{column}"
        valid_df[helper] = pd.to_numeric(valid_df[column], errors="coerce")
        helper_columns.append(helper)

    generated = 0
    skipped = 0
    for group_value, group in valid_df.groupby(group_column, sort=True):
        output_path = output_dir / f"{safe_filename(group_value)}.csv"
        if output_path.exists() and not overwrite:
            skipped += 1
            continue

        sorted_group = group.sort_values(
            by=helper_columns,
            ascending=[True] * len(helper_columns),
            kind="mergesort",
            na_position="last",
        ).drop(columns=helper_columns)
        sorted_group.to_csv(output_path, index=False)
        generated += 1

    return generated, skipped, output_dir


def find_input_csvs(data_root: Path) -> list[Path]:
    """Find source CSV files while excluding previously generated split folders."""
    return sorted(
        (
            path
            for path in data_root.rglob("*.csv")
            if path.is_file()
            and not any(OUTPUT_DIR_PREFIX in part for part in path.parts)
        ),
        key=lambda path: str(path).lower(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split processed GNSS CSV files by a signal-aware identity column."
    )
    parser.add_argument("--input-csv", type=Path, default=None, help="Process only one CSV file.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Root for batch processing.")
    parser.add_argument(
        "--group-column",
        default="signal_id",
        help="Identity column used to split rows. Use sv_id for legacy satellite splits.",
    )
    parser.add_argument(
        "--sort-columns",
        nargs="+",
        default=["TOW", "TimeNanos"],
        help="Numeric ordering columns applied within each split file.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing split CSV files.")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N source CSV files.")
    args = parser.parse_args()

    if args.input_csv:
        input_files = [resolve_path(args.input_csv)]
    else:
        data_root = resolve_path(args.data_root)
        input_files = find_input_csvs(data_root)

    if args.limit is not None:
        input_files = input_files[: args.limit]

    logging.info("Matched source CSV files: %d", len(input_files))

    processed = 0
    failed = 0
    generated_parts = 0
    skipped_parts = 0

    for csv_path in tqdm(input_files, desc=f"Splitting CSV by {args.group_column}"):
        try:
            generated, skipped, output_dir = split_one_csv(
                csv_path,
                group_column=args.group_column,
                sort_columns=args.sort_columns,
                overwrite=args.overwrite,
            )
            processed += 1
            generated_parts += generated
            skipped_parts += skipped
            logging.info(
                "Processed %s -> %s (generated=%d, skipped=%d)",
                csv_path,
                output_dir,
                generated,
                skipped,
            )
        except Exception as exc:
            failed += 1
            logging.error("Failed %s: %s", csv_path, exc)

    logging.info("Processed source CSV files: %d", processed)
    logging.info("Failed source CSV files: %d", failed)
    logging.info("Generated satellite CSV files: %d", generated_parts)
    logging.info("Skipped satellite CSV files: %d", skipped_parts)


if __name__ == "__main__":
    main()

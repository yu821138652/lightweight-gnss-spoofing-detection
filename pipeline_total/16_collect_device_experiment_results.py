"""Collect device-level validation and test metrics from training directories."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def metric_identity(path: Path) -> tuple[str, str]:
    match = re.match(r"(val|test)_metrics_(.+)\.json$", path.name)
    if not match:
        raise ValueError(f"Unsupported metric filename: {path.name}")
    return match.group(1), match.group(2)


def infer_window(run_dir: Path, default_window: int) -> int:
    for candidate in (run_dir, *run_dir.parents):
        match = re.search(r"(?:^|_)l(\d+)(?:_|$)", candidate.name.lower())
        if match:
            return int(match.group(1))
    return default_window


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--default-window", type=int, default=5)
    parser.add_argument("--name-contains", default="device", help="Only include runs whose directory or metric name contains this text.")
    args = parser.parse_args()
    if not args.training_root.exists():
        raise FileNotFoundError(args.training_root)

    records: dict[tuple[str, str], dict] = {}
    for metric_path in args.training_root.rglob("*_metrics_*.json"):
        if args.name_contains and args.name_contains.lower() not in str(metric_path).lower():
            continue
        try:
            split, model = metric_identity(metric_path)
            metrics = json.loads(metric_path.read_text(encoding="utf-8"))
        except (ValueError, json.JSONDecodeError):
            continue
        if "macro_f1" not in metrics:
            continue
        run_dir = metric_path.parent
        key = (str(run_dir.resolve()), model)
        record = records.setdefault(
            key,
            {
                "run_dir": str(run_dir),
                "model": model,
                "time_steps": infer_window(run_dir, args.default_window),
            },
        )
        for name in ["macro_f1", "precision", "recall", "far", "samples", "model_size_bytes", "best_iteration"]:
            if name in metrics:
                record[f"{split}_{name}"] = metrics[name]

    rows = list(records.values())
    if not rows:
        raise ValueError("No matching device-level metric JSON files were found.")
    result = pd.DataFrame(rows)
    result = result.sort_values(["val_recall", "val_macro_f1"], ascending=False, na_position="last")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    print(result.to_string(index=False))
    print(f"Saved: {args.output_csv}")


if __name__ == "__main__":
    main()

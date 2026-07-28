"""Summarize response-state test metrics across folds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, nargs="+", required=True)
    parser.add_argument("--output-csv", type=Path)
    return parser.parse_args()


def short_metrics(row: dict[str, Any]) -> dict[str, float]:
    return {
        "macro_f1": float(row["macro_f1"]),
        "far": float(row["far"]),
        "abnormal_recall": float(row["abnormal_recall"]),
        "anomaly_recall": float(row["per_class"]["1"]["recall"]),
        "anomaly_support": float(row["per_class"]["1"]["support"]),
        "direct_recall": float(row["per_class"]["2"]["recall"]),
        "direct_support": float(row["per_class"]["2"]["support"]),
    }


def fold_name(path: Path) -> str:
    parts = list(path.parts)
    for part in reversed(parts):
        if part.startswith("fold_"):
            return part
    return path.parent.name


def load_rows(paths: list[Path]) -> list[dict[str, str | float]]:
    rows: list[dict[str, str | float]] = []
    for path in paths:
        metrics = json.loads(path.read_text(encoding="utf-8"))
        fold = fold_name(path)
        scenario = ",".join(sorted(metrics.get("by_scenario", {}).keys()))
        overall = short_metrics(metrics["overall"])
        rows.append({"fold": fold, "group_type": "overall", "group": scenario or "overall", **overall})
        for device, values in sorted(metrics.get("by_device", {}).items()):
            rows.append({"fold": fold, "group_type": "device", "group": device, **short_metrics(values)})
        for scenario_name, values in sorted(metrics.get("by_scenario", {}).items()):
            rows.append({"fold": fold, "group_type": "scenario", "group": scenario_name, **short_metrics(values)})
    return rows


def aggregate_overall(rows: list[dict[str, str | float]]) -> dict[str, float]:
    selected = [row for row in rows if row["group_type"] == "overall"]
    result = {
        key: mean(float(row[key]) for row in selected)
        for key in ("macro_f1", "far", "abnormal_recall", "direct_recall")
    }
    anomaly_rows = [row for row in selected if float(row["anomaly_support"]) > 0]
    result["anomaly_recall"] = mean(float(row["anomaly_recall"]) for row in anomaly_rows) if anomaly_rows else 0.0
    return result


def main() -> None:
    args = parse_args()
    rows = load_rows(args.metrics)
    summary = {
        "folds": sorted({str(row["fold"]) for row in rows if row["group_type"] == "overall"}),
        "overall_mean": aggregate_overall(rows),
    }
    print(json.dumps(summary, indent=2))
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "fold", "group_type", "group", "macro_f1", "far", "abnormal_recall",
            "anomaly_recall", "anomaly_support", "direct_recall", "direct_support",
        ]
        with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()

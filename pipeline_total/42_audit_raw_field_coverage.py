#!/usr/bin/env python3
"""Audit per-device coverage of raw Android GNSS-log fields that the current
processed CSV discards.

Motivation
----------
`04_build_labeled_processed_csv.py` parses all 38 raw columns but keeps only a
small subset (Cn0DbHz, AgcDb, ReceivedSvTimeUncertaintyNanos,
PseudorangeRateUncertaintyMetersPerSecond, plus derived Cn0DbHz_dt/_std).
The strategic question is whether any discarded field carries recoverable,
device-available information for spoofing detection.  A field is only worth
rebuilding the central CSV for if it is actually populated across devices.

This script parses every raw log under --data-root and, for each
(device x field), reports:
  - present_ratio: fraction of Raw rows where the column exists and is non-empty
  - finite_ratio:  fraction of Raw rows with a finite numeric value
  - distinct-ish variability via std of finite values (0 std => constant/useless)

It reads raw logs only; it does not touch tensors, labels, or outer test.
Output is a per-device coverage CSV plus a compact console summary.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional


# The 38/37/36-column Android GNSS-log Raw layouts (superset order).
RAW_COLUMNS_38 = [
    "ReadingType", "utcTimeMillis", "TimeNanos", "LeapSecond",
    "TimeUncertaintyNanos", "FullBiasNanos", "BiasNanos",
    "BiasUncertaintyNanos", "DriftNanosPerSecond",
    "DriftUncertaintyNanosPerSecond", "HardwareClockDiscontinuityCount",
    "Svid", "TimeOffsetNanos", "State", "ReceivedSvTimeNanos",
    "ReceivedSvTimeUncertaintyNanos", "Cn0DbHz",
    "PseudorangeRateMetersPerSecond",
    "PseudorangeRateUncertaintyMetersPerSecond",
    "AccumulatedDeltaRangeState", "AccumulatedDeltaRangeMeters",
    "AccumulatedDeltaRangeUncertaintyMeters", "CarrierFrequencyHz",
    "CarrierCycles", "CarrierPhase", "CarrierPhaseUncertainty",
    "MultipathIndicator", "SnrInDb", "ConstellationType", "AgcDb",
    "BasebandCn0DbHz", "FullInterSignalBiasNanos",
    "FullInterSignalBiasUncertaintyNanos", "SatelliteInterSignalBiasNanos",
    "SatelliteInterSignalBiasUncertaintyNanos", "CodeType",
    "ChipsetElapsedRealtimeNanos", "IsFullTracking",
]
COLUMN_MAP = {
    38: RAW_COLUMNS_38,
    37: RAW_COLUMNS_38[:37],
    36: RAW_COLUMNS_38[:36],
}

# Fields the current processed CSV already keeps (directly or as source of a
# derived feature). Everything else is "discarded".
KEPT_FIELDS = {
    "utcTimeMillis", "TimeNanos", "Svid", "ConstellationType",
    "CarrierFrequencyHz", "Cn0DbHz", "AgcDb",
    "ReceivedSvTimeUncertaintyNanos",
    "PseudorangeRateUncertaintyMetersPerSecond", "CodeType",
}

# Fields that are not per-signal numeric measurements; skip variability stats.
NON_NUMERIC = {"ReadingType", "CodeType"}

DEVICE_FOLDER_CANON = {
    "HUAWEI": "HUAWEI_Mate40",
    "HUAWEI_Mate40": "HUAWEI_Mate40",
    "Xiaomi_MI_8": "XiaoMi_MI8",
    "XiaoMi_MI8": "XiaoMi_MI8",
    "Xiaomi_23078RKD5C": "RedMi_K60",
    "RedMi_K60": "RedMi_K60",
    "watch1": "Google_Pixel_Watch1",
    "watch2": "Google_Pixel_Watch2",
    "Google_Pixel_Watch1": "Google_Pixel_Watch1",
    "Google_Pixel_Watch2": "Google_Pixel_Watch2",
    "Google_Pixel6": "Google_Pixel6",
}


def canon_device_from_path(path: Path) -> str:
    """Device name is the parent folder of the log file in the data layout."""
    name = path.parent.name
    return DEVICE_FOLDER_CANON.get(name, name)


class Accum:
    __slots__ = ("total", "present", "finite", "sum", "sumsq", "seen_values")

    def __init__(self) -> None:
        self.total = 0
        self.present = 0
        self.finite = 0
        self.sum = 0.0
        self.sumsq = 0.0
        self.seen_values = set()

    def add(self, raw_value: Optional[str], numeric: bool) -> None:
        self.total += 1
        if raw_value is None or raw_value == "":
            return
        self.present += 1
        if not numeric:
            if len(self.seen_values) < 64:
                self.seen_values.add(raw_value)
            return
        try:
            v = float(raw_value)
        except ValueError:
            return
        if math.isfinite(v):
            self.finite += 1
            self.sum += v
            self.sumsq += v * v
            if len(self.seen_values) < 64:
                self.seen_values.add(round(v, 6))

    def std(self) -> float:
        if self.finite < 2:
            return 0.0
        mean = self.sum / self.finite
        var = max(0.0, self.sumsq / self.finite - mean * mean)
        return math.sqrt(var)


def iter_raw_rows(path: Path):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("Raw,"):
                    yield line.rstrip("\n").split(",")
    except OSError:
        return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data_raw")
    parser.add_argument("--output-csv",
                        default="output/diagnostics/raw_field_coverage_v1/"
                                "raw_field_coverage_by_device.csv")
    args = parser.parse_args()

    root = Path(args.data_root)
    logs = sorted(root.rglob("*.txt"))
    logs = [p for p in logs if any(iter_raw_first(p))]
    print(f"found {len(logs)} candidate logs under {root}")

    # device -> field -> Accum
    acc: Dict[str, Dict[str, Accum]] = defaultdict(lambda: defaultdict(Accum))
    device_rows: Dict[str, int] = defaultdict(int)

    for i, path in enumerate(logs, 1):
        device = canon_device_from_path(path)
        n_rows = 0
        for cols in iter_raw_rows(path):
            layout = COLUMN_MAP.get(len(cols))
            if layout is None:
                continue
            n_rows += 1
            row = dict(zip(layout, cols))
            for field in RAW_COLUMNS_38:
                numeric = field not in NON_NUMERIC
                acc[device][field].add(row.get(field), numeric)
        device_rows[device] += n_rows
        if i % 20 == 0:
            print(f"  parsed {i}/{len(logs)} logs")

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    devices = sorted(acc.keys())

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["field", "kept_in_processed_csv", "device",
                    "raw_rows", "present_ratio", "finite_ratio",
                    "distinct_finite_upto64", "std_finite"])
        for field in RAW_COLUMNS_38:
            kept = field in KEPT_FIELDS
            for device in devices:
                a = acc[device][field]
                if a.total == 0:
                    continue
                w.writerow([
                    field, int(kept), device, a.total,
                    round(a.present / a.total, 4),
                    round(a.finite / a.total, 4),
                    len(a.seen_values),
                    round(a.std(), 6),
                ])

    print(f"\nwrote {out_path}")
    print(f"devices: {devices}")
    print(f"device raw-row totals: {dict(device_rows)}")

    # Console summary: for each discarded field, min/median present_ratio across
    # devices and how many devices have it usable (finite_ratio>0.5 & std>0).
    print("\n=== discarded fields: cross-device usability ===")
    print(f"{'field':<42}{'devs_usable':<12}{'min_present':<12}{'max_present'}")
    for field in RAW_COLUMNS_38:
        if field in KEPT_FIELDS or field in NON_NUMERIC:
            continue
        presents = []
        usable = 0
        for device in devices:
            a = acc[device][field]
            if a.total == 0:
                continue
            presents.append(a.present / a.total)
            if a.finite / a.total > 0.5 and a.std() > 0:
                usable += 1
        if not presents:
            continue
        print(f"{field:<42}{usable}/{len(presents):<10}"
              f"{min(presents):<12.3f}{max(presents):.3f}")


def iter_raw_first(path: Path):
    """Yield True once if the log has at least one Raw row (cheap probe)."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("Raw,"):
                    yield True
                    return
    except OSError:
        return


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Diagnostic 3: per-device clean/attack feature separability audit.

This is a model-agnostic diagnostic.  It answers a single question the failed
model experiments could not isolate: are the absolute C/N0 and AGC values of a
device's *clean* endpoints separable from the *attacked* endpoints, and more
importantly, does one device's clean distribution overlap another device's
attacked distribution?

If Mate40's clean C/N0/AGC distribution overlaps RedMi K60's attacked
distribution, then a single global decision boundary on absolute values cannot
serve both devices, which explains the opposite error directions (Mate40 over-
reports, K60/Pixel6 under-report) observed in the fold-6 diagnostics.  In that
case a causal per-device relative baseline is not merely "worth trying" but
structurally required, and that is itself a reportable paper finding.

It reads the central CSV (raw, *un-standardized* Cn0DbHz / AgcDb) rather than
the per-device standardized tensors, because the whole point is to measure the
cross-device absolute-value shift that the tensor standardization removes.

No tensors, no models, no outer-test leakage concerns: this only computes
descriptive statistics and distribution overlaps on the reviewed labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


# Focus features: the two absolute measurements the current models lean on and
# that carry device identity most strongly.
FEATURES = ["Cn0DbHz", "AgcDb"]

# Scenario groups reported separately: L5 is the known hard case.
SCENARIO_GROUPS = {
    "all": None,
    "L5_only": ["st_L5", "dy_L5"],
    "L1_only": ["st_L1", "dy_L1"],
}

PERCENTILES = [5, 25, 50, 75, 95]


def overlap_coefficient(a: np.ndarray, b: np.ndarray, bins: int = 100) -> float:
    """Histogram overlap coefficient (OVL) between two 1-D samples in [0,1].

    1.0 = identical distributions, 0.0 = disjoint support.  Computed on a shared
    binning over the pooled range.  Returns NaN if either sample is empty.
    """
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    lo = float(min(a.min(), b.min()))
    hi = float(max(a.max(), b.max()))
    if hi <= lo:
        return 1.0
    edges = np.linspace(lo, hi, bins + 1)
    pa, _ = np.histogram(a, bins=edges, density=False)
    pb, _ = np.histogram(b, bins=edges, density=False)
    pa = pa / pa.sum()
    pb = pb / pb.sum()
    return float(np.minimum(pa, pb).sum())


def summarize(values: np.ndarray) -> Dict[str, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0}
    out: Dict[str, float] = {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }
    for p in PERCENTILES:
        out[f"p{p}"] = float(np.percentile(values, p))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    usecols = ["DeviceName", "Scenario", "Label", "AgcDbMissing"] + FEATURES
    print(f"Loading {args.csv} ...", flush=True)
    df = pd.read_csv(args.csv, usecols=usecols)
    print(f"  {len(df):,} rows", flush=True)

    devices = sorted(df["DeviceName"].dropna().unique().tolist())

    report: Dict[str, object] = {
        "csv": str(args.csv),
        "features": FEATURES,
        "devices": devices,
        "note": (
            "Per-device clean(Label=0)/attack(Label=1) distribution summary on "
            "raw un-standardized central CSV. overlap_clean_attack is the same-"
            "device separability (low=separable). cross_device_overlap compares "
            "each device's attacked distribution against every other device's "
            "clean distribution (high=a global absolute threshold cannot serve "
            "both devices)."
        ),
        "groups": {},
    }

    for group_name, scenarios in SCENARIO_GROUPS.items():
        sub = df if scenarios is None else df[df["Scenario"].isin(scenarios)]
        group_out: Dict[str, object] = {"scenarios": scenarios, "per_device": {}, "cross_device_overlap": {}}

        # Per-device clean/attack summaries and same-device overlap.
        per_device_arrays: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
        for dev in devices:
            devdf = sub[sub["DeviceName"] == dev]
            if len(devdf) == 0:
                continue
            dev_out: Dict[str, object] = {}
            per_device_arrays[dev] = {}
            for feat in FEATURES:
                clean = devdf.loc[devdf["Label"] == 0, feat].to_numpy(dtype=float)
                attack = devdf.loc[devdf["Label"] == 1, feat].to_numpy(dtype=float)
                per_device_arrays[dev][feat] = {"clean": clean, "attack": attack}
                dev_out[feat] = {
                    "clean": summarize(clean),
                    "attack": summarize(attack),
                    "overlap_clean_attack": overlap_coefficient(clean, attack),
                }
            group_out["per_device"][dev] = dev_out

        # Cross-device: device A attacked vs device B clean, per feature.
        for feat in FEATURES:
            feat_cross: Dict[str, Dict[str, float]] = {}
            for a_dev in per_device_arrays:
                a_attack = per_device_arrays[a_dev][feat]["attack"]
                if a_attack[np.isfinite(a_attack)].size == 0:
                    continue
                row: Dict[str, float] = {}
                for b_dev in per_device_arrays:
                    b_clean = per_device_arrays[b_dev][feat]["clean"]
                    row[b_dev] = overlap_coefficient(a_attack, b_clean)
                feat_cross[a_dev] = row
            group_out["cross_device_overlap"][feat] = feat_cross

        report["groups"][group_name] = group_out
        print(f"[{group_name}] done ({'all' if scenarios is None else ','.join(scenarios)})", flush=True)

    out_path = args.output_dir / "diagnostics_device_separability.json"
    out_path.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()

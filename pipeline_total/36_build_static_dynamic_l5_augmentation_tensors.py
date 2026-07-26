"""Build E12a tensors: dynamic L5/L15 augmentation for a static outer test.

The static protocol is supplied by a complete-session outer manifest and a
static block manifest.  Every dynamic Session from ``--dynamic-manifest`` is
train-only and gets its own continuous run.  This keeps the static outer test
unchanged while making the dynamic augmentation explicit and auditable.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_static_builder():
    path = Path(__file__).with_name("20_build_static_timeblock_tensors.py")
    spec = importlib.util.spec_from_file_location("_static_tensor_builder_e12a", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = _load_static_builder()
KEYS = base.KEYS


def load_dynamic_manifest(path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(path, encoding="utf-8-sig")
    required = set(KEYS) | {"split"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"Dynamic manifest missing columns: {sorted(missing)}")
    manifest = manifest.loc[:, [*KEYS, "split"]].copy()
    for column in KEYS:
        manifest[column] = manifest[column].astype(str)
    if manifest.duplicated(KEYS).any():
        raise ValueError("Dynamic manifest has duplicate recording identities")
    if not manifest["split"].astype(str).eq("train").all():
        raise ValueError("E12a dynamic manifest may contain train-only Sessions")
    scenarios = set(manifest["Scenario"].astype(str))
    allowed = {"dy_L5", "dy_L_15"}
    if not scenarios or not scenarios.issubset(allowed):
        raise ValueError(f"E12a only accepts dynamic L5/L15 Sessions, got {sorted(scenarios)}")
    return manifest


def _read_rows(csv: Path) -> pd.DataFrame:
    required = [
        *KEYS, "DeviceName", base.SOURCE_COL, "TimeNanos", "TOW", "utcTimeMillis", "signal_id",
        "Label", "LabelStatus", *base.SOURCE_RAW_FEATURES,
    ]
    frame = pd.read_csv(csv, usecols=lambda column: column in set(required))
    missing = set(required).difference(frame.columns)
    if missing:
        raise ValueError(f"Processed CSV missing columns: {sorted(missing)}")
    for column in [*KEYS, "DeviceName", base.SOURCE_COL, "signal_id"]:
        frame[column] = frame[column].astype(str)
    frame["Label"] = (pd.to_numeric(frame["Label"], errors="coerce").fillna(0) > 0).astype(np.int8)
    for column in [*base.SOURCE_RAW_FEATURES, "TimeNanos", "TOW", "utcTimeMillis"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=[*KEYS, base.SOURCE_COL, "DeviceName", "TimeNanos", "utcTimeMillis", "signal_id"])


def _require_manifest_rows(frame: pd.DataFrame, manifest: pd.DataFrame, name: str) -> None:
    available = set(map(tuple, frame.loc[:, KEYS].drop_duplicates().itertuples(index=False, name=None)))
    requested = set(map(tuple, manifest.loc[:, KEYS].itertuples(index=False, name=None)))
    missing = sorted(requested.difference(available))
    if missing:
        raise ValueError(f"{name} contains Sessions absent from reviewed CSV rows: {missing}")


def build(args: argparse.Namespace) -> dict[str, Any]:
    base.configure(args.time_steps)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outer = base._load_outer_manifest(args.static_outer_manifest)
    dynamic_manifest = load_dynamic_manifest(args.dynamic_manifest)
    static_keys = set(map(tuple, outer.loc[:, KEYS].itertuples(index=False, name=None)))
    dynamic_keys = set(map(tuple, dynamic_manifest.loc[:, KEYS].itertuples(index=False, name=None)))
    overlap = static_keys.intersection(dynamic_keys)
    if overlap:
        raise ValueError(f"Static and dynamic manifests overlap: {sorted(overlap)}")
    block = base._load_block_manifest(args.static_block_manifest)
    block.to_csv(args.output_dir / "static_block_manifest.csv", index=False, encoding="utf-8-sig")

    base.LOG.info("Reading %s", args.csv)
    rows = _read_rows(args.csv)
    reviewed = rows.loc[rows["LabelStatus"].astype(str).eq("reviewed")].copy()
    static = reviewed.merge(outer[[*KEYS, "outer_role"]], on=KEYS, how="inner", validate="many_to_one")
    dynamic = reviewed.merge(dynamic_manifest, on=KEYS, how="inner", validate="many_to_one")
    _require_manifest_rows(static, outer, "Static outer manifest")
    _require_manifest_rows(dynamic, dynamic_manifest, "Dynamic train manifest")
    if static.empty or dynamic.empty:
        raise ValueError("E12a requires non-empty reviewed static and dynamic data")
    static["dataset_scope"] = "static"
    dynamic["dataset_scope"] = "dynamic"
    dynamic["outer_role"] = "train"
    frame = pd.concat([static, dynamic], ignore_index=True, copy=False)

    combined_manifest = pd.concat([
        outer.assign(dataset_scope="static"),
        dynamic_manifest.assign(dataset_scope="dynamic", outer_role="train"),
    ], ignore_index=True, sort=False)
    combined_manifest.to_csv(args.output_dir / "recording_manifest.csv", index=False, encoding="utf-8-sig")

    epoch_table = base._canonical_epoch_table(frame)
    device_names = sorted(frame["DeviceName"].unique().tolist())
    device_to_id = {name: index for index, name in enumerate(device_names)}
    (args.output_dir / "device_mapping.json").write_text(
        json.dumps(device_to_id, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    recording_rows = combined_manifest[[*KEYS, "dataset_scope"]].sort_values(
        ["dataset_scope", *KEYS], kind="mergesort"
    ).reset_index(drop=True)
    recording_to_id = {
        tuple(row): index
        for index, row in enumerate(recording_rows[[*KEYS, "dataset_scope"]].itertuples(index=False, name=None))
    }
    source_rows = frame[[*KEYS, "DeviceName", base.SOURCE_COL, "dataset_scope"]].drop_duplicates().sort_values(
        ["dataset_scope", *KEYS, "DeviceName", base.SOURCE_COL], kind="mergesort"
    ).reset_index(drop=True)
    source_to_id = {
        tuple(row): index
        for index, row in enumerate(source_rows[[*KEYS, "DeviceName", base.SOURCE_COL, "dataset_scope"]].itertuples(index=False, name=None))
    }
    signal_values = sorted(frame["signal_id"].unique().tolist())
    signal_to_id = {signal: index for index, signal in enumerate(signal_values)}
    (args.output_dir / "window_trace_index.json").write_text(json.dumps({
        "recordings": recording_rows.to_dict(orient="records"),
        "sources": source_rows.to_dict(orient="records"),
        "signal_ids": signal_values,
        "description": "Integer endpoint traceability indices; dynamic Session windows are train-only.",
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    chunks: dict[str, list[dict[str, list[np.ndarray]]]] = {"train": [], "val": [], "test": []}
    assignments: list[dict[str, object]] = []
    group_cols = [*KEYS, "DeviceName", base.SOURCE_COL, "dataset_scope", "outer_role"]
    for source_key, source_rows_group in tqdm(frame.groupby(group_cols, sort=True), desc="Building E12a source windows"):
        key = tuple(str(source_key[index]) for index in range(3))
        device_name = str(source_key[3])
        source_path = str(source_key[4])
        scope = str(source_key[5])
        outer_role = str(source_key[6])
        source = base._apply_agc_common_mode(base._aggregate_source(source_rows_group), args.agc_common_mode)
        source_times = np.sort(source["TimeNanos"].unique().astype(np.int64))
        if scope == "dynamic":
            splits = np.full(len(source_times), "train", dtype=object)
            segments = np.full(len(source_times), f"dynamic_{recording_to_id[(*key, scope)]}", dtype=object)
        else:
            source_epoch = epoch_table[
                (epoch_table[KEYS] == np.asarray(key)).all(axis=1)
                & epoch_table["DeviceName"].astype(str).eq(device_name)
                & epoch_table[base.SOURCE_COL].astype(str).eq(source_path)
            ].sort_values("TimeNanos")
            if len(source_epoch) != len(source_times):
                raise RuntimeError(f"Canonical epoch mapping mismatch for {source_key}")
            intervals = base._intervals_for_recording(block, key)
            canonical = np.sort(epoch_table[(epoch_table[KEYS] == np.asarray(key)).all(axis=1)]["epoch_utc_millis"].unique())
            epoch_indices = np.searchsorted(canonical, source_epoch["epoch_utc_millis"].to_numpy())
            splits, segments = base._assign_epoch_metadata(
                block, key, source_epoch["epoch_utc_millis"].to_numpy(), intervals, epoch_indices
            )
            if outer_role == "test":
                splits = np.where(splits == "unassigned", "test", splits)
                segments = np.where(segments == "unassigned", "0", segments)
            if np.all(splits == "unassigned"):
                raise ValueError(f"No static block assignment matches source {source_key}")
        source_identity = (*key, device_name, source_path, scope)
        parts = base._make_source_windows(
            source, splits, segments, device_to_id[device_name], recording_to_id[(*key, scope)],
            source_to_id[source_identity], signal_to_id, is_dynamic=scope == "dynamic",
        )
        for split, part in parts.items():
            chunks[split].append(part)
        counts = pd.Series(splits).value_counts().to_dict()
        assignments.append({
            **dict(zip(KEYS, key)), "DeviceName": device_name, base.SOURCE_COL: source_path,
            "dataset_scope": scope, **{f"epochs_{name}": int(value) for name, value in counts.items()},
        })

    datasets = {split: base._stack_windows(chunks[split]) for split in ("train", "val", "test")}
    base._fit_apply_scaler(datasets, args.output_dir)
    raw_dir = args.output_dir / "raw"
    stats_dir = args.output_dir / "stats"
    raw_dir.mkdir(exist_ok=True)
    stats_dir.mkdir(exist_ok=True)
    for split, data in datasets.items():
        common = {"mask": data["mask"], "y": data["y"], "is_dynamic": data["dynamic"], "device_id": data["device"]}
        trace = {name: data[name] for name in (
            "window_time_nanos", "endpoint_utc_millis", "endpoint_tow", "recording_id", "source_id", "signal_id"
        )}
        np.savez_compressed(raw_dir / f"{split}.npz", x=data["raw"], **common, **trace)
        np.savez_compressed(stats_dir / f"{split}.npz", x=data["stats"], **common)
    pd.DataFrame(assignments).to_csv(args.output_dir / "source_epoch_assignment_summary.csv", index=False, encoding="utf-8-sig")
    (raw_dir / "feature_names.json").write_text(json.dumps(base.RAW_NAMES, indent=2), encoding="utf-8")
    (stats_dir / "feature_names.json").write_text(json.dumps(base.STAT_NAMES, indent=2), encoding="utf-8")
    summary: dict[str, Any] = {}
    for split, data in datasets.items():
        active = data["mask"]
        summary[split] = {
            "windows": int(len(data["raw"])), "active": int(active.sum()),
            "positive": int((data["y"][active] == 1).sum()),
            "dynamic_windows": int(data["dynamic"].sum()),
            "dynamic_active": int(active[data["dynamic"]].sum()),
            "dynamic_positive": int((data["y"][data["dynamic"]][active[data["dynamic"]]] == 1).sum()),
        }
    metadata = {
        "experiment": "E12a_static_dynamic_l5_augmentation", "time_steps": base.TIME_STEPS,
        "max_signals": base.MAX_SIGNALS, "raw_features": base.RAW_NAMES, "stats_features": base.STAT_NAMES,
        "static_outer_manifest": str(args.static_outer_manifest), "static_block_manifest": str(args.static_block_manifest),
        "dynamic_train_manifest": str(args.dynamic_manifest), "dynamic_sessions": int(len(dynamic_manifest)),
        "dynamic_scenarios": sorted(dynamic_manifest["Scenario"].unique().tolist()),
        "outer_test_policy": "static Session only; dynamic Sessions are train-only",
        "agc_common_mode": args.agc_common_mode,
    }
    (args.output_dir / "tensor_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=ROOT / "output" / "processed_gnss_data.csv")
    parser.add_argument("--static-outer-manifest", type=Path, required=True)
    parser.add_argument("--static-block-manifest", type=Path, required=True)
    parser.add_argument("--dynamic-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--time-steps", type=int, default=5)
    parser.add_argument("--agc-common-mode", choices=("none", "same_time_band_median"), default="none")
    args = parser.parse_args()
    if args.time_steps < 2:
        parser.error("--time-steps must be at least 2")
    print(json.dumps(build(args), indent=2))


if __name__ == "__main__":
    main()

"""Build band-mean window tensors for static four-way scene classification.

This builder is a deliberate, standalone sibling of
``20_build_static_timeblock_tensors.py``.  Instead of emitting one endpoint per
satellite signal, it collapses each receiver epoch into a compact *band-mean*
vector: for every canonical epoch it averages the four baseline raw features
over all L1 signals and, separately, over all L5 signals of one device.  The
representation mirrors exactly what ``22b_plot_band_mean_dashboards.py`` draws
(blue L1 curve, red L5 curve), so a window is the model-ready form of the very
curves the four-way scene hypothesis was eyeballed on.

Per epoch the ordered feature vector is::

    [ L1: Cn0, Agc, SvTimeUnc, PrRateUnc,
      L5: Cn0, Agc, SvTimeUnc, PrRateUnc,
      L1Present, L5Present ]                                 # 8 means + 2 flags

``TIME_STEPS`` consecutive same-split, same-segment epochs of one device form a
window; the endpoint epoch's scene decides the four-way label:

    * endpoint TOW inside a reviewed attack interval -> spoofing_type_to_label
      of the recording Scenario (L1 -> 1, L5 -> 2, L1+L5 -> 3)
    * otherwise                                        -> 0 (normal)

A window whose endpoint epoch observed only one band (L1 xor L5) cannot express
the four-way scene and is flagged ``single_band_mask=True``: its label is forced
to 0 and every downstream trainer and metric must exclude it.  These windows are
retained in the tensors on purpose so a later, separate path can consume them.

Split assignment reuses the protocol's epoch-level manifest
(``epoch_split_manifest.csv``) verbatim: each device epoch is mapped to its
canonical UTC second and inherits that second's train/val/test/guard split and
segment.  Windows never cross a split, a segment, or a receiver gap, so no
band-mean history borrows observations across a train/validation boundary.

Outputs (compatible in spirit with the training scripts)::

    output_dir/{train,val,test}.npz   # x=[B, TIME_STEPS, 10], y, single_band_mask, ...
    output_dir/feature_names.json
    output_dir/scaler.json
    output_dir/tensor_metadata.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
KEYS = ["Environment", "Scenario", "Session"]
SOURCE_COL = "SourceRelativePath"
TIME_STEPS = 5
EPOCH_GAP_NS = 2_000_000_000  # a missing receiver epoch starts a new run
BANDS = (1, 5)

FEATURES = [
    "Cn0DbHz",
    "AgcDb",
    "ReceivedSvTimeUncertaintyNanos",
    "PseudorangeRateUncertaintyMetersPerSecond",
]
# 8 continuous band means are standardized; the 2 presence flags stay in [0, 1].
FEATURE_NAMES = (
    [f"L1_{name}" for name in FEATURES]
    + [f"L5_{name}" for name in FEATURES]
    + ["L1Present", "L5Present"]
)
CONTINUOUS_COUNT = len(FEATURES) * len(BANDS)  # 8
FEATURE_COUNT = len(FEATURE_NAMES)  # 10

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
LOG = logging.getLogger(__name__)


def _norm_keys(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for key in KEYS:
        out[key] = out[key].astype(str)
    return out


def load_scene_labels(config_path: Path) -> tuple[dict, dict]:
    """Return (intervals_by_recording, scenario_to_class) from the formal config."""
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    labeling = config.get("labeling", {})
    type_map = labeling.get("spoofing_type_to_label", {})
    if not isinstance(type_map, dict) or not type_map:
        raise ValueError(f"labeling.spoofing_type_to_label missing in {config_path}")
    scenario_to_class = {str(k): int(v) for k, v in type_map.items()}
    session_intervals = labeling.get("session_spoofing_tow_intervals", {})
    if not isinstance(session_intervals, dict):
        raise ValueError(f"labeling.session_spoofing_tow_intervals must be a mapping in {config_path}")
    intervals: dict[tuple[str, str, str], list[tuple[float, float]]] = {}
    for environment, scenarios in session_intervals.items():
        if not isinstance(scenarios, dict):
            continue
        for scenario, sessions in scenarios.items():
            if not isinstance(sessions, dict):
                continue
            for session, entry in sessions.items():
                if not isinstance(entry, dict):
                    continue
                if str(entry.get("status", "needs_review")).strip().lower() != "reviewed":
                    continue
                parsed: list[tuple[float, float]] = []
                for raw_interval in entry.get("intervals", []) or []:
                    if not isinstance(raw_interval, (list, tuple)) or len(raw_interval) != 2:
                        raise ValueError(
                            f"Invalid interval {raw_interval!r} for {environment}/{scenario}/{session}"
                        )
                    start_tow, end_tow = float(raw_interval[0]), float(raw_interval[1])
                    if not np.isfinite(start_tow) or not np.isfinite(end_tow) or end_tow < start_tow:
                        raise ValueError(
                            f"Invalid interval {raw_interval!r} for {environment}/{scenario}/{session}"
                        )
                    parsed.append((start_tow, end_tow))
                intervals[(str(environment), str(scenario), str(session))] = sorted(parsed)
    return intervals, scenario_to_class


def load_epoch_manifest(path: Path) -> dict[tuple[str, str, str], dict[int, tuple[str, str]]]:
    """Map (Env, Scenario, Session) -> {canonical_epoch_ms -> (split, segment_key)}.

    Guard epochs are retained with split ``guard`` so windows touching them are
    dropped exactly as in the source-level protocol.
    """
    manifest = _norm_keys(pd.read_csv(path, encoding="utf-8-sig"))
    required = {*KEYS, "canonical_epoch_ms", "split"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"Epoch manifest {path} missing columns: {sorted(missing)}")
    segment_col = next(
        (c for c in ("segment_key", "segment_id", "block_uid") if c in manifest.columns),
        None,
    )
    manifest["canonical_epoch_ms"] = pd.to_numeric(
        manifest["canonical_epoch_ms"], errors="raise"
    ).astype(np.int64)
    lookup: dict[tuple[str, str, str], dict[int, tuple[str, str]]] = {}
    for row in manifest.itertuples(index=False):
        identity = (getattr(row, "Environment"), getattr(row, "Scenario"), getattr(row, "Session"))
        split = str(getattr(row, "split"))
        segment = str(getattr(row, segment_col)) if segment_col else "0"
        lookup.setdefault(identity, {})[int(getattr(row, "canonical_epoch_ms"))] = (split, segment)
    return lookup


def endpoint_class(
    identity: tuple[str, str, str],
    tow: float,
    intervals: dict[tuple[str, str, str], list[tuple[float, float]]],
    scenario_to_class: dict[str, int],
) -> int:
    """Four-way class of an endpoint epoch from reviewed intervals + scenario."""
    scenario = identity[1]
    session_intervals = intervals.get(identity, [])
    for start_tow, end_tow in session_intervals:
        if start_tow <= tow <= end_tow:
            if scenario not in scenario_to_class:
                raise ValueError(f"Scenario {scenario!r} absent from spoofing_type_to_label")
            return int(scenario_to_class[scenario])
    return 0


def band_epoch_table(source: pd.DataFrame) -> pd.DataFrame:
    """Collapse one device's rows into one band-mean row per receiver epoch.

    Returns a frame indexed by TimeNanos with the 8 continuous band means (NaN
    when a band is absent at that epoch), the two presence flags, and endpoint
    UTC/TOW representatives.  This is the epoch-level analogue of the per-TOW
    band mean drawn by the dashboard.
    """
    band = pd.to_numeric(source["FreqBand"], errors="coerce")
    columns: dict[str, pd.Series] = {}
    present: dict[int, pd.Series] = {}
    for band_value in BANDS:
        sub = source[band.eq(band_value)]
        grouped = sub.groupby("TimeNanos", sort=True)
        for feature in FEATURES:
            mean = grouped[feature].mean() if len(sub) else pd.Series(dtype=float)
            columns[f"{'L1' if band_value == 1 else 'L5'}_{feature}"] = mean
        present[band_value] = grouped.size() if len(sub) else pd.Series(dtype=float)
    table = pd.DataFrame(columns)
    # Presence: a band is observed at an epoch iff it contributed >=1 signal row.
    table["L1Present"] = present[1].reindex(table.index).fillna(0).gt(0).astype(np.float32)
    table["L5Present"] = present[5].reindex(table.index).fillna(0).gt(0).astype(np.float32)
    utc = source.groupby("TimeNanos", sort=True)["utcTimeMillis"].median()
    tow = source.groupby("TimeNanos", sort=True)["TOW"].median()
    table["utcTimeMillis"] = utc.reindex(table.index)
    table["TOW"] = tow.reindex(table.index)
    return table.sort_index()


def build_windows(
    table: pd.DataFrame,
    identity: tuple[str, str, str],
    device_id: int,
    recording_id: int,
    epoch_splits: dict[int, tuple[str, str]],
    intervals: dict[tuple[str, str, str], list[tuple[float, float]]],
    scenario_to_class: dict[str, int],
    is_dynamic: bool,
) -> dict[str, list]:
    """Emit all eligible band-mean windows for one device."""
    parts: dict[str, list] = {
        split: {
            "x": [], "y": [], "single_band": [], "device": [], "recording": [],
            "dynamic": [], "window_time_nanos": [], "endpoint_utc_millis": [], "endpoint_tow": [],
        }
        for split in ("train", "val", "test")
    }
    times = table.index.to_numpy(dtype=np.int64)
    if len(times) < TIME_STEPS:
        return parts
    feature_matrix = table[FEATURE_NAMES].to_numpy(dtype=np.float64)
    utc_values = table["utcTimeMillis"].to_numpy(dtype=np.float64)
    tow_values = table["TOW"].to_numpy(dtype=np.float64)
    # Canonical second per epoch drives the manifest split/segment lookup.
    canonical = (np.round(utc_values / 1000.0) * 1000.0).astype(np.int64)
    split_seg = [epoch_splits.get(int(value)) for value in canonical]

    for end_i in range(TIME_STEPS - 1, len(times)):
        window_slice = slice(end_i - TIME_STEPS + 1, end_i + 1)
        window_times = times[window_slice]
        if np.any(np.diff(window_times) > EPOCH_GAP_NS):
            continue
        assignments = split_seg[window_slice]
        if any(item is None for item in assignments):
            continue
        splits = {item[0] for item in assignments}
        segments = {item[1] for item in assignments}
        if len(splits) != 1 or len(segments) != 1:
            continue
        split = next(iter(splits))
        if split not in parts:  # drops 'guard' and any unexpected split
            continue
        x_window = feature_matrix[window_slice].astype(np.float32)  # [T, 10]
        endpoint_tow = float(tow_values[end_i])
        endpoint_present_l1 = feature_matrix[end_i, FEATURE_NAMES.index("L1Present")] > 0
        endpoint_present_l5 = feature_matrix[end_i, FEATURE_NAMES.index("L5Present")] > 0
        single_band = not (endpoint_present_l1 and endpoint_present_l5)
        label = 0 if single_band else endpoint_class(identity, endpoint_tow, intervals, scenario_to_class)
        out = parts[split]
        out["x"].append(x_window)
        out["y"].append(np.int64(label))
        out["single_band"].append(np.bool_(single_band))
        out["device"].append(np.int64(device_id))
        out["recording"].append(np.int32(recording_id))
        out["dynamic"].append(np.bool_(is_dynamic))
        out["window_time_nanos"].append(np.int64(window_times[-1]))
        out["endpoint_utc_millis"].append(np.float64(utc_values[end_i]))
        out["endpoint_tow"].append(np.float64(endpoint_tow))
    return parts


def stack_split(parts: list[dict[str, list]]) -> dict[str, np.ndarray]:
    keys = [
        "x", "y", "single_band", "device", "recording",
        "dynamic", "window_time_nanos", "endpoint_utc_millis", "endpoint_tow",
    ]
    collected: dict[str, list] = {key: [] for key in keys}
    for part in parts:
        for key in keys:
            collected[key].extend(part[key])
    if not collected["x"]:
        return {
            "x": np.empty((0, TIME_STEPS, FEATURE_COUNT), np.float32),
            "y": np.empty((0,), np.int64),
            "single_band": np.empty((0,), np.bool_),
            "device": np.empty((0,), np.int64),
            "recording": np.empty((0,), np.int32),
            "dynamic": np.empty((0,), np.bool_),
            "window_time_nanos": np.empty((0,), np.int64),
            "endpoint_utc_millis": np.empty((0,), np.float64),
            "endpoint_tow": np.empty((0,), np.float64),
        }
    return {
        "x": np.stack(collected["x"]),
        "y": np.asarray(collected["y"], dtype=np.int64),
        "single_band": np.asarray(collected["single_band"], dtype=np.bool_),
        "device": np.asarray(collected["device"], dtype=np.int64),
        "recording": np.asarray(collected["recording"], dtype=np.int32),
        "dynamic": np.asarray(collected["dynamic"], dtype=np.bool_),
        "window_time_nanos": np.asarray(collected["window_time_nanos"], dtype=np.int64),
        "endpoint_utc_millis": np.asarray(collected["endpoint_utc_millis"], dtype=np.float64),
        "endpoint_tow": np.asarray(collected["endpoint_tow"], dtype=np.float64),
    }


def fit_apply_scaler(datasets: dict[str, dict[str, np.ndarray]], output_dir: Path) -> None:
    """Per-device train-only standardization of the 8 continuous band means.

    Missing-band means are NaN; only finite entries contribute to the fit and,
    after standardization, missing entries are set to the (post-scaling) mean of
    0.  Presence flags remain physical [0, 1].
    """
    train = datasets["train"]
    if len(train["x"]) == 0:
        raise ValueError("Train split has no windows; cannot fit scalers")

    def fit(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        finite = np.isfinite(values)
        counts = finite.sum(axis=0).astype(np.float64)
        safe_counts = np.where(counts > 0, counts, 1.0)
        safe_values = np.where(finite, values, 0.0)
        mean = safe_values.sum(axis=0) / safe_counts
        centered = np.where(finite, values - mean.reshape(1, -1), 0.0)
        std = np.sqrt((centered * centered).sum(axis=0) / safe_counts)
        mean = np.where(np.isfinite(mean), mean, 0.0)
        std = np.where(np.isfinite(std) & (std >= 1e-6), std, 1.0)
        return mean.astype(np.float64), std.astype(np.float64)

    train_cont = train["x"][:, :, :CONTINUOUS_COUNT].reshape(-1, CONTINUOUS_COUNT)
    global_mean, global_std = fit(train_cont)
    per_device: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for device in np.unique(train["device"]):
        sel = train["device"] == device
        values = train["x"][sel][:, :, :CONTINUOUS_COUNT].reshape(-1, CONTINUOUS_COUNT)
        per_device[int(device)] = fit(values)

    for data in datasets.values():
        if len(data["x"]) == 0:
            continue
        for device in np.unique(data["device"]):
            sel = data["device"] == device
            mean, std = per_device.get(int(device), (global_mean, global_std))
            block = data["x"][sel][:, :, :CONTINUOUS_COUNT].astype(np.float64)
            standardized = (block - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)
            standardized = np.where(np.isfinite(standardized), standardized, 0.0)
            data["x"][sel, :, :CONTINUOUS_COUNT] = standardized.astype(np.float32)
        data["x"][:, :, CONTINUOUS_COUNT:] = np.nan_to_num(
            data["x"][:, :, CONTINUOUS_COUNT:], nan=0.0, posinf=0.0, neginf=0.0
        )
        if not np.isfinite(data["x"]).all():
            raise ValueError("Non-finite values remain after train-only scaling")

    def serial(pair: tuple[np.ndarray, np.ndarray]) -> dict[str, list[float]]:
        return {"mean": pair[0].tolist(), "std": pair[1].tolist()}

    (output_dir / "scaler.json").write_text(
        json.dumps(
            {
                "continuous_features": FEATURE_NAMES[:CONTINUOUS_COUNT],
                "unscaled_features": FEATURE_NAMES[CONTINUOUS_COUNT:],
                "global": serial((global_mean, global_std)),
                "per_device": {str(k): serial(v) for k, v in per_device.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def build(
    csv_path: Path,
    manifest_path: Path,
    outer_manifest_path: Path,
    config_path: Path,
    output_dir: Path,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    intervals, scenario_to_class = load_scene_labels(config_path)
    epoch_lookup = load_epoch_manifest(manifest_path)

    outer = _norm_keys(pd.read_csv(outer_manifest_path, encoding="utf-8-sig"))
    outer_required = {*KEYS, "split"}
    missing = outer_required.difference(outer.columns)
    if missing:
        raise ValueError(f"Outer manifest {outer_manifest_path} missing columns: {sorted(missing)}")
    kept_recordings = set(map(tuple, outer[KEYS].to_numpy()))

    usecols = [
        *KEYS, "DeviceName", SOURCE_COL, "TimeNanos", "TOW", "utcTimeMillis",
        "FreqBand", "Label", "LabelStatus", *FEATURES,
    ]
    LOG.info("Reading %s", csv_path)
    df = pd.read_csv(csv_path, usecols=lambda c: c in set(usecols))
    missing = set(usecols).difference(df.columns)
    if missing:
        raise ValueError(f"Processed CSV missing columns: {sorted(missing)}")
    df = _norm_keys(df)
    scenario = df["Scenario"].astype(str)
    df = df[(df["LabelStatus"].astype(str) == "reviewed") & scenario.str.startswith("st_")].copy()
    df["_identity"] = list(zip(df["Environment"], df["Scenario"], df["Session"]))
    df = df[df["_identity"].isin(kept_recordings)].copy()
    if df.empty:
        raise ValueError("No reviewed static rows match the outer manifest")
    for column in [*FEATURES, "TimeNanos", "TOW", "utcTimeMillis", "FreqBand"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=[*KEYS, "DeviceName", "TimeNanos", "utcTimeMillis", "FreqBand"])
    df["DeviceName"] = df["DeviceName"].astype(str)

    device_names = sorted(df["DeviceName"].unique())
    device_to_id = {name: i for i, name in enumerate(device_names)}
    recording_rows = (
        df[KEYS].drop_duplicates().sort_values(KEYS, kind="mergesort").reset_index(drop=True)
    )
    recording_to_id = {
        tuple(row): i
        for i, row in enumerate(recording_rows.itertuples(index=False, name=None))
    }

    chunks: dict[str, list[dict[str, list]]] = {"train": [], "val": [], "test": []}
    for source_key, source in tqdm(
        df.groupby([*KEYS, "DeviceName", SOURCE_COL], sort=True),
        desc="Building band-mean windows",
    ):
        identity = tuple(str(source_key[i]) for i in range(3))
        device_name = str(source_key[3])
        epoch_splits = epoch_lookup.get(identity, {})
        if not epoch_splits:
            continue
        table = band_epoch_table(source)
        part = build_windows(
            table,
            identity,
            device_to_id[device_name],
            recording_to_id[identity],
            epoch_splits,
            intervals,
            scenario_to_class,
            is_dynamic=identity[1].startswith("dy_"),
        )
        for split, values in part.items():
            chunks[split].append(values)

    datasets = {split: stack_split(chunks[split]) for split in ("train", "val", "test")}
    fit_apply_scaler(datasets, output_dir)

    for split, data in datasets.items():
        np.savez_compressed(
            output_dir / f"{split}.npz",
            x=data["x"],
            y=data["y"],
            single_band_mask=data["single_band"],
            device_id=data["device"],
            recording_id=data["recording"],
            is_dynamic=data["dynamic"],
            window_time_nanos=data["window_time_nanos"],
            endpoint_utc_millis=data["endpoint_utc_millis"],
            endpoint_tow=data["endpoint_tow"],
        )

    (output_dir / "feature_names.json").write_text(json.dumps(FEATURE_NAMES, indent=2), encoding="utf-8")
    (output_dir / "device_mapping.json").write_text(
        json.dumps(device_to_id, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    def split_stats(data: dict[str, np.ndarray]) -> dict:
        total = int(len(data["y"]))
        single = data["single_band"]
        usable = ~single
        usable_y = data["y"][usable]
        class_counts = {str(c): int((usable_y == c).sum()) for c in range(4)}
        return {
            "windows": total,
            "single_band_excluded": int(single.sum()),
            "usable": int(usable.sum()),
            "usable_class_counts": class_counts,
        }

    metadata = {
        "representation": "band_mean_window",
        "time_steps": TIME_STEPS,
        "feature_count": FEATURE_COUNT,
        "continuous_count": CONTINUOUS_COUNT,
        "feature_names": FEATURE_NAMES,
        "features": FEATURES,
        "bands": list(BANDS),
        "num_classes": 4,
        "class_semantics": {"0": "normal", "1": "L1", "2": "L5", "3": "L1+L5"},
        "single_band_policy": (
            "endpoint epoch observing only one band is labeled 0 and flagged "
            "single_band_mask=True; trainers and metrics must exclude it"
        ),
        "scenario_to_class": scenario_to_class,
        "data_scope": "static",
        "canonical_clock": "utcTimeMillis",
        "window_clock": "TimeNanos",
        "device_mapping": device_to_id,
        "split_stats": {split: split_stats(data) for split, data in datasets.items()},
    }
    (output_dir / "tensor_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    for split, data in datasets.items():
        stats = metadata["split_stats"][split]
        LOG.info(
            "%s windows=%d single_band_excluded=%d usable=%d class_counts=%s",
            split, stats["windows"], stats["single_band_excluded"],
            stats["usable"], stats["usable_class_counts"],
        )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=ROOT / "output" / "processed_gnss_data.csv")
    parser.add_argument(
        "--epoch-manifest", type=Path, required=True,
        help="Protocol fold epoch_split_manifest.csv (canonical_epoch_ms + split).",
    )
    parser.add_argument(
        "--outer-manifest", type=Path, required=True,
        help="Protocol fold recording_split_manifest.csv (recordings + split).",
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "preprocessing.yml")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = build(
        args.csv, args.epoch_manifest, args.outer_manifest, args.config, args.output_dir
    )
    print(json.dumps(summary["split_stats"], indent=2))


if __name__ == "__main__":
    main()

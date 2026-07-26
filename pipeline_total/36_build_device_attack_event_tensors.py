"""Build device-window attack-event tensors from existing signal tensors.

This first hierarchical experiment preserves the signal tensors and changes
only the decision unit: one receiver-device time window is one event sample.
The event label comes from the reviewed Session attack interval regardless of
frequency, so L1 suppression can be attack evidence during an L5-only event.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


SPLITS = ("train", "val", "test")
REQUIRED_STATS = (
    "Cn0DbHzLastW5", "Cn0DbHzSlopeW5", "AgcDbLastW5", "AgcDbStdW5",
    "ReceivedSvTimeUncertaintyNanosStdW5", "IsL5", "SignalHistoryRatioW5",
    "AgcObservedRatioW5",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signal-data-dir", type=Path, required=True)
    parser.add_argument("--label-config", type=Path, default=Path("configs/preprocessing.yml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature-set", choices=("all", "l1_only", "l5_only", "no_cross"), default="all")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {path}") from exc


def reduce(values: np.ndarray, kind: str) -> float:
    if len(values) == 0:
        return float("nan")
    if kind == "median":
        return float(np.median(values))
    if kind == "q25":
        return float(np.quantile(values, 0.25))
    if kind == "q75":
        return float(np.quantile(values, 0.75))
    if kind == "positive_ratio":
        return float(np.mean(values > 0.0))
    if kind == "negative_ratio":
        return float(np.mean(values < 0.0))
    if kind == "coverage_loss_ratio":
        return float(np.mean(values < 0.999))
    raise ValueError(f"Unsupported reduction: {kind}")


def aggregate_window(stats: np.ndarray, mask: np.ndarray, index: dict[str, int]) -> tuple[np.ndarray, list[str]]:
    """Summarize all valid L1/L5 signal endpoints at one device time."""
    if stats.ndim != 3 or stats.shape[1] != 1:
        raise ValueError(f"Expected [signals, 1, features], received {stats.shape}")
    values = stats[:, 0, :]
    active = np.asarray(mask, dtype=bool)
    is_l5 = values[:, index["IsL5"]] >= 0.5
    definitions = (
        ("Cn0DbHzLastW5", "cn0_last", ("median", "q25", "q75")),
        ("Cn0DbHzSlopeW5", "cn0_slope", ("median", "positive_ratio", "negative_ratio")),
        ("AgcDbLastW5", "agc_last", ("median",)),
        ("AgcDbStdW5", "agc_std", ("median",)),
        ("ReceivedSvTimeUncertaintyNanosStdW5", "rx_time_unc_std", ("median",)),
        ("SignalHistoryRatioW5", "history", ("median", "coverage_loss_ratio")),
        ("AgcObservedRatioW5", "agc_observed", ("median", "coverage_loss_ratio")),
    )
    result: list[float] = []
    names: list[str] = []
    summaries: dict[int, dict[str, float]] = {}
    for band, prefix in ((1, "l1"), (5, "l5")):
        selected = active & (is_l5 if band == 5 else ~is_l5)
        result.append(float(np.log1p(selected.sum())))
        names.append(f"{prefix}_log_signal_count")
        summary: dict[str, float] = {}
        for source, short, reductions in definitions:
            source_values = values[selected, index[source]]
            for reduction in reductions:
                key = f"{short}_{reduction}"
                value = reduce(source_values, reduction)
                result.append(value)
                names.append(f"{prefix}_{key}")
                summary[key] = value
        summaries[band] = summary
    for key in ("cn0_last_median", "cn0_slope_median", "agc_last_median", "history_median"):
        result.append(summaries[5][key] - summaries[1][key])
        names.append(f"l5_minus_l1_{key}")
    result.append(summaries[5]["cn0_slope_positive_ratio"] + summaries[1]["cn0_slope_negative_ratio"])
    names.append("coupled_l5_up_plus_l1_down_ratio")
    return np.asarray(result, dtype=np.float32), names


def reviewed_intervals(trace: dict[str, Any], config: dict[str, Any]) -> dict[int, list[tuple[float, float]]]:
    entries = config.get("labeling", {}).get("session_spoofing_tow_intervals", {})
    result: dict[int, list[tuple[float, float]]] = {}
    recordings = trace.get("recordings")
    if not isinstance(recordings, list):
        raise ValueError("window_trace_index.json has no recording index")
    for recording_id, row in enumerate(recordings):
        try:
            entry = entries[row["Environment"]][row["Scenario"]][row["Session"]]
            if entry["status"] != "reviewed":
                raise ValueError(f"Recording is not reviewed: {row}")
            raw_intervals = entry["intervals"]
        except KeyError as exc:
            raise ValueError(f"Missing session label configuration: {row}") from exc
        parsed = [(float(interval[0]), float(interval[1])) for interval in raw_intervals]
        if any(end < start for start, end in parsed):
            raise ValueError(f"Descending label interval for {row}")
        result[recording_id] = parsed
    return result


def labels_from_interval(recording_ids: np.ndarray, tows: np.ndarray, intervals: dict[int, list[tuple[float, float]]]) -> np.ndarray:
    labels = np.zeros(len(recording_ids), dtype=np.int64)
    for row, (recording_id, tow) in enumerate(zip(recording_ids, tows)):
        if not np.isfinite(tow):
            raise ValueError(f"Non-finite endpoint TOW at window {row}")
        labels[row] = int(any(start <= tow <= end for start, end in intervals[int(recording_id)]))
    return labels


def select_feature_indices(names: list[str], feature_set: str) -> list[int]:
    if feature_set == "all":
        selected = names
    elif feature_set == "l1_only":
        selected = [name for name in names if name.startswith("l1_")]
    elif feature_set == "l5_only":
        selected = [name for name in names if name.startswith("l5_")]
    else:
        selected = [name for name in names if not name.startswith("l5_minus_") and name != "coupled_l5_up_plus_l1_down_ratio"]
    if not selected:
        raise ValueError(f"Feature set {feature_set!r} selected no device features")
    return [names.index(name) for name in selected]


def load_device_split(signal_dir: Path, split: str, index: dict[str, int], intervals: dict[int, list[tuple[float, float]]]) -> tuple[dict[str, np.ndarray], list[str]]:
    raw_path = signal_dir / "raw" / f"{split}.npz"
    stats_path = signal_dir / "stats" / f"{split}.npz"
    with np.load(raw_path, allow_pickle=False) as raw, np.load(stats_path, allow_pickle=False) as stats:
        required = {"mask", "y", "device_id", "recording_id", "source_id", "endpoint_tow", "endpoint_utc_millis"}
        if missing := required.difference(raw.files):
            raise ValueError(f"{raw_path} missing {sorted(missing)}")
        if missing := {"x", "mask", "device_id"}.difference(stats.files):
            raise ValueError(f"{stats_path} missing {sorted(missing)}")
        if not np.array_equal(raw["mask"], stats["mask"]) or not np.array_equal(raw["device_id"], stats["device_id"]):
            raise ValueError(f"Raw/stats contracts differ for {split}")
        rows: list[np.ndarray] = []
        names: list[str] | None = None
        for row_stats, row_mask in zip(stats["x"], raw["mask"]):
            features, feature_names = aggregate_window(row_stats, row_mask, index)
            if names is None:
                names = feature_names
            elif names != feature_names:
                raise RuntimeError("Device feature names changed between windows")
            rows.append(features)
        x = np.stack(rows) if rows else np.empty((0, 0), dtype=np.float32)
        recording_ids = raw["recording_id"].astype(np.int64)
        event_y = labels_from_interval(recording_ids, raw["endpoint_tow"], intervals)
        direct_positive = (raw["y"] == 1) & raw["mask"]
        if np.any(direct_positive.any(axis=1) & (event_y == 0)):
            raise ValueError(f"Direct target-band positives fall outside an event interval in {split}")
        return {
            "x": x,
            "y_event": event_y,
            "device_id": raw["device_id"].astype(np.int64),
            "recording_id": recording_ids,
            "source_id": raw["source_id"].astype(np.int64),
            "endpoint_tow": raw["endpoint_tow"].astype(np.float64),
            "endpoint_utc_millis": raw["endpoint_utc_millis"].astype(np.float64),
        }, names or []


def scale_train_only(datasets: dict[str, dict[str, np.ndarray]]) -> dict[str, list[float]]:
    train_x = datasets["train"]["x"]
    if len(train_x) == 0:
        raise ValueError("No train device windows")
    finite = np.isfinite(train_x)
    count = finite.sum(axis=0)
    if np.any(count == 0):
        raise ValueError(f"No finite train value for feature indices {np.flatnonzero(count == 0).tolist()}")
    mean = np.where(finite, train_x, 0.0).sum(axis=0) / count
    variance = np.where(finite, (train_x - mean) ** 2, 0.0).sum(axis=0) / count
    std = np.where(np.sqrt(variance) >= 1e-6, np.sqrt(variance), 1.0)
    for data in datasets.values():
        data["x"] = ((np.where(np.isfinite(data["x"]), data["x"], mean) - mean) / std).astype(np.float32)
    return {"mean": mean.astype(float).tolist(), "std": std.astype(float).tolist()}


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    names = read_json(args.signal_data_dir / "stats" / "feature_names.json")
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise ValueError("Invalid stats feature name list")
    if missing := [name for name in REQUIRED_STATS if name not in names]:
        raise ValueError(f"Signal tensor is missing {missing}")
    index = {name: names.index(name) for name in REQUIRED_STATS}
    trace = read_json(args.signal_data_dir / "window_trace_index.json")
    config = yaml.safe_load(args.label_config.read_text(encoding="utf-8"))
    intervals = reviewed_intervals(trace, config)
    datasets: dict[str, dict[str, np.ndarray]] = {}
    feature_names: list[str] | None = None
    for split in SPLITS:
        data, candidate_names = load_device_split(args.signal_data_dir, split, index, intervals)
        if feature_names is None and candidate_names:
            feature_names = candidate_names
        elif candidate_names and feature_names != candidate_names:
            raise RuntimeError("Feature contract differs between splits")
        datasets[split] = data
    if not feature_names:
        raise ValueError("No device features produced")
    for data in datasets.values():
        if len(data["x"]) == 0:
            data["x"] = np.empty((0, len(feature_names)), dtype=np.float32)
    selected_indices = select_feature_indices(feature_names, args.feature_set)
    feature_names = [feature_names[index] for index in selected_indices]
    for data in datasets.values():
        data["x"] = data["x"][:, selected_indices]
    scaler = scale_train_only(datasets)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, data in datasets.items():
        np.savez_compressed(args.output_dir / f"{split}.npz", **data)
    (args.output_dir / "feature_names.json").write_text(json.dumps(feature_names, indent=2), encoding="utf-8")
    (args.output_dir / "scaler.json").write_text(json.dumps(scaler, indent=2), encoding="utf-8")
    metadata = {
        "task": "device_attack_event",
        "label_semantics": "reviewed session attack interval independent of direct target frequency",
        "signal_data_dir": str(args.signal_data_dir),
        "label_config": str(args.label_config),
        "feature_set": args.feature_set,
        "feature_names": feature_names,
        "device_mapping": read_json(args.signal_data_dir / "device_mapping.json"),
        "recordings": trace["recordings"],
        "splits": {
            split: {"windows": int(len(data["x"])), "positive": int(data["y_event"].sum()), "negative": int(len(data["y_event"]) - data["y_event"].sum())}
            for split, data in datasets.items()
        },
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metadata["splits"], indent=2))


if __name__ == "__main__":
    main()

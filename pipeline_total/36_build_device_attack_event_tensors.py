"""Build device-window attack-event tensors from existing signal tensors.

This first hierarchical experiment preserves the signal tensors and changes
only the decision unit: one receiver-device time window is one event sample.
The event label comes from the reviewed Session attack interval regardless of
frequency, so L1 suppression can be attack evidence during an L5-only event.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


SPLITS = ("train", "val", "test")
ROOT = Path(__file__).resolve().parents[1]
RESPONSE_STATE_TO_ID = {
    "normal": 0,
    "no_observable_response": 0,
    "attack_associated_anomaly": 1,
    "direct_spoof": 2,
}
REQUIRED_STATS = (
    "Cn0DbHzLastW5", "Cn0DbHzSlopeW5", "AgcDbLastW5", "AgcDbStdW5",
    "ReceivedSvTimeUncertaintyNanosStdW5", "IsL5", "SignalHistoryRatioW5",
    "AgcObservedRatioW5",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signal-data-dir", type=Path, required=True)
    parser.add_argument("--label-config", type=Path, default=ROOT / "configs" / "preprocessing.yml")
    parser.add_argument("--response-labels", type=Path, default=ROOT / "docs" / "device_response_intervals.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--device-aggregate-profile", choices=("robust", "sparse_extreme"), default="robust",
        help="signal-to-device aggregation; sparse_extreme retains rare strong cross-band changes",
    )
    parser.add_argument(
        "--feature-set", choices=("all", "l1_only", "l5_only", "no_cross", "causal_delta_only", "causal_delta_with_device", "initial_baseline_delta_only", "initial_baseline_delta_with_device", "initial_baseline_delta_l1_with_device", "initial_baseline_delta_no_cross", "initial_baseline_delta_with_capability", "initial_baseline_delta_no_cross_with_capability"), default="all",
    )
    parser.add_argument(
        "--causal-reference-windows", type=int, default=0,
        help="append within-source differences from preceding device windows",
    )
    parser.add_argument(
        "--initial-baseline-windows", type=int, default=0,
        help="append differences from the initial reviewed-normal device windows in each source stream",
    )
    parser.add_argument(
        "--initial-baseline-policy", choices=("error", "exclude_stream"), default="error",
        help="how to handle streams whose initial baseline is unavailable because an attack starts immediately",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {path}") from exc


def read_response_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    required = {
        "Environment", "Scenario", "Session", "DeviceName", "response_state",
        "start_tow", "end_tow",
    }
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return []
        if missing := required.difference(reader.fieldnames):
            raise ValueError(f"{path} missing response-label columns {sorted(missing)}")
        for line_no, row in enumerate(reader, start=2):
            state = str(row["response_state"]).strip()
            if state not in RESPONSE_STATE_TO_ID:
                raise ValueError(f"Unsupported response_state {state!r} at {path}:{line_no}")
            start = float(row["start_tow"])
            end = float(row["end_tow"])
            if end < start:
                raise ValueError(f"Descending response interval at {path}:{line_no}")
            rows.append({
                "Environment": str(row["Environment"]).strip(),
                "Scenario": str(row["Scenario"]).strip(),
                "Session": str(row["Session"]).strip(),
                "DeviceName": str(row["DeviceName"]).strip(),
                "state_id": RESPONSE_STATE_TO_ID[state],
                "response_state": state,
                "start_tow": start,
                "end_tow": end,
                "line_no": line_no,
            })
    return rows


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
    if kind == "top3_mean":
        return float(np.mean(np.sort(values)[-min(3, len(values)):]))
    if kind == "bottom3_mean":
        return float(np.mean(np.sort(values)[:min(3, len(values))]))
    raise ValueError(f"Unsupported reduction: {kind}")


def aggregate_window(stats: np.ndarray, mask: np.ndarray, index: dict[str, int], profile: str) -> tuple[np.ndarray, list[str]]:
    """Summarize all valid L1/L5 signal endpoints at one device time."""
    if stats.ndim != 3 or stats.shape[1] != 1:
        raise ValueError(f"Expected [signals, 1, features], received {stats.shape}")
    values = stats[:, 0, :]
    active = np.asarray(mask, dtype=bool)
    is_l5 = values[:, index["IsL5"]] >= 0.5
    definitions: list[tuple[str, str, tuple[str, ...]]] = [
        ("Cn0DbHzLastW5", "cn0_last", ("median", "q25", "q75")),
        ("Cn0DbHzSlopeW5", "cn0_slope", ("median", "positive_ratio", "negative_ratio")),
        ("AgcDbLastW5", "agc_last", ("median",)),
        ("AgcDbStdW5", "agc_std", ("median",)),
        ("ReceivedSvTimeUncertaintyNanosStdW5", "rx_time_unc_std", ("median",)),
        ("SignalHistoryRatioW5", "history", ("median", "coverage_loss_ratio")),
        ("AgcObservedRatioW5", "agc_observed", ("median", "coverage_loss_ratio")),
    ]
    if profile == "sparse_extreme":
        definitions[0] = ("Cn0DbHzLastW5", "cn0_last", ("median", "q25", "q75", "top3_mean", "bottom3_mean"))
        definitions[1] = ("Cn0DbHzSlopeW5", "cn0_slope", ("median", "positive_ratio", "negative_ratio", "top3_mean", "bottom3_mean"))
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
    cross_keys = ["cn0_last_median", "cn0_slope_median", "agc_last_median", "history_median"]
    if profile == "sparse_extreme":
        cross_keys.extend(("cn0_last_top3_mean", "cn0_slope_top3_mean", "cn0_slope_bottom3_mean"))
    for key in cross_keys:
        result.append(summaries[5][key] - summaries[1][key])
        names.append(f"l5_minus_l1_{key}")
    result.append(summaries[5]["cn0_slope_positive_ratio"] + summaries[1]["cn0_slope_negative_ratio"])
    names.append("coupled_l5_up_plus_l1_down_ratio")
    if profile == "sparse_extreme":
        result.append(summaries[5]["cn0_slope_top3_mean"] - summaries[1]["cn0_slope_bottom3_mean"])
        names.append("coupled_l5_top3_up_minus_l1_bottom3_down")
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


def response_labels_from_rows(
    recording_ids: np.ndarray,
    source_ids: np.ndarray,
    tows: np.ndarray,
    trace: dict[str, Any],
    response_rows: list[dict[str, Any]],
) -> np.ndarray:
    labels = np.zeros(len(recording_ids), dtype=np.int64)
    if not response_rows:
        return labels
    sources = trace.get("sources")
    recordings = trace.get("recordings")
    if not isinstance(sources, list) or not isinstance(recordings, list):
        raise ValueError("window_trace_index.json must contain source and recording indexes")
    rules: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in response_rows:
        key = (row["Environment"], row["Scenario"], row["Session"], row["DeviceName"])
        rules.setdefault(key, []).append(row)
    for index, (recording_id, source_id, tow) in enumerate(zip(recording_ids, source_ids, tows)):
        source = sources[int(source_id)]
        recording = recordings[int(recording_id)]
        key = (
            str(recording["Environment"]),
            str(recording["Scenario"]),
            str(recording["Session"]),
            str(source["DeviceName"]),
        )
        for rule in rules.get(key, []):
            if rule["start_tow"] <= float(tow) <= rule["end_tow"]:
                labels[index] = max(labels[index], int(rule["state_id"]))
    return labels


def select_feature_indices(names: list[str], feature_set: str) -> list[int]:
    if feature_set == "all":
        selected = names
    elif feature_set == "l1_only":
        selected = [name for name in names if name.startswith("l1_")]
    elif feature_set == "l5_only":
        selected = [name for name in names if name.startswith("l5_")]
    elif feature_set == "no_cross":
        selected = [name for name in names if not name.startswith("l5_minus_") and name != "coupled_l5_up_plus_l1_down_ratio"]
    elif feature_set == "causal_delta_only":
        selected = [name for name in names if name.startswith("causal_delta_")]
    elif feature_set == "initial_baseline_delta_only":
        selected = [name for name in names if name.startswith("initial_baseline_delta_")]
    elif feature_set == "initial_baseline_delta_with_device":
        selected = [name for name in names if name.startswith("initial_baseline_delta_") or name.startswith("device_is_")]
    elif feature_set == "initial_baseline_delta_l1_with_device":
        selected = [name for name in names if name.startswith("initial_baseline_delta_l1_") or name.startswith("device_is_")]
    elif feature_set == "initial_baseline_delta_no_cross":
        selected = [
            name for name in names
            if not name.startswith("initial_baseline_delta_l5_minus_")
            and not name.startswith("initial_baseline_delta_coupled_")
        ]
    elif feature_set == "initial_baseline_delta_with_capability":
        selected = [
            name for name in names
            if name.startswith("initial_baseline_delta_") or name.startswith("capability_")
        ]
    elif feature_set == "initial_baseline_delta_no_cross_with_capability":
        selected = [
            name for name in names
            if (
                name.startswith("capability_")
                or (
                    name.startswith("initial_baseline_delta_")
                    and not name.startswith("initial_baseline_delta_l5_minus_")
                    and not name.startswith("initial_baseline_delta_coupled_")
                )
            )
        ]
    else:
        selected = [name for name in names if name.startswith("causal_delta_") or name.startswith("device_is_")]
    if not selected:
        raise ValueError(f"Feature set {feature_set!r} selected no device features")
    return [names.index(name) for name in selected]


def load_device_split(
    signal_dir: Path,
    split: str,
    index: dict[str, int],
    intervals: dict[int, list[tuple[float, float]]],
    trace: dict[str, Any],
    response_rows: list[dict[str, Any]],
    profile: str,
) -> tuple[dict[str, np.ndarray], list[str]]:
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
            features, feature_names = aggregate_window(row_stats, row_mask, index, profile)
            if names is None:
                names = feature_names
            elif names != feature_names:
                raise RuntimeError("Device feature names changed between windows")
            rows.append(features)
        x = np.stack(rows) if rows else np.empty((0, 0), dtype=np.float32)
        recording_ids = raw["recording_id"].astype(np.int64)
        source_ids = raw["source_id"].astype(np.int64)
        event_y = labels_from_interval(recording_ids, raw["endpoint_tow"], intervals)
        direct_positive = (raw["y"] == 1) & raw["mask"]
        if np.any(direct_positive.any(axis=1) & (event_y == 0)):
            raise ValueError(f"Direct target-band positives fall outside an event interval in {split}")
        response_y = response_labels_from_rows(recording_ids, source_ids, raw["endpoint_tow"], trace, response_rows)
        response_y[direct_positive.any(axis=1)] = RESPONSE_STATE_TO_ID["direct_spoof"]
        return {
            "x": x,
            "y_event": event_y,
            "y_response_state": response_y,
            "device_id": raw["device_id"].astype(np.int64),
            "recording_id": recording_ids,
            "source_id": source_ids,
            "endpoint_tow": raw["endpoint_tow"].astype(np.float64),
            "endpoint_utc_millis": raw["endpoint_utc_millis"].astype(np.float64),
        }, names or []


def append_causal_device_deltas(
    data: dict[str, np.ndarray], names: list[str], history_windows: int,
) -> list[str]:
    """Append source-local differences against preceding device endpoints only."""
    if history_windows == 0:
        return names
    values = data["x"].astype(np.float64, copy=False)
    deltas = np.full_like(values, np.nan, dtype=np.float64)
    recording_ids = data["recording_id"]
    source_ids = data["source_id"]
    endpoint_times = data["endpoint_utc_millis"]
    groups = np.unique(np.column_stack((recording_ids, source_ids)), axis=0)
    for recording_id, source_id in groups:
        group = np.flatnonzero((recording_ids == recording_id) & (source_ids == source_id))
        ordered = group[np.argsort(endpoint_times[group], kind="mergesort")]
        group_values = values[ordered]
        finite = np.isfinite(group_values)
        filled = np.where(finite, group_values, 0.0)
        cumulative_sum = np.vstack((np.zeros((1, values.shape[1])), np.cumsum(filled, axis=0)))
        cumulative_count = np.vstack((np.zeros((1, values.shape[1])), np.cumsum(finite, axis=0)))
        endpoint = np.arange(len(ordered))
        start = np.maximum(0, endpoint - history_windows)
        history_sum = cumulative_sum[endpoint] - cumulative_sum[start]
        history_count = cumulative_count[endpoint] - cumulative_count[start]
        reference = np.divide(history_sum, history_count, out=np.zeros_like(history_sum), where=history_count > 0)
        group_delta = group_values - reference
        group_delta[history_count == 0] = np.nan
        deltas[ordered] = group_delta
    data["x"] = np.concatenate((values, deltas.astype(np.float32)), axis=1).astype(np.float32)
    return [*names, *[f"causal_delta_{name}" for name in names]]


def append_initial_baseline_deltas(
    data: dict[str, np.ndarray], names: list[str], baseline_windows: int,
) -> list[str]:
    """Append source-local differences from a verified normal session prefix.

    A rolling reference eventually adapts to a sustained attack.  This feature
    keeps the reference fixed at the session prefix, which is valid only when
    that prefix is independently reviewed as normal.  The labels are used
    exclusively for this data-contract check, never to calculate the values.
    """
    if baseline_windows == 0:
        return names
    values = data["x"].astype(np.float64, copy=False)
    deltas = np.full_like(values, np.nan, dtype=np.float64)
    recording_ids = data["recording_id"]
    source_ids = data["source_id"]
    endpoint_times = data["endpoint_utc_millis"]
    event_labels = data["y_event"]
    groups = np.unique(np.column_stack((recording_ids, source_ids)), axis=0)
    for recording_id, source_id in groups:
        group = np.flatnonzero((recording_ids == recording_id) & (source_ids == source_id))
        ordered = group[np.argsort(endpoint_times[group], kind="mergesort")]
        prefix = ordered[:baseline_windows]
        if len(prefix) < baseline_windows:
            raise ValueError(
                f"Source {(int(recording_id), int(source_id))} has only {len(prefix)} windows; "
                f"cannot construct a {baseline_windows}-window initial baseline"
            )
        if np.any(event_labels[prefix] != 0):
            raise ValueError(
                f"Initial baseline overlaps a reviewed attack for source {(int(recording_id), int(source_id))}; "
                "choose fewer baseline windows or a different protocol"
            )
        reference_values = values[prefix]
        finite = np.isfinite(reference_values)
        count = finite.sum(axis=0)
        reference = np.divide(
            np.where(finite, reference_values, 0.0).sum(axis=0), count,
            out=np.full(values.shape[1], np.nan, dtype=np.float64), where=count > 0,
        )
        deltas[ordered] = values[ordered] - reference
    data["x"] = np.concatenate((values, deltas.astype(np.float32)), axis=1).astype(np.float32)
    return [*names, *[f"initial_baseline_delta_{name}" for name in names]]


def initial_baseline_eligibility(data: dict[str, np.ndarray], baseline_windows: int) -> tuple[np.ndarray, list[dict[str, int | str]]]:
    """Identify streams that can establish a reviewed-normal initial baseline."""
    keep = np.ones(len(data["x"]), dtype=bool)
    excluded: list[dict[str, int | str]] = []
    recording_ids = data["recording_id"]
    source_ids = data["source_id"]
    endpoint_times = data["endpoint_utc_millis"]
    event_labels = data["y_event"]
    groups = np.unique(np.column_stack((recording_ids, source_ids)), axis=0)
    for recording_id, source_id in groups:
        group = np.flatnonzero((recording_ids == recording_id) & (source_ids == source_id))
        ordered = group[np.argsort(endpoint_times[group], kind="mergesort")]
        prefix = ordered[:baseline_windows]
        reason: str | None = None
        if len(prefix) < baseline_windows:
            reason = "insufficient_windows"
        elif np.any(event_labels[prefix] != 0):
            reason = "attack_in_initial_baseline"
        if reason is not None:
            keep[group] = False
            excluded.append({
                "recording_id": int(recording_id), "source_id": int(source_id), "windows": int(len(group)), "reason": reason,
            })
    return keep, excluded


def filter_rows(data: dict[str, np.ndarray], keep: np.ndarray) -> None:
    """Apply an eligibility mask consistently to every per-window tensor field."""
    if len(keep) != len(data["x"]):
        raise ValueError("Eligibility mask length differs from device windows")
    for key, values in data.items():
        if len(values) != len(keep):
            raise ValueError(f"Field {key!r} has inconsistent device-window length")
        data[key] = values[keep]

def append_device_one_hot(data: dict[str, np.ndarray], device_count: int) -> list[str]:
    """Append a receiver identity indicator after causal feature construction."""
    device_ids = data["device_id"].astype(np.int64)
    if np.any(device_ids < 0) or np.any(device_ids >= device_count):
        raise ValueError("Device identifier is outside the declared mapping")
    one_hot = np.zeros((len(device_ids), device_count), dtype=np.float32)
    one_hot[np.arange(len(device_ids)), device_ids] = 1.0
    data["x"] = np.concatenate((data["x"], one_hot), axis=1).astype(np.float32)
    return [f"device_is_{device_id}" for device_id in range(device_count)]


def append_device_capability_masks(
    datasets: dict[str, dict[str, np.ndarray]], feature_names: list[str]
) -> tuple[list[str], dict[str, dict[str, float]]]:
    """Append train-derived L5/AGC capability masks without device identity."""
    required = ("l5_log_signal_count", "l1_agc_observed_median")
    missing = [name for name in required if name not in feature_names]
    if missing:
        raise ValueError(f"Cannot derive device capability masks; missing {missing}")
    all_ids = [data["device_id"] for data in datasets.values() if len(data["device_id"])]
    train_ids = datasets["train"]["device_id"].astype(np.int64)
    if not all_ids or len(train_ids) == 0:
        raise ValueError("Cannot derive device capability masks without train/device windows")
    device_count = max(int(values.max()) for values in all_ids) + 1
    l5_index = feature_names.index("l5_log_signal_count")
    agc_index = feature_names.index("l1_agc_observed_median")
    capabilities = np.zeros((device_count, 2), dtype=np.float32)
    metadata: dict[str, dict[str, float]] = {}
    train_x = datasets["train"]["x"]
    for device_id in range(device_count):
        rows = train_ids == device_id
        if np.any(rows):
            l5_values = train_x[rows, l5_index]
            agc_values = train_x[rows, agc_index]
            has_l5 = bool(np.any(np.isfinite(l5_values) & (l5_values > 0.0)))
            has_agc = bool(np.any(np.isfinite(agc_values) & (agc_values > 0.0)))
        else:
            has_l5 = False
            has_agc = False
        capabilities[device_id] = (float(has_l5), float(has_agc))
        metadata[str(device_id)] = {"has_l5": float(has_l5), "has_agc": float(has_agc)}
    for data in datasets.values():
        masks = capabilities[data["device_id"].astype(np.int64)]
        data["x"] = np.concatenate((data["x"], masks), axis=1).astype(np.float32)
    return ["capability_has_l5", "capability_has_agc"], metadata

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
    if args.causal_reference_windows < 0 or args.initial_baseline_windows < 0:
        raise ValueError("reference window counts must be non-negative")
    if args.feature_set in ("initial_baseline_delta_only", "initial_baseline_delta_with_device", "initial_baseline_delta_l1_with_device", "initial_baseline_delta_no_cross", "initial_baseline_delta_with_capability", "initial_baseline_delta_no_cross_with_capability") and args.initial_baseline_windows == 0:
        raise ValueError("initial baseline feature sets require --initial-baseline-windows")
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
    response_rows = read_response_rows(args.response_labels)
    datasets: dict[str, dict[str, np.ndarray]] = {}
    feature_names: list[str] | None = None
    for split in SPLITS:
        data, candidate_names = load_device_split(
            args.signal_data_dir, split, index, intervals, trace, response_rows, args.device_aggregate_profile,
        )
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
    causal_names: list[str] | None = None
    for data in datasets.values():
        candidate_names = append_causal_device_deltas(data, feature_names, args.causal_reference_windows)
        if causal_names is None:
            causal_names = candidate_names
        elif causal_names != candidate_names:
            raise RuntimeError("Causal device feature contract differs between splits")
    feature_names = causal_names or feature_names
    initial_baseline_exclusions: dict[str, list[dict[str, int | str]]] = {split: [] for split in SPLITS}
    if args.initial_baseline_windows:
        for split, data in datasets.items():
            keep, excluded = initial_baseline_eligibility(data, args.initial_baseline_windows)
            initial_baseline_exclusions[split] = excluded
            if excluded and args.initial_baseline_policy == "error":
                first = excluded[0]
                raise ValueError(
                    "Initial baseline is unavailable for "
                    f"source ({first['recording_id']}, {first['source_id']}) because {first['reason']}; "
                    "choose --initial-baseline-policy exclude_stream or a different protocol"
                )
            if excluded:
                filter_rows(data, keep)
    baseline_names: list[str] | None = None
    for data in datasets.values():
        candidate_names = append_initial_baseline_deltas(data, feature_names, args.initial_baseline_windows)
        if baseline_names is None:
            baseline_names = candidate_names
        elif baseline_names != candidate_names:
            raise RuntimeError("Initial baseline feature contract differs between splits")
    feature_names = baseline_names or feature_names
    capability_metadata: dict[str, dict[str, float]] = {}
    if args.feature_set == "initial_baseline_delta_with_capability":
        capability_names, capability_metadata = append_device_capability_masks(datasets, feature_names)
        feature_names = [*feature_names, *capability_names]
    if args.feature_set in ("causal_delta_with_device", "initial_baseline_delta_with_device", "initial_baseline_delta_l1_with_device"):
        device_count = max(int(data["device_id"].max()) for data in datasets.values() if len(data["device_id"])) + 1
        device_names: list[str] | None = None
        for data in datasets.values():
            candidate_names = append_device_one_hot(data, device_count)
            if device_names is None:
                device_names = candidate_names
            elif device_names != candidate_names:
                raise RuntimeError("Device one-hot feature contract differs between splits")
        feature_names = [*feature_names, *(device_names or [])]
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
        "response_label_semantics": "0=normal/no observable response, 1=attack-associated anomaly, 2=direct spoof",
        "signal_data_dir": str(args.signal_data_dir),
        "label_config": str(args.label_config),
        "response_labels": str(args.response_labels),
        "feature_set": args.feature_set,
        "device_aggregate_profile": args.device_aggregate_profile,
        "causal_reference_windows": args.causal_reference_windows,
        "initial_baseline_windows": args.initial_baseline_windows,
        "initial_baseline_policy": args.initial_baseline_policy,
        "initial_baseline_exclusions": initial_baseline_exclusions,
        "capability_masks_source": "train_windows_only" if capability_metadata else None,
        "capability_masks": capability_metadata,
        "feature_names": feature_names,
        "device_mapping": read_json(args.signal_data_dir / "device_mapping.json"),
        "recordings": trace["recordings"],
        "response_label_rows": response_rows,
        "splits": {
            split: {
                "windows": int(len(data["x"])),
                "positive": int(data["y_event"].sum()),
                "negative": int(len(data["y_event"]) - data["y_event"].sum()),
                "response_state_counts": {
                    str(state): int((data["y_response_state"] == state).sum())
                    for state in range(3)
                },
            }
            for split, data in datasets.items()
        },
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metadata["splits"], indent=2))


if __name__ == "__main__":
    main()

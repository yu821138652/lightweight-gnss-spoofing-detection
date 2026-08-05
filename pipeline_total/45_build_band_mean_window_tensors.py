"""Build band-mean window tensors for four-way scene classification.

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
      Cn0DbHzL1MinusL5,
      L1Present, L5Present ]                 # 8 means + 1 contrast + 2 flags

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

    output_dir/{train,val,test}.npz   # x=[B, TIME_STEPS, F], y, single_band_mask, ...
    output_dir/feature_names.json
    output_dir/scaler.json
    output_dir/normal_reference.json
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
RATE_RAW_FEATURE = "PseudorangeRateMetersPerSecond"
STATE_RAW_FEATURE = "State"
ADR_STATE_RAW_FEATURE = "AccumulatedDeltaRangeState"
ADR_RAW_FEATURE = "AccumulatedDeltaRangeMeters"
PSEUDORANGE_RAW_FEATURE = "Pseudorange_Calculated"
PAIRED_PRR_CONTINUOUS_NAMES = [
    "PrrPairMedianResidual",
    "PrrPairAbsMedianResidual",
    "PrrPairMadResidual",
]
PAIRED_PRR_AVAILABILITY_NAME = "PrrPairAvailable"
# The continuous band means plus the L1-minus-L5 C/N0 contrast are standardized;
# the 2 presence flags stay in [0, 1].  The contrast makes "L1 rose relative to
# L5" explicit rather than leaving the model to infer it from two levels that a
# per-device scaler shifts independently.
CN0_DIFF_NAME = "Cn0DbHzL1MinusL5"
CAUSAL_RELATIVE_NAMES = [
    "L1_Cn0Relative",
    "L5_Cn0Relative",
    "L1_Cn0AbsRelative",
    "L5_Cn0AbsRelative",
]
# These are calculated after an optional frozen normal reference has been
# subtracted.  They intentionally describe only each physical band's recent
# history; cross-band features remain a separate, later ablation.
CN0_DYNAMICS_NAMES = [
    "L1_Cn0W5SlopeDbHzPerSecond",
    "L1_Cn0W5StdDbHz",
    "L1_Cn0W5ValidCount",
    "L5_Cn0W5SlopeDbHzPerSecond",
    "L5_Cn0W5StdDbHz",
    "L5_Cn0W5ValidCount",
]
CAUSAL_BASELINE_MODES = ("none", "ema", "gated")
# Unlike the online causal baselines, this reference is fitted once from the
# reviewed normal part of the current outer fold's train split and then frozen.
# It implements the explicit known-device/global-fallback experiment.
NORMAL_REFERENCE_MODES = ("none", "train_normal_band_mean")
FEATURE_NAMES = (
    [f"L1_{name}" for name in FEATURES]
    + [f"L5_{name}" for name in FEATURES]
    + [CN0_DIFF_NAME]
    + ["L1Present", "L5Present"]
)
CONTINUOUS_COUNT = len(FEATURES) * len(BANDS) + 1  # 8 band means + 1 C/N0 contrast
FEATURE_COUNT = len(FEATURE_NAMES)  # 11
INCLUDE_PSEUDORANGE_RATE = False
INCLUDE_STATE_ADR = False
INCLUDE_PSEUDORANGE_RESIDUAL = False
INCLUDE_CROSS_BAND = False
INCLUDE_CN0_DYNAMICS = False
INCLUDE_PAIRED_PSEUDORANGE_RATE = False


def feature_names_for_mode(
    include_pseudorange_rate: bool,
    include_state_adr: bool,
    include_pseudorange_residual: bool,
    include_cross_band: bool,
    causal_baseline_mode: str = "none",
    include_cn0_dynamics: bool = False,
    include_paired_pseudorange_rate: bool = False,
) -> list[str]:
    names = [f"L1_{name}" for name in FEATURES] + [f"L5_{name}" for name in FEATURES]
    if include_pseudorange_rate:
        for band_name in ("L1", "L5"):
            names.extend(
                [
                    f"{band_name}_PrrMean",
                    f"{band_name}_PrrSlope",
                    f"{band_name}_PrrMad",
                    f"{band_name}_PrrOutlierRatio",
                ]
            )
    if include_state_adr:
        for band_name in ("L1", "L5"):
            names.extend(
                [
                    f"{band_name}_StateCodeLockRatio",
                    f"{band_name}_StateTowDecodedRatio",
                    f"{band_name}_StateSwitchRatio",
                    f"{band_name}_AdrResetSlipRatio",
                    f"{band_name}_AdrStateSwitchRatio",
                    f"{band_name}_AdrAbsDiffMedian",
                    f"{band_name}_StateAdrMissingRatio",
                ]
            )
    if include_pseudorange_residual:
        for band_name in ("L1", "L5"):
            names.extend(
                [
                    f"{band_name}_PrResidualMad",
                    f"{band_name}_PrResidualP95",
                    f"{band_name}_PrResidualOutlierRatio",
                ]
            )
    if include_cross_band:
        names.extend(
            [
                "Cn0SlopeL1MinusL5",
                "L5UpL1DownRatio",
                "L1UpL5DownRatio",
            ]
        )
    if include_cn0_dynamics:
        names.extend(CN0_DYNAMICS_NAMES)
    if causal_baseline_mode != "none":
        names.extend(CAUSAL_RELATIVE_NAMES)
    names.append(CN0_DIFF_NAME)
    if include_paired_pseudorange_rate:
        names.extend(PAIRED_PRR_CONTINUOUS_NAMES)
        names.append(PAIRED_PRR_AVAILABILITY_NAME)
    names.extend(["L1Present", "L5Present"])
    return names

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


def load_gate_predictions(path: Path = None) -> dict[tuple[int, int, int], float]:
    """Load causal normal probabilities keyed by source/device/receiver epoch."""
    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = {"source_id", "device_id", "window_time_nanos", "prob_normal"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Gate predictions {path} missing columns: {sorted(missing)}")
    for column in ("source_id", "device_id", "window_time_nanos", "prob_normal"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    for column in ("source_id", "device_id", "window_time_nanos"):
        values = frame[column].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or (values != np.trunc(values)).any():
            raise ValueError(f"Gate key column {column!r} in {path} must contain integers")
        frame[column] = values.astype(np.int64)
    key_columns = ["source_id", "device_id", "window_time_nanos"]
    duplicate = frame.duplicated(key_columns, keep=False)
    if duplicate.any():
        sample = frame.loc[duplicate, key_columns].head(5).to_dict("records")
        raise ValueError(f"Duplicate gate prediction keys in {path}: {sample}")
    probabilities = frame["prob_normal"].to_numpy(dtype=np.float64)
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError(f"Gate probabilities in {path} must be finite values in [0, 1]")
    return {
        (int(row.source_id), int(row.device_id), int(row.window_time_nanos)): float(row.prob_normal)
        for row in frame.itertuples(index=False)
    }


def add_causal_cn0_features(
    table: pd.DataFrame,
    epoch_splits: dict[int, tuple[str, str]],
    source_id: int,
    device_id: int,
    mode: str,
    half_life_seconds: float,
    normal_threshold: float,
    gate_predictions: dict[tuple[int, int, int], float],
) -> pd.DataFrame:
    """Append causal L1/L5 deviations from a split-local online baseline.

    The feature at epoch ``t`` always uses the state available before observing
    the gate decision at ``t``.  The state is reset at every split/segment or
    receiver-gap boundary.  In ``gated`` mode, the first ``TIME_STEPS - 1``
    epochs warm up causally before a W5 gate prediction can exist; later missing
    gate predictions freeze the state.
    """
    if mode not in CAUSAL_BASELINE_MODES:
        raise ValueError(f"Unsupported causal baseline mode: {mode!r}")
    if mode == "none":
        return table
    if not np.isfinite(half_life_seconds) or half_life_seconds <= 0:
        raise ValueError("causal baseline half-life must be positive")
    if not 0 <= normal_threshold <= 1:
        raise ValueError("causal normal threshold must be in [0, 1]")
    if mode == "gated" and not gate_predictions:
        raise ValueError("gated causal baseline requires gate predictions")

    result = table.copy()
    times = result.index.to_numpy(dtype=np.int64)
    utc_values = result["utcTimeMillis"].to_numpy(dtype=np.float64)
    canonical = (np.round(utc_values / 1000.0) * 1000.0).astype(np.int64)
    assignments = [epoch_splits.get(int(value)) for value in canonical]
    band_values = {
        "L1": result["L1_Cn0DbHz"].to_numpy(dtype=np.float64),
        "L5": result["L5_Cn0DbHz"].to_numpy(dtype=np.float64),
    }
    relative = {band: np.full(len(result), np.nan, dtype=np.float64) for band in band_values}
    baseline = {band: np.nan for band in band_values}
    previous_observed_time = {band: None for band in band_values}
    previous_stream = None
    stream_age = 0
    stats = {
        "eligible_epochs": 0,
        "warmup_epochs": 0,
        "gate_available_epochs": 0,
        "gate_missing_epochs": 0,
        "update_allowed_epochs": 0,
        "update_frozen_epochs": 0,
        "band_initializations": 0,
        "band_updates": 0,
        "band_freezes": 0,
    }

    for index, (time_nanos, assignment) in enumerate(zip(times, assignments)):
        receiver_gap = index > 0 and int(time_nanos - times[index - 1]) > EPOCH_GAP_NS
        stream = tuple(assignment) if assignment is not None else None
        if receiver_gap or stream != previous_stream:
            baseline = {band: np.nan for band in band_values}
            previous_observed_time = {band: None for band in band_values}
            stream_age = 0
        previous_stream = stream
        if stream is None or stream[0] not in {"train", "val", "test"}:
            continue

        probability = gate_predictions.get((source_id, device_id, int(time_nanos)))
        warmup = stream_age < TIME_STEPS - 1
        stats["eligible_epochs"] += 1
        if mode == "gated":
            if warmup:
                stats["warmup_epochs"] += 1
            elif probability is None:
                stats["gate_missing_epochs"] += 1
            else:
                stats["gate_available_epochs"] += 1
        allow_update = mode == "ema" or warmup or (
            probability is not None and probability >= normal_threshold
        )
        stats["update_allowed_epochs" if allow_update else "update_frozen_epochs"] += 1
        for band, values in band_values.items():
            current = float(values[index])
            if not np.isfinite(current):
                continue
            if not np.isfinite(baseline[band]):
                baseline[band] = current
                relative[band][index] = 0.0
                previous_observed_time[band] = int(time_nanos)
                stats["band_initializations"] += 1
                continue

            relative[band][index] = current - baseline[band]
            previous_time = previous_observed_time[band]
            if allow_update and previous_time is not None:
                delta_seconds = max((int(time_nanos) - previous_time) / 1e9, 0.0)
                alpha = float(np.exp(-np.log(2.0) * delta_seconds / half_life_seconds))
                baseline[band] = alpha * baseline[band] + (1.0 - alpha) * current
                stats["band_updates"] += 1
            elif previous_time is not None:
                stats["band_freezes"] += 1
            previous_observed_time[band] = int(time_nanos)
        stream_age += 1

    result["L1_Cn0Relative"] = relative["L1"]
    result["L5_Cn0Relative"] = relative["L5"]
    result["L1_Cn0AbsRelative"] = np.abs(relative["L1"])
    result["L5_Cn0AbsRelative"] = np.abs(relative["L5"])
    result.attrs["causal_baseline_stats"] = stats
    return result


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


def band_is_normal_at_tow(
    identity: tuple[str, str, str],
    tow: float,
    band: int,
    intervals: dict[tuple[str, str, str], list[tuple[float, float]]],
    scenario_to_class: dict[str, int],
) -> bool:
    """Whether one physical band is a normal-reference sample at an epoch.

    The scene labels are target-band labels.  Outside an attack interval both
    bands are normal.  During an L1 or L5-only interval the non-target band is
    still eligible, while neither band is eligible during an L1+L5 interval.
    """
    if band not in BANDS:
        raise ValueError(f"Unsupported frequency band for normal reference: {band}")
    scene_class = endpoint_class(identity, tow, intervals, scenario_to_class)
    if scene_class == 0:
        return True
    if scene_class == 1:
        return band != 1
    if scene_class == 2:
        return band != 5
    if scene_class == 3:
        return False
    raise ValueError(f"Unsupported scene class {scene_class} for {identity}")


def _reference_summary(values: list[np.ndarray], minimum_epochs: int) -> dict[str, float | int | bool | None]:
    """Serialize one normal C/N0 reference without emitting NaN JSON values."""
    merged = np.concatenate(values) if values else np.empty((0,), dtype=np.float64)
    finite = merged[np.isfinite(merged)]
    count = int(len(finite))
    return {
        "mean": float(finite.mean()) if count else None,
        "count": count,
        "eligible": bool(count >= minimum_epochs),
    }


def fit_normal_band_reference(
    source_tables: list[dict],
    intervals: dict[tuple[str, str, str], list[tuple[float, float]]],
    scenario_to_class: dict[str, int],
    minimum_epochs: int = 1,
) -> dict:
    """Fit fold-local normal L1/L5 means from unique train epochs only.

    ``source_tables`` contains one pre-window, per-device epoch table for each
    source.  This intentionally fits from those raw epoch means rather than
    overlapping W5 tensors, so one epoch contributes exactly once.
    """
    if minimum_epochs < 1:
        raise ValueError("normal reference minimum epochs must be at least 1")
    global_values: dict[int, list[np.ndarray]] = {band: [] for band in BANDS}
    per_device_values: dict[int, dict[int, list[np.ndarray]]] = {}

    for record in source_tables:
        identity = record["identity"]
        device_id = int(record["device_id"])
        table = record["table"]
        epoch_splits = record["epoch_splits"]
        utc_values = table["utcTimeMillis"].to_numpy(dtype=np.float64)
        tow_values = table["TOW"].to_numpy(dtype=np.float64)
        canonical = (np.round(utc_values / 1000.0) * 1000.0).astype(np.int64)
        train_epochs = np.fromiter(
            (
                (assignment := epoch_splits.get(int(epoch))) is not None
                and assignment[0] == "train"
                for epoch in canonical
            ),
            dtype=bool,
            count=len(canonical),
        )
        device_values = per_device_values.setdefault(
            device_id, {band: [] for band in BANDS}
        )
        for band in BANDS:
            band_name = "L1" if band == 1 else "L5"
            values = table[f"{band_name}_Cn0DbHz"].to_numpy(dtype=np.float64)
            target_normal = np.fromiter(
                (
                    band_is_normal_at_tow(
                        identity, float(tow), band, intervals, scenario_to_class
                    )
                    for tow in tow_values
                ),
                dtype=bool,
                count=len(tow_values),
            )
            selected = values[train_epochs & target_normal & np.isfinite(values)]
            if len(selected):
                global_values[band].append(selected)
                device_values[band].append(selected)

    global_reference = {
        str(band): _reference_summary(global_values[band], minimum_epochs)
        for band in BANDS
    }
    if any(not entry["eligible"] for entry in global_reference.values()):
        raise ValueError(
            "No sufficient train-only normal C/N0 reference for all bands: "
            f"{global_reference}"
        )
    return {
        "mode": "train_normal_band_mean",
        "fit_split": "train",
        "fit_unit": "unique_band_mean_epochs",
        "minimum_epochs": int(minimum_epochs),
        "band_normal_semantics": (
            "outside reviewed attack interval, plus the non-target band during "
            "single-band L1/L5 attack intervals"
        ),
        "global": global_reference,
        "per_device": {
            str(device_id): {
                str(band): _reference_summary(values[band], minimum_epochs)
                for band in BANDS
            }
            for device_id, values in sorted(per_device_values.items())
        },
    }


def normal_reference_for_device(reference: dict, device_id: int, band: int) -> tuple[float, str]:
    """Return a device normal mean or the fold-global fallback for one band."""
    device_entry = reference.get("per_device", {}).get(str(int(device_id)), {})
    candidate = device_entry.get(str(band), {})
    if candidate.get("eligible") and candidate.get("mean") is not None:
        return float(candidate["mean"]), "device"
    global_entry = reference.get("global", {}).get(str(band), {})
    if not global_entry.get("eligible") or global_entry.get("mean") is None:
        raise ValueError(f"No eligible global normal reference for band {band}")
    return float(global_entry["mean"]), "global"


def apply_normal_band_reference(
    table: pd.DataFrame, device_id: int, reference: dict
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Subtract the frozen fold-local normal mean from each observed band."""
    result = table.copy()
    applied: dict[str, str] = {}
    for band in BANDS:
        band_name = "L1" if band == 1 else "L5"
        mean, source = normal_reference_for_device(reference, device_id, band)
        result[f"{band_name}_Cn0DbHz"] = result[f"{band_name}_Cn0DbHz"] - mean
        applied[band_name] = source
    result[CN0_DIFF_NAME] = result["L1_Cn0DbHz"] - result["L5_Cn0DbHz"]
    return result, applied


def normal_reference_application_summary(
    source_tables: list[dict], reference: dict
) -> dict[str, dict[str, dict[str, int | list[int]]]]:
    """Describe per-band known-device versus global-fallback use.

    The reference choice is constant for a device/band within one fold, but a
    device can have multiple recording sources.  Reporting both source and
    unique-device counts makes the held-out-device coverage auditable without
    conflating repeated recordings with independent devices.
    """
    summary: dict[str, dict[str, dict[str, int | set[int]]]] = {
        "L1": {
            "device_reference": {"sources": 0, "device_ids": set()},
            "global_fallback": {"sources": 0, "device_ids": set()},
        },
        "L5": {
            "device_reference": {"sources": 0, "device_ids": set()},
            "global_fallback": {"sources": 0, "device_ids": set()},
        },
    }
    for record in source_tables:
        device_id = int(record["device_id"])
        for band in BANDS:
            band_name = "L1" if band == 1 else "L5"
            _, source = normal_reference_for_device(reference, device_id, band)
            bucket = "device_reference" if source == "device" else "global_fallback"
            detail = summary[band_name][bucket]
            detail["sources"] = int(detail["sources"]) + 1
            device_ids = detail["device_ids"]
            assert isinstance(device_ids, set)
            device_ids.add(device_id)

    return {
        band_name: {
            bucket: {
                "sources": int(detail["sources"]),
                "devices": len(detail["device_ids"]),
                "device_ids": sorted(int(value) for value in detail["device_ids"]),
            }
            for bucket, detail in by_source.items()
        }
        for band_name, by_source in summary.items()
    }


def add_cn0_dynamics_features(
    table: pd.DataFrame,
    epoch_splits: dict[int, tuple[str, str]],
) -> pd.DataFrame:
    """Append causal W5 C/N0 slope and spread for each physical band.

    The table is already in receiver-time order.  At epoch ``t`` each feature
    uses only ``t`` and the preceding four *continuous* receiver epochs in the
    same manifest split and segment.  A receiver gap, guard epoch, split, or
    segment boundary starts a fresh run.  Missing band observations are
    excluded from the statistic rather than interpreted as zero; fewer than
    two finite observations leave the statistic undefined (NaN).  The paired
    valid-count feature distinguishes that case from a true zero slope or
    spread, while the raw presence flags preserve the per-epoch missingness
    history for the model.

    Slopes use real receiver timestamps, not row offsets, so a retained
    two-second cadence is measured in dB-Hz per second.
    """
    result = table.copy()
    times = result.index.to_numpy(dtype=np.int64)
    utc_values = result["utcTimeMillis"].to_numpy(dtype=np.float64)
    canonical = (np.round(utc_values / 1000.0) * 1000.0).astype(np.int64)
    assignments = [epoch_splits.get(int(value)) for value in canonical]
    values_by_band = {
        "L1": result["L1_Cn0DbHz"].to_numpy(dtype=np.float64),
        "L5": result["L5_Cn0DbHz"].to_numpy(dtype=np.float64),
    }
    slopes = {band: np.full(len(result), np.nan, dtype=np.float64) for band in values_by_band}
    spreads = {band: np.full(len(result), np.nan, dtype=np.float64) for band in values_by_band}
    valid_counts = {band: np.full(len(result), np.nan, dtype=np.float64) for band in values_by_band}

    run_start = 0
    previous_stream: tuple[str, str] | None = None
    for index, (time_nanos, assignment) in enumerate(zip(times, assignments)):
        stream = (
            (str(assignment[0]), str(assignment[1]))
            if assignment is not None and str(assignment[0]) in {"train", "val", "test"}
            else None
        )
        receiver_gap = index > 0 and int(time_nanos - times[index - 1]) > EPOCH_GAP_NS
        if receiver_gap or stream != previous_stream:
            run_start = index
        previous_stream = stream
        if stream is None:
            continue

        history_start = max(run_start, index - TIME_STEPS + 1)
        history_times = times[history_start:index + 1].astype(np.float64) / 1e9
        for band, values in values_by_band.items():
            history_values = values[history_start:index + 1]
            finite = np.isfinite(history_values)
            finite_count = int(finite.sum())
            valid_counts[band][index] = finite_count
            if not np.isfinite(values[index]) or finite_count < 2:
                continue
            observed_times = history_times[finite]
            observed_values = history_values[finite]
            centered_times = observed_times - observed_times.mean()
            denominator = float(np.dot(centered_times, centered_times))
            if denominator > 0.0:
                centered_values = observed_values - observed_values.mean()
                slopes[band][index] = float(np.dot(centered_times, centered_values) / denominator)
            spreads[band][index] = float(np.std(observed_values))

    result["L1_Cn0W5SlopeDbHzPerSecond"] = slopes["L1"]
    result["L1_Cn0W5StdDbHz"] = spreads["L1"]
    result["L1_Cn0W5ValidCount"] = valid_counts["L1"]
    result["L5_Cn0W5SlopeDbHzPerSecond"] = slopes["L5"]
    result["L5_Cn0W5StdDbHz"] = spreads["L5"]
    result["L5_Cn0W5ValidCount"] = valid_counts["L5"]
    return result


def paired_pseudorange_rate_pairs(source: pd.DataFrame) -> pd.DataFrame:
    """Return same-satellite L1/L5 pseudorange-rate pairs for one source.

    A raw pseudorange rate contains satellite geometry, receiver motion and
    clock effects.  Pairing the two physical bands of the *same* satellite at
    the same receiver epoch removes those common terms before any model sees
    the value.  Multiple code types within one band are collapsed by median.
    """
    columns = [
        "TimeNanos", "utcTimeMillis", "ConstellationType", "Svid", "FreqBand",
        RATE_RAW_FEATURE, "Label",
    ]
    missing = sorted(set(columns).difference(source.columns))
    if missing:
        raise ValueError(f"Paired pseudorange-rate features require columns {missing}")
    values = source[columns].copy()
    for column in columns:
        values[column] = pd.to_numeric(values[column], errors="coerce")
    values = values[
        values["FreqBand"].isin(BANDS)
        & np.isfinite(values[RATE_RAW_FEATURE])
        & values[["TimeNanos", "utcTimeMillis", "ConstellationType", "Svid"]].notna().all(axis=1)
    ].copy()
    if values.empty:
        return pd.DataFrame(
            columns=[
                "TimeNanos", "utcTimeMillis", "ConstellationType", "Svid",
                "PrrPairDifference", "PrrPairL1Label", "PrrPairL5Label",
            ]
        )

    keys = ["TimeNanos", "ConstellationType", "Svid", "FreqBand"]
    collapsed = (
        values.groupby(keys, sort=False)
        .agg(
            utcTimeMillis=("utcTimeMillis", "median"),
            pseudorange_rate=(RATE_RAW_FEATURE, "median"),
            label=("Label", "max"),
        )
        .reset_index()
    )
    base = ["TimeNanos", "ConstellationType", "Svid"]
    rate = collapsed.pivot(index=base, columns="FreqBand", values="pseudorange_rate")
    label = collapsed.pivot(index=base, columns="FreqBand", values="label")
    utc = collapsed.groupby(base, sort=False)["utcTimeMillis"].median()
    empty_rate = pd.Series(np.nan, index=rate.index, dtype=np.float64)
    empty_label = pd.Series(np.nan, index=label.index, dtype=np.float64)
    l1_rate = rate[1] if 1 in rate.columns else empty_rate
    l5_rate = rate[5] if 5 in rate.columns else empty_rate
    l1_label = label[1] if 1 in label.columns else empty_label
    l5_label = label[5] if 5 in label.columns else empty_label
    paired = pd.DataFrame(
        {
            "utcTimeMillis": utc,
            "PrrPairDifference": l1_rate - l5_rate,
            "PrrPairL1Label": l1_label,
            "PrrPairL5Label": l5_label,
        }
    ).reset_index()
    paired = paired.dropna(
        subset=["PrrPairDifference", "PrrPairL1Label", "PrrPairL5Label"]
    )
    return paired.sort_values(base, kind="mergesort").reset_index(drop=True)


def fit_paired_pseudorange_rate_reference(
    pairs: pd.DataFrame,
    minimum_pairs: int,
) -> dict:
    """Fit fold-local train-normal median L1/L5 rate differences.

    The reference is conditioned on device and constellation.  A sparse or
    unseen device/constellation falls back to the train-wide constellation
    median, then to the all-constellation train median.  No validation/test
    label or measurement participates in this fit.
    """
    if minimum_pairs < 1:
        raise ValueError("paired pseudorange-rate minimum pair count must be at least 1")
    required = {
        "device_id", "ConstellationType", "PrrPairDifference", "is_train_normal",
    }
    missing = sorted(required.difference(pairs.columns))
    if missing:
        raise ValueError(f"Paired pseudorange-rate reference input missing {missing}")
    normal = pairs.loc[
        pairs["is_train_normal"].astype(bool)
        & np.isfinite(pd.to_numeric(pairs["PrrPairDifference"], errors="coerce"))
    ].copy()
    if normal.empty:
        raise ValueError("No train-normal paired pseudorange-rate observations")
    normal["device_id"] = pd.to_numeric(normal["device_id"], errors="raise").astype(np.int64)
    normal["ConstellationType"] = pd.to_numeric(
        normal["ConstellationType"], errors="raise"
    ).astype(np.int64)
    normal["PrrPairDifference"] = pd.to_numeric(
        normal["PrrPairDifference"], errors="raise"
    ).astype(np.float64)

    def summary(values: pd.Series) -> dict[str, float | int | bool | None]:
        finite = values.to_numpy(dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        count = int(len(finite))
        return {
            "median": float(np.median(finite)) if count else None,
            "count": count,
            "eligible": bool(count >= minimum_pairs),
        }

    global_all = summary(normal["PrrPairDifference"])
    if not global_all["eligible"]:
        raise ValueError(
            "Insufficient train-normal paired pseudorange-rate observations: "
            f"{global_all}"
        )
    global_by_constellation = {
        str(int(constellation)): summary(group["PrrPairDifference"])
        for constellation, group in normal.groupby("ConstellationType", sort=True)
    }
    per_device: dict[str, dict[str, dict[str, float | int | bool | None]]] = {}
    for device, device_pairs in normal.groupby("device_id", sort=True):
        per_device[str(int(device))] = {
            str(int(constellation)): summary(group["PrrPairDifference"])
            for constellation, group in device_pairs.groupby("ConstellationType", sort=True)
        }
    return {
        "mode": "train_normal_device_constellation_median",
        "fit_split": "train",
        "fit_population": "same_satellite_same_epoch_L1_L5_pairs_with_both_bands_normal",
        "minimum_pairs": int(minimum_pairs),
        "global_all": global_all,
        "global_by_constellation": global_by_constellation,
        "per_device": per_device,
    }


def apply_paired_pseudorange_rate_reference(
    pairs: pd.DataFrame,
    reference: dict,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Subtract frozen paired-rate references and record fallback use."""
    if reference.get("mode") != "train_normal_device_constellation_median":
        raise ValueError(f"Unsupported paired pseudorange-rate reference: {reference.get('mode')!r}")
    result = pairs.copy()
    assignments = {"device_constellation": 0, "global_constellation": 0, "global_all": 0}
    residuals = np.full(len(result), np.nan, dtype=np.float64)
    values = pd.to_numeric(result["PrrPairDifference"], errors="coerce").to_numpy(dtype=np.float64)
    devices = pd.to_numeric(result["device_id"], errors="raise").to_numpy(dtype=np.int64)
    constellations = pd.to_numeric(
        result["ConstellationType"], errors="raise"
    ).to_numpy(dtype=np.int64)
    for index, (value, device, constellation) in enumerate(zip(values, devices, constellations)):
        if not np.isfinite(value):
            continue
        constellation_key = str(int(constellation))
        device_entry = reference.get("per_device", {}).get(str(int(device)), {})
        candidate = device_entry.get(constellation_key)
        assignment = "device_constellation"
        if not candidate or not candidate.get("eligible"):
            candidate = reference.get("global_by_constellation", {}).get(constellation_key)
            assignment = "global_constellation"
        if not candidate or not candidate.get("eligible"):
            candidate = reference.get("global_all")
            assignment = "global_all"
        median = candidate.get("median") if isinstance(candidate, dict) else None
        if median is None or not np.isfinite(float(median)):
            raise ValueError(
                "Paired pseudorange-rate reference has no finite fallback for "
                f"device={device}, constellation={constellation}"
            )
        residuals[index] = value - float(median)
        assignments[assignment] += 1
    result["PrrPairResidual"] = residuals
    return result, assignments


def aggregate_paired_pseudorange_rate_epochs(pairs: pd.DataFrame) -> pd.DataFrame:
    """Robustly summarize paired-rate residuals at each receiver epoch."""
    if pairs.empty:
        return pd.DataFrame(
            columns=[*PAIRED_PRR_CONTINUOUS_NAMES, PAIRED_PRR_AVAILABILITY_NAME]
        )
    required = {"TimeNanos", "PrrPairResidual"}
    missing = sorted(required.difference(pairs.columns))
    if missing:
        raise ValueError(f"Paired pseudorange-rate aggregation missing {missing}")
    grouped = pairs.groupby("TimeNanos", sort=True)["PrrPairResidual"]

    def mad(values: pd.Series) -> float:
        data = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
        data = data[np.isfinite(data)]
        if not len(data):
            return np.nan
        median = float(np.median(data))
        return float(np.median(np.abs(data - median)))

    median = grouped.median()
    abs_median = grouped.apply(
        lambda values: float(np.median(np.abs(pd.to_numeric(values, errors="coerce").dropna())))
        if pd.to_numeric(values, errors="coerce").notna().any() else np.nan
    )
    result = pd.DataFrame(
        {
            PAIRED_PRR_CONTINUOUS_NAMES[0]: median,
            PAIRED_PRR_CONTINUOUS_NAMES[1]: abs_median,
            PAIRED_PRR_CONTINUOUS_NAMES[2]: grouped.apply(mad),
            PAIRED_PRR_AVAILABILITY_NAME: 1.0,
        }
    )
    return result


def band_epoch_table(
    source: pd.DataFrame,
    paired_prr_epochs: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Collapse one device's rows into one band-mean row per receiver epoch.

    Returns a frame indexed by TimeNanos with the 8 continuous band means (NaN
    when a band is absent at that epoch), the two presence flags, and endpoint
    UTC/TOW representatives.  This is the epoch-level analogue of the per-TOW
    band mean drawn by the dashboard.
    """
    source = source.copy()
    if INCLUDE_PSEUDORANGE_RESIDUAL:
        required = {PSEUDORANGE_RAW_FEATURE, "ConstellationType"}
        missing = sorted(required.difference(source.columns))
        if missing:
            raise ValueError(
                f"Pseudorange residual feature set requires columns {missing}; "
                "rebuild the processed CSV with --include-dynamic-raw-features"
            )
        pseudorange = pd.to_numeric(source[PSEUDORANGE_RAW_FEATURE], errors="coerce")
        constellation = pd.to_numeric(source["ConstellationType"], errors="coerce")
        group_median = pseudorange.groupby(
            [source["TimeNanos"], constellation], sort=False
        ).transform("median")
        source["_PrResidualAbs"] = (pseudorange - group_median).abs()
    if INCLUDE_STATE_ADR:
        required = {
            "signal_id", STATE_RAW_FEATURE, ADR_STATE_RAW_FEATURE, ADR_RAW_FEATURE,
        }
        missing = sorted(required.difference(source.columns))
        if missing:
            raise ValueError(
                f"State/ADR feature set requires columns {missing}; "
                "rebuild the processed CSV with --include-dynamic-raw-features"
            )
        source = source.sort_values(["signal_id", "TimeNanos"], kind="mergesort")
        state_group = source.groupby("signal_id", sort=False)[STATE_RAW_FEATURE]
        adr_state_group = source.groupby("signal_id", sort=False)[ADR_STATE_RAW_FEATURE]
        adr_group = source.groupby("signal_id", sort=False)[ADR_RAW_FEATURE]
        source["_StateChanged"] = state_group.diff().abs().gt(0).astype(np.float32)
        source["_AdrStateChanged"] = adr_state_group.diff().abs().gt(0).astype(np.float32)
        source["_AdrAbsDiff"] = adr_group.diff().abs()

    band = pd.to_numeric(source["FreqBand"], errors="coerce")
    columns: dict[str, pd.Series] = {}
    present: dict[int, pd.Series] = {}
    for band_value in BANDS:
        sub = source[band.eq(band_value)]
        grouped = sub.groupby("TimeNanos", sort=True)
        for feature in FEATURES:
            mean = grouped[feature].mean() if len(sub) else pd.Series(dtype=float)
            columns[f"{'L1' if band_value == 1 else 'L5'}_{feature}"] = mean
        if INCLUDE_PSEUDORANGE_RATE:
            if RATE_RAW_FEATURE not in sub.columns:
                raise ValueError(
                    f"{RATE_RAW_FEATURE} is required for the pseudorange-rate feature set "
                    "but is absent from the processed CSV"
                )
            band_name = "L1" if band_value == 1 else "L5"
            rate_grouped = grouped[RATE_RAW_FEATURE]
            rate_mean = rate_grouped.mean()
            rate_median = rate_grouped.median()

            def mad(values: pd.Series) -> float:
                median = float(values.median())
                return float((values - median).abs().median())

            rate_mad = rate_grouped.apply(mad)

            def outlier_ratio(values: pd.Series) -> float:
                median = float(values.median())
                spread = float((values - median).abs().median())
                if not np.isfinite(spread) or spread < 1e-6:
                    return 0.0
                return float(((values - median).abs() > 3.0 * spread).mean())

            rate_outlier = rate_grouped.apply(outlier_ratio)
            columns[f"{band_name}_PrrMean"] = rate_mean
            columns[f"{band_name}_PrrSlope"] = rate_mean.diff().fillna(0.0)
            columns[f"{band_name}_PrrMad"] = rate_mad
            columns[f"{band_name}_PrrOutlierRatio"] = rate_outlier
        if INCLUDE_STATE_ADR:
            band_name = "L1" if band_value == 1 else "L5"
            if len(sub):
                state_grouped = sub.groupby("TimeNanos", sort=True)
                state = pd.to_numeric(sub[STATE_RAW_FEATURE], errors="coerce")
                adr_state = pd.to_numeric(sub[ADR_STATE_RAW_FEATURE], errors="coerce")
                adr_missing = state.isna() | adr_state.isna() | sub[ADR_RAW_FEATURE].isna()
                state_values = state_grouped[STATE_RAW_FEATURE]
                adr_state_values = state_grouped[ADR_STATE_RAW_FEATURE]
                columns[f"{band_name}_StateCodeLockRatio"] = state_values.apply(
                    lambda values: float(((pd.to_numeric(values, errors="coerce").fillna(0).astype(np.int64) & 1) != 0).mean())
                )
                columns[f"{band_name}_StateTowDecodedRatio"] = state_values.apply(
                    lambda values: float(((pd.to_numeric(values, errors="coerce").fillna(0).astype(np.int64) & 8) != 0).mean())
                )
                columns[f"{band_name}_StateSwitchRatio"] = state_grouped["_StateChanged"].mean()
                columns[f"{band_name}_AdrResetSlipRatio"] = adr_state_values.apply(
                    lambda values: float(((pd.to_numeric(values, errors="coerce").fillna(0).astype(np.int64) & 6) != 0).mean())
                )
                columns[f"{band_name}_AdrStateSwitchRatio"] = state_grouped["_AdrStateChanged"].mean()
                columns[f"{band_name}_AdrAbsDiffMedian"] = state_grouped["_AdrAbsDiff"].median()
                columns[f"{band_name}_StateAdrMissingRatio"] = state_grouped.apply(
                    lambda frame: float(adr_missing.loc[frame.index].mean())
                )
            else:
                for suffix in (
                    "StateCodeLockRatio", "StateTowDecodedRatio", "StateSwitchRatio",
                    "AdrResetSlipRatio", "AdrStateSwitchRatio", "AdrAbsDiffMedian",
                    "StateAdrMissingRatio",
                ):
                    columns[f"{band_name}_{suffix}"] = pd.Series(dtype=float)
        if INCLUDE_PSEUDORANGE_RESIDUAL:
            band_name = "L1" if band_value == 1 else "L5"
            if len(sub):
                residual_grouped = sub.groupby("TimeNanos", sort=True)["_PrResidualAbs"]

                def residual_mad(values: pd.Series) -> float:
                    median = float(values.median())
                    return float((values - median).abs().median())

                def residual_outlier_ratio(values: pd.Series) -> float:
                    median = float(values.median())
                    spread = float((values - median).abs().median())
                    if not np.isfinite(spread) or spread < 1e-6:
                        return 0.0
                    return float(((values - median).abs() > 3.0 * spread).mean())

                columns[f"{band_name}_PrResidualMad"] = residual_grouped.apply(residual_mad)
                columns[f"{band_name}_PrResidualP95"] = residual_grouped.quantile(0.95)
                columns[f"{band_name}_PrResidualOutlierRatio"] = residual_grouped.apply(
                    residual_outlier_ratio
                )
            else:
                for suffix in ("PrResidualMad", "PrResidualP95", "PrResidualOutlierRatio"):
                    columns[f"{band_name}_{suffix}"] = pd.Series(dtype=float)
    if INCLUDE_CROSS_BAND:
        l1_mean = columns.get("L1_Cn0DbHz", pd.Series(dtype=float))
        l5_mean = columns.get("L5_Cn0DbHz", pd.Series(dtype=float))
        l1_slope = l1_mean.diff().fillna(0.0)
        l5_slope = l5_mean.diff().fillna(0.0)
        columns["Cn0SlopeL1MinusL5"] = l1_slope - l5_slope
        columns["L5UpL1DownRatio"] = ((l5_slope > 0.0) & (l1_slope < 0.0)).astype(np.float32)
        columns["L1UpL5DownRatio"] = ((l1_slope > 0.0) & (l5_slope < 0.0)).astype(np.float32)


    # Presence is recorded for every band, including when the optional
    # cross-band feature block is disabled.
    for band_value in BANDS:
        sub = source[band.eq(band_value)]
        present[band_value] = sub.groupby("TimeNanos", sort=True).size() if len(sub) else pd.Series(dtype=float)
    table = pd.DataFrame(columns)
    # L1-minus-L5 C/N0 contrast: only defined where both bands are observed;
    # NaN elsewhere so it becomes the neutral post-scaling 0 like any missing
    # band mean.  This is the epoch-level analogue of the gap between the blue
    # and red C/N0 curves in the dashboards.
    table[CN0_DIFF_NAME] = table["L1_Cn0DbHz"] - table["L5_Cn0DbHz"]
    # Presence: a band is observed at an epoch iff it contributed >=1 signal row.
    table["L1Present"] = present[1].reindex(table.index).fillna(0).gt(0).astype(np.float32)
    table["L5Present"] = present[5].reindex(table.index).fillna(0).gt(0).astype(np.float32)
    if INCLUDE_PAIRED_PSEUDORANGE_RATE:
        # A source can be single-frequency even though the fold as a whole has
        # paired observations.  Preserve those epochs with an explicit zero
        # availability flag instead of treating this normal data condition as
        # a construction error.
        paired = (
            paired_prr_epochs.reindex(table.index)
            if paired_prr_epochs is not None
            else pd.DataFrame(index=table.index)
        )
        for name in PAIRED_PRR_CONTINUOUS_NAMES:
            values = (
                paired[name]
                if name in paired.columns
                else pd.Series(np.nan, index=table.index, dtype=np.float64)
            )
            table[name] = pd.to_numeric(values, errors="coerce").reindex(table.index)
        availability = (
            paired[PAIRED_PRR_AVAILABILITY_NAME]
            if PAIRED_PRR_AVAILABILITY_NAME in paired.columns
            else pd.Series(0.0, index=table.index, dtype=np.float64)
        )
        table[PAIRED_PRR_AVAILABILITY_NAME] = pd.to_numeric(
            availability, errors="coerce"
        ).reindex(table.index).fillna(0.0).astype(np.float32)
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
    source_id: int,
    epoch_splits: dict[int, tuple[str, str]],
    intervals: dict[tuple[str, str, str], list[tuple[float, float]]],
    scenario_to_class: dict[str, int],
    is_dynamic: bool,
    causal_baseline_mode: str = "none",
    causal_half_life_seconds: float = 60.0,
    causal_normal_threshold: float = 0.8,
    gate_predictions: dict[tuple[int, int, int], float] = None,
    causal_stats: dict[str, int] = None,
    include_cn0_dynamics: bool = False,
) -> dict[str, list]:
    """Emit all eligible band-mean windows for one device."""
    parts: dict[str, list] = {
        split: {
            "x": [], "y": [], "single_band": [], "device": [], "recording": [], "source": [],
            "stream_key": [], "dynamic": [], "window_time_nanos": [],
            "endpoint_utc_millis": [], "endpoint_tow": [],
        }
        for split in ("train", "val", "test")
    }
    gate_predictions = gate_predictions or {}
    if include_cn0_dynamics:
        table = add_cn0_dynamics_features(table, epoch_splits)
    table = add_causal_cn0_features(
        table,
        epoch_splits,
        source_id,
        device_id,
        causal_baseline_mode,
        causal_half_life_seconds,
        causal_normal_threshold,
        gate_predictions,
    )
    if causal_stats is not None:
        for key, value in table.attrs.get("causal_baseline_stats", {}).items():
            causal_stats[key] = causal_stats.get(key, 0) + int(value)
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
        segment = next(iter(segments))
        if split not in parts:  # drops 'guard' and any unexpected split
            continue
        x_window = feature_matrix[window_slice].astype(np.float32)  # [T, F]
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
        out["source"].append(np.int32(source_id))
        # Online state must not cross a guard or time-block boundary.  Source
        # IDs alone are insufficient because one source can contribute several
        # disjoint manifest segments to the same split.
        out["stream_key"].append(f"{source_id}:{split}:{segment}")
        out["dynamic"].append(np.bool_(is_dynamic))
        out["window_time_nanos"].append(np.int64(window_times[-1]))
        out["endpoint_utc_millis"].append(np.float64(utc_values[end_i]))
        out["endpoint_tow"].append(np.float64(endpoint_tow))
    return parts


def stack_split(parts: list[dict[str, list]]) -> dict[str, np.ndarray]:
    keys = [
        "x", "y", "single_band", "device", "recording", "source",
        "stream_key", "dynamic", "window_time_nanos", "endpoint_utc_millis", "endpoint_tow",
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
            "source": np.empty((0,), np.int32),
            "stream_key": np.empty((0,), dtype="<U1"),
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
        "source": np.asarray(collected["source"], dtype=np.int32),
        "stream_key": np.asarray(collected["stream_key"], dtype=str),
        "dynamic": np.asarray(collected["dynamic"], dtype=np.bool_),
        "window_time_nanos": np.asarray(collected["window_time_nanos"], dtype=np.int64),
        "endpoint_utc_millis": np.asarray(collected["endpoint_utc_millis"], dtype=np.float64),
        "endpoint_tow": np.asarray(collected["endpoint_tow"], dtype=np.float64),
    }


def fit_apply_scaler(
    datasets: dict[str, dict[str, np.ndarray]],
    output_dir: Path,
    scaler_mode: str = "per_device",
    fit_unit: str = "window_timesteps",
) -> None:
    """Apply train-only standardization to the continuous band features.

    Missing-band means are NaN; only finite entries contribute to the fit and,
    after standardization, missing entries are set to the (post-scaling) mean of
    0.  Presence flags remain physical [0, 1].  ``per_device`` uses one scaler
    per training device (falling back to the global train scaler for an unseen
    device); ``global`` applies the same all-device train scaler everywhere.
    """
    if scaler_mode not in {"per_device", "global"}:
        raise ValueError(f"Unsupported scaler mode: {scaler_mode!r}")
    if fit_unit not in {"window_timesteps", "unique_window_endpoints"}:
        raise ValueError(f"Unsupported scaler fit unit: {fit_unit!r}")
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

    def fit_values(data: dict[str, np.ndarray], selector: np.ndarray | None = None) -> np.ndarray:
        indices = (
            np.arange(len(data["x"]), dtype=np.int64)
            if selector is None
            else np.flatnonzero(selector)
        )
        if fit_unit == "window_timesteps":
            return data["x"][indices, :, :CONTINUOUS_COUNT].reshape(-1, CONTINUOUS_COUNT)

        # A causal epoch can occur in several overlapping windows.  Fit from
        # one endpoint per source/device/time key so overlap does not reweight
        # the residual distribution.
        keys = np.rec.fromarrays(
            [
                data["source"][indices],
                data["device"][indices],
                data["window_time_nanos"][indices],
            ],
            names=("source", "device", "time"),
        )
        _, first = np.unique(keys, return_index=True)
        unique_indices = indices[np.sort(first)]
        return data["x"][unique_indices, -1, :CONTINUOUS_COUNT]

    train_cont = fit_values(train)
    global_mean, global_std = fit(train_cont)
    per_device: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    if scaler_mode == "per_device":
        for device in np.unique(train["device"]):
            sel = train["device"] == device
            values = fit_values(train, sel)
            per_device[int(device)] = fit(values)

    for data in datasets.values():
        if len(data["x"]) == 0:
            continue
        if scaler_mode == "global":
            block = data["x"][:, :, :CONTINUOUS_COUNT].astype(np.float64)
            standardized = (
                block - global_mean.reshape(1, 1, -1)
            ) / global_std.reshape(1, 1, -1)
            data["x"][:, :, :CONTINUOUS_COUNT] = np.where(
                np.isfinite(standardized), standardized, 0.0
            ).astype(np.float32)
        else:
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
                "mode": scaler_mode,
                "fit_split": "train",
                "fit_unit": fit_unit,
                "fit_population": (
                    "train_unique_window_endpoints_all_devices"
                    if fit_unit == "unique_window_endpoints"
                    else "train_window_timesteps_all_devices"
                ),
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
    scope: str = "static",
    scaler_mode: str = "per_device",
    include_pseudorange_rate: bool = False,
    include_state_adr: bool = False,
    include_pseudorange_residual: bool = False,
    include_cross_band: bool = False,
    include_cn0_dynamics: bool = False,
    include_paired_pseudorange_rate: bool = False,
    paired_pseudorange_rate_reference_min_pairs: int = 256,
    causal_baseline_mode: str = "none",
    causal_half_life_seconds: float = 60.0,
    causal_normal_threshold: float = 0.8,
    gate_predictions_path: Path = None,
    normal_reference_mode: str = "none",
    normal_reference_minimum_epochs: int = 1,
) -> dict:
    global FEATURE_NAMES, CONTINUOUS_COUNT, FEATURE_COUNT
    global INCLUDE_PSEUDORANGE_RATE, INCLUDE_STATE_ADR
    global INCLUDE_PSEUDORANGE_RESIDUAL, INCLUDE_CROSS_BAND, INCLUDE_CN0_DYNAMICS
    global INCLUDE_PAIRED_PSEUDORANGE_RATE
    INCLUDE_PSEUDORANGE_RATE = bool(include_pseudorange_rate)
    INCLUDE_STATE_ADR = bool(include_state_adr)
    INCLUDE_PSEUDORANGE_RESIDUAL = bool(include_pseudorange_residual)
    INCLUDE_CROSS_BAND = bool(include_cross_band)
    INCLUDE_CN0_DYNAMICS = bool(include_cn0_dynamics)
    INCLUDE_PAIRED_PSEUDORANGE_RATE = bool(include_paired_pseudorange_rate)
    if causal_baseline_mode not in CAUSAL_BASELINE_MODES:
        raise ValueError(f"Unsupported causal baseline mode: {causal_baseline_mode!r}")
    if normal_reference_mode not in NORMAL_REFERENCE_MODES:
        raise ValueError(f"Unsupported normal reference mode: {normal_reference_mode!r}")
    if causal_baseline_mode == "gated" and gate_predictions_path is None:
        raise ValueError("causal baseline mode 'gated' requires a gate prediction CSV")
    if causal_baseline_mode != "none" and scaler_mode != "global":
        raise ValueError("causal baseline modes require --scaler-mode global")
    if normal_reference_mode != "none" and causal_baseline_mode != "none":
        raise ValueError(
            "normal reference and causal baseline modes are mutually exclusive"
        )
    if normal_reference_mode != "none" and scaler_mode != "global":
        raise ValueError("normal reference mode requires --scaler-mode global")
    if normal_reference_minimum_epochs < 1:
        raise ValueError("normal reference minimum epochs must be at least 1")
    if paired_pseudorange_rate_reference_min_pairs < 1:
        raise ValueError("paired pseudorange-rate minimum pair count must be at least 1")
    # Keep this as a list: pandas accepts a list of columns, while a tuple is
    # interpreted as one composite column key by ``table[FEATURE_NAMES]``.
    FEATURE_NAMES = feature_names_for_mode(
        INCLUDE_PSEUDORANGE_RATE, INCLUDE_STATE_ADR, INCLUDE_PSEUDORANGE_RESIDUAL,
        INCLUDE_CROSS_BAND, causal_baseline_mode, INCLUDE_CN0_DYNAMICS,
        INCLUDE_PAIRED_PSEUDORANGE_RATE,
    )
    CONTINUOUS_COUNT = len(FEATURE_NAMES) - (3 if INCLUDE_PAIRED_PSEUDORANGE_RATE else 2)
    FEATURE_COUNT = len(FEATURE_NAMES)
    output_dir.mkdir(parents=True, exist_ok=True)
    intervals, scenario_to_class = load_scene_labels(config_path)
    epoch_lookup = load_epoch_manifest(manifest_path)
    gate_predictions = load_gate_predictions(gate_predictions_path)

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
    if INCLUDE_PSEUDORANGE_RATE:
        usecols.append(RATE_RAW_FEATURE)
    if INCLUDE_PAIRED_PSEUDORANGE_RATE:
        usecols.extend([RATE_RAW_FEATURE, "ConstellationType", "Svid"])
    if INCLUDE_STATE_ADR:
        usecols.extend([
            "signal_id", STATE_RAW_FEATURE, ADR_STATE_RAW_FEATURE, ADR_RAW_FEATURE,
        ])
    if INCLUDE_PSEUDORANGE_RESIDUAL:
        usecols.extend(["ConstellationType", PSEUDORANGE_RAW_FEATURE])
    LOG.info("Reading %s", csv_path)
    df = pd.read_csv(csv_path, usecols=lambda c: c in set(usecols))
    missing = set(usecols).difference(df.columns)
    if missing:
        raise ValueError(f"Processed CSV missing columns: {sorted(missing)}")
    df = _norm_keys(df)
    scenario = df["Scenario"].astype(str)
    reviewed = df["LabelStatus"].astype(str) == "reviewed"
    if scope == "static":
        in_scope = scenario.str.startswith("st_")
    elif scope == "dynamic":
        in_scope = scenario.str.startswith("dy_")
    elif scope == "all":
        in_scope = scenario.str.startswith("st_") | scenario.str.startswith("dy_")
    else:
        raise ValueError(f"Unsupported scope: {scope!r}")
    df = df[reviewed & in_scope].copy()
    df["_identity"] = list(zip(df["Environment"], df["Scenario"], df["Session"]))
    df = df[df["_identity"].isin(kept_recordings)].copy()
    if df.empty:
        raise ValueError(f"No reviewed rows in scope={scope!r} match the outer manifest")
    numeric_columns = [*FEATURES, "TimeNanos", "TOW", "utcTimeMillis", "FreqBand"]
    if INCLUDE_PSEUDORANGE_RATE:
        numeric_columns.append(RATE_RAW_FEATURE)
    if INCLUDE_PAIRED_PSEUDORANGE_RATE:
        numeric_columns.extend([RATE_RAW_FEATURE, "ConstellationType", "Svid"])
    if INCLUDE_STATE_ADR:
        numeric_columns.extend([STATE_RAW_FEATURE, ADR_STATE_RAW_FEATURE, ADR_RAW_FEATURE])
    if INCLUDE_PSEUDORANGE_RESIDUAL:
        numeric_columns.extend(["ConstellationType", PSEUDORANGE_RAW_FEATURE])
    for column in numeric_columns:
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
    source_rows = (
        df[[*KEYS, "DeviceName", SOURCE_COL]]
        .drop_duplicates()
        .sort_values([*KEYS, "DeviceName", SOURCE_COL], kind="mergesort")
        .reset_index(drop=True)
    )
    source_to_id = {
        tuple(row): i
        for i, row in enumerate(source_rows.itertuples(index=False, name=None))
    }

    paired_prr_reference: dict = {"mode": "none"}
    paired_prr_application: dict[str, int] = {}
    paired_prr_epochs_by_source: dict[int, pd.DataFrame] = {}
    if INCLUDE_PAIRED_PSEUDORANGE_RATE:
        pair_records: list[pd.DataFrame] = []
        for source_key, source in tqdm(
            df.groupby([*KEYS, "DeviceName", SOURCE_COL], sort=True),
            desc="Preparing paired pseudorange-rate references",
        ):
            identity = tuple(str(source_key[i]) for i in range(3))
            epoch_splits = epoch_lookup.get(identity, {})
            if not epoch_splits:
                continue
            pairs = paired_pseudorange_rate_pairs(source)
            if pairs.empty:
                continue
            device_name = str(source_key[3])
            source_id = source_to_id[tuple(str(value) for value in source_key)]
            canonical = np.round(
                pd.to_numeric(pairs["utcTimeMillis"], errors="coerce").to_numpy(
                    dtype=np.float64
                ) / 1000.0
            ).astype(np.int64) * 1000
            is_train = np.fromiter(
                (
                    (assignment := epoch_splits.get(int(epoch))) is not None
                    and assignment[0] == "train"
                    for epoch in canonical
                ),
                dtype=bool,
                count=len(canonical),
            )
            l1_normal = pd.to_numeric(pairs["PrrPairL1Label"], errors="coerce").eq(0)
            l5_normal = pd.to_numeric(pairs["PrrPairL5Label"], errors="coerce").eq(0)
            pairs["source_id"] = int(source_id)
            pairs["device_id"] = int(device_to_id[device_name])
            pairs["is_train_normal"] = is_train & l1_normal.to_numpy() & l5_normal.to_numpy()
            pair_records.append(pairs)
        if not pair_records:
            raise ValueError("No same-satellite L1/L5 pseudorange-rate pairs in scope")
        paired_prr_pairs = pd.concat(pair_records, ignore_index=True)
        paired_prr_reference = fit_paired_pseudorange_rate_reference(
            paired_prr_pairs, paired_pseudorange_rate_reference_min_pairs
        )
        paired_prr_pairs, paired_prr_application = apply_paired_pseudorange_rate_reference(
            paired_prr_pairs, paired_prr_reference
        )
        for source_id, source_pairs in paired_prr_pairs.groupby("source_id", sort=False):
            paired_prr_epochs_by_source[int(source_id)] = (
                aggregate_paired_pseudorange_rate_epochs(source_pairs)
            )

    source_tables: list[dict] = []
    for source_key, source in tqdm(
        df.groupby([*KEYS, "DeviceName", SOURCE_COL], sort=True),
        desc="Preparing band-mean epochs",
    ):
        identity = tuple(str(source_key[i]) for i in range(3))
        device_name = str(source_key[3])
        epoch_splits = epoch_lookup.get(identity, {})
        if not epoch_splits:
            continue
        source_tables.append(
            {
                "identity": identity,
                "device_id": device_to_id[device_name],
                "recording_id": recording_to_id[identity],
                "source_id": source_to_id[tuple(str(value) for value in source_key)],
                "epoch_splits": epoch_splits,
                "is_dynamic": identity[1].startswith("dy_"),
                "table": band_epoch_table(
                    source,
                    paired_prr_epochs_by_source.get(
                        source_to_id[tuple(str(value) for value in source_key)]
                    ),
                ),
            }
        )

    normal_reference: dict = {"mode": "none"}
    normal_reference_application: dict = {}
    if normal_reference_mode == "train_normal_band_mean":
        normal_reference = fit_normal_band_reference(
            source_tables,
            intervals,
            scenario_to_class,
            minimum_epochs=normal_reference_minimum_epochs,
        )
        normal_reference_application = normal_reference_application_summary(
            source_tables, normal_reference
        )
        for record in source_tables:
            transformed, _ = apply_normal_band_reference(
                record["table"], int(record["device_id"]), normal_reference
            )
            record["table"] = transformed
    (output_dir / "normal_reference.json").write_text(
        json.dumps(normal_reference, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if INCLUDE_PAIRED_PSEUDORANGE_RATE:
        (output_dir / "paired_pseudorange_rate_reference.json").write_text(
            json.dumps(paired_prr_reference, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    chunks: dict[str, list[dict[str, list]]] = {"train": [], "val": [], "test": []}
    causal_stats: dict[str, int] = {}
    for record in source_tables:
        part = build_windows(
            record["table"],
            record["identity"],
            int(record["device_id"]),
            int(record["recording_id"]),
            int(record["source_id"]),
            record["epoch_splits"],
            intervals,
            scenario_to_class,
            is_dynamic=bool(record["is_dynamic"]),
            causal_baseline_mode=causal_baseline_mode,
            causal_half_life_seconds=causal_half_life_seconds,
            causal_normal_threshold=causal_normal_threshold,
            gate_predictions=gate_predictions,
            causal_stats=causal_stats,
            include_cn0_dynamics=INCLUDE_CN0_DYNAMICS,
        )
        for split, values in part.items():
            chunks[split].append(values)

    datasets = {split: stack_split(chunks[split]) for split in ("train", "val", "test")}
    scaler_fit_unit = (
        "unique_window_endpoints"
        if causal_baseline_mode != "none"
        else "window_timesteps"
    )
    fit_apply_scaler(
        datasets,
        output_dir,
        scaler_mode=scaler_mode,
        fit_unit=scaler_fit_unit,
    )

    for split, data in datasets.items():
        np.savez_compressed(
            output_dir / f"{split}.npz",
            x=data["x"],
            y=data["y"],
            single_band_mask=data["single_band"],
            device_id=data["device"],
            recording_id=data["recording"],
            source_id=data["source"],
            stream_key=data["stream_key"],
            is_dynamic=data["dynamic"],
            window_time_nanos=data["window_time_nanos"],
            endpoint_utc_millis=data["endpoint_utc_millis"],
            endpoint_tow=data["endpoint_tow"],
        )

    (output_dir / "feature_names.json").write_text(json.dumps(FEATURE_NAMES, indent=2), encoding="utf-8")
    (output_dir / "device_mapping.json").write_text(
        json.dumps(device_to_id, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    source_records = [
        {
            "source_id": int(source_to_id[tuple(str(value) for value in row)]),
            "Environment": str(row[0]),
            "Scenario": str(row[1]),
            "Session": str(row[2]),
            "DeviceName": str(row[3]),
            SOURCE_COL: str(row[4]),
        }
        for row in source_rows.itertuples(index=False, name=None)
    ]
    (output_dir / "source_mapping.json").write_text(
        json.dumps(source_records, indent=2, ensure_ascii=False), encoding="utf-8"
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
        "feature_set": (
            "train_normal_band_mean_cn0_relative_plus_paired_pseudorange_rate"
            if normal_reference_mode != "none" and INCLUDE_PAIRED_PSEUDORANGE_RATE
            else "paired_pseudorange_rate"
            if INCLUDE_PAIRED_PSEUDORANGE_RATE
            else
            "train_normal_band_mean_cn0_relative_with_w5_dynamics"
            if normal_reference_mode != "none" and INCLUDE_CN0_DYNAMICS
            else "train_normal_band_mean_cn0_relative"
            if normal_reference_mode != "none"
            else "causal_cn0_relative"
            if causal_baseline_mode != "none"
            else "cn0_plus_pseudorange_rate_state_adr_residual_and_cross_band"
            if INCLUDE_PSEUDORANGE_RATE and INCLUDE_STATE_ADR and INCLUDE_PSEUDORANGE_RESIDUAL and INCLUDE_CROSS_BAND
            else "cn0_plus_cross_band"
            if INCLUDE_CROSS_BAND and not (INCLUDE_PSEUDORANGE_RATE or INCLUDE_STATE_ADR or INCLUDE_PSEUDORANGE_RESIDUAL)
            else "cn0_plus_pseudorange_rate_state_adr_and_residual"
            if INCLUDE_PSEUDORANGE_RATE and INCLUDE_STATE_ADR and INCLUDE_PSEUDORANGE_RESIDUAL
            else "cn0_plus_pseudorange_rate_and_state_adr"
            if INCLUDE_PSEUDORANGE_RATE and INCLUDE_STATE_ADR
            else "cn0_plus_pseudorange_rate"
            if INCLUDE_PSEUDORANGE_RATE
            else "cn0_plus_state_adr_and_residual"
            if INCLUDE_STATE_ADR and INCLUDE_PSEUDORANGE_RESIDUAL
            else "cn0_plus_state_adr"
            if INCLUDE_STATE_ADR
            else "cn0_plus_pseudorange_residual"
            if INCLUDE_PSEUDORANGE_RESIDUAL
            else "baseline"
        ),
        "feature_names": FEATURE_NAMES,
        "features": FEATURES,
        "bands": list(BANDS),
        "num_classes": 4,
        "class_semantics": {"0": "normal", "1": "L1", "2": "L5", "3": "L1+L5"},
        "scaler_mode": scaler_mode,
        "scaler_fit_split": "train",
        "scaler_fit_unit": scaler_fit_unit,
        "normal_reference": {
            "mode": normal_reference_mode,
            "reference_file": "normal_reference.json",
            "fit_split": normal_reference.get("fit_split"),
            "fit_unit": normal_reference.get("fit_unit"),
            "minimum_epochs": normal_reference.get("minimum_epochs"),
            "cn0_feature_transform": (
                "L1_Cn0DbHz and L5_Cn0DbHz each equal their raw per-epoch "
                "band mean minus the frozen train-only normal-band reference; "
                "Cn0DbHzL1MinusL5 is recomputed after those subtractions"
                if normal_reference_mode != "none"
                else "L1_Cn0DbHz and L5_Cn0DbHz are raw per-epoch band means"
            ),
            "application": normal_reference_application,
        },
        "cn0_dynamics": {
            "enabled": INCLUDE_CN0_DYNAMICS,
            "feature_names": CN0_DYNAMICS_NAMES if INCLUDE_CN0_DYNAMICS else [],
            "source_features": ["L1_Cn0DbHz", "L5_Cn0DbHz"],
            "input_semantics": (
                "frozen normal-reference C/N0 residuals"
                if normal_reference_mode != "none"
                else "raw band-mean C/N0"
            ),
            "window_epochs": TIME_STEPS,
            "causality": "current epoch plus preceding continuous epochs only",
            "minimum_finite_observations_for_slope_and_std": 2,
            "valid_count_features": [
                "L1_Cn0W5ValidCount", "L5_Cn0W5ValidCount",
            ] if INCLUDE_CN0_DYNAMICS else [],
            "reset_boundaries": ["split", "segment", "receiver_gap"],
            "missing_history": (
                "exclude missing band values; stats are NaN when the endpoint is "
                "missing or history has fewer than two values; valid-count retains "
                "the available-history size before train-only scaling"
            ),
        },
        "paired_pseudorange_rate": {
            "enabled": INCLUDE_PAIRED_PSEUDORANGE_RATE,
            "feature_names": (
                [*PAIRED_PRR_CONTINUOUS_NAMES, PAIRED_PRR_AVAILABILITY_NAME]
                if INCLUDE_PAIRED_PSEUDORANGE_RATE else []
            ),
            "raw_feature": RATE_RAW_FEATURE,
            "pair_key": ["TimeNanos", "ConstellationType", "Svid"],
            "within_band_duplicate_policy": "median per epoch/satellite/band",
            "pair_difference": "PseudorangeRate_L1 - PseudorangeRate_L5",
            "reference": {
                "mode": paired_prr_reference.get("mode"),
                "reference_file": (
                    "paired_pseudorange_rate_reference.json"
                    if INCLUDE_PAIRED_PSEUDORANGE_RATE else None
                ),
                "fit_split": paired_prr_reference.get("fit_split"),
                "fit_population": paired_prr_reference.get("fit_population"),
                "minimum_pairs": paired_prr_reference.get("minimum_pairs"),
                "application": paired_prr_application,
            },
            "epoch_aggregation": {
                "PrrPairMedianResidual": "median across same-satellite pairs",
                "PrrPairAbsMedianResidual": "median absolute residual across pairs",
                "PrrPairMadResidual": "median absolute deviation across pairs",
                "PrrPairAvailable": "1 iff at least one valid L1/L5 same-satellite pair",
            },
            "missing_pair_policy": (
                "continuous pair summaries remain missing until train-only scaling; "
                "PrrPairAvailable remains an unscaled zero"
            ),
        },
        "causal_baseline": {
            "mode": causal_baseline_mode,
            "half_life_seconds": float(causal_half_life_seconds),
            "normal_threshold": float(causal_normal_threshold),
            "gate_predictions": str(gate_predictions_path) if gate_predictions_path else None,
            "feature_at_t_uses": "baseline_before_t_update",
            "reset_boundaries": [
                "recording", "source", "device", "split", "segment", "receiver_gap"
            ],
            "warmup_epochs_without_gate": TIME_STEPS - 1 if causal_baseline_mode == "gated" else 0,
            "construction_stats": causal_stats,
        },
        "single_band_policy": (
            "endpoint epoch observing only one band is labeled 0 and flagged "
            "single_band_mask=True; trainers and metrics must exclude it"
        ),
        "scenario_to_class": scenario_to_class,
        "data_scope": scope,
        "canonical_clock": "utcTimeMillis",
        "window_clock": "TimeNanos",
        "device_mapping": device_to_id,
        "source_count": len(source_to_id),
        "online_rollout_trace": {
            "stream_key": "source_id:split:segment_key; reset state when it changes",
            "reset_boundaries": ["recording", "source", "split", "segment", "receiver_gap"],
        },
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
    parser.add_argument(
        "--scope", choices=("static", "dynamic", "all"), default="static",
        help="Which recordings to include by Scenario prefix: st_ / dy_ / both.",
    )
    parser.add_argument(
        "--scaler-mode", choices=("per_device", "global"), default="per_device",
        help=(
            "Train-only continuous-feature standardization: one scaler per device "
            "or one shared scaler across all training devices."
        ),
    )
    parser.add_argument(
        "--normal-reference-mode",
        choices=NORMAL_REFERENCE_MODES,
        default="none",
        help=(
            "Subtract a frozen train-only normal C/N0 mean per known device/band, "
            "with a fold-global fallback for an unseen or insufficient device."
        ),
    )
    parser.add_argument(
        "--normal-reference-min-epochs",
        type=int,
        default=1,
        help=(
            "Minimum train-only normal band-mean epochs required for a device "
            "reference before using the fold-global fallback."
        ),
    )
    parser.add_argument(
        "--causal-baseline-mode",
        choices=CAUSAL_BASELINE_MODES,
        default="none",
        help=(
            "Causal C/N0 reference features: disabled, always-updating EMA, or "
            "EMA updated only by external high-confidence normal gate scores."
        ),
    )
    parser.add_argument(
        "--causal-half-life-seconds",
        type=float,
        default=60.0,
        help="Time-aware EMA half-life used by causal C/N0 baselines.",
    )
    parser.add_argument(
        "--causal-normal-threshold",
        type=float,
        default=0.8,
        help="Minimum external P(normal) required to update a gated baseline.",
    )
    parser.add_argument(
        "--gate-predictions",
        type=Path,
        default=None,
        help=(
            "CSV with source_id, device_id, window_time_nanos and prob_normal; "
            "required by --causal-baseline-mode gated."
        ),
    )
    parser.add_argument(
        "--include-pseudorange-rate", action="store_true",
        help=(
            "Append per-band pseudorange-rate mean, causal slope, robust MAD and "
            "within-epoch outlier ratio features."
        ),
    )
    parser.add_argument(
        "--include-state-adr", action="store_true",
        help=(
            "Append State/ADR continuity features: tracking-state bit ratios, "
            "state transitions, ADR reset/slip ratio, ADR-state transitions, "
            "ADR difference magnitude and missingness."
        ),
    )
    parser.add_argument(
        "--include-pseudorange-residual", action="store_true",
        help=(
            "Append causal same-epoch, same-constellation robust pseudorange "
            "residual MAD, P95 and outlier-ratio features."
        ),
    )
    parser.add_argument(
        "--include-cross-band", action="store_true",
        help="Append causal L1/L5 C/N0 slope-difference and opposite-trend features.",
    )
    parser.add_argument(
        "--include-cn0-dynamics", action="store_true",
        help=(
            "Append per-band causal W5 C/N0 slope and standard-deviation features, "
            "reset at split, segment, and receiver-gap boundaries."
        ),
    )
    parser.add_argument(
        "--include-paired-pseudorange-rate", action="store_true",
        help=(
            "Append same-satellite, same-epoch L1/L5 pseudorange-rate residual "
            "summaries fitted to a frozen train-normal device/constellation baseline."
        ),
    )
    parser.add_argument(
        "--paired-pseudorange-rate-reference-min-pairs",
        type=int,
        default=256,
        help=(
            "Minimum train-normal same-satellite pair count for a device/constellation "
            "reference before its global fallback is used."
        ),
    )
    args = parser.parse_args()
    if args.causal_half_life_seconds <= 0:
        parser.error("--causal-half-life-seconds must be positive")
    if not 0 <= args.causal_normal_threshold <= 1:
        parser.error("--causal-normal-threshold must be in [0, 1]")
    if args.causal_baseline_mode == "gated" and args.gate_predictions is None:
        parser.error("--causal-baseline-mode gated requires --gate-predictions")
    if args.normal_reference_min_epochs < 1:
        parser.error("--normal-reference-min-epochs must be at least 1")
    if args.paired_pseudorange_rate_reference_min_pairs < 1:
        parser.error("--paired-pseudorange-rate-reference-min-pairs must be at least 1")
    if args.normal_reference_mode != "none" and args.scaler_mode != "global":
        parser.error("normal reference mode requires --scaler-mode global")
    if args.normal_reference_mode != "none" and args.causal_baseline_mode != "none":
        parser.error("normal reference and causal baseline modes are mutually exclusive")
    summary = build(
        args.csv, args.epoch_manifest, args.outer_manifest, args.config,
        args.output_dir, args.scope, args.scaler_mode,
        args.include_pseudorange_rate, args.include_state_adr,
        args.include_pseudorange_residual,
        args.include_cross_band,
        args.include_cn0_dynamics,
        args.include_paired_pseudorange_rate,
        args.paired_pseudorange_rate_reference_min_pairs,
        args.causal_baseline_mode,
        args.causal_half_life_seconds,
        args.causal_normal_threshold,
        args.gate_predictions,
        args.normal_reference_mode,
        args.normal_reference_min_epochs,
    )
    print(json.dumps(summary["split_stats"], indent=2))


if __name__ == "__main__":
    main()

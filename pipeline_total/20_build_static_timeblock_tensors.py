"""Build raw and per-signal-statistic tensors for the static time-block protocol.

This builder is deliberately separate from the legacy random/recording split
builder.  One complete recording (``Environment/Scenario/Session``) is kept
as the outer test set.  The remaining recordings are divided into contiguous
canonical UTC time blocks for train/validation.  Windows are made *after* the
block assignment and a window is emitted only when all of its epochs belong to
the same split and the same continuous run.  Consequently neither a raw
window nor a per-signal statistic history can borrow observations across a
train/validation boundary.

The canonical clock is ``utcTimeMillis``.  ``TimeNanos`` is used only to order
the epochs inside one source log; its absolute origin differs substantially
between devices in the same physical recording.  A block manifest can be
provided with UTC interval columns (``start_utc_millis`` and
``end_utc_millis``); otherwise a deterministic 256-canonical-epoch 80/20
manifest is generated.

Outputs are compatible with ``21_train_static_signal_fusion.py``::

    output_dir/raw/{train,val,test}.npz    # [B, 128, 5, 7]
    output_dir/stats/{train,val,test}.npz  # [B, 128, 1, 19]

Both NPZ families contain ``x``, ``mask``, ``y``, ``is_dynamic`` and
``device_id``.  Scalers are fitted using train blocks only.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
KEYS = ["Environment", "Scenario", "Session"]
SOURCE_COL = "SourceRelativePath"
TIME_STEPS = 5
MAX_SIGNALS = 128
IGNORE_INDEX = -100
EPOCH_GAP_NS = 2_000_000_000  # a missing receiver epoch starts a new run
BLOCK_SIZE = 256
VALIDATION_FRACTION = 0.20
GUARD_EPOCHS = TIME_STEPS - 1

RAW_FEATURES = [
    "Cn0DbHz",
    "Cn0DbHz_dt",
    "Cn0DbHz_std",
    "AgcDb",
    "ReceivedSvTimeUncertaintyNanos",
    "PseudorangeRateUncertaintyMetersPerSecond",
    "FreqBand",
]
STAT_FEATURES = [
    "Cn0DbHz",
    "AgcDb",
    "ReceivedSvTimeUncertaintyNanos",
    "PseudorangeRateUncertaintyMetersPerSecond",
]
RAW_NAMES = RAW_FEATURES
STAT_NAMES: list[str] = []
STAT_SCALED_COUNT = 16

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
LOG = logging.getLogger(__name__)


def configure(time_steps: int) -> None:
    global TIME_STEPS, GUARD_EPOCHS, STAT_NAMES
    TIME_STEPS = int(time_steps)
    GUARD_EPOCHS = max(TIME_STEPS - 1, 0)
    suffix = f"W{TIME_STEPS}"
    STAT_NAMES = [
        f"{feature}{stat}"
        for feature in STAT_FEATURES
        for stat in (f"Last{suffix}", f"Mean{suffix}", f"Std{suffix}", f"Slope{suffix}")
    ] + ["IsL5", f"SignalHistoryRatio{suffix}", f"AgcObservedRatio{suffix}"]


def _norm_key_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for key in KEYS:
        out[key] = out[key].astype(str)
    return out


def _recording_key(row: Mapping[str, object]) -> tuple[str, str, str]:
    return tuple(str(row[k]) for k in KEYS)  # type: ignore[return-value]


def _find_column(columns: Iterable[str], names: Iterable[str]) -> str | None:
    existing = set(columns)
    for name in names:
        if name in existing:
            return name
    return None


@dataclass(frozen=True)
class BlockInterval:
    start: int
    end: int
    split: str
    block_id: int
    segment: str = "0"


def _load_outer_manifest(path: Path) -> pd.DataFrame:
    manifest = _norm_key_frame(pd.read_csv(path, encoding="utf-8-sig"))
    required = set(KEYS) | {"split"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"Outer manifest missing columns: {sorted(missing)}")
    if manifest.duplicated(KEYS).any():
        raise ValueError("Outer manifest contains duplicate recording identities")
    allowed = {"train", "dev", "development", "val", "test"}
    unknown = set(manifest["split"].astype(str)).difference(allowed)
    if unknown:
        raise ValueError(f"Unsupported outer split values: {sorted(unknown)}")
    # Direction 2 has one complete test recording and development recordings.
    # Treat an explicitly marked val recording as development too; its inner
    # blocks will be used for validation, never as an outer test.
    manifest["outer_role"] = np.where(manifest["split"].astype(str) == "test", "test", "dev")
    if int((manifest["outer_role"] == "test").sum()) != 1:
        raise ValueError("Expected exactly one outer test recording")
    return manifest


def _canonical_epoch_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per source/TimeNanos with a robust UTC representative."""
    source_cols = [*KEYS, SOURCE_COL, "DeviceName", "TimeNanos"]
    # Median UTC over all signals at one receiver epoch removes small per-row
    # timing jitter in watch logs while preserving the real recording clock.
    grouped = (
        df.groupby(source_cols, as_index=False, sort=False)["utcTimeMillis"]
        .median()
        .rename(columns={"utcTimeMillis": "epoch_utc_millis"})
    )
    grouped["epoch_utc_millis"] = grouped["epoch_utc_millis"].round().astype(np.int64)
    return grouped.sort_values([*KEYS, SOURCE_COL, "TimeNanos"], kind="mergesort")


def _auto_block_manifest(
    df: pd.DataFrame, outer: pd.DataFrame, output_path: Path, block_size: int = BLOCK_SIZE
) -> pd.DataFrame:
    """Create deterministic recording-level UTC block intervals.

    Blocks are defined on the sorted union of canonical UTC epochs in one
    recording, not on raw ``TimeNanos``.  The last approximately 20% of blocks
    are validation blocks.  ``GUARD_EPOCHS`` canonical epochs around the
    boundary are marked ``gap`` and never contribute windows.
    """
    epoch_table = _canonical_epoch_table(df)
    dev_keys = set(map(tuple, outer.loc[outer["outer_role"] == "dev", KEYS].to_numpy()))
    rows: list[dict[str, object]] = []
    for key, group in epoch_table.groupby(KEYS, sort=True):
        key_tuple = tuple(str(v) for v in key)
        if key_tuple not in dev_keys:
            continue
        epochs = np.sort(group["epoch_utc_millis"].unique().astype(np.int64))
        if len(epochs) < TIME_STEPS:
            continue
        n_blocks = max(2, int(np.ceil(len(epochs) / block_size)))
        n_val = max(1, int(np.ceil(n_blocks * VALIDATION_FRACTION)))
        val_first_block = n_blocks - n_val
        # Assign labels on the canonical epoch index first.  This makes the
        # guard symmetric and ensures every emitted interval has both UTC and
        # epoch bounds (the earlier implementation left the guard overlapping
        # the train interval).
        labels = np.full(len(epochs), "train", dtype=object)
        labels[val_first_block * block_size :] = "val"
        boundary = val_first_block * block_size
        if GUARD_EPOCHS:
            guard_lo = max(0, boundary - GUARD_EPOCHS)
            guard_hi = min(len(epochs), boundary + GUARD_EPOCHS)
            labels[guard_lo:guard_hi] = "gap"
        run_start = 0
        run_id = 0
        for i in range(1, len(labels) + 1):
            if i < len(labels) and labels[i] == labels[run_start]:
                continue
            lo, hi = run_start, i
            rows.append({**dict(zip(KEYS, key_tuple)), "block_id": run_id,
                         "split": str(labels[run_start]), "epoch_start": lo,
                         "epoch_end": hi, "start_utc_millis": int(epochs[lo]),
                         "end_utc_millis": int(epochs[hi]) if hi < len(epochs) else int(epochs[-1] + 1_000)})
            run_id += 1
            run_start = i
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("Automatic block manifest is empty")
    # Keep the same column metadata as ``_load_block_manifest`` so the
    # in-memory generated manifest follows exactly the supplied-manifest path.
    result.attrs["start_utc_col"] = "start_utc_millis"
    result.attrs["end_utc_col"] = "end_utc_millis"
    result.attrs["start_epoch_col"] = "epoch_start"
    result.attrs["end_epoch_col"] = "epoch_end"
    # The epoch-index columns make the generated assignment auditable and
    # avoid ambiguity when several source clocks have different offsets.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    return result


def _load_block_manifest(path: Path) -> pd.DataFrame:
    block = _norm_key_frame(pd.read_csv(path, encoding="utf-8-sig"))
    # The protocol generator's authoritative epoch manifest uses
    # ``canonical_epoch_ms`` + ``split`` (and the block manifest uses
    # ``canonical_start_ms``/``canonical_end_ms`` + ``epoch_split``).  Normalize
    # both forms to the interval representation consumed below.
    if "split" not in block.columns and "epoch_split" in block.columns:
        block["split"] = block["epoch_split"].astype(str)
    if "canonical_epoch_ms" in block.columns:
        block["start_utc_millis"] = pd.to_numeric(block["canonical_epoch_ms"], errors="raise").astype("int64")
        block["end_utc_millis"] = block["start_utc_millis"] + 1000
    elif "canonical_start_ms" in block.columns and "canonical_end_ms" in block.columns:
        block["start_utc_millis"] = pd.to_numeric(block["canonical_start_ms"], errors="raise").astype("int64")
        block["end_utc_millis"] = pd.to_numeric(block["canonical_end_ms"], errors="raise").astype("int64") + 1000
    required = set(KEYS) | {"split"}
    missing = required.difference(block.columns)
    if missing:
        raise ValueError(f"Block manifest missing columns: {sorted(missing)}")
    allowed = {"train", "val", "gap", "guard", "dev", "test"}
    unknown = set(block["split"].astype(str)).difference(allowed)
    if unknown:
        raise ValueError(f"Unsupported block split values: {sorted(unknown)}")
    block["split"] = block["split"].replace({"dev": "train", "guard": "gap"})
    if "canonical_epoch_ms" in block.columns:
        if "segment_key" in block.columns:
            block["_segment"] = block["segment_key"].astype(str)
        elif "segment_id" in block.columns:
            block["_segment"] = block["segment_id"].astype(str)
        elif "block_uid" in block.columns:
            block["_segment"] = block["block_uid"].astype(str)
        else:
            block["_segment"] = "0"
        block["canonical_epoch_ms"] = pd.to_numeric(block["canonical_epoch_ms"], errors="raise").astype(np.int64)
        block.attrs["epoch_manifest"] = True
    duplicate_keys = [*KEYS, "canonical_epoch_ms"] if "canonical_epoch_ms" in block.columns else ([*KEYS, "block_id"] if "block_id" in block else KEYS)
    if block.duplicated(duplicate_keys).any():
        LOG.warning("Block manifest has duplicate recording/block rows; interval order will decide assignment")
    start_utc = _find_column(block.columns, ["start_utc_millis", "start_utc_ms", "start_utc", "utc_start_millis", "canonical_start_ms"])
    end_utc = _find_column(block.columns, ["end_utc_millis", "end_utc_ms", "end_utc", "utc_end_millis", "canonical_end_ms"])
    start_epoch = _find_column(block.columns, ["epoch_start", "start_epoch", "canonical_epoch_start"])
    end_epoch = _find_column(block.columns, ["epoch_end", "end_epoch", "canonical_epoch_end"])
    if not ((start_utc and end_utc) or (start_epoch and end_epoch)):
        raise ValueError("Block manifest needs UTC interval or epoch_start/epoch_end columns")
    block.attrs["start_utc_col"] = start_utc
    block.attrs["end_utc_col"] = end_utc
    block.attrs["start_epoch_col"] = start_epoch
    block.attrs["end_epoch_col"] = end_epoch
    return block


def _intervals_for_recording(block: pd.DataFrame, key: tuple[str, str, str]) -> list[BlockInterval]:
    if block.attrs.get("epoch_manifest"):
        # Epoch manifests are looked up directly by canonical_epoch_ms; turning
        # every row into an interval here would be both redundant and costly.
        return []
    subset = block[(block[KEYS] == np.asarray(key)).all(axis=1)].copy()
    if subset.empty:
        return []
    start_col = block.attrs.get("start_utc_col")
    end_col = block.attrs.get("end_utc_col")
    if not start_col or not end_col:
        return []
    subset = subset.sort_values([start_col, end_col], kind="mergesort")
    out: list[BlockInterval] = []
    for i, row in subset.iterrows():
        start = int(row[start_col]); end = int(row[end_col])
        if end <= start:
            raise ValueError(f"Invalid UTC interval for {key}: {start}..{end}")
        block_id = int(row.get("block_id", i))
        if "segment_key" in row.index:
            segment = str(row["segment_key"])
        elif "segment_id" in row.index:
            segment = str(row["segment_id"])
        elif "block_uid" in row.index:
            segment = str(row["block_uid"])
        else:
            segment = str(block_id)
        out.append(BlockInterval(start, end, str(row["split"]), block_id, segment))
    return out


def _assign_epoch_metadata(
    block: pd.DataFrame,
    key: tuple[str, str, str],
    epoch_utc: np.ndarray,
    intervals: list[BlockInterval],
    epoch_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return final split and canonical segment for each source epoch."""
    splits = np.full(len(epoch_utc), "unassigned", dtype=object)
    segments = np.full(len(epoch_utc), "unassigned", dtype=object)
    if block.attrs.get("epoch_manifest"):
        subset = block[(block[KEYS] == np.asarray(key)).all(axis=1)]
        if subset.empty:
            return splits, segments
        lookup = subset.drop_duplicates("canonical_epoch_ms").set_index("canonical_epoch_ms")
        canonical = (np.asarray(epoch_utc, dtype=np.int64) // 1000) * 1000
        for i, value in enumerate(canonical):
            if value in lookup.index:
                row = lookup.loc[value]
                splits[i] = str(row["split"])
                segments[i] = str(row["_segment"])
        return splits, segments
    if intervals:
        for interval in intervals:
            hit = (epoch_utc >= interval.start) & (epoch_utc < interval.end)
            splits[hit] = interval.split
            segments[hit] = interval.segment
        return splits, segments
    # Epoch-index-only custom manifests.
    if epoch_indices is not None:
        subset = block[(block[KEYS] == np.asarray(key)).all(axis=1)]
        start_col = block.attrs.get("start_epoch_col")
        end_col = block.attrs.get("end_epoch_col")
        if start_col and end_col:
            for i, row in subset.iterrows():
                lo, hi = int(row[start_col]), int(row[end_col])
                hit = (epoch_indices >= lo) & (epoch_indices < hi)
                splits[hit] = str(row["split"])
                segments[hit] = str(row.get("segment_id", row.get("block_id", i)))
    return splits, segments


def _assign_epoch_splits(
    epoch_utc: np.ndarray,
    intervals: list[BlockInterval],
    epoch_indices: np.ndarray | None = None,
    block_rows: pd.DataFrame | None = None,
) -> np.ndarray:
    """Assign each source epoch to train/val/gap/test or ``unassigned``.

    UTC intervals are preferred.  For manifests generated with only
    ``epoch_start/epoch_end``, ``epoch_indices`` are interpreted against the
    sorted canonical epoch union supplied by the caller.
    """
    assigned = np.full(len(epoch_utc), "unassigned", dtype=object)
    if intervals:
        for interval in intervals:
            # Half-open intervals prevent a boundary epoch from being assigned
            # twice.  The generated final interval ends at max+1000ms.
            hit = (epoch_utc >= interval.start) & (epoch_utc < interval.end)
            assigned[hit] = interval.split
        return assigned
    if block_rows is not None and epoch_indices is not None:
        start_col = block_rows.attrs.get("start_epoch_col")
        end_col = block_rows.attrs.get("end_epoch_col")
        if start_col and end_col:
            for _, row in block_rows.iterrows():
                lo, hi = int(row[start_col]), int(row[end_col])
                hit = (epoch_indices >= lo) & (epoch_indices < hi)
                assigned[hit] = str(row["split"])
    return assigned


def _slope(times_ns: np.ndarray, values: np.ndarray) -> float:
    valid = np.isfinite(values)
    if int(valid.sum()) < 2:
        return 0.0
    t = times_ns[valid].astype(np.float64) * 1e-9
    v = values[valid].astype(np.float64)
    t -= t.mean()
    denominator = float(np.dot(t, t))
    if denominator <= 0:
        return 0.0
    return float(np.dot(t, v - v.mean()) / denominator)


def _stats(times_ns: np.ndarray, values: np.ndarray, endpoint: float) -> list[float]:
    valid = values[np.isfinite(values)]
    if len(valid):
        mean = float(valid.mean()); std = float(valid.std(ddof=0))
    else:
        mean = float("nan"); std = float("nan")
    return [float(endpoint), mean, std, _slope(times_ns, values)]


def _aggregate_source(df: pd.DataFrame) -> pd.DataFrame:
    aggregation: dict[str, str] = {
        "Label": "max",
        "TOW": "median",
        "FreqBand": "median",
        "utcTimeMillis": "median",
        **{feature: "median" for feature in RAW_FEATURES if feature != "FreqBand"},
    }
    # Keep only fields actually present in the processed CSV.
    aggregation = {k: v for k, v in aggregation.items() if k in df.columns}
    return (
        df.groupby(["TimeNanos", "signal_id"], as_index=False, sort=False)
        .agg(aggregation)
        .sort_values(["TimeNanos", "signal_id"], kind="mergesort")
    )


def _make_source_windows(
    source: pd.DataFrame,
    epoch_splits: np.ndarray,
    epoch_segments: np.ndarray,
    device_id: int,
    recording_id: int,
    source_id: int,
    signal_to_id: dict[str, int],
) -> dict[str, dict[str, list[np.ndarray]]]:
    """Build all eligible raw/stats windows for one source in one pass."""
    def empty_part() -> dict[str, list[np.ndarray]]:
        return {
            "raw": [], "stats": [], "mask": [], "y": [], "dynamic": [], "device": [],
            "window_time_nanos": [], "endpoint_utc_millis": [], "endpoint_tow": [],
            "recording_id": [], "source_id": [], "signal_id": [],
        }

    outputs = {split: empty_part() for split in ("train", "val", "test")}
    times = np.sort(source["TimeNanos"].unique().astype(np.int64))
    if len(times) < TIME_STEPS:
        return outputs
    # source rows are sorted by TimeNanos; map epoch to split.
    split_by_time = {int(t): str(s) for t, s in zip(times, epoch_splits)}
    segment_by_time = {int(t): str(s) for t, s in zip(times, epoch_segments)}
    observations: dict[str, dict[int, tuple[np.ndarray, float, int, float, float]]] = {}
    endpoint_ids: dict[int, list[str]] = {}
    for row in source.itertuples(index=False):
        identity = str(row.signal_id); t = int(row.TimeNanos)
        values = np.asarray([getattr(row, f) for f in RAW_FEATURES], dtype=np.float64)
        observations.setdefault(identity, {})[t] = (
            values,
            float(getattr(row, "utcTimeMillis", np.nan)),
            int(getattr(row, "Label", 0) > 0),
            float(getattr(row, "FreqBand", np.nan)),
            float(getattr(row, "TOW", np.nan)),
        )
        endpoint_ids.setdefault(t, []).append(identity)

    # Only consecutive epochs in the same split are eligible.  This explicitly
    # rejects windows crossing a block boundary and windows over source gaps.
    for end_i in range(TIME_STEPS - 1, len(times)):
        window_times = times[end_i - TIME_STEPS + 1 : end_i + 1]
        if np.any(np.diff(window_times) > EPOCH_GAP_NS):
            continue
        labels = [split_by_time.get(int(t), "unassigned") for t in window_times]
        if len(set(labels)) != 1 or labels[0] not in outputs:
            continue
        split_filter = labels[0]
        segments = [segment_by_time.get(int(t), "unassigned") for t in window_times]
        # A canonical segment identifies a continuous source interval in the
        # protocol.  Never let a W5 history cross a segment or a guard, even
        # when both neighboring epochs happen to share the same raw split.
        if any(segment in {"unassigned", "nan", "None"} for segment in segments):
            continue
        if len(set(segments)) != 1:
            continue
        endpoint = int(window_times[-1])
        identities = np.asarray(sorted(set(endpoint_ids.get(endpoint, []))), dtype=object)
        if len(identities) > MAX_SIGNALS:
            raise ValueError(f"{split_filter}: source window has {len(identities)} signals > {MAX_SIGNALS}")
        raw_x = np.full((MAX_SIGNALS, TIME_STEPS, len(RAW_FEATURES)), np.nan, dtype=np.float32)
        stats_x = np.full((MAX_SIGNALS, 1, len(STAT_NAMES)), np.nan, dtype=np.float32)
        mask = np.zeros(MAX_SIGNALS, dtype=bool)
        y = np.full(MAX_SIGNALS, IGNORE_INDEX, dtype=np.int64)
        signal_ids = np.full(MAX_SIGNALS, -1, dtype=np.int32)
        endpoint_utc_values: list[float] = []
        endpoint_tow_values: list[float] = []
        for slot, identity in enumerate(identities):
            signal = observations[str(identity)]
            observed = [(int(t), signal[int(t)]) for t in window_times if int(t) in signal]
            if endpoint not in signal:
                continue
            hist_t = np.asarray([item[0] for item in observed], dtype=np.int64)
            hist_values = np.stack([item[1][0] for item in observed])
            endpoint_values, endpoint_utc, endpoint_label, endpoint_band, endpoint_tow = signal[endpoint]
            raw_values = np.asarray([item[1][0] for item in observed], dtype=np.float64)
            for item_i, (t, _record) in enumerate(observed):
                raw_x[slot, int(np.searchsorted(window_times, t)), :] = raw_values[item_i].astype(np.float32)
            vector: list[float] = []
            for stat_feature in STAT_FEATURES:
                feature_i = RAW_FEATURES.index(stat_feature)
                vector.extend(_stats(hist_t, hist_values[:, feature_i], endpoint_values[feature_i]))
            observed_count = len(observed)
            agc_count = int(np.isfinite(hist_values[:, RAW_FEATURES.index("AgcDb")]).sum())
            vector.extend([
                1.0 if endpoint_band == 5.0 else 0.0,
                observed_count / TIME_STEPS,
                agc_count / TIME_STEPS,
            ])
            stats_x[slot, 0, :] = np.asarray(vector, dtype=np.float32)
            y[slot] = endpoint_label
            mask[slot] = True
            signal_ids[slot] = signal_to_id[str(identity)]
            endpoint_utc_values.append(endpoint_utc)
            endpoint_tow_values.append(endpoint_tow)
        if not mask.any():
            continue
        out = outputs[split_filter]
        out["raw"].append(raw_x); out["stats"].append(stats_x); out["mask"].append(mask)
        out["y"].append(y); out["dynamic"].append(np.asarray(False)); out["device"].append(np.asarray(device_id))
        out["window_time_nanos"].append(np.asarray(endpoint, dtype=np.int64))
        out["endpoint_utc_millis"].append(np.asarray(np.nanmedian(endpoint_utc_values), dtype=np.float64))
        out["endpoint_tow"].append(np.asarray(np.nanmedian(endpoint_tow_values), dtype=np.float64))
        out["recording_id"].append(np.asarray(recording_id, dtype=np.int32))
        out["source_id"].append(np.asarray(source_id, dtype=np.int32))
        out["signal_id"].append(signal_ids)
    return outputs


def _empty_arrays() -> dict[str, np.ndarray]:
    return {
        "raw": np.empty((0, MAX_SIGNALS, TIME_STEPS, len(RAW_FEATURES)), np.float32),
        "stats": np.empty((0, MAX_SIGNALS, 1, len(STAT_NAMES)), np.float32),
        "mask": np.empty((0, MAX_SIGNALS), bool),
        "y": np.empty((0, MAX_SIGNALS), np.int64),
        "dynamic": np.empty((0,), bool),
        "device": np.empty((0,), np.int64),
        "window_time_nanos": np.empty((0,), np.int64),
        "endpoint_utc_millis": np.empty((0,), np.float64),
        "endpoint_tow": np.empty((0,), np.float64),
        "recording_id": np.empty((0,), np.int32),
        "source_id": np.empty((0,), np.int32),
        "signal_id": np.empty((0, MAX_SIGNALS), np.int32),
    }


def _stack_windows(parts: list[dict[str, list[np.ndarray]]]) -> dict[str, np.ndarray]:
    if not parts:
        return _empty_arrays()
    result: dict[str, np.ndarray] = {}
    for key in (
        "raw", "stats", "mask", "y", "dynamic", "device", "window_time_nanos",
        "endpoint_utc_millis", "endpoint_tow", "recording_id", "source_id", "signal_id",
    ):
        values = [value for part in parts for value in part[key]]
        if not values:
            return _empty_arrays()
        result[key] = np.stack(values)
    return result


def _fit_apply_scaler(datasets: dict[str, dict[str, np.ndarray]], output_dir: Path) -> None:
    """Train-only per-device standardization for raw and stats tensors."""
    train = datasets["train"]
    all_train_raw = train["raw"][:, :, :, :][train["mask"]]
    all_train_stats = train["stats"][:, :, 0, :][train["mask"]]
    # Boolean indexing over [window, signal] leaves the singleton temporal
    # axis in place; remove it before fitting feature-wise statistics.
    if all_train_stats.ndim == 3:
        all_train_stats = all_train_stats[:, 0, :]
    if len(all_train_raw) == 0:
        raise ValueError("No active train samples; cannot fit scalers")

    def fit(values: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(values[:, :count], dtype=np.float64)
        finite = np.isfinite(values)
        counts = finite.sum(axis=0).astype(np.float64)
        safe_counts = np.where(counts > 0, counts, 1.0)
        safe_values = np.where(finite, values, 0.0)
        mean = safe_values.sum(axis=0) / safe_counts
        centered = np.where(finite, values - mean.reshape(1, -1), 0.0)
        std = np.sqrt((centered * centered).sum(axis=0) / safe_counts)
        mean = np.where(np.isfinite(mean), mean, 0.0)
        std = np.where(np.isfinite(std) & (std >= 1e-6), std, 1.0)
        return mean.astype(np.float32), std.astype(np.float32)

    raw_global_mean, raw_global_std = fit(all_train_raw.reshape(-1, all_train_raw.shape[-1]), len(RAW_FEATURES))
    stats_global_mean, stats_global_std = fit(all_train_stats, STAT_SCALED_COUNT)
    raw_device: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    stats_device: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for device in np.unique(train["device"]):
        sel = train["device"] == device
        active = train["mask"][sel]
        raw_values = train["raw"][sel][active]
        stats_values = train["stats"][sel, :, 0, :][active]
        if stats_values.ndim == 3:
            stats_values = stats_values[:, 0, :]
        raw_device[int(device)] = fit(raw_values.reshape(-1, raw_values.shape[-1]), len(RAW_FEATURES))
        stats_device[int(device)] = fit(stats_values, STAT_SCALED_COUNT)

    for data in datasets.values():
        for device in np.unique(data["device"]):
            sel = data["device"] == device
            raw_mean, raw_std = raw_device.get(int(device), (raw_global_mean, raw_global_std))
            stats_mean, stats_std = stats_device.get(int(device), (stats_global_mean, stats_global_std))
            raw = data["raw"][sel]
            raw = np.where(np.isfinite(raw), raw, raw_mean.reshape(1, 1, -1))
            data["raw"][sel] = (raw - raw_mean.reshape(1, 1, -1)) / raw_std.reshape(1, 1, -1)
            stats = data["stats"][sel]
            scaled = stats[:, :, 0, :STAT_SCALED_COUNT]
            scaled = np.where(np.isfinite(scaled), scaled, stats_mean.reshape(1, 1, -1))
            data["stats"][sel, :, 0, :STAT_SCALED_COUNT] = (scaled - stats_mean.reshape(1, 1, -1)) / stats_std.reshape(1, 1, -1)
            # Coverage ratios and IsL5 remain physical [0, 1].
            data["stats"][sel, :, 0, STAT_SCALED_COUNT:] = np.nan_to_num(
                data["stats"][sel, :, 0, STAT_SCALED_COUNT:], nan=0.0, posinf=0.0, neginf=0.0
            )
        if not np.isfinite(data["raw"]).all() or not np.isfinite(data["stats"]).all():
            raise ValueError("Non-finite values remain after train-only scaling")

    def serial(pair: tuple[np.ndarray, np.ndarray]) -> dict[str, list[float]]:
        return {"mean": pair[0].tolist(), "std": pair[1].tolist()}
    (output_dir / "raw_scaler.json").write_text(json.dumps({
        "features": RAW_NAMES, "global": serial((raw_global_mean, raw_global_std)),
        "per_device": {str(k): serial(v) for k, v in raw_device.items()},
    }, indent=2), encoding="utf-8")
    (output_dir / "stats_scaler.json").write_text(json.dumps({
        "scaled_features": STAT_NAMES[:STAT_SCALED_COUNT], "unscaled_features": STAT_NAMES[STAT_SCALED_COUNT:],
        "global": serial((stats_global_mean, stats_global_std)),
        "per_device": {str(k): serial(v) for k, v in stats_device.items()},
    }, indent=2), encoding="utf-8")


def build_fold(
    csv: Path,
    outer_manifest_path: Path,
    output_dir: Path,
    block_manifest_path: Path | None,
    time_steps: int,
    block_size: int,
) -> dict[str, dict[str, int]]:
    configure(time_steps)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_required = [
        *KEYS, "DeviceName", SOURCE_COL, "TimeNanos", "TOW", "utcTimeMillis", "signal_id",
        "Label", "LabelStatus", *RAW_FEATURES,
    ]
    LOG.info("Reading %s", csv)
    df = pd.read_csv(csv, usecols=lambda c: c in set(raw_required))
    missing = set(raw_required).difference(df.columns)
    if missing:
        raise ValueError(f"Processed CSV missing columns: {sorted(missing)}")
    df = df[(df["LabelStatus"].astype(str) == "reviewed") & df["Scenario"].astype(str).str.startswith("st_")].copy()
    for key in KEYS + ["DeviceName", SOURCE_COL, "signal_id"]:
        df[key] = df[key].astype(str)
    df["Label"] = (pd.to_numeric(df["Label"], errors="coerce").fillna(0) > 0).astype(np.int8)
    for feature in RAW_FEATURES + ["TimeNanos", "TOW", "utcTimeMillis"]:
        df[feature] = pd.to_numeric(df[feature], errors="coerce")
    df = df.dropna(subset=[*KEYS, SOURCE_COL, "DeviceName", "TimeNanos", "utcTimeMillis", "signal_id"])
    outer = _load_outer_manifest(outer_manifest_path)
    # Restrict to recordings listed by the outer protocol and retain its role.
    df = df.merge(outer[[*KEYS, "outer_role"]], on=KEYS, how="inner", validate="many_to_one")
    if df.empty:
        raise ValueError("No static reviewed rows match outer manifest")
    if block_manifest_path is None:
        block_manifest_path = output_dir / "block_manifest.csv"
        block = _auto_block_manifest(df, outer, block_manifest_path, block_size)
    else:
        block = _load_block_manifest(block_manifest_path)
        block.to_csv(output_dir / "block_manifest.csv", index=False, encoding="utf-8-sig")

    epoch_table = _canonical_epoch_table(df)
    device_names = sorted(df["DeviceName"].astype(str).unique())
    device_to_id = {name: i for i, name in enumerate(device_names)}
    (output_dir / "device_mapping.json").write_text(json.dumps(device_to_id, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "outer_recording_manifest.csv").write_text(outer.to_csv(index=False), encoding="utf-8-sig")

    recording_rows = outer[KEYS].sort_values(KEYS, kind="mergesort").reset_index(drop=True)
    recording_to_id = {
        tuple(row): index
        for index, row in enumerate(recording_rows[KEYS].astype(str).itertuples(index=False, name=None))
    }
    source_rows = (
        df[[*KEYS, "DeviceName", SOURCE_COL]]
        .drop_duplicates()
        .sort_values([*KEYS, "DeviceName", SOURCE_COL], kind="mergesort")
        .reset_index(drop=True)
    )
    source_to_id = {
        tuple(row): index
        for index, row in enumerate(
            source_rows[[*KEYS, "DeviceName", SOURCE_COL]].astype(str).itertuples(index=False, name=None)
        )
    }
    signal_values = sorted(df["signal_id"].astype(str).unique().tolist())
    signal_to_id = {signal: index for index, signal in enumerate(signal_values)}
    trace_index = {
        "recordings": recording_rows.to_dict(orient="records"),
        "sources": source_rows.to_dict(orient="records"),
        "signal_ids": signal_values,
        "description": "Integer indices stored in raw/*.npz for endpoint-level prediction traceability.",
    }
    (output_dir / "window_trace_index.json").write_text(
        json.dumps(trace_index, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    chunks: dict[str, list[dict[str, list[np.ndarray]]]] = {"train": [], "val": [], "test": []}
    assignment_rows: list[dict[str, object]] = []
    source_group_cols = [*KEYS, "DeviceName", SOURCE_COL, "outer_role"]
    for source_key, source_rows in tqdm(df.groupby(source_group_cols, sort=True), desc="Building source windows"):
        key_tuple = tuple(str(source_key[i]) for i in range(3))
        device_name = str(source_key[3]); outer_role = str(source_key[5])
        source = _aggregate_source(source_rows)
        source_times = np.sort(source["TimeNanos"].unique().astype(np.int64))
        source_epoch = epoch_table[
            (epoch_table[KEYS] == np.asarray(key_tuple)).all(axis=1)
            & (epoch_table["DeviceName"].astype(str) == device_name)
            & (epoch_table[SOURCE_COL].astype(str) == str(source_key[4]))
        ].sort_values("TimeNanos")
        if len(source_epoch) != len(source_times):
            raise RuntimeError(f"Canonical epoch mapping mismatch for {source_key}")
        intervals = _intervals_for_recording(block, key_tuple)
        canonical_union = np.sort(
            epoch_table[(epoch_table[KEYS] == np.asarray(key_tuple)).all(axis=1)]["epoch_utc_millis"].unique()
        )
        epoch_indices = np.searchsorted(canonical_union, source_epoch["epoch_utc_millis"].to_numpy())
        if outer_role == "test" and not block.attrs.get("epoch_manifest") and not intervals:
            splits = np.full(len(source_times), "test", dtype=object)
            segments = np.zeros(len(source_times), dtype=object)
        else:
            splits, segments = _assign_epoch_metadata(
                block, key_tuple, source_epoch["epoch_utc_millis"].to_numpy(), intervals, epoch_indices
            )
            # A complete outer test source may not be listed in a development
            # block manifest.  It remains test, with source-gap segmentation.
            if outer_role == "test":
                splits = np.where(splits == "unassigned", "test", splits)
                segments = np.where(segments == "unassigned", "0", segments)
            if np.all(splits == "unassigned"):
                raise ValueError(f"No block interval/epoch assignment matches source {source_key}")
        source_identity = (*key_tuple, device_name, str(source_key[4]))
        source_parts = _make_source_windows(
            source,
            splits,
            segments,
            device_to_id[device_name],
            recording_to_id[key_tuple],
            source_to_id[source_identity],
            signal_to_id,
        )
        for split, part in source_parts.items():
            chunks[split].append(part)
        # Audit epoch assignment counts.
        counts = pd.Series(splits).value_counts().to_dict()
        assignment_rows.append({**dict(zip(KEYS, key_tuple)), "DeviceName": device_name,
                                SOURCE_COL: str(source_key[4]), **{f"epochs_{k}": int(v) for k, v in counts.items()}})

    datasets: dict[str, dict[str, np.ndarray]] = {}
    for split in ("train", "val", "test"):
        datasets[split] = _stack_windows(chunks[split])
        LOG.info("%s raw=%s stats=%s active=%d positive=%d", split, datasets[split]["raw"].shape,
                 datasets[split]["stats"].shape, int(datasets[split]["mask"].sum()),
                 int((datasets[split]["y"][datasets[split]["mask"]] == 1).sum()))
    _fit_apply_scaler(datasets, output_dir)

    raw_dir = output_dir / "raw"; stats_dir = output_dir / "stats"
    raw_dir.mkdir(exist_ok=True); stats_dir.mkdir(exist_ok=True)
    for split, data in datasets.items():
        common = {"mask": data["mask"], "y": data["y"], "is_dynamic": data["dynamic"], "device_id": data["device"]}
        trace = {
            "window_time_nanos": data["window_time_nanos"],
            "endpoint_utc_millis": data["endpoint_utc_millis"],
            "endpoint_tow": data["endpoint_tow"],
            "recording_id": data["recording_id"],
            "source_id": data["source_id"],
            "signal_id": data["signal_id"],
        }
        np.savez_compressed(raw_dir / f"{split}.npz", x=data["raw"], **common, **trace)
        np.savez_compressed(stats_dir / f"{split}.npz", x=data["stats"], **common)
    pd.DataFrame(assignment_rows).to_csv(output_dir / "source_epoch_assignment_summary.csv", index=False, encoding="utf-8-sig")
    (output_dir / "raw" / "feature_names.json").write_text(json.dumps(RAW_NAMES, indent=2), encoding="utf-8")
    (output_dir / "stats" / "feature_names.json").write_text(json.dumps(STAT_NAMES, indent=2), encoding="utf-8")
    metadata = {
        "time_steps": TIME_STEPS, "max_signals": MAX_SIGNALS,
        "raw_feature_count": len(RAW_FEATURES), "stats_feature_count": len(STAT_NAMES),
        "raw_features": RAW_NAMES, "stats_features": STAT_NAMES,
        "canonical_clock": "utcTimeMillis", "window_clock": "TimeNanos",
        "trace_index": "window_trace_index.json",
        "block_size_canonical_epochs": block_size, "guard_epochs": GUARD_EPOCHS,
        "windows_cross_split_boundary": False, "stats_history_crosses_boundary": False,
    }
    (output_dir / "tensor_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {
        split: {"windows": int(len(data["raw"])), "active": int(data["mask"].sum()),
                "positive": int((data["y"][data["mask"]] == 1).sum())}
        for split, data in datasets.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=ROOT / "output" / "processed_gnss_data.csv")
    parser.add_argument("--outer-manifest", type=Path, required=True)
    parser.add_argument("--block-manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--time-steps", type=int, default=5)
    parser.add_argument("--block-size", type=int, default=BLOCK_SIZE)
    args = parser.parse_args()
    if args.time_steps < 2:
        parser.error("--time-steps must be at least 2")
    if args.block_size < args.time_steps:
        parser.error("--block-size must be at least --time-steps")
    summary = build_fold(args.csv, args.outer_manifest, args.output_dir, args.block_manifest, args.time_steps, args.block_size)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

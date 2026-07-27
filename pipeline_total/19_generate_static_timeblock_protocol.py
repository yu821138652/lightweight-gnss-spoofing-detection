"""Generate an outer-Session / inner-time-block signal CV protocol.

The protocol is deliberately stricter than a random window split:

* one or more complete reviewed recordings are held out as ``test`` per fold;
* the other recordings remain in the development pool;
* development recordings are divided into deterministic contiguous blocks of
  canonical epochs (the default block size is 256);
* a deterministic subset of blocks is assigned to validation; strict v2 mode
  controls the validation epoch fraction before using label balance as a
  tie-break, while the historical default remains reproducible;
* ``W-1`` epochs on both sides of every train/validation boundary are marked
  ``guard`` for a W-step causal window (W=5 by default).

``TimeNanos`` is not a suitable cross-device axis in this dataset: individual
receivers have large clock offsets.  ``utcTimeMillis`` is therefore mapped to
the canonical one-second epoch ``floor(utcTimeMillis / 1000) * 1000``.  Every
row from every device in one recording with the same canonical epoch receives
the same assignment.  The resulting manifests contain both block intervals
and an epoch-level lookup, so a tensor builder can map rows without guessing
at boundaries.

The script only writes manifests; it never reads or writes model tensors.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RECORDING_KEYS = ["Environment", "Scenario", "Session"]
STATIC_PREFIX = "st_"
DYNAMIC_PREFIX = "dy_"
DATA_SCOPES = ("static", "mixed")
DEFAULT_BLOCK_EPOCHS = 256
DEFAULT_TIME_STEPS = 5
DEFAULT_CANONICAL_MS = 1000
DEFAULT_SEGMENT_GAP_SECONDS = 2.0
DEFAULT_VAL_FRACTION = 0.20
DEFAULT_VAL_SIZE_TOLERANCE = 0.02
MAX_VALIDATION_DP_STATES = 2_000_000


def stable_int(value: str) -> int:
    """Return a process-independent integer tie-break key."""

    return int(hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:16], 16)


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA256 digest for a protocol input file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def recording_key_frame(df: pd.DataFrame) -> pd.Series:
    """Build a collision-resistant string key for a recording identity."""

    return pd.Series(
        list(map(tuple, df[RECORDING_KEYS].astype(str).to_numpy())),
        index=df.index,
        dtype="object",
    )


def read_recordings(path: Path, data_scope: str = "static") -> pd.DataFrame:
    """Read and validate the recording identities used by an outer protocol."""

    source = pd.read_csv(path, encoding="utf-8-sig")
    missing = set(RECORDING_KEYS).difference(source.columns)
    if missing:
        raise ValueError(f"Source recording manifest is missing columns: {sorted(missing)}")
    if data_scope not in DATA_SCOPES:
        raise ValueError(f"Unsupported data scope {data_scope!r}; expected one of {DATA_SCOPES}")
    optional = [column for column in ("test_fold",) if column in source.columns]
    recordings = source[[*RECORDING_KEYS, *optional]].drop_duplicates().copy()
    scenario = recordings["Scenario"].astype(str)
    selected = scenario.str.startswith(STATIC_PREFIX)
    if data_scope == "mixed":
        selected |= scenario.str.startswith(DYNAMIC_PREFIX)
    recordings = recordings.loc[selected].copy()
    if len(recordings) < 2:
        raise ValueError(
            f"Expected at least two reviewed {data_scope} recordings; "
            f"found {len(recordings)} in {path}"
        )
    if recordings.duplicated(RECORDING_KEYS).any():
        raise ValueError("Recording manifest contains duplicate recording identities.")
    recordings = recordings.sort_values(RECORDING_KEYS, kind="mergesort").reset_index(drop=True)
    recordings.insert(0, "recording_id", np.arange(len(recordings), dtype=np.int32))
    recordings["recording_key"] = recording_key_frame(recordings)
    if "test_fold" in recordings.columns:
        fold = pd.to_numeric(recordings["test_fold"], errors="coerce")
        if fold.isna().any() or (fold < 1).any() or not np.allclose(fold, np.rint(fold)):
            raise ValueError("test_fold must contain positive integer fold identifiers")
        recordings["test_fold"] = np.rint(fold).astype(np.int32)
    return recordings


def _aggregate_chunk(
    chunk: pd.DataFrame,
    target_keys: set[tuple[str, str, str]],
    data_scope: str,
) -> pd.DataFrame:
    """Aggregate one CSV chunk to canonical recording epochs."""

    if chunk.empty:
        return pd.DataFrame()
    chunk = chunk.copy()
    chunk["_recording_key"] = recording_key_frame(chunk)
    chunk = chunk.loc[chunk["_recording_key"].isin(target_keys)].copy()
    if chunk.empty:
        return pd.DataFrame()
    status = chunk["LabelStatus"].astype(str).eq("reviewed")
    scenario = chunk["Scenario"].astype(str)
    selected = scenario.str.startswith(STATIC_PREFIX)
    if data_scope == "mixed":
        selected |= scenario.str.startswith(DYNAMIC_PREFIX)
    chunk = chunk.loc[status & selected].copy()
    if chunk.empty:
        return pd.DataFrame()
    if chunk["utcTimeMillis"].isna().any():
        raise ValueError("Reviewed static rows contain missing utcTimeMillis values.")
    chunk["canonical_epoch_ms"] = (
        pd.to_numeric(chunk["utcTimeMillis"], errors="raise").astype("int64")
        // DEFAULT_CANONICAL_MS
    ) * DEFAULT_CANONICAL_MS
    chunk["Label"] = (pd.to_numeric(chunk["Label"], errors="coerce").fillna(0) > 0).astype("int8")
    group_cols = [*RECORDING_KEYS, "canonical_epoch_ms"]
    grouped = (
        chunk.groupby(group_cols, sort=False, observed=True)
        .agg(
            row_count=("Label", "size"),
            positive_rows=("Label", "sum"),
            positive_epoch=("Label", "max"),
            device_names=("DeviceName", lambda values: "\x1f".join(sorted(set(values.astype(str))))),
        )
        .reset_index()
    )
    return grouped


def load_epoch_table(
    csv_path: Path,
    recordings: pd.DataFrame,
    data_scope: str = "static",
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """Read the large processed CSV without materialising all signal rows."""

    usecols = [
        *RECORDING_KEYS,
        "DeviceName",
        "utcTimeMillis",
        "Label",
        "LabelStatus",
    ]
    target_keys = set(map(tuple, recordings[RECORDING_KEYS].astype(str).to_numpy()))
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize):
        part = _aggregate_chunk(chunk, target_keys, data_scope)
        if not part.empty:
            parts.append(part)
    if not parts:
        raise ValueError(f"No reviewed {data_scope} rows for the requested recordings in {csv_path}")
    raw = pd.concat(parts, ignore_index=True)
    group_cols = [*RECORDING_KEYS, "canonical_epoch_ms"]

    def merge_devices(values: Iterable[str]) -> str:
        names: set[str] = set()
        for value in values:
            if pd.isna(value):
                continue
            names.update(str(value).split("\x1f"))
        return "\x1f".join(sorted(names))

    epochs = (
        raw.groupby(group_cols, sort=False, observed=True)
        .agg(
            row_count=("row_count", "sum"),
            positive_rows=("positive_rows", "sum"),
            positive_epoch=("positive_epoch", "max"),
            device_names=("device_names", merge_devices),
        )
        .reset_index()
    )
    epochs["device_count"] = epochs["device_names"].map(
        lambda value: 0 if not value else len(str(value).split("\x1f"))
    ).astype("int16")
    epochs = epochs.drop(columns="device_names")
    epochs["recording_key"] = recording_key_frame(epochs)
    return epochs


def _split_segments(epochs: pd.DataFrame, gap_seconds: float, block_epochs: int) -> pd.DataFrame:
    """Add deterministic continuous-segment and epoch indices."""

    result = epochs.sort_values("canonical_epoch_ms", kind="mergesort").copy().reset_index(drop=True)
    gaps = result["canonical_epoch_ms"].diff().fillna(0).astype("int64") / 1000.0
    # ``>=`` intentionally treats a two-second hole as a segment boundary.
    result["segment_id"] = (gaps >= gap_seconds).cumsum().astype("int16")
    result["epoch_index"] = np.arange(len(result), dtype=np.int32)
    result["segment_epoch_index"] = result.groupby("segment_id", sort=False).cumcount().astype(np.int32)
    result["block_id"] = (result["segment_epoch_index"] // int(block_epochs)).astype(np.int16)
    return result


def _block_table(epoch_table: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["segment_id", "block_id"]
    blocks = (
        epoch_table.groupby(group_cols, sort=True, observed=True)
        .agg(
            epoch_count=("canonical_epoch_ms", "size"),
            canonical_start_ms=("canonical_epoch_ms", "min"),
            canonical_end_ms=("canonical_epoch_ms", "max"),
            row_count=("row_count", "sum"),
            positive_rows=("positive_rows", "sum"),
            positive_epochs=("positive_epoch", "sum"),
            device_count=("device_count", "max"),
        )
        .reset_index()
    )
    blocks["positive_ratio"] = blocks["positive_epochs"] / blocks["epoch_count"].clip(lower=1)
    blocks["block_uid"] = blocks.apply(
        lambda row: f"s{int(row.segment_id):02d}_b{int(row.block_id):03d}", axis=1
    )
    return blocks


def _candidate_score(
    blocks: pd.DataFrame,
    selected: tuple[int, ...],
    total_epochs: int,
    total_positive: int,
    val_fraction: float,
    tie_key: str,
) -> tuple[float, ...]:
    """Score label quality after epoch-size feasibility has been established.

    ``choose_validation_blocks`` first restricts candidates to a narrow band
    around the best attainable validation epoch count.  Consequently label
    support cannot pull the selection from (for example) 20% to 70% merely to
    obtain a closer positive ratio.
    """

    chosen = blocks.iloc[list(selected)]
    val_epochs = int(chosen["epoch_count"].sum())
    val_positive = int(chosen["positive_epochs"].sum())
    train_epochs = total_epochs - val_epochs
    train_positive = total_positive - val_positive
    val_ratio = val_positive / max(val_epochs, 1)
    total_ratio = total_positive / max(total_epochs, 1)
    has_both_classes = 0 < total_positive < total_epochs
    missing_val_positive = int(has_both_classes and val_positive == 0)
    missing_val_negative = int(has_both_classes and val_positive == val_epochs)
    missing_train_positive = int(has_both_classes and train_positive == 0)
    missing_train_negative = int(has_both_classes and train_positive == train_epochs)
    missing_class_count = (
        missing_val_positive
        + missing_val_negative
        + missing_train_positive
        + missing_train_negative
    )
    fraction_error = abs(val_epochs / max(total_epochs, 1) - val_fraction)
    ratio_error = abs(val_ratio - total_ratio)
    # Prefer a contiguous interval within one segment when scores tie, but do
    # not force it when a non-contiguous choice is materially better balanced.
    segment_count = len(set(int(blocks.iloc[i]["segment_id"]) for i in selected))
    adjacency_breaks = 0
    for left, right in zip(selected, selected[1:]):
        l = blocks.iloc[left]
        r = blocks.iloc[right]
        if int(l["segment_id"]) != int(r["segment_id"]) or int(r["block_id"]) != int(l["block_id"]) + 1:
            adjacency_breaks += 1
    continuity_penalty = 0.01 * (segment_count - 1) + 0.005 * adjacency_breaks
    tie = (stable_int(f"{tie_key}|{','.join(map(str, selected))}") % 10**9) / 10**9
    return (
        float(missing_class_count),
        float(missing_val_positive),
        float(missing_train_positive),
        ratio_error,
        fraction_error,
        continuity_penalty,
        tie,
    )


def _legacy_candidate_score(
    blocks: pd.DataFrame,
    selected: tuple[int, ...],
    total_epochs: int,
    total_positive: int,
    val_fraction: float,
    tie_key: str,
) -> tuple[float, ...]:
    """Retain the historical label-first score for static baseline reproduction."""

    chosen = blocks.iloc[list(selected)]
    val_epochs = int(chosen["epoch_count"].sum())
    val_positive = int(chosen["positive_epochs"].sum())
    train_positive = total_positive - val_positive
    val_ratio = val_positive / max(val_epochs, 1)
    total_ratio = total_positive / max(total_epochs, 1)
    has_both_classes = 0 < total_positive < total_epochs
    missing_positive = int(has_both_classes and val_positive == 0)
    missing_negative = int(has_both_classes and val_positive == val_epochs)
    class_penalty = 3.0 * missing_positive + 1.0 * missing_negative
    train_empty_penalty = 0.5 if total_positive > 0 and train_positive == 0 else 0.0
    fraction_error = abs(val_epochs / max(total_epochs, 1) - val_fraction)
    ratio_error = abs(val_ratio - total_ratio)
    segment_count = len(set(int(blocks.iloc[index]["segment_id"]) for index in selected))
    adjacency_breaks = 0
    for left, right in zip(selected, selected[1:]):
        left_block = blocks.iloc[left]
        right_block = blocks.iloc[right]
        if (
            int(left_block["segment_id"]) != int(right_block["segment_id"])
            or int(right_block["block_id"]) != int(left_block["block_id"]) + 1
        ):
            adjacency_breaks += 1
    continuity_penalty = 0.01 * (segment_count - 1) + 0.005 * adjacency_breaks
    tie = (stable_int(f"{tie_key}|{','.join(map(str, selected))}") % 10**9) / 10**9
    return (
        class_penalty + train_empty_penalty,
        ratio_error,
        fraction_error,
        continuity_penalty,
        tie,
    )


def choose_validation_blocks_legacy(
    blocks: pd.DataFrame,
    val_fraction: float,
    tie_key: str,
) -> set[int]:
    """Reproduce the historical fixed-block-count, label-first selection."""

    if len(blocks) <= 1:
        return set()
    blocks = blocks.reset_index(drop=True)
    total_epochs = int(blocks["epoch_count"].sum())
    total_positive = int(blocks["positive_epochs"].sum())
    target_count = max(1, int(round(len(blocks) * val_fraction)))
    target_count = min(target_count, len(blocks) - 1)
    if len(blocks) <= 24:
        candidates: Iterable[tuple[int, ...]] = itertools.combinations(
            range(len(blocks)), target_count
        )
    else:
        order = sorted(
            range(len(blocks)),
            key=lambda index: stable_int(f"{tie_key}|block|{index}"),
        )
        candidates = [tuple(order[:target_count])]
    best: tuple[float, ...] | None = None
    best_selected: tuple[int, ...] | None = None
    for selected in candidates:
        selected = tuple(sorted(selected))
        score = _legacy_candidate_score(
            blocks,
            selected,
            total_epochs,
            total_positive,
            val_fraction,
            tie_key,
        )
        if best is None or score < best:
            best, best_selected = score, selected
    assert best_selected is not None
    return set(best_selected)


def _validation_subset_states(
    blocks: pd.DataFrame,
    tie_key: str,
) -> dict[tuple[int, int], tuple[tuple[int, ...], int]]:
    """Enumerate aggregate (epoch, positive) states without enumerating 2^N subsets.

    Blocks in this protocol usually contain the same number of epochs, so many
    subsets collapse to the same aggregate state.  One deterministic,
    continuity-favouring representative is retained for each state.  The
    primary size, class-coverage, and positive-ratio objectives depend only on
    the aggregate state and therefore remain exact.
    """

    epoch_counts = blocks["epoch_count"].to_numpy(dtype=np.int64)
    positive_counts = blocks["positive_epochs"].to_numpy(dtype=np.int64)
    segment_ids = blocks["segment_id"].to_numpy(dtype=np.int64)
    block_ids = blocks["block_id"].to_numpy(dtype=np.int64)
    # Value: (selected indices, continuity units).  One unit is 0.005 in the
    # final score; crossing a segment contributes three units (one adjacency
    # break plus the two-unit segment penalty).
    states: dict[tuple[int, int], tuple[tuple[int, ...], int]] = {(0, 0): ((), 0)}
    for index, (epoch_count, positive_count) in enumerate(
        zip(epoch_counts.tolist(), positive_counts.tolist())
    ):
        additions: dict[tuple[int, int], tuple[tuple[int, ...], int]] = {}
        for (epochs_so_far, positives_so_far), (selected, continuity_units) in states.items():
            next_selected = (*selected, index)
            next_units = continuity_units
            if selected:
                previous = selected[-1]
                if segment_ids[previous] != segment_ids[index]:
                    next_units += 3
                elif block_ids[index] != block_ids[previous] + 1:
                    next_units += 1
            aggregate = (
                int(epochs_so_far + epoch_count),
                int(positives_so_far + positive_count),
            )
            candidate = (next_selected, next_units)
            incumbent = additions.get(aggregate, states.get(aggregate))
            if incumbent is None:
                additions[aggregate] = candidate
                continue
            candidate_key = (
                next_units,
                stable_int(f"{tie_key}|representative|{','.join(map(str, next_selected))}"),
            )
            incumbent_key = (
                incumbent[1],
                stable_int(f"{tie_key}|representative|{','.join(map(str, incumbent[0]))}"),
            )
            if candidate_key < incumbent_key:
                additions[aggregate] = candidate
        states.update(additions)
        if len(states) > MAX_VALIDATION_DP_STATES:
            raise RuntimeError(
                "Validation subset search exceeded "
                f"{MAX_VALIDATION_DP_STATES:,} aggregate states for {tie_key!r}; "
                "increase --block-epochs or narrow the source recording."
            )
    return states


def choose_validation_blocks(
    blocks: pd.DataFrame,
    val_fraction: float,
    tie_key: str,
    size_tolerance: float = DEFAULT_VAL_SIZE_TOLERANCE,
) -> set[int]:
    """Choose deterministic validation blocks with epoch fraction as stage one.

    First find the best attainable validation epoch count.  Only candidates
    within ``size_tolerance`` of that optimum (measured against all recording
    epochs) may compete on positive/negative support and positive ratio.  This
    makes label quality a local tie-break rather than a reason to select a
    grossly oversized validation partition.
    """

    if len(blocks) <= 1:
        return set()
    if size_tolerance < 0.0:
        raise ValueError("Validation size tolerance must be non-negative")
    blocks = blocks.reset_index(drop=True)
    total_epochs = int(blocks["epoch_count"].sum())
    total_positive = int(blocks["positive_epochs"].sum())
    target_epochs = total_epochs * val_fraction
    states = _validation_subset_states(blocks, tie_key)
    candidates = [
        (epoch_count, positive_count, representative[0])
        for (epoch_count, positive_count), representative in states.items()
        if 0 < epoch_count < total_epochs
    ]
    if not candidates:
        return set()
    minimum_size_error = min(abs(epoch_count - target_epochs) for epoch_count, _, _ in candidates)
    tolerance_epochs = max(1, int(round(total_epochs * size_tolerance)))
    candidates = [
        (epoch_count, positive_count, selected)
        for epoch_count, positive_count, selected in candidates
        if abs(epoch_count - target_epochs) <= minimum_size_error + tolerance_epochs
    ]
    best: tuple[float, ...] | None = None
    best_selected: tuple[int, ...] | None = None
    for _, _, selected in candidates:
        score = _candidate_score(blocks, selected, total_epochs, total_positive, val_fraction, tie_key)
        if best is None or score < best:
            best, best_selected = score, selected
    assert best_selected is not None
    return set(best_selected)


def apply_guards(epoch_table: pd.DataFrame, time_steps: int) -> pd.DataFrame:
    """Mark symmetric W-1 epoch embargoes around train/validation changes."""

    result = epoch_table.copy()
    result["split"] = result["raw_split"]
    result["is_guard"] = False
    result["guard_reason"] = ""
    radius = max(time_steps - 1, 0)
    if radius == 0:
        return result
    for _, group_index in result.groupby("segment_key", sort=False).groups.items():
        indices = np.asarray(sorted(group_index), dtype=np.int64)
        raw = result.loc[indices, "raw_split"].to_numpy(dtype=object)
        for position in range(1, len(indices)):
            if raw[position] == raw[position - 1]:
                continue
            left = max(0, position - radius)
            right = min(len(indices), position + radius)
            guarded = indices[left:right]
            result.loc[guarded, "split"] = "guard"
            result.loc[guarded, "is_guard"] = True
            result.loc[guarded, "guard_reason"] = f"boundary_w{time_steps}"
    return result


def scenario_validation_audit(
    all_epochs: pd.DataFrame,
    recordings: pd.DataFrame,
    test_ids: set[int],
    fold: int,
) -> pd.DataFrame:
    """Summarise usable validation support for every development Scenario."""

    development_recordings = recordings.loc[
        ~recordings["recording_id"].isin(test_ids), ["recording_id", "Scenario"]
    ].copy()
    expected_scenarios = sorted(development_recordings["Scenario"].astype(str).unique())
    rows: list[dict[str, object]] = []
    for scenario in expected_scenarios:
        recording_ids = set(
            development_recordings.loc[
                development_recordings["Scenario"].astype(str).eq(scenario), "recording_id"
            ].astype(int)
        )
        scenario_epochs = all_epochs.loc[all_epochs["recording_id"].isin(recording_ids)]
        raw_train = scenario_epochs["raw_split"].eq("train")
        raw_val = scenario_epochs["raw_split"].eq("val")
        train = scenario_epochs["split"].eq("train")
        val = scenario_epochs["split"].eq("val")
        guard = scenario_epochs["split"].eq("guard")
        epoch_count = int(len(scenario_epochs))
        val_epochs = int(val.sum())
        val_positive = int(scenario_epochs.loc[val, "positive_epoch"].sum())
        rows.append(
            {
                "fold": int(fold),
                "Scenario": scenario,
                "development_recordings": int(len(recording_ids)),
                "recordings_with_raw_val": int(scenario_epochs.loc[raw_val, "recording_id"].nunique()),
                "recordings_with_final_val": int(scenario_epochs.loc[val, "recording_id"].nunique()),
                "epoch_count": epoch_count,
                "raw_train_epochs": int(raw_train.sum()),
                "raw_val_epochs": int(raw_val.sum()),
                "raw_val_fraction": float(raw_val.sum() / max(epoch_count, 1)),
                "train_epochs": int(train.sum()),
                "val_epochs": val_epochs,
                "guard_epochs": int(guard.sum()),
                "train_fraction": float(train.sum() / max(epoch_count, 1)),
                "val_fraction": float(val_epochs / max(epoch_count, 1)),
                "guard_fraction": float(guard.sum() / max(epoch_count, 1)),
                "val_positive_epochs": val_positive,
                "val_negative_epochs": int(val_epochs - val_positive),
                "val_has_positive": bool(val_positive > 0),
                "val_has_negative": bool(val_epochs - val_positive > 0),
                "validation_covered": bool(val_epochs > 0),
            }
        )
    return pd.DataFrame(rows)


def build_fold(
    epochs: pd.DataFrame,
    recordings: pd.DataFrame,
    fold: int,
    test_recording_ids: Iterable[int],
    block_epochs: int,
    val_fraction: float,
    time_steps: int,
    segment_gap_seconds: float,
    output_dir: Path,
    strict_validation: bool = False,
    val_size_tolerance: float = DEFAULT_VAL_SIZE_TOLERANCE,
) -> dict[str, int | float | str]:
    """Build one outer-test fold and write all manifest levels."""

    test_ids = {int(value) for value in test_recording_ids}
    if not test_ids:
        raise ValueError(f"Fold {fold}: expected at least one outer test recording")
    known_ids = set(recordings["recording_id"].astype(int).tolist())
    unknown_ids = sorted(test_ids.difference(known_ids))
    if unknown_ids:
        raise ValueError(f"Fold {fold}: unknown outer test recording ids {unknown_ids}")
    fold_epochs: list[pd.DataFrame] = []
    block_rows: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    for recording_id, rec in recordings.set_index("recording_id").iterrows():
        rec_epochs = epochs.loc[epochs["recording_key"] == rec["recording_key"]].copy()
        if rec_epochs.empty:
            raise ValueError(f"No epoch rows found for recording {recording_id}")
        rec_epochs = _split_segments(rec_epochs, segment_gap_seconds, block_epochs)
        rec_epochs["recording_id"] = int(recording_id)
        rec_epochs["segment_key"] = rec_epochs["segment_id"].map(
            lambda value: f"r{int(recording_id):02d}_s{int(value):02d}"
        )
        blocks = _block_table(rec_epochs)
        blocks["recording_id"] = int(recording_id)
        blocks["block_uid"] = blocks.apply(
            lambda row: f"r{int(recording_id):02d}_s{int(row.segment_id):02d}_b{int(row.block_id):03d}",
            axis=1,
        )
        block_uid_lookup = blocks.set_index(["segment_id", "block_id"])["block_uid"]
        rec_epochs["block_uid"] = [
            str(block_uid_lookup.loc[(int(segment), int(block))])
            for segment, block in zip(rec_epochs["segment_id"], rec_epochs["block_id"])
        ]
        is_outer_test = int(recording_id) in test_ids
        if is_outer_test:
            blocks["raw_split"] = "test"
            rec_epochs["raw_split"] = "test"
        else:
            recording_name = " / ".join(str(rec[column]) for column in RECORDING_KEYS)
            if strict_validation and len(blocks) <= 1:
                raise ValueError(
                    f"Fold {fold}: development recording {recording_name!r} has only "
                    f"{len(blocks)} time block; strict validation requires at least two."
                )
            if strict_validation:
                validation_indices = choose_validation_blocks(
                    blocks,
                    val_fraction,
                    str(rec["recording_key"]),
                    size_tolerance=val_size_tolerance,
                )
            else:
                validation_indices = choose_validation_blocks_legacy(
                    blocks,
                    val_fraction,
                    str(rec["recording_key"]),
                )
            if strict_validation and not validation_indices:
                raise ValueError(
                    f"Fold {fold}: development recording {recording_name!r} has no "
                    "validation block under the strict protocol."
                )
            blocks["raw_split"] = "train"
            if validation_indices:
                blocks.loc[sorted(validation_indices), "raw_split"] = "val"
            block_lookup = blocks.set_index(["segment_id", "block_id"])["raw_split"]
            rec_epochs["raw_split"] = [
                str(block_lookup.loc[(int(segment), int(block))])
                for segment, block in zip(rec_epochs["segment_id"], rec_epochs["block_id"])
            ]
        fold_epochs.append(rec_epochs)
        block_rows.append(blocks)
        summaries.append(
            {
                "fold": fold,
                "recording_id": int(recording_id),
                "Environment": rec["Environment"],
                "Scenario": rec["Scenario"],
                "Session": rec["Session"],
                "outer_test": is_outer_test,
                "epoch_count": int(len(rec_epochs)),
                "positive_epochs": int(rec_epochs["positive_epoch"].sum()),
                "block_count": int(len(blocks)),
                "val_block_count": int((blocks["raw_split"] == "val").sum()),
            }
        )

    all_epochs = pd.concat(fold_epochs, ignore_index=True)
    all_epochs = apply_guards(all_epochs, time_steps)
    all_blocks = pd.concat(block_rows, ignore_index=True)
    for summary in summaries:
        recording_rows = all_epochs[all_epochs["recording_id"] == int(summary["recording_id"])]
        summary.update(
            {
                "raw_train_epochs": int((recording_rows["raw_split"] == "train").sum()),
                "raw_val_epochs": int((recording_rows["raw_split"] == "val").sum()),
                "raw_val_fraction": float(
                    (recording_rows["raw_split"] == "val").sum() / max(len(recording_rows), 1)
                ),
                "train_epochs": int((recording_rows["split"] == "train").sum()),
                "val_epochs": int((recording_rows["split"] == "val").sum()),
                "guard_epochs": int((recording_rows["split"] == "guard").sum()),
                "test_epochs": int((recording_rows["split"] == "test").sum()),
                "train_positive_epochs": int(
                    recording_rows.loc[recording_rows["split"] == "train", "positive_epoch"].sum()
                ),
                "val_positive_epochs": int(
                    recording_rows.loc[recording_rows["split"] == "val", "positive_epoch"].sum()
                ),
                "val_negative_epochs": int(
                    ((recording_rows["split"] == "val") & (recording_rows["positive_epoch"] == 0)).sum()
                ),
                "test_positive_epochs": int(
                    recording_rows.loc[recording_rows["split"] == "test", "positive_epoch"].sum()
                ),
            }
        )
    if strict_validation:
        unusable_recordings = [
            " / ".join(str(summary[column]) for column in RECORDING_KEYS)
            for summary in summaries
            if not bool(summary["outer_test"])
            and (
                int(summary["val_block_count"]) == 0
                or int(summary["raw_val_epochs"]) == 0
                or int(summary["val_epochs"]) == 0
            )
        ]
        if unusable_recordings:
            raise ValueError(
                f"Fold {fold}: strict validation left development recordings without a "
                "usable validation block after guards: "
                + "; ".join(unusable_recordings)
            )

    scenario_audit = scenario_validation_audit(all_epochs, recordings, test_ids, fold)
    covered_scenarios = sorted(
        scenario_audit.loc[scenario_audit["validation_covered"], "Scenario"].astype(str).tolist()
    )
    missing_scenarios = sorted(
        scenario_audit.loc[~scenario_audit["validation_covered"], "Scenario"].astype(str).tolist()
    )
    if strict_validation and missing_scenarios:
        raise ValueError(
            f"Fold {fold}: development Scenarios missing from final validation: {missing_scenarios}"
        )
    # A block's final split is ``guard`` only when every epoch is guarded;
    # epoch_split_manifest.csv is authoritative for mixed boundary blocks.
    block_split = (
        all_epochs.groupby(["recording_id", "segment_id", "block_id"], sort=False)["split"]
        .agg(lambda values: "guard" if set(values) == {"guard"} else str(values.iloc[0]))
        .rename("epoch_split")
        .reset_index()
    )
    all_blocks = all_blocks.merge(block_split, on=["recording_id", "segment_id", "block_id"], how="left")

    fold_dir = output_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    recording_manifest = recordings.copy()
    recording_manifest["split"] = np.where(
        recording_manifest["recording_id"].isin(test_ids), "test", "development"
    )
    recording_manifest["outer_test"] = recording_manifest["recording_id"].isin(test_ids)
    recording_manifest.drop(columns=["recording_key"]).to_csv(
        fold_dir / "recording_split_manifest.csv", index=False, encoding="utf-8-sig"
    )
    block_columns = [
        "fold", "recording_id", *RECORDING_KEYS, "segment_id", "block_id", "block_uid",
        "epoch_count", "canonical_start_ms", "canonical_end_ms", "row_count", "positive_rows",
        "positive_epochs", "positive_ratio", "device_count", "raw_split", "epoch_split",
    ]
    all_blocks.insert(0, "fold", fold)
    # ``all_blocks`` already carries recording keys through ``_block_table``;
    # merge only as a defensive fallback for future table changes.
    missing_recording_columns = [column for column in RECORDING_KEYS if column not in all_blocks.columns]
    if missing_recording_columns:
        all_blocks = all_blocks.merge(
            recordings[["recording_id", *RECORDING_KEYS]], on="recording_id", how="left", validate="many_to_one"
        )
    all_blocks[block_columns].to_csv(fold_dir / "time_block_manifest.csv", index=False, encoding="utf-8-sig")
    epoch_columns = [
        "fold", "recording_id", *RECORDING_KEYS, "segment_id", "segment_key", "block_id", "block_uid",
        "epoch_index", "segment_epoch_index", "canonical_epoch_ms", "row_count", "positive_rows",
        "positive_epoch", "device_count", "raw_split", "split", "is_guard", "guard_reason",
    ]
    all_epochs.insert(0, "fold", fold)
    missing_recording_columns = [column for column in RECORDING_KEYS if column not in all_epochs.columns]
    if missing_recording_columns:
        all_epochs = all_epochs.merge(
            recordings[["recording_id", *RECORDING_KEYS]], on="recording_id", how="left", validate="many_to_one"
        )
    all_epochs[epoch_columns].to_csv(fold_dir / "epoch_split_manifest.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summaries).to_csv(fold_dir / "recording_summary.csv", index=False, encoding="utf-8-sig")
    scenario_audit.to_csv(
        fold_dir / "scenario_validation_summary.csv", index=False, encoding="utf-8-sig"
    )

    # Integrity checks are intentionally strict: they catch accidental row
    # duplication and ensure the outer test recording never shares an epoch.
    if all_epochs.duplicated(["recording_id", "canonical_epoch_ms"]).any():
        raise RuntimeError(f"Fold {fold}: duplicate recording/canonical epoch rows")
    test_rows = all_epochs[all_epochs["recording_id"].isin(test_ids)]
    if not test_rows["split"].eq("test").all():
        raise RuntimeError(f"Fold {fold}: outer test recordings were not kept intact")
    development_rows = all_epochs[~all_epochs["recording_id"].isin(test_ids)]
    if development_rows["split"].eq("test").any():
        raise RuntimeError(f"Fold {fold}: development recording leaked into outer test")
    if set(all_epochs["split"].unique()) - {"train", "val", "test", "guard"}:
        raise RuntimeError(f"Fold {fold}: unexpected split values")

    development = all_epochs.loc[~all_epochs["recording_id"].isin(test_ids)]
    development_epochs = int(len(development))
    raw_train_epochs = int(development["raw_split"].eq("train").sum())
    raw_val_epochs = int(development["raw_split"].eq("val").sum())
    train_epochs = int(development["split"].eq("train").sum())
    val_epochs = int(development["split"].eq("val").sum())
    guard_epochs = int(development["split"].eq("guard").sum())
    return {
        "fold": fold,
        "test_recording_count": len(test_ids),
        "test_recording_ids": ";".join(map(str, sorted(test_ids))),
        "epoch_count": int(len(all_epochs)),
        "development_epochs": development_epochs,
        "raw_train_epochs": raw_train_epochs,
        "raw_val_epochs": raw_val_epochs,
        "raw_val_fraction_development": float(raw_val_epochs / max(development_epochs, 1)),
        "train_epochs": train_epochs,
        "val_epochs": val_epochs,
        "guard_epochs": guard_epochs,
        "train_fraction_development": float(train_epochs / max(development_epochs, 1)),
        "val_fraction_development": float(val_epochs / max(development_epochs, 1)),
        "guard_fraction_development": float(guard_epochs / max(development_epochs, 1)),
        "test_epochs": int((all_epochs["split"] == "test").sum()),
        "development_scenario_count": int(len(scenario_audit)),
        "validation_scenario_count": int(len(covered_scenarios)),
        "validation_scenarios": ";".join(covered_scenarios),
        "missing_validation_scenarios": ";".join(missing_scenarios),
        "train_positive_epochs": int(
            all_epochs.loc[all_epochs["split"] == "train", "positive_epoch"].sum()
        ),
        "val_positive_epochs": int(all_epochs.loc[all_epochs["split"] == "val", "positive_epoch"].sum()),
        "val_negative_epochs": int(
            ((all_epochs["split"] == "val") & (all_epochs["positive_epoch"] == 0)).sum()
        ),
        "test_positive_epochs": int(
            all_epochs.loc[all_epochs["split"] == "test", "positive_epoch"].sum()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=ROOT / "output" / "processed_gnss_data.csv")
    parser.add_argument(
        "--source-recording-manifest",
        type=Path,
        default=ROOT / "docs" / "protocols" / "static_session_cv_4fold" / "fold_1" / "recording_split_manifest.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output" / "protocols" / "static_time_block_outer_v1",
    )
    parser.add_argument("--time-steps", type=int, default=DEFAULT_TIME_STEPS)
    parser.add_argument("--block-epochs", type=int, default=DEFAULT_BLOCK_EPOCHS)
    parser.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    parser.add_argument(
        "--val-size-tolerance",
        type=float,
        default=DEFAULT_VAL_SIZE_TOLERANCE,
        help=(
            "maximum absolute recording-level epoch fraction that label balance may trade "
            "beyond the best attainable validation size (default: 0.02)"
        ),
    )
    parser.add_argument("--segment-gap-seconds", type=float, default=DEFAULT_SEGMENT_GAP_SECONDS)
    parser.add_argument(
        "--data-scope",
        choices=DATA_SCOPES,
        default="static",
        help="static keeps reviewed st_ recordings; mixed keeps reviewed st_ and dy_ recordings",
    )
    parser.add_argument(
        "--strict-validation",
        action="store_true",
        help=(
            "enable epoch-fraction-first v2 selection and fail if any development recording "
            "has fewer than two blocks or no usable final validation epochs, or if any "
            "development Scenario is absent from validation"
        ),
    )
    args = parser.parse_args()
    if args.time_steps < 2:
        parser.error("--time-steps must be at least 2")
    if args.block_epochs < args.time_steps:
        parser.error("--block-epochs must be >= --time-steps")
    if not 0.0 < args.val_fraction < 0.5:
        parser.error("--val-fraction must be between 0 and 0.5")
    if not 0.0 <= args.val_size_tolerance < 0.5:
        parser.error("--val-size-tolerance must be between 0 (inclusive) and 0.5")
    if args.segment_gap_seconds <= 0:
        parser.error("--segment-gap-seconds must be positive")
    if not args.csv.exists():
        raise FileNotFoundError(args.csv)
    if not args.source_recording_manifest.exists():
        raise FileNotFoundError(args.source_recording_manifest)

    input_files = {
        "processed_csv": {
            "path": str(args.csv.resolve()),
            "sha256": file_sha256(args.csv),
        },
        "outer_recording_manifest": {
            "path": str(args.source_recording_manifest.resolve()),
            "sha256": file_sha256(args.source_recording_manifest),
        },
    }

    recordings = read_recordings(args.source_recording_manifest, args.data_scope)
    epochs = load_epoch_table(args.csv, recordings, args.data_scope)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if "test_fold" in recordings.columns:
        fold_values = sorted(recordings["test_fold"].astype(int).unique().tolist())
        if fold_values != list(range(1, len(fold_values) + 1)):
            raise ValueError(f"test_fold identifiers must be contiguous from 1; found {fold_values}")
        fold_plan = [
            (
                fold,
                recordings.loc[recordings["test_fold"].eq(fold), "recording_id"].astype(int).tolist(),
            )
            for fold in fold_values
        ]
    else:
        fold_plan = [
            (fold, [int(recording_id)])
            for fold, recording_id in enumerate(recordings["recording_id"].tolist(), start=1)
        ]

    fold_assignment_rows: list[dict[str, object]] = []
    fold_summaries: list[dict[str, object]] = []
    for fold, test_recording_ids in fold_plan:
        summary = build_fold(
            epochs=epochs,
            recordings=recordings,
            fold=fold,
            test_recording_ids=test_recording_ids,
            block_epochs=int(args.block_epochs),
            val_fraction=float(args.val_fraction),
            time_steps=int(args.time_steps),
            segment_gap_seconds=float(args.segment_gap_seconds),
            output_dir=args.output_dir,
            strict_validation=bool(args.strict_validation),
            val_size_tolerance=float(args.val_size_tolerance),
        )
        fold_summaries.append(summary)
        for test_recording_id in test_recording_ids:
            test_recording = recordings.loc[
                recordings["recording_id"] == test_recording_id
            ].iloc[0]
            fold_assignment_rows.append(
                {
                    "fold": fold,
                    "role": "test",
                    "recording_id": int(test_recording_id),
                    **{column: test_recording[column] for column in RECORDING_KEYS},
                }
            )
    pd.DataFrame(fold_assignment_rows).to_csv(
        args.output_dir / "fold_assignment.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(fold_summaries).to_csv(args.output_dir / "fold_summary.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "protocol": args.output_dir.resolve().name,
        "data_scope": args.data_scope,
        "recordings": int(len(recordings)),
        "outer_folds": int(len(fold_plan)),
        "outer_test_recordings_per_fold": [len(test_ids) for _, test_ids in fold_plan],
        "inputs": input_files,
        "time_steps": int(args.time_steps),
        "block_epochs": int(args.block_epochs),
        "validation_fraction": float(args.val_fraction),
        "validation_size_tolerance": float(args.val_size_tolerance),
        "validation_selection": (
            (
                "minimise validation epoch-count error; within best error plus tolerance, "
                "optimise train/validation class coverage and positive-epoch ratio"
            )
            if args.strict_validation
            else "legacy fixed-block-count label-first selection"
        ),
        "strict_validation": bool(args.strict_validation),
        "segment_gap_seconds": float(args.segment_gap_seconds),
        "canonical_epoch": "floor(utcTimeMillis / 1000) * 1000",
        "guard_epochs_each_side": int(args.time_steps - 1),
        "split_values": ["train", "val", "guard", "test"],
        "epoch_manifest_key": ["Environment", "Scenario", "Session", "canonical_epoch_ms"],
        "actual_split_by_fold": [
            {
                key: summary[key]
                for key in (
                    "fold",
                    "development_epochs",
                    "raw_train_epochs",
                    "raw_val_epochs",
                    "raw_val_fraction_development",
                    "train_epochs",
                    "val_epochs",
                    "guard_epochs",
                    "train_fraction_development",
                    "val_fraction_development",
                    "guard_fraction_development",
                )
            }
            for summary in fold_summaries
        ],
        "validation_scenario_coverage_by_fold": [
            {
                key: summary[key]
                for key in (
                    "fold",
                    "development_scenario_count",
                    "validation_scenario_count",
                    "validation_scenarios",
                    "missing_validation_scenarios",
                )
            }
            for summary in fold_summaries
        ],
        "scenario_audit_file": "fold_N/scenario_validation_summary.csv",
    }
    (args.output_dir / "protocol_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Generated {args.data_scope} outer-Session/time-block manifests")
    print(f"  recordings={len(recordings)} folds={len(fold_plan)}")
    print(f"  canonical epoch=floor(utcTimeMillis/{DEFAULT_CANONICAL_MS})*{DEFAULT_CANONICAL_MS}ms")
    print(
        f"  block_epochs={args.block_epochs} val_fraction={args.val_fraction} "
        f"size_tolerance={args.val_size_tolerance} guard={args.time_steps - 1} "
        f"strict={args.strict_validation}"
    )
    print(pd.DataFrame(fold_summaries).to_string(index=False))


if __name__ == "__main__":
    main()

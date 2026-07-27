"""Generate an outer-Session / inner-time-block signal CV protocol.

The protocol is deliberately stricter than a random window split:

* one or more complete reviewed recordings are held out as ``test`` per fold;
* the other recordings remain in the development pool;
* development recordings are divided into deterministic contiguous blocks of
  canonical epochs (the default block size is 256);
* a deterministic subset of blocks is assigned to validation; strict v2 mode
  controls the validation epoch fraction before using label balance as a
  tie-break, while the historical default remains reproducible;
* the explicit reviewed-state-stratified mode instead pools state-pure blocks
  across development Sessions, deriving clean/attack context directly from
  reviewed preprocessing.yml TOW intervals rather than row-level labels;
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
import yaml


ROOT = Path(__file__).resolve().parents[1]
RECORDING_KEYS = ["Environment", "Scenario", "Session"]
STATIC_PREFIX = "st_"
DYNAMIC_PREFIX = "dy_"
DATA_SCOPES = ("static", "mixed")
VALIDATION_MODES = ("recording-local", "reviewed-state-stratified")
DEFAULT_VALIDATION_MODE = "recording-local"
STATE_VALUES = ("clean", "attack")
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


def load_reviewed_attack_intervals(
    path: Path,
    recordings: pd.DataFrame,
) -> dict[tuple[str, str, str], tuple[tuple[float, float], ...]]:
    """Resolve reviewed Session attack-context intervals from preprocessing.yml.

    These intervals describe event time, independently of target frequency.
    For example, every L1 and L5 observation during a reviewed ``*_L5`` event
    belongs to the same attack context even though formal row-level ``Label``
    remains positive only for L5 observations.
    """

    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    sessions = config.get("labeling", {}).get("session_spoofing_tow_intervals", {})
    if not isinstance(sessions, dict):
        raise ValueError(
            "labeling.session_spoofing_tow_intervals must be a mapping in "
            f"{path}"
        )

    result: dict[tuple[str, str, str], tuple[tuple[float, float], ...]] = {}
    for values in recordings[RECORDING_KEYS].astype(str).itertuples(index=False, name=None):
        environment, scenario, session = values
        entry = sessions.get(environment, {}).get(scenario, {}).get(session)
        identity = " / ".join(values)
        if not isinstance(entry, dict):
            raise ValueError(f"Missing Session label entry for {identity!r} in {path}")
        if str(entry.get("status", "needs_review")).strip().lower() != "reviewed":
            raise ValueError(f"Session {identity!r} is not reviewed in {path}")
        raw_intervals = entry.get("intervals", []) or []
        if not isinstance(raw_intervals, list):
            raise ValueError(f"Session {identity!r} intervals must be a list")
        parsed: list[tuple[float, float]] = []
        for raw in raw_intervals:
            if not isinstance(raw, (list, tuple)) or len(raw) != 2:
                raise ValueError(f"Session {identity!r} contains invalid interval {raw!r}")
            start, end = float(raw[0]), float(raw[1])
            if not np.isfinite(start) or not np.isfinite(end) or end < start:
                raise ValueError(f"Session {identity!r} contains invalid interval {raw!r}")
            parsed.append((start, end))
        parsed.sort()
        for previous, current in zip(parsed, parsed[1:]):
            if current[0] <= previous[1]:
                raise ValueError(
                    f"Session {identity!r} contains overlapping reviewed intervals: "
                    f"{previous!r}, {current!r}"
                )
        result[values] = tuple(parsed)
    return result


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
    attack_intervals: dict[
        tuple[str, str, str], tuple[tuple[float, float], ...]
    ] | None = None,
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
    if attack_intervals is not None:
        tow = pd.to_numeric(chunk["TOW"], errors="coerce")
        if tow.isna().any() or not np.isfinite(tow.to_numpy(dtype=np.float64)).all():
            raise ValueError("Reviewed rows contain missing or non-finite TOW values.")
        chunk["_tow"] = tow.astype(np.float64)
        chunk["_attack_context_row"] = False
        for recording_key, indices in chunk.groupby("_recording_key", sort=False).groups.items():
            key = tuple(str(value) for value in recording_key)
            if key not in attack_intervals:
                raise ValueError(f"No reviewed attack intervals resolved for recording {key!r}")
            values = chunk.loc[indices, "_tow"].to_numpy(dtype=np.float64)
            inside = np.zeros(len(values), dtype=bool)
            for start, end in attack_intervals[key]:
                inside |= (values >= start) & (values <= end)
            chunk.loc[indices, "_attack_context_row"] = inside
    group_cols = [*RECORDING_KEYS, "canonical_epoch_ms"]
    aggregations: dict[str, tuple[str, str | object]] = {
        "row_count": ("Label", "size"),
        "positive_rows": ("Label", "sum"),
        "positive_epoch": ("Label", "max"),
        "device_names": (
            "DeviceName",
            lambda values: "\x1f".join(sorted(set(values.astype(str)))),
        ),
    }
    if attack_intervals is not None:
        aggregations.update(
            {
                "tow_min": ("_tow", "min"),
                "tow_max": ("_tow", "max"),
                "attack_context_rows": ("_attack_context_row", "sum"),
                "attack_context_epoch": ("_attack_context_row", "max"),
            }
        )
    grouped = chunk.groupby(group_cols, sort=False, observed=True).agg(**aggregations).reset_index()
    return grouped


def load_epoch_table(
    csv_path: Path,
    recordings: pd.DataFrame,
    data_scope: str = "static",
    chunksize: int = 500_000,
    attack_intervals: dict[
        tuple[str, str, str], tuple[tuple[float, float], ...]
    ] | None = None,
) -> pd.DataFrame:
    """Read the large processed CSV without materialising all signal rows."""

    usecols = [
        *RECORDING_KEYS,
        "DeviceName",
        "utcTimeMillis",
        "Label",
        "LabelStatus",
    ]
    if attack_intervals is not None:
        usecols.append("TOW")
    target_keys = set(map(tuple, recordings[RECORDING_KEYS].astype(str).to_numpy()))
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize):
        part = _aggregate_chunk(chunk, target_keys, data_scope, attack_intervals)
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

    aggregations = {
        "row_count": ("row_count", "sum"),
        "positive_rows": ("positive_rows", "sum"),
        "positive_epoch": ("positive_epoch", "max"),
        "device_names": ("device_names", merge_devices),
    }
    if attack_intervals is not None:
        aggregations.update(
            {
                "tow_min": ("tow_min", "min"),
                "tow_max": ("tow_max", "max"),
                "attack_context_rows": ("attack_context_rows", "sum"),
                "attack_context_epoch": ("attack_context_epoch", "max"),
            }
        )
    epochs = raw.groupby(group_cols, sort=False, observed=True).agg(**aggregations).reset_index()
    epochs["device_count"] = epochs["device_names"].map(
        lambda value: 0 if not value else len(str(value).split("\x1f"))
    ).astype("int16")
    epochs = epochs.drop(columns="device_names")
    if attack_intervals is not None:
        epochs["attack_context_rows"] = epochs["attack_context_rows"].astype(np.int32)
        epochs["attack_time_state"] = np.where(
            epochs["attack_context_epoch"].astype(bool), "attack", "clean"
        )
        epochs["mixed_attack_context_epoch"] = (
            epochs["attack_context_rows"].gt(0)
            & epochs["attack_context_rows"].lt(epochs["row_count"])
        )
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


def _split_state_blocks(
    epochs: pd.DataFrame,
    gap_seconds: float,
    block_epochs: int,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    time_steps: int = DEFAULT_TIME_STEPS,
) -> pd.DataFrame:
    """Create state-pure, balanced contiguous blocks inside physical segments.

    A short attack run is retained as one block.  Longer state runs use
    balanced blocks instead of a full block plus a tiny tail, which prevents a
    mechanically created tail from losing every W-step validation window to
    the boundary guard.
    """

    if "attack_time_state" not in epochs.columns:
        raise ValueError("State-stratified validation requires attack_time_state epochs")
    result = _split_segments(epochs, gap_seconds, block_epochs)
    states = set(result["attack_time_state"].astype(str).unique())
    if states - set(STATE_VALUES):
        raise ValueError(f"Unexpected attack-time states: {sorted(states)}")
    result["state_run_id"] = np.int16(-1)
    result["state_run_epoch_index"] = np.int32(-1)
    result["state_run_epoch_count"] = np.int32(-1)
    result["state_block_id"] = np.int16(-1)
    result["block_id"] = np.int16(-1)
    result["splittable_attack_run"] = False
    result["short_attack_atom"] = False
    for _, segment_indices in result.groupby("segment_id", sort=False).groups.items():
        ordered = np.asarray(sorted(segment_indices), dtype=np.int64)
        segment_states = result.loc[ordered, "attack_time_state"].astype(str).to_numpy()
        run_starts = np.r_[True, segment_states[1:] != segment_states[:-1]]
        run_ids = np.cumsum(run_starts).astype(np.int16) - 1
        result.loc[ordered, "state_run_id"] = run_ids
        next_block_id = 0
        for run_id in np.unique(run_ids):
            run_indices = ordered[run_ids == run_id]
            run_length = len(run_indices)
            state = str(result.loc[run_indices[0], "attack_time_state"])
            splittable_attack = bool(
                state == "attack"
                and int(np.ceil(val_fraction * run_length)) - (int(time_steps) - 1)
                >= int(time_steps)
            )
            if state == "attack" and not splittable_attack:
                block_count = 1
            else:
                minimum_blocks = max(1, int(np.ceil(run_length / block_epochs)))
                block_count = max(5, 5 * int(np.ceil(minimum_blocks / 5)))
                block_count = min(run_length, block_count)
            base_size, remainder = divmod(run_length, block_count)
            sizes = [base_size + int(index < remainder) for index in range(block_count)]
            offset = 0
            for state_block_id, size in enumerate(sizes):
                block_indices = run_indices[offset : offset + size]
                result.loc[block_indices, "state_block_id"] = np.int16(state_block_id)
                result.loc[block_indices, "block_id"] = np.int16(next_block_id)
                offset += size
                next_block_id += 1
            result.loc[run_indices, "state_run_epoch_index"] = np.arange(
                run_length, dtype=np.int32
            )
            result.loc[run_indices, "state_run_epoch_count"] = np.int32(run_length)
            result.loc[run_indices, "splittable_attack_run"] = splittable_attack
            result.loc[run_indices, "short_attack_atom"] = bool(
                state == "attack" and not splittable_attack
            )
    if (result[["state_run_id", "state_block_id", "block_id"]] < 0).any().any():
        raise RuntimeError("Failed to assign state-pure block identifiers")
    purity = result.groupby(["segment_id", "block_id"], observed=True)[
        "attack_time_state"
    ].nunique()
    if not purity.eq(1).all():
        raise RuntimeError("A state-stratified time block spans clean and attack epochs")
    block_sizes = result.groupby(["segment_id", "block_id"], observed=True).size()
    if int(block_sizes.max()) > int(block_epochs):
        raise RuntimeError("A balanced state block exceeds --block-epochs")
    return result


def _state_block_table(epoch_table: pd.DataFrame) -> pd.DataFrame:
    """Aggregate state-pure canonical epoch blocks for global selection."""

    group_cols = ["segment_id", "block_id"]
    state_counts = epoch_table.groupby(group_cols, observed=True)["attack_time_state"].nunique()
    if not state_counts.eq(1).all():
        raise RuntimeError("State block aggregation received a mixed-state block")
    blocks = (
        epoch_table.groupby(group_cols, sort=True, observed=True)
        .agg(
            state_run_id=("state_run_id", "first"),
            state_run_epoch_count=("state_run_epoch_count", "first"),
            state_block_id=("state_block_id", "first"),
            splittable_attack_run=("splittable_attack_run", "first"),
            short_attack_atom=("short_attack_atom", "first"),
            attack_time_state=("attack_time_state", "first"),
            epoch_count=("canonical_epoch_ms", "size"),
            canonical_start_ms=("canonical_epoch_ms", "min"),
            canonical_end_ms=("canonical_epoch_ms", "max"),
            tow_min=("tow_min", "min"),
            tow_max=("tow_max", "max"),
            row_count=("row_count", "sum"),
            positive_rows=("positive_rows", "sum"),
            positive_epochs=("positive_epoch", "sum"),
            attack_context_rows=("attack_context_rows", "sum"),
            mixed_attack_context_epochs=("mixed_attack_context_epoch", "sum"),
            device_count=("device_count", "max"),
        )
        .reset_index()
    )
    blocks["positive_ratio"] = blocks["positive_epochs"] / blocks["epoch_count"].clip(lower=1)
    return blocks


def _annotate_isolated_usable_epochs(blocks: pd.DataFrame, time_steps: int) -> pd.DataFrame:
    """Estimate final state-pure val support when each block is selected alone."""

    result = blocks.copy()
    radius = max(int(time_steps) - 1, 0)
    result["isolated_usable_epochs"] = 0
    for _, indices in result.groupby(["recording_id", "segment_id"], sort=False).groups.items():
        ordered = result.loc[list(indices)].sort_values("block_id", kind="mergesort").index.tolist()
        for position, index in enumerate(ordered):
            boundaries = int(position > 0) + int(position + 1 < len(ordered))
            result.loc[index, "isolated_usable_epochs"] = max(
                0,
                int(result.loc[index, "epoch_count"]) - radius * boundaries,
            )
    result["isolated_usable_epochs"] = result["isolated_usable_epochs"].astype(np.int32)
    return result


def choose_state_stratified_blocks(
    development_blocks: pd.DataFrame,
    val_fraction: float,
    time_steps: int,
    tie_key: str,
) -> tuple[set[str], pd.DataFrame]:
    """Select globally pooled clean/attack validation blocks per Scenario.

    A separate deterministic subset-sum search is run for each Scenario and
    reviewed event state, then the Scenario options are combined.  Scenario
    coverage is hard, global state-specific target error is primary, and
    Scenario-level balance is secondary.  This avoids forcing every short
    Session to donate 20% of its attack run.
    """

    if development_blocks.empty:
        raise ValueError("State-stratified validation received no development blocks")
    blocks = _annotate_isolated_usable_epochs(development_blocks, time_steps).reset_index(drop=True)
    scenarios = sorted(blocks["Scenario"].astype(str).unique())
    selected_uids: set[str] = set()
    audit_rows: list[dict[str, object]] = []

    for state in STATE_VALUES:
        state_blocks = blocks.loc[blocks["attack_time_state"].astype(str).eq(state)].copy()
        total_epochs = int(state_blocks["epoch_count"].sum())
        if total_epochs <= 0:
            raise ValueError(f"Development pool has no {state!r} canonical epochs")
        eligible = state_blocks.loc[
            state_blocks["isolated_usable_epochs"].ge(int(time_steps))
        ].copy()
        missing = [
            scenario
            for scenario in scenarios
            if not eligible["Scenario"].astype(str).eq(scenario).any()
        ]
        if missing:
            raise ValueError(
                f"No independently W{time_steps}-usable {state} validation block for "
                f"development Scenarios: {missing}"
            )

        eligible = eligible.sort_values(
            ["Scenario", "recording_id", "segment_id", "block_id"], kind="mergesort"
        ).reset_index(drop=True)
        target_epochs = float(total_epochs * val_fraction)
        scenario_options: list[list[tuple[int, int, int, float]]] = []
        for scenario in scenarios:
            scenario_candidates = eligible.loc[
                eligible["Scenario"].astype(str).eq(scenario)
            ]
            scenario_total = int(
                state_blocks.loc[
                    state_blocks["Scenario"].astype(str).eq(scenario), "epoch_count"
                ].sum()
            )
            scenario_target = float(scenario_total * val_fraction)
            scenario_upper = min(
                scenario_total,
                int(np.ceil(scenario_target)) + int(scenario_candidates["epoch_count"].max()),
            )
            # key=selected epochs, value=(global candidate bitset, block count)
            subset_dp: dict[int, tuple[int, int]] = {0: (0, 0)}
            for candidate_index, row in scenario_candidates.iterrows():
                epoch_count = int(row["epoch_count"])
                bit = 1 << int(candidate_index)
                additions: dict[int, tuple[int, int]] = {}
                for epochs_so_far, (bits_so_far, count_so_far) in subset_dp.items():
                    next_epochs = epochs_so_far + epoch_count
                    if next_epochs > scenario_upper:
                        continue
                    candidate = (bits_so_far | bit, count_so_far + 1)
                    incumbent = additions.get(next_epochs, subset_dp.get(next_epochs))
                    if incumbent is None or (
                        candidate[1],
                        stable_int(f"{tie_key}|{state}|{scenario}|{candidate[0]:x}"),
                    ) < (
                        incumbent[1],
                        stable_int(f"{tie_key}|{state}|{scenario}|{incumbent[0]:x}"),
                    ):
                        additions[next_epochs] = candidate
                subset_dp.update(additions)
            options = [
                (
                    epochs_selected,
                    bits,
                    count,
                    abs(epochs_selected - scenario_target) / max(scenario_total, 1),
                )
                for epochs_selected, (bits, count) in subset_dp.items()
                if epochs_selected > 0
            ]
            if not options:
                raise ValueError(
                    f"No W{time_steps}-usable {state} validation option for Scenario {scenario}"
                )
            scenario_options.append(options)

        # key=global selected epochs, value=(bits, max Scenario error,
        # summed Scenario error, block count).  Every transition chooses one
        # non-empty option, so all Scenarios remain covered by construction.
        combined: dict[int, tuple[int, float, float, int]] = {0: (0, 0.0, 0.0, 0)}
        for options in scenario_options:
            next_combined: dict[int, tuple[int, float, float, int]] = {}
            for epochs_so_far, (bits_so_far, max_error, sum_error, count_so_far) in combined.items():
                for option_epochs, option_bits, option_count, option_error in options:
                    next_epochs = epochs_so_far + option_epochs
                    candidate = (
                        bits_so_far | option_bits,
                        max(max_error, option_error),
                        sum_error + option_error,
                        count_so_far + option_count,
                    )
                    incumbent = next_combined.get(next_epochs)
                    candidate_rank = (
                        candidate[1],
                        candidate[2],
                        candidate[3],
                        stable_int(f"{tie_key}|{state}|combined|{candidate[0]:x}"),
                    )
                    if incumbent is None:
                        next_combined[next_epochs] = candidate
                    else:
                        incumbent_rank = (
                            incumbent[1],
                            incumbent[2],
                            incumbent[3],
                            stable_int(f"{tie_key}|{state}|combined|{incumbent[0]:x}"),
                        )
                        if candidate_rank < incumbent_rank:
                            next_combined[next_epochs] = candidate
            combined = next_combined

        selected_epochs, selected_value = min(
            combined.items(),
            key=lambda item: (
                abs(item[0] - target_epochs),
                item[1][1],
                item[1][2],
                item[1][3],
                stable_int(f"{tie_key}|{state}|final|{item[1][0]:x}"),
            ),
        )
        selected_bits, max_scenario_error, summed_scenario_error, selected_count = selected_value
        selected_indices = [
            index for index in range(len(eligible)) if selected_bits & (1 << index)
        ]
        chosen = eligible.iloc[selected_indices]
        selected_uids.update(chosen["block_uid"].astype(str))
        audit_rows.append(
            {
                "attack_time_state": state,
                "total_epochs": total_epochs,
                "target_raw_val_epochs": target_epochs,
                "selected_raw_val_epochs": int(selected_epochs),
                "selected_raw_val_fraction": float(selected_epochs / total_epochs),
                "selected_block_count": int(selected_count),
                "eligible_block_count": int(len(eligible)),
                "scenario_count": int(len(scenarios)),
                "max_scenario_fraction_error": float(max_scenario_error),
                "summed_scenario_fraction_error": float(summed_scenario_error),
            }
        )
    return selected_uids, pd.DataFrame(audit_rows)


def _usable_mask_window_count(
    epochs: pd.DataFrame,
    mask: pd.Series,
    split: str,
    time_steps: int,
) -> int:
    """Count W-step endpoints for an arbitrary epoch mask and final split."""

    selected = epochs.loc[mask & epochs["split"].astype(str).eq(split)].sort_values(
        ["recording_id", "segment_id", "epoch_index"], kind="mergesort"
    )
    windows = 0
    for _, group in selected.groupby(["recording_id", "segment_id"], sort=False):
        indices = group["epoch_index"].to_numpy(dtype=np.int64)
        if len(indices) == 0:
            continue
        run_ids = np.cumsum(np.r_[True, np.diff(indices) != 1]) - 1
        windows += sum(max(0, int(count) - time_steps + 1) for count in np.bincount(run_ids))
    return int(windows)


def _best_clean_size_error(
    recording_blocks: pd.DataFrame,
    required_neighbors: set[str],
    val_fraction: float,
) -> float:
    """Return the nearest contiguous clean validation size for an attack plan.

    A short edge slice of a long attack run sometimes needs the adjacent clean
    block assigned to validation so that the W-step guard removes only one side
    of the attack slice.  Prefix and suffix attack choices are otherwise tied,
    so use the resulting clean-size error to choose their direction jointly.
    """

    clean = recording_blocks.loc[
        recording_blocks["attack_time_state"].astype(str).eq("clean")
    ].copy()
    if clean.empty:
        return float("inf")
    clean_total = int(clean["epoch_count"].sum())
    target = float(clean_total * val_fraction)
    best_error = float("inf")
    for _, run in clean.groupby(["segment_id", "state_run_id"], sort=False):
        run = run.sort_values("block_id", kind="mergesort").reset_index(drop=True)
        for left in range(len(run)):
            epoch_sum = 0
            uids: set[str] = set()
            for right in range(left, len(run)):
                epoch_sum += int(run.iloc[right]["epoch_count"])
                uids.add(str(run.iloc[right]["block_uid"]))
                if epoch_sum < clean_total and required_neighbors.issubset(uids):
                    best_error = min(best_error, abs(epoch_sum - target))
    return best_error


def plan_attack_validation(
    development_blocks: pd.DataFrame,
    val_fraction: float,
    time_steps: int,
    tie_key: str,
) -> tuple[set[str], pd.DataFrame, pd.DataFrame]:
    """Plan per-run long attack splits and per-Scenario short-atom assignment."""

    attack = development_blocks.loc[
        development_blocks["attack_time_state"].astype(str).eq("attack")
    ].copy()
    if attack.empty:
        raise ValueError("Development pool contains no reviewed attack context")
    run_keys = ["recording_id", "segment_id", "state_run_id"]
    selected_uids: set[str] = set()
    long_rows: list[dict[str, object]] = []
    long_selected_by_scenario: dict[str, int] = {}

    long_blocks = attack.loc[attack["splittable_attack_run"].astype(bool)]
    run_options_by_recording: dict[
        int, list[tuple[tuple[int, int, int], pd.DataFrame, list[dict[str, object]]]]
    ] = {}
    for run_key, run in long_blocks.groupby(run_keys, sort=False):
        run = run.sort_values("state_block_id", kind="mergesort")
        run_epochs = int(run["epoch_count"].sum())
        if run_epochs < 41 or len(run) < 5:
            raise RuntimeError(
                f"Splittable attack run {run_key} must have A>=41 and at least five blocks"
            )
        target_epochs = int(np.ceil(val_fraction * run_epochs))
        candidates: list[dict[str, object]] = []
        for edge in ("prefix", "suffix"):
            for count in range(1, len(run)):
                chosen = run.iloc[:count] if edge == "prefix" else run.iloc[len(run) - count :]
                val_epochs = int(chosen["epoch_count"].sum())
                train_epochs = run_epochs - val_epochs
                # An edge slice has one internal train/val boundary.  The
                # recording-level clean selector later removes an optional
                # second boundary when the run is too short to tolerate it.
                if val_epochs - (time_steps - 1) < time_steps:
                    continue
                if train_epochs - (time_steps - 1) < time_steps:
                    continue
                uids = set(chosen["block_uid"].astype(str))
                required_neighbor: str | None = None
                if val_epochs - 2 * (time_steps - 1) < time_steps:
                    neighbor_id = (
                        int(run.iloc[0]["block_id"]) - 1
                        if edge == "prefix"
                        else int(run.iloc[-1]["block_id"]) + 1
                    )
                    neighbor = development_blocks.loc[
                        development_blocks["recording_id"].eq(int(run_key[0]))
                        & development_blocks["segment_id"].eq(int(run_key[1]))
                        & development_blocks["block_id"].eq(neighbor_id)
                        & development_blocks["attack_time_state"].astype(str).eq("clean")
                    ]
                    if not neighbor.empty:
                        required_neighbor = str(neighbor.iloc[0]["block_uid"])
                candidates.append(
                    {
                        "attack_size_error": abs(val_epochs - target_epochs),
                        "uids": uids,
                        "edge": edge,
                        "val_epochs": val_epochs,
                        "selected_count": count,
                        "required_neighbor": required_neighbor,
                    }
                )
        if not candidates:
            raise ValueError(f"Attack run {run_key} has no valid within-Session 80/20 split")
        best_size_error = min(int(item["attack_size_error"]) for item in candidates)
        candidates = [
            item for item in candidates if int(item["attack_size_error"]) == best_size_error
        ]
        run_options_by_recording.setdefault(int(run_key[0]), []).append(
            (run_key, run, candidates)
        )

    for recording_id, run_groups in run_options_by_recording.items():
        recording_blocks = development_blocks.loc[
            development_blocks["recording_id"].eq(recording_id)
        ]
        combinations: list[
            tuple[tuple[float, ...], tuple[dict[str, object], ...]]
        ] = []
        option_lists = [options for _, _, options in run_groups]
        for combination in itertools.product(*option_lists):
            required_neighbors = {
                str(item["required_neighbor"])
                for item in combination
                if item["required_neighbor"] is not None
            }
            clean_error = _best_clean_size_error(
                recording_blocks, required_neighbors, val_fraction
            )
            if not np.isfinite(clean_error):
                continue
            all_uids = sorted(
                uid for item in combination for uid in item["uids"]  # type: ignore[union-attr]
            )
            rank = (
                clean_error,
                sum(int(item["selected_count"]) for item in combination),
                stable_int(
                    f"{tie_key}|long-joint|{recording_id}|{';'.join(all_uids)}"
                ),
            )
            combinations.append((rank, combination))
        if not combinations:
            raise ValueError(
                f"Recording {recording_id} has no long-attack edge plan compatible "
                "with a contiguous clean validation interval"
            )
        _, chosen_combination = min(combinations, key=lambda item: item[0])
        for (run_key, run, _), chosen in zip(run_groups, chosen_combination):
            chosen_uids = set(chosen["uids"])  # type: ignore[arg-type]
            edge = str(chosen["edge"])
            val_epochs = int(chosen["val_epochs"])
            run_epochs = int(run["epoch_count"].sum())
            target_epochs = int(np.ceil(val_fraction * run_epochs))
            selected_uids.update(chosen_uids)
            first = run.iloc[0]
            scenario = str(first["Scenario"])
            long_selected_by_scenario[scenario] = (
                long_selected_by_scenario.get(scenario, 0) + val_epochs
            )
            long_rows.append(
                {
                    "fold": int(str(tie_key).split("_")[-1]),
                    "recording_id": int(first["recording_id"]),
                    **{column: first[column] for column in RECORDING_KEYS},
                    "segment_id": int(first["segment_id"]),
                    "state_run_id": int(first["state_run_id"]),
                    "attack_epochs": run_epochs,
                    "target_raw_val_epochs": target_epochs,
                    "raw_val_epochs": val_epochs,
                    "raw_val_fraction": float(val_epochs / run_epochs),
                    "validation_edge": edge,
                    "state_block_count": int(len(run)),
                    "selected_block_count": int(len(chosen_uids)),
                    "selected_block_uids": ";".join(sorted(chosen_uids)),
                }
            )

    short = attack.loc[attack["short_attack_atom"].astype(bool)].copy()
    atom_rows: list[dict[str, object]] = []
    atom_records: list[dict[str, object]] = []
    for run_key, run in short.groupby(run_keys, sort=False):
        if len(run) != 1:
            raise RuntimeError(f"Short attack atom {run_key} was mechanically split")
        row = run.iloc[0]
        atom_records.append(
            {
                "uid": str(row["block_uid"]),
                "epochs": int(row["epoch_count"]),
                "scenario": str(row["Scenario"]),
                "row": row,
            }
        )

    scenarios = sorted(attack["Scenario"].astype(str).unique())
    for scenario in scenarios:
        scenario_attack = attack.loc[attack["Scenario"].astype(str).eq(scenario)]
        scenario_total = int(scenario_attack["epoch_count"].sum())
        target = float(val_fraction * scenario_total)
        long_val = int(long_selected_by_scenario.get(scenario, 0))
        atoms = [record for record in atom_records if record["scenario"] == scenario]
        candidates: list[tuple[tuple[float, ...], tuple[int, ...], int]] = []
        for flags in itertools.product((0, 1), repeat=len(atoms)):
            atom_val = sum(
                atom["epochs"] for atom, selected in zip(atoms, flags) if selected
            )
            val_exists = bool(long_val > 0 or any(flags))
            long_train_exists = bool(
                scenario_attack["splittable_attack_run"].astype(bool).any()
            )
            train_exists = bool(long_train_exists or any(not flag for flag in flags))
            if not val_exists or not train_exists:
                continue
            rank = (
                abs(long_val + atom_val - target),
                sum(flags),
                stable_int(f"{tie_key}|short-atoms|{scenario}|{flags}"),
            )
            candidates.append((rank, flags, atom_val))
        if not candidates:
            raise ValueError(
                f"Scenario {scenario!r} cannot provide both attack train and validation atoms"
            )
        _, flags, atom_val = min(candidates, key=lambda item: item[0])
        for atom, selected in zip(atoms, flags):
            if selected:
                selected_uids.add(str(atom["uid"]))
            row = atom["row"]
            atom_rows.append(
                {
                    "fold": int(str(tie_key).split("_")[-1]),
                    "recording_id": int(row["recording_id"]),
                    **{column: row[column] for column in RECORDING_KEYS},
                    "segment_id": int(row["segment_id"]),
                    "state_run_id": int(row["state_run_id"]),
                    "block_uid": str(atom["uid"]),
                    "attack_epochs": int(atom["epochs"]),
                    "assignment": "val" if selected else "train",
                    "scenario_attack_epochs": scenario_total,
                    "scenario_target_raw_val_epochs": target,
                    "scenario_long_raw_val_epochs": long_val,
                    "scenario_short_atom_raw_val_epochs": atom_val,
                }
            )
    return selected_uids, pd.DataFrame(long_rows), pd.DataFrame(atom_rows)


def choose_clean_validation_by_recording(
    all_epochs: pd.DataFrame,
    development_blocks: pd.DataFrame,
    attack_val_uids: set[str],
    val_fraction: float,
    time_steps: int,
    tie_key: str,
) -> tuple[set[str], pd.DataFrame]:
    """Choose one contiguous clean validation interval inside every Session."""

    selected_clean_uids: set[str] = set()
    audit_rows: list[dict[str, object]] = []
    for recording_id, recording_blocks in development_blocks.groupby("recording_id", sort=True):
        clean = recording_blocks.loc[
            recording_blocks["attack_time_state"].astype(str).eq("clean")
        ].copy()
        if clean.empty:
            continue
        clean_total = int(clean["epoch_count"].sum())
        target = float(clean_total * val_fraction)
        recording_epochs = all_epochs.loc[all_epochs["recording_id"].eq(int(recording_id))].copy()
        attack_selected = recording_blocks.loc[
            recording_blocks["block_uid"].astype(str).isin(attack_val_uids)
        ].copy()

        required_neighbors: set[str] = set()
        long_selected = attack_selected.loc[
            attack_selected["splittable_attack_run"].astype(bool)
        ]
        for run_key, selected_run in long_selected.groupby(
            ["segment_id", "state_run_id"], sort=False
        ):
            val_length = int(selected_run["epoch_count"].sum())
            if val_length - 2 * (time_steps - 1) >= time_steps:
                continue
            run_all = recording_blocks.loc[
                recording_blocks["segment_id"].eq(int(run_key[0]))
                & recording_blocks["state_run_id"].eq(int(run_key[1]))
                & recording_blocks["attack_time_state"].astype(str).eq("attack")
            ].sort_values("block_id")
            selected_ids = set(selected_run["block_id"].astype(int))
            if int(run_all.iloc[0]["block_id"]) in selected_ids:
                neighbor_id = int(run_all.iloc[0]["block_id"]) - 1
            else:
                neighbor_id = int(run_all.iloc[-1]["block_id"]) + 1
            neighbor = clean.loc[
                clean["segment_id"].eq(int(run_key[0])) & clean["block_id"].eq(neighbor_id)
            ]
            if not neighbor.empty:
                required_neighbors.add(str(neighbor.iloc[0]["block_uid"]))

        candidate_map: dict[frozenset[str], int] = {}
        for _, run in clean.groupby(["segment_id", "state_run_id"], sort=False):
            run = run.sort_values("block_id", kind="mergesort").reset_index(drop=True)
            for left in range(len(run)):
                epoch_sum = 0
                uids: list[str] = []
                for right in range(left, len(run)):
                    epoch_sum += int(run.iloc[right]["epoch_count"])
                    uids.append(str(run.iloc[right]["block_uid"]))
                    key = frozenset(uids)
                    if epoch_sum < clean_total and required_neighbors.issubset(key):
                        candidate_map[key] = epoch_sum
        if not candidate_map:
            raise ValueError(
                f"Recording {recording_id} has no contiguous clean candidate satisfying "
                f"required attack-boundary neighbors {sorted(required_neighbors)}"
            )
        ranked = sorted(
            candidate_map.items(),
            key=lambda item: (
                abs(item[1] - target),
                len(item[0]),
                stable_int(f"{tie_key}|clean|{recording_id}|{';'.join(sorted(item[0]))}"),
            ),
        )[:256]
        feasible: list[tuple[tuple[float, ...], frozenset[str], pd.DataFrame]] = []
        for clean_uids, clean_val_epochs in ranked:
            val_uids = set(attack_val_uids).union(clean_uids)
            candidate_epochs = recording_epochs.copy()
            candidate_epochs["raw_split"] = np.where(
                candidate_epochs["block_uid"].astype(str).isin(val_uids), "val", "train"
            )
            candidate_epochs = apply_guards(candidate_epochs, time_steps)
            clean_mask = candidate_epochs["attack_time_state"].astype(str).eq("clean")
            if _usable_mask_window_count(
                candidate_epochs, clean_mask, "train", time_steps
            ) <= 0 or _usable_mask_window_count(
                candidate_epochs, clean_mask, "val", time_steps
            ) <= 0:
                continue

            attack_supported = True
            attack_rows = recording_blocks.loc[
                recording_blocks["attack_time_state"].astype(str).eq("attack")
            ]
            for run_key, run in attack_rows.groupby(
                ["segment_id", "state_run_id"], sort=False
            ):
                run_mask = (
                    candidate_epochs["segment_id"].eq(int(run_key[0]))
                    & candidate_epochs["state_run_id"].eq(int(run_key[1]))
                    & candidate_epochs["attack_time_state"].astype(str).eq("attack")
                )
                if bool(run["splittable_attack_run"].iloc[0]):
                    required_splits = ("train", "val")
                else:
                    assigned = "val" if str(run.iloc[0]["block_uid"]) in attack_val_uids else "train"
                    required_splits = (assigned,)
                if any(
                    _usable_mask_window_count(
                        candidate_epochs, run_mask, required_split, time_steps
                    )
                    <= 0
                    for required_split in required_splits
                ):
                    attack_supported = False
                    break
            if not attack_supported:
                continue
            guard_epochs = int(candidate_epochs["split"].eq("guard").sum())
            rank = (
                abs(clean_val_epochs - target),
                guard_epochs,
                len(clean_uids),
                stable_int(f"{tie_key}|clean-final|{recording_id}|{sorted(clean_uids)}"),
            )
            feasible.append((rank, clean_uids, candidate_epochs))
        if not feasible:
            raise ValueError(
                f"Recording {recording_id} has no contiguous clean split preserving W{time_steps} "
                "train/validation support"
            )
        _, chosen_uids, chosen_epochs = min(feasible, key=lambda item: item[0])
        selected_clean_uids.update(chosen_uids)
        first = recording_blocks.iloc[0]
        chosen_raw_val = int(
            (
                chosen_epochs["attack_time_state"].astype(str).eq("clean")
                & chosen_epochs["raw_split"].astype(str).eq("val")
            ).sum()
        )
        audit_rows.append(
            {
                "fold": int(str(tie_key).split("_")[-1]),
                "recording_id": int(recording_id),
                **{column: first[column] for column in RECORDING_KEYS},
                "clean_epochs": clean_total,
                "target_raw_val_epochs": target,
                "raw_val_epochs": chosen_raw_val,
                "raw_val_fraction": float(chosen_raw_val / clean_total),
                "selected_block_count": int(len(chosen_uids)),
                "selected_block_uids": ";".join(sorted(chosen_uids)),
                "required_attack_neighbor_count": int(len(required_neighbors)),
            }
        )
    return selected_clean_uids, pd.DataFrame(audit_rows)


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


def _usable_state_window_count(
    epochs: pd.DataFrame,
    state: str,
    split: str,
    time_steps: int,
) -> int:
    """Count protocol-level W-step endpoints wholly inside one state/split run."""

    selected = epochs.loc[
        epochs["attack_time_state"].astype(str).eq(state) & epochs["split"].astype(str).eq(split)
    ].sort_values(["recording_id", "segment_id", "epoch_index"], kind="mergesort")
    windows = 0
    for _, group in selected.groupby(["recording_id", "segment_id"], sort=False):
        indices = group["epoch_index"].to_numpy(dtype=np.int64)
        if len(indices) == 0:
            continue
        run_starts = np.r_[True, np.diff(indices) != 1]
        run_ids = np.cumsum(run_starts) - 1
        for count in np.bincount(run_ids):
            windows += max(0, int(count) - int(time_steps) + 1)
    return int(windows)


def state_validation_audits(
    all_epochs: pd.DataFrame,
    recordings: pd.DataFrame,
    test_ids: set[int],
    fold: int,
    time_steps: int,
    val_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return global-state and Scenario-by-state train/val/guard audits."""

    development_recordings = recordings.loc[~recordings["recording_id"].isin(test_ids)].copy()
    development_ids = set(development_recordings["recording_id"].astype(int))
    development = all_epochs.loc[all_epochs["recording_id"].isin(development_ids)].copy()
    scenarios = sorted(development_recordings["Scenario"].astype(str).unique())
    scenario_rows: list[dict[str, object]] = []
    for scenario in scenarios:
        scenario_ids = set(
            development_recordings.loc[
                development_recordings["Scenario"].astype(str).eq(scenario), "recording_id"
            ].astype(int)
        )
        scenario_epochs = development.loc[development["recording_id"].isin(scenario_ids)]
        for state in STATE_VALUES:
            state_epochs = scenario_epochs.loc[
                scenario_epochs["attack_time_state"].astype(str).eq(state)
            ]
            total = int(len(state_epochs))
            raw_val = int(state_epochs["raw_split"].eq("val").sum())
            train = int(state_epochs["split"].eq("train").sum())
            val = int(state_epochs["split"].eq("val").sum())
            guard = int(state_epochs["split"].eq("guard").sum())
            usable = _usable_state_window_count(
                scenario_epochs,
                state,
                "val",
                time_steps,
            )
            usable_train = _usable_state_window_count(
                scenario_epochs,
                state,
                "train",
                time_steps,
            )
            scenario_rows.append(
                {
                    "fold": int(fold),
                    "Scenario": scenario,
                    "attack_time_state": state,
                    "development_recordings": int(len(scenario_ids)),
                    "epoch_count": total,
                    "raw_val_epochs": raw_val,
                    "raw_val_fraction": float(raw_val / max(total, 1)),
                    "train_epochs": train,
                    "val_epochs": val,
                    "guard_epochs": guard,
                    "train_fraction": float(train / max(total, 1)),
                    "val_fraction": float(val / max(total, 1)),
                    "guard_fraction": float(guard / max(total, 1)),
                    "usable_train_w5_windows": usable_train,
                    "usable_val_w5_windows": usable,
                    "row_label_positive_epochs": int(state_epochs["positive_epoch"].sum()),
                    "mixed_attack_context_epochs": int(
                        state_epochs["mixed_attack_context_epoch"].sum()
                    ),
                    "validation_supported": bool(usable > 0),
                    "train_supported": bool(usable_train > 0),
                }
            )
    scenario_audit = pd.DataFrame(scenario_rows)

    state_rows: list[dict[str, object]] = []
    for state in STATE_VALUES:
        state_epochs = development.loc[development["attack_time_state"].astype(str).eq(state)]
        total = int(len(state_epochs))
        raw_val = int(state_epochs["raw_split"].eq("val").sum())
        train = int(state_epochs["split"].eq("train").sum())
        val = int(state_epochs["split"].eq("val").sum())
        guard = int(state_epochs["split"].eq("guard").sum())
        supported = scenario_audit.loc[
            scenario_audit["attack_time_state"].astype(str).eq(state)
            & scenario_audit["validation_supported"]
        ]
        state_rows.append(
            {
                "fold": int(fold),
                "attack_time_state": state,
                "epoch_count": total,
                "target_raw_val_epochs": float(total * val_fraction),
                "raw_val_epochs": raw_val,
                "raw_val_fraction": float(raw_val / max(total, 1)),
                "train_epochs": train,
                "val_epochs": val,
                "guard_epochs": guard,
                "train_fraction": float(train / max(total, 1)),
                "val_fraction": float(val / max(total, 1)),
                "guard_fraction": float(guard / max(total, 1)),
                "usable_val_w5_windows": _usable_state_window_count(
                    development, state, "val", time_steps
                ),
                "usable_train_w5_windows": _usable_state_window_count(
                    development, state, "train", time_steps
                ),
                "scenario_count": int(len(scenarios)),
                "supported_scenario_count": int(supported["Scenario"].nunique()),
                "row_label_positive_epochs": int(state_epochs["positive_epoch"].sum()),
                "mixed_attack_context_epochs": int(
                    state_epochs["mixed_attack_context_epoch"].sum()
                ),
            }
        )
    return pd.DataFrame(state_rows), scenario_audit


def recording_state_validation_audit(
    all_epochs: pd.DataFrame,
    recordings: pd.DataFrame,
    test_ids: set[int],
    fold: int,
    time_steps: int,
) -> pd.DataFrame:
    """Audit final train/val/guard and W-step support per recording and state."""

    rows: list[dict[str, object]] = []
    for _, recording in recordings.loc[
        ~recordings["recording_id"].isin(test_ids)
    ].sort_values("recording_id").iterrows():
        recording_id = int(recording["recording_id"])
        recording_epochs = all_epochs.loc[all_epochs["recording_id"].eq(recording_id)]
        for state in STATE_VALUES:
            state_epochs = recording_epochs.loc[
                recording_epochs["attack_time_state"].astype(str).eq(state)
            ]
            total = int(len(state_epochs))
            raw_val = int(state_epochs["raw_split"].eq("val").sum()) if total else 0
            train = int(state_epochs["split"].eq("train").sum()) if total else 0
            val = int(state_epochs["split"].eq("val").sum()) if total else 0
            guard = int(state_epochs["split"].eq("guard").sum()) if total else 0
            rows.append(
                {
                    "fold": int(fold),
                    "recording_id": recording_id,
                    **{column: recording[column] for column in RECORDING_KEYS},
                    "attack_time_state": state,
                    "epoch_count": total,
                    "raw_val_epochs": raw_val,
                    "raw_val_fraction": float(raw_val / total) if total else np.nan,
                    "train_epochs": train,
                    "val_epochs": val,
                    "guard_epochs": guard,
                    "usable_train_w5_windows": (
                        _usable_state_window_count(
                            recording_epochs, state, "train", time_steps
                        )
                        if total
                        else 0
                    ),
                    "usable_val_w5_windows": (
                        _usable_state_window_count(
                            recording_epochs, state, "val", time_steps
                        )
                        if total
                        else 0
                    ),
                    "state_present": bool(total > 0),
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


def build_reviewed_state_stratified_fold(
    epochs: pd.DataFrame,
    recordings: pd.DataFrame,
    fold: int,
    test_recording_ids: Iterable[int],
    block_epochs: int,
    val_fraction: float,
    time_steps: int,
    segment_gap_seconds: float,
    output_dir: Path,
) -> dict[str, int | float | str]:
    """Build one fold using globally pooled reviewed clean/attack strata."""

    test_ids = {int(value) for value in test_recording_ids}
    known_ids = set(recordings["recording_id"].astype(int))
    if not test_ids or test_ids - known_ids:
        raise ValueError(f"Fold {fold}: invalid outer test recording ids {sorted(test_ids)}")
    required_state_columns = {
        "attack_time_state",
        "attack_context_rows",
        "mixed_attack_context_epoch",
        "tow_min",
        "tow_max",
    }
    missing_state_columns = required_state_columns.difference(epochs.columns)
    if missing_state_columns:
        raise ValueError(
            f"Fold {fold}: state-stratified epochs are missing {sorted(missing_state_columns)}"
        )

    fold_epochs: list[pd.DataFrame] = []
    block_rows: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    for recording_id, rec in recordings.set_index("recording_id").iterrows():
        rec_epochs = epochs.loc[epochs["recording_key"] == rec["recording_key"]].copy()
        if rec_epochs.empty:
            raise ValueError(f"Fold {fold}: no epochs for recording {recording_id}")
        rec_epochs = _split_state_blocks(
            rec_epochs,
            segment_gap_seconds,
            block_epochs,
            val_fraction,
            time_steps,
        )
        rec_epochs["recording_id"] = int(recording_id)
        rec_epochs["segment_key"] = rec_epochs["segment_id"].map(
            lambda value: f"r{int(recording_id):02d}_s{int(value):02d}"
        )
        blocks = _state_block_table(rec_epochs)
        blocks["recording_id"] = int(recording_id)
        blocks["block_uid"] = blocks.apply(
            lambda row: (
                f"r{int(recording_id):02d}_s{int(row.segment_id):02d}_"
                f"b{int(row.block_id):03d}"
            ),
            axis=1,
        )
        block_uid_lookup = blocks.set_index(["segment_id", "block_id"])["block_uid"]
        rec_epochs["block_uid"] = [
            str(block_uid_lookup.loc[(int(segment), int(block))])
            for segment, block in zip(rec_epochs["segment_id"], rec_epochs["block_id"])
        ]
        fold_epochs.append(rec_epochs)
        block_rows.append(blocks)
        summaries.append(
            {
                "fold": int(fold),
                "recording_id": int(recording_id),
                "Environment": rec["Environment"],
                "Scenario": rec["Scenario"],
                "Session": rec["Session"],
                "outer_test": bool(int(recording_id) in test_ids),
                "epoch_count": int(len(rec_epochs)),
                "clean_epochs": int(rec_epochs["attack_time_state"].eq("clean").sum()),
                "attack_epochs": int(rec_epochs["attack_time_state"].eq("attack").sum()),
                "mixed_attack_context_epochs": int(
                    rec_epochs["mixed_attack_context_epoch"].sum()
                ),
                "positive_epochs": int(rec_epochs["positive_epoch"].sum()),
                "block_count": int(len(blocks)),
            }
        )

    all_epochs = pd.concat(fold_epochs, ignore_index=True)
    all_blocks = pd.concat(block_rows, ignore_index=True)
    all_blocks = all_blocks.merge(
        recordings[["recording_id", *RECORDING_KEYS]],
        on="recording_id",
        how="left",
        validate="many_to_one",
    )
    all_blocks["raw_split"] = np.where(
        all_blocks["recording_id"].isin(test_ids), "test", "train"
    )
    development_blocks = all_blocks.loc[~all_blocks["recording_id"].isin(test_ids)].copy()
    attack_val_uids, long_attack_audit, short_atom_audit = plan_attack_validation(
        development_blocks,
        val_fraction,
        time_steps,
        tie_key=f"fold_{fold}",
    )
    clean_val_uids, clean_selection_audit = choose_clean_validation_by_recording(
        all_epochs,
        development_blocks,
        attack_val_uids,
        val_fraction,
        time_steps,
        tie_key=f"fold_{fold}",
    )
    selected_uids = attack_val_uids.union(clean_val_uids)
    all_blocks.loc[all_blocks["block_uid"].isin(selected_uids), "raw_split"] = "val"
    split_lookup = all_blocks.set_index("block_uid")["raw_split"]
    all_epochs["raw_split"] = all_epochs["block_uid"].map(split_lookup)
    if all_epochs["raw_split"].isna().any():
        raise RuntimeError(f"Fold {fold}: failed to map state block splits to canonical epochs")
    all_epochs = apply_guards(all_epochs, time_steps)

    for summary in summaries:
        recording_rows = all_epochs.loc[
            all_epochs["recording_id"].eq(int(summary["recording_id"]))
        ]
        recording_blocks = all_blocks.loc[
            all_blocks["recording_id"].eq(int(summary["recording_id"]))
        ]
        summary.update(
            {
                "val_block_count": int(recording_blocks["raw_split"].eq("val").sum()),
                "raw_train_epochs": int(recording_rows["raw_split"].eq("train").sum()),
                "raw_val_epochs": int(recording_rows["raw_split"].eq("val").sum()),
                "raw_val_fraction": float(
                    recording_rows["raw_split"].eq("val").sum()
                    / max(len(recording_rows), 1)
                ),
                "train_epochs": int(recording_rows["split"].eq("train").sum()),
                "val_epochs": int(recording_rows["split"].eq("val").sum()),
                "guard_epochs": int(recording_rows["split"].eq("guard").sum()),
                "test_epochs": int(recording_rows["split"].eq("test").sum()),
                "train_positive_epochs": int(
                    recording_rows.loc[
                        recording_rows["split"].eq("train"), "positive_epoch"
                    ].sum()
                ),
                "val_positive_epochs": int(
                    recording_rows.loc[
                        recording_rows["split"].eq("val"), "positive_epoch"
                    ].sum()
                ),
            }
        )

    scenario_audit = scenario_validation_audit(all_epochs, recordings, test_ids, fold)
    state_audit, scenario_state_audit = state_validation_audits(
        all_epochs,
        recordings,
        test_ids,
        fold,
        time_steps,
        val_fraction,
    )
    recording_state_audit = recording_state_validation_audit(
        all_epochs,
        recordings,
        test_ids,
        fold,
        time_steps,
    )

    clean_recording_failures = recording_state_audit.loc[
        recording_state_audit["attack_time_state"].astype(str).eq("clean")
        & recording_state_audit["state_present"]
        & (
            recording_state_audit["usable_train_w5_windows"].le(0)
            | recording_state_audit["usable_val_w5_windows"].le(0)
        )
    ]
    if not clean_recording_failures.empty:
        raise ValueError(
            f"Fold {fold}: development Sessions missing clean W{time_steps} train/val support:\n"
            + clean_recording_failures[
                [
                    *RECORDING_KEYS,
                    "epoch_count",
                    "raw_val_epochs",
                    "usable_train_w5_windows",
                    "usable_val_w5_windows",
                ]
            ].to_string(index=False)
        )

    if not long_attack_audit.empty:
        final_train_windows: list[int] = []
        final_val_windows: list[int] = []
        final_guard_epochs: list[int] = []
        for row in long_attack_audit.itertuples(index=False):
            run_mask = (
                all_epochs["recording_id"].eq(int(row.recording_id))
                & all_epochs["segment_id"].eq(int(row.segment_id))
                & all_epochs["state_run_id"].eq(int(row.state_run_id))
                & all_epochs["attack_time_state"].astype(str).eq("attack")
            )
            final_train_windows.append(
                _usable_mask_window_count(all_epochs, run_mask, "train", time_steps)
            )
            final_val_windows.append(
                _usable_mask_window_count(all_epochs, run_mask, "val", time_steps)
            )
            final_guard_epochs.append(int((run_mask & all_epochs["split"].eq("guard")).sum()))
        long_attack_audit["guard_epochs"] = final_guard_epochs
        long_attack_audit["usable_train_w5_windows"] = final_train_windows
        long_attack_audit["usable_val_w5_windows"] = final_val_windows
        invalid_long = long_attack_audit.loc[
            long_attack_audit["usable_train_w5_windows"].le(0)
            | long_attack_audit["usable_val_w5_windows"].le(0)
        ]
        if not invalid_long.empty:
            raise ValueError(
                f"Fold {fold}: splittable attack runs lost W{time_steps} support:\n"
                + invalid_long.to_string(index=False)
            )

    if not short_atom_audit.empty:
        assigned_windows: list[int] = []
        atom_guard_epochs: list[int] = []
        for row in short_atom_audit.itertuples(index=False):
            atom_mask = (
                all_epochs["recording_id"].eq(int(row.recording_id))
                & all_epochs["segment_id"].eq(int(row.segment_id))
                & all_epochs["state_run_id"].eq(int(row.state_run_id))
                & all_epochs["attack_time_state"].astype(str).eq("attack")
            )
            assigned_windows.append(
                _usable_mask_window_count(
                    all_epochs,
                    atom_mask,
                    str(row.assignment),
                    time_steps,
                )
            )
            atom_guard_epochs.append(int((atom_mask & all_epochs["split"].eq("guard")).sum()))
        short_atom_audit["guard_epochs"] = atom_guard_epochs
        short_atom_audit["usable_assigned_w5_windows"] = assigned_windows
        invalid_atoms = short_atom_audit.loc[
            short_atom_audit["usable_assigned_w5_windows"].le(0)
        ]
        if not invalid_atoms.empty:
            raise ValueError(
                f"Fold {fold}: short attack atoms lost all assigned W{time_steps} windows:\n"
                + invalid_atoms.to_string(index=False)
            )
    unsupported = scenario_state_audit.loc[~scenario_state_audit["validation_supported"]]
    if not unsupported.empty:
        raise ValueError(
            f"Fold {fold}: Scenario/state pairs have no usable W{time_steps} validation "
            "window:\n"
            + unsupported[
                ["Scenario", "attack_time_state", "epoch_count", "raw_val_epochs", "val_epochs"]
            ].to_string(index=False)
        )
    attack_scenario_failures = scenario_state_audit.loc[
        scenario_state_audit["attack_time_state"].astype(str).eq("attack")
        & (
            ~scenario_state_audit["validation_supported"].astype(bool)
            | ~scenario_state_audit["train_supported"].astype(bool)
        )
    ]
    if not attack_scenario_failures.empty:
        raise ValueError(
            f"Fold {fold}: Scenarios missing attack W{time_steps} train/val support:\n"
            + attack_scenario_failures[
                [
                    "Scenario",
                    "epoch_count",
                    "raw_val_epochs",
                    "usable_train_w5_windows",
                    "usable_val_w5_windows",
                ]
            ].to_string(index=False)
        )
    if (state_audit["train_epochs"] <= 0).any() or (state_audit["val_epochs"] <= 0).any():
        raise ValueError(f"Fold {fold}: both clean and attack states require train and validation")

    block_split = (
        all_epochs.groupby(["recording_id", "segment_id", "block_id"], sort=False)["split"]
        .agg(
            lambda values: (
                "guard"
                if set(values) == {"guard"}
                else next(str(value) for value in values if str(value) != "guard")
            )
        )
        .rename("epoch_split")
        .reset_index()
    )
    all_blocks = all_blocks.merge(
        block_split,
        on=["recording_id", "segment_id", "block_id"],
        how="left",
        validate="one_to_one",
    )

    if all_epochs.duplicated(["recording_id", "canonical_epoch_ms"]).any():
        raise RuntimeError(f"Fold {fold}: duplicate recording/canonical epoch rows")
    if not all_epochs.loc[all_epochs["recording_id"].isin(test_ids), "split"].eq("test").all():
        raise RuntimeError(f"Fold {fold}: outer test recordings were not kept intact")
    development = all_epochs.loc[~all_epochs["recording_id"].isin(test_ids)]
    if development["split"].eq("test").any():
        raise RuntimeError(f"Fold {fold}: development recording leaked into outer test")
    if set(all_epochs["split"].unique()) - {"train", "val", "test", "guard"}:
        raise RuntimeError(f"Fold {fold}: unexpected split values")

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

    all_blocks.insert(0, "fold", int(fold))
    block_columns = [
        "fold",
        "recording_id",
        *RECORDING_KEYS,
        "segment_id",
        "state_run_id",
        "state_run_epoch_count",
        "state_block_id",
        "splittable_attack_run",
        "short_attack_atom",
        "block_id",
        "block_uid",
        "attack_time_state",
        "epoch_count",
        "canonical_start_ms",
        "canonical_end_ms",
        "tow_min",
        "tow_max",
        "row_count",
        "positive_rows",
        "positive_epochs",
        "positive_ratio",
        "attack_context_rows",
        "mixed_attack_context_epochs",
        "device_count",
        "raw_split",
        "epoch_split",
    ]
    all_blocks[block_columns].to_csv(
        fold_dir / "time_block_manifest.csv", index=False, encoding="utf-8-sig"
    )

    all_epochs.insert(0, "fold", int(fold))
    missing_recording_columns = [
        column for column in RECORDING_KEYS if column not in all_epochs.columns
    ]
    if missing_recording_columns:
        all_epochs = all_epochs.merge(
            recordings[["recording_id", *RECORDING_KEYS]],
            on="recording_id",
            how="left",
            validate="many_to_one",
        )
    epoch_columns = [
        "fold",
        "recording_id",
        *RECORDING_KEYS,
        "segment_id",
        "segment_key",
        "state_run_id",
        "state_run_epoch_index",
        "state_run_epoch_count",
        "state_block_id",
        "splittable_attack_run",
        "short_attack_atom",
        "block_id",
        "block_uid",
        "epoch_index",
        "segment_epoch_index",
        "canonical_epoch_ms",
        "tow_min",
        "tow_max",
        "attack_time_state",
        "attack_context_rows",
        "mixed_attack_context_epoch",
        "row_count",
        "positive_rows",
        "positive_epoch",
        "device_count",
        "raw_split",
        "split",
        "is_guard",
        "guard_reason",
    ]
    all_epochs[epoch_columns].to_csv(
        fold_dir / "epoch_split_manifest.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(summaries).to_csv(
        fold_dir / "recording_summary.csv", index=False, encoding="utf-8-sig"
    )
    scenario_audit.to_csv(
        fold_dir / "scenario_validation_summary.csv", index=False, encoding="utf-8-sig"
    )
    state_audit.to_csv(
        fold_dir / "state_validation_summary.csv", index=False, encoding="utf-8-sig"
    )
    scenario_state_audit.to_csv(
        fold_dir / "scenario_state_validation_summary.csv", index=False, encoding="utf-8-sig"
    )
    recording_state_audit.to_csv(
        fold_dir / "recording_state_validation_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    clean_selection_audit.to_csv(
        fold_dir / "clean_session_assignment.csv", index=False, encoding="utf-8-sig"
    )
    long_attack_audit.to_csv(
        fold_dir / "long_attack_run_assignment.csv", index=False, encoding="utf-8-sig"
    )
    short_atom_audit.to_csv(
        fold_dir / "short_atom_assignment.csv", index=False, encoding="utf-8-sig"
    )

    development_epochs = int(len(development))
    raw_train_epochs = int(development["raw_split"].eq("train").sum())
    raw_val_epochs = int(development["raw_split"].eq("val").sum())
    train_epochs = int(development["split"].eq("train").sum())
    val_epochs = int(development["split"].eq("val").sum())
    guard_epochs = int(development["split"].eq("guard").sum())
    covered_scenarios = sorted(scenario_audit["Scenario"].astype(str).unique())
    result: dict[str, int | float | str] = {
        "fold": int(fold),
        "test_recording_count": int(len(test_ids)),
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
        "test_epochs": int(all_epochs["split"].eq("test").sum()),
        "development_scenario_count": int(len(covered_scenarios)),
        "validation_scenario_count": int(len(covered_scenarios)),
        "validation_scenarios": ";".join(covered_scenarios),
        "missing_validation_scenarios": "",
        "train_positive_epochs": int(
            all_epochs.loc[all_epochs["split"].eq("train"), "positive_epoch"].sum()
        ),
        "val_positive_epochs": int(
            all_epochs.loc[all_epochs["split"].eq("val"), "positive_epoch"].sum()
        ),
        "val_negative_epochs": int(
            (all_epochs["split"].eq("val") & all_epochs["positive_epoch"].eq(0)).sum()
        ),
        "test_positive_epochs": int(
            all_epochs.loc[all_epochs["split"].eq("test"), "positive_epoch"].sum()
        ),
    }
    for _, row in state_audit.iterrows():
        state = str(row["attack_time_state"])
        result[f"{state}_raw_val_fraction"] = float(row["raw_val_fraction"])
        result[f"{state}_val_fraction"] = float(row["val_fraction"])
        result[f"{state}_usable_val_w5_windows"] = int(row["usable_val_w5_windows"])
    return result


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
    parser.add_argument(
        "--validation-mode",
        choices=VALIDATION_MODES,
        default=DEFAULT_VALIDATION_MODE,
        help=(
            "recording-local preserves the historical selector; reviewed-state-stratified "
            "pools reviewed clean/attack context blocks across development Sessions"
        ),
    )
    parser.add_argument(
        "--label-config",
        type=Path,
        default=ROOT / "configs" / "preprocessing.yml",
        help="reviewed Session TOW intervals used only by reviewed-state-stratified mode",
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
    if args.validation_mode == "reviewed-state-stratified" and args.strict_validation:
        parser.error(
            "--strict-validation belongs to recording-local mode; "
            "reviewed-state-stratified has its own mandatory Scenario/state checks"
        )
    if not args.csv.exists():
        raise FileNotFoundError(args.csv)
    if not args.source_recording_manifest.exists():
        raise FileNotFoundError(args.source_recording_manifest)
    if args.validation_mode == "reviewed-state-stratified" and not args.label_config.exists():
        raise FileNotFoundError(args.label_config)

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
    attack_intervals = None
    if args.validation_mode == "reviewed-state-stratified":
        attack_intervals = load_reviewed_attack_intervals(args.label_config, recordings)
        input_files["reviewed_label_config"] = {
            "path": str(args.label_config.resolve()),
            "sha256": file_sha256(args.label_config),
        }
    epochs = load_epoch_table(
        args.csv,
        recordings,
        args.data_scope,
        attack_intervals=attack_intervals,
    )
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
        if args.validation_mode == "reviewed-state-stratified":
            summary = build_reviewed_state_stratified_fold(
                epochs=epochs,
                recordings=recordings,
                fold=fold,
                test_recording_ids=test_recording_ids,
                block_epochs=int(args.block_epochs),
                val_fraction=float(args.val_fraction),
                time_steps=int(args.time_steps),
                segment_gap_seconds=float(args.segment_gap_seconds),
                output_dir=args.output_dir,
            )
        else:
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
        "generator": {
            "script": "pipeline_total/19_generate_static_timeblock_protocol.py",
            "sha256": file_sha256(Path(__file__).resolve()),
        },
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
            "per-Session contiguous clean split; per-run splittable attack 80/20; "
            "per-Scenario indivisible short-attack atom assignment"
            if args.validation_mode == "reviewed-state-stratified"
            else (
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
    if args.validation_mode == "reviewed-state-stratified":
        assert attack_intervals is not None
        metadata.update(
            {
                "validation_mode": args.validation_mode,
                "validation_size_tolerance_applies": False,
                "state_long_attack_edge_policy": (
                    "after exact attack-size matching, choose prefix/suffix jointly per "
                    "recording to minimise the feasible clean validation-size error"
                ),
                "attack_time_state_semantics": (
                    "attack iff any reviewed row in a recording/canonical UTC epoch has TOW "
                    "inside an inclusive preprocessing.yml Session interval; independent of "
                    "FreqBand and row-level Label"
                ),
                "state_values": list(STATE_VALUES),
                "state_blocking": (
                    "clean uses balanced contiguous blocks; attack runs satisfying "
                    "ceil(val_fraction*A)-(W-1)>=W use at least five balanced blocks; "
                    "shorter attack runs remain indivisible atoms"
                ),
                "reviewed_intervals": [
                    {
                        "Environment": key[0],
                        "Scenario": key[1],
                        "Session": key[2],
                        "intervals": [list(interval) for interval in intervals],
                    }
                    for key, intervals in sorted(attack_intervals.items())
                ],
                "state_validation_by_fold": [
                    {
                        key: summary[key]
                        for key in (
                            "fold",
                            "clean_raw_val_fraction",
                            "attack_raw_val_fraction",
                            "clean_val_fraction",
                            "attack_val_fraction",
                            "clean_usable_val_w5_windows",
                            "attack_usable_val_w5_windows",
                        )
                    }
                    for summary in fold_summaries
                ],
                "state_audit_file": "fold_N/state_validation_summary.csv",
                "scenario_state_audit_file": (
                    "fold_N/scenario_state_validation_summary.csv"
                ),
                "recording_state_audit_file": (
                    "fold_N/recording_state_validation_summary.csv"
                ),
                "clean_session_assignment_file": "fold_N/clean_session_assignment.csv",
                "long_attack_run_assignment_file": (
                    "fold_N/long_attack_run_assignment.csv"
                ),
                "short_atom_assignment_file": "fold_N/short_atom_assignment.csv",
            }
        )
    (args.output_dir / "protocol_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Generated {args.data_scope} outer-Session/time-block manifests")
    print(f"  recordings={len(recordings)} folds={len(fold_plan)}")
    print(f"  canonical epoch=floor(utcTimeMillis/{DEFAULT_CANONICAL_MS})*{DEFAULT_CANONICAL_MS}ms")
    if args.validation_mode == "reviewed-state-stratified":
        print(
            f"  block_epochs={args.block_epochs} val_fraction={args.val_fraction} "
            f"guard={args.time_steps - 1} mode={args.validation_mode}"
        )
    else:
        print(
            f"  block_epochs={args.block_epochs} val_fraction={args.val_fraction} "
            f"size_tolerance={args.val_size_tolerance} guard={args.time_steps - 1} "
            f"strict={args.strict_validation}"
        )
    print(pd.DataFrame(fold_summaries).to_string(index=False))


if __name__ == "__main__":
    main()

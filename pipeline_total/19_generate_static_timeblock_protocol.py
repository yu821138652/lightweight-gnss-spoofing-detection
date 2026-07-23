"""Generate the outer-Session / inner-time-block static CV protocol.

The protocol is deliberately stricter than a random window split:

* one complete reviewed static recording is held out as ``test`` per fold;
* the other recordings remain in the development pool;
* development recordings are divided into deterministic contiguous blocks of
  canonical epochs (the default block size is 256);
* a deterministic, label-aware subset of blocks is assigned to validation;
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
DEFAULT_BLOCK_EPOCHS = 256
DEFAULT_TIME_STEPS = 5
DEFAULT_CANONICAL_MS = 1000
DEFAULT_SEGMENT_GAP_SECONDS = 2.0
DEFAULT_VAL_FRACTION = 0.20


def stable_int(value: str) -> int:
    """Return a process-independent integer tie-break key."""

    return int(hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:16], 16)


def recording_key_frame(df: pd.DataFrame) -> pd.Series:
    """Build a collision-resistant string key for a recording identity."""

    return pd.Series(
        list(map(tuple, df[RECORDING_KEYS].astype(str).to_numpy())),
        index=df.index,
        dtype="object",
    )


def read_recordings(path: Path) -> pd.DataFrame:
    """Read and validate at least two reviewed static recording identities."""

    source = pd.read_csv(path, encoding="utf-8-sig")
    missing = set(RECORDING_KEYS).difference(source.columns)
    if missing:
        raise ValueError(f"Source recording manifest is missing columns: {sorted(missing)}")
    recordings = source[RECORDING_KEYS].drop_duplicates().copy()
    recordings = recordings.loc[recordings["Scenario"].astype(str).str.startswith(STATIC_PREFIX)].copy()
    if len(recordings) < 2:
        raise ValueError(
            "Expected at least two reviewed static recordings; "
            f"found {len(recordings)} in {path}"
        )
    if recordings.duplicated(RECORDING_KEYS).any():
        raise ValueError("Recording manifest contains duplicate recording identities.")
    recordings = recordings.sort_values(RECORDING_KEYS, kind="mergesort").reset_index(drop=True)
    recordings.insert(0, "recording_id", np.arange(len(recordings), dtype=np.int32))
    recordings["recording_key"] = recording_key_frame(recordings)
    return recordings


def _aggregate_chunk(chunk: pd.DataFrame, target_keys: set[tuple[str, str, str]]) -> pd.DataFrame:
    """Aggregate one CSV chunk to canonical recording epochs."""

    if chunk.empty:
        return pd.DataFrame()
    chunk = chunk.copy()
    chunk["_recording_key"] = recording_key_frame(chunk)
    chunk = chunk.loc[chunk["_recording_key"].isin(target_keys)].copy()
    if chunk.empty:
        return pd.DataFrame()
    status = chunk["LabelStatus"].astype(str).eq("reviewed")
    static = chunk["Scenario"].astype(str).str.startswith(STATIC_PREFIX)
    chunk = chunk.loc[status & static].copy()
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


def load_epoch_table(csv_path: Path, recordings: pd.DataFrame, chunksize: int = 500_000) -> pd.DataFrame:
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
        part = _aggregate_chunk(chunk, target_keys)
        if not part.empty:
            parts.append(part)
    if not parts:
        raise ValueError(f"No reviewed static rows for the requested recordings in {csv_path}")
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
    chosen = blocks.iloc[list(selected)]
    val_epochs = int(chosen["epoch_count"].sum())
    val_positive = int(chosen["positive_epochs"].sum())
    train_positive = total_positive - val_positive
    val_ratio = val_positive / max(val_epochs, 1)
    total_ratio = total_positive / max(total_epochs, 1)
    has_both_classes = 0 < total_positive < total_epochs
    missing_positive = int(has_both_classes and val_positive == 0)
    missing_negative = int(has_both_classes and val_positive == val_epochs)
    # If one block cannot contain both classes, preserve positive support over
    # an all-negative validation block.  This makes Recall/F1 observable while
    # retaining a positive block in most recordings.
    class_penalty = 3.0 * missing_positive + 1.0 * missing_negative
    train_empty_penalty = 0.5 if total_positive > 0 and train_positive == 0 else 0.0
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
        class_penalty + train_empty_penalty,
        ratio_error,
        fraction_error,
        continuity_penalty,
        tie,
    )


def choose_validation_blocks(blocks: pd.DataFrame, val_fraction: float, tie_key: str) -> set[int]:
    """Choose a deterministic, approximately 20% label-aware block subset."""

    if len(blocks) <= 1:
        return set()
    blocks = blocks.reset_index(drop=True)
    total_epochs = int(blocks["epoch_count"].sum())
    total_positive = int(blocks["positive_epochs"].sum())
    target_count = max(1, int(round(len(blocks) * val_fraction)))
    target_count = min(target_count, len(blocks) - 1)
    # Enumerating combinations is tiny for the current recordings (at most
    # eight blocks); retain a bounded deterministic fallback for future data.
    if len(blocks) <= 24:
        candidates = itertools.combinations(range(len(blocks)), target_count)
    else:
        order = sorted(
            range(len(blocks)),
            key=lambda i: stable_int(f"{tie_key}|block|{i}"),
        )
        candidates = [tuple(order[:target_count])]
    best: tuple[float, ...] | None = None
    best_selected: tuple[int, ...] | None = None
    for selected in candidates:
        selected = tuple(sorted(selected))
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


def build_fold(
    epochs: pd.DataFrame,
    recordings: pd.DataFrame,
    fold: int,
    test_recording_id: int,
    block_epochs: int,
    val_fraction: float,
    time_steps: int,
    segment_gap_seconds: float,
    output_dir: Path,
) -> dict[str, int | float | str]:
    """Build one outer-test fold and write all manifest levels."""

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
        if int(recording_id) == int(test_recording_id):
            blocks["raw_split"] = "test"
            rec_epochs["raw_split"] = "test"
        else:
            validation_indices = choose_validation_blocks(blocks, val_fraction, str(rec["recording_key"]))
            blocks["raw_split"] = "train"
            if validation_indices:
                blocks.loc[sorted(validation_indices), "raw_split"] = "val"
            block_lookup = blocks.set_index(["segment_id", "block_id"])["raw_split"]
            rec_epochs["raw_split"] = [
                "test" if int(recording_id) == int(test_recording_id)
                else str(block_lookup.loc[(int(segment), int(block))])
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
                "outer_test": int(recording_id) == int(test_recording_id),
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
        recording_manifest["recording_id"].eq(test_recording_id), "test", "development"
    )
    recording_manifest["outer_test"] = recording_manifest["recording_id"].eq(test_recording_id)
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

    # Integrity checks are intentionally strict: they catch accidental row
    # duplication and ensure the outer test recording never shares an epoch.
    if all_epochs.duplicated(["recording_id", "canonical_epoch_ms"]).any():
        raise RuntimeError(f"Fold {fold}: duplicate recording/canonical epoch rows")
    test_rows = all_epochs[all_epochs["recording_id"] == test_recording_id]
    if not test_rows["split"].eq("test").all():
        raise RuntimeError(f"Fold {fold}: outer test recording was not kept intact")
    if set(all_epochs["split"].unique()) - {"train", "val", "test", "guard"}:
        raise RuntimeError(f"Fold {fold}: unexpected split values")

    return {
        "fold": fold,
        "test_recording_id": test_recording_id,
        "epoch_count": int(len(all_epochs)),
        "train_epochs": int((all_epochs["split"] == "train").sum()),
        "val_epochs": int((all_epochs["split"] == "val").sum()),
        "guard_epochs": int((all_epochs["split"] == "guard").sum()),
        "test_epochs": int((all_epochs["split"] == "test").sum()),
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
    parser.add_argument("--segment-gap-seconds", type=float, default=DEFAULT_SEGMENT_GAP_SECONDS)
    args = parser.parse_args()
    if args.time_steps < 2:
        parser.error("--time-steps must be at least 2")
    if args.block_epochs < args.time_steps:
        parser.error("--block-epochs must be >= --time-steps")
    if not 0.0 < args.val_fraction < 0.5:
        parser.error("--val-fraction must be between 0 and 0.5")
    if args.segment_gap_seconds <= 0:
        parser.error("--segment-gap-seconds must be positive")
    if not args.csv.exists():
        raise FileNotFoundError(args.csv)
    if not args.source_recording_manifest.exists():
        raise FileNotFoundError(args.source_recording_manifest)

    recordings = read_recordings(args.source_recording_manifest)
    epochs = load_epoch_table(args.csv, recordings)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fold_assignment_rows: list[dict[str, object]] = []
    fold_summaries: list[dict[str, object]] = []
    for fold, test_recording_id in enumerate(recordings["recording_id"].tolist(), start=1):
        summary = build_fold(
            epochs=epochs,
            recordings=recordings,
            fold=fold,
            test_recording_id=int(test_recording_id),
            block_epochs=int(args.block_epochs),
            val_fraction=float(args.val_fraction),
            time_steps=int(args.time_steps),
            segment_gap_seconds=float(args.segment_gap_seconds),
            output_dir=args.output_dir,
        )
        fold_summaries.append(summary)
        test_recording = recordings.loc[recordings["recording_id"] == test_recording_id].iloc[0]
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
        "recordings": int(len(recordings)),
        "outer_folds": int(len(recordings)),
        "time_steps": int(args.time_steps),
        "block_epochs": int(args.block_epochs),
        "validation_fraction": float(args.val_fraction),
        "segment_gap_seconds": float(args.segment_gap_seconds),
        "canonical_epoch": "floor(utcTimeMillis / 1000) * 1000",
        "guard_epochs_each_side": int(args.time_steps - 1),
        "split_values": ["train", "val", "guard", "test"],
        "epoch_manifest_key": ["Environment", "Scenario", "Session", "canonical_epoch_ms"],
    }
    (args.output_dir / "protocol_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Generated static outer-Session/time-block manifests")
    print(f"  recordings={len(recordings)} folds={len(recordings)}")
    print(f"  canonical epoch=floor(utcTimeMillis/{DEFAULT_CANONICAL_MS})*{DEFAULT_CANONICAL_MS}ms")
    print(f"  block_epochs={args.block_epochs} val_fraction={args.val_fraction} guard={args.time_steps - 1}")
    print(pd.DataFrame(fold_summaries).to_string(index=False))


if __name__ == "__main__":
    main()

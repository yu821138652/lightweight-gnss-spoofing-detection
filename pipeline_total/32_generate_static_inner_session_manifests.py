"""Generate inner leave-one-Session-out manifests for a fixed outer test.

The input outer manifest must contain one complete ``test`` recording and two
or more ``development`` recordings.  For every development recording, this
script emits an inner fold that uses that whole Session as validation and all
other development Sessions as training.  It reuses an existing per-epoch block
manifest only as a canonical-time inventory; no time-block validation remains.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


KEYS = ("Environment", "Scenario", "Session")


def read_outer(path: Path) -> pd.DataFrame:
    outer = pd.read_csv(path, encoding="utf-8-sig")
    missing = set(KEYS) | {"split"}
    missing.difference_update(outer.columns)
    if missing:
        raise ValueError(f"Outer manifest missing columns: {sorted(missing)}")
    if outer.duplicated(list(KEYS)).any():
        raise ValueError("Outer manifest has duplicate recording identities")
    outer = outer.copy()
    for column in KEYS:
        outer[column] = outer[column].astype(str)
    outer["outer_role"] = outer["split"].astype(str).map(lambda value: "test" if value == "test" else "dev")
    if int(outer["outer_role"].eq("test").sum()) != 1:
        raise ValueError("Expected exactly one complete outer test Session")
    if int(outer["outer_role"].eq("dev").sum()) < 2:
        raise ValueError("Need at least two development Sessions for inner leave-one-Session-out validation")
    return outer


def read_block(path: Path) -> pd.DataFrame:
    block = pd.read_csv(path, encoding="utf-8-sig")
    missing = set(KEYS) | {"split"}
    missing.difference_update(block.columns)
    if missing:
        raise ValueError(f"Block manifest missing columns: {sorted(missing)}")
    if "canonical_epoch_ms" not in block.columns:
        raise ValueError("Expected a per-epoch block manifest with canonical_epoch_ms")
    block = block.copy()
    for column in KEYS:
        block[column] = block[column].astype(str)
    return block


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outer-manifest", type=Path, required=True)
    parser.add_argument("--block-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--write-refit-all-development", action="store_true",
        help="also write a block manifest that assigns every development Session to train and preserves the outer test",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    outer = read_outer(args.outer_manifest)
    block = read_block(args.block_manifest)
    dev = outer.loc[outer["outer_role"].eq("dev")].copy().sort_values(list(KEYS), kind="mergesort").reset_index(drop=True)
    dev_keys = set(map(tuple, dev.loc[:, KEYS].itertuples(index=False, name=None)))
    outer_keys = set(map(tuple, outer.loc[:, KEYS].itertuples(index=False, name=None)))
    block_keys = set(map(tuple, block.loc[:, KEYS].itertuples(index=False, name=None)))
    unexpected = block_keys.difference(outer_keys)
    missing = dev_keys.difference(block_keys)
    if unexpected:
        raise ValueError(f"Block manifest contains recordings outside the outer protocol: {sorted(unexpected)}")
    if missing:
        raise ValueError(f"Block manifest lacks development recordings: {sorted(missing)}")

    summary: list[dict[str, object]] = []
    for index, row in dev.iterrows():
        held_out = tuple(row[column] for column in KEYS)
        inner = block.copy()
        is_validation = (inner.loc[:, KEYS] == held_out).all(axis=1)
        is_outer_test = (inner.loc[:, KEYS] == tuple(outer.loc[outer["outer_role"].eq("test"), KEYS].iloc[0])).all(axis=1)
        inner["split"] = "train"
        inner.loc[is_validation, "split"] = "val"
        inner.loc[is_outer_test, "split"] = "test"
        if "raw_split" in inner.columns:
            inner["raw_split"] = inner["split"]
        if "is_guard" in inner.columns:
            inner["is_guard"] = False
        if "guard_reason" in inner.columns:
            inner["guard_reason"] = ""

        recording = outer.copy()
        recording["inner_role"] = recording["outer_role"].map({"test": "test", "dev": "train"})
        recording.loc[(recording.loc[:, KEYS] == held_out).all(axis=1), "inner_role"] = "val"
        fold_name = f"inner_{index + 1:02d}"
        summary.append({
            "inner_fold": fold_name,
            "validation_recording_id": row.get("recording_id", ""),
            "validation_environment": held_out[0],
            "validation_scenario": held_out[1],
            "validation_session": held_out[2],
            "train_sessions": int(recording["inner_role"].eq("train").sum()),
            "val_sessions": int(recording["inner_role"].eq("val").sum()),
            "outer_test_sessions": int(recording["inner_role"].eq("test").sum()),
            "train_epochs": int((~is_validation).sum()),
            "val_epochs": int(is_validation.sum()),
        })
        if not args.dry_run:
            fold_dir = args.output_dir / fold_name
            fold_dir.mkdir(parents=True, exist_ok=True)
            inner.to_csv(fold_dir / "block_manifest.csv", index=False, encoding="utf-8-sig")
            recording.to_csv(fold_dir / "recording_split_manifest.csv", index=False, encoding="utf-8-sig")

    result = pd.DataFrame(summary)
    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output_dir / "inner_fold_summary.csv", index=False, encoding="utf-8-sig")
        if args.write_refit_all_development:
            test_key = tuple(outer.loc[outer["outer_role"].eq("test"), KEYS].iloc[0])
            refit = block.copy()
            is_outer_test = (refit.loc[:, KEYS] == test_key).all(axis=1)
            refit["split"] = "train"
            refit.loc[is_outer_test, "split"] = "test"
            if "raw_split" in refit.columns:
                refit["raw_split"] = refit["split"]
            if "is_guard" in refit.columns:
                refit["is_guard"] = False
            if "guard_reason" in refit.columns:
                refit["guard_reason"] = ""
            recording = outer.copy()
            recording["inner_role"] = recording["outer_role"].map({"test": "test", "dev": "train"})
            refit_dir = args.output_dir / "refit_all_development"
            refit_dir.mkdir(parents=True, exist_ok=True)
            refit.to_csv(refit_dir / "block_manifest.csv", index=False, encoding="utf-8-sig")
            recording.to_csv(refit_dir / "recording_split_manifest.csv", index=False, encoding="utf-8-sig")
    print(result.to_string(index=False))
    if not args.dry_run:
        print(f"Wrote {len(result)} inner Session manifests to {args.output_dir}")


if __name__ == "__main__":
    main()

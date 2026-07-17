"""Generate balanced static-session cross-validation split manifests.

Each fold contains one new-building and one playground recording in validation,
one of each environment in test, and keeps the remaining recordings in train.
All devices from one physical recording remain in the same split.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RECORDING_COLUMNS = ["Environment", "Scenario", "Session"]


def stable_key(row: pd.Series) -> str:
    value = "|".join(str(row[column]) for column in RECORDING_COLUMNS)
    return hashlib.md5(value.encode("utf-8")).hexdigest()


def build_fold_manifests(recordings: pd.DataFrame, folds: int, output_dir: Path) -> pd.DataFrame:
    environments = sorted(recordings["Environment"].astype(str).unique())
    if len(environments) != 2:
        raise ValueError(f"Expected exactly two environments for balanced cross-validation, got: {environments}")

    assignment_parts: list[pd.DataFrame] = []
    for environment in environments:
        subset = recordings.loc[recordings["Environment"].astype(str) == environment].copy()
        if len(subset) < folds:
            raise ValueError(
                f"Environment {environment!r} has {len(subset)} recordings, fewer than requested folds={folds}."
            )
        subset["_stable_key"] = subset.apply(stable_key, axis=1)
        subset = subset.sort_values(["_stable_key", "Scenario", "Session"], kind="mergesort").reset_index(drop=True)
        subset["cv_fold"] = subset.index % folds
        assignment_parts.append(subset.drop(columns="_stable_key"))

    assignments = pd.concat(assignment_parts, ignore_index=True)
    assignments["recording_id"] = pd.factorize(
        pd.MultiIndex.from_frame(assignments[RECORDING_COLUMNS].astype(str)), sort=True
    )[0].astype("int32")

    for fold_index in range(folds):
        manifest = assignments[["recording_id", *RECORDING_COLUMNS]].copy()
        manifest["split"] = "train"
        manifest.loc[assignments["cv_fold"] == fold_index, "split"] = "test"
        manifest.loc[assignments["cv_fold"] == (fold_index + 1) % folds, "split"] = "val"

        coverage = manifest.groupby(["split", "Environment"], dropna=False).size()
        if coverage.size != len(environments) * 3 or (coverage == 0).any():
            raise RuntimeError(f"Fold {fold_index + 1} does not cover both environments in every split.")
        fold_dir = output_dir / f"fold_{fold_index + 1}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(fold_dir / "recording_split_manifest.csv", index=False, encoding="utf-8-sig")

    assignments.to_csv(output_dir / "fold_assignment.csv", index=False, encoding="utf-8-sig")
    return assignments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=PROJECT_ROOT / "output" / "tensors_static_cross_env" / "recording_split_manifest.csv",
        help="Any recording-level manifest that lists the reviewed static recordings.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "docs" / "protocols" / "static_session_cv_4fold",
    )
    parser.add_argument("--folds", type=int, default=4)
    args = parser.parse_args()
    if args.folds < 2:
        parser.error("--folds must be at least 2")
    if not args.source_manifest.exists():
        raise FileNotFoundError(args.source_manifest)

    source = pd.read_csv(args.source_manifest, encoding="utf-8-sig")
    required = set(RECORDING_COLUMNS)
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(f"Source manifest is missing columns: {sorted(missing)}")
    recordings = source.drop_duplicates(RECORDING_COLUMNS).copy()
    if recordings.duplicated(RECORDING_COLUMNS).any():
        raise RuntimeError("Source manifest contains duplicate recording identities.")
    if "Scenario" in recordings.columns and not recordings["Scenario"].astype(str).str.startswith("st_").all():
        raise ValueError("This generator only accepts static (st_*) recordings.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    assignments = build_fold_manifests(recordings, args.folds, args.output_dir)
    summary = (
        assignments.groupby(["cv_fold", "Environment"], dropna=False)
        .size()
        .rename("recording_count")
        .reset_index()
    )
    print(
        "Generated static Session-CV manifests:\n"
        f"  recordings={len(assignments)} folds={args.folds}\n"
        f"  assignment={args.output_dir / 'fold_assignment.csv'}\n"
        f"  fold manifests={args.output_dir / 'fold_<n>' / 'recording_split_manifest.csv'}\n"
        f"{summary.to_string(index=False)}"
    )


if __name__ == "__main__":
    main()

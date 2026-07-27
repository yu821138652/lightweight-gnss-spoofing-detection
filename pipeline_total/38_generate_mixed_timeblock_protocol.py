"""Generate a balanced outer-CV protocol for reviewed static and dynamic recordings.

The source manifest is explicit and recording-level.  Each complete recording
is assigned to exactly one outer test fold; every other recording is part of
that fold's development pool.  This script does not split development epochs
into train/validation blocks.  That inner split belongs to the tensor-building
stage, where window guards can be applied with the selected time-step length.

The assignment balances motion state, environment, frequency band, row count,
and positive-row count.  It first enforces an explicit worst-fold positive-row
deviation cap, then minimises the original weighted balance objective.  A
seeded multi-start search followed by subset-swap improvement makes the result
deterministic while preserving exact fold sizes.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RECORDING_COLUMNS = ["Environment", "Scenario", "Session"]
REQUIRED_COLUMNS = [*RECORDING_COLUMNS, "rows", "positive_rows"]
DERIVED_SOURCE_COLUMNS = {"split", "test_fold", "cv_fold", "motion"}
EXPECTED_BANDS = {"L1", "L5", "L15"}
DEFAULT_FOLDS = 4
DEFAULT_EXPECTED_RECORDINGS = 24
DEFAULT_SEED = 20260727
DEFAULT_RESTARTS = 128
# An exact set-partition audit of the reviewed v2 manifest found a 20.5833%
# minimax lower bound and a clear objective knee at the 25% cap.
DEFAULT_MAX_POSITIVE_ROW_DEVIATION = 0.25

# Marginal distributions are primary.  Interactions prevent a marginally
# balanced assignment from concentrating one motion/environment combination.
GROUP_WEIGHTS = {
    "motion": 2.0,
    "environment": 2.0,
    "band": 2.0,
    "motion_environment": 0.75,
    "motion_band": 0.75,
    "environment_band": 0.50,
    "motion_environment_band": 0.25,
}
CONTINUOUS_WEIGHTS = {
    "rows": 2.0,
    "positive_rows": 2.0,
    "positive_rate": 0.5,
}


@dataclass(frozen=True)
class BalanceModel:
    """Precomputed arrays used by the assignment objective."""

    folds: int
    fold_sizes: np.ndarray
    outer_group_names: list[str]
    outer_group_members: list[np.ndarray]
    group_matrices: dict[str, np.ndarray]
    group_values: dict[str, list[str]]
    rows: np.ndarray
    positive_rows: np.ndarray
    feature_matrix: np.ndarray
    feature_targets: np.ndarray
    feature_coefficients: np.ndarray
    row_feature_index: int
    positive_feature_index: int
    global_positive_rate: float


def _parse_boolean(value: Any, column: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in {0, 1}:
        return bool(value)
    if isinstance(value, (float, np.floating)) and np.isfinite(value) and value in {0.0, 1.0}:
        return bool(int(value))
    token = str(value).strip().lower()
    if token in {"true", "1", "yes", "dynamic", "dy"}:
        return True
    if token in {"false", "0", "no", "static", "st"}:
        return False
    raise ValueError(f"Column {column!r} contains an invalid boolean value: {value!r}")


def _scenario_motion(scenario: str) -> str:
    token = str(scenario).strip().lower()
    if re.match(r"^(st|static)(?:_|-)", token):
        return "static"
    if re.match(r"^(dy|dynamic)(?:_|-)", token):
        return "dynamic"
    raise ValueError(
        f"Cannot infer static/dynamic state from Scenario={scenario!r}; "
        "use an st_/static_ or dy_/dynamic_ prefix."
    )


def _normalise_band(value: Any) -> str:
    token = re.sub(r"[^A-Z0-9]", "", str(value).upper())
    if token in {"L15", "L1L5"}:
        return "L15"
    if token in {"L1", "L5"}:
        return token
    raise ValueError(f"Cannot normalise frequency band value {value!r} to L1, L5, or L15")


def _scenario_band(scenario: str) -> str:
    token = re.sub(r"^(st|static|dy|dynamic)[_-]", "", str(scenario).strip(), flags=re.IGNORECASE)
    return _normalise_band(token)


def _strict_nonnegative_integer(series: pd.Series, column: str, positive: bool = False) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any() or not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        bad = series.loc[numeric.isna()].head(5).tolist()
        raise ValueError(f"Column {column!r} contains non-numeric or non-finite values: {bad}")
    rounded = np.rint(numeric.to_numpy(dtype=np.float64))
    if not np.allclose(numeric.to_numpy(dtype=np.float64), rounded, rtol=0.0, atol=1e-9):
        raise ValueError(f"Column {column!r} must contain integer counts")
    result = pd.Series(rounded.astype(np.int64), index=series.index, name=column)
    lower_bound = 1 if positive else 0
    if (result < lower_bound).any():
        relation = "positive" if positive else "non-negative"
        raise ValueError(f"Column {column!r} must contain {relation} integer counts")
    return result


def read_source_manifest(
    path: Path,
    expected_recordings: int,
    folds: int,
) -> tuple[pd.DataFrame, list[str]]:
    """Read, normalise, and strictly validate a recording-level manifest."""

    source = pd.read_csv(path, encoding="utf-8-sig")
    return normalise_source_manifest(source, expected_recordings, folds, str(path))


def normalise_source_manifest(
    source: pd.DataFrame,
    expected_recordings: int,
    folds: int,
    source_name: str = "<in-memory>",
) -> tuple[pd.DataFrame, list[str]]:
    """Validate and normalise an already-loaded source manifest."""

    missing = set(REQUIRED_COLUMNS).difference(source.columns)
    if missing:
        raise ValueError(f"Source manifest is missing required columns: {sorted(missing)}")
    if source.empty:
        raise ValueError("Source manifest is empty")
    if len(source) != expected_recordings:
        raise ValueError(
            f"Expected exactly {expected_recordings} recording rows, found {len(source)} in {source_name}"
        )
    if expected_recordings < folds:
        raise ValueError("Expected recording count must be at least the number of folds")

    ignored_columns = sorted(DERIVED_SOURCE_COLUMNS.intersection(source.columns))
    frame = source.drop(columns=ignored_columns, errors="ignore").copy()
    for column in RECORDING_COLUMNS:
        text = frame[column].astype("string")
        invalid = text.isna() | text.str.strip().eq("")
        if invalid.any():
            raise ValueError(f"Column {column!r} contains missing or empty recording identities")
        if text.ne(text.str.strip()).any():
            raise ValueError(f"Column {column!r} contains leading or trailing whitespace")
        frame[column] = text.astype(str)
    if frame.duplicated(RECORDING_COLUMNS).any():
        duplicates = frame.loc[frame.duplicated(RECORDING_COLUMNS, keep=False), RECORDING_COLUMNS]
        raise ValueError(
            "Source manifest contains duplicate recording identities:\n"
            + duplicates.to_string(index=False)
        )

    if "outer_group" in frame.columns:
        outer_group = frame["outer_group"].astype("string")
        invalid = outer_group.isna() | outer_group.str.strip().eq("")
        if invalid.any():
            raise ValueError("outer_group contains missing or empty values")
        if outer_group.ne(outer_group.str.strip()).any():
            raise ValueError("outer_group contains leading or trailing whitespace")
        frame["outer_group"] = outer_group.astype(str)
    else:
        frame["outer_group"] = frame[RECORDING_COLUMNS].astype(str).agg("|".join, axis=1)
    environment_per_group = frame.groupby("outer_group")["Environment"].nunique()
    if (environment_per_group > 1).any():
        bad = environment_per_group[environment_per_group > 1].index.tolist()
        raise ValueError(f"outer_group cannot span multiple environments: {bad}")

    frame["rows"] = _strict_nonnegative_integer(frame["rows"], "rows", positive=True)
    frame["positive_rows"] = _strict_nonnegative_integer(frame["positive_rows"], "positive_rows")
    if (frame["positive_rows"] > frame["rows"]).any():
        raise ValueError("positive_rows cannot exceed rows")
    for status_column in ["LabelStatus", "label_status"]:
        if status_column in frame.columns:
            statuses = frame[status_column].astype(str).str.strip().str.lower()
            if not statuses.eq("reviewed").all():
                bad = sorted(frame.loc[statuses.ne("reviewed"), status_column].astype(str).unique())
                raise ValueError(
                    f"{status_column} must be 'reviewed' for every recording; found: {bad}"
                )

    inferred_motion = frame["Scenario"].map(_scenario_motion)
    if "is_dynamic" in frame.columns:
        declared_dynamic = frame["is_dynamic"].map(lambda value: _parse_boolean(value, "is_dynamic"))
        inferred_dynamic = inferred_motion.eq("dynamic")
        if not declared_dynamic.eq(inferred_dynamic).all():
            bad = frame.loc[declared_dynamic.ne(inferred_dynamic), RECORDING_COLUMNS + ["is_dynamic"]]
            raise ValueError(
                "is_dynamic conflicts with the Scenario prefix:\n" + bad.to_string(index=False)
            )
    else:
        declared_dynamic = inferred_motion.eq("dynamic")
    frame["is_dynamic"] = declared_dynamic.astype(bool)
    frame["motion"] = inferred_motion

    inferred_band = frame["Scenario"].map(_scenario_band)
    if "band" in frame.columns:
        declared_band = frame["band"].map(_normalise_band)
        if not declared_band.eq(inferred_band).all():
            bad = frame.loc[declared_band.ne(inferred_band), RECORDING_COLUMNS + ["band"]]
            raise ValueError("band conflicts with Scenario:\n" + bad.to_string(index=False))
    frame["band"] = inferred_band

    motions = set(frame["motion"])
    if motions != {"static", "dynamic"}:
        raise ValueError(f"Expected both static and dynamic recordings, got motion values: {sorted(motions)}")
    bands = set(frame["band"])
    if bands != EXPECTED_BANDS:
        raise ValueError(f"Expected L1, L5, and L15 recordings, got bands: {sorted(bands)}")
    environments = sorted(frame["Environment"].unique())
    if len(environments) < 2:
        raise ValueError(f"Expected at least two environments, got: {environments}")

    frame = frame.sort_values(RECORDING_COLUMNS, kind="mergesort").reset_index(drop=True)
    if "recording_id" in frame.columns:
        frame["recording_id"] = _strict_nonnegative_integer(frame["recording_id"], "recording_id")
        if frame["recording_id"].duplicated().any():
            raise ValueError("recording_id must be unique when supplied")
    else:
        frame.insert(0, "recording_id", np.arange(len(frame), dtype=np.int32))
    return frame, ignored_columns


def _one_hot(values: pd.Series) -> tuple[np.ndarray, list[str]]:
    categories = sorted(values.astype(str).unique())
    category_index = {value: index for index, value in enumerate(categories)}
    matrix = np.zeros((len(values), len(categories)), dtype=np.float64)
    for row_index, value in enumerate(values.astype(str)):
        matrix[row_index, category_index[value]] = 1.0
    return matrix, categories


def build_balance_model(frame: pd.DataFrame, folds: int) -> BalanceModel:
    quotient, remainder = divmod(len(frame), folds)
    fold_sizes = np.asarray(
        [quotient + (1 if fold < remainder else 0) for fold in range(folds)],
        dtype=np.int64,
    )
    group_series = {
        "motion": frame["motion"],
        "environment": frame["Environment"],
        "band": frame["band"],
        "motion_environment": frame["motion"] + "|" + frame["Environment"],
        "motion_band": frame["motion"] + "|" + frame["band"],
        "environment_band": frame["Environment"] + "|" + frame["band"],
        "motion_environment_band": (
            frame["motion"] + "|" + frame["Environment"] + "|" + frame["band"]
        ),
    }
    matrices: dict[str, np.ndarray] = {}
    values: dict[str, list[str]] = {}
    for name, series in group_series.items():
        matrices[name], values[name] = _one_hot(series)
    proportions = fold_sizes.astype(np.float64) / float(len(frame))
    feature_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    coefficient_parts: list[np.ndarray] = []
    for name, matrix in matrices.items():
        expected = proportions[:, None] * matrix.sum(axis=0, keepdims=True)
        coefficients = GROUP_WEIGHTS[name] / (
            folds * matrix.shape[1] * np.maximum(expected, 1.0)
        )
        feature_parts.append(matrix)
        target_parts.append(expected)
        coefficient_parts.append(coefficients)

    rows = frame["rows"].to_numpy(dtype=np.float64)
    positive_rows = frame["positive_rows"].to_numpy(dtype=np.float64)
    row_feature_index = sum(part.shape[1] for part in feature_parts)
    positive_feature_index = row_feature_index + 1
    expected_rows = proportions[:, None] * rows.sum()
    expected_positive = proportions[:, None] * positive_rows.sum()
    feature_parts.extend([rows[:, None], positive_rows[:, None]])
    target_parts.extend([expected_rows, expected_positive])
    coefficient_parts.extend(
        [
            CONTINUOUS_WEIGHTS["rows"] / (folds * np.maximum(expected_rows, 1.0) ** 2),
            CONTINUOUS_WEIGHTS["positive_rows"]
            / (folds * np.maximum(expected_positive, 1.0) ** 2),
        ]
    )
    grouped = frame.groupby("outer_group", sort=True).indices
    outer_group_names = [str(name) for name in grouped]
    outer_group_members = [np.asarray(grouped[name], dtype=np.int64) for name in grouped]
    if max(map(len, outer_group_members)) > int(fold_sizes.min()):
        raise ValueError("An outer_group is larger than the smallest test-fold capacity")
    return BalanceModel(
        folds=folds,
        fold_sizes=fold_sizes,
        outer_group_names=outer_group_names,
        outer_group_members=outer_group_members,
        group_matrices=matrices,
        group_values=values,
        rows=rows,
        positive_rows=positive_rows,
        feature_matrix=np.concatenate(feature_parts, axis=1),
        feature_targets=np.concatenate(target_parts, axis=1),
        feature_coefficients=np.concatenate(coefficient_parts, axis=1),
        row_feature_index=row_feature_index,
        positive_feature_index=positive_feature_index,
        global_positive_rate=float(positive_rows.sum() / max(rows.sum(), 1.0)),
    )


def objective_components(assignment: np.ndarray, model: BalanceModel) -> dict[str, float]:
    """Return interpretable terms of the weighted balance objective."""

    n = len(assignment)
    if set(assignment.tolist()) != set(range(model.folds)):
        raise ValueError("Every fold must be represented in an assignment")
    observed_sizes = np.bincount(assignment, minlength=model.folds)
    if not np.array_equal(observed_sizes, model.fold_sizes):
        raise ValueError(f"Assignment fold sizes {observed_sizes.tolist()} != {model.fold_sizes.tolist()}")
    proportions = model.fold_sizes.astype(np.float64) / float(n)
    components: dict[str, float] = {}
    for name, matrix in model.group_matrices.items():
        observed = np.zeros((model.folds, matrix.shape[1]), dtype=np.float64)
        np.add.at(observed, assignment, matrix)
        expected = proportions[:, None] * matrix.sum(axis=0, keepdims=True)
        components[name] = float(np.mean(np.square(observed - expected) / np.maximum(expected, 1.0)))

    row_sums = np.bincount(assignment, weights=model.rows, minlength=model.folds)
    positive_sums = np.bincount(assignment, weights=model.positive_rows, minlength=model.folds)
    expected_rows = proportions * model.rows.sum()
    expected_positive = proportions * model.positive_rows.sum()
    components["rows"] = float(
        np.mean(np.square((row_sums - expected_rows) / np.maximum(expected_rows, 1.0)))
    )
    components["positive_rows"] = float(
        np.mean(np.square((positive_sums - expected_positive) / np.maximum(expected_positive, 1.0)))
    )
    global_rate = float(model.positive_rows.sum() / max(model.rows.sum(), 1.0))
    fold_rates = positive_sums / np.maximum(row_sums, 1.0)
    components["positive_rate"] = float(
        np.mean(np.square((fold_rates - global_rate) / max(global_rate, 1e-12)))
    )
    return components


def objective_score(assignment: np.ndarray, model: BalanceModel) -> float:
    observed_sizes = np.bincount(assignment, minlength=model.folds)
    if not np.array_equal(observed_sizes, model.fold_sizes):
        raise ValueError(f"Assignment fold sizes {observed_sizes.tolist()} != {model.fold_sizes.tolist()}")
    loads = np.zeros_like(model.feature_targets)
    np.add.at(loads, assignment, model.feature_matrix)
    return float(sum(_fold_objective(loads[fold], fold, model) for fold in range(model.folds)))


def _fold_objective(load: np.ndarray, fold: int, model: BalanceModel) -> float:
    quadratic = np.sum(
        model.feature_coefficients[fold]
        * np.square(load - model.feature_targets[fold])
    )
    rows = load[model.row_feature_index]
    positives = load[model.positive_feature_index]
    rate = positives / max(rows, 1.0)
    relative_rate_error = (rate - model.global_positive_rate) / max(
        model.global_positive_rate, 1e-12
    )
    rate_term = (
        CONTINUOUS_WEIGHTS["positive_rate"]
        * relative_rate_error**2
        / model.folds
    )
    return float(quadratic + rate_term)


def _max_positive_row_deviation(loads: np.ndarray, model: BalanceModel) -> float:
    expected = model.feature_targets[:, model.positive_feature_index]
    observed = loads[:, model.positive_feature_index]
    deviations = np.abs(observed - expected) / np.maximum(expected, 1.0)
    return float(deviations.max())


def _search_rank(
    loads: np.ndarray,
    score: float,
    model: BalanceModel,
    max_positive_row_deviation: float | None,
) -> tuple[float, float]:
    """Rank candidates by cap violation, then by the original balance score."""

    if max_positive_row_deviation is None:
        return 0.0, float(score)
    violation = max(
        0.0,
        _max_positive_row_deviation(loads, model) - max_positive_row_deviation,
    )
    return float(violation), float(score)


def _canonicalise_fold_labels(assignment: np.ndarray, fold_sizes: np.ndarray) -> np.ndarray:
    """Remove arbitrary fold-label permutations from the deterministic result."""

    members = {
        fold: tuple(np.flatnonzero(assignment == fold).tolist())
        for fold in sorted(set(assignment.tolist()))
    }
    remap: dict[int, int] = {}
    observed_sizes = np.bincount(assignment, minlength=len(fold_sizes))
    for size in sorted(set(fold_sizes.tolist()), reverse=True):
        old_labels = [
            fold for fold in members if int(observed_sizes[fold]) == int(size)
        ]
        target_labels = np.flatnonzero(fold_sizes == size).tolist()
        ordered = sorted(old_labels, key=lambda fold: members[fold])
        if len(ordered) != len(target_labels):
            raise RuntimeError("Cannot canonicalise fold labels with unequal fold capacities")
        remap.update(zip(ordered, target_labels))
    return np.asarray([remap[int(value)] for value in assignment], dtype=np.int16)


def _random_group_assignment(
    model: BalanceModel, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray] | None:
    """Assign indivisible outer groups while filling exact recording capacities."""

    sizes = np.asarray([len(members) for members in model.outer_group_members], dtype=np.int64)
    shuffled = rng.permutation(len(sizes)).tolist()
    order = sorted(shuffled, key=lambda group: -int(sizes[group]))
    capacities = model.fold_sizes.copy()
    group_folds = np.full(len(sizes), -1, dtype=np.int16)
    for group in order:
        feasible = np.flatnonzero(capacities >= sizes[group])
        if not len(feasible):
            return None
        fold = int(rng.choice(feasible))
        group_folds[group] = fold
        capacities[fold] -= sizes[group]
    if np.any(capacities != 0):
        return None
    assignment = np.full(int(model.fold_sizes.sum()), -1, dtype=np.int16)
    for group, members in enumerate(model.outer_group_members):
        assignment[members] = group_folds[group]
    return assignment, group_folds


def optimise_assignment(
    model: BalanceModel,
    seed: int,
    restarts: int,
    max_positive_row_deviation: float | None = DEFAULT_MAX_POSITIVE_ROW_DEVIATION,
) -> tuple[np.ndarray, float]:
    """Search grouped assignments, enforcing the positive-row cap before score."""

    if restarts < 1:
        raise ValueError("search restarts must be at least 1")
    if max_positive_row_deviation is not None and max_positive_row_deviation < 0.0:
        raise ValueError("max positive-row deviation must be non-negative")
    rng = np.random.default_rng(seed)
    best_assignment: np.ndarray | None = None
    best_score = np.inf
    best_rank = (np.inf, np.inf)
    tolerance = 1e-12
    group_features = np.stack(
        [model.feature_matrix[members].sum(axis=0) for members in model.outer_group_members]
    )
    group_sizes = np.asarray([len(members) for members in model.outer_group_members])

    for _ in range(restarts):
        generated = _random_group_assignment(model, rng)
        if generated is None:
            continue
        candidate, group_folds = generated
        loads = np.zeros_like(model.feature_targets)
        np.add.at(loads, candidate, model.feature_matrix)
        fold_scores = np.asarray(
            [_fold_objective(loads[fold], fold, model) for fold in range(model.folds)]
        )
        current_score = float(fold_scores.sum())
        current_rank = _search_rank(
            loads,
            current_score,
            model,
            max_positive_row_deviation,
        )
        while True:
            best_swap: tuple[tuple[int, ...], tuple[int, ...]] | None = None
            swap_score = current_score
            best_left_load: np.ndarray | None = None
            best_right_load: np.ndarray | None = None
            best_left_score = 0.0
            best_right_score = 0.0
            swap_rank = current_rank
            groups_by_fold = {
                fold: np.flatnonzero(group_folds == fold).tolist()
                for fold in range(model.folds)
            }
            subset_cache: dict[int, list[tuple[tuple[int, ...], int, np.ndarray]]] = {}
            for fold, groups in groups_by_fold.items():
                subsets: list[tuple[tuple[int, ...], int, np.ndarray]] = []
                for count in (1, 2):
                    for selected in itertools.combinations(groups, count):
                        selected_array = np.asarray(selected, dtype=np.int64)
                        subsets.append(
                            (
                                selected,
                                int(group_sizes[selected_array].sum()),
                                group_features[selected_array].sum(axis=0),
                            )
                        )
                subset_cache[fold] = subsets
            for left_fold in range(model.folds - 1):
                for right_fold in range(left_fold + 1, model.folds):
                    for left_groups, left_size, left_features in subset_cache[left_fold]:
                        for right_groups, right_size, right_features in subset_cache[right_fold]:
                            if left_size != right_size:
                                continue
                            left_load = loads[left_fold] - left_features + right_features
                            right_load = loads[right_fold] - right_features + left_features
                            left_score = _fold_objective(left_load, left_fold, model)
                            right_score = _fold_objective(right_load, right_fold, model)
                            score = float(
                                current_score
                                - fold_scores[left_fold]
                                - fold_scores[right_fold]
                                + left_score
                                + right_score
                            )
                            candidate_loads = loads.copy()
                            candidate_loads[left_fold] = left_load
                            candidate_loads[right_fold] = right_load
                            rank = _search_rank(
                                candidate_loads,
                                score,
                                model,
                                max_positive_row_deviation,
                            )
                            if (
                                rank[0] < swap_rank[0] - tolerance
                                or (
                                    abs(rank[0] - swap_rank[0]) <= tolerance
                                    and rank[1] < swap_rank[1] - tolerance
                                )
                            ):
                                swap_score = score
                                swap_rank = rank
                                best_swap = (left_groups, right_groups)
                                best_left_load = left_load
                                best_right_load = right_load
                                best_left_score = left_score
                                best_right_score = right_score
            if best_swap is None:
                break
            left_groups, right_groups = best_swap
            left_fold = int(group_folds[left_groups[0]])
            right_fold = int(group_folds[right_groups[0]])
            assert best_left_load is not None and best_right_load is not None
            loads[left_fold] = best_left_load
            loads[right_fold] = best_right_load
            fold_scores[left_fold] = best_left_score
            fold_scores[right_fold] = best_right_score
            for group in left_groups:
                group_folds[group] = right_fold
                candidate[model.outer_group_members[group]] = right_fold
            for group in right_groups:
                group_folds[group] = left_fold
                candidate[model.outer_group_members[group]] = left_fold
            current_score = swap_score
            current_rank = swap_rank

        canonical = _canonicalise_fold_labels(candidate, model.fold_sizes)
        try:
            validate_assignment(canonical, model)
        except RuntimeError:
            continue
        canonical_score = objective_score(canonical, model)
        canonical_loads = np.zeros_like(model.feature_targets)
        np.add.at(canonical_loads, canonical, model.feature_matrix)
        canonical_rank = _search_rank(
            canonical_loads,
            canonical_score,
            model,
            max_positive_row_deviation,
        )
        canonical_key = tuple(canonical.tolist())
        incumbent_key = tuple(best_assignment.tolist()) if best_assignment is not None else ()
        if (
            canonical_rank[0] < best_rank[0] - tolerance
            or (
                abs(canonical_rank[0] - best_rank[0]) <= tolerance
                and canonical_rank[1] < best_rank[1] - tolerance
            )
            or (
                abs(canonical_rank[0] - best_rank[0]) <= tolerance
                and abs(canonical_rank[1] - best_rank[1]) <= tolerance
                and (best_assignment is None or canonical_key < incumbent_key)
            )
        ):
            best_assignment = canonical.copy()
            best_score = canonical_score
            best_rank = canonical_rank

    if best_assignment is None:
        raise RuntimeError("Unable to construct a capacity-feasible grouped fold assignment")
    if best_rank[0] > tolerance:
        actual = max_positive_row_deviation + best_rank[0]
        raise RuntimeError(
            "Unable to satisfy the maximum positive-row deviation after "
            f"{restarts} restarts: cap={max_positive_row_deviation:.6f}, "
            f"best={actual:.6f}"
        )
    return best_assignment, float(best_score)


def validate_assignment(
    assignment: np.ndarray,
    model: BalanceModel,
    max_positive_row_deviation: float | None = None,
) -> None:
    if len(assignment) != int(model.fold_sizes.sum()):
        raise RuntimeError("Assignment length changed during optimisation")
    counts = np.bincount(assignment, minlength=model.folds)
    if not np.array_equal(counts, model.fold_sizes):
        raise RuntimeError(f"Unexpected test fold sizes: {counts.tolist()}")
    if (assignment < 0).any() or (assignment >= model.folds).any():
        raise RuntimeError("Assignment contains an out-of-range fold")
    for name, members in zip(model.outer_group_names, model.outer_group_members):
        assigned_folds = np.unique(assignment[members])
        if len(assigned_folds) != 1:
            raise RuntimeError(f"outer_group {name!r} was split across folds")
    for dimension in ("motion", "environment", "band"):
        matrix = model.group_matrices[dimension]
        observed = np.zeros((model.folds, matrix.shape[1]), dtype=np.int64)
        np.add.at(observed, assignment, matrix.astype(np.int64))
        missing = np.argwhere(observed == 0)
        if missing.size:
            details = [
                f"fold {int(fold) + 1}: {model.group_values[dimension][int(category)]}"
                for fold, category in missing
            ]
            raise RuntimeError(
                f"Every fold must cover each {dimension} category; missing {details}"
            )
    scenario_matrix = model.group_matrices["motion_band"]
    scenario_totals = scenario_matrix.sum(axis=0).astype(np.int64)
    test_counts = np.zeros((model.folds, scenario_matrix.shape[1]), dtype=np.int64)
    np.add.at(test_counts, assignment, scenario_matrix.astype(np.int64))
    missing_from_development = np.argwhere(test_counts >= scenario_totals[None, :])
    if missing_from_development.size:
        details = [
            f"fold {int(fold) + 1}: {model.group_values['motion_band'][int(category)]}"
            for fold, category in missing_from_development
        ]
        raise RuntimeError(f"Every development pool must retain each Scenario; missing {details}")
    if max_positive_row_deviation is not None:
        loads = np.zeros_like(model.feature_targets)
        np.add.at(loads, assignment, model.feature_matrix)
        actual = _max_positive_row_deviation(loads, model)
        if actual > max_positive_row_deviation + 1e-12:
            raise RuntimeError(
                "Assignment exceeds the maximum positive-row deviation: "
                f"cap={max_positive_row_deviation:.6f}, actual={actual:.6f}"
            )


def _safe_column_token(value: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    return token or "unknown"


def build_summaries(
    frame: pd.DataFrame,
    assignment: np.ndarray,
    model: BalanceModel,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    category_rows: list[dict[str, Any]] = []
    total_rows = int(frame["rows"].sum())
    total_positive = int(frame["positive_rows"].sum())

    category_dimensions = {
        "motion": frame["motion"],
        "Environment": frame["Environment"],
        "band": frame["band"],
    }
    for fold_zero in range(model.folds):
        fold = fold_zero + 1
        test = frame.loc[assignment == fold_zero]
        target_fraction = model.fold_sizes[fold_zero] / len(frame)
        test_rows = int(test["rows"].sum())
        test_positive = int(test["positive_rows"].sum())
        expected_rows = total_rows * target_fraction
        expected_positive = total_positive * target_fraction
        row: dict[str, Any] = {
            "fold": fold,
            "test_recordings": int(len(test)),
            "development_recordings": int(len(frame) - len(test)),
            "test_rows": test_rows,
            "test_positive_rows": test_positive,
            "test_positive_rate": test_positive / max(test_rows, 1),
            "row_deviation_pct": 100.0 * (test_rows - expected_rows) / max(expected_rows, 1.0),
            "positive_row_deviation_pct": (
                100.0 * (test_positive - expected_positive) / max(expected_positive, 1.0)
            ),
        }
        for dimension, series in category_dimensions.items():
            values = sorted(series.astype(str).unique())
            for value in values:
                observed = int((test[dimension] == value).sum())
                expected = float((series.astype(str) == value).sum() * target_fraction)
                prefix = "environment" if dimension == "Environment" else dimension
                row[f"test_{prefix}_{_safe_column_token(value)}"] = observed
                category_rows.append(
                    {
                        "fold": fold,
                        "dimension": dimension,
                        "value": value,
                        "test_recordings": observed,
                        "expected_recordings": expected,
                        "difference": observed - expected,
                    }
                )
        summary_rows.append(row)
    return pd.DataFrame(summary_rows), pd.DataFrame(category_rows)


def _manifest_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assignment_sha256(frame: pd.DataFrame) -> str:
    columns = [*RECORDING_COLUMNS, "outer_group", "test_fold"]
    payload = frame[columns].to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_protocol(
    frame: pd.DataFrame,
    assignment: np.ndarray,
    summary: pd.DataFrame,
    category_summary: pd.DataFrame,
    output_dir: Path,
    metadata: dict[str, Any],
    overwrite: bool,
) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Use --overwrite only after checking it."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    assignments = frame.copy()
    assignments["test_fold"] = assignment + 1
    assignments.to_csv(output_dir / "fold_assignment.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "fold_balance_summary.csv", index=False, encoding="utf-8-sig")
    category_summary.to_csv(
        output_dir / "fold_balance_categories.csv", index=False, encoding="utf-8-sig"
    )

    test_coverage: list[tuple[str, str, str]] = []
    for fold_zero in range(metadata["outer_folds"]):
        fold_dir = output_dir / f"fold_{fold_zero + 1}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        manifest = assignments.copy()
        manifest["split"] = np.where(
            assignment == fold_zero,
            "test",
            "development",
        )
        if int(manifest["split"].eq("test").sum()) != int(metadata["fold_sizes"][fold_zero]):
            raise RuntimeError(f"Fold {fold_zero + 1} has an unexpected test recording count")
        test_coverage.extend(
            map(tuple, manifest.loc[manifest["split"].eq("test"), RECORDING_COLUMNS].to_numpy())
        )
        manifest.to_csv(
            fold_dir / "recording_split_manifest.csv", index=False, encoding="utf-8-sig"
        )
    if len(test_coverage) != len(frame) or len(set(test_coverage)) != len(frame):
        raise RuntimeError("Every recording must occur in exactly one outer test fold")

    (output_dir / "protocol_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _synthetic_manifest() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    index = 0
    for environment_index, environment in enumerate(["env_a", "env_b"]):
        for motion_prefix in ["st", "dy"]:
            for band in ["L1", "L5", "L_15"]:
                for repeat in range(2):
                    row_count = 50_000 + 7_000 * index + 3_000 * environment_index
                    positive = int(row_count * (0.08 + 0.02 * ((index + repeat) % 5)))
                    rows.append(
                        {
                            "Environment": environment,
                            "Scenario": f"{motion_prefix}_{band}",
                            "Session": f"session_{index:02d}",
                            "rows": row_count,
                            "positive_rows": positive,
                        }
                    )
                    index += 1
    return pd.DataFrame(rows)


def run_self_test(seed: int, restarts: int) -> None:
    source = _synthetic_manifest()
    source["is_dynamic"] = source["Scenario"].str.startswith("dy_")
    frame, ignored = normalise_source_manifest(
        source,
        expected_recordings=DEFAULT_EXPECTED_RECORDINGS,
        folds=DEFAULT_FOLDS,
    )
    if ignored:
        raise AssertionError(f"Unexpected ignored columns in synthetic input: {ignored}")
    model = build_balance_model(frame, DEFAULT_FOLDS)
    first, first_score = optimise_assignment(model, seed, restarts)
    second, second_score = optimise_assignment(model, seed, restarts)
    validate_assignment(first, model, DEFAULT_MAX_POSITIVE_ROW_DEVIATION)
    first_loads = np.zeros_like(model.feature_targets)
    np.add.at(first_loads, first, model.feature_matrix)
    if _max_positive_row_deviation(first_loads, model) > DEFAULT_MAX_POSITIVE_ROW_DEVIATION + 1e-12:
        raise AssertionError("Synthetic assignment exceeds the positive-row deviation cap")
    if not np.array_equal(first, second) or first_score != second_score:
        raise AssertionError("Seeded optimisation is not deterministic")
    components = objective_components(first, model)
    reconstructed_score = sum(
        GROUP_WEIGHTS[name] * components[name] for name in GROUP_WEIGHTS
    ) + sum(CONTINUOUS_WEIGHTS[name] * components[name] for name in CONTINUOUS_WEIGHTS)
    if not np.isclose(first_score, reconstructed_score, rtol=0.0, atol=1e-12):
        raise AssertionError("Fast swap objective differs from the reported objective components")
    for column in ["motion", "Environment", "band"]:
        table = pd.crosstab(first + 1, frame[column])
        if int(table.max().max() - table.min().min()) > 1:
            raise AssertionError(f"Synthetic assignment is unexpectedly imbalanced for {column}")
    grouped_source = source.copy()
    grouped_source["outer_group"] = [f"group_{index:02d}" for index in range(len(grouped_source))]
    grouped_source.loc[[0, 1], "outer_group"] = "paired_event"
    grouped_frame, _ = normalise_source_manifest(
        grouped_source,
        expected_recordings=DEFAULT_EXPECTED_RECORDINGS,
        folds=DEFAULT_FOLDS,
    )
    grouped_model = build_balance_model(grouped_frame, DEFAULT_FOLDS)
    grouped_assignment, _ = optimise_assignment(grouped_model, seed, restarts)
    validate_assignment(grouped_assignment, grouped_model)
    paired_folds = grouped_assignment[grouped_frame["outer_group"].eq("paired_event").to_numpy()]
    if len(np.unique(paired_folds)) != 1:
        raise AssertionError("Synthetic outer_group was split across folds")
    bad = frame.copy()
    bad.loc[0, "positive_rows"] = bad.loc[0, "rows"] + 1
    try:
        normalise_source_manifest(bad, DEFAULT_EXPECTED_RECORDINGS, DEFAULT_FOLDS)
    except ValueError as error:
        if "positive_rows cannot exceed rows" not in str(error):
            raise
    else:
        raise AssertionError("Synthetic invalid-count validation did not trigger")
    print(f"Self-test passed: recordings={len(frame)} folds={model.folds} score={first_score:.8f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-recording-manifest",
        "--source-manifest",
        dest="source_recording_manifest",
        type=Path,
        help=(
            "Explicit recording-level CSV with Environment, Scenario, Session, rows, "
            "and positive_rows; repeated outer_group values keep one acquisition event intact."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output" / "protocols" / "mixed_timeblock_outer_cv4_v2",
    )
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--expected-recordings", type=int, default=DEFAULT_EXPECTED_RECORDINGS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--search-restarts", type=int, default=DEFAULT_RESTARTS)
    parser.add_argument(
        "--max-positive-row-deviation",
        type=float,
        default=DEFAULT_MAX_POSITIVE_ROW_DEVIATION,
        help=(
            "Maximum absolute relative deviation of test positive rows from the fold target "
            f"(default: {DEFAULT_MAX_POSITIVE_ROW_DEVIATION:.2f})."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and optimise without writing files.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing files in a non-empty output directory.")
    parser.add_argument("--self-test", action="store_true", help="Run a deterministic in-memory smoke test and exit.")
    args = parser.parse_args()
    if args.folds < 2:
        parser.error("--folds must be at least 2")
    if args.expected_recordings < args.folds:
        parser.error("--expected-recordings must be at least --folds")
    if args.search_restarts < 1:
        parser.error("--search-restarts must be at least 1")
    if not np.isfinite(args.max_positive_row_deviation) or args.max_positive_row_deviation < 0.0:
        parser.error("--max-positive-row-deviation must be finite and non-negative")
    if not args.self_test and args.source_recording_manifest is None:
        parser.error("--source-recording-manifest is required unless --self-test is used")
    if args.dry_run and args.overwrite:
        parser.error("--dry-run and --overwrite cannot be used together")
    return args


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test(args.seed, args.search_restarts)
        return

    source_path = args.source_recording_manifest.resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    frame, ignored_columns = read_source_manifest(
        source_path,
        expected_recordings=int(args.expected_recordings),
        folds=int(args.folds),
    )
    model = build_balance_model(frame, int(args.folds))
    assignment, score = optimise_assignment(
        model,
        int(args.seed),
        int(args.search_restarts),
        float(args.max_positive_row_deviation),
    )
    validate_assignment(
        assignment,
        model,
        float(args.max_positive_row_deviation),
    )
    summary, category_summary = build_summaries(frame, assignment, model)
    assigned_frame = frame.copy()
    assigned_frame["test_fold"] = assignment + 1
    components = objective_components(assignment, model)
    assignment_loads = np.zeros_like(model.feature_targets)
    np.add.at(assignment_loads, assignment, model.feature_matrix)
    actual_max_positive_deviation = _max_positive_row_deviation(assignment_loads, model)
    metadata: dict[str, Any] = {
        "protocol": "mixed_timeblock_outer_cv4_v2",
        "protocol_version": 2,
        "source_recording_manifest": str(source_path),
        "source_manifest_sha256": _manifest_sha256(source_path),
        "recordings": int(len(frame)),
        "outer_groups": int(len(model.outer_group_names)),
        "outer_folds": int(model.folds),
        "fold_sizes": model.fold_sizes.astype(int).tolist(),
        "seed": int(args.seed),
        "search_restarts": int(args.search_restarts),
        "max_positive_row_deviation_cap": float(args.max_positive_row_deviation),
        "actual_max_positive_row_deviation": actual_max_positive_deviation,
        "assignment_objective": float(score),
        "objective_components": components,
        "group_weights": GROUP_WEIGHTS,
        "continuous_weights": CONTINUOUS_WEIGHTS,
        "assignment_sha256": _assignment_sha256(assigned_frame),
        "recording_key": RECORDING_COLUMNS,
        "outer_group_key": "outer_group",
        "source_columns_ignored": ignored_columns,
        "outer_split_values": ["development", "test"],
        "inner_split": "deferred_to_time-block tensor construction",
        "invariants": [
            "each complete recording is test in exactly one fold",
            "recordings from one outer_group are test in the same fold",
            "no recording is split across outer folds",
            "all non-test recordings are development",
            "every fold test covers static/dynamic, both environments, and L1/L5/L15",
            "every fold development pool retains all six motion-by-band Scenarios",
            "every fold test positive-row count is within the configured relative-deviation cap",
        ],
    }

    print("Balanced mixed static/dynamic outer-CV assignment")
    print(
        f"  recordings={len(frame)} folds={model.folds} "
        f"fold_sizes={model.fold_sizes.tolist()} seed={args.seed} score={score:.8f} "
        f"max_positive_deviation={actual_max_positive_deviation:.4%}"
    )
    print(summary.to_string(index=False))
    if args.dry_run:
        print("Dry run complete; no files were written.")
        return
    write_protocol(
        frame=frame,
        assignment=assignment,
        summary=summary,
        category_summary=category_summary,
        output_dir=args.output_dir.resolve(),
        metadata=metadata,
        overwrite=bool(args.overwrite),
    )
    print(f"Protocol written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

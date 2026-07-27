"""Aggregate locked mixed signal-level outer-CV test reports.

The input reports are produced independently by
``39_evaluate_mixed_signal_groups.py``.  This script never loads tensors or
checkpoints and cannot change predictions.  It validates the fold reports,
sums their confusion matrices for pooled metrics, and also reports fold- and
complete-Session-equal summaries.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


COUNT_COLUMNS = (
    "samples",
    "negative_support",
    "positive_support",
    "tn",
    "fp",
    "fn",
    "tp",
)
METRIC_COLUMNS = ("macro_f1", "precision", "recall", "far", "specificity")
IDENTITY_COLUMNS = ("Environment", "Scenario", "Session")
REQUIRED_COLUMNS = {
    "group_level",
    "motion",
    "device_id",
    "device_name",
    "band",
    "session_count",
    "raw_feature_set",
    "stats_feature_set",
    *IDENTITY_COLUMNS,
    *COUNT_COLUMNS,
    *METRIC_COLUMNS,
}
POOLED_GROUP_SPECS = {
    "overall": ("overall", ()),
    "motion": ("motion", ("motion",)),
    "scenario": ("scenario", ("Scenario", "motion")),
    "environment_motion": ("session", ("Environment", "motion")),
    "environment_scenario": ("session", ("Environment", "Scenario", "motion")),
    "device_motion_band": (
        "device_motion_band",
        ("device_id", "device_name", "motion", "band"),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--model-subdir", default="tcn")
    parser.add_argument("--fold-assignment", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if not args.model_subdir or Path(args.model_subdir).name != args.model_subdir:
        parser.error("--model-subdir must be one directory name")
    if args.output_dir is None:
        args.output_dir = args.training_root
    return args


def _safe_divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _metrics_from_counts(counts: dict[str, int]) -> dict[str, float | int]:
    tn, fp, fn, tp = (counts[name] for name in ("tn", "fp", "fn", "tp"))
    negative_support = tn + fp
    positive_support = fn + tp
    samples = negative_support + positive_support
    if counts["samples"] != samples:
        raise ValueError(f"Confusion counts do not sum to samples: {counts}")
    if counts["negative_support"] != negative_support:
        raise ValueError(f"Negative support differs from tn + fp: {counts}")
    if counts["positive_support"] != positive_support:
        raise ValueError(f"Positive support differs from fn + tp: {counts}")
    positive_f1 = _safe_divide(2 * tp, 2 * tp + fp + fn)
    negative_f1 = _safe_divide(2 * tn, 2 * tn + fp + fn)
    return {
        **counts,
        "precision": _safe_divide(tp, tp + fp),
        "recall": _safe_divide(tp, positive_support),
        "far": _safe_divide(fp, negative_support),
        "macro_f1": (positive_f1 + negative_f1) / 2,
        "specificity": _safe_divide(tn, negative_support),
    }


def _statistics(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _equal_weight_summary(rows: pd.DataFrame) -> dict[str, Any]:
    def metric_summary(selected: pd.DataFrame) -> dict[str, Any]:
        return {
            "count": int(len(selected)),
            "metrics": {
                metric: _statistics(selected[metric].astype(float).tolist())
                for metric in METRIC_COLUMNS
            },
        }

    positive_rows = rows.loc[rows["positive_support"] > 0]
    summary = metric_summary(rows)
    summary.update({
        "count": int(len(rows)),
        "positive_count": int((rows["positive_support"] > 0).sum()),
        "negative_count": int((rows["negative_support"] > 0).sum()),
        "positive_support_only": metric_summary(positive_rows),
    })
    return summary


def _fold_number(path: Path) -> int:
    match = re.fullmatch(r"fold_(\d+)", path.parent.parent.name)
    if match is None:
        raise ValueError(f"Cannot infer fold number from {path}")
    return int(match.group(1))


def _load_feature_contract(metrics_path: Path) -> dict[str, Any]:
    detail_path = metrics_path.with_name("test_metrics_signal_tcn_stats_mlp_fusion.json")
    if not detail_path.is_file():
        raise FileNotFoundError(detail_path)
    detail = json.loads(detail_path.read_text(encoding="utf-8"))
    required = {
        "parameter_count",
        "raw_feature_set",
        "raw_feature_names",
        "stats_feature_set",
        "stats_feature_names",
    }
    missing = sorted(required.difference(detail))
    if missing:
        raise ValueError(f"{detail_path} lacks feature contract fields: {missing}")
    return {key: detail[key] for key in sorted(required)}


def _validate_report_metrics(report: pd.DataFrame, path: Path) -> None:
    for row_index, row in report.iterrows():
        counts = {column: int(row[column]) for column in COUNT_COLUMNS}
        recomputed = _metrics_from_counts(counts)
        for metric in METRIC_COLUMNS:
            if not np.isclose(
                float(row[metric]), float(recomputed[metric]), rtol=0.0, atol=1e-12
            ):
                raise ValueError(
                    f"{path} row {row_index + 2} has {metric}={row[metric]}, "
                    f"expected {recomputed[metric]} from its confusion matrix"
                )


def _validate_fold_partitions(report: pd.DataFrame, path: Path) -> None:
    sessions = report.loc[report["group_level"] == "session"]
    if not sessions["session_count"].eq(1).all():
        raise ValueError(f"{path} complete-Session rows must each have session_count=1")

    def validate_level(level: str, keys: tuple[str, ...]) -> None:
        summaries = report.loc[report["group_level"] == level]
        if keys:
            expected_keys = {
                tuple(str(value) for value in values)
                for values in sessions[list(keys)].drop_duplicates().itertuples(index=False, name=None)
            }
            actual_keys = {
                tuple(str(row[key]) for key in keys)
                for _, row in summaries.iterrows()
            }
            if expected_keys != actual_keys or len(summaries) != len(actual_keys):
                raise ValueError(f"{path} has incomplete or duplicate {level} partition rows")
        elif len(summaries) != 1:
            raise ValueError(f"{path} must contain exactly one {level} row")

        for _, summary in summaries.iterrows():
            selected = sessions
            for key in keys:
                selected = selected.loc[selected[key].astype(str) == str(summary[key])]
            if selected.empty:
                raise ValueError(f"{path} {level} row does not select any Session rows")
            if int(summary["session_count"]) != len(selected):
                raise ValueError(f"{path} {level} session_count differs from its Session rows")
            for column in COUNT_COLUMNS:
                expected = int(selected[column].sum())
                if int(summary[column]) != expected:
                    raise ValueError(
                        f"{path} {level} {column}={summary[column]}, "
                        f"expected {expected} from its Session rows"
                    )

    validate_level("overall", ())
    validate_level("motion", ("motion",))
    validate_level("scenario", ("Scenario", "motion"))


def _load_reports(training_root: Path, model_subdir: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    paths = sorted(
        training_root.glob(f"fold_*/{model_subdir}/test_metrics_mixed_groups.csv"),
        key=_fold_number,
    )
    if not paths:
        raise FileNotFoundError(
            training_root / f"fold_*/{model_subdir}/test_metrics_mixed_groups.csv"
        )
    folds = [_fold_number(path) for path in paths]
    if folds != list(range(1, len(folds) + 1)):
        raise ValueError(f"Fold numbers must be contiguous from 1, found {folds}")

    reports: list[pd.DataFrame] = []
    contracts: list[dict[str, Any]] = []
    for fold, path in zip(folds, paths):
        report = pd.read_csv(path, encoding="utf-8-sig")
        missing = sorted(REQUIRED_COLUMNS.difference(report.columns))
        if missing:
            raise ValueError(f"{path} lacks columns: {missing}")
        for column in (*COUNT_COLUMNS, "session_count"):
            report[column] = pd.to_numeric(report[column], errors="raise").astype(np.int64)
        for column in METRIC_COLUMNS:
            report[column] = pd.to_numeric(report[column], errors="raise").astype(float)
        if len(report.loc[report["group_level"] == "overall"]) != 1:
            raise ValueError(f"{path} must contain exactly one overall row")
        session_rows = report.loc[report["group_level"] == "session"]
        expected_sessions = int(report.loc[report["group_level"] == "overall", "session_count"].iloc[0])
        if len(session_rows) != expected_sessions:
            raise ValueError(
                f"{path} reports {expected_sessions} Sessions but has {len(session_rows)} Session rows"
            )
        if session_rows.duplicated(list(IDENTITY_COLUMNS)).any():
            raise ValueError(f"{path} contains duplicate complete-Session rows")
        _validate_report_metrics(report, path)
        _validate_fold_partitions(report, path)
        report.insert(0, "fold", fold)
        reports.append(report)
        contracts.append(_load_feature_contract(path))

    if any(contract != contracts[0] for contract in contracts[1:]):
        raise ValueError("Fold checkpoints do not share one feature/model contract")
    return pd.concat(reports, ignore_index=True), contracts[0]


def _validate_assignment(session_rows: pd.DataFrame, path: Path) -> None:
    assignment = pd.read_csv(path, encoding="utf-8-sig")
    required = {*IDENTITY_COLUMNS, "test_fold"}
    missing = sorted(required.difference(assignment.columns))
    if missing:
        raise ValueError(f"{path} lacks columns: {missing}")
    assignment["test_fold"] = pd.to_numeric(assignment["test_fold"], errors="raise").astype(int)
    if assignment.duplicated(list(IDENTITY_COLUMNS)).any():
        raise ValueError(f"{path} contains duplicate recording identities")
    actual = session_rows[[*IDENTITY_COLUMNS, "fold"]].copy()
    expected = assignment[[*IDENTITY_COLUMNS, "test_fold"]].rename(columns={"test_fold": "fold"})
    merged = expected.merge(actual, on=list(IDENTITY_COLUMNS), how="outer", suffixes=("_expected", "_actual"), indicator=True)
    if not (merged["_merge"] == "both").all():
        raise ValueError("Evaluated Sessions differ from the protocol fold assignment")
    if not (merged["fold_expected"] == merged["fold_actual"]).all():
        raise ValueError("At least one Session was evaluated in the wrong outer fold")


def _pooled_rows(reports: pd.DataFrame) -> pd.DataFrame:
    output: list[dict[str, Any]] = []
    profile = reports[["raw_feature_set", "stats_feature_set"]].drop_duplicates()
    if len(profile) != 1:
        raise ValueError("Fold reports mix multiple feature profiles")
    raw_feature_set, stats_feature_set = profile.iloc[0].tolist()

    for level, (source_level, keys) in POOLED_GROUP_SPECS.items():
        source = reports.loc[reports["group_level"] == source_level]
        groups = [((), source)] if not keys else source.groupby(list(keys), dropna=False, sort=True)
        for key_values, rows in groups:
            if keys and not isinstance(key_values, tuple):
                key_values = (key_values,)
            dimensions = dict(zip(keys, key_values))
            counts = {column: int(rows[column].sum()) for column in COUNT_COLUMNS}
            row: dict[str, Any] = {
                "group_level": level,
                "Environment": "ALL",
                "Scenario": "ALL",
                "Session": "ALL",
                "motion": "ALL",
                "device_id": "ALL",
                "device_name": "ALL",
                "band": "ALL",
                "session_count": int(rows["session_count"].sum()),
                "raw_feature_set": raw_feature_set,
                "stats_feature_set": stats_feature_set,
            }
            row.update(dimensions)
            row.update(_metrics_from_counts(counts))
            output.append(row)
    return pd.DataFrame(output)


def _fold_table(reports: pd.DataFrame) -> pd.DataFrame:
    folds = reports.loc[reports["group_level"] == "overall"].copy()
    columns = ["fold", "session_count", *COUNT_COLUMNS, *METRIC_COLUMNS]
    return folds[columns].sort_values("fold").reset_index(drop=True)


def _summary(
    reports: pd.DataFrame,
    pooled: pd.DataFrame,
    feature_contract: dict[str, Any],
    training_root: Path,
) -> dict[str, Any]:
    folds = _fold_table(reports)
    sessions = reports.loc[reports["group_level"] == "session"].copy()
    overall = pooled.loc[pooled["group_level"] == "overall"].iloc[0]
    by_motion = {
        motion: _equal_weight_summary(sessions.loc[sessions["motion"] == motion])
        for motion in ("static", "dynamic")
    }
    by_scenario = {
        str(scenario): _equal_weight_summary(rows)
        for scenario, rows in sessions.groupby("Scenario", sort=True)
    }
    worst_columns = [
        "fold",
        *IDENTITY_COLUMNS,
        "motion",
        "samples",
        "positive_support",
        *METRIC_COLUMNS,
    ]
    worst = sessions.sort_values(["macro_f1", "fold"]).head(8)[worst_columns]
    pooled_overall = {key: int(overall[key]) for key in COUNT_COLUMNS}
    pooled_overall.update({key: float(overall[key]) for key in METRIC_COLUMNS})
    return {
        "aggregation": {
            "pooled": "sum disjoint outer-test confusion matrices, then recompute metrics",
            "fold_equal": "unweighted arithmetic summary of outer-fold overall metrics",
            "session_equal": "unweighted arithmetic summary of complete-Session metrics",
            "metric_std": "population standard deviation (ddof=0)",
            "zero_division": 0,
            "decision_rule": "locked checkpoint argmax; no test-time threshold tuning",
        },
        "training_root": str(training_root),
        "fold_count": int(folds["fold"].nunique()),
        "session_count": int(len(sessions)),
        "feature_contract": feature_contract,
        "pooled_overall": pooled_overall,
        "fold_equal": {
            "count": int(len(folds)),
            "metrics": {
                metric: _statistics(folds[metric].astype(float).tolist())
                for metric in METRIC_COLUMNS
            },
        },
        "session_equal": {
            "overall": _equal_weight_summary(sessions),
            "by_motion": by_motion,
            "by_scenario": by_scenario,
        },
        "worst_sessions_by_macro_f1": worst.to_dict(orient="records"),
    }


def main() -> None:
    args = parse_args()
    reports, feature_contract = _load_reports(args.training_root, args.model_subdir)
    sessions = reports.loc[reports["group_level"] == "session"]
    if sessions.duplicated(list(IDENTITY_COLUMNS)).any():
        raise ValueError("A complete Session appears in more than one outer test fold")
    if args.fold_assignment is not None:
        _validate_assignment(sessions, args.fold_assignment)

    pooled = _pooled_rows(reports)
    folds = _fold_table(reports)
    session_columns = [
        "fold",
        *IDENTITY_COLUMNS,
        "motion",
        "session_count",
        "raw_feature_set",
        "stats_feature_set",
        *COUNT_COLUMNS,
        *METRIC_COLUMNS,
    ]
    session_table = sessions[session_columns].sort_values(["fold", *IDENTITY_COLUMNS])
    summary = _summary(reports, pooled, feature_contract, args.training_root)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pooled_path = args.output_dir / "cv_test_pooled_group_metrics.csv"
    fold_path = args.output_dir / "cv_test_fold_metrics.csv"
    session_path = args.output_dir / "cv_test_session_metrics.csv"
    summary_path = args.output_dir / "cv_test_summary.json"
    pooled.to_csv(pooled_path, index=False, encoding="utf-8-sig")
    folds.to_csv(fold_path, index=False, encoding="utf-8-sig")
    session_table.to_csv(session_path, index=False, encoding="utf-8-sig")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"wrote {pooled_path} ({len(pooled)} pooled groups)")
    print(f"wrote {fold_path} ({len(folds)} folds)")
    print(f"wrote {session_path} ({len(session_table)} Sessions)")
    print(f"wrote {summary_path}")
    print(json.dumps(summary["pooled_overall"], ensure_ascii=False))


if __name__ == "__main__":
    main()

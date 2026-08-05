"""Summarize aggregate band-mean predictions with audited recording metadata.

The tensor builder assigns ``recording_id`` after applying its scope filter.
Consequently, static-only predictions use IDs local to the static recordings,
whereas mixed predictions use IDs over the full recording catalog.  This
script reconstructs the possible local ID mappings from the four-fold protocol
manifests and accepts a mapping only when every prediction ``(fold,
recording_id)`` resolves to a test recording in that fold.

New exports are also required to have unique source/device/receiver-time keys.
Legacy exports lack source IDs, so their weaker recording/device/TOW key is
reported with its collision count rather than being treated as unique.

Outputs contain fixed four-class metrics for the overall result, each true
class, each device, and each recording.  Per-device and per-recording tables
also expose recall/support for every true class and a complete 4x4 confusion
matrix.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


LABELS = [0, 1, 2, 3]
CLASS_NAMES = {0: "normal", 1: "L1", 2: "L5", 3: "L1+L5"}
CLASS_SLUGS = {0: "normal", 1: "L1", 2: "L5", 3: "L1_L5"}
RECORDING_KEYS = ["Environment", "Scenario", "Session"]
PREDICTION_REQUIRED = {
    "fold",
    "recording_id",
    "device_id",
    "device_name",
    "endpoint_tow",
    "true_class",
    "pred_class",
}
MANIFEST_REQUIRED = {
    "recording_id",
    *RECORDING_KEYS,
    "test_fold",
    "split",
    "outer_test",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
LOG = logging.getLogger(__name__)


def integer_column(frame: pd.DataFrame, column: str, context: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="raise")
    numeric = values.to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError(f"{context} column {column!r} must contain finite integers")
    return values.astype(np.int64)


def load_predictions(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, encoding="utf-8-sig")
    missing = PREDICTION_REQUIRED.difference(frame.columns)
    if missing:
        raise ValueError(f"Predictions {path} missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"Predictions {path} are empty")
    for column in ("fold", "recording_id", "device_id", "true_class", "pred_class"):
        frame[column] = integer_column(frame, column, f"Predictions {path}")
    has_source_key = {"source_id", "window_time_nanos"}.issubset(frame.columns)
    has_partial_source_key = bool(
        {"source_id", "window_time_nanos"}.intersection(frame.columns)
    )
    if has_partial_source_key and not has_source_key:
        raise ValueError(
            f"Predictions {path} must contain both source_id and window_time_nanos"
        )
    if has_source_key:
        for column in ("source_id", "window_time_nanos"):
            frame[column] = integer_column(frame, column, f"Predictions {path}")
    frame["endpoint_tow"] = pd.to_numeric(frame["endpoint_tow"], errors="raise")
    if not np.isfinite(frame["endpoint_tow"].to_numpy(dtype=np.float64)).all():
        raise ValueError(f"Predictions {path} endpoint_tow must be finite")
    for column in ("true_class", "pred_class"):
        unexpected = sorted(set(frame[column].unique()).difference(LABELS))
        if unexpected:
            raise ValueError(f"Predictions {path} contain invalid {column}: {unexpected}")
    frame["device_name"] = frame["device_name"].astype(str).str.strip()
    if frame["device_name"].eq("").any():
        raise ValueError(f"Predictions {path} contain blank device_name values")

    device_pairs = frame[["device_id", "device_name"]].drop_duplicates()
    if device_pairs["device_id"].duplicated(keep=False).any():
        raise ValueError("One device_id maps to multiple device_name values")
    if device_pairs["device_name"].duplicated(keep=False).any():
        raise ValueError("One device_name maps to multiple device_id values")
    prediction_key = (
        ["fold", "source_id", "device_id", "window_time_nanos"]
        if has_source_key
        else ["fold", "recording_id", "device_id", "endpoint_tow"]
    )
    duplicate = frame.duplicated(prediction_key, keep=False)
    if has_source_key and duplicate.any():
        sample = frame.loc[duplicate, prediction_key].head(5).to_dict("records")
        raise ValueError(f"Predictions {path} contain duplicate endpoint keys: {sample}")
    frame.attrs["prediction_key"] = prediction_key
    frame.attrs["prediction_key_strict"] = bool(has_source_key)
    frame.attrs["legacy_key_collision_rows"] = int(duplicate.sum())
    return frame


def protocol_fold_ids(protocol_dir: Path) -> List[int]:
    if not protocol_dir.is_dir():
        raise FileNotFoundError(protocol_dir)
    folds = []
    for path in protocol_dir.iterdir():
        match = re.fullmatch(r"fold_(\d+)", path.name)
        if path.is_dir() and match:
            folds.append(int(match.group(1)))
    folds.sort()
    if len(folds) != 4:
        raise ValueError(f"Expected exactly four fold directories under {protocol_dir}, found {folds}")
    return folds


def parse_bool_column(values: pd.Series, context: str) -> pd.Series:
    normalized = values.astype(str).str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False}
    invalid = sorted(set(normalized.unique()).difference(mapping))
    if invalid:
        raise ValueError(f"{context} contains invalid boolean values: {invalid}")
    return normalized.map(mapping).astype(bool)


def load_protocol_manifests(protocol_dir: Path) -> Tuple[List[int], pd.DataFrame, pd.DataFrame]:
    folds = protocol_fold_ids(protocol_dir)
    frames = []
    expected_recording_ids = None
    for fold in folds:
        path = protocol_dir / f"fold_{fold}" / "recording_split_manifest.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, encoding="utf-8-sig")
        missing = MANIFEST_REQUIRED.difference(frame.columns)
        if missing:
            raise ValueError(f"Protocol manifest {path} missing columns: {sorted(missing)}")
        frame["recording_id"] = integer_column(frame, "recording_id", f"Protocol manifest {path}")
        if frame["recording_id"].duplicated().any():
            duplicates = frame.loc[frame["recording_id"].duplicated(False), "recording_id"].tolist()
            raise ValueError(f"Protocol manifest {path} has duplicate recording_id values: {duplicates[:5]}")
        for key in RECORDING_KEYS:
            frame[key] = frame[key].astype(str)
        frame["split"] = frame["split"].astype(str).str.strip().str.lower()
        frame["test_fold"] = integer_column(frame, "test_fold", f"Protocol manifest {path}")
        expected_test = frame["test_fold"].eq(fold)
        if not expected_test.equals(frame["split"].eq("test")):
            raise ValueError(f"Protocol manifest {path} disagrees between test_fold and split")
        frame["outer_test"] = parse_bool_column(
            frame["outer_test"], f"Protocol manifest {path} outer_test"
        )
        if not frame["outer_test"].equals(frame["split"].eq("test")):
            raise ValueError(f"Protocol manifest {path} disagrees between outer_test and split")

        recording_ids = set(frame["recording_id"].tolist())
        if expected_recording_ids is None:
            expected_recording_ids = recording_ids
        elif recording_ids != expected_recording_ids:
            raise ValueError(f"Protocol manifest {path} has a different recording catalog")
        frame.insert(0, "fold", fold)
        frames.append(frame)

    manifests = pd.concat(frames, ignore_index=True)
    stable_columns = [*RECORDING_KEYS, "test_fold"]
    unstable = manifests.groupby("recording_id")[stable_columns].nunique(dropna=False).gt(1)
    if unstable.any(axis=None):
        recording_ids = unstable.index[unstable.any(axis=1)].tolist()
        raise ValueError(
            "Recording identity/test_fold metadata are inconsistent across protocol folds: "
            f"{recording_ids[:5]}"
        )
    test_counts = manifests["split"].eq("test").groupby(manifests["recording_id"]).sum()
    if not test_counts.eq(1).all():
        recording_ids = test_counts.index[~test_counts.eq(1)].tolist()
        raise ValueError(
            f"Each protocol recording must be test in exactly one fold: {recording_ids[:5]}"
        )
    catalog_columns = ["recording_id", *RECORDING_KEYS, "test_fold"]
    catalog_rows = manifests[catalog_columns].drop_duplicates()
    if catalog_rows["recording_id"].duplicated(keep=False).any():
        raise ValueError("recording_id metadata are inconsistent across protocol folds")
    if catalog_rows[RECORDING_KEYS].duplicated(keep=False).any():
        raise ValueError("A recording identity maps to multiple protocol recording_id values")
    catalog = catalog_rows.sort_values("recording_id", kind="mergesort").reset_index(drop=True)
    return folds, manifests, catalog


def scope_catalog(catalog: pd.DataFrame, scope: str) -> pd.DataFrame:
    if scope == "static":
        selected = catalog[catalog["Scenario"].str.startswith("st_")].copy()
    elif scope == "dynamic":
        selected = catalog[catalog["Scenario"].str.startswith("dy_")].copy()
    elif scope == "all":
        selected = catalog.copy()
    else:
        raise ValueError(scope)
    selected = selected.sort_values(RECORDING_KEYS, kind="mergesort").reset_index(drop=True)
    selected = selected.rename(columns={"recording_id": "protocol_recording_id"})
    selected.insert(0, "recording_id", np.arange(len(selected), dtype=np.int64))
    return selected


def infer_recording_mapping(
    predictions: pd.DataFrame,
    manifests: pd.DataFrame,
    catalog: pd.DataFrame,
) -> Tuple[str, pd.DataFrame]:
    predicted_pairs = predictions[["fold", "recording_id"]].drop_duplicates()
    test_pairs = manifests.loc[
        manifests["split"].eq("test"), ["fold", "recording_id"]
    ].rename(columns={"recording_id": "protocol_recording_id"})

    valid = []
    for scope in ("all", "static", "dynamic"):
        local_catalog = scope_catalog(catalog, scope)
        expected_pairs = local_catalog[["recording_id", "test_fold"]].rename(
            columns={"test_fold": "fold"}
        )
        expected_pair_set = set(
            zip(expected_pairs["fold"].astype(int), expected_pairs["recording_id"].astype(int))
        )
        predicted_pair_set = set(
            zip(predicted_pairs["fold"].astype(int), predicted_pairs["recording_id"].astype(int))
        )
        if predicted_pair_set != expected_pair_set:
            continue
        mapped = predicted_pairs.merge(local_catalog, on="recording_id", how="left", validate="many_to_one")
        if mapped["protocol_recording_id"].isna().any():
            continue
        mapped["protocol_recording_id"] = mapped["protocol_recording_id"].astype(np.int64)
        checked = mapped.merge(
            test_pairs.assign(_is_test=True),
            on=["fold", "protocol_recording_id"],
            how="left",
            validate="one_to_one",
        )
        if checked["_is_test"].fillna(False).all():
            checked = checked.drop(columns="_is_test").sort_values(
                ["fold", "recording_id"], kind="mergesort"
            ).reset_index(drop=True)
            valid.append((scope, checked))

    if not valid:
        sample = predicted_pairs.sort_values(["fold", "recording_id"]).head(10).to_dict("records")
        raise ValueError(
            "No all/static/dynamic recording-id mapping sends every prediction pair "
            f"to its protocol test fold; sample prediction pairs={sample}"
        )

    signatures: Dict[Tuple[Tuple[int, int, int], ...], List[Tuple[str, pd.DataFrame]]] = {}
    for scope, mapped in valid:
        signature = tuple(
            (int(row.fold), int(row.recording_id), int(row.protocol_recording_id))
            for row in mapped.itertuples(index=False)
        )
        signatures.setdefault(signature, []).append((scope, mapped))
    if len(signatures) != 1:
        alternatives = [scope for scope, _ in valid]
        raise ValueError(f"Ambiguous recording-id mapping; valid scopes={alternatives}")
    equivalent = next(iter(signatures.values()))
    scope_names = [scope for scope, _ in equivalent]
    preferred = next((item for item in equivalent if item[0] != "all"), equivalent[0])
    mapping_mode = preferred[0] if len(scope_names) == 1 else "+".join(scope_names)
    return mapping_mode, preferred[1]


def attach_recording_metadata(
    predictions: pd.DataFrame, mapping: pd.DataFrame
) -> pd.DataFrame:
    enriched = predictions.merge(
        mapping,
        on=["fold", "recording_id"],
        how="left",
        validate="many_to_one",
    )
    if enriched["protocol_recording_id"].isna().any():
        raise AssertionError("Validated recording mapping did not cover every prediction row")
    enriched["protocol_recording_id"] = enriched["protocol_recording_id"].astype(np.int64)
    return enriched


def metric_bundle(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, Any]:
    true = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(y_pred, dtype=np.int64)
    if len(true) == 0:
        raise ValueError("Cannot summarize an empty prediction group")
    matrix = confusion_matrix(true, pred, labels=LABELS)
    precision, recall, f1, support = precision_recall_fscore_support(
        true, pred, labels=LABELS, zero_division=0
    )
    per_class = []
    for class_id in LABELS:
        per_class.append(
            {
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "precision": float(precision[class_id]),
                "recall": float(recall[class_id]),
                "f1": float(f1[class_id]),
                "support": int(support[class_id]),
                "predicted_support": int(matrix[:, class_id].sum()),
            }
        )
    return {
        "windows": int(len(true)),
        "accuracy": float(np.mean(true == pred)),
        "macro_f1": float(np.mean(f1)),
        "confusion_matrix": matrix.tolist(),
        "confusion_labels": [CLASS_NAMES[class_id] for class_id in LABELS],
        "per_class": per_class,
    }


def grouped_metric_rows(frame: pd.DataFrame, group_columns: List[str]) -> List[Dict[str, Any]]:
    rows = []
    for key, group in frame.groupby(group_columns, sort=True):
        keys = key if isinstance(key, tuple) else (key,)
        row = {column: value.item() if hasattr(value, "item") else value
               for column, value in zip(group_columns, keys)}
        metrics = metric_bundle(group["true_class"], group["pred_class"])
        row.update({name: metrics[name] for name in (
            "windows", "accuracy", "macro_f1", "confusion_matrix", "confusion_labels"
        )})
        for class_metrics in metrics["per_class"]:
            slug = CLASS_SLUGS[class_metrics["class_id"]]
            row[f"recall_{slug}"] = class_metrics["recall"]
            row[f"support_{slug}"] = class_metrics["support"]
        rows.append(row)
    return rows


def csv_ready(records: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    converted = []
    for source in records:
        row = dict(source)
        matrix = row.get("confusion_matrix")
        if matrix is not None:
            row["confusion_matrix"] = json.dumps(matrix, separators=(",", ":"))
            for true_index, true_id in enumerate(LABELS):
                for pred_index, pred_id in enumerate(LABELS):
                    row[
                        f"confusion_true_{CLASS_SLUGS[true_id]}_pred_{CLASS_SLUGS[pred_id]}"
                    ] = int(matrix[true_index][pred_index])
        labels = row.get("confusion_labels")
        if labels is not None:
            row["confusion_labels"] = json.dumps(labels, ensure_ascii=False, separators=(",", ":"))
        converted.append(row)
    return pd.DataFrame(converted)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_outputs(
    output_dir: Path,
    overall: Dict[str, Any],
    per_class: List[Dict[str, Any]],
    per_device: List[Dict[str, Any]],
    per_recording: List[Dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payloads = {
        "overall_metrics": (overall, [overall]),
        "per_class_metrics": (per_class, per_class),
        "per_device_metrics": (per_device, per_device),
        "per_recording_metrics": (per_recording, per_recording),
    }
    for stem, (json_payload, csv_records) in payloads.items():
        write_json(output_dir / f"{stem}.json", json_payload)
        csv_ready(csv_records).to_csv(
            output_dir / f"{stem}.csv", index=False, encoding="utf-8-sig"
        )

    matrix = np.asarray(overall["confusion_matrix"], dtype=np.int64)
    confusion_frame = pd.DataFrame(
        matrix,
        columns=[f"pred_{CLASS_SLUGS[class_id]}" for class_id in LABELS],
    )
    confusion_frame.insert(0, "true_class", [CLASS_NAMES[class_id] for class_id in LABELS])
    confusion_frame.to_csv(
        output_dir / "overall_confusion_matrix.csv", index=False, encoding="utf-8-sig"
    )


def summarize(predictions_path: Path, protocol_dir: Path, output_dir: Path) -> Dict[str, Any]:
    predictions = load_predictions(predictions_path)
    protocol_folds, manifests, catalog = load_protocol_manifests(protocol_dir)
    prediction_folds = sorted(int(value) for value in predictions["fold"].unique())
    if prediction_folds != protocol_folds:
        raise ValueError(
            f"Prediction folds {prediction_folds} do not match protocol folds {protocol_folds}"
        )
    mapping_mode, mapping = infer_recording_mapping(predictions, manifests, catalog)
    enriched = attach_recording_metadata(predictions, mapping)

    metrics = metric_bundle(enriched["true_class"], enriched["pred_class"])
    overall = {name: metrics[name] for name in (
        "windows", "accuracy", "macro_f1", "confusion_matrix", "confusion_labels"
    )}
    overall.update(
        {
            "predictions": str(predictions_path.resolve()),
            "protocol_dir": str(protocol_dir.resolve()),
            "protocol_folds": protocol_folds,
            "recording_id_mapping": mapping_mode,
            "prediction_key": predictions.attrs["prediction_key"],
            "prediction_key_strict": predictions.attrs["prediction_key_strict"],
            "legacy_key_collision_rows": predictions.attrs["legacy_key_collision_rows"],
            "prediction_recording_pairs": int(len(mapping)),
        }
    )
    per_class = metrics["per_class"]
    per_device = grouped_metric_rows(enriched, ["device_id", "device_name"])
    per_recording = grouped_metric_rows(
        enriched,
        [
            "fold",
            "recording_id",
            "protocol_recording_id",
            "Environment",
            "Scenario",
            "Session",
        ],
    )
    write_outputs(output_dir, overall, per_class, per_device, per_recording)

    LOG.info(
        "rows=%d folds=%s recording_pairs=%d id_mapping=%s accuracy=%.6f macro_f1=%.6f",
        len(enriched),
        protocol_folds,
        len(mapping),
        mapping_mode,
        overall["accuracy"],
        overall["macro_f1"],
    )
    pixel6 = [row for row in per_device if row["device_name"] == "Google_Pixel6"]
    if pixel6:
        LOG.info(
            "Google_Pixel6 L5 recall=%.6f support=%d",
            pixel6[0]["recall_L5"],
            pixel6[0]["support_L5"],
        )
    return overall


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summarize(args.predictions, args.protocol_dir, args.output_dir)


if __name__ == "__main__":
    main()

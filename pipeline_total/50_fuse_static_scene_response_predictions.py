#!/usr/bin/env python3
"""Fuse static band-scene and device-response predictions into joint diagnoses.

This is a *late-fusion* evaluator.  It deliberately does not train a new model
or relabel the two tasks as one flat class space:

* the band-mean branch estimates the physical scene at a receiver epoch
  (normal / L1 / L5 / L1+L5) from dual-band devices;
* the response-state branch estimates each device's observable state
  (normal / anomaly / direct) from its own receiver response.

For a common outer fold and endpoint TOW, dual-band probabilities are averaged
across available devices to form one scene consensus.  That consensus is then
attached to every response-state row at the same epoch, including a single-band
Watch.  The final record can therefore say, for example, ``anomaly under L5``
or ``direct spoof under L1+L5`` without falsely treating the two labels as the
same target.

Both upstream prediction files must come from outer-test inference only.  The
script writes the complete fused table plus state, scene, and joint-diagnosis
metrics.  It never uses validation or training predictions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support


ROOT = Path(__file__).resolve().parents[1]
SCENE_NAMES = {0: "normal", 1: "L1", 2: "L5", 3: "L1+L5"}
STATE_NAMES = {0: "normal", 1: "anomaly", 2: "direct"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--band-training-root", type=Path,
        default=ROOT / "output" / "training" / "static_scene_response_fusion_v1" / "band_scene",
        help="Root containing fold_<n>/test_predictions_band_mean_window_tcn.csv.",
    )
    parser.add_argument(
        "--response-training-root", type=Path,
        default=ROOT / "output" / "hierarchical_event_v1" / "static_scene_response_fusion_v1" / "response_state",
        help="Root containing fold_<n>/direct_override_mlp_h32_valcal_all/test_response_state_predictions.csv.",
    )
    parser.add_argument(
        "--response-output-name", type=str,
        default="direct_override_mlp_h32_valcal_all",
        help="Per-fold response override directory name.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "output" / "hierarchical_event_v1" / "static_scene_response_fusion_v1" / "fusion",
    )
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(1, 8)))
    parser.add_argument(
        "--tow-decimals", type=int, default=3,
        help="Decimal places used to align independently built endpoint TOW values.",
    )
    args = parser.parse_args()
    if args.tow_decimals < 0 or args.tow_decimals > 9:
        parser.error("--tow-decimals must be in [0, 9]")
    return args


def metric_bundle(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int], names: dict[int, str]) -> dict[str, Any]:
    if len(y_true) == 0:
        return {"samples": 0, "macro_f1": None, "accuracy": None, "per_class": {}}
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0,
    )
    return {
        "samples": int(len(y_true)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "accuracy": float((y_true == y_pred).mean()),
        "confusion_matrix_labels": [names[label] for label in labels],
        "confusion_matrix": matrix.tolist(),
        "per_class": {
            names[label]: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(labels)
        },
    }


def response_state_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    y_true = frame["true_state"].to_numpy(dtype=np.int64)
    y_pred = frame["pred_state"].to_numpy(dtype=np.int64)
    result = metric_bundle(y_true, y_pred, [0, 1, 2], STATE_NAMES)
    normal = y_true == 0
    abnormal = y_true > 0
    result["far"] = float(((y_pred[normal] > 0).mean())) if normal.any() else None
    result["abnormal_recall"] = float(((y_pred[abnormal] > 0).mean())) if abnormal.any() else None
    return result


def require_columns(frame: pd.DataFrame, columns: set[str], path: Path) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns {missing}; rerun the updated exporter")


def read_band_predictions(path: Path, fold: int, decimals: int) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    probability_columns = [f"prob_{SCENE_NAMES[index]}" for index in range(4)]
    require_columns(frame, {"endpoint_tow", "true_class", "pred_class", *probability_columns}, path)
    if "fold" not in frame or frame["fold"].isna().all():
        frame["fold"] = fold
    frame["fold"] = frame["fold"].astype(int)
    if not (frame["fold"] == fold).all():
        raise ValueError(f"{path} contains rows not tagged fold {fold}")
    frame["tow_key"] = frame["endpoint_tow"].round(decimals)
    for column in probability_columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    return frame


def scene_consensus(frame: pd.DataFrame) -> pd.DataFrame:
    probability_columns = [f"prob_{SCENE_NAMES[index]}" for index in range(4)]
    grouped = frame.groupby(["fold", "tow_key"], sort=True)
    rows: list[dict[str, Any]] = []
    for (fold, tow_key), group in grouped:
        true_values = sorted(set(group["true_class"].astype(int)))
        if len(true_values) != 1:
            raise ValueError(
                f"Scene labels disagree at fold={fold}, TOW={tow_key}: {true_values}. "
                "Band endpoints at the same receiver epoch must share one reviewed scene label."
            )
        probabilities = group[probability_columns].to_numpy(dtype=float).mean(axis=0)
        predicted = int(probabilities.argmax())
        row = {
            "fold": int(fold),
            "tow_key": float(tow_key),
            "scene_true": int(true_values[0]),
            "scene_pred": predicted,
            "scene_confidence": float(probabilities[predicted]),
            "scene_device_votes": int(len(group)),
        }
        row.update({f"scene_prob_{SCENE_NAMES[index]}": float(probabilities[index]) for index in range(4)})
        rows.append(row)
    return pd.DataFrame(rows)


def read_response_predictions(path: Path, fold: int, decimals: int) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    require_columns(
        frame,
        {"endpoint_tow", "true_state", "pred_state", "device_name", "recording_name"},
        path,
    )
    if "fold" not in frame or frame["fold"].isna().all():
        frame["fold"] = fold
    frame["fold"] = frame["fold"].astype(int)
    if not (frame["fold"] == fold).all():
        raise ValueError(f"{path} contains rows not tagged fold {fold}")
    frame["tow_key"] = frame["endpoint_tow"].round(decimals)
    frame["true_state"] = frame["true_state"].astype(int)
    frame["pred_state"] = frame["pred_state"].astype(int)
    return frame


def joint_rows(response: pd.DataFrame, scenes: pd.DataFrame) -> pd.DataFrame:
    merged = response.merge(scenes, how="left", on=["fold", "tow_key"], validate="many_to_one")
    merged["scene_available"] = merged["scene_true"].notna()
    for column in ("scene_true", "scene_pred"):
        merged[column] = merged[column].astype("Int64")
    merged["true_state_name"] = merged["true_state"].map(STATE_NAMES)
    merged["pred_state_name"] = merged["pred_state"].map(STATE_NAMES)
    merged["scene_true_name"] = merged["scene_true"].map(SCENE_NAMES).fillna("unknown")
    merged["scene_pred_name"] = merged["scene_pred"].map(SCENE_NAMES).fillna("unknown")
    merged["joint_true"] = "state_" + merged["true_state_name"] + "__scene_" + merged["scene_true_name"]
    merged["joint_pred"] = "state_" + merged["pred_state_name"] + "__scene_" + merged["scene_pred_name"]
    return merged


def joint_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    context = frame[frame["scene_available"]].copy()
    result: dict[str, Any] = {
        "response_windows": int(len(frame)),
        "scene_context_windows": int(len(context)),
        "scene_context_coverage": float(len(context) / len(frame)) if len(frame) else 0.0,
    }
    if context.empty:
        result["abnormal_joint"] = {"samples": 0, "joint_recall": None}
        result["by_true_state"] = {}
        return result
    abnormal = context[context["true_state"] > 0].copy()
    correct_state = abnormal["pred_state"].to_numpy(dtype=int) == abnormal["true_state"].to_numpy(dtype=int)
    correct_scene = abnormal["scene_pred"].to_numpy(dtype=int) == abnormal["scene_true"].to_numpy(dtype=int)
    result["abnormal_joint"] = {
        "samples": int(len(abnormal)),
        "state_correct": int(correct_state.sum()),
        "scene_correct": int(correct_scene.sum()),
        "joint_correct": int((correct_state & correct_scene).sum()),
        "joint_recall": float((correct_state & correct_scene).mean()) if len(abnormal) else None,
    }
    result["by_true_state"] = {}
    for state in (1, 2):
        subset = abnormal[abnormal["true_state"] == state]
        if subset.empty:
            result["by_true_state"][STATE_NAMES[state]] = {"samples": 0, "joint_recall": None}
            continue
        state_ok = subset["pred_state"].to_numpy(dtype=int) == state
        scene_ok = subset["scene_pred"].to_numpy(dtype=int) == subset["scene_true"].to_numpy(dtype=int)
        result["by_true_state"][STATE_NAMES[state]] = {
            "samples": int(len(subset)),
            "joint_correct": int((state_ok & scene_ok).sum()),
            "joint_recall": float((state_ok & scene_ok).mean()),
        }
    return result


def device_coverage(frame: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, group in frame.groupby("device_name", dropna=False, sort=True):
        result[str(name)] = {
            "response_windows": int(len(group)),
            "scene_context_coverage": float(group["scene_available"].mean()),
            "abnormal_support": int((group["true_state"] > 0).sum()),
        }
    return result


def main() -> None:
    args = parse_args()
    all_band: list[pd.DataFrame] = []
    all_response: list[pd.DataFrame] = []
    for fold in args.folds:
        band_path = args.band_training_root / f"fold_{fold}" / "test_predictions_band_mean_window_tcn.csv"
        response_path = (
            args.response_training_root / f"fold_{fold}" / args.response_output_name /
            "test_response_state_predictions.csv"
        )
        if not band_path.is_file():
            raise FileNotFoundError(f"Missing band-scene test predictions: {band_path}")
        if not response_path.is_file():
            raise FileNotFoundError(f"Missing response-state test predictions: {response_path}")
        all_band.append(read_band_predictions(band_path, fold, args.tow_decimals))
        all_response.append(read_response_predictions(response_path, fold, args.tow_decimals))

    band = pd.concat(all_band, ignore_index=True)
    response = pd.concat(all_response, ignore_index=True)
    scenes = scene_consensus(band)
    fused = joint_rows(response, scenes)

    scene_metric_rows = scenes.dropna(subset=["scene_true", "scene_pred"])
    result = {
        "protocol": "static_time_block_outer_v2",
        "fusion_type": "late_fusion_scene_context_plus_device_response",
        "folds": [int(fold) for fold in args.folds],
        "tow_decimals": args.tow_decimals,
        "semantics": {
            "scene": "0=normal, 1=L1, 2=L5, 3=L1+L5; consensus is the mean dual-band posterior at one endpoint TOW",
            "response_state": "0=normal/no observable response, 1=attack-associated anomaly, 2=direct spoof",
            "joint": "state and scene remain separate axes; joint correctness requires both axes correct",
        },
        "response_state": response_state_metrics(fused),
        "scene_consensus": metric_bundle(
            scene_metric_rows["scene_true"].to_numpy(dtype=np.int64),
            scene_metric_rows["scene_pred"].to_numpy(dtype=np.int64),
            [0, 1, 2, 3], SCENE_NAMES,
        ),
        "joint_diagnosis": joint_metrics(fused),
        "by_device_scene_context": device_coverage(fused),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fused_path = args.output_dir / "fused_static_scene_response_test_predictions.csv"
    metrics_path = args.output_dir / "static_scene_response_fusion_metrics.json"
    fused.to_csv(fused_path, index=False, encoding="utf-8-sig")
    metrics_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "fused_predictions": str(fused_path),
        "metrics": str(metrics_path),
        "response_windows": result["joint_diagnosis"]["response_windows"],
        "scene_context_coverage": result["joint_diagnosis"]["scene_context_coverage"],
        "response_abnormal_recall": result["response_state"]["abnormal_recall"],
        "scene_macro_f1": result["scene_consensus"]["macro_f1"],
        "abnormal_joint_recall": result["joint_diagnosis"]["abnormal_joint"]["joint_recall"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

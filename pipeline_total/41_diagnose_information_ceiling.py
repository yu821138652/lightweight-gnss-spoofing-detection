#!/usr/bin/env python3
"""Model-agnostic diagnostics for the mixed static+dynamic signal task.

This script does NOT train the deployable detector and does NOT read the
outer test tensors.  It runs two independent, cheap diagnostics on the
existing mixed 4-fold development tensors so we can distinguish two very
different explanations for the current `dy_L5` / device failures:

  D1  Information ceiling
      Fit a deliberately strong tabular model (HistGradientBoosting) on the
      exact per-endpoint features the deployable model already sees (the
      train-only per-device standardized W5 stats).  If even this strong
      model cannot separate a subgroup (e.g. `dy_L5`), the bottleneck is the
      information in these features, not the TCN16 capacity.  If it clearly
      beats the TCN16 baseline, the bottleneck is architecture/optimization.

  D2  Shortcut probe
      Fit the same strong model on metadata ONLY -- device id, scenario id,
      frequency band -- with no signal features at all.  Its validation
      Macro-F1 is a lower bound on how much of the task is solvable from
      dataset shortcuts.  This quantifies why state-stratified validation can
      move while held-out test does not.

Both diagnostics are evaluated ONLY on the inner validation split of each
development fold.  The outer test tensors are never opened here.

All feature semantics (train-only per-device standardization, endpoint
flattening over the 128 signal slots, IGNORE_INDEX handling) are copied from
`21_train_static_signal_fusion.py` so results are directly comparable to the
deployable baseline.

Usage:

    python pipeline_total/41_diagnose_information_ceiling.py \
      --tensor-root output/tensors/mixed_timeblock_outer_cv4_w5_v2 \
      --output-dir output/diagnostics/mixed_v2_information_ceiling_v1 \
      --folds 1 2 3 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import f1_score, precision_recall_fscore_support
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "scikit-learn is required for this diagnostic. "
        "Install it into the project environment first."
    ) from exc


IGNORE_INDEX = -100

# The stats branch the deployable compact11 model actually consumes.  We keep
# the full 19-d stats tensor here because the point of the information-ceiling
# probe is to give the strong model the most information available, not to
# reproduce the compact11 ablation.  IsL5 is an online-known routing attribute
# and is kept.
STATS_FEATURE_NAMES_FILE = "feature_names.json"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_endpoints(
    fold_dir: Path, split: str
) -> Dict[str, np.ndarray]:
    """Flatten [B, S, ...] stats tensors to active per-endpoint rows.

    Returns a dict of parallel arrays, one row per active (window, slot)
    endpoint that carries a real label (mask True and y != IGNORE_INDEX).
    """
    stats_path = fold_dir / "stats" / f"{split}.npz"
    raw_path = fold_dir / "raw" / f"{split}.npz"
    with np.load(stats_path) as stats:
        x = np.asarray(stats["x"], dtype=np.float32)  # [B, S, 1, F]
        mask = np.asarray(stats["mask"]).astype(bool)  # [B, S]
        y = np.asarray(stats["y"]).astype(np.int64)  # [B, S]
        device_id = np.asarray(stats["device_id"]).astype(np.int64)  # [B]
        is_dynamic = np.asarray(stats["is_dynamic"]).astype(bool)  # [B]
    # raw tensor carries recording/source trace for subgroup attribution
    with np.load(raw_path) as raw:
        recording_id = np.asarray(raw["recording_id"]).astype(np.int64)  # [B]
        signal_id = np.asarray(raw["signal_id"]).astype(np.int64)  # [B, S]

    n_windows, n_slots, _, n_feat = x.shape
    x2 = x.reshape(n_windows, n_slots, n_feat)  # drop the time dim of length 1

    active = mask & (y != IGNORE_INDEX)  # [B, S]
    w_idx, s_idx = np.nonzero(active)

    feats = x2[w_idx, s_idx, :]  # [N, F]
    labels = y[w_idx, s_idx]  # [N]
    dev = device_id[w_idx]  # [N] (device is per-window)
    dyn = is_dynamic[w_idx]  # [N]
    rec = recording_id[w_idx]  # [N]
    sig = signal_id[w_idx, s_idx]  # [N]

    return {
        "features": feats,
        "labels": labels,
        "device_id": dev,
        "is_dynamic": dyn,
        "recording_id": rec,
        "signal_id": sig,
    }


def load_stats_feature_names(fold_dir: Path) -> List[str]:
    return _read_json(fold_dir / "stats" / STATS_FEATURE_NAMES_FILE)


def load_recording_scenarios(fold_dir: Path) -> Dict[int, Tuple[str, str, str]]:
    """Map recording_id -> (Environment, Scenario, Session)."""
    trace = _read_json(fold_dir / "window_trace_index.json")
    recs = trace["recordings"]
    out: Dict[int, Tuple[str, str, str]] = {}
    for i, r in enumerate(recs):
        out[i] = (r["Environment"], r["Scenario"], r["Session"])
    return out


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def subgroup_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """Binary metrics with positive == spoofed (label 1)."""
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], average=None, zero_division=0
    )
    # FAR = FP / (FP + TN) = 1 - specificity = fraction of true negatives
    # predicted positive.
    neg = y_true == 0
    far = float((y_pred[neg] == 1).mean()) if neg.any() else 0.0
    return {
        "support": int(y_true.size),
        "positives": int((y_true == 1).sum()),
        "negatives": int((y_true == 0).sum()),
        "macro_f1": macro_f1(y_true, y_pred),
        "precision_pos": float(p[1]),
        "recall_pos": float(r[1]),
        "far": far,
    }


def fit_strong_model(
    x_train: np.ndarray, y_train: np.ndarray, seed: int
) -> HistGradientBoostingClassifier:
    # Class-balanced sample weights so the strong model is not trivially
    # dominated by the majority (normal) class, matching the class-balanced
    # CrossEntropy used by the deployable baseline.
    n_pos = max(int((y_train == 1).sum()), 1)
    n_neg = max(int((y_train == 0).sum()), 1)
    w_pos = 0.5 / n_pos * (n_pos + n_neg)
    w_neg = 0.5 / n_neg * (n_pos + n_neg)
    sample_weight = np.where(y_train == 1, w_pos, w_neg).astype(np.float64)
    clf = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.1,
        max_depth=None,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=seed,
    )
    clf.fit(x_train, y_train, sample_weight=sample_weight)
    return clf


def run_fold(
    fold_dir: Path, seed: int
) -> Dict[str, object]:
    train = load_endpoints(fold_dir, "train")
    val = load_endpoints(fold_dir, "val")
    stats_names = load_stats_feature_names(fold_dir)
    rec_scen = load_recording_scenarios(fold_dir)

    result: Dict[str, object] = {"stats_features": stats_names}

    # ----- D1: information ceiling on the deployable stats features -----
    d1_model = fit_strong_model(train["features"], train["labels"], seed)
    val_pred_d1 = d1_model.predict(val["features"])

    d1: Dict[str, object] = {}
    d1["overall_val"] = subgroup_metrics(val["labels"], val_pred_d1)

    # dynamic vs static
    for name, sel in (
        ("static_val", ~val["is_dynamic"]),
        ("dynamic_val", val["is_dynamic"]),
    ):
        if sel.any():
            d1[name] = subgroup_metrics(val["labels"][sel], val_pred_d1[sel])

    # per-scenario (esp. dy_L5) on the validation split
    scen_of = np.array(
        [rec_scen[int(r)][1] for r in val["recording_id"]], dtype=object
    )
    scen_metrics: Dict[str, Dict[str, float]] = {}
    for scen in sorted(set(scen_of.tolist())):
        sel = scen_of == scen
        if sel.any():
            scen_metrics[scen] = subgroup_metrics(
                val["labels"][sel], val_pred_d1[sel]
            )
    d1["by_scenario_val"] = scen_metrics

    # per-device on the validation split
    dev_metrics: Dict[str, Dict[str, float]] = {}
    for dev in sorted(set(val["device_id"].tolist())):
        sel = val["device_id"] == dev
        if sel.any():
            dev_metrics[str(dev)] = subgroup_metrics(
                val["labels"][sel], val_pred_d1[sel]
            )
    d1["by_device_val"] = dev_metrics
    result["D1_information_ceiling"] = d1

    # ----- D2: shortcut probe on metadata only -----
    # Features: device_id (one-hot), scenario_id (one-hot), band (IsL5).
    is_l5_idx = stats_names.index("IsL5")

    def meta_features(bundle: Dict[str, np.ndarray]) -> np.ndarray:
        dev = bundle["device_id"].reshape(-1, 1).astype(np.float64)
        band = bundle["features"][:, is_l5_idx].reshape(-1, 1).astype(np.float64)
        scen = np.array(
            [rec_scen[int(r)][1] for r in bundle["recording_id"]], dtype=object
        )
        return dev, band, scen

    dev_tr, band_tr, scen_tr = meta_features(train)
    dev_va, band_va, scen_va = meta_features(val)

    scenarios = sorted(set(scen_tr.tolist()) | set(scen_va.tolist()))
    scen_index = {s: i for i, s in enumerate(scenarios)}
    devices = sorted(
        set(train["device_id"].tolist()) | set(val["device_id"].tolist())
    )
    dev_index = {d: i for i, d in enumerate(devices)}

    def onehot(dev, band, scen) -> np.ndarray:
        dev_oh = np.zeros((dev.shape[0], len(dev_index)), dtype=np.float64)
        for i, d in enumerate(dev[:, 0].astype(np.int64)):
            dev_oh[i, dev_index[int(d)]] = 1.0
        scen_oh = np.zeros((dev.shape[0], len(scen_index)), dtype=np.float64)
        for i, s in enumerate(scen):
            scen_oh[i, scen_index[s]] = 1.0
        return np.concatenate([dev_oh, scen_oh, band], axis=1)

    meta_tr = onehot(dev_tr, band_tr, scen_tr)
    meta_va = onehot(dev_va, band_va, scen_va)

    d2_model = fit_strong_model(meta_tr, train["labels"], seed)
    val_pred_d2 = d2_model.predict(meta_va)

    d2: Dict[str, object] = {}
    d2["feature_layout"] = {
        "device_onehot": len(dev_index),
        "scenario_onehot": len(scen_index),
        "band_is_l5": 1,
    }
    d2["overall_val"] = subgroup_metrics(val["labels"], val_pred_d2)
    for name, sel in (
        ("static_val", ~val["is_dynamic"]),
        ("dynamic_val", val["is_dynamic"]),
    ):
        if sel.any():
            d2[name] = subgroup_metrics(val["labels"][sel], val_pred_d2[sel])
    result["D2_shortcut_probe"] = d2

    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tensor-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_results: Dict[str, object] = {
        "tensor_root": str(args.tensor_root),
        "seed": args.seed,
        "note": (
            "Model-agnostic diagnostics on inner validation only. "
            "Outer test tensors are never read here."
        ),
        "folds": {},
    }

    for fold in args.folds:
        fold_dir = args.tensor_root / f"fold_{fold}"
        if not fold_dir.exists():
            raise SystemExit(f"Missing fold dir: {fold_dir}")
        print(f"[fold {fold}] loading and fitting strong models ...", flush=True)
        res = run_fold(fold_dir, args.seed)
        all_results["folds"][str(fold)] = res

        d1o = res["D1_information_ceiling"]["overall_val"]
        d2o = res["D2_shortcut_probe"]["overall_val"]
        dyl5 = (
            res["D1_information_ceiling"]["by_scenario_val"].get("dy_L5")
        )
        print(
            f"[fold {fold}] D1 val macroF1={d1o['macro_f1']:.4f}  "
            f"D2(shortcut) val macroF1={d2o['macro_f1']:.4f}"
            + (
                f"  D1 dy_L5 recall={dyl5['recall_pos']:.4f} "
                f"far={dyl5['far']:.4f}"
                if dyl5
                else "  (no dy_L5 in this val)"
            ),
            flush=True,
        )

    out_path = args.output_dir / "diagnostics_information_ceiling.json"
    out_path.write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

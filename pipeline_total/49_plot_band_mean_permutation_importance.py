#!/usr/bin/env python3
"""Cross-fold permutation importance for the band-mean four-way scene classifier.

For every feature name (AgcDb, Cn0DbHz, ...) this shuffles that feature's values
across the usable test windows of each fold, re-scores the four-way task, and
reports the drop in aggregate macro-F1.  A positive drop means the model relied
on the feature; a *negative* drop (macro-F1 rises when the feature is destroyed)
flags a feature that hurts cross-recording generalization -- exactly the AGC
story this line of work uncovered.

The permutation runs on the full-feature baseline checkpoints
(``band_mean_multiclass_cv``); running it there is what makes the AGC harm
visible as a number rather than an ablation result.  Features are grouped by
band-agnostic name so L1_AgcDb and L5_AgcDb are shuffled together (independently
per band, preserving each band's marginal), matching how ``--drop-features``
removes a feature across both bands.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import BandMeanWindowClassifier

NUM_CLASSES = 4
CLASS_NAMES = {0: "normal", 1: "L1", 2: "L5", 3: "L1+L5"}
PROTECTED_FEATURES = ("L1Present", "L5Present")

plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
LOG = logging.getLogger(__name__)


def load_fold_scenarios(assignment_path: Path) -> dict[int, str]:
    frame = pd.read_csv(assignment_path, encoding="utf-8-sig")
    test_rows = frame[frame["role"].astype(str) == "test"]
    return {int(row.fold): str(row.Scenario) for row in test_rows.itertuples(index=False)}


def feature_groups(all_names: list[str]) -> dict[str, list[int]]:
    """Group column indices by band-agnostic feature name (excluding presence)."""
    groups: dict[str, list[int]] = {}
    for index, name in enumerate(all_names):
        if name in PROTECTED_FEATURES:
            continue
        suffix = name.split("_", 1)[1] if "_" in name else name
        groups.setdefault(suffix, []).append(index)
    return groups


@torch.no_grad()
def load_fold_usable(tensor_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    with np.load(tensor_dir / "test.npz", allow_pickle=False) as data:
        x = np.asarray(data["x"], dtype=np.float32)
        y = np.asarray(data["y"], dtype=np.int64)
        single_band = np.asarray(data["single_band_mask"], dtype=bool)
    names = json.loads((tensor_dir / "feature_names.json").read_text(encoding="utf-8"))
    usable = ~single_band
    return x[usable], y[usable], names


@torch.no_grad()
def predict(model: BandMeanWindowClassifier, x: np.ndarray, device: torch.device, batch: int = 4096) -> np.ndarray:
    preds: list[np.ndarray] = []
    for start in range(0, len(x), batch):
        chunk = torch.from_numpy(x[start:start + batch]).to(device)
        preds.append(model(chunk).argmax(-1).cpu().numpy())
    return np.concatenate(preds) if preds else np.empty((0,), np.int64)


def build_model(checkpoint: dict, device: torch.device) -> BandMeanWindowClassifier:
    model = BandMeanWindowClassifier(
        input_dim=int(checkpoint["input_dim"]),
        time_steps=int(checkpoint["time_steps"]),
        encoder=str(checkpoint["encoder"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        dropout=float(checkpoint["dropout"]),
        num_classes=int(checkpoint.get("num_classes", NUM_CLASSES)),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol-dir", type=Path,
        default=ROOT / "output" / "protocols" / "static_time_block_outer_v2",
    )
    parser.add_argument(
        "--tensors-root", type=Path,
        default=ROOT / "output" / "tensors" / "band_mean_window_static_v1",
    )
    parser.add_argument(
        "--training-root", type=Path,
        default=ROOT / "output" / "training" / "band_mean_multiclass_cv",
    )
    parser.add_argument("--encoder", choices=("lstm", "gru", "tcn"), default="tcn")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--repeats", type=int, default=20,
        help="Independent shuffles per feature; the mean drop and its std are reported.",
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    output_dir = args.output_dir or args.training_root
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scenarios = load_fold_scenarios(args.protocol_dir / "fold_assignment.csv")
    folds = sorted(scenarios)
    rng = np.random.default_rng(args.seed)

    # Load every fold's usable test windows, model, and unpermuted predictions.
    fold_data: dict[int, dict] = {}
    all_names: list[str] | None = None
    for fold in folds:
        tensor_dir = args.tensors_root / f"fold_{fold}"
        checkpoint_path = args.training_root / f"fold_{fold}" / f"best_band_mean_window_{args.encoder}.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        x, y, names = load_fold_usable(tensor_dir)
        if all_names is None:
            all_names = names
        elif names != all_names:
            raise ValueError(f"fold {fold} feature names differ from fold {folds[0]}")
        model = build_model(checkpoint, device)
        fold_data[fold] = {"x": x, "y": y, "model": model}

    assert all_names is not None
    groups = feature_groups(all_names)

    def aggregate_macro_f1(permute: str | None, seed: int | None = None) -> float:
        local_rng = np.random.default_rng(seed) if seed is not None else rng
        y_true_all: list[np.ndarray] = []
        y_pred_all: list[np.ndarray] = []
        for fold in folds:
            data = fold_data[fold]
            x = data["x"].copy()
            if permute is not None:
                for column in groups[permute]:
                    # Shuffle this feature's endpoint values across windows,
                    # broadcasting the same permutation over the time axis so a
                    # window keeps its own temporal shape but gets another
                    # window's band level -- destroying only cross-window signal.
                    order = local_rng.permutation(len(x))
                    x[:, :, column] = x[order][:, :, column]
            y_true_all.append(data["y"])
            y_pred_all.append(predict(data["model"], x, device))
        y_true = np.concatenate(y_true_all)
        y_pred = np.concatenate(y_pred_all)
        return float(f1_score(y_true, y_pred, average="macro", labels=list(range(NUM_CLASSES)), zero_division=0))

    baseline = aggregate_macro_f1(None)
    LOG.info("baseline aggregate macro_f1=%.4f", baseline)

    # A single permutation is a high-variance estimate: for a mid-strength
    # feature one draw can flip the sign of the drop.  Repeat each feature with
    # independent shuffles and report mean +/- std so the ranking is stable.
    rows: list[dict] = []
    for name in sorted(groups):
        permuted_scores = np.array([
            aggregate_macro_f1(name, seed=args.seed + 1000 * (rep + 1) + hash(name) % 997)
            for rep in range(args.repeats)
        ])
        drops = baseline - permuted_scores
        rows.append({
            "feature": name,
            "macro_f1_permuted_mean": float(permuted_scores.mean()),
            "macro_f1_drop_mean": float(drops.mean()),
            "macro_f1_drop_std": float(drops.std(ddof=1)) if args.repeats > 1 else 0.0,
            "repeats": args.repeats,
        })
        LOG.info(
            "permute %-12s macro_f1=%.4f drop=%+.4f +/- %.4f (n=%d)",
            name, permuted_scores.mean(), drops.mean(),
            drops.std(ddof=1) if args.repeats > 1 else 0.0, args.repeats,
        )

    frame = pd.DataFrame(rows).sort_values("macro_f1_drop_mean", ascending=False).reset_index(drop=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "band_mean_permutation_importance.csv", index=False, encoding="utf-8-sig")
    (output_dir / "band_mean_permutation_baseline.json").write_text(
        json.dumps({"baseline_macro_f1": baseline, "repeats": args.repeats}, indent=2), encoding="utf-8"
    )

    colors = ["#b42318" if drop < 0 else "#1f5f99" for drop in frame["macro_f1_drop_mean"]]
    figure, axis = plt.subplots(figsize=(9, max(4, len(frame) * 0.7)))
    axis.barh(
        frame["feature"], frame["macro_f1_drop_mean"],
        xerr=frame["macro_f1_drop_std"], color=colors,
        error_kw={"ecolor": "0.3", "capsize": 4, "elinewidth": 1},
    )
    axis.invert_yaxis()
    axis.axvline(0, color="0.3", linewidth=0.8)
    axis.set_xlabel(f"Aggregate Macro-F1 drop after permuting the feature (mean +/- std, n={args.repeats})")
    axis.set_title(
        f"Band-mean feature permutation importance (baseline Macro-F1={baseline:.3f})\n"
        "blue = model relies on it; red = feature HURTS cross-recording generalization",
        fontsize=11,
    )
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "band_mean_permutation_importance.png", dpi=180, bbox_inches="tight", facecolor="white")
    LOG.info("wrote %s", output_dir / "band_mean_permutation_importance.png")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Measure test-set permutation importance for a locked signal fusion checkpoint.

This is a post-hoc diagnostic. Do not use a held-out test result from this
script to select model features and still describe that test as blind.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import SignalRawStatsFusion


def read_names(path: Path) -> list[str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(name, str) for name in value):
        raise ValueError(f"Expected a list of feature names: {path}")
    return value


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    raw: torch.Tensor,
    stats: torch.Tensor,
    mask: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    device: torch.device,
    perturbation: tuple[str, int] | None,
    seed: int,
) -> dict[str, float]:
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    model.eval()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for start in range(0, len(labels), batch_size):
        end = min(start + batch_size, len(labels))
        raw_batch = raw[start:end].clone()
        stats_batch = stats[start:end].clone()
        mask_batch = mask[start:end]
        labels_batch = labels[start:end]
        if perturbation is not None:
            family, feature_index = perturbation
            active = mask_batch.bool()
            count = int(active.sum())
            if count > 1:
                order = torch.randperm(count, generator=generator)
                if family == "raw":
                    values = raw_batch[active, :, feature_index].clone()
                    raw_batch[active, :, feature_index] = values[order]
                else:
                    values = stats_batch[active, 0, feature_index].clone()
                    stats_batch[active, 0, feature_index] = values[order]
        logits = model(raw_batch.to(device), stats_batch.to(device))
        active = mask_batch & labels_batch.ne(-100)
        predictions.append(logits.argmax(-1)[active.to(device)].cpu().numpy())
        targets.append(labels_batch[active].cpu().numpy())
    y_true = np.concatenate(targets)
    y_pred = np.concatenate(predictions)
    negatives = int((y_true == 0).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "far": float(fp / negatives) if negatives else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    raw_dir = args.data_dir / "raw"
    stats_dir = args.data_dir / "stats"
    raw_names = read_names(raw_dir / "feature_names.json")
    stats_names = read_names(stats_dir / "feature_names.json")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    required = {"state_dict", "encoder", "raw_feature_names", "stats_input_dim", "hidden_dim", "dropout"}
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"Checkpoint missing keys: {sorted(missing)}")
    selected_raw_names = checkpoint["raw_feature_names"]
    raw_indices = [raw_names.index(name) for name in selected_raw_names]
    with np.load(raw_dir / "test.npz", allow_pickle=False) as raw_npz, np.load(
        stats_dir / "test.npz", allow_pickle=False
    ) as stats_npz:
        raw = torch.from_numpy(np.asarray(raw_npz["x"])[..., raw_indices].copy()).float()
        stats = torch.from_numpy(np.asarray(stats_npz["x"]).copy()).float()
        mask = torch.from_numpy(np.asarray(raw_npz["mask"]).copy()).bool()
        labels = torch.from_numpy(np.asarray(raw_npz["y"]).copy()).long()
    if stats.shape[-1] != int(checkpoint["stats_input_dim"]):
        raise ValueError("Stats feature count differs from checkpoint")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SignalRawStatsFusion(
        raw_input_dim=raw.shape[-1],
        stats_input_dim=stats.shape[-1],
        encoder=str(checkpoint["encoder"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        dropout=float(checkpoint["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    baseline = evaluate(model, raw, stats, mask, labels, args.batch_size, device, None, args.seed)

    rows: list[dict[str, float | str]] = []
    features = [("raw", index, name) for index, name in enumerate(selected_raw_names)]
    features.extend(("stats", index, name) for index, name in enumerate(stats_names))
    for index, (family, feature_index, name) in enumerate(features):
        result = evaluate(
            model, raw, stats, mask, labels, args.batch_size, device, (family, feature_index), args.seed + index + 1
        )
        rows.append({
            "family": family,
            "feature": name,
            "macro_f1": result["macro_f1"],
            "recall": result["recall"],
            "far": result["far"],
            "macro_f1_drop": baseline["macro_f1"] - result["macro_f1"],
            "recall_drop": baseline["recall"] - result["recall"],
            "far_increase": result["far"] - baseline["far"],
        })
        print(f"{family}:{name} macro_f1_drop={rows[-1]['macro_f1_drop']:.6f}", flush=True)

    result_frame = pd.DataFrame(rows).sort_values("macro_f1_drop", ascending=False).reset_index(drop=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_frame.to_csv(args.output_dir / "feature_permutation_importance.csv", index=False, encoding="utf-8-sig")
    (args.output_dir / "feature_permutation_baseline.json").write_text(
        json.dumps(baseline, indent=2), encoding="utf-8"
    )

    colors = result_frame["family"].map({"raw": "#2f6f9f", "stats": "#d9822b"})
    figure, axes = plt.subplots(1, 2, figsize=(15, max(7, len(result_frame) * 0.28)))
    axes[0].barh(result_frame["family"] + ": " + result_frame["feature"], result_frame["macro_f1_drop"], color=colors)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Macro-F1 drop after permutation")
    axes[0].set_title("Feature importance")
    axes[0].grid(axis="x", alpha=0.25)
    axes[1].barh(result_frame["family"] + ": " + result_frame["feature"], result_frame["recall_drop"], color=colors)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Recall drop after permutation")
    axes[1].set_title("Recall sensitivity")
    axes[1].grid(axis="x", alpha=0.25)
    figure.suptitle(
        f"Permutation importance | baseline Macro-F1={baseline['macro_f1']:.4f}, Recall={baseline['recall']:.4f}",
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(args.output_dir / "feature_permutation_importance.png", dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()

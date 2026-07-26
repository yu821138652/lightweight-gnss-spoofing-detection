"""Train E10: an L5-only expert using all outer-development Sessions.

This is a diagnostic experiment for L5-only test Sessions.  During training,
loss is computed only on active L5 satellite endpoints.  During test, the
expert predicts L5 endpoints and all non-L5 endpoints are forced to normal.
It therefore cannot replace a universal L1/L5 detector; its purpose is to test
whether L1 suppression observations are harming L5 target-spoof detection.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_baseline():
    path = Path(__file__).with_name("21_train_static_signal_fusion.py")
    spec = importlib.util.spec_from_file_location("_static_fusion_l5_expert", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


baseline = _load_baseline()
LOG = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--encoder", choices=baseline.ENCODERS, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--raw-feature-set", choices=tuple(baseline.RAW_FEATURE_SETS), default="full")
    parser.add_argument("--stats-feature-set", choices=baseline.STATS_FEATURE_SETS, default="full")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.hidden_dim < 1 or args.num_workers < 0:
        parser.error("epochs, batch-size, hidden-dim, and num-workers are invalid")
    if not 0.0 <= args.dropout < 1.0:
        parser.error("dropout must be in [0, 1)")
    if args.test_only and args.checkpoint is None:
        parser.error("--test-only requires --checkpoint")
    return args


def l5_active(data: Any, is_l5_index: int) -> torch.Tensor:
    return data.mask & data.y.ne(baseline.IGNORE_INDEX) & data.stats[..., 0, is_l5_index].ge(0.5)


def l5_class_weights(data: Any, is_l5_index: int) -> torch.Tensor:
    labels = data.y[l5_active(data, is_l5_index)]
    if not labels.numel():
        raise ValueError("Training split has no active L5 labels")
    counts = torch.bincount(labels, minlength=2).float()
    missing = torch.nonzero(counts.eq(0), as_tuple=False).flatten().tolist()
    if missing:
        raise ValueError(f"L5 training data must contain both classes; counts={counts.tolist()}")
    return counts.sum() / (2.0 * counts)


def valid_l5(logits: torch.Tensor, stats: torch.Tensor, mask: torch.Tensor, labels: torch.Tensor, is_l5_index: int):
    active = mask & labels.ne(baseline.IGNORE_INDEX) & stats[..., 0, is_l5_index].ge(0.5)
    return logits[active], labels[active]


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    negatives = int((y_true == 0).sum())
    positives = int((y_true == 1).sum())
    return {
        "samples": int(len(y_true)),
        "negative_support": negatives,
        "positive_support": positives,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "far": float(fp / negatives) if negatives else 0.0,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def load_model(checkpoint: dict[str, Any], device: torch.device) -> nn.Module:
    required = {"encoder", "raw_input_dim", "stats_input_dim", "hidden_dim", "dropout", "state_dict", "expert_band"}
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise ValueError(f"Checkpoint is missing fields: {missing}")
    if checkpoint["expert_band"] != "L5":
        raise ValueError(f"Expected an L5-expert checkpoint, got {checkpoint['expert_band']!r}")
    model = baseline.make_model(
        int(checkpoint["raw_input_dim"]), int(checkpoint["stats_input_dim"]),
        str(checkpoint["encoder"]), int(checkpoint["hidden_dim"]), float(checkpoint["dropout"]), device,
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model


@torch.no_grad()
def evaluate(model: nn.Module, data: Any, data_dir: Path, device: torch.device, batch_size: int, is_l5_index: int) -> dict[str, Any]:
    model.eval()
    predictions = torch.zeros_like(data.y)
    for start in range(0, len(data), batch_size):
        stop = min(start + batch_size, len(data))
        raw = data.raw[start:stop].to(device)
        stats = data.stats[start:stop].to(device)
        l5 = stats[..., 0, is_l5_index].ge(0.5)
        expert_prediction = model(raw, stats).argmax(-1).cpu()
        predictions[start:stop] = torch.where(l5.cpu(), expert_prediction, torch.zeros_like(expert_prediction))

    active = data.mask & data.y.ne(baseline.IGNORE_INDEX)
    l5 = data.stats[..., 0, is_l5_index].ge(0.5)
    labels = data.y.cpu().numpy()
    predicted = predictions.cpu().numpy()
    result: dict[str, Any] = {
        "overall_with_l1_forced_normal": metrics(labels[active.cpu().numpy()], predicted[active.cpu().numpy()]),
        "l5_only": metrics(labels[(active & l5).cpu().numpy()], predicted[(active & l5).cpu().numpy()]),
    }
    device_grid = data.device_id
    if device_grid.ndim == 1:
        device_grid = device_grid[:, None].expand_as(data.y)
    mapping_path = data_dir / "device_mapping.json"
    mapping = json.loads(mapping_path.read_text(encoding="utf-8")) if mapping_path.is_file() else {}
    names = {int(value): name for name, value in mapping.items()}
    by_device: dict[str, dict[str, float | int]] = {}
    for device_id in torch.unique(device_grid[active & l5], sorted=True).tolist():
        group = active & l5 & device_grid.eq(device_id)
        by_device[names.get(int(device_id), str(device_id))] = metrics(
            labels[group.cpu().numpy()], predicted[group.cpu().numpy()]
        )
    result["l5_by_device"] = by_device
    return result


def main() -> None:
    args = parse_args()
    baseline.seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_dir, stats_dir, raw_indices, raw_names, all_stats_names = baseline.load_data_contract(args.data_dir, args.raw_feature_set)
    stats_names = baseline.select_stats_feature_names(all_stats_names, args.stats_feature_set)
    stats_indices = baseline.feature_indices(all_stats_names, stats_names, "Stats")
    if "IsL5" not in stats_names:
        raise ValueError("E10 requires IsL5 in the selected stats feature set")
    is_l5_index = stats_names.index("IsL5")
    raw_count = len(baseline.load_feature_names(raw_dir / "feature_names.json"))
    stats_count = len(all_stats_names)

    if args.test_only:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        test = baseline.load_split("test", raw_dir, stats_dir, raw_count, stats_count, raw_indices, stats_indices)
        if int(checkpoint["raw_input_dim"]) != test.raw.shape[-1] or int(checkpoint["stats_input_dim"]) != test.stats.shape[-1]:
            raise ValueError("Checkpoint feature dimensions do not match the selected test features")
        model = load_model(checkpoint, device)
        result = evaluate(model, test, args.data_dir, device, args.batch_size, is_l5_index)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = args.output_dir / "test_metrics_l5_expert.json"
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        LOG.info("locked L5-expert test=%s", json.dumps(result))
        return

    train = baseline.load_split("train", raw_dir, stats_dir, raw_count, stats_count, raw_indices, stats_indices)
    weights = l5_class_weights(train, is_l5_index)
    model = baseline.make_model(train.raw.shape[-1], train.stats.shape[-1], args.encoder, args.hidden_dim, args.dropout, device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    LOG.info("E10 L5-only refit encoder=%s device=%s params=%d epochs=%d train_windows=%d l5_class_weights=%s", args.encoder, device, parameter_count, args.epochs, len(train), weights.tolist())
    if args.dry_run:
        raw, stats, mask, labels = next(iter(loader))
        logits, target = valid_l5(model(raw.to(device), stats.to(device)), stats.to(device), mask.to(device), labels.to(device), is_l5_index)
        LOG.info("dry-run l5_logits=%s l5_labels=%d", tuple(logits.shape), target.numel())
        return

    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        label_count = 0
        for raw, stats, mask, labels in tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}"):
            raw, stats, mask, labels = raw.to(device), stats.to(device), mask.to(device), labels.to(device)
            logits, target = valid_l5(model(raw, stats), stats, mask, labels, is_l5_index)
            if not target.numel():
                continue
            loss = criterion(logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * target.numel()
            label_count += target.numel()
        if not label_count:
            raise ValueError("Training epoch has no active L5 labels")
        LOG.info("epoch=%d l5_train_loss=%.4f", epoch, total_loss / label_count)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / f"best_refit_l5_expert_signal_{args.encoder}_stats_mlp_fusion.pt"
    torch.save({
        "model": f"refit_l5_expert_signal_{args.encoder}_stats_mlp_fusion",
        "expert_band": "L5",
        "decision_protocol": "l5_expert_prediction_with_non_l5_forced_normal",
        "encoder": args.encoder,
        "raw_time_steps": int(train.raw.shape[-2]),
        "raw_input_dim": int(train.raw.shape[-1]),
        "raw_feature_indices": raw_indices,
        "raw_feature_names": raw_names,
        "raw_feature_set": args.raw_feature_set,
        "stats_input_dim": int(train.stats.shape[-1]),
        "stats_feature_names": stats_names,
        "stats_feature_set": args.stats_feature_set,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "weight_decay": args.weight_decay,
        "parameter_count": parameter_count,
        "class_weights": weights.tolist(),
        "selection_protocol": "fixed_epoch_refit_on_all_outer_development_sessions_l5_only_loss",
        "refit_epochs": args.epochs,
        "state_dict": model.state_dict(),
    }, checkpoint_path)
    (args.output_dir / "refit_l5_expert_metadata.json").write_text(json.dumps({
        "checkpoint": str(checkpoint_path), "epochs": args.epochs, "parameter_count": parameter_count,
        "train_windows": len(train), "expert_band": "L5",
        "selection_protocol": "fixed_epoch_refit_on_all_outer_development_sessions_l5_only_loss",
    }, indent=2), encoding="utf-8")
    LOG.info("complete; outer test was not read; checkpoint=%s", checkpoint_path)


if __name__ == "__main__":
    main()
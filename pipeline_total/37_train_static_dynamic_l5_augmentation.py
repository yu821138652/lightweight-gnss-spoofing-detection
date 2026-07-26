"""Train or evaluate E12a dynamic-augmented L5 expert.

Selection mode holds out the tensor directory's complete static validation
Session and never reads its outer test tensors.  ``--refit`` trains a fixed
number of epochs on all static development plus dynamic train-only data.  The
outer test can then be read only through ``--test-only``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_l5_expert():
    path = Path(__file__).with_name("34_refit_static_l5_expert.py")
    spec = importlib.util.spec_from_file_location("_l5_expert_e12a", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


expert = _load_l5_expert()
baseline = expert.baseline
LOG = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--encoder", choices=baseline.ENCODERS, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--raw-feature-set", choices=tuple(baseline.RAW_FEATURE_SETS), default="full")
    parser.add_argument("--stats-feature-set", choices=baseline.STATS_FEATURE_SETS, default="full")
    parser.add_argument("--refit", action="store_true", help="fixed-epoch all-development refit; does not read val/test")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1 or args.hidden_dim < 1 or args.num_workers < 0:
        parser.error("epochs, patience, batch-size, hidden-dim, and num-workers must be positive")
    if not 0 <= args.dropout < 1:
        parser.error("--dropout must be in [0, 1)")
    if args.test_only and args.checkpoint is None:
        parser.error("--test-only requires --checkpoint")
    if args.test_only and args.refit:
        parser.error("--test-only and --refit cannot be used together")
    return args


@torch.no_grad()
def evaluate_l5(model: nn.Module, data: Any, device: torch.device, batch_size: int, is_l5_index: int) -> dict[str, float | int]:
    model.eval()
    predictions = torch.zeros_like(data.y)
    for start in range(0, len(data), batch_size):
        stop = min(start + batch_size, len(data))
        logits = model(data.raw[start:stop].to(device), data.stats[start:stop].to(device))
        predictions[start:stop] = logits.argmax(-1).cpu()
    active = expert.l5_active(data, is_l5_index)
    return expert.metrics(data.y[active].cpu().numpy(), predictions[active].cpu().numpy())


def checkpoint_payload(
    model: nn.Module,
    train: Any,
    args: argparse.Namespace,
    raw_indices: list[int],
    raw_names: list[str],
    stats_names: list[str],
    weights: torch.Tensor,
    parameter_count: int,
    selection: str,
    selected_epoch: int,
) -> dict[str, Any]:
    return {
        "model": "e12a_static_dynamic_l5_expert_signal_fusion",
        "experiment": "E12a_static_dynamic_l5_augmentation",
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
        "selection_protocol": selection,
        "selected_epoch": selected_epoch,
        "state_dict": model.state_dict(),
    }


def main() -> None:
    args = parse_args()
    baseline.seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_dir, stats_dir, raw_indices, raw_names, all_stats_names = baseline.load_data_contract(args.data_dir, args.raw_feature_set)
    stats_names = baseline.select_stats_feature_names(all_stats_names, args.stats_feature_set)
    stats_indices = baseline.feature_indices(all_stats_names, stats_names, "Stats")
    if "IsL5" not in stats_names:
        raise ValueError("E12a requires IsL5 in the selected stats feature set")
    is_l5_index = stats_names.index("IsL5")
    raw_count = len(baseline.load_feature_names(raw_dir / "feature_names.json"))
    stats_count = len(all_stats_names)

    if args.test_only:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if checkpoint.get("experiment") != "E12a_static_dynamic_l5_augmentation":
            raise ValueError("Checkpoint is not an E12a static+dynamic L5 augmentation model")
        test = baseline.load_split("test", raw_dir, stats_dir, raw_count, stats_count, raw_indices, stats_indices)
        if checkpoint["raw_input_dim"] != test.raw.shape[-1] or checkpoint["stats_input_dim"] != test.stats.shape[-1]:
            raise ValueError("Checkpoint feature dimensions do not match selected test features")
        model = expert.load_model(checkpoint, device)
        result = expert.evaluate(model, test, args.data_dir, device, args.batch_size, is_l5_index)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = args.output_dir / "test_metrics_e12a_l5_expert.json"
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        LOG.info("E12a outer static test=%s", json.dumps(result))
        return

    train = baseline.load_split("train", raw_dir, stats_dir, raw_count, stats_count, raw_indices, stats_indices)
    val = None if args.refit else baseline.load_split("val", raw_dir, stats_dir, raw_count, stats_count, raw_indices, stats_indices)
    weights = expert.l5_class_weights(train, is_l5_index)
    model = baseline.make_model(train.raw.shape[-1], train.stats.shape[-1], args.encoder, args.hidden_dim, args.dropout, device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    mode = "fixed refit" if args.refit else "inner static Session selection"
    LOG.info("E12a %s encoder=%s device=%s params=%d train_windows=%d l5_class_weights=%s", mode, args.encoder, device, parameter_count, len(train), weights.tolist())
    if args.dry_run:
        raw, stats, mask, labels = next(iter(loader))
        logits, target = expert.valid_l5(model(raw.to(device), stats.to(device)), stats.to(device), mask.to(device), labels.to(device), is_l5_index)
        LOG.info("dry-run l5_logits=%s l5_labels=%d", tuple(logits.shape), target.numel())
        return

    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "best_e12a_static_dynamic_l5_expert_signal_fusion.pt"
    best_score = float("-inf")
    best_epoch = 0
    best_val_metrics: dict[str, float | int] | None = None
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        label_count = 0
        for raw, stats, mask, labels in tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}"):
            raw, stats, mask, labels = raw.to(device), stats.to(device), mask.to(device), labels.to(device)
            logits, target = expert.valid_l5(model(raw, stats), stats, mask, labels, is_l5_index)
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
        if args.refit:
            LOG.info("epoch=%d l5_train_loss=%.4f", epoch, total_loss / label_count)
            continue
        assert val is not None
        val_metrics = evaluate_l5(model, val, device, args.batch_size, is_l5_index)
        LOG.info("epoch=%d l5_train_loss=%.4f static_val=%s", epoch, total_loss / label_count, json.dumps(val_metrics))
        score = float(val_metrics["macro_f1"])
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_val_metrics = val_metrics
            stale = 0
            torch.save(checkpoint_payload(
                model, train, args, raw_indices, raw_names, stats_names, weights, parameter_count,
                "inner_static_session_validation_with_dynamic_train_augmentation", best_epoch,
            ), checkpoint_path)
        else:
            stale += 1
            if stale >= args.patience:
                LOG.info("early stopping after %d epochs without static validation macro-F1 improvement", args.patience)
                break
    if args.refit:
        best_epoch = args.epochs
        torch.save(checkpoint_payload(
            model, train, args, raw_indices, raw_names, stats_names, weights, parameter_count,
            "fixed_epoch_refit_on_all_static_development_sessions_plus_dynamic_l5_l15_train_only", best_epoch,
        ), checkpoint_path)
    metadata = {
        "checkpoint": str(checkpoint_path), "epochs_completed": best_epoch,
        "parameter_count": parameter_count, "train_windows": len(train), "mode": mode,
        "best_static_val_l5_metrics": best_val_metrics, "outer_test_read": False,
    }
    (args.output_dir / "e12a_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    LOG.info("complete; outer test was not read; checkpoint=%s", checkpoint_path)


if __name__ == "__main__":
    main()

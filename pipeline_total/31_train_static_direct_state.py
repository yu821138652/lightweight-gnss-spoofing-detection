"""Train E7: direct three-class static signal-state classification.

The state is one of: normal, formal target-band spoofed, or non-target-band
observation during a reviewed single-band static attack interval.  The third
class is an observed attack-associated context, not a claim that the signal
itself was spoofed.  This is a distinct primary task from E0's formal binary
target-band spoof detector.
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
from sklearn.metrics import f1_score, recall_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import SignalRawStatsFusion


def _load_module(name: str, filename: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


baseline = _load_module("_static_fusion_baseline_e7", "21_train_static_signal_fusion.py")
e3 = _load_module("_auxiliary_state_e7", "27_train_static_auxiliary_state.py")
experts = _load_module("_band_experts_e7", "25_train_static_band_experts.py")
LOG = logging.getLogger(__name__)
STATE_NAMES = e3.AUXILIARY_NAMES
NUM_STATES = len(STATE_NAMES)


def checkpoint_path(output_dir: Path, encoder: str) -> Path:
    return output_dir / f"best_direct_state_signal_{encoder}_stats_mlp_fusion.pt"


def state_weights(data: e3.AuxiliaryStateDataset) -> torch.Tensor:
    active = data.auxiliary.ne(baseline.IGNORE_INDEX)
    counts = torch.bincount(data.auxiliary[active], minlength=NUM_STATES).float()
    if int(counts.min().item()) == 0:
        raise ValueError(f"E7 training split needs all states, found {counts.tolist()}")
    return counts.sum() / (NUM_STATES * counts)


def make_model(args: argparse.Namespace, raw_dim: int, stats_dim: int, device: torch.device) -> SignalRawStatsFusion:
    return SignalRawStatsFusion(
        raw_input_dim=raw_dim,
        stats_input_dim=stats_dim,
        encoder=args.encoder,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        num_classes=NUM_STATES,
    ).to(device)


def state_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    support = {name: int((labels == index).sum()) for index, name in enumerate(STATE_NAMES)}
    recall = {
        name: float(recall_score(labels, predictions, labels=[index], average="macro", zero_division=0))
        for index, name in enumerate(STATE_NAMES)
    }
    return {
        "samples": int(len(labels)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", labels=list(range(NUM_STATES)), zero_division=0)),
        "state_support": support,
        "state_recall": recall,
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    labels_all: list[np.ndarray] = []
    predictions_all: list[np.ndarray] = []
    loss_total = 0.0
    count = 0
    for raw, stats, mask, _formal, states in loader:
        raw = raw.to(device)
        stats = stats.to(device)
        mask = mask.to(device)
        states = states.to(device)
        active = mask & states.ne(baseline.IGNORE_INDEX)
        if not active.any():
            continue
        logits = model(raw, stats)
        selected = states[active]
        loss_total += float(nn.functional.cross_entropy(logits[active], selected).item()) * selected.numel()
        count += selected.numel()
        labels_all.append(selected.cpu().numpy())
        predictions_all.append(logits[active].argmax(-1).cpu().numpy())
    if not count:
        raise ValueError("Evaluation split has no active E7 labels")
    labels = np.concatenate(labels_all)
    predictions = np.concatenate(predictions_all)
    result = state_metrics(labels, predictions)
    result["loss"] = loss_total / count
    # This projection makes the E0 comparison explicit: only class 1 is a
    # formal target-band spoof prediction; class 2 remains a separate state.
    result["formal_binary_projection"] = experts.metrics(labels == 1, predictions == 1)
    result["attack_associated_projection"] = experts.metrics(labels != 0, predictions != 0)
    return result


def save_checkpoint(
    path: Path,
    args: argparse.Namespace,
    train: e3.AuxiliaryStateDataset,
    raw_names: list[str],
    stats_names: list[str],
    model: nn.Module,
    parameter_count: int,
    weights: torch.Tensor,
    val_metrics: dict[str, Any],
) -> None:
    torch.save(
        {
            "model": f"direct_state_signal_{args.encoder}_stats_mlp_fusion",
            "architecture": "raw_stats_fusion_direct_three_class_state",
            "states": list(STATE_NAMES),
            "state_semantics": "normal|formal_target_spoofed|non_target_during_reviewed_single_band_attack",
            "label_config": str(args.label_config.resolve()),
            "label_config_sha256": train.label_config_sha256,
            "encoder": args.encoder,
            "raw_time_steps": int(train.raw.shape[-2]),
            "raw_input_dim": int(train.raw.shape[-1]),
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
            "train_state_support": train.auxiliary_support,
            "state_dict": model.state_dict(),
            "val_metrics": val_metrics,
        },
        path,
    )


def validate_checkpoint(checkpoint: dict[str, Any], data: e3.AuxiliaryStateDataset, raw_names: list[str], stats_names: list[str]) -> None:
    required = {
        "architecture", "states", "label_config_sha256", "encoder", "raw_time_steps", "raw_input_dim",
        "raw_feature_names", "stats_input_dim", "stats_feature_names", "hidden_dim", "dropout", "state_dict",
    }
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise ValueError(f"Checkpoint missing metadata: {missing}")
    if checkpoint["architecture"] != "raw_stats_fusion_direct_three_class_state":
        raise ValueError(f"Unsupported checkpoint architecture: {checkpoint['architecture']!r}")
    if tuple(checkpoint["states"]) != STATE_NAMES:
        raise ValueError("Checkpoint state definitions differ")
    if checkpoint["label_config_sha256"] != data.label_config_sha256:
        raise ValueError("Label configuration changed since checkpoint selection")
    if tuple(data.raw.shape[-2:]) != (int(checkpoint["raw_time_steps"]), int(checkpoint["raw_input_dim"])):
        raise ValueError("Checkpoint raw input shape differs")
    if data.stats.shape[-1] != int(checkpoint["stats_input_dim"]):
        raise ValueError("Checkpoint stats input dimension differs")
    if checkpoint["raw_feature_names"] != raw_names or checkpoint["stats_feature_names"] != stats_names:
        raise ValueError("Checkpoint feature names/order differ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label-config", type=Path, default=ROOT / "configs" / "preprocessing.yml")
    parser.add_argument("--encoder", choices=baseline.ENCODERS, default="tcn")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--raw-feature-set", choices=tuple(baseline.RAW_FEATURE_SETS), default="full")
    parser.add_argument("--stats-feature-set", choices=baseline.STATS_FEATURE_SETS, default="cn0_agc_coverage_rx_time_std")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--test-only", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.hidden_dim < 1 or args.patience < 1 or args.num_workers < 0:
        parser.error("epochs, batch-size, hidden-dim, patience, and num-workers are invalid")
    if not 0.0 <= args.dropout < 1.0:
        parser.error("dropout must be in [0, 1)")
    return args


def main() -> None:
    args = parse_args()
    baseline.seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_dir, stats_dir, raw_indices, raw_names, all_stats_names = baseline.load_data_contract(args.data_dir, args.raw_feature_set)
    stats_names = baseline.select_stats_feature_names(all_stats_names, args.stats_feature_set)
    stats_indices = baseline.feature_indices(all_stats_names, stats_names, "Stats")
    raw_count = len(baseline.load_feature_names(raw_dir / "feature_names.json"))
    stats_count = len(all_stats_names)
    train = e3.load_split("train", raw_dir, stats_dir, raw_count, stats_count, raw_indices, stats_indices, args.data_dir, args.label_config)
    pin_memory = device.type == "cuda"

    if args.test_only:
        path = checkpoint_path(args.output_dir, args.encoder)
        if not path.is_file():
            raise FileNotFoundError(path)
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError(f"Invalid checkpoint: {path}")
        test = e3.load_split("test", raw_dir, stats_dir, raw_count, stats_count, raw_indices, stats_indices, args.data_dir, args.label_config)
        baseline.validate_compatible(train, test, "test")
        validate_checkpoint(checkpoint, test, raw_names, stats_names)
        model_args = argparse.Namespace(encoder=checkpoint["encoder"], hidden_dim=checkpoint["hidden_dim"], dropout=checkpoint["dropout"])
        model = make_model(model_args, int(checkpoint["raw_input_dim"]), int(checkpoint["stats_input_dim"]), device)
        model.load_state_dict(checkpoint["state_dict"])
        metrics = evaluate(model, DataLoader(test, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=pin_memory), device)
        metrics.update({"checkpoint": str(path), "parameter_count": int(checkpoint["parameter_count"]), "train_state_support": checkpoint["train_state_support"]})
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "test_metrics_direct_state.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        LOG.info("locked checkpoint test=%s", json.dumps(metrics))
        return

    val = e3.load_split("val", raw_dir, stats_dir, raw_count, stats_count, raw_indices, stats_indices, args.data_dir, args.label_config)
    baseline.validate_compatible(train, val, "val")
    weights = state_weights(train)
    model = make_model(args, train.raw.shape[-1], train.stats.shape[-1], device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    LOG.info("architecture=direct_state encoder=%s device=%s params=%d train_state_support=%s", args.encoder, device, parameter_count, json.dumps(train.auxiliary_support))
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=pin_memory)
    if args.dry_run:
        raw, stats, _, _, _ = next(iter(train_loader))
        LOG.info("dry-run logits=%s", tuple(model(raw.to(device), stats.to(device)).shape))
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = -float("inf")
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        count = 0
        for raw, stats, mask, _formal, states in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            raw = raw.to(device)
            stats = stats.to(device)
            mask = mask.to(device)
            states = states.to(device)
            active = mask & states.ne(baseline.IGNORE_INDEX)
            logits = model(raw, stats)
            selected = states[active]
            loss = criterion(logits[active], selected)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * selected.numel()
            count += selected.numel()
        metrics = evaluate(model, val_loader, device)
        LOG.info("epoch=%d train_loss=%.4f val=%s", epoch, total_loss / count, json.dumps(metrics))
        if float(metrics["macro_f1"]) > best:
            best = float(metrics["macro_f1"])
            stale = 0
            save_checkpoint(checkpoint_path(args.output_dir, args.encoder), args, train, raw_names, stats_names, model, parameter_count, weights, metrics)
            (args.output_dir / "val_metrics_direct_state.json").write_text(json.dumps({**metrics, "parameter_count": parameter_count, "train_state_support": train.auxiliary_support}, indent=2), encoding="utf-8")
        else:
            stale += 1
            if stale >= args.patience:
                LOG.info("early stopping")
                break
    LOG.info("complete; test was not read")


if __name__ == "__main__":
    main()

"""Train E11: L5-only shared encoder with device-conditional classifier heads.

Known receiver device IDs route L5 predictions to small device heads.  The raw
and statistics encoders remain shared; a fallback head is trained with an
auxiliary loss for an unseen device.  This diagnostic still forces non-L5 test
endpoints to normal, so it is not a universal L1/L5 deployment model.
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
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import SignalRawStatsDeviceConditionalHeads


def load_module(name: str, filename: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


baseline = load_module("_e11_baseline", "21_train_static_signal_fusion.py")
e10 = load_module("_e11_l5_expert", "34_refit_static_l5_expert.py")
LOG = logging.getLogger(__name__)


class DeviceFusionDataset(Dataset):
    """Expose device metadata without changing the baseline dataset contract."""

    def __init__(self, source: Any) -> None:
        self.source = source

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int):
        return (
            self.source.raw[index], self.source.stats[index], self.source.mask[index],
            self.source.y[index], self.source.device_id[index],
        )


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
    parser.add_argument("--fallback-loss-weight", type=float, default=0.25)
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
    if not 0.0 <= args.dropout < 1.0 or args.fallback_loss_weight < 0.0:
        parser.error("dropout must be in [0, 1), fallback-loss-weight must be non-negative")
    if args.test_only and args.checkpoint is None:
        parser.error("--test-only requires --checkpoint")
    return args


def device_grid(device_id: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return device_id[:, None].expand_as(labels) if device_id.ndim == 1 else device_id


def device_class_weights(data: Any, is_l5_index: int) -> tuple[dict[int, torch.Tensor], torch.Tensor, list[int]]:
    active = e10.l5_active(data, is_l5_index)
    grid = device_grid(data.device_id, data.y)
    device_ids = [int(value) for value in torch.unique(grid[active], sorted=True).tolist()]
    if not device_ids:
        raise ValueError("Training split has no active L5 device IDs")
    weights: dict[int, torch.Tensor] = {}
    for device_id in device_ids:
        labels = data.y[active & grid.eq(device_id)]
        counts = torch.bincount(labels, minlength=2).float()
        if torch.any(counts.eq(0)):
            raise ValueError(f"Device {device_id} lacks an L5 class: counts={counts.tolist()}")
        weights[device_id] = counts.sum() / (2.0 * counts)
    return weights, e10.l5_class_weights(data, is_l5_index), device_ids


def device_balanced_loss(
    routed: torch.Tensor,
    fallback: torch.Tensor,
    stats: torch.Tensor,
    mask: torch.Tensor,
    labels: torch.Tensor,
    device_id: torch.Tensor,
    is_l5_index: int,
    weights: dict[int, torch.Tensor],
    fallback_weights: torch.Tensor,
    fallback_loss_weight: float,
) -> tuple[torch.Tensor, int]:
    active = mask & labels.ne(baseline.IGNORE_INDEX) & stats[..., 0, is_l5_index].ge(0.5)
    grid = device_grid(device_id, labels)
    losses: list[torch.Tensor] = []
    for known_device_id, class_weight in weights.items():
        selected = active & grid.eq(known_device_id)
        if selected.any():
            losses.append(nn.functional.cross_entropy(routed[selected], labels[selected], weight=class_weight.to(routed.device)))
    if not losses:
        raise ValueError("Training batch has no active L5 endpoints for known devices")
    routed_loss = torch.stack(losses).mean()
    fallback_loss = nn.functional.cross_entropy(
        fallback[active], labels[active], weight=fallback_weights.to(fallback.device)
    )
    return routed_loss + fallback_loss_weight * fallback_loss, int(active.sum().item())


def make_model(args: argparse.Namespace, raw_dim: int, stats_dim: int, device_ids: list[int], device: torch.device) -> nn.Module:
    return SignalRawStatsDeviceConditionalHeads(
        raw_input_dim=raw_dim, stats_input_dim=stats_dim, device_ids=device_ids,
        encoder=args.encoder, hidden_dim=args.hidden_dim, dropout=args.dropout,
    ).to(device)


@torch.no_grad()
def evaluate(model: nn.Module, data: Any, data_dir: Path, device: torch.device, batch_size: int, is_l5_index: int) -> dict[str, Any]:
    model.eval()
    predictions = torch.zeros_like(data.y)
    for start in range(0, len(data), batch_size):
        stop = min(start + batch_size, len(data))
        raw, stats = data.raw[start:stop].to(device), data.stats[start:stop].to(device)
        device_id = data.device_id[start:stop].to(device)
        l5 = stats[..., 0, is_l5_index].ge(0.5)
        routed = model(raw, stats, device_id).argmax(-1).cpu()
        predictions[start:stop] = torch.where(l5.cpu(), routed, torch.zeros_like(routed))
    active = data.mask & data.y.ne(baseline.IGNORE_INDEX)
    l5 = data.stats[..., 0, is_l5_index].ge(0.5)
    labels, predicted = data.y.cpu().numpy(), predictions.cpu().numpy()
    result: dict[str, Any] = {
        "overall_with_l1_forced_normal": e10.metrics(labels[active.cpu().numpy()], predicted[active.cpu().numpy()]),
        "l5_only": e10.metrics(labels[(active & l5).cpu().numpy()], predicted[(active & l5).cpu().numpy()]),
    }
    mapping = json.loads((data_dir / "device_mapping.json").read_text(encoding="utf-8"))
    names = {int(value): name for name, value in mapping.items()}
    grid = device_grid(data.device_id, data.y)
    by_device: dict[str, dict[str, float | int]] = {}
    for device_id in torch.unique(grid[active & l5], sorted=True).tolist():
        selected = active & l5 & grid.eq(device_id)
        by_device[names.get(int(device_id), str(device_id))] = e10.metrics(labels[selected.cpu().numpy()], predicted[selected.cpu().numpy()])
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
        raise ValueError("E11 requires IsL5 in the selected stats feature set")
    is_l5_index = stats_names.index("IsL5")
    raw_count, stats_count = len(baseline.load_feature_names(raw_dir / "feature_names.json")), len(all_stats_names)

    if args.test_only:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        required = {"architecture", "device_ids", "state_dict", "raw_input_dim", "stats_input_dim", "encoder", "hidden_dim", "dropout"}
        missing = sorted(required.difference(checkpoint))
        if missing or checkpoint.get("architecture") != "shared_fusion_l5_device_conditional_heads":
            raise ValueError(f"Invalid E11 checkpoint; missing={missing}")
        test = baseline.load_split("test", raw_dir, stats_dir, raw_count, stats_count, raw_indices, stats_indices)
        if int(checkpoint["raw_input_dim"]) != test.raw.shape[-1] or int(checkpoint["stats_input_dim"]) != test.stats.shape[-1]:
            raise ValueError("Checkpoint feature dimensions do not match the selected test features")
        model = SignalRawStatsDeviceConditionalHeads(
            raw_input_dim=int(checkpoint["raw_input_dim"]), stats_input_dim=int(checkpoint["stats_input_dim"]),
            device_ids=[int(value) for value in checkpoint["device_ids"]], encoder=str(checkpoint["encoder"]),
            hidden_dim=int(checkpoint["hidden_dim"]), dropout=float(checkpoint["dropout"]),
        ).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        result = evaluate(model, test, args.data_dir, device, args.batch_size, is_l5_index)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "test_metrics_l5_device_heads.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        LOG.info("locked E11 test=%s", json.dumps(result))
        return

    train = baseline.load_split("train", raw_dir, stats_dir, raw_count, stats_count, raw_indices, stats_indices)
    per_device_weights, fallback_weights, device_ids = device_class_weights(train, is_l5_index)
    model = make_model(args, train.raw.shape[-1], train.stats.shape[-1], device_ids, device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    loader = DataLoader(DeviceFusionDataset(train), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    LOG.info("E11 device-head refit encoder=%s device=%s params=%d epochs=%d train_windows=%d device_ids=%s", args.encoder, device, parameter_count, args.epochs, len(train), device_ids)
    LOG.info("E11 L5 class weights by device=%s fallback=%s", {key: value.tolist() for key, value in per_device_weights.items()}, fallback_weights.tolist())
    if args.dry_run:
        raw, stats, mask, labels, batch_device_id = next(iter(loader))
        routed, fallback = model(raw.to(device), stats.to(device), batch_device_id.to(device), return_fallback=True)
        loss, count = device_balanced_loss(routed, fallback, stats.to(device), mask.to(device), labels.to(device), batch_device_id.to(device), is_l5_index, per_device_weights, fallback_weights, args.fallback_loss_weight)
        LOG.info("dry-run routed_logits=%s l5_labels=%d loss=%.4f", tuple(routed.shape), count, float(loss.item()))
        return

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, label_count = 0.0, 0
        for raw, stats, mask, labels, batch_device_id in tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}"):
            raw, stats, mask, labels, batch_device_id = raw.to(device), stats.to(device), mask.to(device), labels.to(device), batch_device_id.to(device)
            routed, fallback = model(raw, stats, batch_device_id, return_fallback=True)
            loss, count = device_balanced_loss(routed, fallback, stats, mask, labels, batch_device_id, is_l5_index, per_device_weights, fallback_weights, args.fallback_loss_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * count
            label_count += count
        LOG.info("epoch=%d device_balanced_l5_train_loss=%.4f", epoch, total_loss / label_count)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / f"best_refit_l5_device_heads_signal_{args.encoder}_stats_mlp_fusion.pt"
    torch.save({
        "model": f"refit_l5_device_heads_signal_{args.encoder}_stats_mlp_fusion",
        "architecture": "shared_fusion_l5_device_conditional_heads",
        "expert_band": "L5",
        "routing_source": "receiver device ID",
        "decision_protocol": "l5_device_head_prediction_with_non_l5_forced_normal",
        "encoder": args.encoder, "device_ids": device_ids,
        "raw_time_steps": int(train.raw.shape[-2]), "raw_input_dim": int(train.raw.shape[-1]),
        "raw_feature_names": raw_names, "raw_feature_set": args.raw_feature_set,
        "stats_input_dim": int(train.stats.shape[-1]), "stats_feature_names": stats_names, "stats_feature_set": args.stats_feature_set,
        "hidden_dim": args.hidden_dim, "dropout": args.dropout, "weight_decay": args.weight_decay,
        "fallback_loss_weight": args.fallback_loss_weight, "parameter_count": parameter_count,
        "l5_class_weights_by_device": {str(key): value.tolist() for key, value in per_device_weights.items()},
        "selection_protocol": "fixed_epoch_refit_on_all_outer_development_sessions_l5_only_device_balanced_loss",
        "refit_epochs": args.epochs, "state_dict": model.state_dict(),
    }, checkpoint_path)
    (args.output_dir / "refit_l5_device_heads_metadata.json").write_text(json.dumps({
        "checkpoint": str(checkpoint_path), "epochs": args.epochs, "parameter_count": parameter_count,
        "device_ids": device_ids, "fallback_loss_weight": args.fallback_loss_weight,
        "selection_protocol": "fixed_epoch_refit_on_all_outer_development_sessions_l5_only_device_balanced_loss",
    }, indent=2), encoding="utf-8")
    LOG.info("complete; outer test was not read; checkpoint=%s", checkpoint_path)


if __name__ == "__main__":
    main()
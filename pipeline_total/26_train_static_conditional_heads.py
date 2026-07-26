"""Train E1: shared fusion encoder with frequency-conditional binary heads.

The model keeps the original target-band-only labels and raw/stats inputs.  It
uses the unscaled physical ``IsL5`` sidecar only to select an L1 or L5 output
head.  Training samples are individual active satellite windows; every batch
contains equal counts of L1-/L1+/L5-/L5+ windows.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import SignalRawStatsConditionalHeads


def _load_module(name: str, filename: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


baseline = _load_module("_static_fusion_baseline", "21_train_static_signal_fusion.py")
experts = _load_module("_static_band_experts", "25_train_static_band_experts.py")
LOG = logging.getLogger(__name__)
GROUPS = ("L1-", "L1+", "L5-", "L5+")


class EndpointDataset(Dataset):
    """Expose active [window, signal] tensor endpoints without copying tensors."""

    def __init__(self, source: experts.BandExpertDataset) -> None:
        active = (source.mask & source.y.ne(baseline.IGNORE_INDEX)).cpu().numpy()
        window_index, signal_index = np.nonzero(active)
        self.source = source
        self.window_index = torch.from_numpy(window_index.astype(np.int64, copy=False))
        self.signal_index = torch.from_numpy(signal_index.astype(np.int64, copy=False))
        labels = source.y[active].cpu().numpy().astype(np.int64, copy=False)
        bands = source.is_l5[active].cpu().numpy().astype(bool, copy=False)
        self.groups: dict[str, np.ndarray] = {
            "L1-": np.flatnonzero(~bands & (labels == 0)),
            "L1+": np.flatnonzero(~bands & (labels == 1)),
            "L5-": np.flatnonzero(bands & (labels == 0)),
            "L5+": np.flatnonzero(bands & (labels == 1)),
        }
        missing = [name for name, indices in self.groups.items() if not len(indices)]
        if missing:
            raise ValueError(f"Training endpoints have empty frequency x class groups: {missing}")

    def __len__(self) -> int:
        return len(self.window_index)

    def __getitem__(self, index: int):
        window = int(self.window_index[index])
        signal = int(self.signal_index[index])
        return (
            self.source.raw[window, signal],
            self.source.stats[window, signal],
            self.source.y[window, signal],
            self.source.is_l5[window, signal],
        )


class FrequencyClassBalancedBatchSampler(Sampler[list[int]]):
    """Draw exactly the same endpoint count from each frequency x class group."""

    def __init__(
        self,
        groups: dict[str, np.ndarray],
        batch_size: int,
        steps_per_epoch: int,
        seed: int,
    ) -> None:
        if batch_size % len(GROUPS):
            raise ValueError("--batch-size must be divisible by 4 for strict frequency x class balancing")
        self.groups = groups
        self.quota = batch_size // len(GROUPS)
        self.steps_per_epoch = steps_per_epoch
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __iter__(self) -> Iterator[list[int]]:
        generator = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        for _ in range(self.steps_per_epoch):
            batch = np.concatenate([
                generator.choice(self.groups[name], size=self.quota, replace=True)
                for name in GROUPS
            ])
            generator.shuffle(batch)
            yield batch.tolist()


def make_model(args: argparse.Namespace, raw_dim: int, stats_dim: int, device: torch.device) -> SignalRawStatsConditionalHeads:
    return SignalRawStatsConditionalHeads(
        raw_input_dim=raw_dim,
        stats_input_dim=stats_dim,
        encoder=args.encoder,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, float | int], dict[str, dict[str, float | int]]]:
    model.eval()
    all_labels: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []
    all_bands: list[np.ndarray] = []
    losses: list[float] = []
    for raw, stats, mask, labels, is_l5 in loader:
        raw = raw.to(device)
        stats = stats.to(device)
        mask = mask.to(device)
        labels = labels.to(device)
        is_l5 = is_l5.to(device)
        active = mask & labels.ne(baseline.IGNORE_INDEX)
        if not active.any():
            continue
        logits = model(raw, stats, is_l5)
        selected_logits = logits[active]
        selected_labels = labels[active]
        losses.append(float(nn.functional.cross_entropy(selected_logits, selected_labels).item()) * selected_labels.numel())
        all_labels.append(selected_labels.cpu().numpy())
        all_predictions.append(selected_logits.argmax(-1).cpu().numpy())
        all_bands.append(is_l5[active].cpu().numpy())
    if not all_labels:
        raise ValueError("Evaluation split has no active labels")
    labels = np.concatenate(all_labels)
    predictions = np.concatenate(all_predictions)
    bands = np.concatenate(all_bands).astype(bool, copy=False)
    overall = experts.metrics(labels, predictions)
    overall["loss"] = sum(losses) / len(labels)
    return overall, {
        "L1": experts.metrics(labels[~bands], predictions[~bands]),
        "L5": experts.metrics(labels[bands], predictions[bands]),
    }


@torch.no_grad()
def write_group_metrics(
    data: experts.BandExpertDataset,
    model: nn.Module,
    device: torch.device,
    batch_size: int,
    data_dir: Path,
    output_path: Path,
) -> None:
    mapping = json.loads((data_dir / "device_mapping.json").read_text(encoding="utf-8"))
    inverse_mapping = {int(value): str(key) for key, value in mapping.items()}
    prediction = np.zeros(tuple(data.y.shape), dtype=np.int64)
    model.eval()
    for start in range(0, len(data), batch_size):
        stop = min(start + batch_size, len(data))
        logits = model(
            data.raw[start:stop].to(device),
            data.stats[start:stop].to(device),
            data.is_l5[start:stop].to(device),
        )
        prediction[start:stop] = logits.argmax(-1).cpu().numpy()
    mask = (data.mask & data.y.ne(baseline.IGNORE_INDEX)).cpu().numpy()
    labels = data.y.cpu().numpy()
    bands = data.is_l5.cpu().numpy().astype(bool, copy=False)
    device_ids = data.device_id.cpu().numpy()
    if device_ids.ndim == 1:
        device_ids = np.broadcast_to(device_ids[:, None], labels.shape)
    rows: list[dict[str, object]] = []
    for device_id in sorted(np.unique(device_ids[mask]).tolist()):
        for is_l5, band_name in ((False, "L1"), (True, "L5")):
            selected = mask & (device_ids == device_id) & (bands == is_l5)
            if not selected.any():
                continue
            row: dict[str, object] = {
                "device_id": int(device_id),
                "device_name": inverse_mapping.get(int(device_id), f"unknown_{device_id}"),
                "band": band_name,
            }
            row.update(experts.metrics(labels[selected], prediction[selected]))
            rows.append(row)
    total: dict[str, object] = {"device_id": "ALL", "device_name": "ALL", "band": "ALL"}
    total.update(experts.metrics(labels[mask], prediction[mask]))
    rows.append(total)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def checkpoint_path(output_dir: Path, encoder: str) -> Path:
    return output_dir / f"best_conditional_heads_signal_{encoder}_stats_mlp_fusion.pt"


def save_checkpoint(
    path: Path,
    args: argparse.Namespace,
    train: experts.BandExpertDataset,
    raw_names: list[str],
    stats_names: list[str],
    model: nn.Module,
    parameter_count: int,
    group_support: dict[str, int],
    val_metrics: dict[str, float | int],
    val_by_band: dict[str, dict[str, float | int]],
) -> None:
    torch.save({
        "model": f"conditional_heads_signal_{args.encoder}_stats_mlp_fusion",
        "architecture": "shared_fusion_encoder_frequency_conditional_heads",
        "routing_source": "stats:IsL5 (unscaled physical sidecar)",
        "sampling": "strict_endpoint_frequency_x_class_balanced",
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
        "steps_per_epoch": args.steps_per_epoch,
        "parameter_count": parameter_count,
        "train_group_support": group_support,
        "state_dict": model.state_dict(),
        "val_metrics": val_metrics,
        "val_metrics_by_band": val_by_band,
    }, path)


def validate_checkpoint(checkpoint: dict[str, Any], data: experts.BandExpertDataset, raw_names: list[str], stats_names: list[str]) -> None:
    required = {
        "architecture", "encoder", "raw_time_steps", "raw_input_dim", "raw_feature_names",
        "stats_input_dim", "stats_feature_names", "hidden_dim", "dropout", "state_dict",
    }
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise ValueError(f"Checkpoint is missing metadata: {missing}")
    if checkpoint["architecture"] != "shared_fusion_encoder_frequency_conditional_heads":
        raise ValueError(f"Unsupported checkpoint architecture: {checkpoint['architecture']!r}")
    if tuple(data.raw.shape[-2:]) != (int(checkpoint["raw_time_steps"]), int(checkpoint["raw_input_dim"])):
        raise ValueError("Checkpoint raw input shape differs from selected tensor features")
    if data.stats.shape[-1] != int(checkpoint["stats_input_dim"]):
        raise ValueError("Checkpoint stats input dimension differs from selected tensor features")
    if checkpoint["raw_feature_names"] != raw_names or checkpoint["stats_feature_names"] != stats_names:
        raise ValueError("Checkpoint feature names/order differ from selected tensor features")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--encoder", choices=baseline.ENCODERS, default="tcn")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8192,
        help="Endpoint training batch size; must be divisible by 4.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=256,
        help="Window batch size for validation and test evaluation.",
    )
    parser.add_argument("--steps-per-epoch", type=int, default=None)
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
    if (
        args.epochs < 1
        or args.batch_size < 4
        or args.eval_batch_size < 1
        or args.patience < 1
        or args.hidden_dim < 1
    ):
        parser.error("epochs, batch sizes, patience, and hidden-dim must be positive")
    if args.batch_size % 4:
        parser.error("--batch-size must be divisible by 4")
    if args.steps_per_epoch is not None and args.steps_per_epoch < 1:
        parser.error("--steps-per-epoch must be positive")
    if not 0.0 <= args.dropout < 1.0 or args.num_workers < 0:
        parser.error("invalid dropout or num-workers")
    return args


def main() -> None:
    args = parse_args()
    baseline.seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_dir, stats_dir, raw_indices, raw_names, all_stats_names = baseline.load_data_contract(args.data_dir, args.raw_feature_set)
    stats_names = baseline.select_stats_feature_names(all_stats_names, args.stats_feature_set)
    stats_indices = baseline.feature_indices(all_stats_names, stats_names, "Stats")
    if "IsL5" not in all_stats_names:
        raise ValueError("Stats tensors need unscaled IsL5 for conditional head routing")
    raw_count = len(baseline.load_feature_names(raw_dir / "feature_names.json"))
    stats_count = len(all_stats_names)
    is_l5_index = all_stats_names.index("IsL5")
    train = experts.load_split("train", raw_dir, stats_dir, raw_count, stats_count, raw_indices, stats_indices, is_l5_index)
    val = experts.load_split("val", raw_dir, stats_dir, raw_count, stats_count, raw_indices, stats_indices, is_l5_index)
    baseline.validate_compatible(train, val, "val")
    pin_memory = device.type == "cuda"

    if args.test_only:
        path = checkpoint_path(args.output_dir, args.encoder)
        if not path.is_file():
            raise FileNotFoundError(path)
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError(f"Invalid checkpoint: {path}")
        test = experts.load_split("test", raw_dir, stats_dir, raw_count, stats_count, raw_indices, stats_indices, is_l5_index)
        baseline.validate_compatible(train, test, "test")
        validate_checkpoint(checkpoint, test, raw_names, stats_names)
        model_args = argparse.Namespace(
            encoder=str(checkpoint["encoder"]), hidden_dim=int(checkpoint["hidden_dim"]), dropout=float(checkpoint["dropout"]),
        )
        model = make_model(model_args, int(checkpoint["raw_input_dim"]), int(checkpoint["stats_input_dim"]), device)
        model.load_state_dict(checkpoint["state_dict"])
        metrics, by_band = evaluate(
            model,
            DataLoader(
                test,
                batch_size=args.eval_batch_size,
                num_workers=args.num_workers,
                pin_memory=pin_memory,
            ),
            device,
        )
        metrics.update({
            "checkpoint": str(path), "parameter_count": int(checkpoint["parameter_count"]),
            "raw_feature_names": raw_names, "stats_feature_names": stats_names, "metrics_by_band": by_band,
        })
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "test_metrics_conditional_heads.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        write_group_metrics(
            test,
            model,
            device,
            args.eval_batch_size,
            args.data_dir,
            args.output_dir / "test_metrics_by_device_band.csv",
        )
        LOG.info("locked checkpoint test=%s", json.dumps(metrics))
        return

    endpoints = EndpointDataset(train)
    group_support = {name: int(len(indices)) for name, indices in endpoints.groups.items()}
    if args.steps_per_epoch is None:
        # Preserve the baseline's number of optimizer updates.  Unlike the
        # baseline, each E1 training item is one satellite endpoint rather
        # than a whole multi-satellite window, so its batch is intentionally
        # much larger than the evaluation window batch.
        args.steps_per_epoch = math.ceil(len(train) / args.eval_batch_size)
    sampler = FrequencyClassBalancedBatchSampler(endpoints.groups, args.batch_size, args.steps_per_epoch, args.seed)
    train_loader = DataLoader(endpoints, batch_sampler=sampler, num_workers=args.num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(
        val,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    model = make_model(args, train.raw.shape[-1], train.stats.shape[-1], device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    LOG.info(
        "architecture=conditional_heads encoder=%s device=%s params=%d steps_per_epoch=%d group_support=%s",
        args.encoder, device, parameter_count, args.steps_per_epoch, json.dumps(group_support),
    )
    if args.dry_run:
        raw, stats, _, is_l5 = next(iter(train_loader))
        logits = model(raw.unsqueeze(1).to(device), stats.unsqueeze(1).to(device), is_l5.unsqueeze(1).to(device))
        LOG.info("dry-run logits=%s group_quota=%d", tuple(logits.shape), args.batch_size // 4)
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = -float("inf")
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for raw, stats, labels, is_l5 in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            logits = model(raw.unsqueeze(1).to(device), stats.unsqueeze(1).to(device), is_l5.unsqueeze(1).to(device))[:, 0]
            loss = nn.functional.cross_entropy(logits, labels.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        val_metrics, val_by_band = evaluate(model, val_loader, device)
        LOG.info("epoch=%d train_loss=%.4f val=%s val_by_band=%s", epoch, total_loss / len(train_loader), json.dumps(val_metrics), json.dumps(val_by_band))
        if float(val_metrics["macro_f1"]) > best:
            best = float(val_metrics["macro_f1"])
            stale = 0
            save_checkpoint(
                checkpoint_path(args.output_dir, args.encoder), args, train, raw_names, stats_names,
                model, parameter_count, group_support, val_metrics, val_by_band,
            )
            (args.output_dir / "val_metrics_conditional_heads.json").write_text(json.dumps({
                **val_metrics, "metrics_by_band": val_by_band, "parameter_count": parameter_count,
                "raw_feature_names": raw_names, "stats_feature_names": stats_names,
                "train_group_support": group_support, "steps_per_epoch": args.steps_per_epoch,
            }, indent=2), encoding="utf-8")
        else:
            stale += 1
            if stale >= args.patience:
                LOG.info("early stopping")
                break
    LOG.info("complete; test was not read")


if __name__ == "__main__":
    main()

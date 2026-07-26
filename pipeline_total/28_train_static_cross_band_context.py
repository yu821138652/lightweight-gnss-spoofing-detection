"""Train E5a: formal signal detector with causal same-epoch band context.

The formal target-band-only binary labels, W5 windows, Session/time-block
protocol, and compact11 signal features are unchanged.  At every endpoint the
model receives only online-observable peer summaries from the same device and
source: L1/L5 visible counts, C/N0 and AGC means, and their causal W5 changes.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import SignalRawStatsCrossBandContext


def _load_module(name: str, filename: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


baseline = _load_module("_static_fusion_baseline_e5", "21_train_static_signal_fusion.py")
experts = _load_module("_static_band_experts_e5", "25_train_static_band_experts.py")
LOG = logging.getLogger(__name__)


class ContextDataset(baseline.FusionDataset):
    """Fusion tensors plus unscaled physical band and device sidecars."""

    def __init__(self, *args, is_l5_index: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        raw_path = Path(args[0])
        stats_path = Path(args[1])
        with np.load(stats_path, allow_pickle=False) as stats:
            self.is_l5 = torch.from_numpy(np.asarray(stats["x"][..., 0, is_l5_index]).copy()).ge(0.5)
        with np.load(raw_path, allow_pickle=False) as raw:
            device_id = np.asarray(raw["device_id"]).copy()
        if device_id.ndim == 1 and device_id.shape[0] == len(self):
            self.device_id = torch.from_numpy(device_id)
        elif device_id.shape == tuple(self.y.shape):
            self.device_id = torch.from_numpy(device_id)
        else:
            raise ValueError(f"Unexpected device_id shape {device_id.shape}; labels={tuple(self.y.shape)}")

    def __getitem__(self, index: int):
        raw, stats, mask, label = super().__getitem__(index)
        return raw, stats, mask, label, self.is_l5[index]


def load_split(
    split: str,
    raw_dir: Path,
    stats_dir: Path,
    raw_count: int,
    stats_count: int,
    raw_indices: list[int],
    stats_indices: list[int],
    is_l5_index: int,
) -> ContextDataset:
    return ContextDataset(
        raw_dir / f"{split}.npz",
        stats_dir / f"{split}.npz",
        raw_count,
        stats_count,
        raw_indices,
        stats_indices,
        is_l5_index=is_l5_index,
    )


def make_model(args: argparse.Namespace, raw_dim: int, stats_dim: int, cn0_index: int, agc_index: int, device: torch.device) -> SignalRawStatsCrossBandContext:
    return SignalRawStatsCrossBandContext(
        raw_input_dim=raw_dim,
        stats_input_dim=stats_dim,
        cn0_index=cn0_index,
        agc_index=agc_index,
        encoder=args.encoder,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[dict[str, float | int], dict[str, dict[str, float | int]]]:
    model.eval()
    labels_all: list[np.ndarray] = []
    predictions_all: list[np.ndarray] = []
    bands_all: list[np.ndarray] = []
    loss_total = 0.0
    count = 0
    for raw, stats, mask, labels, is_l5 in loader:
        raw = raw.to(device)
        stats = stats.to(device)
        mask = mask.to(device)
        labels = labels.to(device)
        is_l5 = is_l5.to(device)
        active = mask & labels.ne(baseline.IGNORE_INDEX)
        if not active.any():
            continue
        logits = model(raw, stats, mask, is_l5)
        selected_labels = labels[active]
        selected_logits = logits[active]
        loss_total += float(nn.functional.cross_entropy(selected_logits, selected_labels).item()) * selected_labels.numel()
        count += selected_labels.numel()
        labels_all.append(selected_labels.cpu().numpy())
        predictions_all.append(selected_logits.argmax(-1).cpu().numpy())
        bands_all.append(is_l5[active].cpu().numpy())
    if not count:
        raise ValueError("Evaluation split has no active labels")
    labels_array = np.concatenate(labels_all)
    predictions_array = np.concatenate(predictions_all)
    bands = np.concatenate(bands_all).astype(bool, copy=False)
    overall = experts.metrics(labels_array, predictions_array)
    overall["loss"] = loss_total / count
    return overall, {
        "L1": experts.metrics(labels_array[~bands], predictions_array[~bands]),
        "L5": experts.metrics(labels_array[bands], predictions_array[bands]),
    }


@torch.no_grad()
def write_group_metrics(data: ContextDataset, model: nn.Module, device: torch.device, batch_size: int, data_dir: Path, output_path: Path) -> None:
    prediction = np.zeros(tuple(data.y.shape), dtype=np.int64)
    model.eval()
    for start in range(0, len(data), batch_size):
        stop = min(start + batch_size, len(data))
        prediction[start:stop] = model(
            data.raw[start:stop].to(device),
            data.stats[start:stop].to(device),
            data.mask[start:stop].to(device),
            data.is_l5[start:stop].to(device),
        ).argmax(-1).cpu().numpy()
    mapping = json.loads((data_dir / "device_mapping.json").read_text(encoding="utf-8"))
    inverse_mapping = {int(value): str(name) for name, value in mapping.items()}
    labels = data.y.cpu().numpy()
    mask = (data.mask & data.y.ne(baseline.IGNORE_INDEX)).cpu().numpy()
    is_l5 = data.is_l5.cpu().numpy().astype(bool, copy=False)
    device_id = data.device_id.cpu().numpy()
    if device_id.ndim == 1:
        device_id = np.broadcast_to(device_id[:, None], labels.shape)
    rows: list[dict[str, object]] = []
    for value in sorted(np.unique(device_id[mask]).tolist()):
        for band, band_name in ((False, "L1"), (True, "L5")):
            selected = mask & (device_id == value) & (is_l5 == band)
            if not selected.any():
                continue
            row: dict[str, object] = {"device_id": int(value), "device_name": inverse_mapping.get(int(value), f"unknown_{value}"), "band": band_name}
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
    return output_dir / f"best_cross_band_context_signal_{encoder}_stats_mlp_fusion.pt"


def save_checkpoint(path: Path, args: argparse.Namespace, train: ContextDataset, raw_names: list[str], stats_names: list[str], model: nn.Module, parameter_count: int, class_weights: torch.Tensor, val_metrics: dict[str, float | int], val_by_band: dict[str, dict[str, float | int]]) -> None:
    torch.save({
        "model": f"cross_band_context_signal_{args.encoder}_stats_mlp_fusion",
        "architecture": "shared_fusion_causal_same_epoch_cross_band_context",
        "context_features": [
            "l1_visible_count", "l1_cn0_mean", "l1_agc_mean", "l1_cn0_delta_w5", "l1_agc_delta_w5",
            "l5_visible_count", "l5_cn0_mean", "l5_agc_mean", "l5_cn0_delta_w5", "l5_agc_delta_w5",
        ],
        "context_source": "active signals in same device/source/current endpoint; no labels, TOW, scenario, or future samples",
        "encoder": args.encoder,
        "raw_time_steps": int(train.raw.shape[-2]), "raw_input_dim": int(train.raw.shape[-1]),
        "raw_feature_names": raw_names, "raw_feature_set": args.raw_feature_set,
        "stats_input_dim": int(train.stats.shape[-1]), "stats_feature_names": stats_names, "stats_feature_set": args.stats_feature_set,
        "cn0_raw_index": raw_names.index("Cn0DbHz"), "agc_raw_index": raw_names.index("AgcDb"),
        "hidden_dim": args.hidden_dim, "dropout": args.dropout, "weight_decay": args.weight_decay,
        "parameter_count": parameter_count, "train_class_weights": class_weights.tolist(),
        "state_dict": model.state_dict(), "val_metrics": val_metrics, "val_metrics_by_band": val_by_band,
    }, path)


def validate_checkpoint(checkpoint: dict[str, Any], data: ContextDataset, raw_names: list[str], stats_names: list[str]) -> None:
    required = {"architecture", "encoder", "raw_time_steps", "raw_input_dim", "raw_feature_names", "stats_input_dim", "stats_feature_names", "cn0_raw_index", "agc_raw_index", "hidden_dim", "dropout", "state_dict"}
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise ValueError(f"Checkpoint is missing metadata: {missing}")
    if checkpoint["architecture"] != "shared_fusion_causal_same_epoch_cross_band_context":
        raise ValueError(f"Unsupported checkpoint architecture: {checkpoint['architecture']!r}")
    if tuple(data.raw.shape[-2:]) != (int(checkpoint["raw_time_steps"]), int(checkpoint["raw_input_dim"])) or data.stats.shape[-1] != int(checkpoint["stats_input_dim"]):
        raise ValueError("Checkpoint input dimensions differ from selected tensors")
    if checkpoint["raw_feature_names"] != raw_names or checkpoint["stats_feature_names"] != stats_names:
        raise ValueError("Checkpoint feature names/order differ from selected tensor features")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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
    parser.add_argument("--raw-feature-set", choices=("full",), default="full", help="E5a context needs physical C/N0 and AGC raw features")
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
    if "Cn0DbHz" not in raw_names or "AgcDb" not in raw_names:
        raise ValueError("E5a raw feature profile requires Cn0DbHz and AgcDb")
    stats_names = baseline.select_stats_feature_names(all_stats_names, args.stats_feature_set)
    stats_indices = baseline.feature_indices(all_stats_names, stats_names, "Stats")
    if "IsL5" not in all_stats_names:
        raise ValueError("E5a requires unscaled stats:IsL5 for physical peer grouping")
    raw_count = len(baseline.load_feature_names(raw_dir / "feature_names.json"))
    stats_count = len(all_stats_names)
    is_l5_index = all_stats_names.index("IsL5")
    train = load_split("train", raw_dir, stats_dir, raw_count, stats_count, raw_indices, stats_indices, is_l5_index)
    pin_memory = device.type == "cuda"

    if args.test_only:
        path = checkpoint_path(args.output_dir, args.encoder)
        if not path.is_file():
            raise FileNotFoundError(path)
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError(f"Invalid checkpoint: {path}")
        test = load_split("test", raw_dir, stats_dir, raw_count, stats_count, raw_indices, stats_indices, is_l5_index)
        baseline.validate_compatible(train, test, "test")
        validate_checkpoint(checkpoint, test, raw_names, stats_names)
        model_args = argparse.Namespace(encoder=checkpoint["encoder"], hidden_dim=checkpoint["hidden_dim"], dropout=checkpoint["dropout"])
        model = make_model(model_args, int(checkpoint["raw_input_dim"]), int(checkpoint["stats_input_dim"]), int(checkpoint["cn0_raw_index"]), int(checkpoint["agc_raw_index"]), device)
        model.load_state_dict(checkpoint["state_dict"])
        metrics, by_band = evaluate(model, DataLoader(test, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=pin_memory), device)
        metrics.update({"checkpoint": str(path), "parameter_count": int(checkpoint["parameter_count"]), "metrics_by_band": by_band})
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "test_metrics_cross_band_context.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        write_group_metrics(test, model, device, args.batch_size, args.data_dir, args.output_dir / "test_metrics_by_device_band.csv")
        LOG.info("locked checkpoint test=%s", json.dumps(metrics))
        return

    val = load_split("val", raw_dir, stats_dir, raw_count, stats_count, raw_indices, stats_indices, is_l5_index)
    baseline.validate_compatible(train, val, "val")
    class_weights = baseline.class_weights(train)
    model = make_model(args, train.raw.shape[-1], train.stats.shape[-1], raw_names.index("Cn0DbHz"), raw_names.index("AgcDb"), device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    LOG.info("architecture=cross_band_context encoder=%s device=%s params=%d context_dim=10", args.encoder, device, parameter_count)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=pin_memory)
    if args.dry_run:
        raw, stats, mask, _, is_l5 = next(iter(train_loader))
        logits = model(raw.to(device), stats.to(device), mask.to(device), is_l5.to(device))
        LOG.info("dry-run logits=%s active=%d", tuple(logits.shape), int(mask.sum()))
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = -float("inf")
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_total = 0.0
        count = 0
        for raw, stats, mask, labels, is_l5 in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            raw = raw.to(device); stats = stats.to(device); mask = mask.to(device); labels = labels.to(device); is_l5 = is_l5.to(device)
            active = mask & labels.ne(baseline.IGNORE_INDEX)
            logits = model(raw, stats, mask, is_l5)
            targets = labels[active]
            loss = criterion(logits[active], targets)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            loss_total += float(loss.item()) * targets.numel(); count += targets.numel()
        metrics, by_band = evaluate(model, val_loader, device)
        LOG.info("epoch=%d train_loss=%.4f val=%s val_by_band=%s", epoch, loss_total / count, json.dumps(metrics), json.dumps(by_band))
        if float(metrics["macro_f1"]) > best:
            best = float(metrics["macro_f1"]); stale = 0
            save_checkpoint(checkpoint_path(args.output_dir, args.encoder), args, train, raw_names, stats_names, model, parameter_count, class_weights, metrics, by_band)
            (args.output_dir / "val_metrics_cross_band_context.json").write_text(json.dumps({**metrics, "metrics_by_band": by_band, "parameter_count": parameter_count}, indent=2), encoding="utf-8")
        else:
            stale += 1
            if stale >= args.patience:
                LOG.info("early stopping")
                break
    LOG.info("complete; test was not read")


if __name__ == "__main__":
    main()

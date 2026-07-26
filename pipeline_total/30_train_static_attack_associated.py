"""Train E6: direct attack-associated anomaly detection on static signals.

This is a separate task from formal target-band spoof detection.  Every active
signal inside a reviewed static attack TOW interval is positive; every active
signal outside is negative.  Labels are derived at load time from the existing
tensor trace and YAML, without changing the central CSV or tensor ``y``.
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

from models import SignalRawStatsFusion


def _load_module(name: str, filename: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
    return module


baseline = _load_module("_static_fusion_baseline_e6", "21_train_static_signal_fusion.py")
e5b = _load_module("_context_attack_aux_e6", "29_train_static_context_attack_aux.py")
experts = e5b.e5.experts
LOG = logging.getLogger(__name__)


def checkpoint_path(output_dir: Path, encoder: str) -> Path:
    return output_dir / f"best_attack_associated_signal_{encoder}_stats_mlp_fusion.pt"


def make_model(args: argparse.Namespace, raw_dim: int, stats_dim: int, device: torch.device) -> SignalRawStatsFusion:
    return SignalRawStatsFusion(raw_input_dim=raw_dim, stats_input_dim=stats_dim, encoder=args.encoder, hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[dict[str, float | int], dict[str, dict[str, float | int]]]:
    model.eval(); labels_all: list[np.ndarray] = []; predictions_all: list[np.ndarray] = []; bands_all: list[np.ndarray] = []; loss_total = 0.0; count = 0
    for raw, stats, mask, _formal, is_l5, target in loader:
        raw = raw.to(device); stats = stats.to(device); mask = mask.to(device); target = target.to(device); is_l5 = is_l5.to(device)
        active = mask & target.ne(baseline.IGNORE_INDEX)
        if not active.any():
            continue
        logits = model(raw, stats); selected = target[active]
        loss_total += float(nn.functional.cross_entropy(logits[active], selected).item()) * selected.numel(); count += selected.numel()
        labels_all.append(selected.cpu().numpy()); predictions_all.append(logits[active].argmax(-1).cpu().numpy()); bands_all.append(is_l5[active].cpu().numpy())
    if not count:
        raise ValueError("Evaluation split has no active attack-associated labels")
    labels = np.concatenate(labels_all); predictions = np.concatenate(predictions_all); bands = np.concatenate(bands_all).astype(bool, copy=False)
    overall = experts.metrics(labels, predictions); overall["loss"] = loss_total / count
    return overall, {"L1": experts.metrics(labels[~bands], predictions[~bands]), "L5": experts.metrics(labels[bands], predictions[bands])}


@torch.no_grad()
def write_group_metrics(data: e5b.AttackAssociatedContextDataset, model: nn.Module, device: torch.device, batch_size: int, data_dir: Path, output_path: Path) -> None:
    prediction = np.zeros(tuple(data.attack_associated.shape), dtype=np.int64); model.eval()
    for start in range(0, len(data), batch_size):
        stop = min(start + batch_size, len(data)); prediction[start:stop] = model(data.raw[start:stop].to(device), data.stats[start:stop].to(device)).argmax(-1).cpu().numpy()
    mapping = json.loads((data_dir / "device_mapping.json").read_text(encoding="utf-8")); inverse = {int(value): str(name) for name, value in mapping.items()}
    labels = data.attack_associated.cpu().numpy(); mask = (data.mask & data.attack_associated.ne(baseline.IGNORE_INDEX)).cpu().numpy(); bands = data.is_l5.cpu().numpy().astype(bool, copy=False); device_ids = data.device_id.cpu().numpy()
    if device_ids.ndim == 1: device_ids = np.broadcast_to(device_ids[:, None], labels.shape)
    rows: list[dict[str, object]] = []
    for value in sorted(np.unique(device_ids[mask]).tolist()):
        for band, name in ((False, "L1"), (True, "L5")):
            selected = mask & (device_ids == value) & (bands == band)
            if selected.any():
                row: dict[str, object] = {"device_id": int(value), "device_name": inverse.get(int(value), f"unknown_{value}"), "band": name}; row.update(experts.metrics(labels[selected], prediction[selected])); rows.append(row)
    total: dict[str, object] = {"device_id": "ALL", "device_name": "ALL", "band": "ALL"}; total.update(experts.metrics(labels[mask], prediction[mask])); rows.append(total)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def save_checkpoint(path: Path, args: argparse.Namespace, train: e5b.AttackAssociatedContextDataset, raw_names: list[str], stats_names: list[str], model: nn.Module, parameter_count: int, weights: torch.Tensor, val_metrics: dict[str, float | int], by_band: dict[str, dict[str, float | int]]) -> None:
    torch.save({
        "model": f"attack_associated_signal_{args.encoder}_stats_mlp_fusion", "architecture": "raw_stats_fusion_attack_associated_binary",
        "label_semantics": "all active signals inside reviewed static attack interval are positive", "label_config": str(args.label_config.resolve()), "label_config_sha256": train.label_config_sha256,
        "encoder": args.encoder, "raw_time_steps": int(train.raw.shape[-2]), "raw_input_dim": int(train.raw.shape[-1]), "raw_feature_names": raw_names, "raw_feature_set": args.raw_feature_set,
        "stats_input_dim": int(train.stats.shape[-1]), "stats_feature_names": stats_names, "stats_feature_set": args.stats_feature_set,
        "hidden_dim": args.hidden_dim, "dropout": args.dropout, "weight_decay": args.weight_decay, "parameter_count": parameter_count,
        "class_weights": weights.tolist(), "train_attack_support": train.attack_support, "state_dict": model.state_dict(), "val_metrics": val_metrics, "val_metrics_by_band": by_band,
    }, path)


def validate_checkpoint(checkpoint: dict[str, Any], data: e5b.AttackAssociatedContextDataset, raw_names: list[str], stats_names: list[str]) -> None:
    required = {"architecture", "label_config_sha256", "encoder", "raw_time_steps", "raw_input_dim", "raw_feature_names", "stats_input_dim", "stats_feature_names", "hidden_dim", "dropout", "state_dict"}
    missing = sorted(required.difference(checkpoint))
    if missing: raise ValueError(f"Checkpoint missing metadata: {missing}")
    if checkpoint["architecture"] != "raw_stats_fusion_attack_associated_binary": raise ValueError("Unsupported checkpoint architecture")
    if checkpoint["label_config_sha256"] != data.label_config_sha256: raise ValueError("Label configuration changed since checkpoint selection")
    if tuple(data.raw.shape[-2:]) != (int(checkpoint["raw_time_steps"]), int(checkpoint["raw_input_dim"])) or data.stats.shape[-1] != int(checkpoint["stats_input_dim"]): raise ValueError("Checkpoint input dimensions differ")
    if checkpoint["raw_feature_names"] != raw_names or checkpoint["stats_feature_names"] != stats_names: raise ValueError("Checkpoint feature names/order differ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--label-config", type=Path, default=ROOT / "configs" / "preprocessing.yml")
    parser.add_argument("--encoder", choices=baseline.ENCODERS, default="tcn"); parser.add_argument("--epochs", type=int, default=30); parser.add_argument("--batch-size", type=int, default=256); parser.add_argument("--lr", type=float, default=1e-3); parser.add_argument("--weight-decay", type=float, default=1e-3); parser.add_argument("--hidden-dim", type=int, default=16); parser.add_argument("--dropout", type=float, default=0.1); parser.add_argument("--patience", type=int, default=6); parser.add_argument("--seed", type=int, default=2026); parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--raw-feature-set", choices=tuple(baseline.RAW_FEATURE_SETS), default="full"); parser.add_argument("--stats-feature-set", choices=baseline.STATS_FEATURE_SETS, default="cn0_agc_coverage_rx_time_std")
    mode = parser.add_mutually_exclusive_group(); mode.add_argument("--dry-run", action="store_true"); mode.add_argument("--test-only", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.hidden_dim < 1 or args.patience < 1 or args.num_workers < 0 or not 0.0 <= args.dropout < 1.0: parser.error("invalid training argument")
    return args


def main() -> None:
    args = parse_args(); baseline.seed_all(args.seed); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_dir, stats_dir, raw_indices, raw_names, all_stats_names = baseline.load_data_contract(args.data_dir, args.raw_feature_set); stats_names = baseline.select_stats_feature_names(all_stats_names, args.stats_feature_set); stats_indices = baseline.feature_indices(all_stats_names, stats_names, "Stats")
    if "IsL5" not in all_stats_names: raise ValueError("Full stats tensor lacks IsL5 sidecar")
    raw_count = len(baseline.load_feature_names(raw_dir / "feature_names.json")); stats_count = len(all_stats_names); is_l5_index = all_stats_names.index("IsL5")
    train = e5b.load_split("train", raw_dir, stats_dir, raw_count, stats_count, raw_indices, stats_indices, is_l5_index, args.data_dir, args.label_config); pin_memory = device.type == "cuda"
    if args.test_only:
        path = checkpoint_path(args.output_dir, args.encoder)
        if not path.is_file(): raise FileNotFoundError(path)
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        if not isinstance(checkpoint, dict): raise ValueError(f"Invalid checkpoint: {path}")
        test = e5b.load_split("test", raw_dir, stats_dir, raw_count, stats_count, raw_indices, stats_indices, is_l5_index, args.data_dir, args.label_config); baseline.validate_compatible(train, test, "test"); validate_checkpoint(checkpoint, test, raw_names, stats_names)
        model_args = argparse.Namespace(encoder=checkpoint["encoder"], hidden_dim=checkpoint["hidden_dim"], dropout=checkpoint["dropout"]); model = make_model(model_args, int(checkpoint["raw_input_dim"]), int(checkpoint["stats_input_dim"]), device); model.load_state_dict(checkpoint["state_dict"])
        metrics, by_band = evaluate(model, DataLoader(test, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=pin_memory), device); metrics.update({"checkpoint": str(path), "parameter_count": int(checkpoint["parameter_count"]), "metrics_by_band": by_band, "train_attack_support": checkpoint["train_attack_support"]})
        args.output_dir.mkdir(parents=True, exist_ok=True); (args.output_dir / "test_metrics_attack_associated.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8"); write_group_metrics(test, model, device, args.batch_size, args.data_dir, args.output_dir / "test_metrics_by_device_band.csv"); LOG.info("locked checkpoint test=%s", json.dumps(metrics)); return
    val = e5b.load_split("val", raw_dir, stats_dir, raw_count, stats_count, raw_indices, stats_indices, is_l5_index, args.data_dir, args.label_config); baseline.validate_compatible(train, val, "val")
    weights = e5b.class_weights(train.attack_associated); model = make_model(args, train.raw.shape[-1], train.stats.shape[-1], device); parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad); LOG.info("architecture=attack_associated encoder=%s device=%s params=%d train_support=%s", args.encoder, device, parameter_count, json.dumps(train.attack_support))
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin_memory); val_loader = DataLoader(val, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=pin_memory)
    if args.dry_run:
        raw, stats, _, _, _, _ = next(iter(train_loader)); LOG.info("dry-run logits=%s", tuple(model(raw.to(device), stats.to(device)).shape)); return
    args.output_dir.mkdir(parents=True, exist_ok=True); criterion = nn.CrossEntropyLoss(weight=weights.to(device)); optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay); best = -float("inf"); stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train(); loss_total = 0.0; count = 0
        for raw, stats, mask, _formal, _is_l5, target in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            raw = raw.to(device); stats = stats.to(device); mask = mask.to(device); target = target.to(device); active = mask & target.ne(baseline.IGNORE_INDEX); logits = model(raw, stats); selected = target[active]; loss = criterion(logits[active], selected)
            optimizer.zero_grad(); loss.backward(); optimizer.step(); loss_total += float(loss.item()) * selected.numel(); count += selected.numel()
        metrics, by_band = evaluate(model, val_loader, device); LOG.info("epoch=%d train_loss=%.4f val=%s val_by_band=%s", epoch, loss_total / count, json.dumps(metrics), json.dumps(by_band))
        if float(metrics["macro_f1"]) > best:
            best = float(metrics["macro_f1"]); stale = 0; save_checkpoint(checkpoint_path(args.output_dir, args.encoder), args, train, raw_names, stats_names, model, parameter_count, weights, metrics, by_band); (args.output_dir / "val_metrics_attack_associated.json").write_text(json.dumps({**metrics, "metrics_by_band": by_band, "parameter_count": parameter_count, "train_attack_support": train.attack_support}, indent=2), encoding="utf-8")
        else:
            stale += 1
            if stale >= args.patience: LOG.info("early stopping"); break
    LOG.info("complete; test was not read")


if __name__ == "__main__":
    main()

"""Train E5b: cross-band context plus a training-only attack-associated head.

The primary head predicts the unchanged formal target-band-only label.  The
auxiliary head predicts whether an endpoint belongs to a reviewed static attack
interval, regardless of frequency band.  Its label is never an input feature
and its output is not used for the deployable primary decision.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score, recall_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import SignalRawStatsCrossBandContextAuxiliary


def _load_module(name: str, filename: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


baseline = _load_module("_static_fusion_baseline_e5b", "21_train_static_signal_fusion.py")
e5 = _load_module("_static_cross_band_context_e5b", "28_train_static_cross_band_context.py")
LOG = logging.getLogger(__name__)


def reviewed_attack_intervals(path: Path) -> tuple[dict[tuple[str, str, str], list[tuple[float, float]]], str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    sessions = config.get("labeling", {}).get("session_spoofing_tow_intervals", {})
    if not isinstance(sessions, dict):
        raise ValueError("Invalid labeling.session_spoofing_tow_intervals")
    result: dict[tuple[str, str, str], list[tuple[float, float]]] = {}
    for environment, scenarios in sessions.items():
        if not isinstance(scenarios, dict):
            continue
        for scenario in ("st_L1", "st_L5", "st_L_15"):
            for session, entry in (scenarios.get(scenario, {}) or {}).items():
                if not isinstance(entry, dict) or str(entry.get("status")) != "reviewed":
                    continue
                intervals = [(float(start), float(end)) for start, end in (entry.get("intervals", []) or [])]
                if any(end < start for start, end in intervals):
                    raise ValueError(f"Invalid attack interval for {environment}/{scenario}/{session}")
                result[(str(environment), scenario, str(session))] = intervals
    if not result:
        raise ValueError("No reviewed static attack intervals in label config")
    return result, hashlib.sha256(path.read_bytes()).hexdigest()


class AttackAssociatedContextDataset(e5.ContextDataset):
    """E5 tensors plus an isolated training-only attack interval target."""

    def __init__(self, *args, data_dir: Path, label_config: Path, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        raw_path = Path(args[0])
        with np.load(raw_path, allow_pickle=False) as raw:
            recording_id = np.asarray(raw["recording_id"]).copy()
            endpoint_tow = np.asarray(raw["endpoint_tow"]).copy()
        if recording_id.shape != (len(self),) or endpoint_tow.shape != (len(self),):
            raise ValueError(f"Unexpected trace arrays in {raw_path}")
        trace = json.loads((data_dir / "window_trace_index.json").read_text(encoding="utf-8"))
        recordings = trace.get("recordings")
        if not isinstance(recordings, list):
            raise ValueError("window_trace_index.json has no recordings list")
        intervals, self.label_config_sha256 = reviewed_attack_intervals(label_config)
        target = np.full(tuple(self.y.shape), baseline.IGNORE_INDEX, dtype=np.int64)
        active = self.mask.cpu().numpy() & self.y.ne(baseline.IGNORE_INDEX).cpu().numpy()
        target[active] = 0
        for window, value in enumerate(recording_id.astype(np.int64, copy=False)):
            if value < 0 or value >= len(recordings):
                raise ValueError(f"Unknown recording_id {value}")
            record = recordings[int(value)]
            key = tuple(str(record[name]) for name in ("Environment", "Scenario", "Session"))
            tow = float(endpoint_tow[window])
            if np.isfinite(tow) and any(start <= tow <= end for start, end in intervals.get(key, [])):
                target[window, active[window]] = 1
        self.attack_associated = torch.from_numpy(target).long()
        support = torch.bincount(self.attack_associated[self.attack_associated.ne(baseline.IGNORE_INDEX)], minlength=2)
        if int(support.min().item()) == 0:
            raise ValueError(f"Attack-associated target needs both classes, found {support.tolist()}")
        self.attack_support = {"outside_attack": int(support[0]), "inside_attack": int(support[1])}

    def __getitem__(self, index: int):
        raw, stats, mask, label, is_l5 = super().__getitem__(index)
        return raw, stats, mask, label, is_l5, self.attack_associated[index]


def load_split(split: str, raw_dir: Path, stats_dir: Path, raw_count: int, stats_count: int, raw_indices: list[int], stats_indices: list[int], is_l5_index: int, data_dir: Path, label_config: Path) -> AttackAssociatedContextDataset:
    return AttackAssociatedContextDataset(
        raw_dir / f"{split}.npz", stats_dir / f"{split}.npz", raw_count, stats_count,
        raw_indices, stats_indices, is_l5_index=is_l5_index, data_dir=data_dir, label_config=label_config,
    )


def class_weights(target: torch.Tensor) -> torch.Tensor:
    values = target[target.ne(baseline.IGNORE_INDEX)]
    counts = torch.bincount(values, minlength=2).float()
    if int(counts.min().item()) == 0:
        raise ValueError(f"Both target classes are required, found {counts.tolist()}")
    return counts.sum() / (2.0 * counts)


def make_model(args: argparse.Namespace, raw_dim: int, stats_dim: int, cn0_index: int, agc_index: int, device: torch.device) -> SignalRawStatsCrossBandContextAuxiliary:
    return SignalRawStatsCrossBandContextAuxiliary(
        raw_input_dim=raw_dim, stats_input_dim=stats_dim, cn0_index=cn0_index, agc_index=agc_index,
        encoder=args.encoder, hidden_dim=args.hidden_dim, dropout=args.dropout,
    ).to(device)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[dict[str, Any], dict[str, dict[str, float | int]]]:
    model.eval()
    labels_all: list[np.ndarray] = []
    predictions_all: list[np.ndarray] = []
    bands_all: list[np.ndarray] = []
    attack_all: list[np.ndarray] = []
    attack_predictions: list[np.ndarray] = []
    primary_loss = 0.0
    auxiliary_loss = 0.0
    count = 0
    for raw, stats, mask, labels, is_l5, attack_target in loader:
        raw = raw.to(device); stats = stats.to(device); mask = mask.to(device)
        labels = labels.to(device); is_l5 = is_l5.to(device); attack_target = attack_target.to(device)
        active = mask & labels.ne(baseline.IGNORE_INDEX)
        if not active.any():
            continue
        primary_logits, attack_logits = model(raw, stats, mask, is_l5)
        selected_labels = labels[active]
        selected_attack = attack_target[active]
        primary_loss += float(nn.functional.cross_entropy(primary_logits[active], selected_labels).item()) * selected_labels.numel()
        auxiliary_loss += float(nn.functional.cross_entropy(attack_logits[active], selected_attack).item()) * selected_labels.numel()
        count += selected_labels.numel()
        labels_all.append(selected_labels.cpu().numpy()); predictions_all.append(primary_logits[active].argmax(-1).cpu().numpy())
        bands_all.append(is_l5[active].cpu().numpy()); attack_all.append(selected_attack.cpu().numpy()); attack_predictions.append(attack_logits[active].argmax(-1).cpu().numpy())
    if not count:
        raise ValueError("Evaluation split has no active labels")
    labels_array = np.concatenate(labels_all); predictions_array = np.concatenate(predictions_all)
    bands = np.concatenate(bands_all).astype(bool, copy=False)
    attack_array = np.concatenate(attack_all); attack_pred = np.concatenate(attack_predictions)
    result: dict[str, Any] = e5.experts.metrics(labels_array, predictions_array)
    result.update({
        "primary_loss": primary_loss / count,
        "attack_auxiliary_loss": auxiliary_loss / count,
        "attack_auxiliary_macro_f1": float(f1_score(attack_array, attack_pred, average="macro", zero_division=0)),
        "attack_auxiliary_recall": float(recall_score(attack_array, attack_pred, zero_division=0)),
        "attack_auxiliary_support": {"outside_attack": int((attack_array == 0).sum()), "inside_attack": int((attack_array == 1).sum())},
    })
    return result, {"L1": e5.experts.metrics(labels_array[~bands], predictions_array[~bands]), "L5": e5.experts.metrics(labels_array[bands], predictions_array[bands])}


@torch.no_grad()
def write_group_metrics(data: AttackAssociatedContextDataset, model: nn.Module, device: torch.device, batch_size: int, data_dir: Path, output_path: Path) -> None:
    prediction = np.zeros(tuple(data.y.shape), dtype=np.int64)
    model.eval()
    for start in range(0, len(data), batch_size):
        stop = min(start + batch_size, len(data))
        primary, _ = model(data.raw[start:stop].to(device), data.stats[start:stop].to(device), data.mask[start:stop].to(device), data.is_l5[start:stop].to(device))
        prediction[start:stop] = primary.argmax(-1).cpu().numpy()
    mapping = json.loads((data_dir / "device_mapping.json").read_text(encoding="utf-8"))
    inverse = {int(value): str(name) for name, value in mapping.items()}
    labels = data.y.cpu().numpy(); mask = (data.mask & data.y.ne(baseline.IGNORE_INDEX)).cpu().numpy(); bands = data.is_l5.cpu().numpy().astype(bool, copy=False)
    device_ids = data.device_id.cpu().numpy()
    if device_ids.ndim == 1:
        device_ids = np.broadcast_to(device_ids[:, None], labels.shape)
    rows: list[dict[str, object]] = []
    for value in sorted(np.unique(device_ids[mask]).tolist()):
        for band, name in ((False, "L1"), (True, "L5")):
            selected = mask & (device_ids == value) & (bands == band)
            if selected.any():
                row: dict[str, object] = {"device_id": int(value), "device_name": inverse.get(int(value), f"unknown_{value}"), "band": name}
                row.update(e5.experts.metrics(labels[selected], prediction[selected])); rows.append(row)
    total: dict[str, object] = {"device_id": "ALL", "device_name": "ALL", "band": "ALL"}; total.update(e5.experts.metrics(labels[mask], prediction[mask])); rows.append(total)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def checkpoint_path(output_dir: Path, encoder: str) -> Path:
    return output_dir / f"best_context_attack_aux_signal_{encoder}_stats_mlp_fusion.pt"


def save_checkpoint(path: Path, args: argparse.Namespace, train: AttackAssociatedContextDataset, raw_names: list[str], stats_names: list[str], model: nn.Module, parameter_count: int, primary_weights: torch.Tensor, auxiliary_weights: torch.Tensor, val_metrics: dict[str, Any], by_band: dict[str, dict[str, float | int]]) -> None:
    torch.save({
        "model": f"context_attack_aux_signal_{args.encoder}_stats_mlp_fusion",
        "architecture": "cross_band_context_formal_binary_plus_attack_associated_auxiliary",
        "primary_semantics": "formal_target_band_only_binary", "auxiliary_semantics": "reviewed_static_attack_interval_all_active_signals",
        "label_config": str(args.label_config.resolve()), "label_config_sha256": train.label_config_sha256,
        "aux_loss_weight": args.aux_loss_weight, "encoder": args.encoder,
        "raw_time_steps": int(train.raw.shape[-2]), "raw_input_dim": int(train.raw.shape[-1]), "raw_feature_names": raw_names, "raw_feature_set": args.raw_feature_set,
        "stats_input_dim": int(train.stats.shape[-1]), "stats_feature_names": stats_names, "stats_feature_set": args.stats_feature_set,
        "cn0_raw_index": raw_names.index("Cn0DbHz"), "agc_raw_index": raw_names.index("AgcDb"), "hidden_dim": args.hidden_dim, "dropout": args.dropout,
        "weight_decay": args.weight_decay, "parameter_count": parameter_count, "primary_class_weights": primary_weights.tolist(), "auxiliary_class_weights": auxiliary_weights.tolist(),
        "train_attack_auxiliary_support": train.attack_support, "state_dict": model.state_dict(), "val_metrics": val_metrics, "val_metrics_by_band": by_band,
    }, path)


def validate_checkpoint(checkpoint: dict[str, Any], data: AttackAssociatedContextDataset, raw_names: list[str], stats_names: list[str]) -> None:
    required = {"architecture", "encoder", "raw_time_steps", "raw_input_dim", "raw_feature_names", "stats_input_dim", "stats_feature_names", "cn0_raw_index", "agc_raw_index", "hidden_dim", "dropout", "label_config_sha256", "state_dict"}
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise ValueError(f"Checkpoint is missing metadata: {missing}")
    if checkpoint["architecture"] != "cross_band_context_formal_binary_plus_attack_associated_auxiliary":
        raise ValueError(f"Unsupported checkpoint architecture: {checkpoint['architecture']!r}")
    if checkpoint["label_config_sha256"] != data.label_config_sha256:
        raise ValueError("Label configuration changed since checkpoint selection")
    if tuple(data.raw.shape[-2:]) != (int(checkpoint["raw_time_steps"]), int(checkpoint["raw_input_dim"])) or data.stats.shape[-1] != int(checkpoint["stats_input_dim"]):
        raise ValueError("Checkpoint input dimensions differ from selected tensors")
    if checkpoint["raw_feature_names"] != raw_names or checkpoint["stats_feature_names"] != stats_names:
        raise ValueError("Checkpoint feature names/order differ from selected tensor features")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label-config", type=Path, default=ROOT / "configs" / "preprocessing.yml")
    parser.add_argument("--encoder", choices=baseline.ENCODERS, default="tcn"); parser.add_argument("--epochs", type=int, default=30); parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3); parser.add_argument("--weight-decay", type=float, default=1e-3); parser.add_argument("--hidden-dim", type=int, default=16); parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--aux-loss-weight", type=float, default=0.05); parser.add_argument("--patience", type=int, default=6); parser.add_argument("--seed", type=int, default=2026); parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--raw-feature-set", choices=("full",), default="full"); parser.add_argument("--stats-feature-set", choices=baseline.STATS_FEATURE_SETS, default="cn0_agc_coverage_rx_time_std")
    mode = parser.add_mutually_exclusive_group(); mode.add_argument("--dry-run", action="store_true"); mode.add_argument("--test-only", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.hidden_dim < 1 or args.patience < 1 or args.num_workers < 0 or args.aux_loss_weight < 0:
        parser.error("invalid positive training argument")
    if not 0.0 <= args.dropout < 1.0:
        parser.error("dropout must be in [0, 1)")
    return args


def main() -> None:
    args = parse_args(); baseline.seed_all(args.seed); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_dir, stats_dir, raw_indices, raw_names, all_stats_names = baseline.load_data_contract(args.data_dir, args.raw_feature_set)
    if "Cn0DbHz" not in raw_names or "AgcDb" not in raw_names or "IsL5" not in all_stats_names:
        raise ValueError("E5b requires raw Cn0DbHz/AgcDb and unscaled stats IsL5")
    stats_names = baseline.select_stats_feature_names(all_stats_names, args.stats_feature_set); stats_indices = baseline.feature_indices(all_stats_names, stats_names, "Stats")
    raw_count = len(baseline.load_feature_names(raw_dir / "feature_names.json")); stats_count = len(all_stats_names); is_l5_index = all_stats_names.index("IsL5")
    train = load_split("train", raw_dir, stats_dir, raw_count, stats_count, raw_indices, stats_indices, is_l5_index, args.data_dir, args.label_config); pin_memory = device.type == "cuda"
    if args.test_only:
        path = checkpoint_path(args.output_dir, args.encoder)
        if not path.is_file(): raise FileNotFoundError(path)
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        if not isinstance(checkpoint, dict): raise ValueError(f"Invalid checkpoint: {path}")
        test = load_split("test", raw_dir, stats_dir, raw_count, stats_count, raw_indices, stats_indices, is_l5_index, args.data_dir, args.label_config)
        baseline.validate_compatible(train, test, "test"); validate_checkpoint(checkpoint, test, raw_names, stats_names)
        model_args = argparse.Namespace(encoder=checkpoint["encoder"], hidden_dim=checkpoint["hidden_dim"], dropout=checkpoint["dropout"])
        model = make_model(model_args, int(checkpoint["raw_input_dim"]), int(checkpoint["stats_input_dim"]), int(checkpoint["cn0_raw_index"]), int(checkpoint["agc_raw_index"]), device); model.load_state_dict(checkpoint["state_dict"])
        metrics, by_band = evaluate(model, DataLoader(test, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=pin_memory), device)
        metrics.update({"checkpoint": str(path), "parameter_count": int(checkpoint["parameter_count"]), "metrics_by_band": by_band, "train_attack_auxiliary_support": checkpoint["train_attack_auxiliary_support"]})
        args.output_dir.mkdir(parents=True, exist_ok=True); (args.output_dir / "test_metrics_context_attack_aux.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        write_group_metrics(test, model, device, args.batch_size, args.data_dir, args.output_dir / "test_metrics_by_device_band.csv"); LOG.info("locked checkpoint test=%s", json.dumps(metrics)); return
    val = load_split("val", raw_dir, stats_dir, raw_count, stats_count, raw_indices, stats_indices, is_l5_index, args.data_dir, args.label_config); baseline.validate_compatible(train, val, "val")
    primary_weights = baseline.class_weights(train); auxiliary_weights = class_weights(train.attack_associated)
    model = make_model(args, train.raw.shape[-1], train.stats.shape[-1], raw_names.index("Cn0DbHz"), raw_names.index("AgcDb"), device); parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    LOG.info("architecture=context_attack_aux encoder=%s device=%s params=%d aux_weight=%.3f train_attack_support=%s", args.encoder, device, parameter_count, args.aux_loss_weight, json.dumps(train.attack_support))
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin_memory); val_loader = DataLoader(val, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=pin_memory)
    if args.dry_run:
        raw, stats, mask, _, is_l5, _ = next(iter(train_loader)); primary, attack = model(raw.to(device), stats.to(device), mask.to(device), is_l5.to(device)); LOG.info("dry-run primary=%s auxiliary=%s", tuple(primary.shape), tuple(attack.shape)); return
    args.output_dir.mkdir(parents=True, exist_ok=True); primary_criterion = nn.CrossEntropyLoss(weight=primary_weights.to(device)); auxiliary_criterion = nn.CrossEntropyLoss(weight=auxiliary_weights.to(device)); optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = -float("inf"); stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train(); total_loss = 0.0; count = 0
        for raw, stats, mask, labels, is_l5, attack_target in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            raw = raw.to(device); stats = stats.to(device); mask = mask.to(device); labels = labels.to(device); is_l5 = is_l5.to(device); attack_target = attack_target.to(device)
            active = mask & labels.ne(baseline.IGNORE_INDEX); primary, attack = model(raw, stats, mask, is_l5)
            targets = labels[active]; loss = primary_criterion(primary[active], targets) + args.aux_loss_weight * auxiliary_criterion(attack[active], attack_target[active])
            optimizer.zero_grad(); loss.backward(); optimizer.step(); total_loss += float(loss.item()) * targets.numel(); count += targets.numel()
        metrics, by_band = evaluate(model, val_loader, device); LOG.info("epoch=%d train_loss=%.4f val=%s val_by_band=%s", epoch, total_loss / count, json.dumps(metrics), json.dumps(by_band))
        if float(metrics["macro_f1"]) > best:
            best = float(metrics["macro_f1"]); stale = 0; save_checkpoint(checkpoint_path(args.output_dir, args.encoder), args, train, raw_names, stats_names, model, parameter_count, primary_weights, auxiliary_weights, metrics, by_band)
            (args.output_dir / "val_metrics_context_attack_aux.json").write_text(json.dumps({**metrics, "metrics_by_band": by_band, "parameter_count": parameter_count, "train_attack_auxiliary_support": train.attack_support}, indent=2), encoding="utf-8")
        else:
            stale += 1
            if stale >= args.patience: LOG.info("early stopping"); break
    LOG.info("complete; test was not read")


if __name__ == "__main__":
    main()

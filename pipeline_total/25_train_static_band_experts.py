"""Train frequency-routed satellite-level raw-plus-stats fusion experts.

This is a targeted follow-up to the static compact11 fusion baseline.  Each
signal is routed by the physically observed, unscaled ``IsL5`` tensor sidecar:
L1 and L5 use independent lightweight fusion classifiers.  Routing never uses
the scenario, attack interval, or label, so the same rule is available online.

The script intentionally reuses the tensor/feature contract from
``21_train_static_signal_fusion.py`` while writing a distinct checkpoint format
and output directory.  It is an experimental model, not a replacement for the
locked shared-baseline checkpoints.
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
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import SignalRawStatsFusion


def _load_baseline_module():
    """Reuse the established tensor validation and feature-profile selectors."""
    path = Path(__file__).with_name("21_train_static_signal_fusion.py")
    spec = importlib.util.spec_from_file_location("_static_fusion_baseline", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load baseline helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


baseline = _load_baseline_module()
IGNORE_INDEX = baseline.IGNORE_INDEX
ENCODERS = baseline.ENCODERS
RAW_FEATURE_SETS = baseline.RAW_FEATURE_SETS
STATS_FEATURE_SETS = baseline.STATS_FEATURE_SETS
LOG = logging.getLogger(__name__)


class BandExpertDataset(baseline.FusionDataset):
    """Fusion tensors plus the unscaled physical L5 routing sidecar."""

    def __init__(self, *args, is_l5_index: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        stats_path = Path(args[1])
        with np.load(stats_path, allow_pickle=False) as stats:
            full_stats = np.asarray(stats["x"])
            if full_stats.ndim != 4 or full_stats.shape[-2] != 1:
                raise ValueError(f"Expected stats x=[B,S,1,F], got {full_stats.shape} in {stats_path}")
            if is_l5_index >= full_stats.shape[-1]:
                raise ValueError(f"IsL5 index {is_l5_index} is outside {stats_path}")
            self.is_l5 = torch.from_numpy(full_stats[..., 0, is_l5_index].copy()).ge(0.5)
        self.device_id = self._load_device_ids(Path(args[0]))

    def _load_device_ids(self, raw_path: Path) -> torch.Tensor:
        with np.load(raw_path, allow_pickle=False) as raw:
            values = np.asarray(raw["device_id"]).copy()
        if values.ndim == 1 and values.shape[0] == len(self):
            return torch.from_numpy(values)
        if values.shape == tuple(self.y.shape):
            return torch.from_numpy(values)
        raise ValueError(f"Unexpected device_id shape {values.shape}; labels={tuple(self.y.shape)}")

    def __getitem__(self, index: int):
        raw, stats, mask, label = super().__getitem__(index)
        return raw, stats, mask, label, self.is_l5[index]


def load_split(
    split: str,
    raw_dir: Path,
    stats_dir: Path,
    raw_feature_count: int,
    stats_feature_count: int,
    raw_feature_indices: list[int],
    stats_feature_indices: list[int],
    is_l5_index: int,
) -> BandExpertDataset:
    return BandExpertDataset(
        raw_dir / f"{split}.npz",
        stats_dir / f"{split}.npz",
        raw_feature_count,
        stats_feature_count,
        raw_feature_indices,
        stats_feature_indices,
        is_l5_index=is_l5_index,
    )


def class_weights_by_band(data: BandExpertDataset, is_l5: bool) -> torch.Tensor:
    active = data.mask & data.y.ne(IGNORE_INDEX) & data.is_l5.eq(is_l5)
    labels = data.y[active]
    counts = torch.bincount(labels, minlength=2).float()
    if int(counts.min().item()) == 0:
        band = "L5" if is_l5 else "L1"
        raise ValueError(f"Training split has no {band} samples for one class: {counts.tolist()}")
    return counts.sum() / (2.0 * counts)


def band_support(data: BandExpertDataset, is_l5: bool) -> dict[str, int]:
    active = data.mask & data.y.ne(IGNORE_INDEX) & data.is_l5.eq(is_l5)
    labels = data.y[active]
    return {"negative": int(labels.eq(0).sum()), "positive": int(labels.eq(1).sum())}


def selected_logits(
    model: nn.Module,
    raw: torch.Tensor,
    stats: torch.Tensor,
    selected: torch.Tensor,
) -> torch.Tensor:
    """Run one expert on a non-rectangular subset of signal slots."""
    selected_raw = raw[selected]
    selected_stats = stats[selected]
    if not len(selected_raw):
        return raw.new_empty((0, 2))
    return model(selected_raw.unsqueeze(0), selected_stats.unsqueeze(0))[0]


def routed_logits(
    l1_model: nn.Module,
    l5_model: nn.Module,
    raw: torch.Tensor,
    stats: torch.Tensor,
    is_l5: torch.Tensor,
) -> torch.Tensor:
    output = raw.new_zeros((*is_l5.shape, 2))
    l5_mask = is_l5.bool()
    output[~l5_mask] = selected_logits(l1_model, raw, stats, ~l5_mask)
    output[l5_mask] = selected_logits(l5_model, raw, stats, l5_mask)
    return output


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    if not len(y_true):
        return {
            "samples": 0, "negative_support": 0, "positive_support": 0,
            "tn": 0, "fp": 0, "fn": 0, "tp": 0,
            "macro_f1": 0.0, "precision": 0.0, "recall": 0.0, "far": 0.0,
        }
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    negative_support = int((y_true == 0).sum())
    positive_support = int((y_true == 1).sum())
    return {
        "samples": int(len(y_true)),
        "negative_support": negative_support,
        "positive_support": positive_support,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "far": float(fp / negative_support) if negative_support else 0.0,
    }


@torch.no_grad()
def evaluate(
    l1_model: nn.Module,
    l5_model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, float | int], dict[str, dict[str, float | int]]]:
    l1_model.eval()
    l5_model.eval()
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
        active = mask & labels.ne(IGNORE_INDEX)
        if not active.any():
            continue
        logits = routed_logits(l1_model, l5_model, raw, stats, is_l5)
        active_logits = logits[active]
        active_labels = labels[active]
        losses.append(float(nn.functional.cross_entropy(active_logits, active_labels).item()) * active_labels.numel())
        all_labels.append(active_labels.cpu().numpy())
        all_predictions.append(active_logits.argmax(-1).cpu().numpy())
        all_bands.append(is_l5[active].cpu().numpy())
    if not all_labels:
        raise ValueError("Evaluation split has no active labels")
    labels = np.concatenate(all_labels)
    predictions = np.concatenate(all_predictions)
    bands = np.concatenate(all_bands)
    overall = metrics(labels, predictions)
    overall["loss"] = sum(losses) / len(labels)
    by_band = {
        "L1": metrics(labels[~bands], predictions[~bands]),
        "L5": metrics(labels[bands], predictions[bands]),
    }
    return overall, by_band


def make_model(args: argparse.Namespace, raw_input_dim: int, stats_input_dim: int, device: torch.device) -> SignalRawStatsFusion:
    return SignalRawStatsFusion(
        raw_input_dim=raw_input_dim,
        stats_input_dim=stats_input_dim,
        encoder=args.encoder,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)


def checkpoint_path(output_dir: Path, encoder: str) -> Path:
    return output_dir / f"best_band_experts_signal_{encoder}_stats_mlp_fusion.pt"


def save_checkpoint(
    path: Path,
    args: argparse.Namespace,
    train: BandExpertDataset,
    raw_feature_names: list[str],
    stats_feature_names: list[str],
    l1_model: nn.Module,
    l5_model: nn.Module,
    l1_count: int,
    l5_count: int,
    val_metrics: dict[str, float | int],
    val_by_band: dict[str, dict[str, float | int]],
) -> None:
    torch.save(
        {
            "model": f"band_experts_signal_{args.encoder}_stats_mlp_fusion",
            "architecture": "frequency_routed_independent_experts",
            "routing_source": "stats:IsL5 (unscaled physical sidecar)",
            "encoder": args.encoder,
            "raw_time_steps": int(train.raw.shape[-2]),
            "raw_input_dim": int(train.raw.shape[-1]),
            "raw_feature_names": raw_feature_names,
            "raw_feature_set": args.raw_feature_set,
            "stats_input_dim": int(train.stats.shape[-1]),
            "stats_feature_names": stats_feature_names,
            "stats_feature_set": args.stats_feature_set,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "weight_decay": args.weight_decay,
            "l1_parameter_count": l1_count,
            "l5_parameter_count": l5_count,
            "parameter_count": l1_count + l5_count,
            "l1_state_dict": l1_model.state_dict(),
            "l5_state_dict": l5_model.state_dict(),
            "val_metrics": val_metrics,
            "val_metrics_by_band": val_by_band,
        },
        path,
    )


def load_models_from_checkpoint(
    checkpoint: dict[str, Any],
    device: torch.device,
) -> tuple[SignalRawStatsFusion, SignalRawStatsFusion]:
    required = {
        "architecture", "encoder", "raw_input_dim", "stats_input_dim", "hidden_dim", "dropout",
        "l1_state_dict", "l5_state_dict",
    }
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise ValueError(f"Band-expert checkpoint is missing metadata: {missing}")
    if checkpoint["architecture"] != "frequency_routed_independent_experts":
        raise ValueError(f"Unsupported checkpoint architecture: {checkpoint['architecture']!r}")
    model_args = argparse.Namespace(
        encoder=str(checkpoint["encoder"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        dropout=float(checkpoint["dropout"]),
    )
    l1_model = make_model(model_args, int(checkpoint["raw_input_dim"]), int(checkpoint["stats_input_dim"]), device)
    l5_model = make_model(model_args, int(checkpoint["raw_input_dim"]), int(checkpoint["stats_input_dim"]), device)
    l1_model.load_state_dict(checkpoint["l1_state_dict"])
    l5_model.load_state_dict(checkpoint["l5_state_dict"])
    return l1_model, l5_model


def validate_checkpoint(
    checkpoint: dict[str, Any],
    data: BandExpertDataset,
    raw_feature_names: list[str],
    stats_feature_names: list[str],
) -> None:
    if tuple(data.raw.shape[-2:]) != (int(checkpoint["raw_time_steps"]), int(checkpoint["raw_input_dim"])):
        raise ValueError("Checkpoint raw input shape differs from selected tensor features")
    if data.stats.shape[-1] != int(checkpoint["stats_input_dim"]):
        raise ValueError("Checkpoint stats input dimension differs from selected tensor features")
    if checkpoint.get("raw_feature_names") != raw_feature_names:
        raise ValueError("Checkpoint raw feature names/order differ from selected tensor features")
    if checkpoint.get("stats_feature_names") != stats_feature_names:
        raise ValueError("Checkpoint stats feature names/order differ from selected tensor features")


def write_group_metrics(
    data: BandExpertDataset,
    l1_model: nn.Module,
    l5_model: nn.Module,
    device: torch.device,
    batch_size: int,
    data_dir: Path,
    output_path: Path,
) -> None:
    mapping_path = data_dir / "device_mapping.json"
    inverse_mapping: dict[int, str] = {}
    if mapping_path.is_file():
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        inverse_mapping = {int(value): str(key) for key, value in mapping.items()}

    prediction = np.zeros(tuple(data.y.shape), dtype=np.int64)
    l1_model.eval()
    l5_model.eval()
    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            stop = min(start + batch_size, len(data))
            logits = routed_logits(
                l1_model,
                l5_model,
                data.raw[start:stop].to(device),
                data.stats[start:stop].to(device),
                data.is_l5[start:stop].to(device),
            )
            prediction[start:stop] = logits.argmax(-1).cpu().numpy()

    mask = (data.mask & data.y.ne(IGNORE_INDEX)).cpu().numpy()
    labels = data.y.cpu().numpy()
    bands = data.is_l5.cpu().numpy()
    device_ids = data.device_id.cpu().numpy()
    if device_ids.ndim == 1:
        device_ids = np.broadcast_to(device_ids[:, None], labels.shape)
    rows: list[dict[str, object]] = []
    for device_id in sorted(np.unique(device_ids[mask]).tolist()):
        for is_l5, band in ((False, "L1"), (True, "L5")):
            selected = mask & (device_ids == device_id) & (bands == is_l5)
            if not selected.any():
                continue
            row: dict[str, object] = {
                "device_id": int(device_id),
                "device_name": inverse_mapping.get(int(device_id), f"unknown_{device_id}"),
                "band": band,
            }
            row.update(metrics(labels[selected], prediction[selected]))
            rows.append(row)
    total: dict[str, object] = {"device_id": "ALL", "device_name": "ALL", "band": "ALL"}
    total.update(metrics(labels[mask], prediction[mask]))
    rows.append(total)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--encoder", choices=ENCODERS, default="tcn")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--raw-feature-set", choices=tuple(RAW_FEATURE_SETS), default="full")
    parser.add_argument("--stats-feature-set", choices=STATS_FEATURE_SETS, default="cn0_agc_coverage_rx_time_std")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--test-only", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.hidden_dim < 1 or args.patience < 1:
        parser.error("epochs, batch-size, hidden-dim, and patience must be positive")
    if not 0.0 <= args.dropout < 1.0:
        parser.error("--dropout must be in [0, 1)")
    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    baseline.seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_dir, stats_dir, raw_indices, raw_names, all_stats_names = baseline.load_data_contract(
        args.data_dir, args.raw_feature_set
    )
    if "FreqBand" not in raw_names:
        raise ValueError("Band experts require the physical FreqBand raw feature in the tensor contract")
    raw_indices = [index for index, name in zip(raw_indices, raw_names) if name != "FreqBand"]
    raw_names = [name for name in raw_names if name != "FreqBand"]
    if not raw_names:
        raise ValueError("Band routing removed every raw feature")
    stats_names = baseline.select_stats_feature_names(all_stats_names, args.stats_feature_set)
    stats_indices = baseline.feature_indices(all_stats_names, stats_names, "Stats")
    if "IsL5" not in all_stats_names:
        raise ValueError("Stats tensors need the unscaled IsL5 sidecar for physical band routing")
    is_l5_index = all_stats_names.index("IsL5")
    raw_count = len(baseline.load_feature_names(raw_dir / "feature_names.json"))
    stats_count = len(all_stats_names)

    train = load_split("train", raw_dir, stats_dir, raw_count, stats_count, raw_indices, stats_indices, is_l5_index)
    if not len(train):
        raise ValueError("Training split contains no windows")
    train_support = {"L1": band_support(train, False), "L5": band_support(train, True)}
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
        l1_model, l5_model = load_models_from_checkpoint(checkpoint, device)
        test_loader = DataLoader(test, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=pin_memory)
        result, by_band = evaluate(l1_model, l5_model, test_loader, device)
        result.update({
            "checkpoint": str(path),
            "parameter_count": int(checkpoint["parameter_count"]),
            "l1_parameter_count": int(checkpoint["l1_parameter_count"]),
            "l5_parameter_count": int(checkpoint["l5_parameter_count"]),
            "raw_feature_names": raw_names,
            "stats_feature_names": stats_names,
            "metrics_by_band": by_band,
        })
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "test_metrics_band_experts.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        write_group_metrics(
            test, l1_model, l5_model, device, args.batch_size, args.data_dir,
            args.output_dir / "test_metrics_by_device_band.csv",
        )
        LOG.info("locked checkpoint test=%s", json.dumps(result))
        return

    val = load_split("val", raw_dir, stats_dir, raw_count, stats_count, raw_indices, stats_indices, is_l5_index)
    baseline.validate_compatible(train, val, "val")
    l1_model = make_model(args, train.raw.shape[-1], train.stats.shape[-1], device)
    l5_model = make_model(args, train.raw.shape[-1], train.stats.shape[-1], device)
    l1_count = sum(parameter.numel() for parameter in l1_model.parameters() if parameter.requires_grad)
    l5_count = sum(parameter.numel() for parameter in l5_model.parameters() if parameter.requires_grad)
    LOG.info(
        "architecture=band_experts encoder=%s device=%s L1_params=%d L5_params=%d total=%d train_support=%s",
        args.encoder, device, l1_count, l5_count, l1_count + l5_count, json.dumps(train_support),
    )
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=pin_memory)
    if args.dry_run:
        raw, stats, mask, labels, is_l5 = next(iter(train_loader))
        logits = routed_logits(l1_model, l5_model, raw.to(device), stats.to(device), is_l5.to(device))
        LOG.info("dry-run logits=%s active=%d", tuple(logits.shape), int(mask.sum()))
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    l1_weights = class_weights_by_band(train, False).to(device)
    l5_weights = class_weights_by_band(train, True).to(device)
    criterion_l1 = nn.CrossEntropyLoss(weight=l1_weights)
    criterion_l5 = nn.CrossEntropyLoss(weight=l5_weights)
    optimizer = torch.optim.AdamW(
        list(l1_model.parameters()) + list(l5_model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    best = -float("inf")
    stale = 0
    for epoch in range(1, args.epochs + 1):
        l1_model.train()
        l5_model.train()
        loss_total = 0.0
        batch_count = 0
        for raw, stats, mask, labels, is_l5 in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            raw = raw.to(device)
            stats = stats.to(device)
            mask = mask.to(device)
            labels = labels.to(device)
            is_l5 = is_l5.to(device)
            active = mask & labels.ne(IGNORE_INDEX)
            l1_active = active & ~is_l5
            l5_active = active & is_l5
            losses: list[torch.Tensor] = []
            if l1_active.any():
                losses.append(criterion_l1(selected_logits(l1_model, raw, stats, l1_active), labels[l1_active]))
            if l5_active.any():
                losses.append(criterion_l5(selected_logits(l5_model, raw, stats, l5_active), labels[l5_active]))
            if not losses:
                continue
            loss = torch.stack(losses).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_total += float(loss.item())
            batch_count += 1
        if not batch_count:
            raise ValueError("Training epoch has no active labels")
        val_metrics, val_by_band = evaluate(l1_model, l5_model, val_loader, device)
        LOG.info(
            "epoch=%d train_loss=%.4f val=%s val_by_band=%s",
            epoch, loss_total / batch_count, json.dumps(val_metrics), json.dumps(val_by_band),
        )
        score = float(val_metrics["macro_f1"])
        if score > best:
            best = score
            stale = 0
            save_checkpoint(
                checkpoint_path(args.output_dir, args.encoder), args, train, raw_names, stats_names,
                l1_model, l5_model, l1_count, l5_count, val_metrics, val_by_band,
            )
            (args.output_dir / "val_metrics_band_experts.json").write_text(
                json.dumps({
                    **val_metrics,
                    "metrics_by_band": val_by_band,
                    "parameter_count": l1_count + l5_count,
                    "l1_parameter_count": l1_count,
                    "l5_parameter_count": l5_count,
                    "raw_feature_names": raw_names,
                    "stats_feature_names": stats_names,
                    "train_support": train_support,
                }, indent=2),
                encoding="utf-8",
            )
        else:
            stale += 1
            if stale >= args.patience:
                LOG.info("early stopping")
                break
    LOG.info("complete; test was not read")


if __name__ == "__main__":
    main()

"""Train the static raw-sequence plus per-signal-statistics fusion baseline.

``--data-dir`` must contain the paired tensor families written by
``20_build_static_timeblock_tensors.py``::

    data_dir/raw/{train,val,test}.npz
    data_dir/stats/{train,val,test}.npz

Each family must also contain ``feature_names.json``.  The raw branch selects
five features by name, deliberately excluding the precomputed
``Cn0DbHz_dt`` and ``Cn0DbHz_std`` columns.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import SignalRawStatsFusion


logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
LOG = logging.getLogger(__name__)
IGNORE_INDEX = -100
RAW_FEATURE_NAMES = [
    "Cn0DbHz",
    "AgcDb",
    "ReceivedSvTimeUncertaintyNanos",
    "PseudorangeRateUncertaintyMetersPerSecond",
    "FreqBand",
]
REQUIRED_ARRAYS = {"x", "mask", "y", "is_dynamic", "device_id"}
ENCODERS = ("lstm", "tcn")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_feature_names(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing tensor feature-name metadata: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{path} must contain a non-empty JSON list of feature names")
    duplicates = sorted({name for name in value if value.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate feature names in {path}: {duplicates}")
    return value


def load_data_contract(data_dir: Path) -> tuple[Path, Path, list[int], list[str]]:
    raw_dir = data_dir / "raw"
    stats_dir = data_dir / "stats"
    raw_names = load_feature_names(raw_dir / "feature_names.json")
    stats_names = load_feature_names(stats_dir / "feature_names.json")
    missing = [name for name in RAW_FEATURE_NAMES if name not in raw_names]
    if missing:
        raise ValueError(
            f"Raw tensor metadata {raw_dir / 'feature_names.json'} is missing required features: {missing}"
        )
    raw_indices = [raw_names.index(name) for name in RAW_FEATURE_NAMES]
    return raw_dir, stats_dir, raw_indices, stats_names


class FusionDataset(Dataset):
    def __init__(
        self,
        raw_path: Path,
        stats_path: Path,
        raw_feature_count: int,
        stats_feature_count: int,
        raw_feature_indices: list[int],
    ) -> None:
        if not raw_path.is_file():
            raise FileNotFoundError(raw_path)
        if not stats_path.is_file():
            raise FileNotFoundError(stats_path)
        with np.load(raw_path, allow_pickle=False) as raw, np.load(stats_path, allow_pickle=False) as stats:
            raw_missing = REQUIRED_ARRAYS.difference(raw.files)
            stats_missing = REQUIRED_ARRAYS.difference(stats.files)
            if raw_missing:
                raise ValueError(f"{raw_path} is missing arrays: {sorted(raw_missing)}")
            if stats_missing:
                raise ValueError(f"{stats_path} is missing arrays: {sorted(stats_missing)}")

            raw_x = np.asarray(raw["x"])
            stats_x = np.asarray(stats["x"])
            if raw_x.ndim != 4:
                raise ValueError(f"Expected raw x=[B,S,T,F], got {raw_x.shape} in {raw_path}")
            if stats_x.ndim != 4 or stats_x.shape[-2] != 1:
                raise ValueError(f"Expected stats x=[B,S,1,F], got {stats_x.shape} in {stats_path}")
            if raw_x.shape[:2] != stats_x.shape[:2]:
                raise ValueError(f"Raw/stats window shapes differ: {raw_x.shape} vs {stats_x.shape}")
            if raw_x.shape[-1] != raw_feature_count:
                raise ValueError(
                    f"Raw x has {raw_x.shape[-1]} features but feature_names.json lists {raw_feature_count}"
                )
            if stats_x.shape[-1] != stats_feature_count:
                raise ValueError(
                    f"Stats x has {stats_x.shape[-1]} features but feature_names.json lists {stats_feature_count}"
                )

            for key in ("mask", "y", "is_dynamic", "device_id"):
                if not np.array_equal(raw[key], stats[key]):
                    raise ValueError(f"Raw/stats {key} mismatch for {raw_path.name}")
            mask = np.asarray(raw["mask"])
            labels = np.asarray(raw["y"])
            if mask.shape != raw_x.shape[:2] or labels.shape != raw_x.shape[:2]:
                raise ValueError(
                    f"mask/y shapes must match [B,S]={raw_x.shape[:2]}, got {mask.shape}/{labels.shape}"
                )
            active_labels = labels[mask.astype(bool)]
            unexpected = sorted(set(np.unique(active_labels).tolist()).difference({0, 1}))
            if unexpected:
                raise ValueError(f"Active labels must be binary in {raw_path}; found {unexpected}")

            self.raw = torch.from_numpy(raw_x[..., raw_feature_indices].copy()).float()
            self.stats = torch.from_numpy(stats_x.copy()).float()
            self.mask = torch.from_numpy(mask.copy()).bool()
            self.y = torch.from_numpy(labels.copy()).long()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int):
        return self.raw[index], self.stats[index], self.mask[index], self.y[index]


def load_split(
    split: str,
    raw_dir: Path,
    stats_dir: Path,
    raw_feature_count: int,
    stats_feature_count: int,
    raw_feature_indices: list[int],
) -> FusionDataset:
    return FusionDataset(
        raw_dir / f"{split}.npz",
        stats_dir / f"{split}.npz",
        raw_feature_count,
        stats_feature_count,
        raw_feature_indices,
    )


def validate_compatible(reference: FusionDataset, other: FusionDataset, split: str) -> None:
    if reference.raw.shape[2:] != other.raw.shape[2:]:
        raise ValueError(
            f"{split} raw input shape {tuple(other.raw.shape[2:])} differs from train "
            f"{tuple(reference.raw.shape[2:])}"
        )
    if reference.stats.shape[2:] != other.stats.shape[2:]:
        raise ValueError(
            f"{split} stats input shape {tuple(other.stats.shape[2:])} differs from train "
            f"{tuple(reference.stats.shape[2:])}"
        )


def class_weights(data: FusionDataset) -> torch.Tensor:
    active = data.mask & data.y.ne(IGNORE_INDEX)
    labels = data.y[active]
    if labels.numel() == 0:
        raise ValueError("Training split has no active labels; cannot compute class weights")
    counts = torch.bincount(labels, minlength=2).float()
    missing = torch.nonzero(counts.eq(0), as_tuple=False).flatten().tolist()
    if missing:
        raise ValueError(
            "Training split must contain both classes for class-weighted binary training; "
            f"counts={{0: {int(counts[0])}, 1: {int(counts[1])}}}, missing={missing}"
        )
    return counts.sum() / (2.0 * counts)


def valid(logits: torch.Tensor, mask: torch.Tensor, labels: torch.Tensor):
    active = mask & labels.ne(IGNORE_INDEX)
    return logits[active], labels[active]


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float | int]:
    model.eval()
    losses: list[float] = []
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for raw, stats, mask, labels in loader:
        raw = raw.to(device)
        stats = stats.to(device)
        mask = mask.to(device)
        labels = labels.to(device)
        logits, target = valid(model(raw, stats), mask, labels)
        if not target.numel():
            continue
        losses.append(float(criterion(logits, target).item()) * target.numel())
        predictions.append(logits.argmax(-1).cpu().numpy())
        targets.append(target.cpu().numpy())
    if not targets:
        raise ValueError("Evaluation split has no active labels")
    y_true = np.concatenate(targets)
    y_pred = np.concatenate(predictions)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    negative_support = int((y_true == 0).sum())
    positive_support = int((y_true == 1).sum())
    return {
        "loss": sum(losses) / len(y_true),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "far": float(fp / negative_support) if negative_support else 0.0,
        "samples": int(len(y_true)),
        "negative_support": negative_support,
        "positive_support": positive_support,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def make_model(
    raw_input_dim: int,
    stats_input_dim: int,
    encoder: str,
    hidden_dim: int,
    dropout: float,
    device: torch.device,
) -> SignalRawStatsFusion:
    return SignalRawStatsFusion(
        raw_input_dim=raw_input_dim,
        stats_input_dim=stats_input_dim,
        encoder=encoder,
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)


def resolve_checkpoint(args: argparse.Namespace) -> Path:
    if args.checkpoint is not None:
        return args.checkpoint
    if args.encoder is not None:
        return args.output_dir / f"best_signal_{args.encoder}_stats_mlp_fusion.pt"
    candidates = sorted(args.output_dir.glob("best_signal_*_stats_mlp_fusion.pt"))
    if len(candidates) != 1:
        raise ValueError(
            "--test-only needs --checkpoint, --encoder for checkpoint lookup, or exactly one "
            f"best_signal_*_stats_mlp_fusion.pt in {args.output_dir}; found {len(candidates)}"
        )
    return candidates[0]


def checkpoint_architecture(checkpoint: dict[str, Any]) -> dict[str, Any]:
    required = ["encoder", "raw_time_steps", "raw_feature_names", "stats_input_dim", "hidden_dim", "dropout"]
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise ValueError(f"Checkpoint is missing architecture metadata: {missing}")
    encoder = str(checkpoint["encoder"])
    if encoder not in ENCODERS:
        raise ValueError(f"Checkpoint has unsupported encoder={encoder!r}")
    raw_feature_names = checkpoint["raw_feature_names"]
    if not isinstance(raw_feature_names, list) or not all(isinstance(name, str) for name in raw_feature_names):
        raise ValueError("Checkpoint raw_feature_names must be a list of strings")
    architecture = {
        "encoder": encoder,
        "raw_time_steps": int(checkpoint["raw_time_steps"]),
        "raw_feature_names": raw_feature_names,
        "raw_input_dim": int(checkpoint.get("raw_input_dim", len(raw_feature_names))),
        "stats_input_dim": int(checkpoint["stats_input_dim"]),
        "hidden_dim": int(checkpoint["hidden_dim"]),
        "dropout": float(checkpoint["dropout"]),
        "stats_feature_names": checkpoint.get("stats_feature_names"),
    }
    if architecture["raw_time_steps"] < 2 or architecture["raw_input_dim"] <= 0:
        raise ValueError(f"Invalid raw checkpoint dimensions: {architecture}")
    if architecture["stats_input_dim"] <= 0 or architecture["hidden_dim"] <= 0:
        raise ValueError(f"Invalid checkpoint dimensions: {architecture}")
    if not 0.0 <= architecture["dropout"] < 1.0:
        raise ValueError(f"Invalid checkpoint dropout: {architecture['dropout']}")
    return architecture


def validate_checkpoint_inputs(
    architecture: dict[str, Any],
    data: FusionDataset,
    stats_feature_names: list[str],
    split: str,
) -> None:
    expected_raw_shape = (architecture["raw_time_steps"], architecture["raw_input_dim"])
    if tuple(data.raw.shape[-2:]) != expected_raw_shape:
        raise ValueError(
            f"Checkpoint expects raw [T,F]={expected_raw_shape}, but {split} has "
            f"{tuple(data.raw.shape[-2:])}"
        )
    if int(data.stats.shape[-1]) != architecture["stats_input_dim"]:
        raise ValueError(
            f"Checkpoint expects {architecture['stats_input_dim']} stats features, but {split} has "
            f"{data.stats.shape[-1]}"
        )
    if architecture["raw_feature_names"] != RAW_FEATURE_NAMES:
        raise ValueError(
            "Checkpoint raw feature order differs from the required data contract: "
            f"{architecture['raw_feature_names']} != {RAW_FEATURE_NAMES}"
        )
    checkpoint_stats_names = architecture["stats_feature_names"]
    if checkpoint_stats_names is not None and checkpoint_stats_names != stats_feature_names:
        raise ValueError("Checkpoint stats feature names/order differ from stats/feature_names.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Tensor root containing raw/ and stats/ subdirectories.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--encoder",
        choices=ENCODERS,
        default=None,
        help="Training encoder; in test-only mode it is used only to locate a checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint to evaluate with --test-only; architecture is read from the checkpoint.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-workers", type=int, default=0)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument(
        "--test-only",
        action="store_true",
        help="Evaluate a validation-selected checkpoint on test without training.",
    )
    args = parser.parse_args()
    if not args.test_only and args.encoder is None:
        parser.error("--encoder is required for training and --dry-run")
    if args.checkpoint is not None and not args.test_only:
        parser.error("--checkpoint is only valid with --test-only")
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.hidden_dim < 1:
        parser.error("--hidden-dim must be positive")
    if not 0.0 <= args.dropout < 1.0:
        parser.error("--dropout must be in [0, 1)")
    if args.patience < 1:
        parser.error("--patience must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_dir, stats_dir, raw_feature_indices, stats_feature_names = load_data_contract(args.data_dir)
    raw_feature_count = len(load_feature_names(raw_dir / "feature_names.json"))
    stats_feature_count = len(stats_feature_names)

    train = load_split(
        "train", raw_dir, stats_dir, raw_feature_count, stats_feature_count, raw_feature_indices
    )
    if len(train) == 0:
        raise ValueError("Training split contains no windows")
    # Training needs both classes to construct the weighted loss.  A locked
    # test-only evaluation does not, because it never updates parameters and
    # should remain usable even when a development split is single-class.
    weights = None if args.test_only else class_weights(train)
    pin_memory = device.type == "cuda"

    if args.test_only:
        checkpoint_path = resolve_checkpoint(args)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
            raise ValueError(f"Checkpoint {checkpoint_path} does not contain a model state_dict")
        architecture = checkpoint_architecture(checkpoint)
        validate_checkpoint_inputs(architecture, train, stats_feature_names, "train")
        test = load_split(
            "test", raw_dir, stats_dir, raw_feature_count, stats_feature_count, raw_feature_indices
        )
        validate_compatible(train, test, "test")
        validate_checkpoint_inputs(architecture, test, stats_feature_names, "test")
        model = make_model(
            architecture["raw_input_dim"],
            architecture["stats_input_dim"],
            architecture["encoder"],
            architecture["hidden_dim"],
            architecture["dropout"],
            device,
        )
        model.load_state_dict(checkpoint["state_dict"])
        parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        criterion = nn.CrossEntropyLoss()
        test_loader = DataLoader(
            test,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
        metrics = evaluate(model, test_loader, criterion, device)
        metrics["parameter_count"] = parameter_count
        metrics["checkpoint"] = str(checkpoint_path)
        metrics["encoder"] = architecture["encoder"]
        model_name = str(
            checkpoint.get("model", f"signal_{architecture['encoder']}_stats_mlp_fusion")
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = args.output_dir / f"test_metrics_{model_name}.json"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        LOG.info("locked checkpoint test=%s", json.dumps(metrics))
        return

    val = load_split(
        "val", raw_dir, stats_dir, raw_feature_count, stats_feature_count, raw_feature_indices
    )
    validate_compatible(train, val, "val")
    encoder = str(args.encoder)
    model = make_model(
        train.raw.shape[-1],
        train.stats.shape[-1],
        encoder,
        args.hidden_dim,
        args.dropout,
        device,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    LOG.info("encoder=%s device=%s params=%d", encoder, device, parameter_count)
    train_loader = DataLoader(
        train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    if args.dry_run:
        raw, stats, _, _ = next(iter(train_loader))
        logits = model(raw.to(device), stats.to(device))
        LOG.info("dry-run logits=%s", tuple(logits.shape))
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_name = f"signal_{encoder}_stats_mlp_fusion"
    checkpoint_path = args.output_dir / f"best_{model_name}.pt"
    assert weights is not None
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    best = -float("inf")
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        label_count = 0
        for raw, stats, mask, labels in tqdm(
            train_loader, desc=f"Epoch {epoch}/{args.epochs}"
        ):
            raw = raw.to(device)
            stats = stats.to(device)
            mask = mask.to(device)
            labels = labels.to(device)
            logits, target = valid(model(raw, stats), mask, labels)
            if not target.numel():
                continue
            loss = criterion(logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * target.numel()
            label_count += target.numel()
        if label_count == 0:
            raise ValueError("Training epoch has no active labels")
        metrics = evaluate(model, val_loader, criterion, device)
        LOG.info(
            "epoch=%d train_loss=%.4f val=%s",
            epoch,
            total_loss / label_count,
            json.dumps(metrics),
        )
        selection_score = metrics["macro_f1"] if metrics["positive_support"] > 0 else -metrics["loss"]
        if selection_score > best:
            best = float(selection_score)
            stale = 0
            torch.save(
                {
                    "model": model_name,
                    "encoder": encoder,
                    "raw_time_steps": int(train.raw.shape[-2]),
                    "raw_input_dim": int(train.raw.shape[-1]),
                    "raw_feature_indices": raw_feature_indices,
                    "raw_feature_names": RAW_FEATURE_NAMES,
                    "stats_input_dim": int(train.stats.shape[-1]),
                    "stats_feature_names": stats_feature_names,
                    "hidden_dim": args.hidden_dim,
                    "dropout": args.dropout,
                    "weight_decay": args.weight_decay,
                    "parameter_count": parameter_count,
                    "state_dict": model.state_dict(),
                    "val_metrics": metrics,
                },
                checkpoint_path,
            )
            (args.output_dir / f"val_metrics_{model_name}.json").write_text(
                json.dumps({**metrics, "parameter_count": parameter_count}, indent=2),
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

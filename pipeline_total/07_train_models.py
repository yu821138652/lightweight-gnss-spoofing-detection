"""Train project-owned lightweight GNSS per-signal baselines.

The input NPZ contains one device-local GNSS window per sample with shape
``[batch, signal, time, feature]``. Labels and masks are per signal slot.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import SignalGRU, SignalLSTM, SignalMLP, SignalTCN, SignalTransformerTiny


logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
LOGGER = logging.getLogger(__name__)
IGNORE_INDEX = -100


class GNSSWindowDataset(Dataset):
    """Load the device-local GNSS windows produced by step 05."""

    def __init__(self, npz_path: Path):
        with np.load(npz_path) as data:
            self.x = torch.from_numpy(data["x"]).float()
            self.mask = torch.from_numpy(data["mask"]).bool()
            self.y = torch.from_numpy(data["y"]).long()
        if self.x.ndim != 4 or self.mask.shape != self.y.shape or self.x.shape[:2] != self.y.shape:
            raise ValueError(f"Invalid tensor shapes in {npz_path}: x={self.x.shape}, mask={self.mask.shape}, y={self.y.shape}")

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int):
        return self.x[index], self.mask[index], self.y[index]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def masked_class_weights(dataset: GNSSWindowDataset) -> torch.Tensor | None:
    labels = dataset.y[dataset.mask]
    counts = torch.bincount(labels, minlength=2).float()
    if torch.any(counts == 0):
        return None
    return counts.sum() / (2.0 * counts)


def build_model(name: str, input_dim: int, time_steps: int, hidden_dim: int, dropout: float) -> nn.Module:
    if name == "signal_mlp":
        return SignalMLP(input_dim=input_dim, time_steps=time_steps, hidden_dim=hidden_dim, dropout=dropout)
    if name == "signal_gru":
        return SignalGRU(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
    if name == "signal_tcn":
        return SignalTCN(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
    if name == "signal_lstm":
        return SignalLSTM(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
    if name == "signal_transformer_tiny":
        return SignalTransformerTiny(input_dim=input_dim, time_steps=time_steps, hidden_dim=hidden_dim, dropout=dropout)
    raise ValueError(f"Unknown model: {name}")


def valid_logits_and_targets(logits: torch.Tensor, mask: torch.Tensor, targets: torch.Tensor):
    valid_mask = mask & (targets != IGNORE_INDEX)
    return logits[valid_mask], targets[valid_mask]


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    for x, mask, y in loader:
        x, mask, y = x.to(device), mask.to(device), y.to(device)
        logits, valid_targets = valid_logits_and_targets(model(x), mask, y)
        if valid_targets.numel() == 0:
            continue
        losses.append(float(criterion(logits, valid_targets).item()) * valid_targets.numel())
        predictions.append(logits.argmax(dim=-1).cpu().numpy())
        targets.append(valid_targets.cpu().numpy())

    if not targets:
        return {"loss": float("nan"), "macro_f1": 0.0, "precision": 0.0, "recall": 0.0, "far": 0.0, "samples": 0}

    y_true = np.concatenate(targets)
    y_pred = np.concatenate(predictions)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, _, _ = matrix.ravel()
    return {
        "loss": float(sum(losses) / len(y_true)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "far": float(fp / (fp + tn)) if fp + tn else 0.0,
        "samples": int(len(y_true)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "output" / "tensors_mixed", help="Directory containing train.npz and val.npz.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "output" / "training", help="Where checkpoints and validation metrics are written.")
    parser.add_argument(
        "--model",
        choices=["signal_mlp", "signal_gru", "signal_tcn", "signal_lstm", "signal_transformer_tiny"],
        default="signal_mlp",
        help="Lightweight per-signal baseline to train.",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Maximum training epochs. Validation macro-F1 controls early stopping.")
    parser.add_argument("--batch-size", type=int, default=256, help="Number of GNSS windows per batch.")
    parser.add_argument("--lr", type=float, default=1e-3, help="AdamW learning rate.")
    parser.add_argument("--hidden-dim", type=int, default=32, help="Hidden width for the lightweight baseline.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout applied before the classifier.")
    parser.add_argument("--patience", type=int, default=6, help="Stop after this many non-improving validation epochs.")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed for reproducible baseline runs.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers; keep 0 on Windows unless verified stable.")
    parser.add_argument("--evaluate-test", action="store_true", help="Evaluate test.npz after training. Use only after architecture and hyperparameters are locked.")
    parser.add_argument("--test-only", action="store_true", help="Evaluate an existing best checkpoint without training or validation-driven model selection.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Checkpoint used by --test-only. Defaults to <output-dir>/best_<model>.pt.")
    parser.add_argument("--dry-run", action="store_true", help="Load train/val data and run one forward pass without optimization, checkpointing, or test access.")
    args = parser.parse_args()
    if args.test_only and (args.dry_run or args.evaluate_test):
        parser.error("--test-only cannot be combined with --dry-run or --evaluate-test")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_path = args.data_dir / "train.npz"
    val_path = args.data_dir / "val.npz"
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError("train.npz and val.npz must exist before training.")

    train_data = GNSSWindowDataset(train_path)
    val_data = GNSSWindowDataset(val_path)
    input_dim = int(train_data.x.shape[-1])
    time_steps = int(train_data.x.shape[-2])
    if tuple(val_data.x.shape[-2:]) != (time_steps, input_dim):
        raise ValueError("train and validation tensors use different time/feature dimensions.")

    model = build_model(args.model, input_dim, time_steps, args.hidden_dim, args.dropout).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    LOGGER.info("Model=%s device=%s parameters=%d input=[signal, %d, %d]", args.model, device, parameter_count, time_steps, input_dim)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)

    if args.dry_run:
        x, mask, y = next(iter(train_loader))
        with torch.no_grad():
            logits = model(x.to(device))
            _, valid_targets = valid_logits_and_targets(logits, mask.to(device), y.to(device))
        LOGGER.info("Dry run passed: logits=%s valid_labels=%d; no weights updated and no test data read.", tuple(logits.shape), valid_targets.numel())
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    class_weights = masked_class_weights(train_data)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)

    if args.test_only:
        checkpoint_path = args.checkpoint or args.output_dir / f"best_{args.model}.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint.get("model") != args.model:
            raise ValueError(f"Checkpoint model is {checkpoint.get('model')}, not {args.model}")
        model = build_model(
            checkpoint["model"],
            int(checkpoint["input_dim"]),
            int(checkpoint["time_steps"]),
            int(checkpoint["hidden_dim"]),
            float(checkpoint["dropout"]),
        ).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        test_path = args.data_dir / "test.npz"
        if not test_path.exists():
            raise FileNotFoundError(f"Missing test tensor: {test_path}")
        test_data = GNSSWindowDataset(test_path)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)
        test_metrics = evaluate(model, test_loader, criterion, device)
        (args.output_dir / f"test_metrics_{args.model}.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
        LOGGER.info("Test-only metrics=%s", json.dumps(test_metrics, ensure_ascii=False))
        return

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_f1 = -1.0
    stale_epochs = 0
    checkpoint_path = args.output_dir / f"best_{args.model}.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_labels = 0
        for x, mask, y in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            x, mask, y = x.to(device), mask.to(device), y.to(device)
            logits, valid_targets = valid_logits_and_targets(model(x), mask, y)
            if valid_targets.numel() == 0:
                continue
            loss = criterion(logits, valid_targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * valid_targets.numel()
            total_labels += valid_targets.numel()

        val_metrics = evaluate(model, val_loader, criterion, device)
        LOGGER.info("Epoch %d: train_loss=%.4f val=%s", epoch, total_loss / max(total_labels, 1), json.dumps(val_metrics, ensure_ascii=False))
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            stale_epochs = 0
            torch.save({
                "model": args.model,
                "input_dim": input_dim,
                "time_steps": time_steps,
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
                "state_dict": model.state_dict(),
                "val_metrics": val_metrics,
            }, checkpoint_path)
            (args.output_dir / f"val_metrics_{args.model}.json").write_text(json.dumps(val_metrics, indent=2), encoding="utf-8")
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                LOGGER.info("Early stopping after %d epochs without validation macro-F1 improvement.", args.patience)
                break

    if args.evaluate_test:
        test_path = args.data_dir / "test.npz"
        if not test_path.exists():
            raise FileNotFoundError(f"Missing test tensor: {test_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        test_data = GNSSWindowDataset(test_path)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)
        test_metrics = evaluate(model, test_loader, criterion, device)
        (args.output_dir / f"test_metrics_{args.model}.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
        LOGGER.info("Final test metrics=%s", json.dumps(test_metrics, ensure_ascii=False))
    else:
        LOGGER.info("Training complete. Test data was not read; use --evaluate-test only after locking the baseline configuration.")


if __name__ == "__main__":
    main()

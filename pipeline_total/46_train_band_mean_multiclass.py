"""Train a four-way band-mean window scene classifier.

Consumes the tensors written by ``45_build_band_mean_window_tensors.py``::

    data_dir/{train,val,test}.npz   # x=[B, T, F], y in {0,1,2,3}, single_band_mask, ...

The four scene classes are ``0=normal``, ``1=L1``, ``2=L5``, ``3=L1+L5``.  A
window whose endpoint epoch observed only one physical band is flagged
``single_band_mask=True``; such windows are excluded from the loss and from
every reported metric, because a single band cannot express the four-way scene.
They are deliberately kept in the tensors for a later, separate path.

Model selection uses macro-F1 over the *usable* validation windows.  Because the
static leave-one-recording-out folds each hold out a single scene type, an
individual fold's test split cannot contain all four classes; the validation
split (time blocks drawn from the development recordings) does, so it is the
primary four-way separability signal.  A full 4x4 test confusion matrix is
obtained by aggregating predictions across all folds, not from one fold.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import BandMeanWindowClassifier


logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
LOG = logging.getLogger(__name__)
NUM_CLASSES = 4
CLASS_NAMES = {0: "normal", 1: "L1", 2: "L5", 3: "L1+L5"}
ENCODERS = ("lstm", "gru", "tcn")
REQUIRED_ARRAYS = {"x", "y", "single_band_mask", "device_id"}
# The two presence flags encode which bands were observed; they are the backbone
# of the four-way task and are never droppable in an ablation.
PROTECTED_FEATURES = ("L1Present", "L5Present")


def load_feature_names(data_dir: Path) -> list[str]:
    path = data_dir / "feature_names.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing tensor feature-name metadata: {path}")
    names = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
        raise ValueError(f"{path} must be a JSON list of feature names")
    return names


def load_tensor_metadata(data_dir: Path) -> dict:
    path = data_dir / "tensor_metadata.json"
    if not path.is_file():
        return {}
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return metadata


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_contract(data_dir: Path) -> dict[str, str]:
    """Fingerprint tensor artifacts that affect training or exported predictions."""
    required = (
        "feature_names.json",
        "tensor_metadata.json",
        "train.npz",
        "val.npz",
        "test.npz",
    )
    missing = [name for name in required if not (data_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Tensor directory {data_dir} is missing contract artifacts: {missing}"
        )
    optional = (
        "scaler.json", "normal_reference.json", "device_mapping.json", "source_mapping.json"
    )
    names = [*required, *[name for name in optional if (data_dir / name).is_file()]]
    return {
        name: sha256_file(data_dir / name)
        for name in names
    }


def causal_contract(value: object) -> dict:
    """Normalize legacy and expanded metadata for compatibility checks."""
    if not isinstance(value, dict):
        return {"mode": "none"}
    mode = str(value.get("mode", "none"))
    if mode == "none":
        return {"mode": "none"}
    return value


def validate_tensor_contract(
    checkpoint_contract: object, data_dir: Path, predict_split: str
) -> None:
    """Bind new checkpoints to their metadata/scaler and selected split bytes."""
    if checkpoint_contract is None:
        return  # Legacy checkpoints predate artifact fingerprints.
    if not isinstance(checkpoint_contract, dict):
        raise ValueError("Checkpoint tensor_contract must be a JSON-like object")
    required = [
        "feature_names.json",
        "tensor_metadata.json",
        f"{predict_split}.npz",
    ]
    if "scaler.json" in checkpoint_contract:
        required.append("scaler.json")
    for name in ("normal_reference.json", "device_mapping.json", "source_mapping.json"):
        if name in checkpoint_contract:
            required.append(name)
    for name in required:
        expected = checkpoint_contract.get(name)
        path = data_dir / name
        if not isinstance(expected, str) or not path.is_file():
            raise ValueError(
                f"Checkpoint tensor contract requires {name}, but it is missing or invalid"
            )
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"Checkpoint/tensor artifact mismatch for {name}: "
                f"checkpoint={expected}, tensor={actual}"
            )


def resolve_feature_selection(
    all_names: list[str], drop_features: list[str] | None
) -> tuple[list[int], list[str]]:
    """Resolve a --drop-features ablation into kept column indices and names.

    Presence flags are protected. Dropping happens by feature name (e.g. AgcDb
    drops both L1_AgcDb and L5_AgcDb) so the same flag works across bands.
    """
    if not drop_features:
        return list(range(len(all_names))), list(all_names)
    drop_tokens = {token.strip() for token in drop_features if token.strip()}
    kept_indices: list[int] = []
    kept_names: list[str] = []
    for index, name in enumerate(all_names):
        protected = name in PROTECTED_FEATURES
        # Match either the full feature name or its band-agnostic suffix.
        suffix = name.split("_", 1)[1] if "_" in name else name
        dropped = (not protected) and (name in drop_tokens or suffix in drop_tokens)
        if not dropped:
            kept_indices.append(index)
            kept_names.append(name)
    if len(kept_names) == len(all_names):
        raise ValueError(f"--drop-features {sorted(drop_tokens)} matched no columns in {all_names}")
    if not any(name not in PROTECTED_FEATURES for name in kept_names):
        raise ValueError("Feature ablation cannot drop every continuous feature")
    return kept_indices, kept_names


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_usable_split(
    path: Path, feature_indices: list[int] | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load one split and drop single-band windows (excluded everywhere)."""
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        missing = REQUIRED_ARRAYS.difference(data.files)
        if missing:
            raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
        x = np.asarray(data["x"], dtype=np.float32)
        y = np.asarray(data["y"], dtype=np.int64)
        single_band = np.asarray(data["single_band_mask"], dtype=bool)
    if x.ndim != 3:
        raise ValueError(f"Expected x=[B,T,F], got {x.shape} in {path}")
    if y.shape != x.shape[:1] or single_band.shape != x.shape[:1]:
        raise ValueError(f"y/single_band_mask must have shape [B] in {path}")
    if feature_indices is not None:
        x = x[:, :, feature_indices]
    usable = ~single_band
    x_usable = x[usable]
    y_usable = y[usable]
    unexpected = sorted(set(np.unique(y_usable).tolist()).difference(range(NUM_CLASSES)))
    if unexpected:
        raise ValueError(f"Usable labels must be in 0..{NUM_CLASSES - 1}; found {unexpected} in {path}")
    return torch.from_numpy(x_usable), torch.from_numpy(y_usable)


def load_usable_split_traced(
    path: Path, feature_indices: list[int] | None = None
) -> dict[str, np.ndarray]:
    """Load one split, drop single-band windows, and keep endpoint trace fields.

    Used by ``--test-only`` to export per-window predictions with enough
    provenance (recording, device, endpoint TOW) for a cross-fold aggregation
    into one true 4x4 confusion matrix.
    """
    if not path.is_file():
        raise FileNotFoundError(path)
    trace_fields = (
        "recording_id", "source_id", "device_id", "endpoint_tow", "window_time_nanos",
    )
    with np.load(path, allow_pickle=False) as data:
        missing = REQUIRED_ARRAYS.difference(data.files)
        if missing:
            raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
        x = np.asarray(data["x"], dtype=np.float32)
        y = np.asarray(data["y"], dtype=np.int64)
        single_band = np.asarray(data["single_band_mask"], dtype=bool)
        trace = {
            name: np.asarray(data[name]) if name in data.files else None
            for name in trace_fields
        }
    if x.ndim != 3:
        raise ValueError(f"Expected x=[B,T,F], got {x.shape} in {path}")
    if y.shape != x.shape[:1] or single_band.shape != x.shape[:1]:
        raise ValueError(f"y/single_band_mask must have shape [B] in {path}")
    if feature_indices is not None:
        x = x[:, :, feature_indices]
    usable = ~single_band
    result: dict[str, np.ndarray] = {"x": x[usable], "y": y[usable]}
    for name, values in trace.items():
        if values is not None and values.shape[:1] == x.shape[:1]:
            result[name] = values[usable]
    return result


def class_weights(y: torch.Tensor, multipliers: list[float] | None = None) -> torch.Tensor:
    counts = torch.bincount(y, minlength=NUM_CLASSES).float()
    present = counts > 0
    if int(present.sum()) < 2:
        raise ValueError(f"Training split has fewer than two classes; counts={counts.tolist()}")
    # Inverse-frequency weights; absent classes get zero weight so they do not
    # distort the normalization (they contribute no samples anyway).
    weights = torch.zeros(NUM_CLASSES)
    weights[present] = counts.sum() / (present.sum().float() * counts[present])
    # Optional per-class multipliers let us push a recall-starved class (e.g. L1,
    # L1+L5) harder without rebuilding tensors.  A multiplier of 1.0 is a no-op.
    if multipliers is not None:
        if len(multipliers) != NUM_CLASSES:
            raise ValueError(f"--class-weight-mult needs {NUM_CLASSES} values, got {len(multipliers)}")
        weights = weights * torch.tensor(multipliers, dtype=weights.dtype)
    return weights


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> dict:
    model.eval()
    losses: list[float] = []
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        losses.append(float(criterion(logits, y).item()) * y.numel())
        preds.append(logits.argmax(-1).cpu().numpy())
        targets.append(y.cpu().numpy())
    if not targets:
        raise ValueError("Evaluation split has no usable windows")
    y_true = np.concatenate(targets)
    y_pred = np.concatenate(preds)
    labels = list(range(NUM_CLASSES))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = {
        CLASS_NAMES[c]: {
            "precision": float(precision[c]),
            "recall": float(recall[c]),
            "f1": float(f1[c]),
            "support": int(support[c]),
        }
        for c in labels
    }
    return {
        "loss": sum(losses) / len(y_true),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "accuracy": float((y_true == y_pred).mean()),
        "samples": int(len(y_true)),
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_labels": [CLASS_NAMES[c] for c in labels],
        "per_class": per_class,
        "class_support": {CLASS_NAMES[c]: int(support[c]) for c in labels},
    }


def make_loader(x: torch.Tensor, y: torch.Tensor, batch_size: int, shuffle: bool, pin: bool) -> DataLoader:
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle, pin_memory=pin)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--encoder", choices=ENCODERS, default="tcn")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--class-weight-mult", type=float, nargs=NUM_CLASSES, default=None,
        metavar=("NORMAL", "L1", "L5", "L1L5"),
        help="Per-class multipliers on the inverse-frequency loss weights (order: normal L1 L5 L1+L5).",
    )
    parser.add_argument(
        "--drop-features", type=str, nargs="+", default=None,
        help=(
            "Feature-ablation: drop these features by name or band-agnostic suffix "
            "(e.g. AgcDb drops both L1_AgcDb and L5_AgcDb). Presence flags are protected."
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument(
        "--test-only", action="store_true",
        help="Load the fold checkpoint, predict its test split, and export per-window predictions.",
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=None,
        help="With --test-only, the checkpoint to evaluate; defaults to output-dir/best_band_mean_window_<encoder>.pt.",
    )
    parser.add_argument(
        "--predict-split", choices=("train", "val", "test"), default="test",
        help="With --test-only, tensor split to predict; default preserves test export behavior.",
    )
    parser.add_argument(
        "--fold", type=int, default=None,
        help="Optional fold tag recorded in the exported test-prediction CSV.",
    )
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if not 0.0 <= args.dropout < 1.0:
        parser.error("--dropout must be in [0, 1)")
    if args.patience < 1:
        parser.error("--patience must be positive")
    return args


@torch.no_grad()
def run_test_only(args: argparse.Namespace, device: torch.device) -> None:
    """Load a fold checkpoint, predict its test split, and export per-window rows."""
    checkpoint_path = args.checkpoint or (
        args.output_dir / f"best_band_mean_window_{args.encoder}.pt"
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    # ``weights_only`` was added after the PyTorch version bundled with some
    # project environments.  Keep the newer explicit form where available, but
    # retain compatibility with the older unpickler API used for this project.
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint {checkpoint_path} has no state_dict")
    checkpoint_encoder = str(checkpoint.get("encoder", args.encoder))
    if checkpoint_encoder != args.encoder:
        raise ValueError(
            f"Checkpoint encoder={checkpoint_encoder} does not match --encoder={args.encoder}"
        )
    tensor_metadata = load_tensor_metadata(args.data_dir)
    tensor_scaler_mode = str(tensor_metadata.get("scaler_mode", "legacy_per_device"))
    checkpoint_scaler_mode = str(checkpoint.get("scaler_mode", "legacy_per_device"))
    if tensor_scaler_mode != checkpoint_scaler_mode:
        raise ValueError(
            "Checkpoint/tensor scaler mismatch: "
            f"checkpoint={checkpoint_scaler_mode}, tensor={tensor_scaler_mode}"
        )
    tensor_causal = causal_contract(tensor_metadata.get("causal_baseline"))
    checkpoint_causal = causal_contract(checkpoint.get("causal_baseline"))
    if tensor_causal != checkpoint_causal:
        raise ValueError(
            "Checkpoint/tensor causal-baseline mismatch: "
            f"checkpoint={checkpoint_causal}, tensor={tensor_causal}"
        )
    time_steps = int(checkpoint["time_steps"])
    feature_dim = int(checkpoint["input_dim"])
    model = BandMeanWindowClassifier(
        input_dim=feature_dim,
        time_steps=time_steps,
        encoder=str(checkpoint["encoder"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        dropout=float(checkpoint["dropout"]),
        num_classes=int(checkpoint.get("num_classes", NUM_CLASSES)),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # Reapply the exact feature ablation the checkpoint was trained with, so the
    # test tensor's feature axis matches the model's expected input dimension.
    all_names = load_feature_names(args.data_dir)
    feature_indices, selected_names = resolve_feature_selection(
        all_names, checkpoint.get("drop_features")
    )
    checkpoint_names = checkpoint.get("feature_names")
    if checkpoint_names is not None and checkpoint_names != selected_names:
        raise ValueError(
            "Checkpoint/tensor feature order mismatch: "
            f"checkpoint={checkpoint_names}, tensor={selected_names}"
        )
    split_name = args.predict_split
    validate_tensor_contract(checkpoint.get("tensor_contract"), args.data_dir, split_name)
    predicted = load_usable_split_traced(args.data_dir / f"{split_name}.npz", feature_indices)
    if predicted["x"].shape[1:] != (time_steps, feature_dim):
        raise ValueError(
            f"Checkpoint expects [T,F]=({time_steps},{feature_dim}), "
            f"{split_name} has {tuple(predicted['x'].shape[1:])}"
        )
    if len(predicted["y"]) == 0:
        raise ValueError(f"{split_name} split has no usable windows")
    x = torch.from_numpy(predicted["x"]).to(device)
    logits = model(x)
    probabilities = torch.softmax(logits, dim=1).cpu().numpy()
    preds = logits.argmax(-1).cpu().numpy()
    y_true = predicted["y"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / f"{split_name}_predictions_band_mean_window_{args.encoder}.csv"
    rows = len(y_true)
    mapping_path = args.data_dir / "device_mapping.json"
    device_names: dict[int, str] = {}
    if mapping_path.is_file():
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        device_names = {int(value): str(name) for name, value in mapping.items()}
    fields = [
        "fold", "recording_id", "source_id", "device_id", "device_name",
        "window_time_nanos", "endpoint_tow",
        "true_class", "pred_class", *[f"prob_{CLASS_NAMES[c]}" for c in range(NUM_CLASSES)],
    ]
    with predictions_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i in range(rows):
            device_id = int(predicted["device_id"][i]) if "device_id" in predicted else -1
            row = {
                "fold": args.fold if args.fold is not None else "",
                "recording_id": int(predicted["recording_id"][i]) if "recording_id" in predicted else "",
                "source_id": int(predicted["source_id"][i]) if "source_id" in predicted else "",
                "device_id": device_id if device_id >= 0 else "",
                "device_name": device_names.get(device_id, ""),
                "window_time_nanos": int(predicted["window_time_nanos"][i]) if "window_time_nanos" in predicted else "",
                "endpoint_tow": float(predicted["endpoint_tow"][i]) if "endpoint_tow" in predicted else "",
                "true_class": int(y_true[i]),
                "pred_class": int(preds[i]),
            }
            row.update({f"prob_{CLASS_NAMES[c]}": float(probabilities[i, c]) for c in range(NUM_CLASSES)})
            writer.writerow(row)
    present = sorted(set(y_true.tolist()))
    LOG.info(
        "test-only split=%s fold=%s usable=%d classes_present=%s exported=%s",
        split_name, args.fold, rows, [CLASS_NAMES[c] for c in present], predictions_path,
    )


def main() -> None:
    args = parse_args()
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.test_only:
        run_test_only(args, device)
        return

    all_names = load_feature_names(args.data_dir)
    tensor_metadata = load_tensor_metadata(args.data_dir)
    scaler_mode = str(tensor_metadata.get("scaler_mode", "legacy_per_device"))
    causal_baseline = tensor_metadata.get("causal_baseline", {"mode": "none"})
    normal_reference = tensor_metadata.get("normal_reference", {"mode": "none"})
    artifact_contract = tensor_contract(args.data_dir)
    feature_indices, feature_names = resolve_feature_selection(all_names, args.drop_features)
    if args.drop_features:
        LOG.info("feature ablation: dropped=%s kept=%s", sorted(args.drop_features), feature_names)
    train_x, train_y = load_usable_split(args.data_dir / "train.npz", feature_indices)
    val_x, val_y = load_usable_split(args.data_dir / "val.npz", feature_indices)
    if train_x.shape[1:] != val_x.shape[1:]:
        raise ValueError(f"train/val feature shapes differ: {tuple(train_x.shape[1:])} vs {tuple(val_x.shape[1:])}")
    time_steps, feature_dim = int(train_x.shape[1]), int(train_x.shape[2])

    train_counts = torch.bincount(train_y, minlength=NUM_CLASSES).tolist()
    val_counts = torch.bincount(val_y, minlength=NUM_CLASSES).tolist()
    LOG.info(
        "encoder=%s device=%s T=%d F=%d train_usable=%d val_usable=%d",
        args.encoder, device, time_steps, feature_dim, len(train_y), len(val_y),
    )
    LOG.info("train class counts=%s val class counts=%s", train_counts, val_counts)

    model = BandMeanWindowClassifier(
        input_dim=feature_dim,
        time_steps=time_steps,
        encoder=args.encoder,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        num_classes=NUM_CLASSES,
    ).to(device)
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    LOG.info("params=%d", parameter_count)

    pin = device.type == "cuda"
    train_loader = make_loader(train_x, train_y, args.batch_size, True, pin)
    val_loader = make_loader(val_x, val_y, args.batch_size, False, pin)

    if args.dry_run:
        x, _ = next(iter(train_loader))
        logits = model(x.to(device))
        LOG.info("dry-run logits=%s", tuple(logits.shape))
        return

    weights = class_weights(train_y, args.class_weight_mult).to(device)
    LOG.info("class weights=%s", weights.tolist())
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_name = f"band_mean_window_{args.encoder}"
    checkpoint_path = args.output_dir / f"best_{model_name}.pt"
    best = -float("inf")
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        count = 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * y.numel()
            count += int(y.numel())
        metrics = evaluate(model, val_loader, criterion, device)
        LOG.info(
            "epoch=%d train_loss=%.4f val_macro_f1=%.4f val_acc=%.4f",
            epoch, total_loss / count, metrics["macro_f1"], metrics["accuracy"],
        )
        if metrics["macro_f1"] > best:
            best = float(metrics["macro_f1"])
            stale = 0
            torch.save(
                {
                    "model": model_name,
                    "encoder": args.encoder,
                    "time_steps": time_steps,
                    "input_dim": feature_dim,
                    "hidden_dim": args.hidden_dim,
                    "dropout": args.dropout,
                    "num_classes": NUM_CLASSES,
                    "parameter_count": parameter_count,
                    "state_dict": model.state_dict(),
                    "val_metrics": metrics,
                    "train_class_counts": train_counts,
                    "feature_names": feature_names,
                    "drop_features": sorted(args.drop_features) if args.drop_features else [],
                    "scaler_mode": scaler_mode,
                    "data_scope": tensor_metadata.get("data_scope"),
                    "causal_baseline": causal_baseline,
                    "normal_reference": normal_reference,
                    "tensor_contract": artifact_contract,
                },
                checkpoint_path,
            )
            (args.output_dir / f"val_metrics_{model_name}.json").write_text(
                json.dumps({**metrics, "parameter_count": parameter_count}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        else:
            stale += 1
            if stale >= args.patience:
                LOG.info("early stopping")
                break

    LOG.info("best val macro_f1=%.4f checkpoint=%s", best, checkpoint_path)
    best_metrics = json.loads((args.output_dir / f"val_metrics_{model_name}.json").read_text(encoding="utf-8"))
    LOG.info("best val confusion matrix (rows=true, cols=pred) labels=%s", best_metrics["confusion_matrix_labels"])
    for label, row in zip(best_metrics["confusion_matrix_labels"], best_metrics["confusion_matrix"]):
        LOG.info("  %-7s %s", label, row)


if __name__ == "__main__":
    main()

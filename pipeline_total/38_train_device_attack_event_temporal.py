"""Train a causal temporal device attack-event detector.

Unlike the independent-window baseline, this model receives each device stream
in chronological order within one recording.  A unidirectional GRU therefore
can use preceding observations to retain an alarm through a sustained attack,
but cannot access future windows.  The validation split selects the alarm
threshold and optional post-trigger hold duration; outer-test evaluation is
available only through an explicit ``--test-only`` run.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=4, help="number of complete device streams")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    parser.add_argument("--hold-windows", type=int, nargs="+", default=[0, 10, 30])
    parser.add_argument("--max-val-far", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if min(args.hidden_dim, args.layers, args.epochs, args.batch_size, args.patience) < 1 or args.num_workers < 0:
        parser.error("training dimensions and counts must be positive")
    if not 0.0 <= args.dropout < 1.0 or not 0.0 <= args.max_val_far <= 1.0:
        parser.error("dropout and max-val-far must be in [0, 1]")
    if any(not 0.0 < value < 1.0 for value in args.thresholds):
        parser.error("thresholds must be strictly between zero and one")
    if any(value < 0 for value in args.hold_windows):
        parser.error("hold-windows must be non-negative")
    if args.test_only and args.checkpoint is None:
        parser.error("--test-only requires --checkpoint")
    return args


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EventTensor:
    REQUIRED = {"x", "y_event", "device_id", "recording_id", "source_id", "endpoint_tow", "endpoint_utc_millis"}

    def __init__(self, path: Path) -> None:
        with np.load(path, allow_pickle=False) as data:
            if missing := self.REQUIRED.difference(data.files):
                raise ValueError(f"{path} is missing {sorted(missing)}")
            self.x = data["x"].astype(np.float32)
            self.y = data["y_event"].astype(np.int64)
            self.device_id = data["device_id"].astype(np.int64)
            self.recording_id = data["recording_id"].astype(np.int64)
            self.source_id = data["source_id"].astype(np.int64)
            self.endpoint_tow = data["endpoint_tow"].astype(np.float64)
            self.endpoint_time = data["endpoint_utc_millis"].astype(np.float64)
        if self.x.ndim != 2 or len(self.x) != len(self.y):
            raise ValueError(f"Invalid device tensor shapes in {path}")
        if not np.isfinite(self.endpoint_time).all():
            raise ValueError(f"Non-finite endpoint time in {path}")

    def groups(self) -> list[np.ndarray]:
        grouped: dict[tuple[int, int], list[int]] = defaultdict(list)
        for index, key in enumerate(zip(self.recording_id, self.source_id)):
            grouped[(int(key[0]), int(key[1]))].append(index)
        result: list[np.ndarray] = []
        for key in sorted(grouped):
            indices = np.asarray(grouped[key], dtype=np.int64)
            ordered = indices[np.argsort(self.endpoint_time[indices], kind="mergesort")]
            if len(ordered) and np.any(np.diff(self.endpoint_time[ordered]) < 0):
                raise RuntimeError(f"Failed to sort sequence {key}")
            result.append(ordered)
        return result


class SequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, data: EventTensor) -> None:
        self.data = data
        self.sequences = data.groups()
        if not self.sequences:
            raise ValueError("No device sequences available")

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = self.sequences[index]
        return (
            torch.from_numpy(self.data.x[indices]),
            torch.from_numpy(self.data.y[indices]),
            torch.from_numpy(indices),
        )


def collate_sequences(batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([len(item[0]) for item in batch], dtype=torch.long)
    max_length = int(lengths.max())
    feature_count = batch[0][0].shape[1]
    x = torch.zeros((len(batch), max_length, feature_count), dtype=torch.float32)
    y = torch.full((len(batch), max_length), -100, dtype=torch.long)
    indices = torch.full((len(batch), max_length), -1, dtype=torch.long)
    for row, (sequence_x, sequence_y, sequence_indices) in enumerate(batch):
        length = len(sequence_x)
        x[row, :length] = sequence_x
        y[row, :length] = sequence_y
        indices[row, :length] = sequence_indices
    return x, y, indices, lengths


class CausalEventGRU(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, 2))

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.gru(packed)
        output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=x.shape[1])
        return self.head(output)


def class_weights(data: EventTensor) -> torch.Tensor:
    counts = torch.bincount(torch.from_numpy(data.y), minlength=2).float()
    if torch.any(counts == 0):
        raise ValueError(f"Train split lacks an event class: {counts.tolist()}")
    return counts.sum() / (2.0 * counts)


def window_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "samples": int(len(y_true)), "negative_support": int((y_true == 0).sum()), "positive_support": int((y_true == 1).sum()),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "far": float(fp / (fp + tn)) if fp + tn else 0.0,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def contiguous_runs(labels: np.ndarray, value: int) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, label in enumerate(labels):
        if label == value and start is None:
            start = index
        elif label != value and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(labels)))
    return runs


def event_metrics(data: EventTensor, predicted: np.ndarray) -> dict[str, float | int | None]:
    event_count = detected = predicted_event_count = false_alarm_runs = 0
    coverage: list[float] = []
    delays: list[int] = []
    for indices in data.groups():
        labels = data.y[indices]
        predictions = predicted[indices]
        for start, end in contiguous_runs(labels, 1):
            event_count += 1
            local = np.flatnonzero(predictions[start:end] == 1)
            if len(local):
                detected += 1
                delays.append(int(local[0]))
            coverage.append(float(predictions[start:end].mean()))
        for start, end in contiguous_runs(predictions, 1):
            predicted_event_count += 1
            if not np.any(labels[start:end] == 1):
                false_alarm_runs += 1
    return {
        "true_events": event_count,
        "detected_events": detected,
        "event_recall": float(detected / event_count) if event_count else 0.0,
        "mean_attack_coverage": float(np.mean(coverage)) if coverage else 0.0,
        "median_detection_delay_windows": float(np.median(delays)) if delays else None,
        "predicted_event_runs": predicted_event_count,
        "false_alarm_runs": false_alarm_runs,
    }


def apply_alarm_policy(probability: np.ndarray, data: EventTensor, threshold: float, hold_windows: int) -> np.ndarray:
    """Apply a causal threshold/hold alarm policy independently to every stream."""
    predicted = np.zeros(len(probability), dtype=np.int64)
    for indices in data.groups():
        remaining = 0
        for index in indices:
            if probability[index] >= threshold:
                predicted[index] = 1
                remaining = hold_windows
            elif remaining > 0:
                predicted[index] = 1
                remaining -= 1
    return predicted


@torch.no_grad()
def probabilities(model: nn.Module, dataset: SequenceDataset, device: torch.device, batch_size: int, num_workers: int) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_sequences)
    result = np.empty(len(dataset.data.y), dtype=np.float64)
    model.eval()
    for x, _, indices, lengths in loader:
        logits = model(x.to(device), lengths)
        batch_probability = logits.softmax(dim=-1)[..., 1].cpu().numpy()
        for row, length in enumerate(lengths.tolist()):
            result[indices[row, :length].numpy()] = batch_probability[row, :length]
    return result


def evaluate_probability(data: EventTensor, probability: np.ndarray, threshold: float, hold_windows: int, device_names: dict[int, str]) -> dict[str, Any]:
    predicted = apply_alarm_policy(probability, data, threshold, hold_windows)
    result: dict[str, Any] = {
        "alarm_policy": {"threshold": threshold, "hold_windows": hold_windows},
        "overall": window_metrics(data.y, predicted),
        "event": event_metrics(data, predicted),
        "by_device": {},
    }
    for device_id in np.unique(data.device_id):
        selected = data.device_id == device_id
        result["by_device"][device_names.get(int(device_id), str(device_id))] = window_metrics(data.y[selected], predicted[selected])
    return result


def select_alarm_policy(data: EventTensor, probability: np.ndarray, thresholds: list[float], holds: list[int], max_far: float, device_names: dict[int, str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidates = [evaluate_probability(data, probability, threshold, hold, device_names) for threshold in thresholds for hold in holds]
    eligible = [candidate for candidate in candidates if candidate["overall"]["far"] <= max_far]
    if eligible:
        selected = max(eligible, key=lambda candidate: (candidate["overall"]["macro_f1"], candidate["event"]["event_recall"], -candidate["overall"]["far"]))
    else:
        selected = min(candidates, key=lambda candidate: (candidate["overall"]["far"], -candidate["overall"]["macro_f1"]))
    return selected, candidates


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def train(args: argparse.Namespace, metadata: dict[str, Any], device_names: dict[int, str], device: torch.device) -> None:
    train_data = EventTensor(args.data_dir / "train.npz")
    val_data = EventTensor(args.data_dir / "val.npz")
    train_dataset, val_dataset = SequenceDataset(train_data), SequenceDataset(val_data)
    if args.dry_run:
        print(json.dumps({"device": str(device), "train_streams": len(train_dataset), "val_streams": len(val_dataset), "features": int(train_data.x.shape[1])}))
        return
    model = CausalEventGRU(train_data.x.shape[1], args.hidden_dim, args.layers, args.dropout).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_data).to(device), ignore_index=-100)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_sequences)
    best_score, best_state, stale = float("-inf"), None, 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_windows = 0
        for x, y, _, lengths in loader:
            logits = model(x.to(device), lengths)
            loss = criterion(logits.reshape(-1, 2), y.to(device).reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            valid = int((y != -100).sum())
            total_loss += float(loss.item()) * valid
            total_windows += valid
        val_probability = probabilities(model, val_dataset, device, args.batch_size, args.num_workers)
        argmax_metrics = evaluate_probability(val_data, val_probability, 0.5, 0, device_names)["overall"]
        row = {"epoch": epoch, "train_loss": total_loss / total_windows, **argmax_metrics}
        history.append(row)
        score = float(argmax_metrics["macro_f1"])
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("No checkpoint selected")
    model.load_state_dict(best_state)
    val_probability = probabilities(model, val_dataset, device, args.batch_size, args.num_workers)
    selected, candidates = select_alarm_policy(val_data, val_probability, args.thresholds, args.hold_windows, args.max_val_far, device_names)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "best_device_event_gru.pt"
    torch.save({
        "model": "causal_gru", "input_dim": int(train_data.x.shape[1]), "hidden_dim": args.hidden_dim,
        "layers": args.layers, "dropout": args.dropout, "best_val_macro_f1_argmax": best_score,
        "alarm_policy": selected["alarm_policy"], "state_dict": best_state, "task": "device_attack_event",
        "label_semantics": metadata["label_semantics"],
    }, checkpoint_path)
    save_json(args.output_dir / "training_history.json", history)
    save_json(args.output_dir / "val_metrics_device_event_temporal.json", selected)
    save_json(args.output_dir / "val_alarm_policy_candidates.json", {"max_val_far": args.max_val_far, "candidates": candidates, "selected": selected["alarm_policy"]})
    print(json.dumps({"checkpoint": str(checkpoint_path), "best_val_macro_f1_argmax": best_score, "selected_alarm_policy": selected["alarm_policy"], "epochs": len(history)}, ensure_ascii=False, indent=2))


def test_only(args: argparse.Namespace, device_names: dict[int, str], device: torch.device) -> None:
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    required = {"model", "input_dim", "hidden_dim", "layers", "dropout", "alarm_policy", "state_dict"}
    if missing := required.difference(checkpoint):
        raise ValueError(f"Invalid checkpoint; missing {sorted(missing)}")
    if checkpoint["model"] != "causal_gru":
        raise ValueError("Checkpoint is not a causal GRU device-event model")
    test_data = EventTensor(args.data_dir / "test.npz")
    if test_data.x.shape[1] != int(checkpoint["input_dim"]):
        raise ValueError("Checkpoint and test feature dimensions differ")
    model = CausalEventGRU(int(checkpoint["input_dim"]), int(checkpoint["hidden_dim"]), int(checkpoint["layers"]), float(checkpoint["dropout"])).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    probability = probabilities(model, SequenceDataset(test_data), device, args.batch_size, args.num_workers)
    policy = checkpoint["alarm_policy"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(args.output_dir / "test_metrics_device_event_temporal.json", evaluate_probability(test_data, probability, float(policy["threshold"]), int(policy["hold_windows"]), device_names))


def main() -> None:
    args = parse_args()
    seed_all(args.seed)
    metadata = json.loads((args.data_dir / "metadata.json").read_text(encoding="utf-8"))
    device_names = {int(value): str(key) for key, value in metadata.get("device_mapping", {}).items()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.test_only:
        test_only(args, device_names, device)
    else:
        train(args, metadata, device_names, device)


if __name__ == "__main__":
    main()

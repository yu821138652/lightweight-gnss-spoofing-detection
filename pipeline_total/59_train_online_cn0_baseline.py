"""Train and deploy a causal online C/N0-baseline scene classifier.

This is intentionally separate from ``46_train_band_mean_multiclass.py``.
Ordinary band-mean training treats W5 windows as independent samples, while
this route maintains an L1/L5 state for each chronological source segment::

    input(t) = [L1_Cn0(t), L5_Cn0(t), L1Present(t), L5Present(t),
                x_L1(t-), x_L5(t-)]
    pred(t)  = classifier(W5 ending at t)
    x_b(t)   = alpha * x_b(t-) + (1 - alpha) * k_b(t), only if pred(t)=normal

The current absolute C/N0 values are retained.  The two online baselines are
additional model inputs; no residual-only representation and no external gate
is used.  Training, validation, and test all execute the same chronological
rollout.  Labels are used only for loss and metrics, never for a state update.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import random
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import BandMeanWindowClassifier


logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
LOG = logging.getLogger(__name__)

NUM_CLASSES = 4
NORMAL_CLASS = 0
CLASS_NAMES = {0: "normal", 1: "L1", 2: "L5", 3: "L1+L5"}
ENCODERS = ("lstm", "gru", "tcn")
BASE_FEATURE_NAMES = ["L1_Cn0DbHz", "L5_Cn0DbHz", "L1Present", "L5Present"]
ONLINE_FEATURE_NAMES = [*BASE_FEATURE_NAMES, "L1_Cn0Baseline", "L5_Cn0Baseline"]
EPOCH_GAP_NS = 2_000_000_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_contract(data_dir: Path) -> dict[str, str]:
    names = [
        "feature_names.json",
        "tensor_metadata.json",
        "scaler.json",
        "train.npz",
        "val.npz",
        "test.npz",
    ]
    missing = [name for name in names if not (data_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Tensor directory {data_dir} is missing {missing}")
    return {name: sha256_file(data_dir / name) for name in names}


def load_tensor_metadata(data_dir: Path) -> dict:
    path = data_dir / "tensor_metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return metadata


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_base_feature_indices(feature_names: list[str]) -> list[int]:
    missing = [name for name in BASE_FEATURE_NAMES if name not in feature_names]
    if missing:
        raise ValueError(
            "Online baseline requires absolute band-mean inputs "
            f"{BASE_FEATURE_NAMES}; missing={missing}"
        )
    return [feature_names.index(name) for name in BASE_FEATURE_NAMES]


@dataclass
class SplitData:
    x: np.ndarray
    y: np.ndarray
    single_band: np.ndarray
    recording_id: np.ndarray
    source_id: np.ndarray
    device_id: np.ndarray
    window_time_nanos: np.ndarray
    endpoint_tow: np.ndarray
    stream_key: np.ndarray


@dataclass
class Episode:
    stream_key: str
    indices: np.ndarray


@dataclass
class RolloutResult:
    metrics: dict
    indices: np.ndarray
    predictions: np.ndarray
    probabilities: np.ndarray


def load_split(data_dir: Path, split: str, feature_indices: list[int]) -> SplitData:
    path = data_dir / f"{split}.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    required = {
        "x", "y", "single_band_mask", "recording_id", "source_id", "device_id",
        "window_time_nanos", "endpoint_tow", "stream_key",
    }
    with np.load(path, allow_pickle=False) as data:
        missing = required.difference(data.files)
        if missing:
            raise ValueError(
                f"{path} lacks chronological trace fields {sorted(missing)}. "
                "Rebuild tensors with 45_build_band_mean_window_tensors.py."
            )
        raw_x = np.asarray(data["x"], dtype=np.float32)
        result = SplitData(
            x=raw_x[:, :, feature_indices],
            y=np.asarray(data["y"], dtype=np.int64),
            single_band=np.asarray(data["single_band_mask"], dtype=bool),
            recording_id=np.asarray(data["recording_id"], dtype=np.int32),
            source_id=np.asarray(data["source_id"], dtype=np.int32),
            device_id=np.asarray(data["device_id"], dtype=np.int64),
            window_time_nanos=np.asarray(data["window_time_nanos"], dtype=np.int64),
            endpoint_tow=np.asarray(data["endpoint_tow"], dtype=np.float64),
            stream_key=np.asarray(data["stream_key"]).astype(str),
        )
    if result.x.ndim != 3 or result.x.shape[-1] != len(BASE_FEATURE_NAMES):
        raise ValueError(f"Expected [B,T,{len(BASE_FEATURE_NAMES)}] selected tensors, got {result.x.shape}")
    count = len(result.x)
    fields = (
        result.y, result.single_band, result.recording_id, result.source_id,
        result.device_id, result.window_time_nanos, result.endpoint_tow, result.stream_key,
    )
    if any(len(values) != count for values in fields):
        raise ValueError(f"Trace arrays in {path} do not match x rows")
    return result


def _validate_episode_windows(data: SplitData, indices: np.ndarray, stream_key: str) -> None:
    times = data.window_time_nanos[indices]
    if len(times) > 1 and np.any(np.diff(times) <= 0):
        raise ValueError(f"Stream {stream_key!r} has non-increasing or duplicate endpoint times")
    for previous, current in zip(indices, indices[1:]):
        if not np.allclose(data.x[previous, 1:], data.x[current, :-1], atol=1e-6, rtol=0.0):
            raise ValueError(
                f"Stream {stream_key!r} has non-overlapping consecutive W5 windows; "
                "a missing segment boundary may not have been encoded"
            )


def build_episodes(data: SplitData) -> list[Episode]:
    grouped: dict[tuple[str, int, int], list[int]] = {}
    for index, (stream_key, source_id, device_id) in enumerate(
        zip(data.stream_key, data.source_id, data.device_id)
    ):
        grouped.setdefault((str(stream_key), int(source_id), int(device_id)), []).append(index)

    episodes: list[Episode] = []
    for (stream_key, _, _), members in sorted(grouped.items()):
        ordered = np.asarray(
            sorted(members, key=lambda index: int(data.window_time_nanos[index])), dtype=np.int64
        )
        start = 0
        for position in range(1, len(ordered) + 1):
            at_end = position == len(ordered)
            gap = (
                not at_end
                and int(data.window_time_nanos[ordered[position]] - data.window_time_nanos[ordered[position - 1]])
                > EPOCH_GAP_NS
            )
            if not at_end and not gap:
                continue
            chunk = ordered[start:position]
            _validate_episode_windows(data, chunk, stream_key)
            episodes.append(Episode(stream_key=stream_key, indices=chunk))
            start = position
    if not episodes:
        raise ValueError("No chronological episodes found")
    return episodes


class StreamState:
    """One causal state machine; only its own earlier predictions can update it."""

    def __init__(self, data: SplitData, episode: Episode, alpha: float):
        self.data = data
        self.episode = episode
        self.alpha = alpha
        self.position = 0
        self.time_steps = int(data.x.shape[1])
        self.baseline = np.full(2, np.nan, dtype=np.float32)
        self.history: deque[np.ndarray] = deque(maxlen=self.time_steps)
        self._prime_warmup()

    @property
    def done(self) -> bool:
        return self.position >= len(self.episode.indices)

    @property
    def row_index(self) -> int:
        if self.done:
            raise IndexError("State is already exhausted")
        return int(self.episode.indices[self.position])

    def _initialise_missing_bands(self, row: np.ndarray) -> None:
        for band in range(2):
            current = float(row[band])
            present = bool(row[2 + band] > 0.5)
            if present and np.isfinite(current) and not np.isfinite(self.baseline[band]):
                self.baseline[band] = current

    def _prime_warmup(self) -> None:
        first_window = self.data.x[int(self.episode.indices[0])]
        for row in first_window:
            self._initialise_missing_bands(row)
            self.history.append(self.baseline.copy())
        if len(self.history) != self.time_steps:
            raise AssertionError("Initial online state lacks a complete W5 baseline history")

    def prepare_input(self) -> np.ndarray:
        """Return the current W5 using baseline values available before its prediction."""
        if self.done:
            raise IndexError("Cannot prepare an exhausted state")
        if self.position > 0:
            current_row = self.data.x[self.row_index, -1]
            self._initialise_missing_bands(current_row)
            self.history.append(self.baseline.copy())
        state_history = np.stack(tuple(self.history), axis=0)
        result = np.concatenate((self.data.x[self.row_index], state_history), axis=1)
        # A never-observed band has no valid baseline.  Its corresponding
        # presence feature is zero, while zero keeps the tensor finite.
        return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def advance(self, predicted_normal: bool) -> None:
        """Apply the endpoint decision after inference so it affects only the future."""
        if self.done:
            raise IndexError("Cannot advance an exhausted state")
        if predicted_normal:
            endpoint = self.data.x[self.row_index, -1]
            for band in range(2):
                current = float(endpoint[band])
                present = bool(endpoint[2 + band] > 0.5)
                if not present or not np.isfinite(current):
                    continue
                if not np.isfinite(self.baseline[band]):
                    self.baseline[band] = current
                else:
                    self.baseline[band] = self.alpha * self.baseline[band] + (1.0 - self.alpha) * current
        self.position += 1


def calculate_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, loss: float | None = None
) -> dict:
    labels = list(range(NUM_CLASSES))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    result = {
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "accuracy": float((y_true == y_pred).mean()),
        "samples": int(len(y_true)),
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_labels": [CLASS_NAMES[label] for label in labels],
        "per_class": {
            CLASS_NAMES[label]: {
                "precision": float(precision[label]),
                "recall": float(recall[label]),
                "f1": float(f1[label]),
                "support": int(support[label]),
            }
            for label in labels
        },
        "class_support": {CLASS_NAMES[label]: int(support[label]) for label in labels},
    }
    if loss is not None:
        result["loss"] = float(loss)
    return result


def rollout_split(
    model: nn.Module,
    data: SplitData,
    episodes: list[Episode],
    alpha: float,
    device: torch.device,
    criterion: nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    rollout_batch_size: int = 256,
    optimizer_step_interval: int = 32,
) -> RolloutResult:
    """Execute the same chronological state protocol for train, validation, or test.

    Distinct streams are batched, but entries within one stream are never
    reordered.  In training, the argmax used for a state update is detached;
    labels only enter the CE loss for usable double-band endpoints.
    """
    if (criterion is None) != (optimizer is None):
        raise ValueError("criterion and optimizer must be provided together for online training")
    if rollout_batch_size < 1:
        raise ValueError("rollout_batch_size must be positive")
    if optimizer_step_interval < 1:
        raise ValueError("optimizer_step_interval must be positive")

    states = [StreamState(data, episode, alpha) for episode in episodes]
    total_loss = 0.0
    total_count = 0
    output_indices: list[int] = []
    output_predictions: list[int] = []
    output_probabilities: list[np.ndarray] = []

    was_training = model.training
    # Dropout must not make a stochastic gate decision that changes later
    # inputs.  eval() keeps BatchNorm/Dropout inference-consistent but still
    # permits gradients during the online training pass.
    model.eval()
    pending_batches = 0
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    while True:
        active = [state for state in states if not state.done]
        if not active:
            break
        prepared = [(state, state.prepare_input()) for state in active]
        # Every endpoint receives a model decision, including a single-band
        # endpoint.  Such endpoints are not semantically evaluable by the
        # four-class task, so they are excluded only from CE/metrics/export;
        # their available band can still causally refresh its own baseline.
        normal_decisions: dict[int, bool] = {}

        for start in range(0, len(prepared), rollout_batch_size):
            batch = prepared[start:start + rollout_batch_size]
            features = torch.from_numpy(np.stack([item[1] for item in batch])).to(device)
            targets = torch.as_tensor([data.y[item[0].row_index] for item in batch], dtype=torch.long, device=device)
            usable_mask = torch.as_tensor(
                [not data.single_band[item[0].row_index] for item in batch],
                dtype=torch.bool,
                device=device,
            )
            if optimizer is None:
                with torch.no_grad():
                    logits = model(features)
            else:
                logits = model(features)
                if bool(usable_mask.any()):
                    loss = criterion(logits[usable_mask], targets[usable_mask])
                    loss.backward()
                    usable_count = int(usable_mask.sum().item())
                    total_loss += float(loss.detach().item()) * usable_count
                    total_count += usable_count
                    pending_batches += 1
                    # One optimizer step per very small chronological batch makes
                    # a warm-started classifier catastrophically forget.  State
                    # decisions remain strictly ordered, while gradients from the
                    # next few causal rounds are accumulated before weights move.
                    if pending_batches >= optimizer_step_interval:
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        pending_batches = 0

            probabilities = torch.softmax(logits.detach(), dim=1).cpu().numpy()
            predictions = logits.detach().argmax(dim=1).cpu().numpy()
            for (state, _), prediction, probability in zip(batch, predictions, probabilities):
                row_index = state.row_index
                normal_decisions[id(state)] = int(prediction) == NORMAL_CLASS
                if not data.single_band[row_index]:
                    output_indices.append(row_index)
                    output_predictions.append(int(prediction))
                    output_probabilities.append(np.asarray(probability, dtype=np.float32))

        for state, _ in prepared:
            state.advance(normal_decisions[id(state)])

    if optimizer is not None and pending_batches:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    if was_training:
        model.train()
    if not output_indices:
        raise ValueError("No usable double-band endpoints were available for rollout")
    order = np.argsort(np.asarray(output_indices, dtype=np.int64), kind="mergesort")
    indices = np.asarray(output_indices, dtype=np.int64)[order]
    predictions = np.asarray(output_predictions, dtype=np.int64)[order]
    probabilities = np.stack(output_probabilities, axis=0)[order]
    targets = data.y[indices]
    mean_loss = total_loss / total_count if total_count else None
    return RolloutResult(
        metrics=calculate_metrics(targets, predictions, mean_loss),
        indices=indices,
        predictions=predictions,
        probabilities=probabilities,
    )


def class_weights(y: np.ndarray, single_band: np.ndarray, multipliers: list[float] | None) -> torch.Tensor:
    labels = y[~single_band]
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float32)
    present = counts > 0
    if int(present.sum()) < 2:
        raise ValueError(f"Training split has fewer than two classes: {counts.tolist()}")
    weights = np.zeros(NUM_CLASSES, dtype=np.float32)
    weights[present] = counts.sum() / (float(present.sum()) * counts[present])
    if multipliers is not None:
        if len(multipliers) != NUM_CLASSES:
            raise ValueError(f"Expected {NUM_CLASSES} class-weight multipliers")
        weights *= np.asarray(multipliers, dtype=np.float32)
    return torch.from_numpy(weights)


def build_model(args: argparse.Namespace, time_steps: int) -> BandMeanWindowClassifier:
    return BandMeanWindowClassifier(
        input_dim=len(ONLINE_FEATURE_NAMES),
        time_steps=time_steps,
        encoder=args.encoder,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        num_classes=NUM_CLASSES,
    )


def load_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint {path} has no state_dict")
    return checkpoint


def warm_start_from_absolute_checkpoint(
    model: BandMeanWindowClassifier, path: Path, args: argparse.Namespace, time_steps: int, device: torch.device
) -> dict:
    checkpoint = load_checkpoint(path, device)
    expected = BASE_FEATURE_NAMES
    if checkpoint.get("feature_names") != expected:
        raise ValueError(
            f"Warm-start checkpoint features must be {expected}, got {checkpoint.get('feature_names')}"
        )
    for field, expected_value in (
        ("encoder", args.encoder), ("time_steps", time_steps),
        ("hidden_dim", args.hidden_dim), ("num_classes", NUM_CLASSES),
    ):
        if checkpoint.get(field) != expected_value:
            raise ValueError(
                f"Warm-start checkpoint {field}={checkpoint.get(field)!r}, expected {expected_value!r}"
            )
    if int(checkpoint.get("input_dim", -1)) != len(BASE_FEATURE_NAMES):
        raise ValueError("Warm-start checkpoint is not a 4D absolute-C/N0 model")

    target_state = model.state_dict()
    source_state = checkpoint["state_dict"]
    for name, target in target_state.items():
        source = source_state.get(name)
        if source is None:
            raise ValueError(f"Warm-start checkpoint lacks parameter {name}")
        if source.shape == target.shape:
            target_state[name] = source.detach().clone()
            continue
        # The first temporal input projection changes from 4 to 6 channels.
        if (
            source.ndim >= 2
            and target.ndim == source.ndim
            and source.shape[0] == target.shape[0]
            and source.shape[1] == len(BASE_FEATURE_NAMES)
            and target.shape[1] == len(ONLINE_FEATURE_NAMES)
            and source.shape[2:] == target.shape[2:]
        ):
            expanded = torch.zeros_like(target)
            expanded[:, :len(BASE_FEATURE_NAMES)] = source.detach()
            target_state[name] = expanded
            continue
        raise ValueError(
            f"Cannot warm-start parameter {name}: source={tuple(source.shape)} target={tuple(target.shape)}"
        )
    model.load_state_dict(target_state)
    return {"path": str(path), "sha256": sha256_file(path)}


def checkpoint_payload(
    model: BandMeanWindowClassifier,
    args: argparse.Namespace,
    time_steps: int,
    parameter_count: int,
    metrics: dict,
    train_counts: list[int],
    metadata: dict,
    contract: dict[str, str],
    warm_start: dict | None,
    selection_epoch: int,
) -> dict:
    return {
        "model": f"online_cn0_baseline_{args.encoder}",
        "encoder": args.encoder,
        "time_steps": time_steps,
        "input_dim": len(ONLINE_FEATURE_NAMES),
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "num_classes": NUM_CLASSES,
        "parameter_count": parameter_count,
        "selection_epoch": selection_epoch,
        "state_dict": model.state_dict(),
        "val_metrics": metrics,
        "train_class_counts": train_counts,
        "feature_names": ONLINE_FEATURE_NAMES,
        "scaler_mode": metadata.get("scaler_mode"),
        "data_scope": metadata.get("data_scope"),
        "online_baseline": {
            "schema_version": 1,
            "alpha": float(args.alpha),
            "state_feature_names": ["L1_Cn0Baseline", "L5_Cn0Baseline"],
            "base_feature_names": BASE_FEATURE_NAMES,
            "update_rule": "argmax_class_equals_normal",
            "normal_class": NORMAL_CLASS,
            "update_order": "feature_uses_pre_update_state_then_prediction_updates_future",
            "warmup": "first W5 window initializes only; no pre-window update",
            "single_band_policy": "model_decision_updates_present_bands_and_exclude_from_loss_metrics",
            "reset_boundaries": ["recording", "source", "split", "segment", "receiver_gap"],
            "training_rollout": "chronological_round_robin_with_detached_argmax",
            "dropout_during_state_decision": "disabled_via_model_eval",
            "optimizer_step_interval": int(args.optimizer_step_interval),
        },
        "tensor_contract": contract,
        "warm_start": warm_start,
    }


def validate_metadata(metadata: dict) -> None:
    if str(metadata.get("scaler_mode")) != "global":
        raise ValueError("Online baseline requires --scaler-mode global; per-device training statistics are forbidden")
    causal = metadata.get("causal_baseline")
    if isinstance(causal, dict) and str(causal.get("mode", "none")) != "none":
        raise ValueError("Online baseline tensors must use absolute C/N0 with --causal-baseline-mode none")
    trace = metadata.get("online_rollout_trace")
    if not isinstance(trace, dict):
        raise ValueError("Tensor metadata lacks online_rollout_trace; rebuild tensors with the current script 45")


def save_metrics(path: Path, metrics: dict, parameter_count: int) -> None:
    path.write_text(
        json.dumps({**metrics, "parameter_count": parameter_count}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def train(args: argparse.Namespace, device: torch.device) -> None:
    metadata = load_tensor_metadata(args.data_dir)
    validate_metadata(metadata)
    names = json.loads((args.data_dir / "feature_names.json").read_text(encoding="utf-8"))
    feature_indices = resolve_base_feature_indices(names)
    contract = tensor_contract(args.data_dir)
    train_data = load_split(args.data_dir, "train", feature_indices)
    val_data = load_split(args.data_dir, "val", feature_indices)
    if train_data.x.shape[1] != val_data.x.shape[1]:
        raise ValueError("Train and validation tensors use different time-step counts")
    time_steps = int(train_data.x.shape[1])
    train_episodes = build_episodes(train_data)
    val_episodes = build_episodes(val_data)
    train_counts = np.bincount(train_data.y[~train_data.single_band], minlength=NUM_CLASSES).tolist()

    model = build_model(args, time_steps).to(device)
    warm_start = None
    if args.init_checkpoint is not None:
        warm_start = warm_start_from_absolute_checkpoint(
            model, args.init_checkpoint, args, time_steps, device
        )
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    LOG.info(
        "online model encoder=%s device=%s T=%d F=%d params=%d train_episodes=%d val_episodes=%d",
        args.encoder, device, time_steps, len(ONLINE_FEATURE_NAMES), parameter_count,
        len(train_episodes), len(val_episodes),
    )
    LOG.info("online alpha=%.6f train usable class counts=%s", args.alpha, train_counts)

    if args.dry_run:
        result = rollout_split(model, train_data, train_episodes, args.alpha, device)
        LOG.info("dry-run online rollout samples=%d macro_f1=%.4f", result.metrics["samples"], result.metrics["macro_f1"])
        return

    weights = class_weights(train_data.y, train_data.single_band, args.class_weight_mult).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_name = f"online_cn0_baseline_{args.encoder}"
    checkpoint_path = args.output_dir / f"best_{model_name}.pt"
    metrics_path = args.output_dir / f"val_metrics_{model_name}.json"
    best = -float("inf")
    stale = 0
    # Expanding the first projection with two zero columns makes the
    # warm-started six-dimensional model exactly the frozen four-dimensional
    # absolute-C/N0 classifier.  It is therefore a legitimate epoch-0
    # candidate, and prevents online fine tuning from being selected when it
    # only makes validation performance worse.
    if warm_start is not None:
        initial_result = rollout_split(model, val_data, val_episodes, args.alpha, device)
        best = float(initial_result.metrics["macro_f1"])
        torch.save(
            checkpoint_payload(
                model, args, time_steps, parameter_count, initial_result.metrics, train_counts,
                metadata, contract, warm_start, selection_epoch=0,
            ),
            checkpoint_path,
        )
        save_metrics(metrics_path, initial_result.metrics, parameter_count)
        LOG.info(
            "epoch=0 warm_start val_f1=%.4f val_acc=%.4f",
            initial_result.metrics["macro_f1"], initial_result.metrics["accuracy"],
        )
    for epoch in range(1, args.epochs + 1):
        train_result = rollout_split(
            model, train_data, train_episodes, args.alpha, device,
            criterion=criterion, optimizer=optimizer, rollout_batch_size=args.rollout_batch_size,
            optimizer_step_interval=args.optimizer_step_interval,
        )
        val_result = rollout_split(model, val_data, val_episodes, args.alpha, device)
        LOG.info(
            "epoch=%d train_loss=%.4f train_f1=%.4f val_f1=%.4f val_acc=%.4f",
            epoch, train_result.metrics.get("loss", float("nan")), train_result.metrics["macro_f1"],
            val_result.metrics["macro_f1"], val_result.metrics["accuracy"],
        )
        if val_result.metrics["macro_f1"] > best:
            best = float(val_result.metrics["macro_f1"])
            stale = 0
            torch.save(
                checkpoint_payload(
                    model, args, time_steps, parameter_count, val_result.metrics, train_counts,
                    metadata, contract, warm_start, selection_epoch=epoch,
                ),
                checkpoint_path,
            )
            save_metrics(metrics_path, val_result.metrics, parameter_count)
        else:
            stale += 1
            if stale >= args.patience:
                LOG.info("early stopping after %d stale epochs", stale)
                break
    LOG.info("best validation macro_f1=%.4f checkpoint=%s", best, checkpoint_path)


def validate_checkpoint_contract(checkpoint: dict, data_dir: Path, predict_split: str) -> None:
    contract = checkpoint.get("tensor_contract")
    if not isinstance(contract, dict):
        raise ValueError("Online checkpoint lacks a tensor contract")
    required = ["feature_names.json", "tensor_metadata.json", "scaler.json", f"{predict_split}.npz"]
    for name in required:
        path = data_dir / name
        if not path.is_file() or contract.get(name) != sha256_file(path):
            raise ValueError(f"Checkpoint/tensor artifact mismatch for {name}")


def run_test_only(args: argparse.Namespace, device: torch.device) -> None:
    checkpoint_path = args.checkpoint or args.output_dir / f"best_online_cn0_baseline_{args.encoder}.pt"
    checkpoint = load_checkpoint(checkpoint_path, device)
    online = checkpoint.get("online_baseline")
    if not isinstance(online, dict):
        raise ValueError("Checkpoint is not an online C/N0-baseline model")
    if checkpoint.get("feature_names") != ONLINE_FEATURE_NAMES:
        raise ValueError("Checkpoint online feature order does not match this trainer")
    if str(checkpoint.get("encoder")) != args.encoder:
        raise ValueError("Checkpoint encoder does not match --encoder")
    metadata = load_tensor_metadata(args.data_dir)
    validate_metadata(metadata)
    if str(checkpoint.get("scaler_mode")) != str(metadata.get("scaler_mode")):
        raise ValueError("Checkpoint/tensor scaler-mode mismatch")
    alpha = float(online.get("alpha"))
    if not 0.0 <= alpha < 1.0:
        raise ValueError("Checkpoint online alpha is invalid")
    validate_checkpoint_contract(checkpoint, args.data_dir, args.predict_split)
    names = json.loads((args.data_dir / "feature_names.json").read_text(encoding="utf-8"))
    data = load_split(args.data_dir, args.predict_split, resolve_base_feature_indices(names))
    time_steps = int(data.x.shape[1])
    model = BandMeanWindowClassifier(
        input_dim=len(ONLINE_FEATURE_NAMES), time_steps=time_steps,
        encoder=args.encoder, hidden_dim=int(checkpoint["hidden_dim"]),
        dropout=float(checkpoint["dropout"]), num_classes=NUM_CLASSES,
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    result = rollout_split(model, data, build_episodes(data), alpha, device)

    mapping_path = args.data_dir / "device_mapping.json"
    device_names: dict[int, str] = {}
    if mapping_path.is_file():
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        device_names = {int(value): str(name) for name, value in mapping.items()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"{args.predict_split}_predictions_band_mean_window_{args.encoder}.csv"
    fields = [
        "fold", "recording_id", "source_id", "device_id", "device_name", "window_time_nanos",
        "endpoint_tow", "stream_key", "true_class", "pred_class",
        *[f"prob_{CLASS_NAMES[index]}" for index in range(NUM_CLASSES)],
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row_index, prediction, probabilities in zip(
            result.indices, result.predictions, result.probabilities
        ):
            device_id = int(data.device_id[row_index])
            row = {
                "fold": args.fold if args.fold is not None else "",
                "recording_id": int(data.recording_id[row_index]),
                "source_id": int(data.source_id[row_index]),
                "device_id": device_id,
                "device_name": device_names.get(device_id, ""),
                "window_time_nanos": int(data.window_time_nanos[row_index]),
                "endpoint_tow": float(data.endpoint_tow[row_index]),
                "stream_key": str(data.stream_key[row_index]),
                "true_class": int(data.y[row_index]),
                "pred_class": int(prediction),
            }
            row.update({f"prob_{CLASS_NAMES[index]}": float(probabilities[index]) for index in range(NUM_CLASSES)})
            writer.writerow(row)
    save_metrics(args.output_dir / f"{args.predict_split}_metrics_{model_name(args)}.json", result.metrics, int(checkpoint["parameter_count"]))
    LOG.info(
        "online %s rollout samples=%d macro_f1=%.4f accuracy=%.4f exported=%s",
        args.predict_split, result.metrics["samples"], result.metrics["macro_f1"],
        result.metrics["accuracy"], csv_path,
    )


def model_name(args: argparse.Namespace) -> str:
    return f"online_cn0_baseline_{args.encoder}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--encoder", choices=ENCODERS, default="tcn")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--alpha", type=float, default=0.98)
    parser.add_argument("--rollout-batch-size", type=int, default=256)
    parser.add_argument("--optimizer-step-interval", type=int, default=32)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--class-weight-mult", type=float, nargs=NUM_CLASSES, default=None,
        metavar=("NORMAL", "L1", "L5", "L1L5"),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--test-only", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--predict-split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--fold", type=int, default=None)
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1 or args.rollout_batch_size < 1 or args.optimizer_step_interval < 1:
        parser.error("epochs, patience, rollout-batch-size, and optimizer-step-interval must be positive")
    if args.hidden_dim < 1 or not 0.0 <= args.dropout < 1.0:
        parser.error("hidden-dim must be positive and dropout must be in [0, 1)")
    if not 0.0 <= args.alpha < 1.0:
        parser.error("--alpha must be in [0, 1)")
    return args


def main() -> None:
    args = parse_args()
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.test_only:
        run_test_only(args, device)
    else:
        train(args, device)


if __name__ == "__main__":
    main()

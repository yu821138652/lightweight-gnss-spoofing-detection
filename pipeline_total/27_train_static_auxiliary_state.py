"""Train E3: binary target-spoofing detection with an auxiliary state head.

The deployable output remains the original formal binary target-band spoof
label.  During training only, a second head classifies active endpoints as:
0 normal, 1 formal target-spoofed, or 2 non-target-band during a reviewed
single-band attack interval.  State 2 is an operational context label; it is
not a claim that the non-target signal was itself spoofed.
"""

from __future__ import annotations

import argparse
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

from models import SignalRawStatsAuxiliaryState


def _load_module(name: str, filename: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


baseline = _load_module("_static_fusion_baseline_e3", "21_train_static_signal_fusion.py")
experts = _load_module("_static_band_experts_e3", "25_train_static_band_experts.py")
LOG = logging.getLogger(__name__)
AUXILIARY_NAMES = ("normal", "target_spoofed", "non_target_single_band_attack")


def load_reviewed_single_band_intervals(path: Path) -> tuple[dict[tuple[str, str, str], tuple[int, list[tuple[float, float]]]], str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    sessions = config.get("labeling", {}).get("session_spoofing_tow_intervals", {})
    if not isinstance(sessions, dict):
        raise ValueError("Invalid labeling.session_spoofing_tow_intervals in label config")
    intervals: dict[tuple[str, str, str], tuple[int, list[tuple[float, float]]]] = {}
    for environment, scenarios in sessions.items():
        if not isinstance(scenarios, dict):
            continue
        for scenario, target_band in (("st_L1", 1), ("st_L5", 5)):
            for session, entry in (scenarios.get(scenario, {}) or {}).items():
                if not isinstance(entry, dict) or str(entry.get("status")) != "reviewed":
                    continue
                ranges = [(float(start), float(end)) for start, end in (entry.get("intervals", []) or [])]
                if any(end < start for start, end in ranges):
                    raise ValueError(f"Invalid TOW interval for {environment}/{scenario}/{session}")
                intervals[(str(environment), scenario, str(session))] = (target_band, ranges)
    if not intervals:
        raise ValueError("No reviewed st_L1/st_L5 intervals in label config")
    return intervals, hashlib.sha256(path.read_bytes()).hexdigest()


class AuxiliaryStateDataset(baseline.FusionDataset):
    """Baseline fusion tensors plus a derived, isolated auxiliary state."""

    def __init__(self, *args, data_dir: Path, label_config: Path, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        raw_path = Path(args[0])
        stats_path = Path(args[1])
        with np.load(raw_path, allow_pickle=False) as raw:
            recording_id = np.asarray(raw["recording_id"]).copy()
            endpoint_tow = np.asarray(raw["endpoint_tow"]).copy()
        with np.load(stats_path, allow_pickle=False) as stats:
            names = json.loads((stats_path.parent / "feature_names.json").read_text(encoding="utf-8"))
            if "IsL5" not in names:
                raise ValueError(f"Missing unscaled IsL5 sidecar in {stats_path}")
            is_l5 = np.asarray(stats["x"][..., 0, names.index("IsL5")]).copy() >= 0.5
        if recording_id.shape != (len(self),) or endpoint_tow.shape != (len(self),):
            raise ValueError(f"Unexpected trace shapes in {raw_path}: recording={recording_id.shape} tow={endpoint_tow.shape}")
        trace = json.loads((data_dir / "window_trace_index.json").read_text(encoding="utf-8"))
        recordings = trace.get("recordings")
        if not isinstance(recordings, list):
            raise ValueError("window_trace_index.json has no recordings list")
        interval_map, self.label_config_sha256 = load_reviewed_single_band_intervals(label_config)
        self.auxiliary = self._build_auxiliary(recording_id, endpoint_tow, is_l5, recordings, interval_map)

    def _build_auxiliary(
        self,
        recording_id: np.ndarray,
        endpoint_tow: np.ndarray,
        is_l5: np.ndarray,
        recordings: list[dict[str, object]],
        interval_map: dict[tuple[str, str, str], tuple[int, list[tuple[float, float]]]],
    ) -> torch.Tensor:
        state = np.full(tuple(self.y.shape), baseline.IGNORE_INDEX, dtype=np.int64)
        active = self.mask.cpu().numpy() & self.y.ne(baseline.IGNORE_INDEX).cpu().numpy()
        labels = self.y.cpu().numpy()
        state[active] = 0
        state[active & (labels == 1)] = 1
        for window, recording_value in enumerate(recording_id.astype(np.int64, copy=False)):
            if recording_value < 0 or recording_value >= len(recordings):
                raise ValueError(f"Unknown recording_id {recording_value} in tensor trace")
            record = recordings[int(recording_value)]
            key = tuple(str(record[name]) for name in ("Environment", "Scenario", "Session"))
            entry = interval_map.get(key)
            if entry is None:
                continue
            target_band, ranges = entry
            tow = float(endpoint_tow[window])
            if not np.isfinite(tow) or not any(start <= tow <= end for start, end in ranges):
                continue
            non_target = is_l5[window] if target_band == 1 else ~is_l5[window]
            selected = active[window] & (labels[window] == 0) & non_target
            state[window, selected] = 2
        result = torch.from_numpy(state)
        support = torch.bincount(result[result.ne(baseline.IGNORE_INDEX)], minlength=3)
        if int(support[2]) == 0:
            raise ValueError("Auxiliary state has no non-target single-band-attack endpoints")
        self.auxiliary_support = {name: int(support[index]) for index, name in enumerate(AUXILIARY_NAMES)}
        return result.long()

    def __getitem__(self, index: int):
        raw, stats, mask, label = super().__getitem__(index)
        return raw, stats, mask, label, self.auxiliary[index]


def load_split(
    split: str,
    raw_dir: Path,
    stats_dir: Path,
    raw_feature_count: int,
    stats_feature_count: int,
    raw_feature_indices: list[int],
    stats_feature_indices: list[int],
    data_dir: Path,
    label_config: Path,
) -> AuxiliaryStateDataset:
    return AuxiliaryStateDataset(
        raw_dir / f"{split}.npz",
        stats_dir / f"{split}.npz",
        raw_feature_count,
        stats_feature_count,
        raw_feature_indices,
        stats_feature_indices,
        data_dir=data_dir,
        label_config=label_config,
    )


def auxiliary_weights(data: AuxiliaryStateDataset) -> torch.Tensor:
    active = data.auxiliary.ne(baseline.IGNORE_INDEX)
    counts = torch.bincount(data.auxiliary[active], minlength=3).float()
    if int(counts.min().item()) == 0:
        raise ValueError(f"E3 training split needs all three auxiliary states, found {counts.tolist()}")
    return counts.sum() / (3.0 * counts)


def make_model(args: argparse.Namespace, raw_dim: int, stats_dim: int, device: torch.device) -> SignalRawStatsAuxiliaryState:
    return SignalRawStatsAuxiliaryState(
        raw_input_dim=raw_dim,
        stats_input_dim=stats_dim,
        encoder=args.encoder,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    binary_labels: list[np.ndarray] = []
    binary_predictions: list[np.ndarray] = []
    auxiliary_labels: list[np.ndarray] = []
    auxiliary_predictions: list[np.ndarray] = []
    binary_loss = 0.0
    auxiliary_loss = 0.0
    count = 0
    for raw, stats, mask, labels, auxiliary in loader:
        raw = raw.to(device)
        stats = stats.to(device)
        active = mask.to(device) & labels.to(device).ne(baseline.IGNORE_INDEX)
        if not active.any():
            continue
        binary_logits, auxiliary_logits = model(raw, stats)
        selected_labels = labels.to(device)[active]
        selected_auxiliary = auxiliary.to(device)[active]
        binary_loss += float(nn.functional.cross_entropy(binary_logits[active], selected_labels).item()) * selected_labels.numel()
        auxiliary_loss += float(nn.functional.cross_entropy(auxiliary_logits[active], selected_auxiliary).item()) * selected_labels.numel()
        count += selected_labels.numel()
        binary_labels.append(selected_labels.cpu().numpy())
        binary_predictions.append(binary_logits[active].argmax(-1).cpu().numpy())
        auxiliary_labels.append(selected_auxiliary.cpu().numpy())
        auxiliary_predictions.append(auxiliary_logits[active].argmax(-1).cpu().numpy())
    if not count:
        raise ValueError("Evaluation split has no active endpoints")
    binary_y = np.concatenate(binary_labels)
    binary_pred = np.concatenate(binary_predictions)
    auxiliary_y = np.concatenate(auxiliary_labels)
    auxiliary_pred = np.concatenate(auxiliary_predictions)
    result: dict[str, Any] = experts.metrics(binary_y, binary_pred)
    result.update({
        "binary_loss": binary_loss / count,
        "auxiliary_loss": auxiliary_loss / count,
        "auxiliary_macro_f1": float(f1_score(auxiliary_y, auxiliary_pred, average="macro", labels=[0, 1, 2], zero_division=0)),
        "auxiliary_support": {name: int((auxiliary_y == index).sum()) for index, name in enumerate(AUXILIARY_NAMES)},
        "auxiliary_recall": {name: float(recall_score(auxiliary_y, auxiliary_pred, labels=[index], average="macro", zero_division=0)) for index, name in enumerate(AUXILIARY_NAMES)},
    })
    return result


def checkpoint_path(output_dir: Path, encoder: str) -> Path:
    return output_dir / f"best_auxiliary_state_signal_{encoder}_stats_mlp_fusion.pt"


def save_checkpoint(
    path: Path,
    args: argparse.Namespace,
    train: AuxiliaryStateDataset,
    raw_names: list[str],
    stats_names: list[str],
    model: nn.Module,
    parameter_count: int,
    binary_weights: torch.Tensor,
    auxiliary_weights_value: torch.Tensor,
    val_metrics: dict[str, Any],
) -> None:
    torch.save({
        "model": f"auxiliary_state_signal_{args.encoder}_stats_mlp_fusion",
        "architecture": "shared_fusion_binary_plus_auxiliary_state_head",
        "auxiliary_states": list(AUXILIARY_NAMES),
        "auxiliary_semantics": "normal|formal_target_spoofed|non_target_during_reviewed_single_band_attack",
        "label_config": str(args.label_config.resolve()),
        "label_config_sha256": train.label_config_sha256,
        "aux_loss_weight": args.aux_loss_weight,
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
        "parameter_count": parameter_count,
        "train_binary_class_weights": binary_weights.tolist(),
        "train_auxiliary_class_weights": auxiliary_weights_value.tolist(),
        "train_auxiliary_support": train.auxiliary_support,
        "state_dict": model.state_dict(),
        "val_metrics": val_metrics,
    }, path)


def validate_checkpoint(checkpoint: dict[str, Any], data: AuxiliaryStateDataset, raw_names: list[str], stats_names: list[str]) -> None:
    required = {
        "architecture", "encoder", "raw_time_steps", "raw_input_dim", "raw_feature_names",
        "stats_input_dim", "stats_feature_names", "hidden_dim", "dropout", "state_dict", "label_config_sha256",
    }
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise ValueError(f"Checkpoint is missing metadata: {missing}")
    if checkpoint["architecture"] != "shared_fusion_binary_plus_auxiliary_state_head":
        raise ValueError(f"Unsupported checkpoint architecture: {checkpoint['architecture']!r}")
    if checkpoint["label_config_sha256"] != data.label_config_sha256:
        raise ValueError("Label configuration changed since checkpoint selection; use the recorded configuration")
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
    parser.add_argument("--label-config", type=Path, default=ROOT / "configs" / "preprocessing.yml")
    parser.add_argument("--encoder", choices=baseline.ENCODERS, default="tcn")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--aux-loss-weight", type=float, default=0.25)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--raw-feature-set", choices=tuple(baseline.RAW_FEATURE_SETS), default="full")
    parser.add_argument("--stats-feature-set", choices=baseline.STATS_FEATURE_SETS, default="cn0_agc_coverage_rx_time_std")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--test-only", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.hidden_dim < 1 or args.patience < 1 or args.num_workers < 0:
        parser.error("epochs, batch-size, hidden-dim, patience, and num-workers are invalid")
    if not 0.0 <= args.dropout < 1.0 or args.aux_loss_weight < 0.0:
        parser.error("dropout must be in [0, 1), aux-loss-weight must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    baseline.seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_dir, stats_dir, raw_indices, raw_names, all_stats_names = baseline.load_data_contract(args.data_dir, args.raw_feature_set)
    stats_names = baseline.select_stats_feature_names(all_stats_names, args.stats_feature_set)
    stats_indices = baseline.feature_indices(all_stats_names, stats_names, "Stats")
    raw_count = len(baseline.load_feature_names(raw_dir / "feature_names.json"))
    stats_count = len(all_stats_names)
    train = load_split("train", raw_dir, stats_dir, raw_count, stats_count, raw_indices, stats_indices, args.data_dir, args.label_config)
    pin_memory = device.type == "cuda"

    if args.test_only:
        path = checkpoint_path(args.output_dir, args.encoder)
        if not path.is_file():
            raise FileNotFoundError(path)
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError(f"Invalid checkpoint: {path}")
        test = load_split("test", raw_dir, stats_dir, raw_count, stats_count, raw_indices, stats_indices, args.data_dir, args.label_config)
        baseline.validate_compatible(train, test, "test")
        validate_checkpoint(checkpoint, test, raw_names, stats_names)
        model_args = argparse.Namespace(encoder=checkpoint["encoder"], hidden_dim=checkpoint["hidden_dim"], dropout=checkpoint["dropout"])
        model = make_model(model_args, int(checkpoint["raw_input_dim"]), int(checkpoint["stats_input_dim"]), device)
        model.load_state_dict(checkpoint["state_dict"])
        metrics = evaluate(model, DataLoader(test, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=pin_memory), device)
        metrics.update({"checkpoint": str(path), "parameter_count": int(checkpoint["parameter_count"]), "train_auxiliary_support": checkpoint["train_auxiliary_support"]})
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "test_metrics_auxiliary_state.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        LOG.info("locked checkpoint test=%s", json.dumps(metrics))
        return

    val = load_split("val", raw_dir, stats_dir, raw_count, stats_count, raw_indices, stats_indices, args.data_dir, args.label_config)
    baseline.validate_compatible(train, val, "val")
    binary_weights = baseline.class_weights(train)
    aux_weights = auxiliary_weights(train)
    model = make_model(args, train.raw.shape[-1], train.stats.shape[-1], device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    LOG.info("architecture=auxiliary_state encoder=%s device=%s params=%d aux_weight=%.3f train_auxiliary_support=%s", args.encoder, device, parameter_count, args.aux_loss_weight, json.dumps(train.auxiliary_support))
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=pin_memory)
    if args.dry_run:
        raw, stats, _, _, _ = next(iter(train_loader))
        binary_logits, auxiliary_logits = model(raw.to(device), stats.to(device))
        LOG.info("dry-run binary_logits=%s auxiliary_logits=%s", tuple(binary_logits.shape), tuple(auxiliary_logits.shape))
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    binary_criterion = nn.CrossEntropyLoss(weight=binary_weights.to(device))
    auxiliary_criterion = nn.CrossEntropyLoss(weight=aux_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = -float("inf")
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        count = 0
        for raw, stats, mask, labels, auxiliary in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            raw = raw.to(device)
            stats = stats.to(device)
            active = mask.to(device) & labels.to(device).ne(baseline.IGNORE_INDEX)
            binary_logits, auxiliary_logits = model(raw, stats)
            targets = labels.to(device)[active]
            aux_targets = auxiliary.to(device)[active]
            loss = binary_criterion(binary_logits[active], targets) + args.aux_loss_weight * auxiliary_criterion(auxiliary_logits[active], aux_targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * targets.numel()
            count += targets.numel()
        metrics = evaluate(model, val_loader, device)
        LOG.info("epoch=%d train_loss=%.4f val=%s", epoch, total_loss / count, json.dumps(metrics))
        if float(metrics["macro_f1"]) > best:
            best = float(metrics["macro_f1"])
            stale = 0
            save_checkpoint(checkpoint_path(args.output_dir, args.encoder), args, train, raw_names, stats_names, model, parameter_count, binary_weights, aux_weights, metrics)
            (args.output_dir / "val_metrics_auxiliary_state.json").write_text(json.dumps({**metrics, "parameter_count": parameter_count, "train_auxiliary_support": train.auxiliary_support}, indent=2), encoding="utf-8")
        else:
            stale += 1
            if stale >= args.patience:
                LOG.info("early stopping")
                break
    LOG.info("complete; test was not read")


if __name__ == "__main__":
    main()

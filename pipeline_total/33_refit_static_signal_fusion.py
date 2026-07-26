"""Refit a locked static signal model on all outer-development data.

This script has no validation split and must only be used after architecture
and fixed epoch count are chosen without the outer test Session.  Its input
tensor directory therefore has every development Session assigned to ``train``
and the one untouched outer Session assigned to ``test``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_baseline():
    path = Path(__file__).with_name("21_train_static_signal_fusion.py")
    spec = importlib.util.spec_from_file_location("_static_fusion_refit", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


baseline = _load_baseline()
LOG = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--encoder", choices=baseline.ENCODERS, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--raw-feature-set", choices=tuple(baseline.RAW_FEATURE_SETS), default="full")
    parser.add_argument("--stats-feature-set", choices=baseline.STATS_FEATURE_SETS, default="full")
    parser.add_argument(
        "--sampling",
        choices=("window_uniform", "session_uniform"),
        default="window_uniform",
        help="Window-uniform baseline or inverse-session-size window sampling.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.hidden_dim < 1 or args.num_workers < 0:
        parser.error("epochs, batch-size, hidden-dim, and num-workers are invalid")
    if not 0.0 <= args.dropout < 1.0:
        parser.error("dropout must be in [0, 1)")
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
    train = baseline.load_split("train", raw_dir, stats_dir, raw_count, stats_count, raw_indices, stats_indices)
    weights = baseline.class_weights(train)
    model = baseline.make_model(train.raw.shape[-1], train.stats.shape[-1], args.encoder, args.hidden_dim, args.dropout, device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    session_window_counts: dict[str, int] | None = None
    sampler = None
    if args.sampling == "session_uniform":
        if train.recording_id is None:
            raise ValueError(
                "session_uniform sampling requires recording_id in the training NPZ; "
                "rebuild tensors with 20_build_static_timeblock_tensors.py"
            )
        session_ids, counts = torch.unique(train.recording_id, sorted=True, return_counts=True)
        if not len(session_ids):
            raise ValueError("session_uniform sampling requires at least one training Session")
        session_window_counts = {
            str(int(session_id)): int(count)
            for session_id, count in zip(session_ids.tolist(), counts.tolist(), strict=True)
        }
        inverse_counts = counts.float().reciprocal()
        sample_weights = inverse_counts[torch.searchsorted(session_ids, train.recording_id)]
        sampler = WeightedRandomSampler(
            weights=sample_weights.double(),
            num_samples=len(train),
            replacement=True,
            generator=torch.Generator().manual_seed(args.seed),
        )
    loader = DataLoader(
        train,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    LOG.info(
        "outer-development refit encoder=%s device=%s params=%d epochs=%d train_windows=%d sampling=%s",
        args.encoder, device, parameter_count, args.epochs, len(train), args.sampling,
    )
    if session_window_counts is not None:
        LOG.info("session-uniform source window counts=%s", session_window_counts)
    if args.dry_run:
        raw, stats, _, _ = next(iter(loader))
        LOG.info("dry-run logits=%s", tuple(model(raw.to(device), stats.to(device)).shape))
        return

    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        label_count = 0
        for raw, stats, mask, labels in tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}"):
            raw = raw.to(device)
            stats = stats.to(device)
            mask = mask.to(device)
            labels = labels.to(device)
            logits, target = baseline.valid(model(raw, stats), mask, labels)
            if not target.numel():
                continue
            loss = criterion(logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * target.numel()
            label_count += target.numel()
        if not label_count:
            raise ValueError("Training epoch has no active labels")
        LOG.info("epoch=%d train_loss=%.4f", epoch, total_loss / label_count)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_name = f"refit_signal_{args.encoder}_stats_mlp_fusion"
    checkpoint_path = args.output_dir / f"best_{model_name}.pt"
    torch.save({
        "model": model_name,
        "encoder": args.encoder,
        "raw_time_steps": int(train.raw.shape[-2]),
        "raw_input_dim": int(train.raw.shape[-1]),
        "raw_feature_indices": raw_indices,
        "raw_feature_names": raw_names,
        "raw_feature_set": args.raw_feature_set,
        "stats_input_dim": int(train.stats.shape[-1]),
        "stats_feature_names": stats_names,
        "stats_feature_set": args.stats_feature_set,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "weight_decay": args.weight_decay,
        "parameter_count": parameter_count,
        "class_weights": weights.tolist(),
        "selection_protocol": "fixed_epoch_refit_on_all_outer_development_sessions",
        "refit_epochs": args.epochs,
        "sampling": args.sampling,
        "session_window_counts": session_window_counts,
        "state_dict": model.state_dict(),
    }, checkpoint_path)
    (args.output_dir / "refit_metadata.json").write_text(json.dumps({
        "checkpoint": str(checkpoint_path), "epochs": args.epochs, "parameter_count": parameter_count,
        "train_windows": len(train), "selection_protocol": "fixed_epoch_refit_on_all_outer_development_sessions",
        "sampling": args.sampling, "session_window_counts": session_window_counts,
    }, indent=2), encoding="utf-8")
    LOG.info("complete; outer test was not read; checkpoint=%s", checkpoint_path)


if __name__ == "__main__":
    main()

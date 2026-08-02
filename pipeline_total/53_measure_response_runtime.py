"""Measure CPU runtime and size of a device-response checkpoint."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


class EventMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float, num_classes: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def make_model(kind: str, input_dim: int, hidden_dim: int, dropout: float, num_classes: int) -> nn.Module:
    if kind == "linear":
        return nn.Linear(input_dim, num_classes)
    if kind == "mlp":
        return EventMLP(input_dim, hidden_dim, dropout, num_classes)
    raise ValueError(f"Unsupported model kind: {kind}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if min(args.warmup, args.iterations, args.batch_size, args.threads) < 1:
        parser.error("runtime counts must be positive")
    return args


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    with np.load(args.data_dir / "test.npz", allow_pickle=False) as data:
        if "x" not in data.files or len(data["x"]) == 0:
            raise ValueError(f"Missing non-empty test.x in {args.data_dir / 'test.npz'}")
        x = torch.from_numpy(data["x"].astype(np.float32))
    input_dim = int(checkpoint["input_dim"])
    if x.shape[1] != input_dim:
        raise ValueError(f"Input dimension mismatch: checkpoint={input_dim}, tensor={x.shape[1]}")
    model = make_model(
        str(checkpoint["model"]),
        input_dim,
        int(checkpoint["hidden_dim"]),
        float(checkpoint["dropout"]),
        int(checkpoint["num_classes"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    one = x[:1]
    batch = x[: min(args.batch_size, len(x))]
    with torch.no_grad():
        for _ in range(args.warmup):
            model(one)
        single_ms: list[float] = []
        for index in range(args.iterations):
            sample = x[index % len(x) : index % len(x) + 1]
            start = time.perf_counter()
            model(sample)
            single_ms.append((time.perf_counter() - start) * 1000.0)
        batch_ms: list[float] = []
        for _ in range(max(10, args.iterations // 10)):
            start = time.perf_counter()
            model(batch)
            batch_ms.append((time.perf_counter() - start) * 1000.0)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    output = {
        "checkpoint": str(args.checkpoint),
        "data_dir": str(args.data_dir),
        "model": str(checkpoint["model"]),
        "input_dim": input_dim,
        "hidden_dim": int(checkpoint["hidden_dim"]),
        "num_classes": int(checkpoint["num_classes"]),
        "parameters": int(parameters),
        "checkpoint_bytes": int(args.checkpoint.stat().st_size),
        "threads": int(args.threads),
        "single_window_latency_ms": {
            "p50": percentile(single_ms, 50),
            "p95": percentile(single_ms, 95),
            "mean": float(np.mean(single_ms)),
        },
        "batch_latency_ms": {
            "batch_size": int(len(batch)),
            "p50": percentile(batch_ms, 50),
            "p95": percentile(batch_ms, 95),
            "throughput_windows_per_second": float(len(batch) / (np.mean(batch_ms) / 1000.0)),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Measure single-thread CPU forward cost of a band-mean scene checkpoint.

Inputs are already aggregated and standardized, so this reports model-only
latency. Raw-log parsing, band aggregation and scaler application are excluded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import BandMeanWindowClassifier  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--gate-checkpoint",
        type=Path,
        default=None,
        help="Optional gate checkpoint measured immediately before the classifier.",
    )
    parser.add_argument(
        "--gate-data-dir",
        type=Path,
        default=None,
        help="Tensor directory for --gate-checkpoint.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if min(args.warmup, args.iterations, args.batch_size, args.threads) < 1:
        parser.error("runtime counts must be positive")
    if (args.gate_checkpoint is None) != (args.gate_data_dir is None):
        parser.error("--gate-checkpoint and --gate-data-dir must be provided together")
    return args


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def causal_contract(value: object) -> dict:
    if not isinstance(value, dict):
        return {"mode": "none"}
    if str(value.get("mode", "none")) == "none":
        return {"mode": "none"}
    return value


def validate_checkpoint_tensor_metadata(
    checkpoint: dict, tensor_metadata: dict, data_dir: Path
) -> None:
    tensor_scaler = str(tensor_metadata.get("scaler_mode", "legacy_per_device"))
    checkpoint_scaler = str(checkpoint.get("scaler_mode", "legacy_per_device"))
    if tensor_scaler != checkpoint_scaler:
        raise ValueError(
            f"Checkpoint/tensor scaler mismatch: {checkpoint_scaler} vs {tensor_scaler}"
        )
    tensor_causal = causal_contract(tensor_metadata.get("causal_baseline"))
    checkpoint_causal = causal_contract(checkpoint.get("causal_baseline"))
    if tensor_causal != checkpoint_causal:
        raise ValueError("Checkpoint/tensor causal-baseline metadata mismatch")
    checkpoint_scope = checkpoint.get("data_scope")
    tensor_scope = tensor_metadata.get("data_scope")
    if checkpoint_scope is not None and checkpoint_scope != tensor_scope:
        raise ValueError(
            f"Checkpoint/tensor data scope mismatch: {checkpoint_scope} vs {tensor_scope}"
        )

    contract = checkpoint.get("tensor_contract")
    if contract is None:
        return
    if not isinstance(contract, dict):
        raise ValueError("Checkpoint tensor_contract must be a JSON-like object")
    required = ["feature_names.json", "tensor_metadata.json", "test.npz"]
    if "scaler.json" in contract:
        required.append("scaler.json")
    for name in ("device_mapping.json", "source_mapping.json"):
        if name in contract:
            required.append(name)
    for name in required:
        expected = contract.get(name)
        path = data_dir / name
        if not isinstance(expected, str) or not path.is_file():
            raise ValueError(f"Checkpoint tensor contract requires {name}")
        if sha256_file(path) != expected:
            raise ValueError(f"Checkpoint/tensor artifact mismatch for {name}")


def selected_feature_indices(data_dir: Path, checkpoint: dict) -> list[int]:
    all_names = json.loads((data_dir / "feature_names.json").read_text(encoding="utf-8"))
    selected = checkpoint.get("feature_names")
    if not isinstance(all_names, list) or not isinstance(selected, list) or not selected:
        raise ValueError("Tensor and checkpoint feature names are required")
    missing = [name for name in selected if name not in all_names]
    if missing:
        raise ValueError(f"Checkpoint features absent from tensor: {missing}")
    return [all_names.index(name) for name in selected]


def load_stage(checkpoint_path: Path, data_dir: Path) -> dict:
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(f"Invalid checkpoint: {checkpoint_path}")

    tensor_metadata = load_json(data_dir / "tensor_metadata.json")
    validate_checkpoint_tensor_metadata(checkpoint, tensor_metadata, data_dir)

    indices = selected_feature_indices(data_dir, checkpoint)
    with np.load(data_dir / "test.npz", allow_pickle=False) as data:
        x = np.asarray(data["x"], dtype=np.float32)[:, :, indices]
        single_band = np.asarray(data["single_band_mask"], dtype=bool)
    x = torch.from_numpy(x[~single_band])
    if len(x) == 0:
        raise ValueError(f"Test split has no usable windows: {data_dir}")

    model = BandMeanWindowClassifier(
        input_dim=int(checkpoint["input_dim"]),
        time_steps=int(checkpoint["time_steps"]),
        encoder=str(checkpoint["encoder"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        dropout=float(checkpoint["dropout"]),
        num_classes=int(checkpoint.get("num_classes", 4)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return {
        "checkpoint": checkpoint,
        "checkpoint_path": checkpoint_path,
        "data_dir": data_dir,
        "tensor_metadata": tensor_metadata,
        "x": x,
        "model": model,
    }


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    classifier_stage = load_stage(args.checkpoint, args.data_dir)
    stages = []
    if args.gate_checkpoint is not None:
        stages.append(load_stage(args.gate_checkpoint, args.gate_data_dir))
    stages.append(classifier_stage)

    batch_size = min([args.batch_size, *[len(stage["x"]) for stage in stages]])
    # ``inference_mode`` is unavailable in older PyTorch releases used by
    # some project environments; ``no_grad`` preserves the runtime
    # measurement semantics in that case.
    inference_context = getattr(torch, "inference_mode", torch.no_grad)
    with inference_context():
        for _ in range(args.warmup):
            for stage in stages:
                stage["model"](stage["x"][:1])
        single_ms: list[float] = []
        for index in range(args.iterations):
            start = time.perf_counter()
            for stage in stages:
                offset = index % len(stage["x"])
                stage["model"](stage["x"][offset : offset + 1])
            single_ms.append((time.perf_counter() - start) * 1000.0)
        batch_ms: list[float] = []
        for _ in range(max(20, args.iterations // 10)):
            start = time.perf_counter()
            for stage in stages:
                stage["model"](stage["x"][:batch_size])
            batch_ms.append((time.perf_counter() - start) * 1000.0)

    checkpoint = classifier_stage["checkpoint"]
    tensor_metadata = classifier_stage["tensor_metadata"]
    stage_summaries = []
    for role, stage in zip(
        (["gate", "classifier"] if len(stages) == 2 else ["classifier"]), stages
    ):
        stage_checkpoint = stage["checkpoint"]
        stage_summaries.append(
            {
                "role": role,
                "checkpoint": str(stage["checkpoint_path"]),
                "data_dir": str(stage["data_dir"]),
                "encoder": str(stage_checkpoint["encoder"]),
                "input_dim": int(stage_checkpoint["input_dim"]),
                "hidden_dim": int(stage_checkpoint["hidden_dim"]),
                "parameters": int(
                    sum(parameter.numel() for parameter in stage["model"].parameters())
                ),
                "checkpoint_bytes": int(stage["checkpoint_path"].stat().st_size),
            }
        )
    parameters = sum(stage["parameters"] for stage in stage_summaries)
    checkpoint_bytes = sum(stage["checkpoint_bytes"] for stage in stage_summaries)
    output = {
        "measurement_scope": (
            "sequential_gate_and_classifier_forwards_only_prebuilt_tensors"
            if len(stages) == 2
            else "model_forward_only_prebuilt_tensor"
        ),
        "checkpoint": str(args.checkpoint),
        "data_dir": str(args.data_dir),
        "scaler_mode": tensor_metadata.get("scaler_mode", "legacy_per_device"),
        "data_scope": tensor_metadata.get("data_scope"),
        "encoder": str(checkpoint["encoder"]),
        "time_steps": int(checkpoint["time_steps"]),
        "input_dim": int(checkpoint["input_dim"]),
        "hidden_dim": int(checkpoint["hidden_dim"]),
        "parameters": int(parameters),
        "checkpoint_bytes": int(checkpoint_bytes),
        "stages": stage_summaries,
        "threads": int(args.threads),
        "interop_threads": 1,
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "platform": platform.platform(),
        "single_window_latency_ms": {
            "p50": percentile(single_ms, 50),
            "p95": percentile(single_ms, 95),
            "mean": float(np.mean(single_ms)),
        },
        "batch_latency_ms": {
            "batch_size": int(batch_size),
            "p50": percentile(batch_ms, 50),
            "p95": percentile(batch_ms, 95),
            "throughput_windows_per_second": float(batch_size / (np.mean(batch_ms) / 1000.0)),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

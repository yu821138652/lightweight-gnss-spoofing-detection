"""Measure deployment-time latency of the self-updating C/N0 baseline path.

Unlike ``56_measure_band_mean_runtime.py``, this script measures the complete
online decision loop: W5 baseline-feature construction, model forward,
``argmax`` and the conditional update of the L1/L5 states.  It deliberately
excludes file reading, raw-log parsing, band aggregation, global scaling,
metrics and CSV output.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
ONLINE_PATH = ROOT / "pipeline_total" / "59_train_online_cn0_baseline.py"
SPEC = importlib.util.spec_from_file_location("online_cn0_runtime", ONLINE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot load online rollout implementation from {ONLINE_PATH}")
ONLINE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ONLINE
SPEC.loader.exec_module(ONLINE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--predict-split", choices=("train", "val", "test"), default="test")
    parser.add_argument(
        "--rollout-batch-size",
        type=int,
        default=1,
        help="1 measures sequential single-stream latency; >1 measures round-robin throughput.",
    )
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.rollout_batch_size < 1 or args.warmup < 0 or args.threads < 1:
        parser.error("rollout-batch-size and threads must be positive; warmup must be non-negative")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def validate_online_schema(checkpoint: dict) -> dict:
    online = checkpoint.get("online_baseline")
    if not isinstance(online, dict) or int(online.get("schema_version", -1)) != 1:
        raise ValueError("Checkpoint lacks online_baseline schema_version=1")
    if checkpoint.get("feature_names") != ONLINE.ONLINE_FEATURE_NAMES:
        raise ValueError("Checkpoint online feature order does not match the rollout implementation")
    if int(checkpoint.get("input_dim", -1)) != len(ONLINE.ONLINE_FEATURE_NAMES):
        raise ValueError("Checkpoint does not have the required six online input dimensions")
    required = {
        "update_rule": "argmax_class_equals_normal",
        "normal_class": ONLINE.NORMAL_CLASS,
        "update_order": "feature_uses_pre_update_state_then_prediction_updates_future",
        "single_band_policy": "model_decision_updates_present_bands_and_exclude_from_loss_metrics",
        "reset_boundaries": ["recording", "source", "split", "segment", "receiver_gap"],
    }
    for name, value in required.items():
        if online.get(name) != value:
            raise ValueError(f"Checkpoint online protocol mismatch for {name}: {online.get(name)!r}")
    alpha = float(online.get("alpha"))
    if not 0.0 <= alpha < 1.0:
        raise ValueError("Checkpoint online alpha is invalid")
    return online


def load_stage(args: argparse.Namespace) -> tuple[torch.nn.Module, object, list, dict, dict]:
    checkpoint = ONLINE.load_checkpoint(args.checkpoint, torch.device("cpu"))
    online = validate_online_schema(checkpoint)
    metadata = ONLINE.load_tensor_metadata(args.data_dir)
    ONLINE.validate_metadata(metadata)
    ONLINE.validate_checkpoint_contract(checkpoint, args.data_dir, args.predict_split)
    names = json.loads((args.data_dir / "feature_names.json").read_text(encoding="utf-8"))
    data = ONLINE.load_split(
        args.data_dir,
        args.predict_split,
        ONLINE.resolve_base_feature_indices(names),
    )
    if int(checkpoint.get("time_steps", -1)) != int(data.x.shape[1]):
        raise ValueError("Checkpoint/tensor time-step mismatch")
    model = ONLINE.BandMeanWindowClassifier(
        input_dim=len(ONLINE.ONLINE_FEATURE_NAMES),
        time_steps=int(data.x.shape[1]),
        encoder=str(checkpoint["encoder"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        dropout=float(checkpoint["dropout"]),
        num_classes=ONLINE.NUM_CLASSES,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    episodes = ONLINE.build_episodes(data)
    return model, data, episodes, checkpoint, online


def warmup(model: torch.nn.Module, data: object, episodes: list, alpha: float, count: int) -> None:
    if count == 0:
        return
    state = ONLINE.StreamState(data, episodes[0], alpha)
    value = torch.from_numpy(state.prepare_input()[None, ...])
    with torch.inference_mode():
        for _ in range(count):
            model(value)


def endpoint_key(data: object, row_index: int, prediction: int) -> bytes:
    return (
        f"{int(data.recording_id[row_index])}|{int(data.source_id[row_index])}|"
        f"{int(data.device_id[row_index])}|{int(data.window_time_nanos[row_index])}|"
        f"{str(data.stream_key[row_index])}|{prediction}"
    ).encode("utf-8")


def record_transition(
    state: object,
    data: object,
    prediction: int,
    counters: dict[str, object],
    checksum_items: list[bytes],
) -> None:
    row_index = state.row_index
    counters["endpoints"] += 1
    if bool(data.single_band[row_index]):
        counters["single_band_endpoints"] += 1
    else:
        counters["usable_endpoints"] += 1
        checksum_items.append(endpoint_key(data, row_index, prediction))
    if prediction == ONLINE.NORMAL_CLASS:
        counters["normal_predictions"] += 1
        endpoint = data.x[row_index, -1]
        for band, name in enumerate(("L1", "L5")):
            if endpoint[2 + band] > 0.5 and np.isfinite(endpoint[band]):
                counters["normal_decision_updates"][name] += 1
    state.advance(predicted_normal=prediction == ONLINE.NORMAL_CLASS)


def measure_single_stream(
    model: torch.nn.Module, data: object, episodes: list, alpha: float
) -> tuple[list[float], dict[str, object], list[bytes]]:
    elapsed_ms: list[float] = []
    counters: dict[str, object] = {
        "endpoints": 0,
        "usable_endpoints": 0,
        "single_band_endpoints": 0,
        "normal_predictions": 0,
        "normal_decision_updates": {"L1": 0, "L5": 0},
    }
    checksum_items: list[bytes] = []
    with torch.inference_mode():
        for episode in episodes:
            state = ONLINE.StreamState(data, episode, alpha)
            while not state.done:
                start = time.perf_counter()
                features = torch.from_numpy(state.prepare_input()[None, ...])
                prediction = int(model(features).argmax(dim=1).item())
                record_transition(state, data, prediction, counters, checksum_items)
                elapsed_ms.append((time.perf_counter() - start) * 1000.0)
    return elapsed_ms, counters, checksum_items


def measure_batched_rollout(
    model: torch.nn.Module,
    data: object,
    episodes: list,
    alpha: float,
    batch_size: int,
) -> tuple[list[float], dict[str, object], list[bytes]]:
    round_ms: list[float] = []
    counters: dict[str, object] = {
        "endpoints": 0,
        "usable_endpoints": 0,
        "single_band_endpoints": 0,
        "normal_predictions": 0,
        "normal_decision_updates": {"L1": 0, "L5": 0},
    }
    checksum_items: list[bytes] = []
    states = [ONLINE.StreamState(data, episode, alpha) for episode in episodes]
    with torch.inference_mode():
        while True:
            active = [state for state in states if not state.done]
            if not active:
                break
            start = time.perf_counter()
            prepared = [(state, state.prepare_input()) for state in active]
            decisions: dict[int, int] = {}
            for offset in range(0, len(prepared), batch_size):
                batch = prepared[offset:offset + batch_size]
                features = torch.from_numpy(np.stack([item[1] for item in batch]))
                predictions = model(features).argmax(dim=1).cpu().numpy()
                for (state, _), prediction in zip(batch, predictions):
                    decisions[id(state)] = int(prediction)
            for state, _ in prepared:
                record_transition(state, data, decisions[id(state)], counters, checksum_items)
            round_ms.append((time.perf_counter() - start) * 1000.0)
    return round_ms, counters, checksum_items


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    model, data, episodes, checkpoint, online = load_stage(args)
    alpha = float(online["alpha"])
    warmup(model, data, episodes, alpha, args.warmup)

    if args.rollout_batch_size == 1:
        timings, counters, checksum_items = measure_single_stream(model, data, episodes, alpha)
        latency = {
            "mode": "sequential_single_stream",
            "unit": "milliseconds_per_endpoint",
            "mean_ms": float(np.mean(timings)),
            "p50_ms": percentile(timings, 50),
            "p95_ms": percentile(timings, 95),
            "max_ms": float(np.max(timings)),
        }
    else:
        timings, counters, checksum_items = measure_batched_rollout(
            model, data, episodes, alpha, args.rollout_batch_size
        )
        total_ms = float(np.sum(timings))
        latency = {
            "mode": "round_robin_batched_throughput",
            "unit": "milliseconds_per_endpoint_average",
            "total_ms": total_ms,
            "mean_ms_per_endpoint": total_ms / int(counters["endpoints"]),
            "endpoints_per_second": int(counters["endpoints"]) * 1000.0 / total_ms,
            "round_p50_ms": percentile(timings, 50),
            "round_p95_ms": percentile(timings, 95),
        }

    digest = hashlib.sha256()
    for item in sorted(checksum_items):
        digest.update(item)
        digest.update(b"\n")
    contract = checkpoint.get("tensor_contract", {})
    payload = {
        "measurement_scope": {
            "included": [
                "online baseline feature construction",
                "CPU model forward",
                "argmax normal decision",
                "conditional L1/L5 state update",
            ],
            "excluded": [
                "file loading",
                "raw-log parsing",
                "band aggregation",
                "global scaling",
                "metrics",
                "CSV export",
            ],
        },
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_bytes": args.checkpoint.stat().st_size,
        "data_dir": str(args.data_dir.resolve()),
        "predict_split": args.predict_split,
        "tensor_contract": contract,
        "model": {
            "encoder": checkpoint["encoder"],
            "time_steps": int(checkpoint["time_steps"]),
            "input_dim": int(checkpoint["input_dim"]),
            "hidden_dim": int(checkpoint["hidden_dim"]),
            "dropout": float(checkpoint["dropout"]),
            "parameters": int(checkpoint["parameter_count"]),
            "base_feature_names": ONLINE.BASE_FEATURE_NAMES,
            "online_feature_names": ONLINE.ONLINE_FEATURE_NAMES,
        },
        "online_protocol": online,
        "episodes": len(episodes),
        "counters": counters,
        "prediction_checksum_usable_endpoints": digest.hexdigest(),
        "latency": latency,
        "configuration": {
            "rollout_batch_size": args.rollout_batch_size,
            "warmup_forwards": args.warmup,
            "threads": args.threads,
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"latency": latency, "counters": counters}, ensure_ascii=False))


if __name__ == "__main__":
    main()

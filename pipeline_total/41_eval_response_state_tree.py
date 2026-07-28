"""Evaluate a two-stage response-state tree.

Stage 1 predicts normal vs abnormal.  Stage 2 predicts anomaly vs direct spoof
for windows that stage 1 marked abnormal.  The final labels are:

0 = normal / no observable response
1 = attack-associated anomaly
2 = direct spoof
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "pipeline_total" / "37_train_device_attack_event.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--abnormal-checkpoint", type=Path, required=True)
    parser.add_argument("--type-checkpoint", type=Path, required=True)
    parser.add_argument("--abnormal-threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def load_train_module() -> Any:
    spec = importlib.util.spec_from_file_location("device_event_train", TRAIN_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {TRAIN_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    args = parse_args()
    if not 0.0 < args.abnormal_threshold < 1.0:
        raise ValueError("abnormal-threshold must be strictly between zero and one")
    train_mod = load_train_module()
    metadata = json.loads((args.data_dir / "metadata.json").read_text(encoding="utf-8"))
    device_names = {int(value): str(key) for key, value in metadata.get("device_mapping", {}).items()}
    data = train_mod.EventDataset(args.data_dir / f"{args.split}.npz", "y_response_state", None, "raw")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    abnormal_checkpoint, abnormal_model = train_mod.load_checkpoint_model(args.abnormal_checkpoint, device, data.x.shape[1])
    type_checkpoint, type_model = train_mod.load_checkpoint_model(args.type_checkpoint, device, data.x.shape[1])
    if int(abnormal_checkpoint.get("num_classes", 2)) != 2 or int(type_checkpoint.get("num_classes", 2)) != 2:
        raise ValueError("Both tree stages must be binary checkpoints")
    abnormal_probability = train_mod.probabilities(abnormal_model, data, device, args.batch_size)
    type_probability = train_mod.probabilities(type_model, data, device, args.batch_size)
    abnormal_pred = abnormal_probability[:, 1] >= args.abnormal_threshold
    type_pred = type_probability.argmax(axis=1).astype(np.int64)
    predicted = np.zeros(len(data), dtype=np.int64)
    predicted[abnormal_pred] = type_pred[abnormal_pred] + 1
    labels = data.y.numpy()
    result: dict[str, Any] = {
        "split": args.split,
        "abnormal_threshold": args.abnormal_threshold,
        "abnormal_checkpoint": str(args.abnormal_checkpoint),
        "type_checkpoint": str(args.type_checkpoint),
        "overall": train_mod.multiclass_metrics(labels, predicted, 3),
    }
    device_ids = data.device_id.numpy()
    recording_ids = data.recording_id.numpy()
    scenario_key, scenario_names = train_mod.scenario_ids(metadata, recording_ids)
    result["by_device"] = train_mod.group_metrics(labels, predicted, device_ids, device_names, 3)
    result["by_scenario"] = train_mod.group_metrics(labels, predicted, scenario_key, scenario_names, 3)
    result["by_recording"] = train_mod.group_metrics(labels, predicted, recording_ids, train_mod.recording_names(metadata), 3)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{args.split}_metrics_response_state_tree.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "metrics": str(output_path),
        "macro_f1": result["overall"]["macro_f1"],
        "far": result["overall"]["far"],
        "abnormal_recall": result["overall"]["abnormal_recall"],
        "anomaly_recall": result["overall"]["per_class"]["1"]["recall"],
        "direct_recall": result["overall"]["per_class"]["2"]["recall"],
    }, indent=2))


if __name__ == "__main__":
    main()

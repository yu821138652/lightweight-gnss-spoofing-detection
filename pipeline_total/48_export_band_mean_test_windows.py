"""Export every band-mean test window across static CV folds for inspection.

The cross-fold aggregation in ``47_aggregate_band_mean_cv.py`` intentionally
drops ``single_band`` windows before scoring, because a single observed band
cannot express the four-way scene.  This exporter is the complement: it loads
each fold's checkpoint, predicts *every* test window (``single_band`` included),
and writes one annotated row per window so the full held-out picture can be
eyeballed.

Each row carries enough provenance to slice the export any way: fold, the fold's
held-out recording scenario, device name, endpoint TOW, the ``single_band`` flag,
the ground-truth four-way class, and the model's prediction.  ``single_band``
rows keep the builder's forced label 0 (normal); their ``pred_class`` is still
the raw model output, so they are informative but must not be mixed into the
four-way metrics.

Reuses the per-fold tensors and checkpoints already produced under
``--tensors-root`` / ``--training-root``; it never trains.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import BandMeanWindowClassifier


NUM_CLASSES = 4
CLASS_NAMES = {0: "normal", 1: "L1", 2: "L5", 3: "L1+L5"}
PROTECTED_FEATURES = ("L1Present", "L5Present")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
LOG = logging.getLogger(__name__)


def resolve_kept_indices(all_names: list[str], drop_features: list[str] | None) -> list[int]:
    """Mirror the trainer's --drop-features selection so a checkpoint trained with
    an ablation is fed exactly the columns it expects at inference time."""
    if not drop_features:
        return list(range(len(all_names)))
    drop_tokens = {token.strip() for token in drop_features if token.strip()}
    kept: list[int] = []
    for index, name in enumerate(all_names):
        protected = name in PROTECTED_FEATURES
        suffix = name.split("_", 1)[1] if "_" in name else name
        dropped = (not protected) and (name in drop_tokens or suffix in drop_tokens)
        if not dropped:
            kept.append(index)
    return kept


def load_fold_scenarios(assignment_path: Path) -> dict[int, dict[str, str]]:
    """Map fold id -> held-out recording {Environment, Scenario, Session}."""
    frame = pd.read_csv(assignment_path, encoding="utf-8-sig")
    required = {"fold", "role", "Environment", "Scenario", "Session"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{assignment_path} missing columns: {sorted(missing)}")
    test_rows = frame[frame["role"].astype(str) == "test"]
    scenarios: dict[int, dict[str, str]] = {}
    for row in test_rows.itertuples(index=False):
        scenarios[int(row.fold)] = {
            "Environment": str(row.Environment),
            "Scenario": str(row.Scenario),
            "Session": str(row.Session),
        }
    if not scenarios:
        raise ValueError(f"No test-role rows in {assignment_path}")
    return scenarios


def load_device_mapping(tensor_dir: Path) -> dict[int, str]:
    path = tensor_dir / "device_mapping.json"
    if not path.is_file():
        return {}
    mapping = json.loads(path.read_text(encoding="utf-8"))
    return {int(v): str(k) for k, v in mapping.items()}


@torch.no_grad()
def predict_fold(
    fold: int,
    tensor_dir: Path,
    checkpoint_path: Path,
    scenario: dict[str, str],
    device: torch.device,
) -> pd.DataFrame:
    test_path = tensor_dir / "test.npz"
    if not test_path.is_file():
        raise FileNotFoundError(test_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
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

    id_to_device = load_device_mapping(tensor_dir)
    with np.load(test_path, allow_pickle=False) as data:
        x = np.asarray(data["x"], dtype=np.float32)
        y = np.asarray(data["y"], dtype=np.int64)
        single_band = np.asarray(data["single_band_mask"], dtype=bool)
        device_id = np.asarray(data["device_id"], dtype=np.int64) if "device_id" in data.files else None
        recording_id = np.asarray(data["recording_id"], dtype=np.int64) if "recording_id" in data.files else None
        endpoint_tow = np.asarray(data["endpoint_tow"], dtype=np.float64) if "endpoint_tow" in data.files else None
    # A checkpoint trained with --drop-features stored fewer columns than the raw
    # 10-feature tensor; replay the same selection before the shape check.
    drop_features = checkpoint.get("drop_features")
    if drop_features:
        names_path = tensor_dir / "feature_names.json"
        all_names = json.loads(names_path.read_text(encoding="utf-8"))
        kept = resolve_kept_indices(all_names, drop_features)
        x = x[:, :, kept]
    if x.shape[1:] != (time_steps, feature_dim):
        raise ValueError(
            f"fold {fold}: checkpoint expects [T,F]=({time_steps},{feature_dim}), test has {tuple(x.shape[1:])}"
        )
    if len(x) == 0:
        raise ValueError(f"fold {fold} test split has no windows")

    logits = model(torch.from_numpy(x).to(device))
    probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
    preds = logits.argmax(-1).cpu().numpy()

    rows = len(y)
    frame = pd.DataFrame({
        "fold": fold,
        "Environment": scenario["Environment"],
        "Scenario": scenario["Scenario"],
        "Session": scenario["Session"],
        "device_id": device_id if device_id is not None else -1,
        "device_name": [id_to_device.get(int(device_id[i]), "") if device_id is not None else "" for i in range(rows)],
        "endpoint_tow": endpoint_tow if endpoint_tow is not None else np.nan,
        "single_band": single_band,
        "true_class": y,
        "true_class_name": [CLASS_NAMES[int(c)] for c in y],
        "pred_class": preds,
        "pred_class_name": [CLASS_NAMES[int(c)] for c in preds],
        "correct": (preds == y),
    })
    for c in range(NUM_CLASSES):
        frame[f"prob_{CLASS_NAMES[c]}"] = probabilities[:, c]
    return frame


def summarize(combined: pd.DataFrame) -> None:
    total = len(combined)
    single = int(combined["single_band"].sum())
    usable = total - single
    LOG.info("exported %d windows: usable=%d single_band=%d", total, usable, single)
    LOG.info("--- usable windows only (the four-way task) ---")
    usable_frame = combined[~combined["single_band"]]
    if not usable_frame.empty:
        acc = float((usable_frame["pred_class"] == usable_frame["true_class"]).mean())
        LOG.info("usable accuracy=%.4f", acc)
        for name, group in usable_frame.groupby("true_class_name"):
            hit = float((group["pred_class"] == group["true_class"]).mean())
            LOG.info("  true=%-7s n=%-6d recall=%.3f", name, len(group), hit)
    LOG.info("--- single_band windows (excluded from task; raw predictions) ---")
    single_frame = combined[combined["single_band"]]
    if not single_frame.empty:
        dist = single_frame["pred_class_name"].value_counts().to_dict()
        LOG.info("  single_band prediction distribution=%s", dist)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol-dir", type=Path,
        default=ROOT / "output" / "protocols" / "static_time_block_outer_v2",
    )
    parser.add_argument(
        "--tensors-root", type=Path,
        default=ROOT / "output" / "tensors" / "band_mean_window_static_v1",
    )
    parser.add_argument(
        "--training-root", type=Path,
        default=ROOT / "output" / "training" / "band_mean_multiclass_cv",
    )
    parser.add_argument("--encoder", choices=("lstm", "gru", "tcn"), default="tcn")
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "output" / "training" / "band_mean_multiclass_cv" / "all_test_windows.csv",
    )
    parser.add_argument("--folds", type=int, nargs="*", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scenarios = load_fold_scenarios(args.protocol_dir / "fold_assignment.csv")
    folds = args.folds or sorted(scenarios)

    frames: list[pd.DataFrame] = []
    for fold in folds:
        if fold not in scenarios:
            raise ValueError(f"fold {fold} absent from fold_assignment.csv")
        tensor_dir = args.tensors_root / f"fold_{fold}"
        checkpoint = args.training_root / f"fold_{fold}" / f"best_band_mean_window_{args.encoder}.pt"
        frame = predict_fold(fold, tensor_dir, checkpoint, scenarios[fold], device)
        LOG.info("fold %d (%s): %d windows", fold, scenarios[fold]["Scenario"], len(frame))
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output, index=False, encoding="utf-8-sig")
    LOG.info("wrote %s", args.output)
    summarize(combined)


if __name__ == "__main__":
    main()

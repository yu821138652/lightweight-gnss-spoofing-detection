"""Plot device-level validation review views for selected recording sessions.

The plots combine the source labels, per-signal model probabilities and raw
signal-feature summaries. They are for diagnosing validation failures only and
never read test.npz.
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Windows workstations in this project use Microsoft YaHei; keep a portable fallback.
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def quantile_10(values: pd.Series) -> float:
    return float(values.quantile(0.10))


def quantile_90(values: pd.Series) -> float:
    return float(values.quantile(0.90))


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def aggregate_device_epochs(predictions: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "Environment",
        "Scenario",
        "Session",
        "DeviceName",
        "SourceRelativePath",
        "endpoint_TimeNanos",
        "endpoint_TOW",
    ]
    aggregation = {
        "true_label": "max",
        "spoofing_probability": ["mean", "max", quantile_90],
        "predicted_label": "mean",
        "signal_id": "size",
        "Cn0DbHz": ["median", quantile_10, quantile_90],
        "Cn0DbHz_dt": "median",
        "Cn0DbHz_std": "median",
        "AgcDb": "median",
        "ReceivedSvTimeUncertaintyNanos": "median",
    }
    aggregated = predictions.groupby(group_columns, dropna=False).agg(aggregation).reset_index()
    aggregated.columns = [
        "_".join(str(part) for part in column if part).rstrip("_") if isinstance(column, tuple) else column
        for column in aggregated.columns
    ]
    return aggregated.rename(
        columns={
            "true_label_max": "true_label",
            "spoofing_probability_mean": "mean_spoofing_probability",
            "spoofing_probability_max": "max_spoofing_probability",
            "spoofing_probability_quantile_90": "p90_spoofing_probability",
            "predicted_label_mean": "predicted_signal_fraction",
            "signal_id_size": "valid_signal_count",
            "Cn0DbHz_median": "cn0_median",
            "Cn0DbHz_quantile_10": "cn0_p10",
            "Cn0DbHz_quantile_90": "cn0_p90",
            "Cn0DbHz_dt_median": "cn0_dt_median",
            "Cn0DbHz_std_median": "cn0_std_median",
            "AgcDb_median": "agc_median",
            "ReceivedSvTimeUncertaintyNanos_median": "received_time_uncertainty_median",
        }
    )


def draw_label_band(axis, x: np.ndarray, labels: np.ndarray) -> None:
    axis.fill_between(
        x,
        0,
        1,
        where=labels.astype(bool),
        step="mid",
        transform=axis.get_xaxis_transform(),
        color="#d94841",
        alpha=0.16,
        label="True spoofing interval",
    )


def plot_device_epoch_view(device_epochs: pd.DataFrame, output_path: Path) -> None:
    device_epochs = device_epochs.sort_values("endpoint_TOW")
    x = device_epochs["endpoint_TOW"].to_numpy()
    labels = device_epochs["true_label"].to_numpy()
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True, constrained_layout=True)

    probability_axis = axes[0]
    draw_label_band(probability_axis, x, labels)
    probability_axis.plot(x, device_epochs["mean_spoofing_probability"], color="#007f7b", label="Mean signal probability")
    probability_axis.plot(x, device_epochs["p90_spoofing_probability"], color="#e07a2f", label="P90 signal probability")
    probability_axis.plot(x, device_epochs["predicted_signal_fraction"], color="#5b4b8a", label="Predicted spoofing fraction")
    probability_axis.axhline(0.5, color="#555555", linewidth=0.9, linestyle="--", label="Signal threshold")
    probability_axis.set_ylim(-0.02, 1.02)
    probability_axis.set_ylabel("Probability / fraction")
    probability_axis.legend(loc="upper right", ncol=2, fontsize=8)
    probability_axis.set_title(
        f"{device_epochs['Scenario'].iloc[0]} | {device_epochs['Session'].iloc[0]} | {device_epochs['DeviceName'].iloc[0]}"
    )

    cn0_axis = axes[1]
    cn0_axis.fill_between(x, device_epochs["cn0_p10"], device_epochs["cn0_p90"], color="#3b8bc2", alpha=0.20, label="C/N0 P10-P90")
    cn0_axis.plot(x, device_epochs["cn0_median"], color="#1f5f99", label="C/N0 median")
    cn0_axis.set_ylabel("C/N0 (dB-Hz)")
    cn0_axis.legend(loc="upper right", fontsize=8)

    agc_axis = axes[2]
    agc_axis.plot(x, device_epochs["agc_median"], color="#9a6b2f", label="AGC median")
    agc_axis.plot(x, device_epochs["cn0_dt_median"], color="#7a5caa", label="C/N0 dt median")
    agc_axis.set_ylabel("AGC / C/N0 dt")
    agc_axis.set_xlabel("GPS TOW (s)")
    agc_axis.legend(loc="upper right", fontsize=8)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "output" / "tensors_mixed")
    parser.add_argument("--csv", type=Path, default=PROJECT_ROOT / "output" / "processed_gnss_data.csv")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--device", default=None, help="Optional single DeviceName to review.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    checkpoint_path = args.model_dir / f"best_{args.model}.pt"
    validation_npz = args.data_dir / "val.npz"
    manifest_path = args.data_dir / "recording_split_manifest.csv"
    for path in (args.csv, checkpoint_path, validation_npz, manifest_path):
        if not path.exists():
            raise FileNotFoundError(path)

    error_export = load_module(
        "validation_error_export_for_plotting",
        PROJECT_ROOT / "pipeline_total" / "09_export_validation_misclassifications.py",
    )
    tensor_builder = load_module(
        "tensor_builder_for_plotting",
        PROJECT_ROOT / "pipeline_total" / "05_build_train_val_test_tensors.py",
    )
    training_module = load_module(
        "training_module_for_plotting",
        PROJECT_ROOT / "pipeline_total" / "07_train_models.py",
    )
    frame, identity_column = error_export.prepare_validation_frame(args.csv, manifest_path, tensor_builder)
    expected_mask, expected_y, metadata = error_export.reconstruct_validation_metadata(
        frame, identity_column, tensor_builder.FEATURE_COLS
    )
    _, probabilities, predicted_labels, dataset = error_export.predict_validation(
        training_module, checkpoint_path, validation_npz, args.batch_size
    )
    if expected_mask.shape != tuple(dataset.mask.shape) or not np.array_equal(expected_mask, dataset.mask.numpy()):
        raise RuntimeError("Reconstructed validation mask differs from val.npz; refusing to plot misaligned rows.")
    if expected_y.shape != tuple(dataset.y.shape) or not np.array_equal(expected_y, dataset.y.numpy()):
        raise RuntimeError("Reconstructed validation labels differ from val.npz; refusing to plot misaligned rows.")
    if len(metadata) != len(probabilities):
        raise RuntimeError("Metadata rows do not match valid validation predictions.")

    metadata["predicted_label"] = predicted_labels.astype(int)
    metadata["spoofing_probability"] = probabilities
    selection = metadata.loc[
        (metadata["Scenario"].astype(str) == args.scenario) & (metadata["Session"].astype(str) == args.session)
    ].copy()
    if args.device:
        selection = selection.loc[selection["DeviceName"].astype(str) == args.device].copy()
    if selection.empty:
        raise ValueError("No validation predictions match the requested Scenario, Session and DeviceName.")

    output_dir = args.output_dir or args.model_dir / "validation_error_review" / safe_filename(args.scenario) / safe_filename(args.session)
    output_dir.mkdir(parents=True, exist_ok=True)
    device_epochs = aggregate_device_epochs(selection)
    device_epochs.to_csv(output_dir / "device_epoch_prediction_summary.csv", index=False, encoding="utf-8-sig")
    for device_name, device_frame in device_epochs.groupby("DeviceName", sort=True):
        plot_device_epoch_view(device_frame, output_dir / f"{safe_filename(str(device_name))}_review.png")
    print(f"Saved {device_epochs['DeviceName'].nunique()} device review plots to {output_dir}")


if __name__ == "__main__":
    main()

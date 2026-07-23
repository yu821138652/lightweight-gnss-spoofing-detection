#!/usr/bin/env python3
"""Render a per-device model-decision overview for one static test Session."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


DEVICE_ORDER = [
    "Google_Pixel6",
    "Google_Pixel_Watch1",
    "Google_Pixel_Watch2",
    "HUAWEI_Mate40",
    "RedMi_K60",
    "XiaoMi_MI8",
]
TARGET_BAND_TOKENS = {
    "L1": ("_L1", "_E1", "_B1"),
    "L5": ("_L5", "_E5", "_B2"),
}


def is_target_band(signal_id: pd.Series, scenario: str) -> pd.Series:
    if "L_15" in scenario:
        return pd.Series(True, index=signal_id.index)
    tokens = TARGET_BAND_TOKENS["L5"] if "L5" in scenario else TARGET_BAND_TOKENS["L1"]
    return signal_id.astype(str).map(lambda value: any(token in value for token in tokens))


def label_intervals(frame: pd.DataFrame, config: dict) -> list[tuple[float, float]]:
    first = frame.iloc[0]
    entry = (
        config.get("labeling", {})
        .get("session_spoofing_tow_intervals", {})
        .get(str(first["Environment"]), {})
        .get(str(first["Scenario"]), {})
        .get(str(first["Session"]), {})
    )
    if not isinstance(entry, dict):
        return []
    return [(float(start), float(end)) for start, end in entry.get("intervals", [])]


def shade_labels(axis: plt.Axes, intervals: list[tuple[float, float]]) -> None:
    for start, end in intervals:
        axis.axvspan(start, end, color="#d84949", alpha=0.16, zorder=0)
        axis.axvline(start, color="#d84949", linestyle=":", linewidth=0.8)
        axis.axvline(end, color="#d84949", linestyle=":", linewidth=0.8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/preprocessing.yml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bin-seconds", type=int, default=10)
    args = parser.parse_args()

    frame = pd.read_csv(args.predictions, encoding="utf-8-sig")
    required = {
        "Environment", "Scenario", "Session", "DeviceName", "signal_id", "EndpointTOW",
        "Label", "Prediction", "PositiveProbability", "ErrorType",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Prediction CSV is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Prediction CSV is empty")
    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    frame["EndpointTOW"] = pd.to_numeric(frame["EndpointTOW"], errors="raise")
    frame["PositiveProbability"] = pd.to_numeric(frame["PositiveProbability"], errors="raise")
    frame["Label"] = pd.to_numeric(frame["Label"], errors="raise").astype(int)
    frame["Prediction"] = pd.to_numeric(frame["Prediction"], errors="raise").astype(int)
    frame["TargetBand"] = is_target_band(frame["signal_id"], str(frame.iloc[0]["Scenario"]))
    devices = [device for device in DEVICE_ORDER if device in set(frame["DeviceName"])]
    devices.extend(sorted(set(frame["DeviceName"]).difference(devices)))
    intervals = label_intervals(frame, config)
    tow_min, tow_max = frame["EndpointTOW"].min(), frame["EndpointTOW"].max()

    fig, axes = plt.subplots(3, len(devices), figsize=(4.2 * len(devices), 9.5), sharex="col")
    fig.suptitle(
        f"Model decisions | {frame.iloc[0]['Environment']} / {frame.iloc[0]['Scenario']} / {frame.iloc[0]['Session']}\n"
        "red span: formal target interval; alert row: model positives; target row: TP/FN on target bands",
        fontsize=13,
    )
    for column, device in enumerate(devices):
        data = frame.loc[frame["DeviceName"] == device].copy()
        alerts = data.loc[data["Prediction"] == 1]
        positives = data.loc[data["Label"] == 1]
        alert_axis, target_axis, probability_axis = axes[:, column]
        for axis in (alert_axis, target_axis, probability_axis):
            shade_labels(axis, intervals)
            axis.grid(alpha=0.2, linewidth=0.5)
            axis.set_xlim(tow_min, tow_max)

        for outcome, color, label in (("TP", "#171717", "TP"), ("FP", "#e67e22", "FP")):
            part = alerts.loc[alerts["ErrorType"] == outcome]
            y = np.where(part["TargetBand"], 5.0, 1.0)
            alert_axis.scatter(part["EndpointTOW"], y, s=2, color=color, alpha=0.45, label=label)
        alert_axis.set_yticks([1, 5], ["other", "target"])
        alert_axis.set_title(device, fontsize=10)

        for outcome, color, label in (("TP", "#171717", "TP"), ("FN", "#2878b5", "FN")):
            part = positives.loc[positives["ErrorType"] == outcome]
            target_axis.scatter(part["EndpointTOW"], np.full(len(part), 5.0), s=2, color=color, alpha=0.45, label=label)
        target_axis.set_yticks([5], ["target"])

        data["TowBin"] = (data["EndpointTOW"] // args.bin_seconds * args.bin_seconds).astype(int)
        for target, color, label in ((True, "#242424", "target bands"), (False, "#e67e22", "other bands")):
            series = data.loc[data["TargetBand"] == target].groupby("TowBin")["PositiveProbability"].mean()
            probability_axis.plot(series.index, series.values, color=color, linewidth=1.1, label=label)
        probability_axis.set_ylim(-0.03, 1.03)
        probability_axis.set_xlabel("TOW (s)")
        if column == 0:
            alert_axis.set_ylabel("model alert")
            target_axis.set_ylabel("true target")
            probability_axis.set_ylabel("mean P(spoof)")
            alert_axis.legend(loc="upper left", fontsize=8)
            target_axis.legend(loc="upper left", fontsize=8)
            probability_axis.legend(loc="upper left", fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()

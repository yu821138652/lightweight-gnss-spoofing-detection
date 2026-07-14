#!/usr/bin/env python3
"""Build second-level evidence and review plots for dynamic spoofing labels.

This script is deliberately a labeling assistant, not an automatic labeler.  It
aggregates signal-level CSV data by device, frequency band, and TOW, then looks
for short anomalies that occur on multiple devices at the same time.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_INPUT_DIR = Path("data_csv/new_building")
DEFAULT_OUTPUT_DIR = Path("output/dynamic_labeling_review")
TARGET_BANDS = {
    "dy_L1": [1],
    "dy_L5": [5],
    "dy_L_15": [1, 5],
}

RAW_COLUMNS = [
    "TOW",
    "DeviceName",
    "FreqBand",
    "signal_id",
    "Cn0DbHz",
    "Cn0DbHz_dt",
    "Cn0DbHz_std",
    "AgcDb",
    "ReceivedSvTimeUncertaintyNanos",
    "PseudorangeRateUncertaintyMetersPerSecond",
    "AccumulatedDeltaRangeUncertaintyMeters",
]


def find_source_csvs(session_dir: Path) -> list[Path]:
    """Return only the per-log CSVs, excluding compatibility split folders."""
    return sorted(
        path
        for path in session_dir.rglob("*.csv")
        if not any("_by_signal_id" in part or "_by_sv_id" in part for part in path.parts)
    )


def robust_local_z(values: pd.Series, window_seconds: int, minimum_scale: float) -> pd.Series:
    """Return a signed rolling robust z-score without learning a global scale."""
    values = pd.to_numeric(values, errors="coerce")
    min_periods = max(5, window_seconds // 3)
    baseline = values.rolling(window_seconds, center=True, min_periods=min_periods).median()
    residual = (values - baseline).abs()
    local_mad = residual.rolling(window_seconds, center=True, min_periods=min_periods).median()
    local_scale = 1.4826 * local_mad

    global_mad = (values - values.median()).abs().median()
    global_scale = max(1.4826 * global_mad, minimum_scale) if pd.notna(global_mad) else minimum_scale
    # Some receiver fields are quantized or nearly constant. Without this
    # floor, one discrete tick can divide by a near-zero local MAD and create
    # a misleading score in the tens. The local scale still dominates when
    # the short-window variability is genuinely informative.
    scale_floor = max(0.25 * global_scale, minimum_scale)
    scale = local_scale.where(local_scale >= scale_floor, scale_floor)
    return ((values - baseline) / scale).clip(-50, 50)


def summarize_device_band(csv_path: Path, target_bands: Iterable[int], window_seconds: int) -> pd.DataFrame:
    """Create one row per observed second for a device and target frequency band."""
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    usecols = [column for column in RAW_COLUMNS if column in header]
    missing = {"TOW", "FreqBand", "Cn0DbHz", "signal_id"} - set(usecols)
    if missing:
        raise ValueError(f"{csv_path}: missing required columns: {sorted(missing)}")

    df = pd.read_csv(csv_path, usecols=usecols, low_memory=False)
    for column in set(usecols) - {"DeviceName", "signal_id"}:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["TOW", "FreqBand"]).copy()
    df["TOW"] = df["TOW"].round().astype(int)
    df["FreqBand"] = df["FreqBand"].astype(int)
    df = df[df["FreqBand"].isin(target_bands)]
    if df.empty:
        return pd.DataFrame()

    device_name = str(df["DeviceName"].dropna().iloc[0]) if "DeviceName" in df and df["DeviceName"].notna().any() else csv_path.parent.name
    summaries: list[pd.DataFrame] = []
    for band, band_df in df.groupby("FreqBand", sort=True):
        grouped = band_df.groupby("TOW", sort=True)
        summary = grouped.agg(
            Cn0Median=("Cn0DbHz", "median"),
            Cn0AbsDtMedian=("Cn0DbHz_dt", lambda values: values.abs().median()),
            Cn0StdMedian=("Cn0DbHz_std", "median"),
            AgcMedian=("AgcDb", "median"),
            ReceivedSvTimeUncertaintyMedian=("ReceivedSvTimeUncertaintyNanos", "median"),
            PseudorangeRateUncertaintyMedian=("PseudorangeRateUncertaintyMetersPerSecond", "median"),
            AdrUncertaintyMedian=("AccumulatedDeltaRangeUncertaintyMeters", "median"),
            SignalCount=("signal_id", "nunique"),
        ).reset_index()

        # Reindex so rolling windows represent seconds rather than observed rows.
        full_tow = pd.RangeIndex(summary["TOW"].min(), summary["TOW"].max() + 1, name="TOW")
        summary = summary.set_index("TOW").reindex(full_tow).reset_index()
        summary["DeviceName"] = device_name
        summary["FreqBand"] = int(band)
        summary["SeriesID"] = f"{device_name}|L{int(band)}"

        feature_sources = {
            # The minimum scales prevent quantized receiver fields from
            # treating a tiny one-step tick as an extreme anomaly.
            "Cn0Median": ("Cn0Median", 0.5),
            "Cn0AbsDtMedian": ("Cn0AbsDtMedian", 0.5),
            "Cn0StdMedian": ("Cn0StdMedian", 0.25),
            "AgcMedian": ("AgcMedian", 0.5),
            "ReceivedSvTimeUncertaintyMedian": ("ReceivedSvTimeUncertaintyMedian", 0.1),
            "PseudorangeRateUncertaintyMedian": ("PseudorangeRateUncertaintyMedian", 0.1),
            "AdrUncertaintyMedian": ("AdrUncertaintyMedian", 0.05),
            "SignalCount": ("SignalCount", 1.0),
        }
        for feature, (source, minimum_scale) in feature_sources.items():
            values = summary[source]
            if "Uncertainty" in feature:
                values = np.log1p(values.clip(lower=0))
            summary[f"z_{feature}"] = robust_local_z(values, window_seconds, minimum_scale)

        z_columns = [column for column in summary.columns if column.startswith("z_")]
        score_matrix = summary[z_columns].abs().to_numpy(dtype=float)
        valid_count = np.isfinite(score_matrix).sum(axis=1)
        sorted_scores = np.sort(np.where(np.isfinite(score_matrix), score_matrix, -np.inf), axis=1)
        top_two = sorted_scores[:, -2:]
        summary["DeviceScore"] = np.where(valid_count >= 2, np.nanmedian(top_two, axis=1), np.nan)
        summaries.append(summary)

    return pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()


def build_consensus(per_series: pd.DataFrame, threshold: float, min_devices: int) -> pd.DataFrame:
    """Require independent device agreement rather than a single noisy receiver."""
    z_columns = [column for column in per_series.columns if column.startswith("z_")]
    per_series = per_series.copy()
    per_series["AnomalousFeatureCount"] = (per_series[z_columns].abs() >= threshold).sum(axis=1)
    per_series["StrongDeviceEvidence"] = (
        (per_series["DeviceScore"] >= threshold)
        & (per_series["AnomalousFeatureCount"] >= 2)
    )
    consensus = per_series.groupby("TOW", as_index=False).agg(
        SeriesAvailable=("SeriesID", "count"),
        DevicesAvailable=("DeviceName", "nunique"),
        DevicesWithStrongEvidence=("DeviceName", lambda values: values[per_series.loc[values.index, "StrongDeviceEvidence"]].nunique()),
        ScoreMedian=("DeviceScore", "median"),
        ScoreP75=("DeviceScore", lambda values: values.quantile(0.75)),
        Cn0ZMedian=("z_Cn0Median", "median"),
        SignalCountZMedian=("z_SignalCount", "median"),
    )
    consensus["CandidateSecond"] = consensus["DevicesWithStrongEvidence"] >= min_devices
    return consensus, per_series


def build_intervals(consensus: pd.DataFrame, min_duration: int, merge_gap: int) -> pd.DataFrame:
    """Convert candidate seconds to short, reviewable intervals."""
    candidates = consensus[consensus["CandidateSecond"]].sort_values("TOW")
    if candidates.empty:
        return pd.DataFrame(columns=[
            "StartTOW", "EndTOW", "DurationSeconds", "PeakDevicesWithStrongEvidence",
            "PeakScoreP75", "MeanScoreMedian", "CandidateSeconds",
        ])

    groups = (candidates["TOW"].diff().fillna(1) > (merge_gap + 1)).cumsum()
    rows = []
    for _, group in candidates.groupby(groups):
        start, end = int(group["TOW"].min()), int(group["TOW"].max())
        duration = end - start + 1
        if duration < min_duration:
            continue
        rows.append({
            "StartTOW": start,
            "EndTOW": end,
            "DurationSeconds": duration,
            "PeakDevicesWithStrongEvidence": int(group["DevicesWithStrongEvidence"].max()),
            "PeakScoreP75": round(float(group["ScoreP75"].max()), 3),
            "MeanScoreMedian": round(float(group["ScoreMedian"].mean()), 3),
            "CandidateSeconds": int(len(group)),
        })
    return pd.DataFrame(rows).sort_values(["PeakDevicesWithStrongEvidence", "PeakScoreP75"], ascending=False) if rows else pd.DataFrame()


def shade_intervals(axis: plt.Axes, intervals: pd.DataFrame) -> None:
    for _, interval in intervals.iterrows():
        axis.axvspan(interval["StartTOW"], interval["EndTOW"], color="tab:red", alpha=0.12)


def plot_overview(per_series: pd.DataFrame, consensus: pd.DataFrame, intervals: pd.DataFrame,
                  scenario: str, session: str, output_path: Path, threshold: float) -> None:
    """Produce a compact review plot focused on sub-minute events."""
    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True, constrained_layout=True)
    series_ids = sorted(per_series["SeriesID"].unique())
    cmap = plt.get_cmap("tab20", max(1, len(series_ids)))

    for index, series_id in enumerate(series_ids):
        data = per_series[per_series["SeriesID"] == series_id]
        color = cmap(index)
        axes[0].plot(data["TOW"], data["DeviceScore"], linewidth=0.8, alpha=0.85, color=color, label=series_id)
        axes[2].plot(data["TOW"], data["z_Cn0Median"], linewidth=0.7, alpha=0.8, color=color, label=series_id)
        axes[3].plot(data["TOW"], data["z_SignalCount"], linewidth=0.7, alpha=0.8, color=color, label=series_id)

    axes[0].axhline(threshold, color="black", linestyle="--", linewidth=0.8, label="strong-evidence threshold")
    axes[0].set_ylabel("device anomaly score")
    display_session = session.replace("新主楼", "new_building")
    axes[0].set_title(f"{scenario} | {display_session} | dynamic-label review evidence")
    axes[0].legend(ncol=4, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.17))

    axes[1].plot(consensus["TOW"], consensus["DevicesWithStrongEvidence"], color="tab:red", linewidth=1.0, label="strong devices")
    axes[1].plot(consensus["TOW"], consensus["DevicesAvailable"], color="tab:gray", linewidth=0.8, alpha=0.8, label="available devices")
    axes[1].set_ylabel("device count")
    axes[1].legend(fontsize=8, loc="upper right")

    axes[2].axhline(0, color="black", linewidth=0.5)
    axes[2].set_ylabel("C/N0 local z")
    axes[3].axhline(0, color="black", linewidth=0.5)
    axes[3].set_ylabel("signal-count local z")
    axes[3].set_xlabel("TOW (s)")

    for axis in axes:
        shade_intervals(axis, intervals)
        axis.grid(alpha=0.18, linewidth=0.5)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, facecolor="white")
    plt.close(fig)


def process_session(session_dir: Path, scenario: str, output_root: Path, args: argparse.Namespace) -> pd.DataFrame:
    session = session_dir.name
    target_bands = TARGET_BANDS[scenario]
    frames = []
    for csv_path in find_source_csvs(session_dir):
        frame = summarize_device_band(csv_path, target_bands, args.window_seconds)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        print(f"[skip] {scenario}/{session}: no usable target-band rows")
        return pd.DataFrame()

    consensus, per_series = build_consensus(pd.concat(frames, ignore_index=True), args.threshold, args.min_devices)
    intervals = build_intervals(consensus, args.min_duration, args.merge_gap)
    output_dir = output_root / scenario / session
    output_dir.mkdir(parents=True, exist_ok=True)
    per_series.to_csv(output_dir / "per_second_device_band_evidence.csv", index=False, encoding="utf-8-sig")
    consensus.to_csv(output_dir / "per_second_consensus.csv", index=False, encoding="utf-8-sig")
    intervals.to_csv(output_dir / "candidate_intervals.csv", index=False, encoding="utf-8-sig")
    plot_overview(per_series, consensus, intervals, scenario, session, output_dir / "overview.png", args.threshold)

    if intervals.empty:
        print(f"[done] {scenario}/{session}: no candidates meeting the current threshold")
        return intervals
    intervals.insert(0, "Session", session)
    intervals.insert(0, "Scenario", scenario)
    print(f"[done] {scenario}/{session}: {len(intervals)} candidate interval(s)")
    return intervals


def main() -> None:
    parser = argparse.ArgumentParser(description="Build short-event dynamic spoofing labeling assistance outputs.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="When: review dynamic new-building CSVs. Why: source of per-log extracted features.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="When: keep review artifacts separate. Why: never overwrite training CSVs or label config.")
    parser.add_argument("--scenario", choices=sorted(TARGET_BANDS), help="When: assign one scenario to a reviewer. Why: limits output to the owned dynamic scenario.")
    parser.add_argument("--session", help="When: recheck one named Session. Why: avoids regenerating other review outputs.")
    parser.add_argument("--window-seconds", type=int, default=31, help="Rolling baseline width in seconds; use an odd value larger than expected short events.")
    parser.add_argument("--threshold", type=float, default=2.5, help="Robust anomaly threshold per device-band series.")
    parser.add_argument("--min-devices", type=int, default=2, help="Minimum agreeing devices; protects against one receiver's motion or dropout.")
    parser.add_argument("--min-duration", type=int, default=3, help="Minimum candidate duration in seconds; deliberately supports short dynamic events.")
    parser.add_argument("--merge-gap", type=int, default=1, help="Merge candidate runs separated by at most this many non-candidate seconds.")
    args = parser.parse_args()
    if args.window_seconds < 7 or args.window_seconds % 2 == 0:
        parser.error("--window-seconds must be an odd integer >= 7")

    scenarios = [args.scenario] if args.scenario else sorted(TARGET_BANDS)
    all_intervals = []
    for scenario in scenarios:
        scenario_dir = args.input_dir / scenario
        if not scenario_dir.exists():
            print(f"[skip] scenario directory not found: {scenario_dir}")
            continue
        sessions = [path for path in sorted(scenario_dir.iterdir()) if path.is_dir()]
        if args.session:
            sessions = [path for path in sessions if path.name == args.session]
            if not sessions:
                print(f"[skip] {scenario}: session not found: {args.session}")
        for session_dir in sessions:
            intervals = process_session(session_dir, scenario, args.output_dir, args)
            if not intervals.empty:
                all_intervals.append(intervals)

    summary_path = args.output_dir / "candidate_interval_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if all_intervals:
        pd.concat(all_intervals, ignore_index=True).to_csv(summary_path, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame(columns=["Scenario", "Session", "StartTOW", "EndTOW"]).to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"[summary] {summary_path}")
    print("Candidates are review aids only. Confirm against the overview and raw feature plots before writing YAML labels.")


if __name__ == "__main__":
    main()

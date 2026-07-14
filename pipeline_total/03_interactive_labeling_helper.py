#!/usr/bin/env python3
"""Interactively select exact TOW intervals from plot-feature CSV files.

The tool reads the current ``*-plot_features.csv`` files and displays a matrix:
devices are columns, labeling features are rows, and every subplot shares one
TOW axis. The mouse position snaps to the nearest TOW that really exists in the
selected recording.

Examples:
    python pipeline_total/03_interactive_labeling_helper.py \
      --input-base data_raw/new_building \
      --scenario dy_L1 \
      --session 2025.07.29.20.04_新主楼

    python pipeline_total/03_interactive_labeling_helper.py \
      --csv data_raw/new_building/dy_L1/.../device/log-plot_features.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backend_bases import MouseButton


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAX_GAP_SECONDS = 10

plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

FEATURE_LABELS = {
    "Cn0DbHz": "C/N0 (dB-Hz)",
    "Cn0DbHz_dt": "C/N0 Change",
    "Cn0DbHz_std": "C/N0 Std Dev",
    "AgcDb": "AGC (dB)",
    "ReceivedSvTimeUncertaintyNanos": "Time Uncertainty (ns)",
    "PseudorangeRateUncertaintyMetersPerSecond": "PR Rate Uncertainty (m/s)",
    "AccumulatedDeltaRangeUncertaintyMeters": "ADR Uncertainty (m)",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")


@dataclass
class PlotDataset:
    path: Path
    frame: pd.DataFrame
    device: str


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def format_tow(value: float) -> str:
    if np.isclose(value, round(value), atol=1e-9):
        return str(int(round(value)))
    return f"{value:.3f}".rstrip("0").rstrip(".")


def find_plot_csvs(
    csv_paths: Iterable[str] | None,
    input_base: str,
    scenario: str | None,
    session: str | None,
) -> list[Path]:
    if csv_paths:
        paths = [resolve_path(path) for path in csv_paths]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"CSV files not found: {missing}")
        return paths

    if not scenario or not session:
        raise ValueError("--scenario and --session are required unless --csv is used")

    base = resolve_path(input_base)
    scenario_dir = base / scenario
    if not scenario_dir.is_dir():
        raise FileNotFoundError(f"Scenario directory not found: {scenario_dir}")

    matches = [
        path
        for path in scenario_dir.rglob("*-plot_features.csv")
        if session in path.parts
    ]
    return sorted(matches, key=lambda path: str(path).lower())


def load_datasets(paths: list[Path], device_filter: str | None) -> list[PlotDataset]:
    datasets: list[PlotDataset] = []
    filter_text = device_filter.lower() if device_filter else None

    for path in paths:
        frame = pd.read_csv(path)
        required = {"TOW"}
        identity_column = "SignalID" if "SignalID" in frame.columns else "SatelliteID"
        required.add(identity_column)
        missing = sorted(required.difference(frame.columns))
        if missing:
            logging.warning("Skipping %s: missing columns %s", path, missing)
            continue

        device = path.parent.name
        if "DeviceName" in frame.columns and not frame.empty:
            device = str(frame["DeviceName"].iloc[0])

        if filter_text and filter_text not in device.lower() and filter_text not in path.parent.name.lower():
            continue

        frame = frame.copy()
        frame["TOW"] = pd.to_numeric(frame["TOW"], errors="coerce")
        for feature in FEATURE_LABELS:
            if feature in frame.columns:
                frame[feature] = pd.to_numeric(frame[feature], errors="coerce")
        frame = frame.dropna(subset=["TOW"])
        if frame.empty:
            logging.warning("Skipping %s: no numeric TOW values", path)
            continue

        datasets.append(PlotDataset(path=path, frame=frame, device=device))

    return datasets


def plot_with_gaps(ax, x: np.ndarray, y: np.ndarray, color, label: str | None) -> None:
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < 2:
        return

    gap_indices = np.where(np.diff(x) > MAX_GAP_SECONDS)[0] + 1
    segments = np.split(np.arange(len(x)), gap_indices)
    for index, segment in enumerate(segments):
        if len(segment) < 2:
            continue
        ax.plot(
            x[segment],
            y[segment],
            color=color,
            alpha=0.72,
            linewidth=0.75,
            label=label if index == 0 else None,
        )


def extract_metadata(datasets: list[PlotDataset]) -> dict[str, str]:
    frame = datasets[0].frame

    def first_value(column: str, fallback: str = "unknown") -> str:
        if column not in frame.columns or frame.empty:
            return fallback
        value = frame[column].iloc[0]
        return fallback if pd.isna(value) else str(value)

    return {
        "environment": first_value("Environment"),
        "scenario": first_value("Scenario"),
        "session": first_value("Session"),
    }


class TowIntervalSelector:
    def __init__(
        self,
        figure,
        axes,
        tow_values: np.ndarray,
        metadata: dict[str, str],
        datasets: list[PlotDataset],
        candidate_output: Path | None,
    ) -> None:
        self.figure = figure
        self.axes = list(np.atleast_1d(axes))
        self.tow_values = np.asarray(tow_values, dtype=float)
        self.metadata = metadata
        self.features = list(FEATURE_LABELS)
        self.datasets = datasets
        self.candidate_output = candidate_output
        self.boundaries: list[float] = []
        self.history: list[list[float]] = []
        self.selection_artists = []
        self.last_saved_signature: tuple[tuple[float, float], ...] | None = None

        self.cursor_lines = [
            axis.axvline(
                self.tow_values[0],
                color="black",
                linestyle=":",
                linewidth=0.9,
                alpha=0.75,
                visible=False,
            )
            for axis in self.axes
        ]
        self.status_text = figure.text(
            0.01,
            0.012,
            "TOW: move the mouse over a plot",
            ha="left",
            va="bottom",
            fontsize=9,
        )

        figure.canvas.mpl_connect("motion_notify_event", self.on_motion)
        figure.canvas.mpl_connect("button_press_event", self.on_click)
        figure.canvas.mpl_connect("key_press_event", self.on_key)

    def snap_tow(self, x_value: float) -> float:
        index = int(np.searchsorted(self.tow_values, x_value))
        if index <= 0:
            return float(self.tow_values[0])
        if index >= len(self.tow_values):
            return float(self.tow_values[-1])
        before = self.tow_values[index - 1]
        after = self.tow_values[index]
        return float(before if abs(x_value - before) <= abs(after - x_value) else after)

    def completed_intervals(self) -> list[tuple[float, float]]:
        if len(self.boundaries) < 2:
            return []
        return [(self.boundaries[0], self.boundaries[1])]

    def update_status(self, exact_x: float, snapped_tow: float) -> None:
        if not self.boundaries:
            click_action = "set start"
        elif len(self.boundaries) == 1:
            click_action = "set end"
        else:
            nearest_index = int(
                np.argmin([abs(snapped_tow - boundary) for boundary in self.boundaries])
            )
            click_action = "adjust start" if nearest_index == 0 else "adjust end"
        self.status_text.set_text(
            f"cursor={exact_x:.3f} s | nearest TOW={format_tow(snapped_tow)} | click={click_action}"
        )

    def on_motion(self, event) -> None:
        if event.inaxes not in self.axes or event.xdata is None:
            return
        snapped = self.snap_tow(float(event.xdata))
        for line in self.cursor_lines:
            line.set_xdata([snapped, snapped])
            line.set_visible(True)
        self.update_status(float(event.xdata), snapped)
        self.figure.canvas.draw_idle()

    def on_click(self, event) -> None:
        if event.inaxes not in self.axes:
            return

        toolbar = getattr(self.figure.canvas.manager, "toolbar", None)
        if toolbar is not None and getattr(toolbar, "mode", ""):
            return

        if event.button == MouseButton.RIGHT:
            self.undo()
            return

        if event.button != MouseButton.LEFT or event.xdata is None:
            return

        snapped = self.snap_tow(float(event.xdata))
        self.history.append(self.boundaries.copy())

        if not self.boundaries:
            self.boundaries = [snapped]
            boundary_kind = "start"
        elif len(self.boundaries) == 1:
            self.boundaries.append(snapped)
            self.boundaries.sort()
            boundary_kind = "end"
        else:
            nearest_index = int(
                np.argmin([abs(snapped - boundary) for boundary in self.boundaries])
            )
            self.boundaries[nearest_index] = snapped
            self.boundaries.sort()
            boundary_kind = "start" if np.isclose(self.boundaries[0], snapped) else "end"

        logging.info("Set %s TOW: %s", boundary_kind, format_tow(snapped))
        self.redraw_selections()

    def clear_selection_artists(self) -> None:
        for artist in self.selection_artists:
            try:
                artist.remove()
            except ValueError:
                pass
        self.selection_artists.clear()

    def redraw_selections(self) -> None:
        self.clear_selection_artists()
        intervals = self.completed_intervals()

        for axis in self.axes:
            if intervals:
                start, end = intervals[0]
                self.selection_artists.append(
                    axis.axvspan(start, end, color="tab:red", alpha=0.14)
                )
                self.selection_artists.append(
                    axis.axvline(start, color="tab:green", linestyle="--", linewidth=1.3)
                )
                self.selection_artists.append(
                    axis.axvline(end, color="tab:red", linestyle="--", linewidth=1.3)
                )

            elif len(self.boundaries) == 1:
                self.selection_artists.append(
                    axis.axvline(
                        self.boundaries[0],
                        color="tab:green",
                        linestyle="--",
                        linewidth=1.3,
                    )
                )

        interval_text = ", ".join(
            f"[{format_tow(start)}, {format_tow(end)}]" for start, end in intervals
        ) or "none"
        if len(self.boundaries) == 1:
            interval_text += f" | start={format_tow(self.boundaries[0])}, waiting for end"
        self.figure.suptitle(
            f"{self.metadata['scenario']} / {self.metadata['session']}\n"
            f"candidate intervals: {interval_text}",
            fontsize=11,
        )
        self.figure.canvas.draw_idle()

    def undo(self) -> None:
        if not self.history:
            return
        self.boundaries = self.history.pop()
        logging.info("Restored previous interval selection")
        self.redraw_selections()

    def reset(self) -> None:
        if self.boundaries:
            self.history.append(self.boundaries.copy())
        self.boundaries.clear()
        logging.info("Cleared all selected boundaries")
        self.redraw_selections()

    def print_intervals(self) -> None:
        intervals = self.completed_intervals()
        if len(self.boundaries) == 1:
            logging.warning("The start boundary has no matching end boundary and was not exported")

        print("\nCandidate interval block (review before setting status=reviewed):")
        print(f'"{self.metadata["session"]}":')
        print("  status: reviewed")
        print("  intervals:")
        if not intervals:
            print("    []")
        else:
            for start, end in intervals:
                print(f"    - [{format_tow(start)}, {format_tow(end)}]")
        print()

        if self.candidate_output and intervals:
            self.save_candidates(intervals)

    def save_candidates(self, intervals: list[tuple[float, float]]) -> None:
        signature = tuple(intervals)
        if signature == self.last_saved_signature:
            logging.info("Candidate intervals are unchanged; not appending a duplicate record")
            return

        output_path = self.candidate_output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "created_at",
            "environment",
            "scenario",
            "session",
            "feature",
            "start_tow",
            "end_tow",
            "devices",
            "source_csvs",
            "review_status",
        ]
        exists = output_path.exists()
        devices = ";".join(dataset.device for dataset in self.datasets)
        source_csvs = ";".join(str(dataset.path) for dataset in self.datasets)
        with output_path.open("a", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if not exists:
                writer.writeheader()
            for start, end in intervals:
                writer.writerow(
                    {
                        "created_at": datetime.now().isoformat(timespec="seconds"),
                        "environment": self.metadata["environment"],
                        "scenario": self.metadata["scenario"],
                        "session": self.metadata["session"],
                        "feature": ";".join(self.features),
                        "start_tow": format_tow(start),
                        "end_tow": format_tow(end),
                        "devices": devices,
                        "source_csvs": source_csvs,
                        "review_status": "candidate",
                    }
                )
        self.last_saved_signature = signature
        logging.info("Appended candidate intervals to %s", output_path)

    def on_key(self, event) -> None:
        if event.key in {"u", "backspace"}:
            self.undo()
        elif event.key == "r":
            self.reset()
        elif event.key in {"enter", "return"}:
            self.print_intervals()
        elif event.key in {"q", "escape"}:
            plt.close(self.figure)


def create_interactive_plot(
    datasets: list[PlotDataset],
    show_legend: bool,
    candidate_output: Path | None,
):
    features = list(FEATURE_LABELS.items())
    row_count = len(features)
    column_count = len(datasets)
    figure_width = max(13.0, 4.8 * column_count)
    figure_height = max(12.0, 2.05 * row_count)
    figure, axes = plt.subplots(
        row_count,
        column_count,
        sharex=True,
        figsize=(figure_width, figure_height),
        squeeze=False,
    )

    all_tows = []
    for column_index, dataset in enumerate(datasets):
        frame = dataset.frame
        identity_column = "SignalID" if "SignalID" in frame.columns else "SatelliteID"
        identities = sorted(str(value) for value in frame[identity_column].dropna().unique())
        colors = plt.cm.turbo(np.linspace(0, 1, max(1, len(identities))))
        color_by_identity = dict(zip(identities, colors))
        signal_frames = {
            identity: frame[frame[identity_column].astype(str) == identity]
            for identity in identities
        }

        all_tows.extend(frame["TOW"].dropna().astype(float).tolist())

        for row_index, (feature, feature_label) in enumerate(features):
            axis = axes[row_index, column_index]
            if row_index == 0:
                axis.set_title(dataset.device, fontsize=10)

            if feature not in frame.columns or not frame[feature].notna().any():
                axis.text(
                    0.5,
                    0.5,
                    "No data",
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                    color="0.45",
                    fontsize=9,
                )
            else:
                for identity in identities:
                    signal_frame = signal_frames[identity]
                    sort_columns = ["TOW"] + (
                        ["TimeNanos"] if "TimeNanos" in signal_frame.columns else []
                    )
                    signal_frame = signal_frame.sort_values(sort_columns, kind="mergesort")
                    x_values = signal_frame["TOW"].to_numpy(dtype=float)
                    y_values = signal_frame[feature].to_numpy(dtype=float)
                    plot_with_gaps(
                        axis,
                        x_values,
                        y_values,
                        color=color_by_identity[identity],
                        label=identity,
                    )

            if column_index == 0:
                axis.set_ylabel(feature_label, fontsize=8)
            axis.grid(True, alpha=0.22)
            axis.tick_params(axis="both", labelsize=7)
            if show_legend and row_index == 0 and identities:
                axis.legend(fontsize=4.5, ncol=4, loc="upper left", framealpha=0.8)

            if row_index == row_count - 1:
                axis.set_xlabel("TOW (s)")

    metadata = extract_metadata(datasets)
    figure.suptitle(
        f"{metadata['scenario']} / {metadata['session']}",
        fontsize=11,
    )
    figure.subplots_adjust(
        left=0.07,
        right=0.99,
        top=0.93,
        bottom=0.055,
        hspace=0.23,
        wspace=0.13,
    )

    try:
        figure.canvas.manager.set_window_title(
            f"TOW Labeling - {metadata['scenario']} - {metadata['session']}"
        )
    except AttributeError:
        pass

    selector = TowIntervalSelector(
        figure=figure,
        axes=axes.ravel(),
        tow_values=np.unique(np.asarray(all_tows, dtype=float)),
        metadata=metadata,
        datasets=datasets,
        candidate_output=candidate_output,
    )
    return figure, selector


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", nargs="+", help="One or more explicit *-plot_features.csv files.")
    parser.add_argument(
        "--input-base",
        default="data_raw/new_building",
        help="Directory containing scenario folders. Default: data_raw/new_building",
    )
    parser.add_argument("--scenario", help="Scenario directory, for example dy_L1.")
    parser.add_argument("--session", help="Exact session directory name.")
    parser.add_argument(
        "--device",
        help="Optional case-insensitive device name/folder filter. Omit to compare all devices.",
    )
    parser.add_argument(
        "--show-legend",
        action="store_true",
        help="Show signal legends on the C/N0 row. Hidden by default to keep the matrix readable.",
    )
    parser.add_argument(
        "--candidate-output",
        type=Path,
        default=None,
        help="Optional CSV that receives candidate intervals when Enter is pressed.",
    )
    parser.add_argument("--save-preview", type=Path, default=None, help="Optionally save the initial figure.")
    parser.add_argument("--list-only", action="store_true", help="List matched plot-feature CSV files and exit.")
    parser.add_argument("--no-show", action="store_true", help="Build the figure without opening a GUI window.")
    args = parser.parse_args()

    try:
        paths = find_plot_csvs(args.csv, args.input_base, args.scenario, args.session)
        datasets = load_datasets(paths, args.device)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    if not datasets:
        parser.error("No matching plot-feature CSV files were found")

    logging.info("Matched %d device CSV files", len(datasets))
    for dataset in datasets:
        logging.info("  %s -> %s", dataset.device, dataset.path)

    if args.list_only:
        return

    candidate_output = resolve_path(args.candidate_output) if args.candidate_output else None
    figure, _selector = create_interactive_plot(
        datasets=datasets,
        show_legend=args.show_legend,
        candidate_output=candidate_output,
    )

    if args.save_preview:
        preview_path = resolve_path(args.save_preview)
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(preview_path, dpi=150, bbox_inches="tight")
        logging.info("Saved preview to %s", preview_path)

    print("\nControls:")
    print("  mouse move    show exact cursor x and nearest real TOW")
    print("  left click    set start/end; after two clicks, adjust the nearest boundary")
    print("  right click   undo the latest boundary")
    print("  Enter         print intervals and optionally append candidate CSV")
    print("  r             clear all boundaries")
    print("  q / Esc       close the window")
    print("  toolbar       zoom and pan before clicking\n")

    if args.no_show:
        plt.close(figure)
    else:
        plt.show()


if __name__ == "__main__":
    main()

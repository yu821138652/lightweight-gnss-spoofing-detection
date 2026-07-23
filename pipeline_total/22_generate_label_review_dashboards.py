"""Generate session-level, satellite-signal label-review dashboards.

``02_batch_plot_feature_images.py`` creates one image per feature and device.
This tool instead reads the mirrored per-log CSV files in ``data_csv/``,
resolves the same formal Session-level labels from
``configs/preprocessing.yml``, and creates one aligned dashboard for each
complete recording. It is intended for manual label review, not for model
evaluation.

Each dashboard contains a label timeline and all seven available GNSS Raw
features for every device in the session.  A review index flags stale mirrored
CSVs whose existing ``Label`` values do not match the current configuration.
"""

from __future__ import annotations

import argparse
import html
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.collections import LineCollection
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger(__name__)
MAX_GAP_SECONDS = 10.0

plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

FEATURES = [
    ("Cn0DbHz", "C/N0 (dB-Hz)"),
    ("Cn0DbHz_dt", "C/N0 change"),
    ("Cn0DbHz_std", "C/N0 rolling std"),
    ("AgcDb", "AGC (dB)"),
    ("ReceivedSvTimeUncertaintyNanos", "SV time uncertainty (ns)"),
    ("PseudorangeRateUncertaintyMetersPerSecond", "PR rate uncertainty (m/s)"),
    ("AccumulatedDeltaRangeUncertaintyMeters", "ADR uncertainty (m)"),
]
METADATA_COLUMNS = [
    "Environment",
    "Scenario",
    "Session",
    "DeviceName",
    "TOW",
    "TimeNanos",
    "signal_id",
    "SignalBand",
    "FreqBand",
    "Label",
    "LabelStatus",
    "LabelSource",
]


@dataclass(frozen=True)
class SessionIdentity:
    environment: str
    scenario: str
    session: str


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def safe_path_component(value: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', "_", str(value)).strip(". ") or "unknown"


def discover_mirrored_csvs(input_dir: Path) -> list[Path]:
    """Return per-log mirror CSVs only, excluding per-signal splits."""

    paths: list[Path] = []
    for pattern in ("gnss_log_*.csv", "log_mimir_*.csv"):
        paths.extend(input_dir.rglob(pattern))
    return sorted(set(paths), key=lambda path: str(path).lower())


def read_identity(path: Path) -> SessionIdentity | None:
    try:
        header = pd.read_csv(
            path,
            usecols=lambda name: name in {"Environment", "Scenario", "Session"},
            nrows=1,
            encoding="utf-8-sig",
        )
    except (OSError, UnicodeDecodeError, pd.errors.ParserError) as error:
        LOG.warning("Skipping unreadable CSV %s: %s", path, error)
        return None
    required = {"Environment", "Scenario", "Session"}
    if header.empty or not required.issubset(header.columns):
        LOG.warning("Skipping %s: missing recording metadata", path)
        return None
    values = header.iloc[0]
    return SessionIdentity(*(str(values[column]) for column in ("Environment", "Scenario", "Session")))


def resolve_formal_label(identity: SessionIdentity, config: dict) -> tuple[list[tuple[float, float]], str, str]:
    """Resolve the one formal Environment/Scenario/Session label entry."""

    labeling = config.get("labeling", {})
    session_entry = (
        labeling.get("session_spoofing_tow_intervals", {})
        .get(identity.environment, {})
        .get(identity.scenario, {})
        .get(identity.session)
    )
    if session_entry is None:
        return [], "needs_review", "missing_session_config"
    if not isinstance(session_entry, dict):
        raise ValueError(
            "Session label entries must be mappings with status and intervals: "
            f"{identity.environment}/{identity.scenario}/{identity.session}"
        )
    intervals = session_entry.get("intervals", []) or []
    status = str(session_entry.get("status", "needs_review"))
    return [(float(start), float(end)) for start, end in intervals], status, "session_config"


def target_bands(scenario: str, config: dict) -> set[int]:
    label_value = config.get("labeling", {}).get("spoofing_type_to_label", {}).get(scenario, 0)
    if label_value == 1:
        return {1}
    if label_value == 2:
        return {5}
    if label_value == 3:
        return {1, 5}
    return set()


def expected_labels(frame: pd.DataFrame, intervals: list[tuple[float, float]], bands: set[int]) -> np.ndarray:
    expected = np.zeros(len(frame), dtype=np.int8)
    if not intervals or not bands:
        return expected
    tow = pd.to_numeric(frame["TOW"], errors="coerce").to_numpy(dtype=float)
    band = pd.to_numeric(frame["FreqBand"], errors="coerce").to_numpy(dtype=float)
    in_interval = np.zeros(len(frame), dtype=bool)
    for start, end in intervals:
        in_interval |= (tow >= start) & (tow <= end)
    expected[in_interval & np.isin(band, list(bands))] = 1
    return expected


def intervals_in_session(
    intervals: list[tuple[float, float]], tow_min: float, tow_max: float
) -> list[tuple[float, float]]:
    """Keep only configured intervals that can affect this recording's rows."""

    return [(start, end) for start, end in intervals if start <= tow_max and end >= tow_min]


def load_session(paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    requested_columns = [*METADATA_COLUMNS, *(column for column, _ in FEATURES)]
    for path in paths:
        try:
            header = pd.read_csv(path, nrows=0, encoding="utf-8-sig")
            usecols = [column for column in requested_columns if column in header.columns]
            frame = pd.read_csv(path, usecols=usecols, encoding="utf-8-sig")
        except (OSError, UnicodeDecodeError, pd.errors.ParserError) as error:
            raise ValueError(f"Unable to read {path}: {error}") from error
        missing = {"DeviceName", "TOW", "signal_id", "FreqBand", "Label"}.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing required review columns: {sorted(missing)}")
        frame["SourceCsv"] = path.name
        frames.append(frame)
    if not frames:
        raise ValueError("No mirrored CSVs supplied for session")
    result = pd.concat(frames, ignore_index=True)
    for column in ["TOW", "TimeNanos", "FreqBand", "Label", *(name for name, _ in FEATURES)]:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    result["signal_id"] = result["signal_id"].astype(str)
    result["DeviceName"] = result["DeviceName"].astype(str)
    result = result.dropna(subset=["TOW"])
    # Some recordings contain two source logs from one named device.  Prefix
    # those signal IDs so plot lines cannot connect across distinct log files.
    source_counts = result.groupby("DeviceName", sort=False)["SourceCsv"].transform("nunique")
    result["plot_signal_id"] = result["signal_id"]
    duplicate_device_log = source_counts > 1
    result.loc[duplicate_device_log, "plot_signal_id"] = (
        result.loc[duplicate_device_log, "SourceCsv"].astype(str)
        + " | "
        + result.loc[duplicate_device_log, "signal_id"].astype(str)
    )
    return result


def split_line_segments(x: np.ndarray, y: np.ndarray) -> list[np.ndarray]:
    """Return continuous polylines, preserving deliberate gaps in one signal."""

    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 2:
        return []
    parts = np.split(np.arange(len(x)), np.where(np.diff(x) > MAX_GAP_SECONDS)[0] + 1)
    return [np.column_stack((x[part], y[part])) for part in parts if len(part) > 1]


def draw_label_timeline(axis, device_frame: pd.DataFrame, intervals: list[tuple[float, float]], bands: set[int]) -> None:
    axis.set_ylim(0.4, 5.6)
    axis.set_yticks([1, 5], ["L1", "L5"])
    axis.grid(axis="x", alpha=0.22)
    axis.grid(axis="y", alpha=0.16)
    for start, end in intervals:
        axis.axvspan(start, end, color="tab:red", alpha=0.13, zorder=0)
        for band in bands:
            axis.broken_barh([(start, end - start)], (band - 0.28, 0.56), facecolors="tab:red", alpha=0.78)
    observed = device_frame.loc[device_frame["Label"].fillna(0).astype(int) > 0]
    for band, group in observed.groupby("FreqBand", sort=True):
        if band not in {1, 5}:
            continue
        axis.scatter(group["TOW"], np.full(len(group), band), marker="|", s=10, color="black", alpha=0.32)
    axis.set_ylabel("Formal label", fontsize=8)


def draw_feature(axis, device_frame: pd.DataFrame, feature: str, intervals: list[tuple[float, float]], target_signals: set[str]) -> None:
    for start, end in intervals:
        axis.axvspan(start, end, color="tab:red", alpha=0.13, zorder=0)
        axis.axvline(start, color="tab:red", linestyle="--", linewidth=0.55, alpha=0.7)
        axis.axvline(end, color="tab:red", linestyle="--", linewidth=0.55, alpha=0.7)
    if feature not in device_frame.columns or not device_frame[feature].notna().any():
        axis.text(0.5, 0.5, "No data", transform=axis.transAxes, ha="center", va="center", color="0.45")
        return
    signals = sorted(device_frame["plot_signal_id"].dropna().unique())
    colors = plt.cm.turbo(np.linspace(0, 1, max(1, len(signals))))
    normal_segments: list[np.ndarray] = []
    normal_colors: list[np.ndarray] = []
    target_segments: list[np.ndarray] = []
    target_colors: list[np.ndarray] = []
    for signal, color in zip(signals, colors):
        group = device_frame.loc[device_frame["plot_signal_id"] == signal].sort_values(["TOW", "TimeNanos"], kind="mergesort")
        segments = split_line_segments(
            group["TOW"].to_numpy(dtype=float),
            group[feature].to_numpy(dtype=float),
        )
        if signal in target_signals:
            target_segments.extend(segments)
            target_colors.extend([color] * len(segments))
        else:
            normal_segments.extend(segments)
            normal_colors.extend([color] * len(segments))
    if normal_segments:
        axis.add_collection(
            LineCollection(normal_segments, colors=normal_colors, linewidths=0.55, alpha=0.45)
        )
    if target_segments:
        axis.add_collection(
            LineCollection(target_segments, colors=target_colors, linewidths=0.85, alpha=0.90)
        )
    axis.autoscale_view()
    axis.grid(True, alpha=0.22)


def write_signal_summary(frame: pd.DataFrame, output_path: Path) -> None:
    summary = (
        frame.groupby(["DeviceName", "SourceCsv", "FreqBand", "SignalBand", "signal_id"], dropna=False, sort=True)
        .agg(
            rows=("Label", "size"),
            positive_rows=("Label", "sum"),
            tow_start=("TOW", "min"),
            tow_end=("TOW", "max"),
            cn0_median=("Cn0DbHz", "median"),
            agc_missing_rate=("AgcDb", lambda values: float(values.isna().mean())),
        )
        .reset_index()
    )
    summary.to_csv(output_path, index=False, encoding="utf-8-sig")


def render_dashboard(
    identity: SessionIdentity,
    frame: pd.DataFrame,
    intervals: list[tuple[float, float]],
    status: str,
    source: str,
    output_path: Path,
    dpi: int,
) -> None:
    devices = sorted(frame["DeviceName"].dropna().unique())
    tow_min = float(frame["TOW"].min())
    tow_max = float(frame["TOW"].max())
    displayed_intervals = intervals_in_session(intervals, tow_min, tow_max)
    padding = max(1.0, (tow_max - tow_min) * 0.015)
    x_limits = (tow_min - padding, tow_max + padding)
    rows, columns = len(FEATURES) + 1, len(devices)
    figure, axes = plt.subplots(
        rows,
        columns,
        sharex="col",
        squeeze=False,
        figsize=(max(14, 4.2 * columns), max(17, 2.0 * rows)),
    )
    bands = target_bands(identity.scenario, _CONFIG)
    label_text = ", ".join(f"[{start:g}, {end:g}]" for start, end in displayed_intervals) or "none in this recording"
    figure.suptitle(
        f"Label review | {identity.environment} / {identity.scenario} / {identity.session}\n"
        f"applicable formal label: {label_text} | status={status} | source={source} | target bands={sorted(bands) or 'none'}",
        fontsize=12,
    )
    for column, device in enumerate(devices):
        device_frame = frame.loc[frame["DeviceName"] == device].copy()
        target_signals = set(
            device_frame.loc[device_frame["Label"].fillna(0).astype(int) > 0, "plot_signal_id"].astype(str)
        )
        axes[0, column].set_title(
            f"{device}\n{device_frame['plot_signal_id'].nunique()} plotted signals, {len(device_frame):,} rows",
            fontsize=9,
        )
        draw_label_timeline(axes[0, column], device_frame, displayed_intervals, bands)
        for row, (feature, label) in enumerate(FEATURES, start=1):
            axis = axes[row, column]
            draw_feature(axis, device_frame, feature, displayed_intervals, target_signals)
            if column == 0:
                axis.set_ylabel(label, fontsize=8)
            axis.tick_params(axis="both", labelsize=7)
        for axis in axes[:, column]:
            axis.set_xlim(*x_limits)
        axes[-1, column].set_xlabel("TOW (s)", fontsize=8)
    figure.text(
        0.01,
        0.008,
        "Red span: formal session-level attack interval. Red label bar: affected frequency. "
        "Black ticks: rows currently labeled positive. Colored curves: independent signal_id; stronger curves contain positive rows.",
        ha="left",
        va="bottom",
        fontsize=7.5,
    )
    figure.subplots_adjust(left=0.065, right=0.99, top=0.89, bottom=0.045, hspace=0.28, wspace=0.16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def build_index_html(records: list[dict], output_dir: Path) -> None:
    rows: list[str] = []
    for record in records:
        image = quote(record["dashboard_rel"].replace("\\", "/"))
        signals = quote(record["signals_rel"].replace("\\", "/"))
        warning = " label mismatch" if record["label_mismatch_rows"] else ""
        rows.append(
            "<tr>"
            f"<td>{html.escape(record['environment'])}</td>"
            f"<td>{html.escape(record['scenario'])}</td>"
            f"<td>{html.escape(record['session'])}</td>"
            f"<td>{html.escape(record['label_status'])}</td>"
            f"<td>{html.escape(record['label_source'])}</td>"
            f"<td>{html.escape(record['formal_intervals'])}</td>"
            f"<td>{html.escape(record['review_priority'])}</td>"
            f"<td>{record['device_count']}</td>"
            f"<td>{record['positive_rows']} / {record['expected_positive_rows']}</td>"
            f"<td class='{warning.strip()}'>{record['label_mismatch_rows']} "
            f"(CSV-only {record['csv_only_positive_rows']}, missing {record['missing_positive_rows']})</td>"
            f"<td><a href='{image}'>dashboard</a> | <a href='{signals}'>signals</a></td>"
            "</tr>"
        )
    document = """<!doctype html>
<html lang=\"en\"><head><meta charset=\"utf-8\"><title>GNSS label review</title>
<style>
body { font-family: Arial, 'Microsoft YaHei', sans-serif; margin: 24px; color: #18212b; }
table { border-collapse: collapse; width: 100%; font-size: 14px; }
th, td { border: 1px solid #c7d0d9; padding: 8px; vertical-align: top; text-align: left; }
th { background: #edf2f5; position: sticky; top: 0; } .label { color: #b42318; font-weight: 700; }
a { color: #075985; } p { max-width: 1050px; line-height: 1.5; }
</style></head><body>
<h1>GNSS Session Label Review</h1>
<p>Each dashboard aligns all devices and all seven signal-level features for one complete Session. The red span is resolved from the current label configuration, but only intervals overlapping the recording's actual TOW range are shown. A non-zero mismatch count means the local mirrored CSV must be rebuilt before it is used for training.</p>
<table><thead><tr><th>Environment</th><th>Scenario</th><th>Session</th><th>Status</th><th>Source</th><th>Applicable intervals</th><th>Review priority</th><th>Devices</th><th>CSV / expected positive rows</th><th>CSV mismatch rows</th><th>Files</th></tr></thead>
<tbody>""" + "\n".join(rows) + "</tbody></table></body></html>"
    (output_dir / "index.html").write_text(document, encoding="utf-8")


def selected(identity: SessionIdentity, args: argparse.Namespace) -> bool:
    return (
        (not args.environment or identity.environment == args.environment)
        and (not args.scenario or identity.scenario == args.scenario)
        and (not args.session or identity.session in args.session)
    )


def review_priority(identity: SessionIdentity, status: str, mismatch_rows: int) -> tuple[int, str]:
    """Order the review index by data risk, while retaining every Session."""

    if mismatch_rows:
        return 0, "P0: config/CSV mismatch"
    if status != "reviewed":
        return 1, "P1: label not reviewed"
    if identity.scenario in {"dy_L5", "dy_L_15"}:
        return 2, "P2: dynamic high-risk"
    if identity.scenario in {"st_L5", "st_L_15"}:
        return 3, "P3: static high-risk"
    return 4, "P4: routine full review"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="data_csv", help="Mirrored per-log CSV root.")
    parser.add_argument("--config", default="configs/preprocessing.yml", help="Formal label configuration.")
    parser.add_argument("--output-dir", default="output/label_review_dashboards", help="Generated review package.")
    parser.add_argument("--environment", help="Only one environment, for example playground.")
    parser.add_argument("--scenario", help="Only one scenario, for example dy_L5.")
    parser.add_argument(
        "--session",
        action="append",
        help="One exact Session name. Repeat --session to render several selected Sessions.",
    )
    parser.add_argument("--dpi", type=int, default=180, help="PNG DPI; use 120 for fast preview.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing dashboards.")
    parser.add_argument("--list-only", action="store_true", help="List selected sessions without rendering.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
    input_dir, config_path, output_dir = map(resolve_path, (args.input_dir, args.config, args.output_dir))
    if not input_dir.is_dir():
        raise FileNotFoundError(input_dir)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if args.dpi < 72:
        parser.error("--dpi must be at least 72")

    global _CONFIG
    _CONFIG = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    grouped: dict[SessionIdentity, list[Path]] = {}
    for path in discover_mirrored_csvs(input_dir):
        identity = read_identity(path)
        if identity is not None and selected(identity, args):
            grouped.setdefault(identity, []).append(path)
    if not grouped:
        raise ValueError("No mirrored CSVs matched the requested filters")
    LOG.info("Matched %d source CSVs in %d complete Sessions", sum(map(len, grouped.values())), len(grouped))
    for identity in sorted(grouped, key=lambda value: (value.environment, value.scenario, value.session)):
        LOG.info("  %s / %s / %s: %d device logs", identity.environment, identity.scenario, identity.session, len(grouped[identity]))
    if args.list_only:
        return

    records: list[dict] = []
    for identity in tqdm(sorted(grouped, key=lambda value: (value.environment, value.scenario, value.session)), desc="Rendering sessions"):
        frame = load_session(grouped[identity])
        intervals, formal_status, formal_source = resolve_formal_label(identity, _CONFIG)
        displayed_intervals = intervals_in_session(
            intervals, float(frame["TOW"].min()), float(frame["TOW"].max())
        )
        expected = expected_labels(frame, intervals, target_bands(identity.scenario, _CONFIG))
        observed = (frame["Label"].fillna(0).astype(int).to_numpy() > 0).astype(np.int8)
        mismatch_rows = int(np.count_nonzero(expected != observed))
        csv_only_rows = int(np.count_nonzero((observed == 1) & (expected == 0)))
        missing_rows = int(np.count_nonzero((observed == 0) & (expected == 1)))
        priority_rank, priority_text = review_priority(identity, formal_status, mismatch_rows)
        session_dir = output_dir / safe_path_component(identity.environment) / safe_path_component(identity.scenario) / safe_path_component(identity.session)
        dashboard = session_dir / "dashboard.png"
        signals = session_dir / "signals.csv"
        if args.overwrite or not dashboard.is_file():
            render_dashboard(identity, frame, intervals, formal_status, formal_source, dashboard, args.dpi)
        write_signal_summary(frame, signals)
        statuses = ";".join(sorted(set(frame.get("LabelStatus", pd.Series(dtype=str)).dropna().astype(str))))
        sources = ";".join(sorted(set(frame.get("LabelSource", pd.Series(dtype=str)).dropna().astype(str))))
        records.append(
            {
                "environment": identity.environment,
                "scenario": identity.scenario,
                "session": identity.session,
                "source_csv_count": len(grouped[identity]),
                "device_count": int(frame["DeviceName"].nunique()),
                "signal_count": int(frame["plot_signal_id"].nunique()),
                "rows": len(frame),
                "formal_intervals": "; ".join(f"[{start:g}, {end:g}]" for start, end in displayed_intervals) or "[]",
                "configured_intervals": "; ".join(f"[{start:g}, {end:g}]" for start, end in intervals) or "[]",
                "formal_status": formal_status,
                "formal_source": formal_source,
                "csv_label_statuses": statuses,
                "csv_label_sources": sources,
                "positive_rows": int(observed.sum()),
                "expected_positive_rows": int(expected.sum()),
                "label_mismatch_rows": mismatch_rows,
                "csv_only_positive_rows": csv_only_rows,
                "missing_positive_rows": missing_rows,
                "review_priority_rank": priority_rank,
                "review_priority": priority_text,
                "dashboard_rel": str(dashboard.relative_to(output_dir)),
                "signals_rel": str(signals.relative_to(output_dir)),
                "label_status": formal_status,
                "label_source": formal_source,
            }
        )
    index = pd.DataFrame(records).sort_values(
        ["review_priority_rank", "environment", "scenario", "session"], kind="mergesort"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    index.to_csv(output_dir / "session_review_index.csv", index=False, encoding="utf-8-sig")
    build_index_html(index.to_dict("records"), output_dir)
    LOG.info("Wrote review package: %s", output_dir)
    LOG.info("CSV/config label mismatches: %d", int(index["label_mismatch_rows"].sum()))


_CONFIG: dict = {}


if __name__ == "__main__":
    main()

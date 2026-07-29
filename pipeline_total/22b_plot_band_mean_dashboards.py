"""Generate session-level band-mean dashboards for scene-type inspection.

``22_generate_label_review_dashboards.py`` draws one faint line per
``signal_id``.  This sibling tool instead collapses each physical band into a
single curve: for every TOW it averages the baseline raw features over all L1
signals (blue) and separately over all L5 signals (red).  The intent is to
eyeball whether L1-only / L5-only / L1+L5 spoofing produce visually separable
band-mean signatures, i.e. whether an explicit four-way scene classifier
(L1 / L5 / L1+L5 / normal) is worth building.

Layout mirrors the review dashboard: one column per device, a label timeline
on top, then one row per baseline feature.  Aggregation is per device on
purpose -- AGC / C/N0 baselines are a device hardware fingerprint, so averaging
across devices would blend distinct baselines and hide the attack response.
"""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger(__name__)
MAX_GAP_SECONDS = 10.0

# Same band palette as the review dashboard: L1 blue, L5 red.
BAND_COLORS = {1: "#1f5f99", 5: "#b42318"}
BAND_LIGHT_COLORS = {1: "#dbeafe", 5: "#fee2e2"}
BAND_LABELS = {1: "L1", 5: "L5"}

plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# The four baseline raw features (FreqBand is the label semantic, not averaged).
FEATURES = [
    ("Cn0DbHz", "C/N0 mean (dB-Hz)"),
    ("AgcDb", "AGC mean (dB)"),
    ("ReceivedSvTimeUncertaintyNanos", "SV time unc. mean (ns)"),
    ("PseudorangeRateUncertaintyMetersPerSecond", "PR rate unc. mean (m/s)"),
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


def resolve_formal_label(identity: SessionIdentity, config: dict) -> tuple[list[tuple[float, float]], str]:
    labeling = config.get("labeling", {})
    session_entry = (
        labeling.get("session_spoofing_tow_intervals", {})
        .get(identity.environment, {})
        .get(identity.scenario, {})
        .get(identity.session)
    )
    if session_entry is None:
        return [], "missing_session_config"
    if not isinstance(session_entry, dict):
        raise ValueError(
            "Session label entries must be mappings with status and intervals: "
            f"{identity.environment}/{identity.scenario}/{identity.session}"
        )
    intervals = session_entry.get("intervals", []) or []
    status = str(session_entry.get("status", "needs_review"))
    return [(float(start), float(end)) for start, end in intervals], status


def target_bands(scenario: str, config: dict) -> set[int]:
    label_value = config.get("labeling", {}).get("spoofing_type_to_label", {}).get(scenario, 0)
    if label_value == 1:
        return {1}
    if label_value == 2:
        return {5}
    if label_value == 3:
        return {1, 5}
    return set()


def intervals_in_session(
    intervals: list[tuple[float, float]], tow_min: float, tow_max: float
) -> list[tuple[float, float]]:
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
        missing = {"DeviceName", "TOW", "signal_id", "FreqBand"}.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        frame["SourceCsv"] = path.name
        frames.append(frame)
    if not frames:
        raise ValueError("No mirrored CSVs supplied for session")
    result = pd.concat(frames, ignore_index=True)
    for column in ["TOW", "TimeNanos", "FreqBand", "Label", *(name for name, _ in FEATURES)]:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    result["DeviceName"] = result["DeviceName"].astype(str)
    result = result.dropna(subset=["TOW"])
    return result


def contiguous_runs(tow: np.ndarray) -> list[np.ndarray]:
    """Split index positions into runs broken by gaps larger than MAX_GAP."""
    if len(tow) == 0:
        return []
    splits = np.where(np.diff(tow) > MAX_GAP_SECONDS)[0] + 1
    return [run for run in np.split(np.arange(len(tow)), splits) if len(run) >= 1]


def band_mean_frame(device_frame: pd.DataFrame, feature: str, band: int) -> pd.DataFrame | None:
    """Return per-TOW mean/std/count of one feature over all signals in a band."""
    band_series = pd.to_numeric(device_frame["FreqBand"], errors="coerce")
    sub = device_frame.loc[band_series.eq(band) & device_frame[feature].notna(), ["TOW", feature]]
    if sub.empty:
        return None
    aggregated = (
        sub.groupby("TOW")[feature]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values("TOW", kind="mergesort")
    )
    aggregated["std"] = aggregated["std"].fillna(0.0)
    return aggregated


def draw_label_timeline(axis, device_frame: pd.DataFrame, intervals: list[tuple[float, float]], bands: set[int]) -> None:
    axis.set_ylim(0.4, 5.6)
    axis.set_yticks([1, 5], ["L1", "L5"])
    axis.grid(axis="x", alpha=0.22)
    axis.grid(axis="y", alpha=0.16)
    for band in (1, 5):
        axis.axhspan(band - 0.28, band + 0.28, color=BAND_LIGHT_COLORS[band], alpha=0.48, zorder=0)
    for start, end in intervals:
        axis.axvspan(start, end, color="#6b7280", alpha=0.12, zorder=0)
        for band in bands:
            axis.broken_barh(
                [(start, end - start)],
                (band - 0.28, 0.56),
                facecolors=BAND_COLORS.get(band, "#6b7280"),
                alpha=0.82,
            )
    observed = device_frame.loc[device_frame["Label"].fillna(0).astype(int) > 0] if "Label" in device_frame else device_frame.iloc[0:0]
    for band, group in observed.groupby("FreqBand", sort=True):
        if band not in {1, 5}:
            continue
        axis.scatter(
            group["TOW"],
            np.full(len(group), band),
            marker="|",
            s=10,
            color=BAND_COLORS[int(band)],
            alpha=0.55,
        )
    axis.set_ylabel("Formal label", fontsize=8)


def draw_band_mean(
    axis,
    device_frame: pd.DataFrame,
    feature: str,
    intervals: list[tuple[float, float]],
    show_std: bool,
) -> bool:
    for start, end in intervals:
        axis.axvspan(start, end, color="#6b7280", alpha=0.12, zorder=0)
        axis.axvline(start, color="#6b7280", linestyle="--", linewidth=0.55, alpha=0.7)
        axis.axvline(end, color="#6b7280", linestyle="--", linewidth=0.55, alpha=0.7)
    if feature not in device_frame.columns:
        axis.text(0.5, 0.5, "No data", transform=axis.transAxes, ha="center", va="center", color="0.45")
        return False
    drew_any = False
    for band in (1, 5):
        aggregated = band_mean_frame(device_frame, feature, band)
        if aggregated is None or aggregated.empty:
            continue
        tow = aggregated["TOW"].to_numpy(dtype=float)
        mean = aggregated["mean"].to_numpy(dtype=float)
        std = aggregated["std"].to_numpy(dtype=float)
        color = BAND_COLORS[band]
        for run in contiguous_runs(tow):
            if len(run) == 1:
                axis.plot(tow[run], mean[run], marker=".", markersize=2.5, color=color, alpha=0.9)
                continue
            axis.plot(
                tow[run], mean[run], color=color, linewidth=1.15, alpha=0.95,
                label=BAND_LABELS[band] if not drew_any else None,
            )
            if show_std:
                axis.fill_between(
                    tow[run], mean[run] - std[run], mean[run] + std[run],
                    color=color, alpha=0.12, linewidth=0,
                )
            drew_any = True
    if not drew_any:
        axis.text(0.5, 0.5, "No data", transform=axis.transAxes, ha="center", va="center", color="0.45")
        return False
    axis.autoscale_view()
    axis.grid(True, alpha=0.22)
    return True


def render_dashboard(
    identity: SessionIdentity,
    frame: pd.DataFrame,
    intervals: list[tuple[float, float]],
    status: str,
    output_path: Path,
    dpi: int,
    show_std: bool,
) -> None:
    devices = sorted(frame["DeviceName"].dropna().unique())
    tow_min = float(frame["TOW"].min())
    tow_max = float(frame["TOW"].max())
    displayed_intervals = intervals_in_session(intervals, tow_min, tow_max)
    padding = max(1.0, (tow_max - tow_min) * 0.015)
    x_limits = (tow_min - padding, tow_max + padding)
    rows, columns = len(FEATURES) + 1, len(devices)
    figure, axes = plt.subplots(
        rows, columns, sharex="col", squeeze=False,
        figsize=(max(14, 4.2 * columns), max(15, 2.0 * rows)),
    )
    bands = target_bands(identity.scenario, _CONFIG)
    label_text = ", ".join(f"[{start:g}, {end:g}]" for start, end in displayed_intervals) or "none in this recording"
    figure.suptitle(
        f"Band-mean view | {identity.environment} / {identity.scenario} / {identity.session}\n"
        f"applicable formal label: {label_text} | status={status} | target bands={sorted(bands) or 'none'}",
        fontsize=12,
    )
    for column, device in enumerate(devices):
        device_frame = frame.loc[frame["DeviceName"] == device].copy()
        n_l1 = int(pd.to_numeric(device_frame["FreqBand"], errors="coerce").eq(1).sum())
        n_l5 = int(pd.to_numeric(device_frame["FreqBand"], errors="coerce").eq(5).sum())
        axes[0, column].set_title(
            f"{device}\nL1 rows {n_l1:,} | L5 rows {n_l5:,}", fontsize=9,
        )
        draw_label_timeline(axes[0, column], device_frame, displayed_intervals, bands)
        for row, (feature, label) in enumerate(FEATURES, start=1):
            axis = axes[row, column]
            drew = draw_band_mean(axis, device_frame, feature, displayed_intervals, show_std)
            if column == 0:
                axis.set_ylabel(label, fontsize=8)
            if drew and row == 1:
                axis.legend(loc="best", fontsize=7, framealpha=0.6)
            axis.tick_params(axis="both", labelsize=7)
        for axis in axes[:, column]:
            axis.set_xlim(*x_limits)
        axes[-1, column].set_xlabel("TOW (s)", fontsize=8)
    figure.text(
        0.01, 0.008,
        "Blue = mean over all L1 signals per TOW. Red = mean over all L5 signals per TOW. "
        "Shaded band is +/-1 std across signals. Gray span is the formal session-level attack interval; "
        "ticks in the top row mark rows currently labeled positive.",
        ha="left", va="bottom", fontsize=7.5,
    )
    figure.subplots_adjust(left=0.065, right=0.99, top=0.89, bottom=0.05, hspace=0.28, wspace=0.16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def selected(identity: SessionIdentity, args: argparse.Namespace) -> bool:
    return (
        (not args.environment or identity.environment == args.environment)
        and (not args.scenario or identity.scenario == args.scenario)
        and (not args.session or identity.session in args.session)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="data_csv", help="Mirrored per-log CSV root.")
    parser.add_argument("--config", default="configs/preprocessing.yml", help="Formal label configuration.")
    parser.add_argument("--output-dir", default="output/band_mean_dashboards", help="Generated dashboards.")
    parser.add_argument("--environment", help="Only one environment, for example playground.")
    parser.add_argument("--scenario", help="Only one scenario, for example dy_L5.")
    parser.add_argument(
        "--session", action="append",
        help="One exact Session name. Repeat --session to render several selected Sessions.",
    )
    parser.add_argument("--dpi", type=int, default=180, help="PNG DPI; use 120 for fast preview.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing dashboards.")
    parser.add_argument("--no-std", action="store_true", help="Hide the +/-1 std band, draw mean lines only.")
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

    for identity in tqdm(sorted(grouped, key=lambda value: (value.environment, value.scenario, value.session)), desc="Rendering sessions"):
        frame = load_session(grouped[identity])
        intervals, status = resolve_formal_label(identity, _CONFIG)
        session_dir = output_dir / safe_path_component(identity.environment) / safe_path_component(identity.scenario) / safe_path_component(identity.session)
        dashboard = session_dir / "band_mean.png"
        if args.overwrite or not dashboard.is_file():
            render_dashboard(identity, frame, intervals, status, dashboard, args.dpi, not args.no_std)
    LOG.info("Wrote band-mean dashboards under: %s", output_dir)


_CONFIG: dict = {}


if __name__ == "__main__":
    main()

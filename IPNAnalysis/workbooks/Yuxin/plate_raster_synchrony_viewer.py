from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

PLATE_ROWS = 4
PLATE_COLS = 6
PLATE_WELL_COUNT = PLATE_ROWS * PLATE_COLS
ROW_LABELS = ["A", "B", "C", "D"]
DISPLAY_MODES = {"raster", "synchrony", "both"}
INDEX_COLUMNS = [
    "root_dir",
    "scan_dir",
    "scan_id",
    "run_id",
    "scan_label",
    "well_id",
    "plate_label",
    "row",
    "col",
    "spike_times_path",
    "network_json_path",
    "has_spike_times",
    "has_network_json",
]
MANIFEST_COLUMNS = [
    "run_id",
    "scan_id",
    "scan_label",
    "scan_dir",
    "n_wells",
    "missing_spike_times",
    "missing_network_json",
]
PLATE_HORIZONTAL_SPACING = 0.03
PLATE_VERTICAL_SPACING = 0.08
FIGURE_MARGIN_LR_PX = 90
FIGURE_MARGIN_TOP_PX = 320
FIGURE_MARGIN_BOTTOM_PX = 70
FIGURE_MARGIN_TB_PX = FIGURE_MARGIN_TOP_PX + FIGURE_MARGIN_BOTTOM_PX
TARGET_PANEL_WIDTH_TO_HEIGHT = 2.0
RASTER_MARKER_SYMBOL = "line-ns-open"
RASTER_MARKER_COLOR = "rgba(90, 90, 90, 0.75)"
TITLE_Y = 0.98
TITLE_FONT_SIZE = 16
LEGEND_Y = 1.15
LEGEND_FONT_SIZE = 11
MODE_BUTTONS_X = 0.0
MODE_BUTTONS_Y = 1.30
SLIDER_X = 0.0
SLIDER_Y = 1.25
SLIDER_LEN = 0.25
DEFAULT_VIEWER_WIDTH_PX = 2400
DEFAULT_VIEWER_MAX_RASTER_POINTS = 12000
DEFAULT_VIEWER_MAX_SYNCHRONY_POINTS = 3000
DEFAULT_INITIAL_WINDOW_S = 300.0
DEFAULT_EXPORT_WIDTH_IN = 24.0
DEFAULT_EXPORT_DPI = 600


def _normalize_run_id(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    digits = "".join(ch for ch in text if ch.isdigit())
    if digits:
        return digits.zfill(6)
    return text


def _run_sort_key(value: Any) -> tuple[int, str, str]:
    normalized = _normalize_run_id(value) or ""
    digits = "".join(ch for ch in normalized if ch.isdigit())
    if digits:
        return (0, f"{int(digits):09d}", normalized)
    return (1, normalized.lower(), normalized)


def _validate_display_mode(display_mode: str) -> None:
    if display_mode not in DISPLAY_MODES:
        raise ValueError(f"display_mode must be one of {DISPLAY_MODES}")


@dataclass(frozen=True, slots=True)
class ViewerConfig:
    display_mode: str = "both"
    marker_size: float = 5.0
    line_width: float = 1.25
    unit_sort_mode: str = "firing_rate_desc"
    max_raster_points_per_well: int | None = DEFAULT_VIEWER_MAX_RASTER_POINTS
    max_synchrony_points: int | None = DEFAULT_VIEWER_MAX_SYNCHRONY_POINTS
    width_px: int = DEFAULT_VIEWER_WIDTH_PX
    height_px: int | None = None
    initial_window_s: float | None = DEFAULT_INITIAL_WINDOW_S

    def __post_init__(self) -> None:
        _validate_display_mode(self.display_mode)


@dataclass(frozen=True, slots=True)
class ExportConfig:
    output_dir: str | Path
    dpi: int = DEFAULT_EXPORT_DPI
    width_in: float = DEFAULT_EXPORT_WIDTH_IN
    height_in: float | None = None
    export_html: bool = True
    export_png: bool = True

    @property
    def resolved_output_dir(self) -> Path:
        return Path(self.output_dir).expanduser().resolve()

    @property
    def resolved_height_in(self) -> float:
        if self.height_in is not None:
            return float(self.height_in)
        return compute_plate_height_px(int(float(self.width_in) * 100)) / 100.0


def well_to_plate_position(well_id: str) -> tuple[int, int, str]:
    well_num = int(str(well_id).replace("well", ""))
    if not 0 <= well_num < PLATE_WELL_COUNT:
        raise ValueError(f"Well {well_id} is outside the supported 24-well range.")
    row = (well_num // PLATE_COLS) + 1
    col = (well_num % PLATE_COLS) + 1
    label = f"{ROW_LABELS[row - 1]}{col}"
    return row, col, label


def scan_context_label(scan_dir: Path) -> str:
    parts = scan_dir.parts
    try:
        network_idx = parts.index("Network")
    except ValueError:
        return scan_dir.as_posix()
    start_idx = max(0, network_idx - 3)
    return "/".join(parts[start_idx : network_idx + 2])


def discover_well_records(root_dir: str | Path) -> pd.DataFrame:
    root = Path(root_dir).expanduser().resolve()
    rows: list[dict[str, Any]] = []

    for well_dir in root.rglob("well*"):
        if not well_dir.is_dir() or not well_dir.name.startswith("well"):
            continue
        scan_dir = well_dir.parent
        if scan_dir.parent.name != "Network":
            continue
        try:
            row_idx, col_idx, plate_label = well_to_plate_position(well_dir.name)
        except ValueError:
            continue

        raw_scan_id = scan_dir.name
        run_id = _normalize_run_id(raw_scan_id) or raw_scan_id
        spike_path = well_dir / "spike_times.npy"
        network_json_path = well_dir / "network_results.json"
        resolved_scan_dir = str(scan_dir.resolve())

        rows.append(
            {
                "root_dir": str(root),
                "scan_dir": resolved_scan_dir,
                "scan_id": raw_scan_id,
                "run_id": run_id,
                "scan_label": scan_context_label(Path(resolved_scan_dir)),
                "well_id": well_dir.name,
                "plate_label": plate_label,
                "row": row_idx,
                "col": col_idx,
                "spike_times_path": str(spike_path),
                "network_json_path": str(network_json_path),
                "has_spike_times": spike_path.exists(),
                "has_network_json": network_json_path.exists(),
            }
        )

    if not rows:
        return pd.DataFrame(columns=INDEX_COLUMNS)

    out = pd.DataFrame(rows).drop_duplicates(subset=["scan_dir", "well_id"]).reset_index(drop=True)
    out["_run_sort_key"] = out["run_id"].apply(_run_sort_key)
    out = out.sort_values(["_run_sort_key", "scan_dir", "well_id"]).drop(columns="_run_sort_key")
    return out.reset_index(drop=True)


def build_run_manifest(index_df: pd.DataFrame) -> pd.DataFrame:
    if index_df.empty:
        return pd.DataFrame(columns=MANIFEST_COLUMNS)

    manifest = (
        index_df.groupby(["run_id", "scan_id", "scan_label", "scan_dir"], as_index=False)
        .agg(
            n_wells=("well_id", "nunique"),
            missing_spike_times=("has_spike_times", lambda s: int((~s).sum())),
            missing_network_json=("has_network_json", lambda s: int((~s).sum())),
        )
        .copy()
    )
    manifest["_run_sort_key"] = manifest["run_id"].apply(_run_sort_key)
    manifest = manifest.sort_values(["_run_sort_key", "scan_dir"]).drop(columns="_run_sort_key")
    return manifest.reset_index(drop=True)


@lru_cache(maxsize=512)
def _load_spike_times_cached(spike_path: str) -> dict[str, np.ndarray]:
    loaded = np.load(spike_path, allow_pickle=True)
    data = loaded.item() if hasattr(loaded, "item") else loaded
    if not isinstance(data, dict):
        raise ValueError(f"Unexpected spike_times payload at {spike_path}")
    return {str(key): np.asarray(value, dtype=float) for key, value in data.items()}


def load_spike_times(spike_path: str | Path) -> dict[str, np.ndarray]:
    return _load_spike_times_cached(str(Path(spike_path).expanduser().resolve()))


@lru_cache(maxsize=512)
def _load_network_plot_data_cached(network_json_path: str) -> dict[str, Any]:
    with open(network_json_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    plot_data = payload.get("plot_data", {})
    for key in ("t", "signal", "signal_smooth", "burst_peak_times", "burst_peak_values"):
        if key in plot_data and plot_data[key] is not None:
            plot_data[key] = np.asarray(plot_data[key], dtype=float)
    return plot_data


def load_network_plot_data(network_json_path: str | Path) -> dict[str, Any]:
    return _load_network_plot_data_cached(str(Path(network_json_path).expanduser().resolve()))


def sort_units(spike_times: dict[str, np.ndarray], mode: str = "firing_rate_desc") -> list[str]:
    if mode == "native":
        return list(spike_times.keys())

    def last_time(values: np.ndarray) -> float:
        return float(values[-1]) if values.size else 0.0

    if mode == "firing_rate_desc":
        duration = max((last_time(values) for values in spike_times.values()), default=1.0) or 1.0
        return sorted(spike_times.keys(), key=lambda unit: len(spike_times[unit]) / duration, reverse=True)

    return sorted(spike_times.keys(), key=str)


def downsample_values(values: np.ndarray, max_points: int | None = None) -> np.ndarray:
    arr = np.asarray(values)
    if max_points is None or max_points <= 0 or arr.size <= max_points:
        return arr
    idx = np.linspace(0, arr.size - 1, num=max_points, dtype=int)
    return arr[np.unique(idx)]


def downsample_xy(
    x: np.ndarray,
    y: np.ndarray,
    max_points: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    n = min(x_arr.size, y_arr.size)
    if n == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    x_arr = x_arr[:n]
    y_arr = y_arr[:n]
    if max_points is None or max_points <= 0 or n <= max_points:
        return x_arr, y_arr
    idx = np.linspace(0, n - 1, num=max_points, dtype=int)
    idx = np.unique(idx)
    return x_arr[idx], y_arr[idx]


def safe_nanmax(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return 0.0
    return float(np.nanmax(arr))


def _raster_points_for_well(
    spike_times: dict[str, np.ndarray],
    well_id: str,
    plate_label: str,
    unit_sort_mode: str,
    max_points_per_well: int | None,
) -> tuple[list[go.Scattergl], float, int]:
    units = sort_units(spike_times, mode=unit_sort_mode)
    nonempty: list[tuple[int, str, np.ndarray]] = []
    max_time = 0.0

    for unit_index, unit_name in enumerate(units, start=1):
        spikes = np.asarray(spike_times[unit_name], dtype=float)
        if spikes.size == 0:
            continue
        nonempty.append((unit_index, str(unit_name), spikes))
        max_time = max(max_time, float(spikes[-1]))

    if not nonempty:
        return [], max_time, max(len(units), 1)

    total_points = sum(spikes.size for _, _, spikes in nonempty)
    stride = 1
    if max_points_per_well is not None and max_points_per_well > 0 and total_points > max_points_per_well:
        stride = int(np.ceil(total_points / max_points_per_well))

    traces: list[go.Scattergl] = []
    for rank, (_, unit_name, spikes) in enumerate(nonempty, start=1):
        sampled = spikes[::stride] if stride > 1 else spikes
        if sampled.size == 0:
            continue
        customdata = np.column_stack(
            [
                np.full(sampled.shape, well_id, dtype=object),
                np.full(sampled.shape, plate_label, dtype=object),
                np.full(sampled.shape, unit_name, dtype=object),
                np.full(sampled.shape, rank, dtype=object),
                np.full(sampled.shape, len(nonempty), dtype=object),
            ]
        )
        traces.append(
            go.Scattergl(
                x=sampled,
                y=np.full(sampled.shape, rank, dtype=float),
                mode="markers",
                marker=dict(symbol=RASTER_MARKER_SYMBOL, size=4, color=RASTER_MARKER_COLOR),
                customdata=customdata,
                hovertemplate=(
                    "%{customdata[1]} / %{customdata[0]}"
                    "<br>Unit %{customdata[2]} (%{customdata[3]}/%{customdata[4]})"
                    "<br>t=%{x:.3f} s"
                    "<extra></extra>"
                ),
                name="Raster",
                legendgroup="raster",
                showlegend=False,
            )
        )

    return traces, max_time, max(len(nonempty), 1)


def _synchrony_traces_for_well(
    plot_data: dict[str, Any],
    well_id: str,
    plate_label: str,
    line_width: float,
    max_points: int | None,
) -> tuple[list[go.BaseTraceType], float, float]:
    traces: list[go.BaseTraceType] = []
    t = np.asarray(plot_data.get("t", []), dtype=float)
    signal = np.asarray(plot_data.get("signal", []), dtype=float)
    smooth = np.asarray(plot_data.get("signal_smooth", []), dtype=float)
    peak_t = np.asarray(plot_data.get("burst_peak_times", []), dtype=float)
    peak_y = np.asarray(plot_data.get("burst_peak_values", []), dtype=float)
    baseline = plot_data.get("baseline")
    threshold = plot_data.get("threshold")

    xmax = float(t[-1]) if t.size else 0.0
    ymax = max(
        safe_nanmax(signal),
        safe_nanmax(smooth),
        safe_nanmax(peak_y),
        float(baseline) if baseline is not None else 0.0,
        float(threshold) if threshold is not None else 0.0,
        0.0,
    )

    signal_x, signal_y = downsample_xy(t, signal, max_points=max_points)
    if signal_x.size and signal_y.size:
        customdata = np.column_stack(
            [
                np.full(signal_x.shape, well_id, dtype=object),
                np.full(signal_x.shape, plate_label, dtype=object),
                signal_y,
            ]
        )
        traces.append(
            go.Scattergl(
                x=signal_x,
                y=signal_y,
                mode="lines",
                line=dict(color="#b22222", width=max(0.5, line_width)),
                customdata=customdata,
                hovertemplate=(
                    "%{customdata[1]} / %{customdata[0]}"
                    "<br>Synchrony=%{customdata[2]:.3f}"
                    "<br>t=%{x:.3f} s"
                    "<extra></extra>"
                ),
                name="Synchrony",
                legendgroup="synchrony",
                showlegend=False,
            )
        )

    smooth_x, smooth_y = downsample_xy(t, smooth, max_points=max_points)
    if smooth_x.size and smooth_y.size:
        customdata = np.column_stack(
            [
                np.full(smooth_x.shape, well_id, dtype=object),
                np.full(smooth_x.shape, plate_label, dtype=object),
                smooth_y,
            ]
        )
        traces.append(
            go.Scattergl(
                x=smooth_x,
                y=smooth_y,
                mode="lines",
                line=dict(color="rgba(255, 140, 0, 0.95)", width=max(0.5, line_width - 0.1)),
                customdata=customdata,
                hovertemplate=(
                    "%{customdata[1]} / %{customdata[0]}"
                    "<br>Smooth synchrony=%{customdata[2]:.3f}"
                    "<br>t=%{x:.3f} s"
                    "<extra></extra>"
                ),
                name="Synchrony smooth",
                legendgroup="synchrony",
                showlegend=False,
            )
        )

    if peak_t.size and peak_y.size:
        peak_n = min(peak_t.size, peak_y.size)
        peak_t = downsample_values(peak_t[:peak_n], max_points=max_points)
        peak_y = downsample_values(peak_y[:peak_n], max_points=max_points)
        peak_n = min(peak_t.size, peak_y.size)
        peak_t = peak_t[:peak_n]
        peak_y = peak_y[:peak_n]
        customdata = np.column_stack(
            [
                np.full(peak_t.shape, well_id, dtype=object),
                np.full(peak_t.shape, plate_label, dtype=object),
                peak_y,
            ]
        )
        traces.append(
            go.Scattergl(
                x=peak_t,
                y=peak_y,
                mode="markers",
                marker=dict(color="red", size=4),
                customdata=customdata,
                hovertemplate=(
                    "%{customdata[1]} / %{customdata[0]}"
                    "<br>Burst peak=%{customdata[2]:.3f}"
                    "<br>t=%{x:.3f} s"
                    "<extra></extra>"
                ),
                name="Burst peaks",
                legendgroup="synchrony",
                showlegend=False,
            )
        )

    if t.size:
        line_x = np.asarray([float(t[0]), float(t[-1])], dtype=float)
        if baseline is not None:
            baseline_val = float(baseline)
            traces.append(
                go.Scatter(
                    x=line_x,
                    y=np.asarray([baseline_val, baseline_val], dtype=float),
                    mode="lines",
                    line=dict(color="rgba(255, 102, 0, 0.7)", width=1, dash="dash"),
                    hoverinfo="skip",
                    name="Baseline",
                    legendgroup="synchrony",
                    showlegend=False,
                )
            )
        if threshold is not None:
            threshold_val = float(threshold)
            traces.append(
                go.Scatter(
                    x=line_x,
                    y=np.asarray([threshold_val, threshold_val], dtype=float),
                    mode="lines",
                    line=dict(color="rgba(192, 57, 43, 0.8)", width=1, dash="dash"),
                    hoverinfo="skip",
                    name="Threshold",
                    legendgroup="synchrony",
                    showlegend=False,
                )
            )

    return traces, xmax, max(ymax, 1.0)


def _axis_ref(prefix: str, axis_index: int) -> str:
    return prefix if axis_index == 1 else f"{prefix}{axis_index}"


def _layout_axis_name(prefix: str, axis_index: int) -> str:
    return f"{prefix}axis" if axis_index == 1 else f"{prefix}axis{axis_index}"


def _facet_axis_refs(axis_index: int) -> tuple[str, str]:
    xref = _axis_ref("x", axis_index)
    yref = _axis_ref("y", axis_index)
    return f"{xref} domain", f"{yref} domain"


def _subplot_axis_index(row: int, col: int) -> int:
    return ((row - 1) * PLATE_COLS) + col


def _xaxis_range_layout_key(axis_index: int) -> str:
    return f"{_layout_axis_name('x', axis_index)}.range"


def _xaxis_window_relayout(window_end: float) -> dict[str, list[float]]:
    return {
        _xaxis_range_layout_key(axis_index): [0.0, float(window_end)]
        for axis_index in range(1, PLATE_WELL_COUNT + 1)
    }


def _build_xspan_slider(global_xmax: float, initial_window_s: float | None) -> tuple[list[dict[str, Any]], float]:
    if global_xmax <= 0:
        return [], 0.0

    max_window = max(1, int(np.floor(global_xmax)))
    min_window = 1 if max_window >= 1 else max_window
    window_values = list(range(min_window, max_window + 1))
    if not window_values:
        window_values = [max_window]

    if initial_window_s is not None and initial_window_s > 0:
        initial_window = min(max_window, max(min_window, int(round(initial_window_s))))
    else:
        initial_window = window_values[0]

    active_index = min(range(len(window_values)), key=lambda i: abs(window_values[i] - initial_window))
    steps = [
        {
            "label": str(window_value),
            "method": "relayout",
            "args": [_xaxis_window_relayout(window_value)],
        }
        for window_value in window_values
    ]

    slider = {
        "active": active_index,
        "currentvalue": {"prefix": "X span (s): "},
        "len": SLIDER_LEN,
        "x": SLIDER_X,
        "xanchor": "left",
        "y": SLIDER_Y,
        "yanchor": "top",
        "pad": {"t": 6, "b": 0},
        "steps": steps,
    }
    return [slider], float(window_values[active_index])


def _panel_domain_fraction(count: int, spacing: float) -> float:
    return (1.0 - (count - 1) * spacing) / count


def compute_plate_height_px(
    width_px: int,
    *,
    target_panel_width_to_height: float = TARGET_PANEL_WIDTH_TO_HEIGHT,
) -> int:
    effective_width_px = max(width_px, FIGURE_MARGIN_LR_PX + 1)
    inner_width_px = max(1.0, float(effective_width_px - FIGURE_MARGIN_LR_PX))
    panel_width_frac = _panel_domain_fraction(PLATE_COLS, PLATE_HORIZONTAL_SPACING)
    panel_height_frac = _panel_domain_fraction(PLATE_ROWS, PLATE_VERTICAL_SPACING)
    inner_height_px = inner_width_px * panel_width_frac / (target_panel_width_to_height * panel_height_frac)
    return max(480, int(round(inner_height_px + FIGURE_MARGIN_TB_PX)))


def _resolve_figure_height_px(width_px: int, height_px: int | None) -> int:
    if height_px is not None:
        return int(height_px)
    return compute_plate_height_px(int(width_px))


def _make_plate_figure(title: str, width_px: int, height_px: int | None) -> go.Figure:
    subplot_titles = [f"{row_label}{col}" for row_label in ROW_LABELS for col in range(1, PLATE_COLS + 1)]
    resolved_height_px = _resolve_figure_height_px(width_px, height_px)
    fig = make_subplots(
        rows=PLATE_ROWS,
        cols=PLATE_COLS,
        specs=[[{"secondary_y": True} for _ in range(PLATE_COLS)] for _ in range(PLATE_ROWS)],
        subplot_titles=subplot_titles,
        horizontal_spacing=PLATE_HORIZONTAL_SPACING,
        vertical_spacing=PLATE_VERTICAL_SPACING,
    )
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", y=TITLE_Y, yanchor="top", font=dict(size=TITLE_FONT_SIZE)),
        width=width_px,
        height=resolved_height_px,
        template="plotly_white",
        hovermode="closest",
        margin=dict(l=50, r=40, t=FIGURE_MARGIN_TOP_PX, b=FIGURE_MARGIN_BOTTOM_PX),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=LEGEND_Y,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.2)",
            borderwidth=1,
            font=dict(size=LEGEND_FONT_SIZE),
        ),
    )
    return fig


def _annotate_missing_panel(fig: go.Figure, row: int, col: int, text: str, *, x: float = 0.5, y: float = 0.5) -> None:
    axis_index = _subplot_axis_index(row, col)
    xref, yref = _facet_axis_refs(axis_index)
    fig.add_annotation(
        x=x,
        y=y,
        xref=xref,
        yref=yref,
        text=text,
        showarrow=False,
        xanchor="center" if x == 0.5 else "right",
        font=dict(size=12 if x == 0.5 else 10, color="gray"),
    )


def _build_scan_figure(scan_df: pd.DataFrame, viewer_config: ViewerConfig, title: str | None = None) -> go.Figure:
    _validate_display_mode(viewer_config.display_mode)
    label = title or scan_df["scan_label"].iloc[0]
    run_id = scan_df["run_id"].iloc[0] if "run_id" in scan_df.columns else scan_df["scan_id"].iloc[0]
    scan_id = scan_df["scan_id"].iloc[0]
    fig = _make_plate_figure(title=f"Plate Raster + Synchrony: {label}", width_px=viewer_config.width_px, height_px=viewer_config.height_px)

    trace_roles: list[str] = []
    legend_state = {"raster": False, "synchrony": False}
    global_xmax = 0.0
    primary_y_ranges: dict[tuple[int, int], float] = {}
    secondary_y_ranges: dict[tuple[int, int], float] = {}

    annotations_by_text = {str(ann.text): ann for ann in fig.layout.annotations}
    for well_num in range(PLATE_WELL_COUNT):
        well_id = f"well{well_num:03d}"
        row, col, plate_label = well_to_plate_position(well_id)
        ann = annotations_by_text.get(plate_label)
        if ann is not None:
            ann.text = f"{plate_label}<br>{well_id}"
            ann.font = dict(size=13)

        matches = scan_df[scan_df["well_id"] == well_id]
        if matches.empty:
            _annotate_missing_panel(fig, row, col, "Missing well")
            primary_y_ranges[(row, col)] = 1.0
            secondary_y_ranges[(row, col)] = 1.0
            continue

        record = matches.iloc[0]
        raster_loaded = False
        sync_loaded = False

        if bool(record["has_spike_times"]):
            raster_traces, spike_xmax, unit_count = _raster_points_for_well(
                load_spike_times(record["spike_times_path"]),
                well_id=well_id,
                plate_label=plate_label,
                unit_sort_mode=viewer_config.unit_sort_mode,
                max_points_per_well=viewer_config.max_raster_points_per_well,
            )
            primary_y_ranges[(row, col)] = float(unit_count + 1)
            global_xmax = max(global_xmax, spike_xmax)
            for trace in raster_traces:
                trace.marker.size = viewer_config.marker_size
                if not legend_state["raster"]:
                    trace.showlegend = True
                    legend_state["raster"] = True
                fig.add_trace(trace, row=row, col=col, secondary_y=False)
                trace_roles.append("raster")
            raster_loaded = bool(raster_traces)
        else:
            primary_y_ranges[(row, col)] = 1.0

        if bool(record["has_network_json"]):
            sync_traces, sync_xmax, sync_ymax = _synchrony_traces_for_well(
                load_network_plot_data(record["network_json_path"]),
                well_id=well_id,
                plate_label=plate_label,
                line_width=viewer_config.line_width,
                max_points=viewer_config.max_synchrony_points,
            )
            secondary_y_ranges[(row, col)] = float(sync_ymax * 1.1)
            global_xmax = max(global_xmax, sync_xmax)
            for trace in sync_traces:
                if not legend_state["synchrony"] and trace.name in {"Synchrony", "Synchrony smooth", "Burst peaks"}:
                    trace.showlegend = True
                    legend_state["synchrony"] = True
                fig.add_trace(trace, row=row, col=col, secondary_y=True)
                trace_roles.append("synchrony")
            sync_loaded = bool(sync_traces)
        else:
            secondary_y_ranges[(row, col)] = 1.0

        if not raster_loaded and not sync_loaded:
            _annotate_missing_panel(fig, row, col, "No plot data")
        elif not raster_loaded or not sync_loaded:
            missing_text = "raster missing" if not raster_loaded else "synchrony missing"
            _annotate_missing_panel(fig, row, col, missing_text, x=0.98, y=0.95)

    fig.update_xaxes(matches="x", showgrid=True, rangeslider_visible=False)
    fig.update_yaxes(matches=None, fixedrange=True)

    for row in range(1, PLATE_ROWS + 1):
        for col in range(1, PLATE_COLS + 1):
            fig.update_yaxes(
                range=[0.0, primary_y_ranges.get((row, col), 1.0)],
                title_text="Unit" if col == 1 else None,
                fixedrange=True,
                row=row,
                col=col,
                secondary_y=False,
            )
            fig.update_yaxes(
                range=[0.0, secondary_y_ranges.get((row, col), 1.0)],
                title_text="Sync" if col == PLATE_COLS else None,
                showgrid=False,
                zeroline=False,
                fixedrange=True,
                row=row,
                col=col,
                secondary_y=True,
            )
            fig.update_xaxes(title_text="Time (s)" if row == PLATE_ROWS else None, row=row, col=col)

    sliders, window_end = _build_xspan_slider(global_xmax, viewer_config.initial_window_s)
    if window_end > 0:
        fig.update_xaxes(range=[0.0, window_end])

    mode_visibility = {
        "both": [True for _ in trace_roles],
        "raster": [role == "raster" for role in trace_roles],
        "synchrony": [role == "synchrony" for role in trace_roles],
    }
    for trace, visible in zip(fig.data, mode_visibility[viewer_config.display_mode]):
        trace.visible = visible

    fig.update_layout(
        title=dict(
            text=f"Plate Raster + Synchrony: {label}<br><sup>run_id={run_id} | scan_id={scan_id}</sup>",
            x=0.5,
            xanchor="center",
            y=TITLE_Y,
            yanchor="top",
            font=dict(size=TITLE_FONT_SIZE),
        ),
        sliders=sliders,
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=MODE_BUTTONS_X,
                y=MODE_BUTTONS_Y,
                xanchor="left",
                yanchor="top",
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="rgba(0, 0, 0, 0.2)",
                borderwidth=1,
                pad={"r": 12, "t": 4, "b": 0},
                buttons=[
                    dict(label="Both", method="update", args=[{"visible": mode_visibility["both"]}]),
                    dict(label="Raster only", method="update", args=[{"visible": mode_visibility["raster"]}]),
                    dict(label="Synchrony only", method="update", args=[{"visible": mode_visibility["synchrony"]}]),
                ],
            )
        ],
    )
    return fig


def _figure_to_iframe_markup(fig: go.Figure) -> str:
    figure_width = fig.layout.width
    figure_height = fig.layout.height
    frame_width_css = "100%" if figure_width is None else f"{int(figure_width)}px"
    frame_height_px = max(480, int(figure_height or 900) + 18)
    figure_html = pio.to_html(
        fig,
        include_plotlyjs=True,
        full_html=True,
        config={"responsive": False},
    )
    return (
        '<div style="width:100%; overflow-x:auto; overflow-y:hidden;">'
        f'<iframe srcdoc="{escape(figure_html)}" '
        f'style="display:block; width:{frame_width_css}; height:{frame_height_px}px; '
        'border:none; max-width:none;" '
        'loading="lazy"></iframe>'
        "</div>"
    )


def _sanitize_filename_part(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    text = text.strip("._")
    return text or "scan"


def _export_figure(fig: go.Figure, export_config: ExportConfig, *, stem: str) -> dict[str, Path]:
    out_dir = export_config.resolved_output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    exported: dict[str, Path] = {}

    base_width_px = max(1200, int(float(export_config.width_in) * 100))
    base_height_px = max(800, int(float(export_config.resolved_height_in) * 100))
    scale = max(1.0, float(export_config.dpi) / 100.0)

    if export_config.export_html:
        html_path = out_dir / f"{stem}.html"
        fig.write_html(html_path, include_plotlyjs=True, full_html=True)
        exported["html"] = html_path

    if export_config.export_png:
        png_path = out_dir / f"{stem}.png"
        try:
            fig.write_image(
                png_path,
                format="png",
                width=base_width_px,
                height=base_height_px,
                scale=scale,
            )
        except Exception as exc:
            raise RuntimeError(
                "PNG export requires Plotly image export support, typically via `pip install kaleido`."
            ) from exc
        exported["png"] = png_path

    return exported


class PlateRasterSynchronyViewer:
    def __init__(self, index_df: pd.DataFrame, viewer_config: ViewerConfig | None = None):
        self.viewer_config = viewer_config or ViewerConfig()
        self.index_df = index_df.copy().reset_index(drop=True)
        self.manifest_df = build_run_manifest(self.index_df)
        self._manifest_by_scan_dir = self.manifest_df.set_index("scan_dir", drop=False) if not self.manifest_df.empty else None
        self._duplicate_run_ids = {
            str(value)
            for value in self.manifest_df["run_id"][self.manifest_df["run_id"].duplicated(keep=False)].tolist()
        }
        self._figure_markup_cache: dict[str, str] = {}
        self._current_scan_dir: str | None = None

    @classmethod
    def from_analysis_root(
        cls,
        root_dir: str | Path,
        viewer_config: ViewerConfig | None = None,
    ) -> PlateRasterSynchronyViewer:
        return cls(discover_well_records(root_dir), viewer_config=viewer_config)

    @property
    def current_scan_dir(self) -> str | None:
        return self._current_scan_dir

    def _ensure_manifest(self) -> None:
        if self.manifest_df.empty or self._manifest_by_scan_dir is None:
            raise RuntimeError("No scans found. Check ANALYSIS_ROOT and the expected directory pattern.")

    def _resolve_manifest_row(self, scan_dir: str | Path) -> pd.Series:
        self._ensure_manifest()
        normalized_scan_dir = str(Path(scan_dir).expanduser().resolve())
        if normalized_scan_dir not in self._manifest_by_scan_dir.index:
            raise ValueError(f"No scan found for scan_dir={normalized_scan_dir!r}")
        row = self._manifest_by_scan_dir.loc[normalized_scan_dir]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        return row

    def _default_export_stem(self, row: pd.Series | dict[str, Any]) -> str:
        run_id = str(row["run_id"])
        scan_id = str(row["scan_id"])
        run_part = _sanitize_filename_part(run_id)
        scan_part = _sanitize_filename_part(scan_id)
        if run_id in self._duplicate_run_ids and scan_part != run_part:
            return f"{run_part}__{scan_part}_plate_overlay"
        if run_id in self._duplicate_run_ids:
            label_part = _sanitize_filename_part(Path(str(row["scan_dir"])).name)
            return f"{run_part}__{label_part}_plate_overlay"
        return f"{run_part}_plate_overlay"

    def resolve_scan_dir(self, run_id: str | int) -> str:
        self._ensure_manifest()
        normalized_run_id = _normalize_run_id(run_id)
        matches = self.manifest_df[self.manifest_df["run_id"] == normalized_run_id]
        if matches.empty:
            raise ValueError(f"No scan found for run_id={run_id!r}")
        if len(matches) > 1:
            raise ValueError(
                f"run_id={normalized_run_id!r} is not unique. Use `manifest_df['scan_dir']` to select a specific scan."
            )
        return str(matches.iloc[0]["scan_dir"])

    def figure_for_scan(self, scan_dir: str | Path) -> go.Figure:
        row = self._resolve_manifest_row(scan_dir)
        normalized_scan_dir = str(row["scan_dir"])
        scan_df = self.index_df[self.index_df["scan_dir"] == normalized_scan_dir].copy()
        if scan_df.empty:
            raise ValueError(f"No wells found for scan_dir={normalized_scan_dir!r}")
        self._current_scan_dir = normalized_scan_dir
        return _build_scan_figure(scan_df, self.viewer_config)

    def render_widget(self, initial_scan_dir: str | Path | None = None) -> Any:
        try:
            import ipywidgets as widgets
        except ImportError as exc:
            raise ImportError(
                "render_widget requires ipywidgets. Install it with `pip install ipywidgets` "
                "or `pip install -r requirements.txt`, then restart the notebook kernel."
            ) from exc

        self._ensure_manifest()
        if initial_scan_dir is None:
            resolved_initial_scan_dir = str(self.manifest_df.iloc[0]["scan_dir"])
        else:
            resolved_initial_scan_dir = str(self._resolve_manifest_row(initial_scan_dir)["scan_dir"])

        dropdown_options = [
            (f"{row.run_id} | {row.scan_label}", row.scan_dir)
            for row in self.manifest_df.itertuples(index=False)
        ]
        scan_dropdown = widgets.Dropdown(
            options=dropdown_options,
            value=resolved_initial_scan_dir,
            description="scan",
            layout=widgets.Layout(width="95%"),
            style={"description_width": "72px"},
        )
        status_html = widgets.HTML()
        figure_html = widgets.HTML(layout=widgets.Layout(width="100%"))

        def _render(scan_dir_value: str) -> None:
            row = self._resolve_manifest_row(scan_dir_value)
            normalized_scan_dir = str(row["scan_dir"])
            self._current_scan_dir = normalized_scan_dir
            status_html.value = (
                "<div>"
                f"<b>run_id:</b> {escape(str(row['run_id']))} &nbsp; "
                f"<b>scan_id:</b> {escape(str(row['scan_id']))} &nbsp; "
                f"<b>wells:</b> {int(row['n_wells'])} &nbsp; "
                f"<b>missing spikes:</b> {int(row['missing_spike_times'])} &nbsp; "
                f"<b>missing synchrony:</b> {int(row['missing_network_json'])}"
                f"<br><code>{escape(str(row['scan_label']))}</code>"
                "</div>"
            )
            if normalized_scan_dir not in self._figure_markup_cache:
                self._figure_markup_cache[normalized_scan_dir] = _figure_to_iframe_markup(
                    self.figure_for_scan(normalized_scan_dir)
                )
            figure_html.value = self._figure_markup_cache[normalized_scan_dir]

        def _on_scan_change(change: dict[str, Any]) -> None:
            if change.get("name") == "value" and change.get("new"):
                _render(str(change["new"]))

        scan_dropdown.observe(_on_scan_change, names="value")
        _render(resolved_initial_scan_dir)

        return widgets.VBox(
            [
                widgets.HTML("<b>Plate viewer</b>"),
                scan_dropdown,
                status_html,
                figure_html,
            ]
        )

    def export_scan(
        self,
        scan_dir: str | Path,
        export_config: ExportConfig,
        stem: str | None = None,
    ) -> dict[str, Path]:
        row = self._resolve_manifest_row(scan_dir)
        export_stem = stem or self._default_export_stem(row)
        fig = self.figure_for_scan(str(row["scan_dir"]))
        return _export_figure(fig, export_config, stem=export_stem)

    def export_all(self, export_config: ExportConfig) -> list[dict[str, str]]:
        self._ensure_manifest()
        exports: list[dict[str, str]] = []
        for row in self.manifest_df.itertuples(index=False):
            row_dict = row._asdict()
            exported = self.export_scan(str(row.scan_dir), export_config, stem=self._default_export_stem(row_dict))
            exports.append(
                {
                    "run_id": str(row.run_id),
                    "scan_id": str(row.scan_id),
                    "scan_dir": str(row.scan_dir),
                    **{key: str(value) for key, value in exported.items()},
                }
            )
        return exports


__all__ = [
    "DISPLAY_MODES",
    "ExportConfig",
    "PlateRasterSynchronyViewer",
    "ViewerConfig",
    "build_run_manifest",
    "compute_plate_height_px",
    "discover_well_records",
]

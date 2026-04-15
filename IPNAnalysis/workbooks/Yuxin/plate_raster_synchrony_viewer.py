from __future__ import annotations

import copy
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
from plotly.offline.offline import get_plotlyjs
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
DEFAULT_COMBINED_HTML_STEM = "plate_raster_synchrony_all_scans"
TIME_DECIMALS = 3
VALUE_DECIMALS = 4


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
    combined_html_stem: str = DEFAULT_COMBINED_HTML_STEM

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


def _axis_ref(prefix: str, axis_index: int) -> str:
    return prefix if axis_index == 1 else f"{prefix}{axis_index}"


def _layout_axis_name(prefix: str, axis_index: int) -> str:
    return f"{prefix}axis" if axis_index == 1 else f"{prefix}axis{axis_index}"


def _subplot_axis_index(row: int, col: int) -> int:
    return ((row - 1) * PLATE_COLS) + col


def _primary_yaxis_number(panel_index: int) -> int:
    return (2 * panel_index) - 1


def _secondary_yaxis_number(panel_index: int) -> int:
    return 2 * panel_index


def _build_panel_metadata() -> tuple[dict[str, Any], ...]:
    panels: list[dict[str, Any]] = []
    for well_num in range(PLATE_WELL_COUNT):
        well_id = f"well{well_num:03d}"
        row, col, plate_label = well_to_plate_position(well_id)
        panel_index = _subplot_axis_index(row, col)
        xaxis_ref = _axis_ref("x", panel_index)
        primary_y_ref = _axis_ref("y", _primary_yaxis_number(panel_index))
        secondary_y_ref = _axis_ref("y", _secondary_yaxis_number(panel_index))
        panels.append(
            {
                "panel_index": panel_index,
                "well_id": well_id,
                "plate_label": plate_label,
                "row": row,
                "col": col,
                "xaxis_ref": xaxis_ref,
                "primary_yaxis_ref": primary_y_ref,
                "secondary_yaxis_ref": secondary_y_ref,
                "xaxis_layout_key": _layout_axis_name("x", panel_index),
                "primary_yaxis_layout_key": _layout_axis_name("y", _primary_yaxis_number(panel_index)),
                "secondary_yaxis_layout_key": _layout_axis_name("y", _secondary_yaxis_number(panel_index)),
                "x_domain_ref": f"{xaxis_ref} domain",
                "primary_y_domain_ref": f"{primary_y_ref} domain",
            }
        )
    return tuple(panels)


PANEL_METADATA = _build_panel_metadata()


def _serialize_numeric_array(values: np.ndarray, *, decimals: int | None = None) -> list[float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return []
    if decimals is not None:
        arr = np.round(arr.astype(np.float32), decimals=decimals)
    return arr.astype(float).tolist()


def _max_window_for_xmax(global_xmax: float) -> int:
    if global_xmax <= 0:
        return 0
    return max(1, int(np.floor(global_xmax)))


def _clamp_window_end(global_xmax: float, requested_window_s: float | None) -> float:
    max_window = _max_window_for_xmax(global_xmax)
    if max_window <= 0:
        return 0.0
    if requested_window_s is None or requested_window_s <= 0:
        return float(max_window)
    return float(max(1, min(max_window, int(round(requested_window_s)))))


def _xaxis_range_layout_key(axis_index: int) -> str:
    return f"{_layout_axis_name('x', axis_index)}.range"


def _xaxis_window_relayout(window_end: float) -> dict[str, list[float]]:
    return {
        _xaxis_range_layout_key(axis_index): [0.0, float(window_end)]
        for axis_index in range(1, PLATE_WELL_COUNT + 1)
    }


def _build_xspan_slider(global_xmax: float, initial_window_s: float | None) -> tuple[list[dict[str, Any]], float]:
    max_window = _max_window_for_xmax(global_xmax)
    if max_window <= 0:
        return [], 0.0

    window_values = list(range(1, max_window + 1))
    initial_window = int(_clamp_window_end(global_xmax, initial_window_s))
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
    subplot_titles = [panel["plate_label"] for panel in PANEL_METADATA]
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


def _configure_static_plate_figure(fig: go.Figure) -> None:
    for annotation, panel_meta in zip(fig.layout.annotations, PANEL_METADATA):
        annotation.text = f"{panel_meta['plate_label']}<br>{panel_meta['well_id']}"
        annotation.font = dict(size=13)

    fig.update_xaxes(matches="x", showgrid=True, rangeslider_visible=False)
    fig.update_yaxes(matches=None, fixedrange=True)

    for panel_meta in PANEL_METADATA:
        row = int(panel_meta["row"])
        col = int(panel_meta["col"])
        fig.update_yaxes(
            title_text="Unit" if col == 1 else None,
            fixedrange=True,
            row=row,
            col=col,
            secondary_y=False,
        )
        fig.update_yaxes(
            title_text="Sync" if col == PLATE_COLS else None,
            showgrid=False,
            zeroline=False,
            fixedrange=True,
            row=row,
            col=col,
            secondary_y=True,
        )
        fig.update_xaxes(title_text="Time (s)" if row == PLATE_ROWS else None, row=row, col=col)


def _sanitize_filename_part(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    text = text.strip("._")
    return text or "scan"


def _scan_title_text(scan_payload: dict[str, Any]) -> str:
    return (
        f"Plate Raster + Synchrony: {scan_payload['scan_label']}"
        f"<br><sup>run_id={scan_payload['run_id']} | scan_id={scan_payload['scan_id']}</sup>"
    )


def _panel_status_to_annotation(panel_meta: dict[str, Any], status: str | None) -> dict[str, Any] | None:
    if not status:
        return None

    annotation = {
        "xref": panel_meta["x_domain_ref"],
        "yref": panel_meta["primary_y_domain_ref"],
        "showarrow": False,
        "font": {"color": "gray"},
    }

    if status == "missing_well":
        annotation.update({"x": 0.5, "y": 0.5, "text": "Missing well", "xanchor": "center", "font": {"size": 12, "color": "gray"}})
        return annotation
    if status == "no_plot_data":
        annotation.update({"x": 0.5, "y": 0.5, "text": "No plot data", "xanchor": "center", "font": {"size": 12, "color": "gray"}})
        return annotation
    if status == "raster_missing":
        annotation.update({"x": 0.98, "y": 0.95, "text": "raster missing", "xanchor": "right", "font": {"size": 10, "color": "gray"}})
        return annotation
    if status == "synchrony_missing":
        annotation.update({"x": 0.98, "y": 0.95, "text": "synchrony missing", "xanchor": "right", "font": {"size": 10, "color": "gray"}})
        return annotation
    return None


def _raster_payload_for_well(
    spike_times: dict[str, np.ndarray],
    well_id: str,
    plate_label: str,
    unit_sort_mode: str,
    max_points_per_well: int | None,
) -> dict[str, Any]:
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
        return {
            "traces": [],
            "xmax": max_time,
            "unit_count": max(len(units), 1),
        }

    total_points = sum(spikes.size for _, _, spikes in nonempty)
    stride = 1
    if max_points_per_well is not None and max_points_per_well > 0 and total_points > max_points_per_well:
        stride = int(np.ceil(total_points / max_points_per_well))

    traces: list[dict[str, Any]] = []
    unit_count = len(nonempty)
    for rank, (_, unit_name, spikes) in enumerate(nonempty, start=1):
        sampled = spikes[::stride] if stride > 1 else spikes
        if sampled.size == 0:
            continue
        traces.append(
            {
                "x": _serialize_numeric_array(sampled, decimals=TIME_DECIMALS),
                "rank": rank,
                "hover_label": f"{plate_label} / {well_id}<br>Unit {unit_name} ({rank}/{unit_count})",
            }
        )

    return {
        "traces": traces,
        "xmax": max_time,
        "unit_count": max(unit_count, 1),
    }


def _synchrony_payload_for_well(
    plot_data: dict[str, Any],
    max_points: int | None,
) -> dict[str, Any]:
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
    smooth_x, smooth_y = downsample_xy(t, smooth, max_points=max_points)

    peak_n = min(peak_t.size, peak_y.size)
    if peak_n:
        peak_t = downsample_values(peak_t[:peak_n], max_points=max_points)
        peak_y = downsample_values(peak_y[:peak_n], max_points=max_points)
    else:
        peak_t = np.array([], dtype=float)
        peak_y = np.array([], dtype=float)

    has_lines = bool(t.size and (baseline is not None or threshold is not None))
    has_any = bool(
        (signal_x.size and signal_y.size)
        or (smooth_x.size and smooth_y.size)
        or (peak_t.size and peak_y.size)
        or has_lines
    )

    return {
        "signal": {
            "x": _serialize_numeric_array(signal_x, decimals=TIME_DECIMALS),
            "y": _serialize_numeric_array(signal_y, decimals=VALUE_DECIMALS),
        },
        "smooth": {
            "x": _serialize_numeric_array(smooth_x, decimals=TIME_DECIMALS),
            "y": _serialize_numeric_array(smooth_y, decimals=VALUE_DECIMALS),
        },
        "peaks": {
            "x": _serialize_numeric_array(peak_t, decimals=TIME_DECIMALS),
            "y": _serialize_numeric_array(peak_y, decimals=VALUE_DECIMALS),
        },
        "baseline": float(baseline) if baseline is not None else None,
        "threshold": float(threshold) if threshold is not None else None,
        "line_start": float(t[0]) if t.size else 0.0,
        "xmax": xmax,
        "ymax": max(ymax, 1.0),
        "has_any": has_any,
    }


def build_scan_payload(scan_df: pd.DataFrame, viewer_config: ViewerConfig) -> dict[str, Any]:
    _validate_display_mode(viewer_config.display_mode)
    if scan_df.empty:
        raise ValueError("scan_df must contain at least one well record.")

    records_by_well = {str(row["well_id"]): row for row in scan_df.to_dict(orient="records")}
    global_xmax = 0.0
    panels: list[dict[str, Any]] = []

    for panel_meta in PANEL_METADATA:
        well_id = str(panel_meta["well_id"])
        record = records_by_well.get(well_id)

        if record is None:
            panels.append(
                {
                    "primary_ymax": 1.0,
                    "secondary_ymax": 1.0,
                    "raster": [],
                    "synchrony": {
                        "signal": {"x": [], "y": []},
                        "smooth": {"x": [], "y": []},
                        "peaks": {"x": [], "y": []},
                        "baseline": None,
                        "threshold": None,
                        "line_start": 0.0,
                        "xmax": 0.0,
                        "ymax": 1.0,
                        "has_any": False,
                    },
                    "status": "missing_well",
                }
            )
            continue

        raster_payload = {"traces": [], "xmax": 0.0, "unit_count": 1}
        synchrony_payload = {
            "signal": {"x": [], "y": []},
            "smooth": {"x": [], "y": []},
            "peaks": {"x": [], "y": []},
            "baseline": None,
            "threshold": None,
            "line_start": 0.0,
            "xmax": 0.0,
            "ymax": 1.0,
            "has_any": False,
        }

        raster_loaded = False
        synchrony_loaded = False

        if bool(record["has_spike_times"]):
            raster_payload = _raster_payload_for_well(
                load_spike_times(record["spike_times_path"]),
                well_id=well_id,
                plate_label=str(panel_meta["plate_label"]),
                unit_sort_mode=viewer_config.unit_sort_mode,
                max_points_per_well=viewer_config.max_raster_points_per_well,
            )
            raster_loaded = bool(raster_payload["traces"])
            global_xmax = max(global_xmax, float(raster_payload["xmax"]))

        if bool(record["has_network_json"]):
            synchrony_payload = _synchrony_payload_for_well(
                load_network_plot_data(record["network_json_path"]),
                max_points=viewer_config.max_synchrony_points,
            )
            synchrony_loaded = bool(synchrony_payload["has_any"])
            global_xmax = max(global_xmax, float(synchrony_payload["xmax"]))

        status: str | None = None
        if not raster_loaded and not synchrony_loaded:
            status = "no_plot_data"
        elif not raster_loaded:
            status = "raster_missing"
        elif not synchrony_loaded:
            status = "synchrony_missing"

        primary_ymax = float(raster_payload["unit_count"] + 1) if bool(record["has_spike_times"]) else 1.0
        secondary_ymax = float(synchrony_payload["ymax"]) * 1.1 if bool(record["has_network_json"]) else 1.0

        panels.append(
            {
                "primary_ymax": max(primary_ymax, 1.0),
                "secondary_ymax": max(secondary_ymax, 1.0),
                "raster": raster_payload["traces"],
                "synchrony": synchrony_payload,
                "status": status,
            }
        )

    first_row = scan_df.iloc[0]
    return {
        "scan_dir": str(first_row["scan_dir"]),
        "scan_id": str(first_row["scan_id"]),
        "run_id": str(first_row["run_id"]),
        "scan_label": str(first_row["scan_label"]),
        "global_xmax": float(global_xmax),
        "panels": panels,
    }


def _build_scan_figure(
    scan_payload: dict[str, Any],
    viewer_config: ViewerConfig,
    *,
    display_mode: str | None = None,
    window_end: float | None = None,
    include_embedded_controls: bool = True,
) -> go.Figure:
    effective_mode = display_mode or viewer_config.display_mode
    _validate_display_mode(effective_mode)

    fig = _make_plate_figure(
        title=_scan_title_text(scan_payload),
        width_px=viewer_config.width_px,
        height_px=viewer_config.height_px,
    )
    _configure_static_plate_figure(fig)

    trace_roles: list[str] = []
    raster_legend_shown = False
    synchrony_legend_shown = False

    for panel_meta, panel_payload in zip(PANEL_METADATA, scan_payload["panels"]):
        row = int(panel_meta["row"])
        col = int(panel_meta["col"])

        should_add_raster = include_embedded_controls or effective_mode in {"raster", "both"}
        should_add_synchrony = include_embedded_controls or effective_mode in {"synchrony", "both"}

        if should_add_raster:
            for raster_trace in panel_payload["raster"]:
                x_values = np.asarray(raster_trace["x"], dtype=float)
                if x_values.size == 0:
                    continue
                trace = go.Scattergl(
                    x=x_values,
                    y=np.full(x_values.shape, float(raster_trace["rank"]), dtype=float),
                    mode="markers",
                    marker=dict(
                        symbol=RASTER_MARKER_SYMBOL,
                        size=float(viewer_config.marker_size),
                        color=RASTER_MARKER_COLOR,
                    ),
                    hovertemplate=f"{raster_trace['hover_label']}<br>t=%{{x:.3f}} s<extra></extra>",
                    name="Raster",
                    legendgroup="raster",
                    showlegend=not raster_legend_shown,
                )
                raster_legend_shown = True
                fig.add_trace(trace, row=row, col=col, secondary_y=False)
                trace_roles.append("raster")

        if should_add_synchrony:
            sync_payload = panel_payload["synchrony"]
            signal_x = np.asarray(sync_payload["signal"]["x"], dtype=float)
            signal_y = np.asarray(sync_payload["signal"]["y"], dtype=float)
            smooth_x = np.asarray(sync_payload["smooth"]["x"], dtype=float)
            smooth_y = np.asarray(sync_payload["smooth"]["y"], dtype=float)
            peak_x = np.asarray(sync_payload["peaks"]["x"], dtype=float)
            peak_y = np.asarray(sync_payload["peaks"]["y"], dtype=float)
            hover_prefix = f"{panel_meta['plate_label']} / {panel_meta['well_id']}"

            if signal_x.size and signal_y.size:
                fig.add_trace(
                    go.Scattergl(
                        x=signal_x,
                        y=signal_y,
                        mode="lines",
                        line=dict(color="#b22222", width=max(0.5, float(viewer_config.line_width))),
                        hovertemplate=f"{hover_prefix}<br>Synchrony=%{{y:.3f}}<br>t=%{{x:.3f}} s<extra></extra>",
                        name="Synchrony",
                        legendgroup="synchrony",
                        showlegend=not synchrony_legend_shown,
                    ),
                    row=row,
                    col=col,
                    secondary_y=True,
                )
                synchrony_legend_shown = True
                trace_roles.append("synchrony")

            if smooth_x.size and smooth_y.size:
                fig.add_trace(
                    go.Scattergl(
                        x=smooth_x,
                        y=smooth_y,
                        mode="lines",
                        line=dict(color="rgba(255, 140, 0, 0.95)", width=max(0.5, float(viewer_config.line_width) - 0.1)),
                        hovertemplate=f"{hover_prefix}<br>Smooth synchrony=%{{y:.3f}}<br>t=%{{x:.3f}} s<extra></extra>",
                        name="Synchrony smooth",
                        legendgroup="synchrony",
                        showlegend=not synchrony_legend_shown,
                    ),
                    row=row,
                    col=col,
                    secondary_y=True,
                )
                synchrony_legend_shown = True
                trace_roles.append("synchrony")

            if peak_x.size and peak_y.size:
                fig.add_trace(
                    go.Scattergl(
                        x=peak_x,
                        y=peak_y,
                        mode="markers",
                        marker=dict(color="red", size=4),
                        hovertemplate=f"{hover_prefix}<br>Burst peak=%{{y:.3f}}<br>t=%{{x:.3f}} s<extra></extra>",
                        name="Burst peaks",
                        legendgroup="synchrony",
                        showlegend=not synchrony_legend_shown,
                    ),
                    row=row,
                    col=col,
                    secondary_y=True,
                )
                synchrony_legend_shown = True
                trace_roles.append("synchrony")

            line_x = np.asarray([sync_payload["line_start"], sync_payload["xmax"]], dtype=float)
            if line_x.size == 2 and sync_payload["baseline"] is not None:
                baseline_value = float(sync_payload["baseline"])
                fig.add_trace(
                    go.Scatter(
                        x=line_x,
                        y=np.asarray([baseline_value, baseline_value], dtype=float),
                        mode="lines",
                        line=dict(color="rgba(255, 102, 0, 0.7)", width=1, dash="dash"),
                        hoverinfo="skip",
                        name="Baseline",
                        legendgroup="synchrony",
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                    secondary_y=True,
                )
                trace_roles.append("synchrony")

            if line_x.size == 2 and sync_payload["threshold"] is not None:
                threshold_value = float(sync_payload["threshold"])
                fig.add_trace(
                    go.Scatter(
                        x=line_x,
                        y=np.asarray([threshold_value, threshold_value], dtype=float),
                        mode="lines",
                        line=dict(color="rgba(192, 57, 43, 0.8)", width=1, dash="dash"),
                        hoverinfo="skip",
                        name="Threshold",
                        legendgroup="synchrony",
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                    secondary_y=True,
                )
                trace_roles.append("synchrony")

        fig.update_yaxes(
            range=[0.0, float(panel_payload["primary_ymax"])],
            row=row,
            col=col,
            secondary_y=False,
        )
        fig.update_yaxes(
            range=[0.0, float(panel_payload["secondary_ymax"])],
            row=row,
            col=col,
            secondary_y=True,
        )

        annotation = _panel_status_to_annotation(panel_meta, panel_payload["status"])
        if annotation is not None:
            fig.add_annotation(**annotation)

    resolved_window_end = _clamp_window_end(scan_payload["global_xmax"], window_end or viewer_config.initial_window_s)
    if resolved_window_end > 0:
        fig.update_xaxes(range=[0.0, resolved_window_end])

    fig.update_layout(
        title=dict(
            text=_scan_title_text(scan_payload),
            x=0.5,
            xanchor="center",
            y=TITLE_Y,
            yanchor="top",
            font=dict(size=TITLE_FONT_SIZE),
        )
    )

    if include_embedded_controls:
        sliders, window_end_from_slider = _build_xspan_slider(scan_payload["global_xmax"], window_end or viewer_config.initial_window_s)
        if window_end_from_slider > 0:
            fig.update_xaxes(range=[0.0, window_end_from_slider])

        mode_visibility = {
            "both": [True for _ in trace_roles],
            "raster": [role == "raster" for role in trace_roles],
            "synchrony": [role == "synchrony" for role in trace_roles],
        }
        for trace, visible in zip(fig.data, mode_visibility[effective_mode]):
            trace.visible = visible

        fig.update_layout(
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
    else:
        fig.update_layout(sliders=[], updatemenus=[])

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


def _json_for_html(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")


def _combined_manifest_records(manifest_df: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in manifest_df.itertuples(index=False):
        records.append(
            {
                "run_id": str(row.run_id),
                "scan_id": str(row.scan_id),
                "scan_label": str(row.scan_label),
                "scan_dir": str(row.scan_dir),
                "n_wells": int(row.n_wells),
                "missing_spike_times": int(row.missing_spike_times),
                "missing_network_json": int(row.missing_network_json),
            }
        )
    return records


def _combined_viewer_layout(viewer_config: ViewerConfig) -> dict[str, Any]:
    base_fig = _make_plate_figure(
        title="",
        width_px=viewer_config.width_px,
        height_px=viewer_config.height_px,
    )
    _configure_static_plate_figure(base_fig)
    base_fig.update_layout(title=dict(text=""), sliders=[], updatemenus=[])
    return base_fig.to_plotly_json()["layout"]


def _build_combined_viewer_html(
    manifest_records: list[dict[str, Any]],
    scan_payloads: list[dict[str, Any]],
    viewer_config: ViewerConfig,
    *,
    initial_index: int,
) -> str:
    layout_payload = _combined_viewer_layout(viewer_config)
    viewer_payload = {
        "display_mode": viewer_config.display_mode,
        "marker_size": float(viewer_config.marker_size),
        "line_width": float(viewer_config.line_width),
        "width_px": int(viewer_config.width_px),
        "height_px": int(_resolve_figure_height_px(viewer_config.width_px, viewer_config.height_px)),
        "initial_window_s": float(viewer_config.initial_window_s or 0.0),
        "raster_marker_symbol": RASTER_MARKER_SYMBOL,
        "raster_marker_color": RASTER_MARKER_COLOR,
    }

    template = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Plate Raster Synchrony Viewer</title>
  <style>
    :root {
      --border: rgba(0, 0, 0, 0.15);
      --surface: #ffffff;
      --surface-alt: #f5f7fa;
      --text: #1f2933;
      --muted: #52606d;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--surface-alt);
      color: var(--text);
      font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
    }
    .viewer-shell {
      padding: 16px 18px 20px;
    }
    .toolbar {
      display: flex;
      gap: 16px;
      align-items: flex-end;
      flex-wrap: wrap;
      margin-bottom: 14px;
      padding: 14px;
      border: 1px solid var(--border);
      background: var(--surface);
      border-radius: 10px;
    }
    .toolbar-group {
      display: flex;
      flex-direction: column;
      gap: 6px;
      min-width: 180px;
    }
    .toolbar-group label {
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.03em;
      text-transform: uppercase;
      color: var(--muted);
    }
    .scan-controls,
    .xspan-controls {
      display: flex;
      gap: 8px;
      align-items: center;
    }
    .toolbar select,
    .toolbar button,
    .toolbar input {
      font: inherit;
    }
    .toolbar select,
    .toolbar button {
      height: 36px;
      padding: 0 12px;
      border-radius: 8px;
      border: 1px solid var(--border);
      background: var(--surface);
      color: var(--text);
    }
    .toolbar button {
      cursor: pointer;
      font-weight: 600;
    }
    .toolbar button:disabled {
      cursor: default;
      opacity: 0.45;
    }
    #scan-select {
      min-width: 420px;
    }
    .xspan-group {
      min-width: 280px;
      flex: 1 1 320px;
    }
    #x-span {
      width: 100%;
    }
    #x-span-value {
      min-width: 3ch;
      text-align: right;
      font-variant-numeric: tabular-nums;
      font-weight: 700;
    }
    .status {
      margin-bottom: 12px;
      padding: 12px 14px;
      border: 1px solid var(--border);
      border-radius: 10px;
      background: var(--surface);
      font-size: 14px;
      line-height: 1.5;
    }
    .plot-shell {
      width: 100%;
      overflow-x: auto;
      overflow-y: hidden;
      border: 1px solid var(--border);
      border-radius: 10px;
      background: var(--surface);
      padding: 8px 0;
    }
    #viewer-plot {
      min-width: 1200px;
    }
  </style>
</head>
<body>
  <div class="viewer-shell">
    <div class="toolbar">
      <div class="toolbar-group">
        <label for="scan-select">Scan</label>
        <div class="scan-controls">
          <button id="prev-scan" type="button">Previous</button>
          <select id="scan-select"></select>
          <button id="next-scan" type="button">Next</button>
        </div>
      </div>
      <div class="toolbar-group">
        <label for="mode-select">Mode</label>
        <select id="mode-select">
          <option value="both">Both</option>
          <option value="raster">Raster only</option>
          <option value="synchrony">Synchrony only</option>
        </select>
      </div>
      <div class="toolbar-group xspan-group">
        <label for="x-span">X span (s)</label>
        <div class="xspan-controls">
          <input id="x-span" type="range" min="1" max="1" step="1" value="1" />
          <span id="x-span-value">1</span>
        </div>
      </div>
    </div>
    <div id="viewer-status" class="status"></div>
    <div class="plot-shell">
      <div id="viewer-plot"></div>
    </div>
  </div>
  <script>__PLOTLY_JS__</script>
  <script>
    const VIEWER_CONFIG = __VIEWER_CONFIG__;
    const PANEL_METADATA = __PANEL_METADATA__;
    const MANIFEST = __MANIFEST__;
    const SCAN_PAYLOADS = __SCAN_PAYLOADS__;
    const BASE_LAYOUT = __BASE_LAYOUT__;
    const INITIAL_INDEX = __INITIAL_INDEX__;

    const plotRoot = document.getElementById("viewer-plot");
    const scanSelect = document.getElementById("scan-select");
    const prevButton = document.getElementById("prev-scan");
    const nextButton = document.getElementById("next-scan");
    const modeSelect = document.getElementById("mode-select");
    const xSpanInput = document.getElementById("x-span");
    const xSpanValue = document.getElementById("x-span-value");
    const statusRoot = document.getElementById("viewer-status");

    const state = {
      scanIndex: Math.max(0, Math.min(INITIAL_INDEX, MANIFEST.length - 1)),
      displayMode: VIEWER_CONFIG.display_mode || "both",
      xSpan: null,
    };

    function deepClone(value) {
      return JSON.parse(JSON.stringify(value));
    }

    function escapeHtml(value) {
      return String(value ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
    }

    function maxWindowForXmax(globalXmax) {
      if (!(globalXmax > 0)) {
        return 0;
      }
      return Math.max(1, Math.floor(globalXmax));
    }

    function clampWindow(globalXmax, requested) {
      const maxWindow = maxWindowForXmax(globalXmax);
      if (maxWindow <= 0) {
        return 0;
      }
      const numeric = Number(requested);
      if (!(numeric > 0)) {
        return maxWindow;
      }
      return Math.max(1, Math.min(maxWindow, Math.round(numeric)));
    }

    function setStatus(manifestRow) {
      statusRoot.innerHTML =
        "<div>" +
        "<b>run_id:</b> " + escapeHtml(manifestRow.run_id) +
        " &nbsp; <b>scan_id:</b> " + escapeHtml(manifestRow.scan_id) +
        " &nbsp; <b>wells:</b> " + escapeHtml(manifestRow.n_wells) +
        " &nbsp; <b>missing spikes:</b> " + escapeHtml(manifestRow.missing_spike_times) +
        " &nbsp; <b>missing synchrony:</b> " + escapeHtml(manifestRow.missing_network_json) +
        "<br><code>" + escapeHtml(manifestRow.scan_label) + "</code>" +
        "</div>";
    }

    function panelStatusAnnotation(panelMeta, status) {
      if (!status) {
        return null;
      }
      if (status === "missing_well") {
        return {
          x: 0.5,
          y: 0.5,
          xref: panelMeta.x_domain_ref,
          yref: panelMeta.primary_y_domain_ref,
          text: "Missing well",
          showarrow: false,
          xanchor: "center",
          font: { size: 12, color: "gray" },
        };
      }
      if (status === "no_plot_data") {
        return {
          x: 0.5,
          y: 0.5,
          xref: panelMeta.x_domain_ref,
          yref: panelMeta.primary_y_domain_ref,
          text: "No plot data",
          showarrow: false,
          xanchor: "center",
          font: { size: 12, color: "gray" },
        };
      }
      if (status === "raster_missing") {
        return {
          x: 0.98,
          y: 0.95,
          xref: panelMeta.x_domain_ref,
          yref: panelMeta.primary_y_domain_ref,
          text: "raster missing",
          showarrow: false,
          xanchor: "right",
          font: { size: 10, color: "gray" },
        };
      }
      if (status === "synchrony_missing") {
        return {
          x: 0.98,
          y: 0.95,
          xref: panelMeta.x_domain_ref,
          yref: panelMeta.primary_y_domain_ref,
          text: "synchrony missing",
          showarrow: false,
          xanchor: "right",
          font: { size: 10, color: "gray" },
        };
      }
      return null;
    }

    function claimLegend(seen, groupName) {
      if (seen[groupName]) {
        return false;
      }
      seen[groupName] = true;
      return true;
    }

    function buildFigure(scanPayload) {
      const layout = deepClone(BASE_LAYOUT);
      const annotations = (BASE_LAYOUT.annotations || []).slice();
      const data = [];
      const legendState = { raster: false, synchrony: false };
      const windowEnd = state.xSpan > 0 ? state.xSpan : Math.max(1, maxWindowForXmax(scanPayload.global_xmax));

      for (let index = 0; index < PANEL_METADATA.length; index += 1) {
        const panelMeta = PANEL_METADATA[index];
        const panelPayload = scanPayload.panels[index];

        layout[panelMeta.xaxis_layout_key].range = [0, windowEnd];
        layout[panelMeta.primary_yaxis_layout_key].range = [0, panelPayload.primary_ymax];
        layout[panelMeta.secondary_yaxis_layout_key].range = [0, panelPayload.secondary_ymax];

        if (state.displayMode !== "synchrony") {
          for (const rasterTrace of panelPayload.raster) {
            if (!rasterTrace.x.length) {
              continue;
            }
            data.push({
              type: "scattergl",
              x: rasterTrace.x,
              y: Array(rasterTrace.x.length).fill(rasterTrace.rank),
              mode: "markers",
              marker: {
                symbol: VIEWER_CONFIG.raster_marker_symbol,
                size: VIEWER_CONFIG.marker_size,
                color: VIEWER_CONFIG.raster_marker_color,
              },
              hovertemplate: rasterTrace.hover_label + "<br>t=%{x:.3f} s<extra></extra>",
              name: "Raster",
              legendgroup: "raster",
              showlegend: claimLegend(legendState, "raster"),
              xaxis: panelMeta.xaxis_ref,
              yaxis: panelMeta.primary_yaxis_ref,
            });
          }
        }

        if (state.displayMode !== "raster") {
          const syncPayload = panelPayload.synchrony;
          const hoverPrefix = panelMeta.plate_label + " / " + panelMeta.well_id;

          if (syncPayload.signal.x.length && syncPayload.signal.y.length) {
            data.push({
              type: "scattergl",
              x: syncPayload.signal.x,
              y: syncPayload.signal.y,
              mode: "lines",
              line: { color: "#b22222", width: Math.max(0.5, VIEWER_CONFIG.line_width) },
              hovertemplate: hoverPrefix + "<br>Synchrony=%{y:.3f}<br>t=%{x:.3f} s<extra></extra>",
              name: "Synchrony",
              legendgroup: "synchrony",
              showlegend: claimLegend(legendState, "synchrony"),
              xaxis: panelMeta.xaxis_ref,
              yaxis: panelMeta.secondary_yaxis_ref,
            });
          }

          if (syncPayload.smooth.x.length && syncPayload.smooth.y.length) {
            data.push({
              type: "scattergl",
              x: syncPayload.smooth.x,
              y: syncPayload.smooth.y,
              mode: "lines",
              line: {
                color: "rgba(255, 140, 0, 0.95)",
                width: Math.max(0.5, VIEWER_CONFIG.line_width - 0.1),
              },
              hovertemplate: hoverPrefix + "<br>Smooth synchrony=%{y:.3f}<br>t=%{x:.3f} s<extra></extra>",
              name: "Synchrony smooth",
              legendgroup: "synchrony",
              showlegend: claimLegend(legendState, "synchrony"),
              xaxis: panelMeta.xaxis_ref,
              yaxis: panelMeta.secondary_yaxis_ref,
            });
          }

          if (syncPayload.peaks.x.length && syncPayload.peaks.y.length) {
            data.push({
              type: "scattergl",
              x: syncPayload.peaks.x,
              y: syncPayload.peaks.y,
              mode: "markers",
              marker: { color: "red", size: 4 },
              hovertemplate: hoverPrefix + "<br>Burst peak=%{y:.3f}<br>t=%{x:.3f} s<extra></extra>",
              name: "Burst peaks",
              legendgroup: "synchrony",
              showlegend: claimLegend(legendState, "synchrony"),
              xaxis: panelMeta.xaxis_ref,
              yaxis: panelMeta.secondary_yaxis_ref,
            });
          }

          if (syncPayload.baseline !== null) {
            data.push({
              type: "scatter",
              x: [syncPayload.line_start, syncPayload.xmax],
              y: [syncPayload.baseline, syncPayload.baseline],
              mode: "lines",
              line: { color: "rgba(255, 102, 0, 0.7)", width: 1, dash: "dash" },
              hoverinfo: "skip",
              name: "Baseline",
              legendgroup: "synchrony",
              showlegend: false,
              xaxis: panelMeta.xaxis_ref,
              yaxis: panelMeta.secondary_yaxis_ref,
            });
          }

          if (syncPayload.threshold !== null) {
            data.push({
              type: "scatter",
              x: [syncPayload.line_start, syncPayload.xmax],
              y: [syncPayload.threshold, syncPayload.threshold],
              mode: "lines",
              line: { color: "rgba(192, 57, 43, 0.8)", width: 1, dash: "dash" },
              hoverinfo: "skip",
              name: "Threshold",
              legendgroup: "synchrony",
              showlegend: false,
              xaxis: panelMeta.xaxis_ref,
              yaxis: panelMeta.secondary_yaxis_ref,
            });
          }
        }

        const annotation = panelStatusAnnotation(panelMeta, panelPayload.status);
        if (annotation) {
          annotations.push(annotation);
        }
      }

      layout.annotations = annotations;
      layout.title.text =
        "Plate Raster + Synchrony: " + scanPayload.scan_label +
        "<br><sup>run_id=" + scanPayload.run_id + " | scan_id=" + scanPayload.scan_id + "</sup>";

      return { data, layout };
    }

    function syncXSpanControl(scanPayload) {
      const maxWindow = maxWindowForXmax(scanPayload.global_xmax);
      state.xSpan = clampWindow(scanPayload.global_xmax, state.xSpan ?? VIEWER_CONFIG.initial_window_s);
      xSpanInput.max = String(Math.max(1, maxWindow));
      xSpanInput.value = String(Math.max(1, state.xSpan || 1));
      xSpanInput.disabled = maxWindow <= 0;
      xSpanValue.textContent = String(Math.max(1, state.xSpan || 1));
    }

    function updateNavigation() {
      scanSelect.value = String(state.scanIndex);
      prevButton.disabled = state.scanIndex <= 0;
      nextButton.disabled = state.scanIndex >= MANIFEST.length - 1;
    }

    function renderCurrentScan() {
      const manifestRow = MANIFEST[state.scanIndex];
      const scanPayload = SCAN_PAYLOADS[state.scanIndex];
      syncXSpanControl(scanPayload);
      updateNavigation();
      setStatus(manifestRow);

      const figureSpec = buildFigure(scanPayload);
      Plotly.react(plotRoot, figureSpec.data, figureSpec.layout, {
        responsive: false,
        displaylogo: false,
      });
    }

    for (let index = 0; index < MANIFEST.length; index += 1) {
      const row = MANIFEST[index];
      const option = document.createElement("option");
      option.value = String(index);
      option.textContent = row.run_id + " | " + row.scan_label;
      scanSelect.appendChild(option);
    }

    modeSelect.value = state.displayMode;
    updateNavigation();

    scanSelect.addEventListener("change", function (event) {
      state.scanIndex = Number(event.target.value);
      renderCurrentScan();
    });

    prevButton.addEventListener("click", function () {
      if (state.scanIndex <= 0) {
        return;
      }
      state.scanIndex -= 1;
      renderCurrentScan();
    });

    nextButton.addEventListener("click", function () {
      if (state.scanIndex >= MANIFEST.length - 1) {
        return;
      }
      state.scanIndex += 1;
      renderCurrentScan();
    });

    modeSelect.addEventListener("change", function (event) {
      state.displayMode = event.target.value;
      renderCurrentScan();
    });

    xSpanInput.addEventListener("input", function (event) {
      xSpanValue.textContent = event.target.value;
    });

    xSpanInput.addEventListener("change", function (event) {
      state.xSpan = Number(event.target.value);
      renderCurrentScan();
    });

    renderCurrentScan();
  </script>
</body>
</html>
"""

    html = template
    html = html.replace("__PLOTLY_JS__", get_plotlyjs())
    html = html.replace("__VIEWER_CONFIG__", _json_for_html(viewer_payload))
    html = html.replace("__PANEL_METADATA__", _json_for_html(list(PANEL_METADATA)))
    html = html.replace("__MANIFEST__", _json_for_html(manifest_records))
    html = html.replace("__SCAN_PAYLOADS__", _json_for_html(scan_payloads))
    html = html.replace("__BASE_LAYOUT__", _json_for_html(layout_payload))
    html = html.replace("__INITIAL_INDEX__", str(int(initial_index)))
    return html


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
        self._scan_payload_cache: dict[str, dict[str, Any]] = {}
        self._combined_html_cache: dict[tuple[str, int], str] = {}
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

    def _resolve_initial_scan_dir(self, initial_scan_dir: str | Path | None = None) -> str:
        self._ensure_manifest()
        if initial_scan_dir is None:
            return str(self.manifest_df.iloc[0]["scan_dir"])
        return str(self._resolve_manifest_row(initial_scan_dir)["scan_dir"])

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

    def scan_payload_for_scan(self, scan_dir: str | Path) -> dict[str, Any]:
        row = self._resolve_manifest_row(scan_dir)
        normalized_scan_dir = str(row["scan_dir"])
        if normalized_scan_dir not in self._scan_payload_cache:
            scan_df = self.index_df[self.index_df["scan_dir"] == normalized_scan_dir].copy()
            if scan_df.empty:
                raise ValueError(f"No wells found for scan_dir={normalized_scan_dir!r}")
            self._scan_payload_cache[normalized_scan_dir] = build_scan_payload(scan_df, self.viewer_config)
        return self._scan_payload_cache[normalized_scan_dir]

    def figure_for_scan(
        self,
        scan_dir: str | Path,
        *,
        display_mode: str | None = None,
        x_span_end: float | None = None,
        include_embedded_controls: bool = True,
    ) -> go.Figure:
        row = self._resolve_manifest_row(scan_dir)
        normalized_scan_dir = str(row["scan_dir"])
        self._current_scan_dir = normalized_scan_dir
        return _build_scan_figure(
            self.scan_payload_for_scan(normalized_scan_dir),
            self.viewer_config,
            display_mode=display_mode,
            window_end=x_span_end,
            include_embedded_controls=include_embedded_controls,
        )

    def combined_viewer_html(self, initial_scan_dir: str | Path | None = None) -> str:
        resolved_initial_scan_dir = self._resolve_initial_scan_dir(initial_scan_dir)
        cache_key = (resolved_initial_scan_dir, id(self.viewer_config))
        if cache_key in self._combined_html_cache:
            return self._combined_html_cache[cache_key]

        manifest_records = _combined_manifest_records(self.manifest_df)
        scan_payloads = [self.scan_payload_for_scan(str(row.scan_dir)) for row in self.manifest_df.itertuples(index=False)]
        initial_index = next(
            (index for index, row in enumerate(manifest_records) if row["scan_dir"] == resolved_initial_scan_dir),
            0,
        )

        html = _build_combined_viewer_html(
            manifest_records,
            scan_payloads,
            self.viewer_config,
            initial_index=initial_index,
        )
        self._combined_html_cache[cache_key] = html
        return html

    def render_widget(self, initial_scan_dir: str | Path | None = None) -> Any:
        try:
            import ipywidgets as widgets
            from IPython.display import display
        except ImportError as exc:
            raise ImportError(
                "render_widget requires ipywidgets and IPython display support. "
                "Install them with `pip install ipywidgets`, then restart the notebook kernel."
            ) from exc

        self._ensure_manifest()
        resolved_initial_scan_dir = self._resolve_initial_scan_dir(initial_scan_dir)
        initial_payload = self.scan_payload_for_scan(resolved_initial_scan_dir)
        initial_window_end = int(_clamp_window_end(initial_payload["global_xmax"], self.viewer_config.initial_window_s) or 1)
        initial_raw_max_window = _max_window_for_xmax(initial_payload["global_xmax"])
        initial_max_window = max(1, initial_raw_max_window)

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
        prev_button = widgets.Button(description="Previous", layout=widgets.Layout(width="110px"))
        next_button = widgets.Button(description="Next", layout=widgets.Layout(width="110px"))
        mode_dropdown = widgets.Dropdown(
            options=[("Both", "both"), ("Raster only", "raster"), ("Synchrony only", "synchrony")],
            value=self.viewer_config.display_mode,
            description="mode",
            layout=widgets.Layout(width="250px"),
            style={"description_width": "72px"},
        )
        x_span_slider = widgets.IntSlider(
            value=initial_window_end,
            min=1,
            max=initial_max_window,
            step=1,
            description="x span",
            continuous_update=False,
            layout=widgets.Layout(width="420px"),
            style={"description_width": "72px"},
        )
        x_span_slider.disabled = initial_raw_max_window <= 0
        status_html = widgets.HTML()
        figure_output = widgets.Output(layout=widgets.Layout(width="100%"))

        state = {"suspended": False}

        def _selected_index() -> int:
            scan_dir_value = str(scan_dropdown.value)
            matches = self.manifest_df.index[self.manifest_df["scan_dir"] == scan_dir_value].tolist()
            return int(matches[0]) if matches else 0

        def _sync_navigation_buttons() -> None:
            idx = _selected_index()
            prev_button.disabled = idx <= 0
            next_button.disabled = idx >= (len(self.manifest_df) - 1)

        def _set_status(scan_dir_value: str) -> None:
            row = self._resolve_manifest_row(scan_dir_value)
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

        def _sync_x_span_slider(scan_dir_value: str, *, preserve_current: bool = True) -> None:
            payload = self.scan_payload_for_scan(scan_dir_value)
            current_value = x_span_slider.value if preserve_current else self.viewer_config.initial_window_s
            raw_max_window = _max_window_for_xmax(payload["global_xmax"])
            max_window = max(1, raw_max_window)
            next_value = int(_clamp_window_end(payload["global_xmax"], current_value) or 1)

            state["suspended"] = True
            try:
                x_span_slider.max = max_window
                x_span_slider.value = next_value
                x_span_slider.disabled = raw_max_window <= 0
            finally:
                state["suspended"] = False

        def _render_current() -> None:
            scan_dir_value = str(scan_dropdown.value)
            self._current_scan_dir = scan_dir_value
            _set_status(scan_dir_value)
            _sync_navigation_buttons()
            fig = self.figure_for_scan(
                scan_dir_value,
                display_mode=str(mode_dropdown.value),
                x_span_end=float(x_span_slider.value),
                include_embedded_controls=False,
            )
            with figure_output:
                figure_output.clear_output(wait=True)
                display(fig)

        def _on_scan_change(change: dict[str, Any]) -> None:
            if change.get("name") != "value" or not change.get("new") or state["suspended"]:
                return
            _sync_x_span_slider(str(change["new"]), preserve_current=True)
            _render_current()

        def _on_mode_change(change: dict[str, Any]) -> None:
            if change.get("name") == "value" and change.get("new") and not state["suspended"]:
                _render_current()

        def _on_x_span_change(change: dict[str, Any]) -> None:
            if change.get("name") == "value" and change.get("new") and not state["suspended"]:
                _render_current()

        def _shift_scan(delta: int) -> None:
            idx = _selected_index()
            next_idx = max(0, min(len(self.manifest_df) - 1, idx + delta))
            if next_idx != idx:
                scan_dropdown.value = str(self.manifest_df.iloc[next_idx]["scan_dir"])

        scan_dropdown.observe(_on_scan_change, names="value")
        mode_dropdown.observe(_on_mode_change, names="value")
        x_span_slider.observe(_on_x_span_change, names="value")
        prev_button.on_click(lambda _: _shift_scan(-1))
        next_button.on_click(lambda _: _shift_scan(1))

        _sync_x_span_slider(resolved_initial_scan_dir, preserve_current=False)
        _render_current()

        return widgets.VBox(
            [
                widgets.HTML("<b>Plate viewer</b>"),
                widgets.HBox([prev_button, next_button]),
                scan_dropdown,
                widgets.HBox([mode_dropdown, x_span_slider]),
                status_html,
                figure_output,
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

    def export_each_scan(self, export_config: ExportConfig) -> list[dict[str, str]]:
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

    def export_combined_html(
        self,
        export_config: ExportConfig,
        *,
        stem: str | None = None,
        initial_scan_dir: str | Path | None = None,
    ) -> Path:
        if not export_config.export_html:
            raise ValueError("export_combined_html requires export_html=True in ExportConfig.")

        out_dir = export_config.resolved_output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        export_stem = stem or export_config.combined_html_stem or DEFAULT_COMBINED_HTML_STEM
        html_path = out_dir / f"{_sanitize_filename_part(export_stem)}.html"
        html_path.write_text(self.combined_viewer_html(initial_scan_dir=initial_scan_dir), encoding="utf-8")
        return html_path

    def export_all(self, export_config: ExportConfig) -> dict[str, str]:
        self._ensure_manifest()
        html_path = self.export_combined_html(export_config)
        return {
            "html": str(html_path),
            "scan_count": str(len(self.manifest_df)),
            "run_ids": ",".join(self.manifest_df["run_id"].astype(str).tolist()),
        }


__all__ = [
    "DISPLAY_MODES",
    "ExportConfig",
    "PlateRasterSynchronyViewer",
    "ViewerConfig",
    "build_run_manifest",
    "build_scan_payload",
    "compute_plate_height_px",
    "discover_well_records",
]

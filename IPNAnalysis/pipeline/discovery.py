from __future__ import annotations

import logging
import os
from pathlib import Path
import re

from .model import DiscoveryResult, PipelineConfig, WellTask


def _extract_run_id(file_path: Path) -> str | None:
    if file_path.parent.name.lower() == "network":
        candidate = file_path.parent.parent.name
        if candidate:
            return candidate
    match = re.search(r"/(\d+)/data\.raw\.h5$", str(file_path))
    if match:
        return match.group(1)
    for candidate in (file_path.parent.name, file_path.parent.parent.name):
        if candidate and candidate.isdigit():
            return candidate
    return None


def _find_files_with_subfolder(root_dir: Path, file_name_pattern: str, subfolder_name: str) -> list[Path]:
    matches: list[Path] = []
    for dirpath, _, filenames in os.walk(root_dir):
        path = Path(dirpath)
        if subfolder_name not in path.parts:
            continue
        for filename in filenames:
            if filename == file_name_pattern:
                matches.append(path / filename)
    return matches


def _load_valid_runs(reference_file: str | None, assay_types: list[str] | None) -> tuple[set[int] | None, list[str]]:
    if not reference_file:
        return None, []

    warnings: list[str] = []
    import pandas as pd

    frame = pd.read_excel(reference_file)
    if "Run #" not in frame.columns or "Assay" not in frame.columns:
        raise ValueError(
            f"Reference file '{reference_file}' must contain 'Run #' and 'Assay' columns"
        )
    assay_types = assay_types or []
    filtered = frame[frame["Assay"].str.lower().isin([item.lower() for item in assay_types])]
    valid_runs = set(filtered["Run #"].astype(int).tolist())
    warnings.append(
        f"Reference filter applied: {len(valid_runs)} run(s) selected from {reference_file}"
    )
    return valid_runs, warnings


def _discover_h5_tasks(path: Path, recording_name: str | None, well_id: str | None) -> list[WellTask]:
    tasks: list[WellTask] = []
    import h5py

    with h5py.File(path, "r") as handle:
        if "recordings" in handle:
            recording_keys = list(handle["recordings"].keys())
            for recording in recording_keys:
                if recording_name and recording != recording_name:
                    continue
                well_keys = list(handle["recordings"][recording].keys())
                for candidate_well in well_keys:
                    if well_id and candidate_well != well_id:
                        continue
                    tasks.append(
                        WellTask(
                            source_path=path,
                            file_type="h5",
                            recording_name=recording,
                            well_id=candidate_well,
                            run_id=_extract_run_id(path),
                            file_group="recordings",
                        )
                    )
        elif "wells" in handle:
            recording = recording_name or "rec0000"
            for candidate_well in handle["wells"].keys():
                if well_id and candidate_well != well_id:
                    continue
                tasks.append(
                    WellTask(
                        source_path=path,
                        file_type="h5",
                        recording_name=recording,
                        well_id=candidate_well,
                        run_id=_extract_run_id(path),
                        file_group="wells",
                    )
                )
    return tasks


def discover_tasks(
    source_path: str | Path,
    config: PipelineConfig,
    *,
    recording_name: str | None = None,
    well_id: str | None = None,
    logger: logging.Logger | None = None,
) -> DiscoveryResult:
    source_path = Path(source_path).resolve()
    result = DiscoveryResult(source_path=source_path)
    filters = config.filters
    valid_runs, filter_messages = _load_valid_runs(
        filters.get("reference_file"),
        list(filters.get("assay_types", [])),
    )
    result.warnings.extend(filter_messages)

    if not source_path.exists():
        raise FileNotFoundError(f"Path does not exist: {source_path}")

    paths: list[Path] = []
    if source_path.is_file():
        paths = [source_path]
    else:
        network_name = config.inputs.get("maxwell_network_folder_name", "Network")
        filename_pattern = config.inputs.get("maxwell_h5_filename_pattern", "data.raw.h5")
        h5_files = _find_files_with_subfolder(source_path, filename_pattern, network_name)
        paths.extend(sorted(h5_files))
        paths.extend(sorted(source_path.rglob("*.nwb")))
        paths.extend(sorted(source_path.rglob("*.raw")))

    if not paths:
        raise FileNotFoundError(f"No supported input files found under {source_path}")

    for path in paths:
        suffix = path.suffix.lower()
        if suffix == ".h5":
            run_id = _extract_run_id(path)
            if valid_runs is not None and run_id is not None and run_id.isdigit():
                if int(run_id) not in valid_runs:
                    result.skipped_paths.append(str(path))
                    continue
            try:
                result.tasks.extend(_discover_h5_tasks(path, recording_name, well_id))
            except Exception as exc:
                message = f"Failed to inspect HDF5 file {path}: {exc}"
                result.warnings.append(message)
                if logger is not None:
                    logger.warning(message)
        elif suffix == ".nwb":
            result.warnings.append(f"NWB discovery is not implemented yet: {path}")
        elif suffix == ".raw":
            result.warnings.append(f"Raw binary discovery is not implemented yet: {path}")
        else:
            result.skipped_paths.append(str(path))

    return result


def format_discovery_summary(discovery: DiscoveryResult) -> str:
    lines = [f"Source: {discovery.source_path}", f"Discovered tasks: {len(discovery.tasks)}"]
    for task in discovery.tasks:
        lines.append(f"- {task.source_path} :: {task.recording_name} :: {task.well_id}")
    for warning in discovery.warnings:
        lines.append(f"[warn] {warning}")
    for skipped in discovery.skipped_paths:
        lines.append(f"[skip] {skipped}")
    return os.linesep.join(lines)

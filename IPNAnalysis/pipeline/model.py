from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


STAGE_ORDER: tuple[str, ...] = ("preprocess", "sort", "merge", "analyze", "report")

STAGE_ALIASES = {
    "preprocess": "preprocess",
    "preprocessing": "preprocess",
    "sort": "sort",
    "sorting": "sort",
    "merge": "merge",
    "analyze": "analyze",
    "analyse": "analyze",
    "analysis": "analyze",
    "report": "report",
    "reports": "report",
}


def normalize_stage_name(name: str | None) -> str | None:
    if name is None:
        return None
    token = str(name).strip().lower().replace("-", "_")
    if not token:
        return None
    normalized = STAGE_ALIASES.get(token)
    if normalized is None:
        valid = ", ".join(STAGE_ORDER)
        raise ValueError(f"Invalid stage '{name}'. Valid values: {valid}")
    return normalized


@dataclass(slots=True)
class PipelineConfig:
    source_path: Path | None = None
    format_name: str = "yaml"
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    execution: dict[str, Any] = field(default_factory=dict)
    stages: dict[str, dict[str, Any]] = field(default_factory=dict)
    plugins: dict[str, dict[str, Any]] = field(default_factory=dict)
    filters: dict[str, Any] = field(default_factory=dict)
    compat: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    def stage_enabled(self, stage_name: str) -> bool:
        return bool(self.stages.get(stage_name, {}).get("enabled", True))

    def plugin_enabled(self, plugin_name: str) -> bool:
        return bool(self.plugins.get(plugin_name, {}).get("enabled", False))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.source_path is not None:
            data["source_path"] = str(self.source_path)
        return data


@dataclass(slots=True)
class WellTask:
    source_path: Path
    file_type: str
    recording_name: str
    well_id: str
    run_id: str | None = None
    file_group: str | None = None

    @property
    def task_id(self) -> str:
        return f"{self.source_path}:{self.recording_name}:{self.well_id}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": str(self.source_path),
            "file_type": self.file_type,
            "recording_name": self.recording_name,
            "well_id": self.well_id,
            "run_id": self.run_id,
            "file_group": self.file_group,
            "task_id": self.task_id,
        }


@dataclass(slots=True)
class DiscoveryResult:
    source_path: Path
    tasks: list[WellTask] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    skipped_paths: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": str(self.source_path),
            "tasks": [task.to_dict() for task in self.tasks],
            "warnings": list(self.warnings),
            "skipped_paths": list(self.skipped_paths),
        }


@dataclass(slots=True)
class RunRequest:
    source_path: Path
    config_path: Path | None = None
    output_root: Path | None = None
    checkpoint_root: Path | None = None
    recording_name: str | None = None
    well_id: str | None = None
    from_stage: str | None = None
    to_stage: str | None = None
    dry_run: bool = False
    force_restart: bool = False
    verbose: bool = False
    overrides: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ReanalysisRequest:
    source_path: Path
    config_path: Path | None = None
    plugins: list[str] = field(default_factory=lambda: ["burst_analysis"])
    recursive: bool = True
    verbose: bool = False
    overrides: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RunResult:
    task: WellTask
    output_dir: Path
    selected_stages: list[str] = field(default_factory=list)
    skipped: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task.to_dict(),
            "output_dir": str(self.output_dir),
            "selected_stages": list(self.selected_stages),
            "skipped": self.skipped,
            "details": self.details,
        }

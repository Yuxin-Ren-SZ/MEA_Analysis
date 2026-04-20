from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import yaml

from .artifacts import ArtifactRegistry
from .model import PipelineConfig, ReanalysisRequest, RunRequest, WellTask


def _timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


class PipelineStateStore:
    def __init__(self, artifacts: ArtifactRegistry):
        self.artifacts = artifacts
        self.artifacts.ensure_layout()

    def _load_state(self) -> dict[str, Any]:
        if self.artifacts.state_file.exists():
            return json.loads(self.artifacts.state_file.read_text(encoding="utf-8"))
        return {
            "last_updated": _timestamp(),
            "stages": {},
        }

    def _write_state(self, state: dict[str, Any]) -> None:
        state["last_updated"] = _timestamp()
        self.artifacts.state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")

    def write_manifest(
        self,
        *,
        request: RunRequest | ReanalysisRequest,
        task: WellTask | None = None,
        config: PipelineConfig,
        legacy_artifacts: dict[str, Any] | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "generated_at": _timestamp(),
            "request": {
                "source_path": str(request.source_path),
                "config_path": str(request.config_path) if request.config_path else None,
            },
            "config": config.to_dict(),
            "legacy_artifacts": legacy_artifacts or {},
        }
        if task is not None:
            payload["task"] = task.to_dict()
        with open(self.artifacts.manifest_file, "w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False)

    def record_compat_import(self, payload: dict[str, Any]) -> None:
        self.artifacts.compat_import_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def mark_stage(self, stage_name: str, status: str, details: dict[str, Any] | None = None) -> None:
        details = details or {}
        state = self._load_state()
        entry = {
            "stage": stage_name,
            "status": status,
            "updated_at": _timestamp(),
            "details": details,
        }
        state.setdefault("stages", {})[stage_name] = entry
        self._write_state(state)
        stage_file = self.artifacts.stage_dir / f"{stage_name}.json"
        stage_file.write_text(json.dumps(entry, indent=2), encoding="utf-8")


@dataclass(slots=True)
class RunContext:
    request: RunRequest
    task: WellTask
    config: PipelineConfig
    effective_config: PipelineConfig
    artifacts: ArtifactRegistry
    state_store: PipelineStateStore
    pipeline: Any
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def logger(self):
        return self.pipeline.logger


@dataclass(slots=True)
class LegacyAnalysisContext:
    request: ReanalysisRequest
    output_dir: Path
    config: PipelineConfig
    effective_config: PipelineConfig
    artifacts: ArtifactRegistry
    state_store: PipelineStateStore
    logger: Any
    data: dict[str, Any] = field(default_factory=dict)

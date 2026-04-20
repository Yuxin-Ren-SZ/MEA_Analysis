from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .compat.legacy_outputs import detect_legacy_artifacts, iter_legacy_well_dirs, looks_like_legacy_well_dir


@dataclass(slots=True)
class ArtifactRegistry:
    output_dir: Path
    ipn_root: Path = field(init=False)
    stage_dir: Path = field(init=False)
    log_dir: Path = field(init=False)
    state_file: Path = field(init=False)
    manifest_file: Path = field(init=False)
    compat_import_file: Path = field(init=False)

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir).resolve()
        self.ipn_root = self.output_dir / ".ipn"
        self.stage_dir = self.ipn_root / "stages"
        self.log_dir = self.ipn_root / "logs"
        self.state_file = self.ipn_root / "state.json"
        self.manifest_file = self.ipn_root / "manifest.yaml"
        self.compat_import_file = self.ipn_root / "compat_import.json"

    def ensure_layout(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ipn_root.mkdir(parents=True, exist_ok=True)
        self.stage_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def legacy_artifacts(self) -> dict[str, str]:
        return detect_legacy_artifacts(self.output_dir)


__all__ = ["ArtifactRegistry", "detect_legacy_artifacts", "iter_legacy_well_dirs", "looks_like_legacy_well_dir"]

from __future__ import annotations

from pathlib import Path


LEGACY_FILE_NAMES = {
    "network_results": "network_results.json",
    "spike_times": "spike_times.npy",
    "metrics_curated": "metrics_curated.xlsx",
    "metrics_unfiltered": "qm_unfiltered.xlsx",
    "template_metrics_curated": "tm_curated.xlsx",
    "template_metrics_unfiltered": "tm_unfiltered.xlsx",
    "rejection_log": "rejection_log.xlsx",
}

LEGACY_DIR_NAMES = {
    "preprocessed_zarr": "preprocessed.zarr",
    "preprocessed_binary": "binary",
    "sorter_output": "sorter_output",
    "analyzer_output": "analyzer_output",
    "checkpoints": "checkpoints",
}

SKIP_WALK_DIRS = {
    ".ipn",
    ".git",
    ".pytest_cache",
    "__pycache__",
    "sorter_output",
    "analyzer_output",
    "preprocessed.zarr",
    "binary",
    ".ipynb_checkpoints",
}


def detect_legacy_artifacts(output_dir: str | Path) -> dict[str, str]:
    output_dir = Path(output_dir)
    artifacts: dict[str, str] = {}

    for key, filename in LEGACY_FILE_NAMES.items():
        candidate = output_dir / filename
        if candidate.exists():
            artifacts[key] = str(candidate)

    for key, dirname in LEGACY_DIR_NAMES.items():
        candidate = output_dir / dirname
        if candidate.exists():
            artifacts[key] = str(candidate)

    checkpoint_dir = output_dir / LEGACY_DIR_NAMES["checkpoints"]
    if checkpoint_dir.exists():
        checkpoint_files = sorted(checkpoint_dir.glob("*.json"))
        if checkpoint_files:
            artifacts["checkpoint_files"] = [str(path) for path in checkpoint_files]

    return artifacts


def looks_like_legacy_well_dir(path: str | Path) -> bool:
    path = Path(path)
    if not path.is_dir():
        return False
    artifacts = detect_legacy_artifacts(path)
    return bool(artifacts)


def iter_legacy_well_dirs(root: str | Path) -> list[Path]:
    root = Path(root)
    if root.is_file():
        parent = root.parent
        return [parent] if looks_like_legacy_well_dir(parent) else []

    if looks_like_legacy_well_dir(root):
        return [root]

    matches: list[Path] = []
    for dirpath, dirnames, _ in __import__("os").walk(root):
        dirnames[:] = [name for name in dirnames if name not in SKIP_WALK_DIRS and not name.startswith(".")]
        candidate = Path(dirpath)
        if looks_like_legacy_well_dir(candidate):
            matches.append(candidate)
    return sorted(set(matches))

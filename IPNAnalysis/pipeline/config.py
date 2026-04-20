from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import yaml

from .compat.legacy_config import looks_like_legacy_config, normalize_legacy_config, strip_comment_keys
from .model import PipelineConfig


DEFAULT_PIPELINE_DATA: dict[str, Any] = {
    "inputs": {
        "file_types": ["h5", "nwb", "raw"],
        "maxwell_h5_filename_pattern": "data.raw.h5",
        "maxwell_network_folder_name": "Network",
    },
    "outputs": {
        "root": None,
        "checkpoint_root": None,
        "output_subdir_after_well": None,
        "preserve_legacy_filenames": True,
        "export_to_phy": False,
        "clean_up": False,
    },
    "execution": {
        "one_well_per_subprocess": False,
        "verbosity": "info",
        "force_restart": False,
    },
    "stages": {
        "preprocess": {
            "enabled": True,
            "cache_format": "zarr",
        },
        "sort": {
            "enabled": True,
            "sorter": "kilosort4",
            "docker_image": None,
            "skip_spikesorting": False,
        },
        "merge": {
            "enabled": True,
            "unitmatch": {
                "enabled": False,
                "dry_run": True,
                "scored_dry_run": True,
                "output_subdir_name": "unitmatch_outputs",
                "throughput_subdir_name": "unitmatch_throughput",
                "max_candidate_pairs": 20000,
                "oversplit_min_probability": 0.80,
                "oversplit_max_suggestions": 2000,
                "apply_merges": False,
                "recursive": False,
                "max_iterations": 5,
                "max_spikes_per_unit": 100,
                "keep_all_iterations": False,
                "generate_reports": True,
                "report_subdir_name": "unitmatch_reports",
                "report_max_heatmap_units": 200,
            },
            "auto_merge": {
                "enabled": False,
                "template_diff_thresh": "0.05,0.15,0.25",
            },
        },
        "analyze": {
            "enabled": True,
            "rerun": False,
        },
        "report": {
            "enabled": True,
            "plot_mode": "separate",
            "raster_sort": "none",
            "plot_debug": False,
            "fixed_y": False,
            "curation": {
                "enabled": True,
                "quality_thresholds": {
                    "presence_ratio": 0.75,
                    "rp_contamination": 0.15,
                    "firing_rate": 0.05,
                    "amplitude_median": -20,
                    "amplitude_cv_median": 0.5,
                },
            },
        },
    },
    "plugins": {
        "burst_analysis": {
            "enabled": True,
            "write_plots": True,
        }
    },
    "filters": {
        "reference_file": None,
        "assay_types": ["network today", "network today/best"],
    },
    "compat": {
        "read_legacy_json": True,
        "import_legacy_checkpoints": True,
        "import_legacy_outputs": True,
        "preserve_legacy_filenames": True,
        "reanalyze_bursts_only": False,
    },
}


TEMPLATE_YAML = """# IPNAnalysis v2 pipeline configuration
inputs:
  file_types: [h5, nwb, raw]
  maxwell_h5_filename_pattern: data.raw.h5
  maxwell_network_folder_name: Network

outputs:
  root: null
  checkpoint_root: null
  output_subdir_after_well: null
  preserve_legacy_filenames: true
  export_to_phy: false
  clean_up: false

execution:
  one_well_per_subprocess: false
  verbosity: info
  force_restart: false

stages:
  preprocess:
    enabled: true
    cache_format: zarr
  sort:
    enabled: true
    sorter: kilosort4
    docker_image: null
    skip_spikesorting: false
  merge:
    enabled: true
    unitmatch:
      enabled: false
      dry_run: true
      scored_dry_run: true
      output_subdir_name: unitmatch_outputs
      throughput_subdir_name: unitmatch_throughput
      max_candidate_pairs: 20000
      oversplit_min_probability: 0.8
      oversplit_max_suggestions: 2000
      apply_merges: false
      recursive: false
      max_iterations: 5
      max_spikes_per_unit: 100
      keep_all_iterations: false
      generate_reports: true
      report_subdir_name: unitmatch_reports
      report_max_heatmap_units: 200
    auto_merge:
      enabled: false
      template_diff_thresh: "0.05,0.15,0.25"
  analyze:
    enabled: true
    rerun: false
  report:
    enabled: true
    plot_mode: separate
    raster_sort: none
    plot_debug: false
    fixed_y: false
    curation:
      enabled: true
      quality_thresholds:
        presence_ratio: 0.75
        rp_contamination: 0.15
        firing_rate: 0.05
        amplitude_median: -20
        amplitude_cv_median: 0.5

plugins:
  burst_analysis:
    enabled: true
    write_plots: true

filters:
  reference_file: null
  assay_types: [network today, network today/best]

compat:
  read_legacy_json: true
  import_legacy_checkpoints: true
  import_legacy_outputs: true
  preserve_legacy_filenames: true
  reanalyze_bursts_only: false
"""


def deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def set_nested_value(mapping: dict[str, Any], dotted_path: str, value: Any) -> None:
    parts = dotted_path.split(".")
    cursor = mapping
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value


def normalize_pipeline_data(data: dict[str, Any]) -> dict[str, Any]:
    raw = strip_comment_keys(deepcopy(data or {}))
    if looks_like_legacy_config(raw):
        raw = normalize_legacy_config(raw)
    return deep_merge(DEFAULT_PIPELINE_DATA, raw)


def build_pipeline_config(
    data: dict[str, Any] | None = None,
    *,
    source_path: str | Path | None = None,
    format_name: str = "yaml",
) -> PipelineConfig:
    normalized = normalize_pipeline_data(data or {})
    source = Path(source_path).resolve() if source_path else None
    return PipelineConfig(
        source_path=source,
        format_name=format_name,
        inputs=deepcopy(normalized["inputs"]),
        outputs=deepcopy(normalized["outputs"]),
        execution=deepcopy(normalized["execution"]),
        stages=deepcopy(normalized["stages"]),
        plugins=deepcopy(normalized["plugins"]),
        filters=deepcopy(normalized["filters"]),
        compat=deepcopy(normalized["compat"]),
        raw=deepcopy(normalized),
    )


def pipeline_config_to_dict(config: PipelineConfig) -> dict[str, Any]:
    return {
        "inputs": deepcopy(config.inputs),
        "outputs": deepcopy(config.outputs),
        "execution": deepcopy(config.execution),
        "stages": deepcopy(config.stages),
        "plugins": deepcopy(config.plugins),
        "filters": deepcopy(config.filters),
        "compat": deepcopy(config.compat),
    }


def apply_overrides(config: PipelineConfig, overrides: dict[str, Any] | None = None) -> PipelineConfig:
    overrides = {key: value for key, value in (overrides or {}).items() if value is not None}
    if not overrides:
        return build_pipeline_config(
            pipeline_config_to_dict(config),
            source_path=config.source_path,
            format_name=config.format_name,
        )

    data = pipeline_config_to_dict(config)
    for dotted_path, value in overrides.items():
        set_nested_value(data, dotted_path, value)
    return build_pipeline_config(data, source_path=config.source_path, format_name=config.format_name)


def load_pipeline_config(config_path: str | Path | None = None) -> PipelineConfig:
    if config_path is None:
        return build_pipeline_config()

    config_path = Path(config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    suffix = config_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        with open(config_path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        return build_pipeline_config(raw, source_path=config_path, format_name="yaml")

    if suffix == ".json":
        with open(config_path, "r", encoding="utf-8") as handle:
            raw = json.load(handle) or {}
        format_name = "legacy_json" if looks_like_legacy_config(raw) else "json"
        return build_pipeline_config(raw, source_path=config_path, format_name=format_name)

    raise ValueError(f"Unsupported config format: {config_path.suffix}")


def write_config_template(destination: str | Path) -> Path:
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(TEMPLATE_YAML, encoding="utf-8")
    return destination

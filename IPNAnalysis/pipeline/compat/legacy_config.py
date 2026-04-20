from __future__ import annotations

from copy import deepcopy
from typing import Any


LEGACY_TOP_LEVEL_KEYS = {"io", "sorting", "merging", "filtering", "plotting", "curation"}


def strip_comment_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: strip_comment_keys(val)
            for key, val in value.items()
            if not str(key).startswith("_comment")
        }
    if isinstance(value, list):
        return [strip_comment_keys(item) for item in value]
    return value


def looks_like_legacy_config(data: dict[str, Any]) -> bool:
    if not isinstance(data, dict):
        return False
    return bool(LEGACY_TOP_LEVEL_KEYS.intersection(data.keys()))


def normalize_legacy_config(data: dict[str, Any]) -> dict[str, Any]:
    data = strip_comment_keys(deepcopy(data))

    io_cfg = data.get("io", {})
    sorting_cfg = data.get("sorting", {})
    merging_cfg = data.get("merging", {})
    filtering_cfg = data.get("filtering", {})
    plotting_cfg = data.get("plotting", {})
    curation_cfg = data.get("curation", {})

    quality_thresholds = dict(curation_cfg.get("quality_thresholds", {}))

    return {
        "inputs": {
            "file_types": ["h5", "nwb", "raw"],
            "maxwell_h5_filename_pattern": "data.raw.h5",
            "maxwell_network_folder_name": "Network",
        },
        "outputs": {
            "root": io_cfg.get("output_dir"),
            "checkpoint_root": io_cfg.get("checkpoint_dir"),
            "output_subdir_after_well": io_cfg.get("output_subdir_after_well"),
            "preserve_legacy_filenames": True,
            "export_to_phy": bool(io_cfg.get("export_to_phy", False)),
            "clean_up": bool(io_cfg.get("clean_up", False)),
        },
        "execution": {
            "one_well_per_subprocess": False,
            "verbosity": "info",
            "force_restart": False,
        },
        "stages": {
            "preprocess": {
                "enabled": True,
                "cache_format": io_cfg.get("preprocessed_storage_format", "zarr"),
            },
            "sort": {
                "enabled": True,
                "sorter": sorting_cfg.get("sorter", "kilosort4"),
                "docker_image": sorting_cfg.get("docker_image"),
                "skip_spikesorting": False,
            },
            "merge": {
                "enabled": True,
                "unitmatch": {
                    "enabled": False,
                    "dry_run": True,
                    "scored_dry_run": bool(merging_cfg.get("unitmatch_scored_dry_run", True)),
                    "output_subdir_name": merging_cfg.get("unitmatch_output_subdir_name", "unitmatch_outputs"),
                    "throughput_subdir_name": merging_cfg.get(
                        "unitmatch_throughput_subdir_name", "unitmatch_throughput"
                    ),
                    "max_candidate_pairs": int(merging_cfg.get("unitmatch_max_candidate_pairs", 20000)),
                    "oversplit_min_probability": float(
                        merging_cfg.get("unitmatch_oversplit_min_probability", 0.80)
                    ),
                    "oversplit_max_suggestions": int(
                        merging_cfg.get("unitmatch_oversplit_max_suggestions", 2000)
                    ),
                    "apply_merges": bool(merging_cfg.get("unitmatch_apply_merges", False)),
                    "recursive": bool(merging_cfg.get("unitmatch_recursive", False)),
                    "max_iterations": int(merging_cfg.get("unitmatch_max_iterations", 5)),
                    "max_spikes_per_unit": int(merging_cfg.get("unitmatch_max_spikes_per_unit", 100)),
                    "keep_all_iterations": bool(merging_cfg.get("unitmatch_keep_all_iterations", False)),
                    "generate_reports": bool(merging_cfg.get("unitmatch_generate_reports", True)),
                    "report_subdir_name": merging_cfg.get("unitmatch_report_subdir_name", "unitmatch_reports"),
                    "report_max_heatmap_units": int(
                        merging_cfg.get("unitmatch_report_max_heatmap_units", 200)
                    ),
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
                "plot_mode": plotting_cfg.get("plot_mode", "separate"),
                "raster_sort": plotting_cfg.get("raster_sort", "none"),
                "plot_debug": bool(plotting_cfg.get("plot_debug", False)),
                "fixed_y": bool(plotting_cfg.get("fixed_y", False)),
                "curation": {
                    "enabled": not bool(curation_cfg.get("no_curation", False)),
                    "quality_thresholds": quality_thresholds,
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
            "reference_file": filtering_cfg.get("reference_file"),
            "assay_types": list(filtering_cfg.get("assay_types", ["network today", "network today/best"])),
        },
        "compat": {
            "read_legacy_json": True,
            "import_legacy_checkpoints": True,
            "import_legacy_outputs": True,
            "preserve_legacy_filenames": True,
            "reanalyze_bursts_only": False,
        },
    }

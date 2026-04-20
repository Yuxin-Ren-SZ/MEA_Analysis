from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .config import load_pipeline_config
from .model import RunRequest


def build_legacy_driver_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Legacy MEA batch driver compatibility wrapper",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("path", type=str)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--output-subdir-after-well", type=str, default=None)
    parser.add_argument("--preprocessed-storage-format", type=str, choices=["zarr", "binary"], default=None)
    parser.add_argument("--export-to-phy", action="store_true")
    parser.add_argument("--clean-up", action="store_true")
    parser.add_argument("--reference", type=str, default=None)
    parser.add_argument("--type", nargs="+", default=None)
    parser.add_argument("--sorter", type=str, default=None)
    parser.add_argument("--docker", type=str, default=None)
    parser.add_argument("--skip-spikesorting", action="store_true")
    parser.add_argument("--plot-mode", choices=["separate", "merged"], default=None)
    parser.add_argument("--raster-sort", choices=["none", "firing_rate", "location_y", "unit_id"], default=None)
    parser.add_argument("--plot-debug", action="store_true")
    parser.add_argument("--fixed-y", action="store_true")
    parser.add_argument("--no-curation", action="store_true")
    parser.add_argument("--params", type=str, default=None)
    parser.add_argument("--force-restart", action="store_true")
    parser.add_argument("--resume-from", "--resume_from", dest="resume_from", type=str, default=None)
    parser.add_argument("--reanalyze-bursts", action="store_true")
    parser.add_argument("--dry", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--unitmatch-merge-units", action="store_true")
    parser.add_argument("--unitmatch-dry-run", action="store_true")
    parser.add_argument("--unitmatch-scored-dry-run", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--unitmatch-output-subdir-name", type=str, default=None)
    parser.add_argument("--unitmatch-throughput-subdir-name", type=str, default=None)
    parser.add_argument("--unitmatch-max-candidate-pairs", type=int, default=None)
    parser.add_argument("--unitmatch-oversplit-min-probability", type=float, default=None)
    parser.add_argument("--unitmatch-oversplit-max-suggestions", type=int, default=None)
    parser.add_argument("--unitmatch-apply-merges", action="store_true")
    parser.add_argument("--unitmatch-recursive", action="store_true")
    parser.add_argument("--unitmatch-max-iterations", type=int, default=None)
    parser.add_argument("--unitmatch-max-spikes-per-unit", type=int, default=None)
    parser.add_argument("--unitmatch-keep-all-iterations", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--unitmatch-generate-reports", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--unitmatch-report-subdir-name", type=str, default=None)
    parser.add_argument("--unitmatch-report-max-heatmap-units", type=int, default=None)
    return parser


def build_legacy_worker_parser() -> argparse.ArgumentParser:
    parser = build_legacy_driver_parser()
    parser.description = "Legacy single-well worker compatibility wrapper"
    parser.add_argument("--well", required=True)
    parser.add_argument("--rec", type=str, default=None)
    parser.add_argument("--auto-merge-units", action="store_true")
    parser.add_argument("--auto-merge-template-diff-thresh", default=None)
    parser.add_argument("--rerun-analyzer", action="store_true")
    return parser


def _load_quality_thresholds(params_value: str | None) -> dict[str, object] | None:
    if not params_value:
        return None
    if os.path.exists(params_value):
        with open(params_value, "r", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(params_value)


def _legacy_overrides(args: argparse.Namespace) -> dict[str, object]:
    overrides: dict[str, object] = {}
    if args.output_subdir_after_well is not None:
        overrides["outputs.output_subdir_after_well"] = args.output_subdir_after_well
    if args.preprocessed_storage_format is not None:
        overrides["stages.preprocess.cache_format"] = args.preprocessed_storage_format
    if args.reference is not None:
        overrides["filters.reference_file"] = args.reference
    if args.type is not None:
        overrides["filters.assay_types"] = list(args.type)
    if args.sorter is not None:
        overrides["stages.sort.sorter"] = args.sorter
    if args.docker is not None:
        overrides["stages.sort.docker_image"] = args.docker
    if args.skip_spikesorting:
        overrides["stages.sort.skip_spikesorting"] = True
    if args.plot_mode is not None:
        overrides["stages.report.plot_mode"] = args.plot_mode
    if args.raster_sort is not None:
        overrides["stages.report.raster_sort"] = args.raster_sort
    if args.plot_debug:
        overrides["stages.report.plot_debug"] = True
    if args.fixed_y:
        overrides["stages.report.fixed_y"] = True
    if args.no_curation:
        overrides["stages.report.curation.enabled"] = False
    thresholds = _load_quality_thresholds(getattr(args, "params", None))
    if thresholds:
        overrides["stages.report.curation.quality_thresholds"] = thresholds
    if args.export_to_phy:
        overrides["outputs.export_to_phy"] = True
    if args.clean_up:
        overrides["outputs.clean_up"] = True
    if args.unitmatch_merge_units:
        overrides["stages.merge.unitmatch.enabled"] = True
    if args.unitmatch_dry_run:
        overrides["stages.merge.unitmatch.dry_run"] = True
    if args.unitmatch_scored_dry_run is not None:
        overrides["stages.merge.unitmatch.scored_dry_run"] = bool(args.unitmatch_scored_dry_run)
    if args.unitmatch_output_subdir_name is not None:
        overrides["stages.merge.unitmatch.output_subdir_name"] = args.unitmatch_output_subdir_name
    if args.unitmatch_throughput_subdir_name is not None:
        overrides["stages.merge.unitmatch.throughput_subdir_name"] = args.unitmatch_throughput_subdir_name
    if args.unitmatch_max_candidate_pairs is not None:
        overrides["stages.merge.unitmatch.max_candidate_pairs"] = int(args.unitmatch_max_candidate_pairs)
    if args.unitmatch_oversplit_min_probability is not None:
        overrides["stages.merge.unitmatch.oversplit_min_probability"] = float(
            args.unitmatch_oversplit_min_probability
        )
    if args.unitmatch_oversplit_max_suggestions is not None:
        overrides["stages.merge.unitmatch.oversplit_max_suggestions"] = int(
            args.unitmatch_oversplit_max_suggestions
        )
    if args.unitmatch_apply_merges:
        overrides["stages.merge.unitmatch.apply_merges"] = True
    if args.unitmatch_recursive:
        overrides["stages.merge.unitmatch.recursive"] = True
    if args.unitmatch_max_iterations is not None:
        overrides["stages.merge.unitmatch.max_iterations"] = int(args.unitmatch_max_iterations)
    if args.unitmatch_max_spikes_per_unit is not None:
        overrides["stages.merge.unitmatch.max_spikes_per_unit"] = int(args.unitmatch_max_spikes_per_unit)
    if args.unitmatch_keep_all_iterations is not None:
        overrides["stages.merge.unitmatch.keep_all_iterations"] = bool(args.unitmatch_keep_all_iterations)
    if args.unitmatch_generate_reports is not None:
        overrides["stages.merge.unitmatch.generate_reports"] = bool(args.unitmatch_generate_reports)
    if args.unitmatch_report_subdir_name is not None:
        overrides["stages.merge.unitmatch.report_subdir_name"] = args.unitmatch_report_subdir_name
    if args.unitmatch_report_max_heatmap_units is not None:
        overrides["stages.merge.unitmatch.report_max_heatmap_units"] = int(
            args.unitmatch_report_max_heatmap_units
        )
    if getattr(args, "reanalyze_bursts", False):
        overrides["compat.reanalyze_bursts_only"] = True
    if getattr(args, "auto_merge_units", False):
        overrides["stages.merge.auto_merge.enabled"] = True
    if getattr(args, "auto_merge_template_diff_thresh", None) is not None:
        overrides["stages.merge.auto_merge.template_diff_thresh"] = args.auto_merge_template_diff_thresh
    if getattr(args, "rerun_analyzer", False):
        overrides["stages.analyze.rerun"] = True
    return overrides


def main_driver(argv: list[str] | None = None) -> int:
    parser = build_legacy_driver_parser()
    args = parser.parse_args(argv)
    from .runner import PipelineRunner

    config = load_pipeline_config(args.config)
    request = RunRequest(
        source_path=Path(args.path).resolve(),
        config_path=Path(args.config).resolve() if args.config else None,
        output_root=Path(args.output_dir).resolve() if args.output_dir else None,
        checkpoint_root=Path(args.checkpoint_dir).resolve() if args.checkpoint_dir else None,
        from_stage=args.resume_from,
        dry_run=bool(args.dry),
        force_restart=bool(args.force_restart),
        verbose=bool(args.debug),
        overrides=_legacy_overrides(args),
    )
    runner = PipelineRunner(config)
    results = runner.run(request)
    print(f"[compat] processed {len(results)} task(s)")
    return 0


def main_worker(argv: list[str] | None = None) -> int:
    parser = build_legacy_worker_parser()
    args = parser.parse_args(argv)
    from .runner import PipelineRunner

    config = load_pipeline_config(args.config)
    request = RunRequest(
        source_path=Path(args.path).resolve(),
        config_path=Path(args.config).resolve() if args.config else None,
        output_root=Path(args.output_dir).resolve() if args.output_dir else None,
        checkpoint_root=Path(args.checkpoint_dir).resolve() if args.checkpoint_dir else None,
        recording_name=args.rec,
        well_id=args.well,
        from_stage=args.resume_from,
        dry_run=False,
        force_restart=bool(args.force_restart),
        verbose=bool(args.debug),
        overrides=_legacy_overrides(args),
    )
    runner = PipelineRunner(config)
    runner.run(request)
    return 0

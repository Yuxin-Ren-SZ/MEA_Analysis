from __future__ import annotations

from copy import deepcopy
import logging
from pathlib import Path
from typing import Any

from .artifacts import ArtifactRegistry, detect_legacy_artifacts, iter_legacy_well_dirs
from .config import apply_overrides
from .context import LegacyAnalysisContext, PipelineStateStore, RunContext
from .discovery import discover_tasks
from .model import PipelineConfig, ReanalysisRequest, RunRequest, RunResult, STAGE_ORDER, WellTask, normalize_stage_name
from .stages import AnalyzeStage, MergeStage, PreprocessStage, ReportStage, SortStage


STAGE_REGISTRY = {
    "preprocess": PreprocessStage(),
    "sort": SortStage(),
    "merge": MergeStage(),
    "analyze": AnalyzeStage(),
    "report": ReportStage(),
}


def _request_overrides(request: RunRequest) -> dict[str, Any]:
    overrides = dict(request.overrides)
    if request.output_root is not None:
        overrides["outputs.root"] = str(request.output_root)
    if request.checkpoint_root is not None:
        overrides["outputs.checkpoint_root"] = str(request.checkpoint_root)
    if request.force_restart:
        overrides["execution.force_restart"] = True
    if request.verbose:
        overrides["execution.verbosity"] = "debug"
    return overrides


def _build_pipeline(task: WellTask, effective_config: PipelineConfig):
    from ..mea_analysis_routine import MEAPipeline, _apply_resume_from_stage

    sort_settings = deepcopy(effective_config.stages.get("sort", {}))
    merge_settings = deepcopy(effective_config.stages.get("merge", {}))
    analyze_settings = deepcopy(effective_config.stages.get("analyze", {}))
    outputs = effective_config.outputs
    execution = effective_config.execution
    preprocess_settings = deepcopy(effective_config.stages.get("preprocess", {}))

    if outputs.get("root") is None:
        raise ValueError(
            "No output root configured. Set outputs.root in the YAML/JSON config or pass --output-root."
        )

    unitmatch = deepcopy(merge_settings.get("unitmatch", {}))
    auto_merge = deepcopy(merge_settings.get("auto_merge", {}))

    pipeline = MEAPipeline(
        file_path=task.source_path,
        stream_id=task.well_id,
        recording_num=task.recording_name,
        output_root=outputs.get("root"),
        checkpoint_root=outputs.get("checkpoint_root"),
        sorter=sort_settings.get("sorter", "kilosort4"),
        docker_image=sort_settings.get("docker_image"),
        verbose=str(execution.get("verbosity", "info")).lower() == "debug",
        cleanup=bool(outputs.get("clean_up", False)),
        force_restart=bool(execution.get("force_restart", False)),
        um_kwargs={
            "merge_units": bool(unitmatch.get("enabled", False)),
            "dry_run": bool(unitmatch.get("dry_run", True)),
            "scored_dry_run": bool(unitmatch.get("scored_dry_run", True)),
            "output_subdir_name": unitmatch.get("output_subdir_name", "unitmatch_outputs"),
            "throughput_subdir_name": unitmatch.get("throughput_subdir_name", "unitmatch_throughput"),
            "max_candidate_pairs": int(unitmatch.get("max_candidate_pairs", 20000)),
            "oversplit_min_probability": float(unitmatch.get("oversplit_min_probability", 0.80)),
            "oversplit_max_suggestions": int(unitmatch.get("oversplit_max_suggestions", 2000)),
            "apply_merges": bool(unitmatch.get("apply_merges", False)),
            "recursive": bool(unitmatch.get("recursive", False)),
            "max_iterations": int(unitmatch.get("max_iterations", 5)),
            "max_spikes_per_unit": int(unitmatch.get("max_spikes_per_unit", 100)),
            "keep_all_iterations": bool(unitmatch.get("keep_all_iterations", False)),
            "generate_reports": bool(unitmatch.get("generate_reports", True)),
            "report_subdir_name": unitmatch.get("report_subdir_name", "unitmatch_reports"),
            "report_max_heatmap_units": int(unitmatch.get("report_max_heatmap_units", 200)),
        },
        am_kwargs={
            "enabled": bool(auto_merge.get("enabled", False)),
            "template_diff_thresh": str(auto_merge.get("template_diff_thresh", "0.05,0.15,0.25")),
        },
        option_kwargs={
            "force_rerun_analyzer": bool(analyze_settings.get("rerun", False)),
            "output_subdir_after_well": outputs.get("output_subdir_after_well"),
            "preprocessed_storage_format": preprocess_settings.get("cache_format", "zarr"),
        },
    )
    return pipeline, _apply_resume_from_stage


def _select_stages(effective_config: PipelineConfig, request: RunRequest) -> list[str]:
    selected = [stage for stage in STAGE_ORDER if effective_config.stage_enabled(stage)]
    if bool(effective_config.stages.get("sort", {}).get("skip_spikesorting", False)):
        selected = [stage for stage in selected if stage in {"preprocess", "sort"}]

    start_stage = normalize_stage_name(request.from_stage)
    end_stage = normalize_stage_name(request.to_stage)

    if start_stage is not None:
        start_index = selected.index(start_stage) if start_stage in selected else 0
        selected = selected[start_index:]
    if end_stage is not None and end_stage in selected:
        end_index = selected.index(end_stage)
        selected = selected[: end_index + 1]
    return selected


class PipelineRunner:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def discover(self, request: RunRequest):
        effective_config = apply_overrides(self.config, _request_overrides(request))
        return discover_tasks(
            request.source_path,
            effective_config,
            recording_name=request.recording_name,
            well_id=request.well_id,
        )

    def run(self, request: RunRequest) -> list[RunResult]:
        effective_config = apply_overrides(self.config, _request_overrides(request))
        discovery = discover_tasks(
            request.source_path,
            effective_config,
            recording_name=request.recording_name,
            well_id=request.well_id,
        )
        if request.dry_run:
            return [
                RunResult(task=task, output_dir=Path("."), selected_stages=_select_stages(effective_config, request))
                for task in discovery.tasks
            ]

        results: list[RunResult] = []
        for task in discovery.tasks:
            results.append(self._run_single_task(task, request, effective_config))
        return results

    def _run_single_task(
        self,
        task: WellTask,
        request: RunRequest,
        effective_config: PipelineConfig,
    ) -> RunResult:
        pipeline, apply_resume_from_stage = _build_pipeline(task, effective_config)
        artifacts = ArtifactRegistry(pipeline.output_dir)
        artifacts.ensure_layout()
        state_store = PipelineStateStore(artifacts)
        legacy_artifacts = detect_legacy_artifacts(pipeline.output_dir)
        state_store.write_manifest(
            request=request,
            task=task,
            config=effective_config,
            legacy_artifacts=legacy_artifacts,
        )
        state_store.record_compat_import(
            {
                "task": task.to_dict(),
                "legacy_artifacts": legacy_artifacts,
            }
        )
        context = RunContext(
            request=request,
            task=task,
            config=self.config,
            effective_config=effective_config,
            artifacts=artifacts,
            state_store=state_store,
            pipeline=pipeline,
        )

        if request.from_stage:
            apply_resume_from_stage(pipeline, request.from_stage)

        if bool(effective_config.compat.get("reanalyze_bursts_only", False)):
            from .plugins import BurstAnalysisPlugin

            plugin = BurstAnalysisPlugin()
            state_store.mark_stage("burst_analysis", "running", {"task_id": task.task_id, "mode": "compat_only"})
            details = plugin.run(context, effective_config.plugins.get("burst_analysis"))
            state_store.mark_stage("burst_analysis", "completed", details)
            return RunResult(
                task=task,
                output_dir=pipeline.output_dir,
                selected_stages=["burst_analysis"],
                skipped=False,
                details=details,
            )

        if pipeline.should_skip() and request.from_stage is None and request.to_stage is None:
            state_store.mark_stage("pipeline", "skipped", {"reason": "legacy_pipeline_already_complete"})
            return RunResult(task=task, output_dir=pipeline.output_dir, selected_stages=[], skipped=True)

        selected_stages = _select_stages(effective_config, request)
        for stage_name in selected_stages:
            stage = STAGE_REGISTRY[stage_name]
            state_store.mark_stage(stage_name, "running", {"task_id": task.task_id})
            details = stage.run(context)
            state_store.mark_stage(stage_name, "completed", details)

        if context.data.get("skip_sorting") and effective_config.plugin_enabled("burst_analysis"):
            from .plugins import BurstAnalysisPlugin

            plugin = BurstAnalysisPlugin()
            state_store.mark_stage("burst_analysis", "running", {"task_id": task.task_id})
            details = plugin.run(context, effective_config.plugins.get("burst_analysis"))
            state_store.mark_stage("burst_analysis", "completed", details)

        if bool(effective_config.outputs.get("clean_up", False)):
            pipeline.cleanup()

        return RunResult(
            task=task,
            output_dir=pipeline.output_dir,
            selected_stages=selected_stages,
            skipped=False,
            details={"legacy_artifacts": legacy_artifacts},
        )

    def reanalyze(self, request: ReanalysisRequest) -> list[dict[str, Any]]:
        effective_config = apply_overrides(self.config, request.overrides)
        plugin_names = request.plugins or ["burst_analysis"]
        logger = logging.getLogger("ipn_reanalyze")
        if not logger.handlers:
            logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.DEBUG if request.verbose else logging.INFO)

        from .plugins import BurstAnalysisPlugin

        plugin_registry = {"burst_analysis": BurstAnalysisPlugin()}
        output_dirs = iter_legacy_well_dirs(request.source_path)
        if not output_dirs:
            raise FileNotFoundError(f"No legacy well output folders found under {request.source_path}")

        results: list[dict[str, Any]] = []
        for output_dir in output_dirs:
            artifacts = ArtifactRegistry(output_dir)
            artifacts.ensure_layout()
            state_store = PipelineStateStore(artifacts)
            legacy_artifacts = detect_legacy_artifacts(output_dir)
            state_store.write_manifest(
                request=request,
                task=None,
                config=effective_config,
                legacy_artifacts=legacy_artifacts,
            )
            state_store.record_compat_import({"output_dir": str(output_dir), "legacy_artifacts": legacy_artifacts})
            context = LegacyAnalysisContext(
                request=request,
                output_dir=output_dir,
                config=self.config,
                effective_config=effective_config,
                artifacts=artifacts,
                state_store=state_store,
                logger=logger,
            )

            for plugin_name in plugin_names:
                plugin = plugin_registry[plugin_name]
                state_store.mark_stage(plugin_name, "running", {"output_dir": str(output_dir)})
                details = plugin.run(context, effective_config.plugins.get(plugin_name))
                state_store.mark_stage(plugin_name, "completed", details)
                results.append({"plugin": plugin_name, "output_dir": str(output_dir), "details": details})

        return results

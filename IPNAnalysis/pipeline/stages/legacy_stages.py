from __future__ import annotations

from typing import Any

from .base import StagePlugin


class PreprocessStage(StagePlugin):
    name = "preprocess"
    produces = ("preprocessed_recording",)

    def run(self, context: Any) -> dict[str, Any]:
        context.pipeline.run_preprocessing()
        return {
            "output_dir": str(context.pipeline.output_dir),
            "cache_format": context.effective_config.stages["preprocess"].get("cache_format", "zarr"),
        }


class SortStage(StagePlugin):
    name = "sort"
    requires = ("preprocess",)

    def run(self, context: Any) -> dict[str, Any]:
        sort_settings = context.effective_config.stages.get("sort", {})
        if bool(sort_settings.get("skip_spikesorting", False)):
            ids = context.pipeline._spike_detection_only()
            context.data["detected_ids"] = ids
            context.data["skip_sorting"] = True
            return {
                "mode": "spike_detection_only",
                "n_detected_units": len(ids),
            }

        context.pipeline.run_sorting()
        return {
            "mode": "sorter",
            "sorter": sort_settings.get("sorter", context.pipeline.sorter),
            "sorter_output": str(context.pipeline.output_dir / "sorter_output"),
        }


class MergeStage(StagePlugin):
    name = "merge"
    requires = ("sort",)

    def run(self, context: Any) -> dict[str, Any]:
        context.pipeline.run_optional_merge_phase()
        return dict(context.pipeline.state.get("merge_phase") or {})


class AnalyzeStage(StagePlugin):
    name = "analyze"
    requires = ("sort",)

    def run(self, context: Any) -> dict[str, Any]:
        context.pipeline.run_analyzer()
        return {
            "analyzer_output": str(context.pipeline.output_dir / "analyzer_output"),
        }


class ReportStage(StagePlugin):
    name = "report"
    requires = ("analyze",)

    def run(self, context: Any) -> dict[str, Any]:
        report_settings = context.effective_config.stages.get("report", {})
        curation_settings = report_settings.get("curation", {})
        context.pipeline.generate_reports(
            thresholds=curation_settings.get("quality_thresholds"),
            no_curation=not bool(curation_settings.get("enabled", True)),
            export_phy=bool(context.effective_config.outputs.get("export_to_phy", False)),
            plot_mode=str(report_settings.get("plot_mode", "separate")),
            plot_debug=bool(report_settings.get("plot_debug", False)),
            raster_sort=str(report_settings.get("raster_sort", "none")),
            fixed_y=bool(report_settings.get("fixed_y", False)),
        )
        return {
            "report_root": str(context.pipeline.output_dir),
        }

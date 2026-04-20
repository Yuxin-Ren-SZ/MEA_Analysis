from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .base import AnalysisPlugin

try:
    from ... import helper_functions as helper
    from ...parameter_free_burst_detector import compute_network_bursts
except Exception:  # pragma: no cover - local script fallback
    import helper_functions as helper  # type: ignore
    from parameter_free_burst_detector import compute_network_bursts  # type: ignore


class BurstAnalysisPlugin(AnalysisPlugin):
    name = "burst_analysis"

    def run(self, context: Any, settings: dict[str, Any] | None = None) -> dict[str, Any]:
        settings = settings or {}
        if getattr(context, "pipeline", None) is not None:
            report_settings = context.effective_config.stages.get("report", {})
            context.pipeline._run_burst_analysis(
                ids_list=context.data.get("detected_ids"),
                plot_mode=str(report_settings.get("plot_mode", "separate")),
                plot_debug=bool(report_settings.get("plot_debug", False)),
                raster_sort=str(report_settings.get("raster_sort", "none")),
                fixed_y=bool(report_settings.get("fixed_y", False)),
            )
            return {
                "mode": "delegated_to_mea_pipeline",
                "output_dir": str(context.artifacts.output_dir),
            }

        output_dir = Path(context.output_dir).resolve()
        spike_times_path = output_dir / "spike_times.npy"
        if not spike_times_path.exists():
            raise FileNotFoundError(f"Spike times not found for burst reanalysis: {spike_times_path}")

        spike_times = np.load(spike_times_path, allow_pickle=True).item()
        if not spike_times:
            raise ValueError(f"No spike times available in {spike_times_path}")

        network_data = compute_network_bursts(SpikeTimes=spike_times, plot=False)
        network_data_clean = helper.recursive_clean(network_data)
        network_data_clean["n_units"] = len(spike_times)

        network_file = output_dir / "network_results.json"
        with open(network_file, "w", encoding="utf-8") as handle:
            json.dump(network_data_clean, handle, indent=2)

        if bool(settings.get("write_plots", True)):
            self._write_plots(
                output_dir=output_dir,
                spike_times=spike_times,
                network_data=network_data,
                plot_mode=str(
                    context.effective_config.stages.get("report", {}).get("plot_mode", "separate")
                ),
                plot_debug=bool(
                    context.effective_config.stages.get("report", {}).get("plot_debug", False)
                ),
                raster_sort=str(
                    context.effective_config.stages.get("report", {}).get("raster_sort", "none")
                ),
                logger=context.logger,
            )

        return {
            "mode": "legacy_output_folder",
            "output_dir": str(output_dir),
            "network_results": str(network_file),
            "n_units": len(spike_times),
        }

    def _write_plots(
        self,
        *,
        output_dir: Path,
        spike_times: dict[Any, np.ndarray],
        network_data: dict[str, Any],
        plot_mode: str,
        plot_debug: bool,
        raster_sort: str,
        logger: Any,
    ) -> None:
        sorted_units = self._sort_units_for_raster(spike_times, raster_sort)
        if plot_mode == "merged":
            fig, ax = plt.subplots(figsize=(12, 5))
            ax_raster = ax
            ax_network = ax.twinx()
            helper.plot_clean_raster(ax_raster, spike_times, sorted_units)
            helper.plot_clean_network(ax_network, **network_data["plot_data"])
        else:
            fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            ax_raster, ax_network = axs
            helper.plot_clean_raster(ax_raster, spike_times, sorted_units)
            helper.plot_clean_network(ax_network, **network_data["plot_data"])

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.05)
        if plot_debug:
            for event in network_data.get("network_bursts", {}).get("events", []):
                ax_network.axvspan(event["start"], event["end"], color="gray", alpha=0.1)
            for event in network_data.get("superbursts", {}).get("events", []):
                ax_network.axvspan(event["start"], event["end"], color="gray", alpha=0.2)

        plt.savefig(output_dir / "raster_burst_plot.svg")
        ax_raster.set_xlim(0, 60)
        ax_network.set_xlim(0, 60)
        plt.savefig(output_dir / "raster_burst_plot_60s.svg")
        ax_raster.set_xlim(0, 30)
        ax_network.set_xlim(0, 30)
        ax_network.set_xlabel("Time (s)")
        plt.savefig(output_dir / "raster_burst_plot_30s.svg")
        plt.savefig(output_dir / "raster_burst_plot.png", dpi=300)
        plt.close(fig)
        logger.info("Rewrote burst plots in %s", output_dir)

    @staticmethod
    def _sort_units_for_raster(spike_times: dict[Any, np.ndarray], raster_sort: str) -> list[Any] | None:
        if raster_sort == "firing_rate":
            return sorted(spike_times.keys(), key=lambda uid: len(spike_times[uid]))
        if raster_sort == "unit_id":
            return sorted(spike_times.keys())
        return None

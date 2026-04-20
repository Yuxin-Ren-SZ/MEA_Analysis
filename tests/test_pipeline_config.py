from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from IPNAnalysis.pipeline.config import build_pipeline_config, load_pipeline_config, write_config_template


class PipelineConfigTests(unittest.TestCase):
    def test_legacy_json_normalizes_into_new_shape(self) -> None:
        legacy = {
            "io": {
                "output_dir": "/tmp/output",
                "checkpoint_dir": "/tmp/output/checkpoints",
                "preprocessed_storage_format": "binary",
                "export_to_phy": True,
            },
            "sorting": {
                "sorter": "mountainsort5",
                "docker_image": "mea-spikesorter",
            },
            "plotting": {
                "plot_mode": "merged",
                "raster_sort": "firing_rate",
            },
            "curation": {
                "no_curation": False,
                "quality_thresholds": {"presence_ratio": 0.9},
            },
        }
        config = build_pipeline_config(legacy, format_name="legacy_json")
        self.assertEqual(config.outputs["root"], "/tmp/output")
        self.assertEqual(config.outputs["checkpoint_root"], "/tmp/output/checkpoints")
        self.assertEqual(config.stages["preprocess"]["cache_format"], "binary")
        self.assertEqual(config.stages["sort"]["sorter"], "mountainsort5")
        self.assertEqual(config.stages["sort"]["docker_image"], "mea-spikesorter")
        self.assertEqual(config.stages["report"]["plot_mode"], "merged")
        self.assertEqual(config.stages["report"]["curation"]["quality_thresholds"]["presence_ratio"], 0.9)

    def test_yaml_template_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            destination = Path(tmpdir) / "mea_pipeline.yaml"
            write_config_template(destination)
            self.assertTrue(destination.exists())
            config = load_pipeline_config(destination)
            self.assertTrue(config.stage_enabled("preprocess"))
            self.assertTrue(config.plugin_enabled("burst_analysis"))
            raw = yaml.safe_load(destination.read_text(encoding="utf-8"))
            self.assertIn("stages", raw)


if __name__ == "__main__":
    unittest.main()

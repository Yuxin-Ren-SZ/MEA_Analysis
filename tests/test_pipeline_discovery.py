from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import importlib.util

from IPNAnalysis.pipeline.artifacts import detect_legacy_artifacts, iter_legacy_well_dirs
from IPNAnalysis.pipeline.config import build_pipeline_config
from IPNAnalysis.pipeline.discovery import discover_tasks


H5PY_AVAILABLE = importlib.util.find_spec("h5py") is not None


class PipelineDiscoveryTests(unittest.TestCase):
    @unittest.skipUnless(H5PY_AVAILABLE, "h5py is not available in this Python environment")
    def test_h5_discovery_finds_recordings_and_wells(self) -> None:
        import h5py

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            h5_path = root / "ProjectA" / "240101" / "M00001" / "000123" / "Network" / "data.raw.h5"
            h5_path.parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(h5_path, "w") as handle:
                recordings = handle.create_group("recordings")
                rec = recordings.create_group("rec0001")
                rec.create_group("well000")
                rec.create_group("well001")

            config = build_pipeline_config({"outputs": {"root": str(root / "AnalyzedData")}})
            discovery = discover_tasks(h5_path, config)
            self.assertEqual(len(discovery.tasks), 2)
            self.assertEqual(discovery.tasks[0].recording_name, "rec0001")
            self.assertEqual(discovery.tasks[0].run_id, "000123")

    def test_legacy_output_detection_finds_well_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            well_dir = root / "ProjectA" / "240101" / "M00001" / "000123" / "well000"
            well_dir.mkdir(parents=True, exist_ok=True)
            (well_dir / "network_results.json").write_text("{}", encoding="utf-8")
            (well_dir / "spike_times.npy").write_bytes(b"npy")
            (well_dir / "sorter_output").mkdir()

            artifacts = detect_legacy_artifacts(well_dir)
            self.assertIn("network_results", artifacts)
            self.assertIn("sorter_output", artifacts)

            discovered = iter_legacy_well_dirs(root)
            self.assertEqual(discovered, [well_dir])


if __name__ == "__main__":
    unittest.main()

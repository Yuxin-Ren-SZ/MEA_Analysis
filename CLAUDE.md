# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

End-to-end pipeline for neuronal spike sorting and network burst analysis on **Maxwell Biosystems MEA** recordings. Built on SpikeInterface with Kilosort4 as the default sorter. Python ≥ 3.9, GPU recommended (≥ 8 GB VRAM).

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Editable install
pip install -e .

# Generate a config template
python IPNAnalysis/config_loader.py mea_config.json

# Dry run (preview what would process)
python IPNAnalysis/run_pipeline_driver.py /data/experiment --config mea_config.json --dry

# Full batch run
python IPNAnalysis/run_pipeline_driver.py /data/experiment --config mea_config.json

# Single well processing
python IPNAnalysis/mea_analysis_routine.py /data/exp/run_001/Network/data.raw.h5 \
  --well well000 --rec rec0001 --config mea_config.json

# Docker build and run
docker build -t mea-spikesorter -f dockers/spikesorter/Dockerfile .
docker run --gpus all -it --rm \
  -v /path/to/experiment:/data/experiment \
  -v /path/to/mea_config.json:/config/mea_config.json:ro \
  mea-spikesorter /data/experiment --config /config/mea_config.json
```

There is no test suite or linter configured for this project.

## Architecture

### Two-Tier Design

1. **`IPNAnalysis/run_pipeline_driver.py`** — Orchestrator. Scans directories, discovers HDF5 files and wells, launches subprocesses for each recording-well combination. Handles reference filtering, dry-runs, and logging.

2. **`IPNAnalysis/mea_analysis_routine.py`** — Core worker (`MEAPipeline` class). Executes the full pipeline for a single well with checkpoint-based resumption. Stages:
   - **Preprocessing** (`run_preprocessing`) — highpass filter 300 Hz, common median reference, cache to Zarr or binary
   - **Spike Sorting** (`run_sorting`) — Kilosort4 via SpikeInterface, optional Docker isolation
   - **Analyzer** (`run_analyzer`) — template computation, quality metrics
   - **Reports** (`generate_reports`) — waveform plots, probe maps, burst analysis, automatic curation

3. **`IPNAnalysis/config_loader.py`** — Shared configuration with three-level priority: CLI flag → config file (`mea_config.json`) → hardcoded defaults. Run directly to generate a config template.

### Supporting Modules (all in `IPNAnalysis/`)

- **`helper_functions.py`** — peak detection, file discovery, raster/network plotting utilities
- **`parameter_free_burst_detector.py`** — adaptive network burst detection with per-unit ISI calibration
- **`meaplotter.py`** — comprehensive visualization (waveforms, rasters, probe maps)
- **`gaussianNetworkBursts.py`** — Gaussian-based burst modeling
- **`spikeMatrix.py`** — spike raster matrix operations
- **`mea_pipeline_gui.py`** — PyQt GUI wrapper

### Other Directories

- `NetworkAnalysis/` — MATLAB-based network analysis tools (legacy)
- `Archive/` — legacy scripts, not maintained
- `dockers/spikesorter/` — Dockerfile, entrypoint, minimal requirements for containerized sorting
- `kubernetics_configs/jobs/` — Kubernetes job templates for distributed GPU processing

## Key Conventions

- **Checkpoint resumption**: Re-running the same command automatically resumes from the last completed stage. Use `--force-restart` to ignore checkpoints.
- **HDF5 structure**: Pipeline expects `recordings/recNNNN/wellNNN` hierarchy inside `.h5` files. A `recording_map` dictionary maps recordings to wells.
- **Path-based metadata**: Project structure is inferred from file paths: `<project>/<date>/<chip>/<run_id>/Network/data.raw.h5`.
- **`--well` and `--rec`** are always CLI-only (never set in config). Run-control flags (`--debug`, `--dry`, `--force-restart`, `--reanalyze-bursts`, `--skip-spikesorting`) are also CLI-only.
- **Preprocessing cleanup**: When recomputed, the pipeline deletes the obsolete alternate cache format (zarr vs binary) to avoid stale data.

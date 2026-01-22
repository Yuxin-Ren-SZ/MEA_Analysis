# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MEA_Analysis is a Python pipeline for analyzing high-density Microelectrode Array (MEA) recordings from Maxwell Biosystems chips. The project focuses on neuronal spike detection, sorting, waveform analysis, and burst detection for CDKL5 disease models and organoid studies.

**Primary Entry Point**: `IPNAnalysis/` contains the main analysis pipeline.

## Common Commands

### Run Pipeline (Single Well)
```bash
cd IPNAnalysis
python mea_analysis_routine.py /path/to/data.raw.h5 \
  --well well000 \
  --output-dir ./AnalyzedData \
  --sorter kilosort4 \
  --clean-up
```

### Run Pipeline (Batch Processing)
```bash
cd IPNAnalysis
python run_pipeline_driver.py /path/to/data/directory \
  --reference experiment_metadata.xlsx \
  --type "network today" "network best" \
  --sorter kilosort4 \
  --output-dir ./AnalyzedData
```

### Run Pipeline via GUI
```bash
cd IPNAnalysis
python pipeline_gui.py
```
Tkinter GUI with file browsers, dropdowns, and checkboxes. Features live command preview and streaming log output.

### Monitor Pipeline Progress
```bash
streamlit run IPNAnalysis/streamlit_checkpoint_analyzer/checkpoint_dashboard.py
```

### Analysis Selection (--analysis flag)

The `--analysis` flag controls which outputs are generated. Available in both `mea_analysis_routine.py` and `run_pipeline_driver.py`.

**Valid analyses**: `probe`, `waveforms`, `raster`, `burst`, `spatial`

**Presets**:
- `default` → probe, waveforms, raster, burst
- `all` → all analyses including spatial
- `minimal` → burst only (no plots)
- `none` → no analyses

**Examples**:
```bash
--analysis default,spatial   # All default + spatial maps
--analysis burst             # Only burst metrics (fastest)
--analysis raster            # Raster plots (auto-enables burst)
--analysis all               # Everything including spatial maps
```

**Note**: If `raster` is specified, `burst` is auto-enabled (required for plot data).

## Architecture

### Pipeline Stages (Checkpointed)

The `MEAPipeline` class in `mea_analysis_routine.py` processes each well through 4 checkpointed stages:

| Stage | Description |
|-------|-------------|
| PREPROCESSING | Load Maxwell .h5 → Bandpass 300-3000Hz → Local CMR (250µm) → Binary |
| SORTING | Kilosort → Remove empty/duplicate units → Checkpoint |
| ANALYZER | SortingAnalyzer → Sparse waveforms (50µm) → Quality/template metrics |
| REPORTS | Export metrics → Curation → Burst detection → Visualizations |

Checkpoints allow resuming failed runs. Use `--force-restart` to ignore checkpoints.

### Data Flow
```
Raw .h5 → Preprocessing (300-3000Hz bandpass, local CMR) → Kilosort → Quality Metrics → Burst Analysis → Outputs
```

### Core Modules (IPNAnalysis/)

| Module | Purpose |
|--------|---------|
| `mea_analysis_routine.py` | Main pipeline class (`MEAPipeline`) - single well processing |
| `run_pipeline_driver.py` | Batch coordinator for multiple wells/files |
| `pipeline_gui.py` | Tkinter GUI for run_pipeline_driver.py |
| `helper_functions.py` | Utilities, ISI burst detection, file I/O |
| `parameter_free_burst_detector.py` | Adaptive hierarchical burst detection (newest) |
| `gaussianNetworkBursts.py` | Gaussian-smoothed burst detection (σ=100ms) |
| `neuron_spatial_maps.py` | Spatial visualization: spike density maps, amplitude heatmaps |

### Output Structure
```
AnalyzedData/{project}/{date}/{chip_id}/{run_id}/{well_id}/
├── quality_metrics.xlsx              # Unit quality (SNR, firing rate, etc.)
├── template_metrics.xlsx             # Waveform shape features
├── network_data.json                 # Network burst statistics
├── spikesorted_spike_times_dict.npy  # Spike times per unit
├── waveforms/                        # PDF waveform plots
├── spike_sorted_raster_plot.svg      # Network raster
├── locations_*.pdf                   # Probe location plots
└── neuron_spatial_*.pdf              # Spatial maps (opt-in)
```

## CLI Reference

### mea_analysis_routine.py

| Flag | Description |
|------|-------------|
| `--well` | Well ID (e.g., well000) [required] |
| `--output-dir` | Output directory [required] |
| `--sorter` | kilosort2, kilosort2_5, kilosort3, kilosort4 (default) |
| `--analysis` | Comma-separated analyses or preset (see above) |
| `--params` | JSON file with quality thresholds |
| `--checkpoint-dir` | Custom checkpoint directory |
| `--docker` | Docker image for containerized sorting |
| `--debug` | Enable debug logging |
| `--clean-up` | Delete intermediate files after processing |
| `--force-restart` | Restart from scratch, ignoring checkpoints |
| `--export-to-phy` | Export for manual curation in Phy GUI |
| `--no-curation` | Skip automatic quality filtering |
| `--skip-spikesorting` | Skip sorting stage (use existing results) |
| `--reanalyze-bursts` | Re-run burst analysis only |

### run_pipeline_driver.py (additional)

| Flag | Description |
|------|-------------|
| `--reference` | Excel file for filtering runs by metadata |
| `--type` | Assay types to include (default: "network today", "network today/best") |
| `--dry` | Dry run (no processing) |

## Data Analysis Quick Start

```python
import numpy as np
import pandas as pd
import json

# Spike times for all units
spike_times = np.load('AnalyzedData/.../spikesorted_spike_times_dict.npy', allow_pickle=True).item()

# Quality metrics
quality_metrics = pd.read_excel('AnalyzedData/.../quality_metrics.xlsx')

# Template metrics (waveform shape features)
template_metrics = pd.read_excel('AnalyzedData/.../template_metrics.xlsx')

# Network burst data
with open('AnalyzedData/.../network_data.json', 'r') as f:
    network_data = json.load(f)
```

### Burst Detection Methods

```python
from helper_functions import detect_bursts_statistics
from gaussianNetworkBursts import plot_network_activity
from parameter_free_burst_detector import compute_network_bursts

# ISI threshold method (simple, fast)
bursts = detect_bursts_statistics(spike_times_array, isi_threshold=0.1)

# Gaussian smoothing method (good for visualization)
bursts_gaussian = plot_network_activity(ax, spike_times_dict, gaussianSigma=0.1)

# Parameter-free method (newest, adaptive, hierarchical)
burst_results = compute_network_bursts(ax_raster, ax_macro, spike_times_dict)
```

## Tech Stack

- **Python**: 3.12
- **SpikeInterface**: 0.103.x
- **Kilosort**: 4.x (supports 2, 2.5, 3, 4)
- **PyTorch**: 2.x + CUDA 12.x
- **Scientific**: numpy, scipy, pandas, matplotlib, seaborn, h5py, zarr

## Quality Thresholds

Defined in `IPNAnalysis/sorting_quality_threshold_params.json`:

| Metric | Threshold | Meaning |
|--------|-----------|---------|
| `num_spikes` | > 300 | Minimum spike count |
| `presence_ratio` | > 0.9 | Unit active throughout recording |
| `rp_contamination` | < 1.0 | Refractory period violations |
| `firing_rate` | > 0.05 Hz | Minimum activity |
| `amplitude_median` | ≤ -20 µV | Minimum signal amplitude |

## Important Constants

```python
# Preprocessing
freq_min, freq_max = 300, 3000  # Hz bandpass
reference_radius_um = 250       # Local CMR radius

# Sparse waveforms
sparse_radius_um = 50
peak_sign = 'neg'

# Burst detection defaults
ISI_threshold = 0.1      # seconds
gaussianSigma = 0.1      # seconds
min_spikes_in_burst = 3
```

## Key Jupyter Notebooks (IPNAnalysis/workbooks/)

| Notebook | Purpose |
|----------|---------|
| `spikeTImesProcessing.ipynb` | Spike time post-processing, custom burst detection |
| `CIRM_figures.ipynb` | Publication figures (CDKL5, organoids) |
| `analysis_general.ipynb` | General workflows, templates |
| `compare_two_sorters.ipynb` | Kilosort version comparison |

## Repository Structure

```
MEA_Analysis/
├── IPNAnalysis/                    # Main analysis pipeline
├── AxonReconPipeline/              # Axon velocity tracking (git submodule)
├── MaxwellBiosystemsDeviceInterface/  # Maxwell MEA hardware control
├── StimulationAnalysis/            # Single-neuron stimulation experiments
├── NetworkAnalysis/                # MATLAB & legacy analysis tools
├── MEAProcessingLibrary/           # Reusable processing utilities
├── Organoid/                       # Organoid-specific analysis
└── dockers/                        # Kilosort spike sorter containers
```

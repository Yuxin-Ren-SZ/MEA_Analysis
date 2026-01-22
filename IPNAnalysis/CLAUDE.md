# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MEA neuronal recording analysis pipeline for Maxwell MEA chips. Two primary use cases:

1. **Pipeline operators**: Run spike sorting on raw .h5 files via `run_pipeline_driver.py` or `mea_analysis_routine.py`
2. **Data analysts**: Load pre-processed data in Jupyter notebooks for custom analysis and figures

**Research Focus**: CDKL5 disease models, network burst characterization, organoid studies

## Common Commands

### Run Pipeline (Single Well)
```bash
python mea_analysis_routine.py /path/to/data.raw.h5 \
  --well well000 \
  --output-dir ./AnalyzedData \
  --sorter kilosort4 \
  --clean-up
```

### Run Pipeline (Batch Processing)
```bash
python run_pipeline_driver.py /path/to/data/directory \
  --reference experiment_metadata.xlsx \
  --type "network today" "network best" \
  --sorter kilosort4 \
  --output-dir ./AnalyzedData
```

### Run Pipeline via GUI
```bash
python pipeline_gui.py
```
A Tkinter GUI that exposes all `run_pipeline_driver.py` options with file browsers, dropdowns, and checkboxes. Features live command preview and streaming log output.

### Monitor Pipeline Progress
```bash
streamlit run streamlit_checkpoint_analyzer/checkpoint_dashboard.py
```

### Analysis Selection (--analysis flag)

The `--analysis` flag controls which outputs are generated. Available in both scripts.

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

## Quick Start for Data Analysis

### Loading Pre-Analyzed Data

Pipeline outputs are in `AnalyzedData/{project}/{date}/{chip_id}/{run_id}/{well_id}/`:

```python
import numpy as np
import pandas as pd
import json

# Spike times for all units
spike_times = np.load('AnalyzedData/.../spikesorted_spike_times_dict.npy', allow_pickle=True).item()

# Quality metrics (firing rate, SNR, presence ratio, etc.)
quality_metrics = pd.read_excel('AnalyzedData/.../quality_metrics.xlsx')

# Template metrics (waveform shape features)
template_metrics = pd.read_excel('AnalyzedData/.../template_metrics.xlsx')

# Network burst data
with open('AnalyzedData/.../network_data.json', 'r') as f:
    network_data = json.load(f)
```

### Burst Analysis Methods

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

## Key Jupyter Notebooks

### Primary Analysis (workbooks/)

| Notebook | Purpose |
|----------|---------|
| `spikeTImesProcessing.ipynb` | Spike time post-processing, custom burst detection |
| `CIRM_figures.ipynb` | Publication figures (CDKL5, organoids) |
| `SpikeSortingAlgorithmFigures.ipynb` | Algorithm validation, quality metrics |
| `analysis_general.ipynb` | General workflows, templates |
| `compare_two_sorters.ipynb` | Kilosort version comparison |

### Visualization (Plotting/)

| Notebook | Purpose |
|----------|---------|
| `plotNetworkActivtitySpikeSortedMetrics.ipynb` | Firing rate plots, burst statistics |
| `amplitude_graphs.ipynb` | Amplitude analysis visualizations |

## Architecture

### Core Modules

- **mea_analysis_routine.py**: Main pipeline class (`MEAPipeline`) with 4 checkpointed stages
- **run_pipeline_driver.py**: Batch processing coordinator for multiple wells/files
- **helper_functions.py**: Burst detection utilities, plotting, file I/O
- **gaussianNetworkBursts.py**: Gaussian-smoothed burst detection (σ=100ms, mean+3σ threshold)
- **parameter_free_burst_detector.py**: Adaptive burst detection with hierarchical merging (burstlets → network_bursts → superbursts)

### Pipeline Output Structure

```
AnalyzedData/{project}/{date}/{chip_id}/{run_id}/{well_id}/
├── quality_metrics.xlsx              # Unit quality (SNR, firing rate, etc.)
├── template_metrics.xlsx             # Waveform shape features
├── network_data.json                 # Network burst statistics
├── spikesorted_spike_times_dict.npy  # Spike times per unit
├── waveforms/                        # PDF waveform plots
├── spike_sorted_raster_plot.svg      # Network raster
├── locations_*.pdf                   # Probe location plots
├── neuron_spatial_density.pdf        # Per-spike scatter (spatial)
├── neuron_spatial_amplitude.pdf      # Unit amplitude heatmap (spatial)
└── neuron_spatial_combined.pdf       # Side-by-side panel view (spatial)
```

### Data Flow

```
Raw .h5 → Preprocessing (300-3000Hz bandpass, CMR) → Kilosort → Quality Metrics → Burst Analysis → Outputs
```

### Pipeline Stages (Checkpointed)

The `MEAPipeline` class in `mea_analysis_routine.py` processes each well through 4 checkpointed stages:

| Stage | Checkpoint | Description |
|-------|------------|-------------|
| 1. PREPROCESSING | - | Load Maxwell .h5 → Bandpass 300-3000Hz → Local CMR (250µm) → Binary |
| 2. SORTING | `SORTING_COMPLETE` | Run Kilosort → Remove empty/duplicate units (0.1ms window) |
| 3. ANALYZER | `ANALYZER_COMPLETE` | SortingAnalyzer → Sparse waveforms (50µm) → Quality/template metrics |
| 4. REPORTS | `REPORTS_COMPLETE` | Export metrics → Curation → Burst detection → Visualizations |

### Supporting Modules

| Module | Purpose |
|--------|---------|
| `helper_functions.py` | File I/O, burst detection (`detect_bursts_statistics`), plotting, directory utilities |
| `parameter_free_burst_detector.py` | Adaptive burst detection with hierarchical merging (burstlets → network_bursts → superbursts) |
| `gaussianNetworkBursts.py` | Gaussian-smoothed burst detection (σ=100ms, threshold=mean+3σ) |
| `neuron_spatial_maps.py` | Spatial visualization: spike density maps, unit amplitude heatmaps |
| `scalebury.py` | Matplotlib scale bar rendering for publication figures |
| `pipeline_gui.py` | Tkinter GUI for `run_pipeline_driver.py` with file browsers and live command preview |

## Tech Stack

- **Python**: 3.12
- **SpikeInterface**: 0.103.x
- **Kilosort**: 4.x (supports 2, 2.5, 3, 4)
- **PyTorch**: 2.x + CUDA 12.x
- **Scientific**: numpy, scipy, pandas, matplotlib, seaborn
- **Data I/O**: h5py, zarr

## CLI Reference

### mea_analysis_routine.py

| Flag | Description |
|------|-------------|
| `--well` | Well ID (e.g., well000) [required] |
| `--output-dir` | Output directory [required] |
| `--sorter` | Spike sorter (kilosort2, kilosort2_5, kilosort3, kilosort4) |
| `--analysis` | Comma-separated analyses or preset (see above) |
| `--params` | JSON file with quality thresholds |
| `--checkpoint-dir` | Checkpoint directory |
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
| `--reference` | Excel file for filtering runs |
| `--type` | Assay types to include (default: "network today", "network today/best") |
| `--dry` | Dry run (no processing) |

## Quality Thresholds

Defined in `sorting_quality_threshold_params.json`:

| Metric | Threshold | Meaning |
|--------|-----------|---------|
| `num_spikes` | > 300 | Minimum spike count |
| `presence_ratio` | > 0.9 | Unit active throughout recording |
| `rp_contamination` | < 1.0 | Refractory period violations acceptable |
| `firing_rate` | > 0.05 Hz | Minimum activity |
| `amplitude_median` | ≤ -20 µV | Minimum signal amplitude |

## Output Metrics Reference

### Quality Metrics
`num_spikes`, `firing_rate`, `presence_ratio`, `snr`, `rp_contamination`, `isi_violations_ratio`, `amplitude_median`, `amplitude_cutoff`, `location_X/Y/Z`

### Template Metrics
`peak_to_valley`, `halfwidth`, `repolarization_slope`, `recovery_slope`, `num_positive_peaks`, `num_negative_peaks`

### Network Metrics
`num_network_bursts`, `network_burst_rate`, `mean_IBI`, `CoV_IBI`, `mean_network_burst_duration`, `CoV_burst_duration`, `mean_network_burst_peak`, `mean_spikes_per_burst`, `network_burst_percentage`, `MeanWithinBurstISI`, `CoVWithinBurstISI`, `active_electrodes`, `NumUnits`

## Important Constants

```python
# Preprocessing
freq_min, freq_max = 300, 3000  # Hz
reference = 'local', radius = 250  # µm

# Sparse waveforms
radius_um = 50
peak_sign = 'neg'

# Burst detection defaults
ISI_threshold = 0.1  # seconds
gaussianSigma = 0.1  # seconds
min_spikes_in_burst = 3
```

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **collaborative research codebase** for analyzing Microelectrode Array (MEA) neuronal recordings from Maxwell MEA chips. The repository contains:

1. **Spike sorting pipeline** - Processes raw .h5 files through Kilosort (run by pipeline operators)
2. **Analysis notebooks** - Post-processing, visualization, and publication figures (most common usage)

**Most users work with pre-processed data** in Jupyter notebooks. The pipeline is typically operated centrally, and analysts load the resulting spike times, metrics, and network data for custom analysis.

**Primary Research Focus**: CDKL5 disease model analysis, network burst characterization, organoid studies

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

### Custom Burst Analysis

```python
from helper_functions import detect_bursts_statistics, plot_raster_with_bursts
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

### Primary Analysis (`workbooks/`)

| Notebook | Purpose |
|----------|---------|
| `spikeTImesProcessing.ipynb` | Spike time post-processing, custom burst detection |
| `CIRM_figures.ipynb` | Publication figures (CDKL5, organoids) |
| `SpikeSortingAlgorithmFigures.ipynb` | Algorithm validation, quality metrics |
| `analysis_general.ipynb` | General workflows, templates |
| `compare_two_sorters.ipynb` | Kilosort version comparison |

### Visualization (`Plotting/`)

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
├── network_data.json                 # Network burst statistics (requires 'burst' in --analysis)
├── spikesorted_spike_times_dict.npy  # Spike times per unit
├── waveforms/                        # PDF waveform plots (requires 'waveforms' in --analysis)
├── spike_sorted_raster_plot.svg      # Network raster (requires 'raster' in --analysis)
├── locations_*.pdf                   # Probe location plots (requires 'probe' in --analysis)
├── neuron_spatial_density.pdf        # Per-spike scatter (requires 'spatial' in --analysis)
├── neuron_spatial_amplitude.pdf      # Unit amplitude heatmap (requires 'spatial' in --analysis)
└── neuron_spatial_combined.pdf       # Side-by-side panel view (requires 'spatial' in --analysis)
```

### Data Flow

```
Raw .h5 → Preprocessing (300-3000Hz bandpass, CMR) → Kilosort → Quality Metrics → Burst Analysis → Outputs
```

### Detailed Pipeline Workflow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         run_pipeline_driver.py                              │
│  (Batch Coordinator)                                                        │
│                                                                             │
│  Input: Directory or single .h5 file                                        │
│    ↓                                                                        │
│  1. Scan for data.raw.h5 files in "Network" subfolders                      │
│  2. Optionally filter by reference Excel (Assay type)                       │
│  3. Open each .h5 → extract well IDs (well000, well001, ...)                │
│  4. For each well → spawn subprocess:                                       │
│     python3 mea_analysis_routine.py <file> --well <well_id> <args>          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
            ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
            │ well000      │ │ well001      │ │ well002      │
            │ subprocess   │ │ subprocess   │ │ subprocess   │
            └──────────────┘ └──────────────┘ └──────────────┘
                    │               │               │
                    └───────────────┼───────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         mea_analysis_routine.py                             │
│  (MEAPipeline class - per well)                                             │
│                                                                             │
│  Stage 1: PREPROCESSING                                                     │
│    └─ Load Maxwell .h5 → Bandpass 300-3000Hz → Local CMR → Save Zarr        │
│                                                                             │
│  Stage 2: SORTING (checkpointed)                                            │
│    └─ Run Kilosort4 → Remove empty/duplicate units                          │
│                                                                             │
│  Stage 3: ANALYZER (checkpointed)                                           │
│    └─ Create SortingAnalyzer → Compute waveforms, templates, quality metrics│
│                                                                             │
│  Stage 4: REPORTS (checkpointed)                                            │
│    └─ Export metrics → Apply curation → Burst detection → Visualizations    │
│                                                                             │
│  Uses: helper_functions.py, parameter_free_burst_detector.py,               │
│        gaussianNetworkBursts.py, scalebury.py                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              OUTPUT FILES                                   │
│  AnalyzedData/{project}/{date}/{chip_id}/{run_id}/{well_id}/                │
│    ├── quality_metrics.xlsx                                                 │
│    ├── template_metrics.xlsx                                                │
│    ├── network_data.json                                                    │
│    ├── spikesorted_spike_times_dict.npy                                     │
│    ├── waveforms/*.pdf                                                      │
│    └── spike_sorted_raster_plot.svg                                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Script Responsibilities

#### run_pipeline_driver.py (Entry Point)

**Purpose**: Batch coordinator that discovers data files and launches per-well processing

**Workflow**:
1. Parse command-line arguments
2. If directory mode:
   - Scan for `data.raw.h5` files in "Network" subfolders using `helper.find_files_with_subfolder()`
   - Optionally filter runs by reference Excel file (Assay type matching)
3. If single file mode:
   - Open the .h5 file directly
4. For each .h5 file:
   - Open with h5py and read `h5f["wells"].keys()` to get well IDs
   - For each well, spawn a **subprocess** calling `mea_analysis_routine.py`

**Key Function**: `launch_sorting_subprocess(file_path, stream_id, extra_args)`
- Builds command: `python3 mea_analysis_routine.py '<file>' --well <well_id> <args>`
- Runs via `subprocess.run()`

#### mea_analysis_routine.py (Core Pipeline)

**Purpose**: Process a single well through the complete spike sorting and analysis pipeline

**Class**: `MEAPipeline`

**Checkpointed Stages**:

| Stage | Checkpoint | Description |
|-------|------------|-------------|
| 1. PREPROCESSING | - | Load Maxwell .h5 via SpikeInterface → Bandpass filter (300-3000 Hz) → Local common median reference (250µm radius) → Convert to Int16 → Save as Zarr/binary |
| 2. SORTING | `SORTING_COMPLETE` | Run Kilosort (2/2.5/3/4) → Remove empty units → Remove excess spikes → Remove duplicates (0.1ms window) |
| 3. ANALYZER | `ANALYZER_COMPLETE` | Create SortingAnalyzer → Estimate sparsity (50µm radius) → Compute extensions: waveforms, templates, noise_levels, quality_metrics, template_metrics, unit_locations, spike_locations |
| 4. REPORTS | `REPORTS_COMPLETE` | Export quality_metrics.xlsx → Apply automatic curation (thresholds from JSON) → Merge similar templates (>0.7 correlation) → Burst detection → Generate visualizations → Export network_data.json |

### Supporting Modules

| Module | Purpose |
|--------|---------|
| `helper_functions.py` | File I/O (`load_json`, `save_json`), burst detection (`detect_bursts_statistics`), plotting (`plot_raster_with_bursts`, `plot_network_activity`), directory utilities |
| `parameter_free_burst_detector.py` | Adaptive network burst detection with dual Gaussian smoothing, hierarchical merging (burstlets → network_bursts → superbursts) |
| `gaussianNetworkBursts.py` | Gaussian-smoothed burst detection (σ=100ms, threshold=mean+3σ) |
| `neuron_spatial_maps.py` | Spatial visualization: spike density maps (dynamic alpha), unit amplitude heatmaps (µV colormap), combined panel views |
| `scalebury.py` | Matplotlib scale bar rendering for publication figures |

## Tech Stack

- **Python**: 3.12.9
- **SpikeInterface**: 0.103.0
- **Kilosort**: 4.1.1
- **PyTorch**: 2.9.0 + CUDA 12.8
- **Scientific**: numpy, scipy, pandas, matplotlib, seaborn
- **Data I/O**: h5py, zarr

## Burst Detection Methods

| Method | Module | Key Parameters | Use Case |
|--------|--------|----------------|----------|
| ISI Threshold | `helper_functions.py` | `isi_threshold=0.1s` | Simple, fast |
| Gaussian | `gaussianNetworkBursts.py` | `gaussianSigma=0.1s`, threshold=mean+3σ | Visualization |
| Parameter-free | `parameter_free_burst_detector.py` | Adaptive bin size, dual smoothing | Advanced analysis |

---

## Pipeline Execution (Operators Only)

### Single Recording
```bash
python mea_analysis_routine.py /path/to/data.raw.h5 \
  --well well000 \
  --output-dir ./AnalyzedData \
  --sorter kilosort4 \
  --clean-up
```

### Batch Processing
```bash
python run_pipeline_driver.py /path/to/data/directory \
  --reference experiment_metadata.xlsx \
  --type "network today" "network best" \
  --sorter kilosort4 \
  --output-dir ./AnalyzedData
```

### Command-Line Arguments

**mea_analysis_routine.py**:
- `--well`: Well ID (e.g., well000) [required]
- `--output-dir`: Output directory [required]
- `--sorter`: Spike sorter (kilosort2, kilosort2_5, kilosort3, kilosort4)
- `--params`: JSON file with quality thresholds
- `--checkpoint-dir`: Checkpoint directory
- `--docker`: Docker image for containerized sorting
- `--debug`: Enable debug logging
- `--clean-up`: Delete intermediate files after processing
- `--force-restart`: Restart from scratch, ignoring checkpoints
- `--export-to-phy`: Export for manual curation in Phy GUI
- `--no-curation`: Skip automatic quality filtering
- `--skip-spikesorting`: Skip sorting stage (use existing results)
- `--reanalyze-bursts`: Re-run burst analysis only

**Analysis Selection Flag** (available in both scripts):
- `--analysis <list>`: Comma-separated list of analyses to run
  - **Valid analyses**: `probe`, `waveforms`, `raster`, `burst`, `spatial`
  - **Presets**: `default` (probe,waveforms,raster,burst), `all`, `minimal` (burst only), `none`
  - **Examples**:
    - `--analysis default,spatial` - Run all default analyses plus spatial maps
    - `--analysis burst` - Only burst metrics (no plots)
    - `--analysis raster` - Raster plots (auto-enables burst)
    - `--analysis all` - All analyses including spatial
  - **Note**: If `raster` is specified, `burst` is auto-enabled (required for plot data)

**run_pipeline_driver.py** (additional):
- `--reference`: Excel file for filtering runs
- `--type`: Assay types to include (default: "network today", "network today/best")
- `--dry`: Dry run (no processing)

### Pipeline Stages (Checkpointed)

1. **PREPROCESSING**: Load Maxwell .h5 → Bandpass 300-3000Hz → Local CMR → Zarr/binary
2. **SORTING**: Kilosort → Remove empty/duplicate units → Checkpoint
3. **ANALYZER**: SortingAnalyzer → Sparse waveforms (50µm) → Extensions → Checkpoint
4. **REPORTS**: Quality metrics → Curation → Burst detection → Visualizations → JSON export

### Quality Thresholds (sorting_quality_threshold_params.json)
- `num_spikes > 300`: Minimum spike count
- `presence_ratio > 0.9`: Unit active throughout recording
- `rp_contamination < 1.0`: Refractory period violations
- `firing_rate > 0.05 Hz`: Minimum activity
- `amplitude_median <= -20 µV`: Minimum signal amplitude

### Monitoring Dashboard
```bash
streamlit run streamlit_checkpoint_analyzer/checkpoint_dashboard.py
```

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

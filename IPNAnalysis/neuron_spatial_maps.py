"""
Neuron Spatial Map Visualization Module

Standalone module for generating comprehensive neuron spatial visualizations
from spike-sorted MEA data. Follows the same modular pattern as
parameter_free_burst_detector.py.

Output Files:
    - neuron_spatial_density.pdf: Per-spike scatter visualization (dynamic alpha)
    - neuron_spatial_amplitude.pdf: Unit amplitude heatmap (µV colormap)
    - neuron_spatial_combined.pdf: Side-by-side panel view
"""

import numpy as np
import matplotlib.pyplot as plt
import spikeinterface.full as si
from pathlib import Path


def plot_neuron_spatial_maps(
    recording,
    spike_locs,
    unit_locations,
    amplitudes,
    output_dir,
    plot=True,
    verbose=True
):
    """
    Generate comprehensive neuron spatial visualization maps.

    Parameters
    ----------
    recording : BaseRecording
        SpikeInterface recording object (for probe map)
    spike_locs : structured array
        Per-spike locations with 'x', 'y' fields from spike_locations extension
    unit_locations : ndarray
        Per-unit locations (N_units, 2)
    amplitudes : ndarray
        Per-unit amplitude_median values
    output_dir : Path or str
        Directory to save output files
    plot : bool
        Whether to generate plots (default True)
    verbose : bool
        Whether to print progress messages

    Returns
    -------
    dict
        Statistics and metadata about the spatial distributions
    """
    if not plot:
        return {"status": "skipped"}

    output_dir = Path(output_dir)
    results = {}

    # 1. Spike density map
    _plot_spike_density_map(recording, spike_locs, output_dir, verbose)

    # 2. Amplitude map
    _plot_amplitude_map(recording, unit_locations, amplitudes, output_dir, verbose)

    # 3. Combined panel
    _plot_combined_spatial(recording, spike_locs, unit_locations, amplitudes,
                           output_dir, verbose)

    # Compute statistics
    results["total_spikes"] = len(spike_locs)
    results["num_units"] = len(unit_locations)
    results["amplitude_range"] = [float(np.min(amplitudes)), float(np.max(amplitudes))]
    results["spatial_extent"] = {
        "x_range": [float(np.min(spike_locs['x'])), float(np.max(spike_locs['x']))],
        "y_range": [float(np.min(spike_locs['y'])), float(np.max(spike_locs['y']))]
    }
    results["status"] = "completed"

    return results


def _plot_spike_density_map(recording, spike_locs, output_dir, verbose=True):
    """Plot per-spike scatter with dynamic alpha."""
    fig, ax = plt.subplots(figsize=(10.5, 6.5))

    si.plot_probe_map(recording, ax=ax, with_channel_ids=False)

    x = spike_locs['x']
    y = spike_locs['y']
    total_spikes = len(spike_locs)

    # Dynamic alpha: scale inversely with spike count
    alpha = np.clip(200 / total_spikes, 0.001, 0.1)

    ax.scatter(x, y, color='purple', alpha=alpha, s=1, rasterized=True)
    ax.invert_yaxis()
    ax.set_title(f"Spike Density Map ({total_spikes:,} spikes, α={alpha:.4f})")
    ax.set_xlabel("X (µm)")
    ax.set_ylabel("Y (µm)")

    fig.tight_layout()
    fig.savefig(output_dir / "neuron_spatial_density.pdf", dpi=150)
    plt.close(fig)

    if verbose:
        print(f"Saved: neuron_spatial_density.pdf")


def _plot_amplitude_map(recording, unit_locations, amplitudes, output_dir, verbose=True):
    """Plot unit locations colored by amplitude."""
    fig, ax = plt.subplots(figsize=(10.5, 6.5))

    si.plot_probe_map(recording, ax=ax, with_channel_ids=False)

    abs_amplitudes = np.abs(amplitudes)

    scatter = ax.scatter(
        unit_locations[:, 0], unit_locations[:, 1],
        s=80, c=abs_amplitudes, cmap='plasma', alpha=0.8,
        edgecolors='white', linewidths=0.5
    )
    ax.invert_yaxis()

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('|Amplitude| (µV)')

    ax.set_title(f"Unit Amplitude Map ({len(unit_locations)} units)")
    ax.set_xlabel("X (µm)")
    ax.set_ylabel("Y (µm)")

    fig.tight_layout()
    fig.savefig(output_dir / "neuron_spatial_amplitude.pdf")
    plt.close(fig)

    if verbose:
        print(f"Saved: neuron_spatial_amplitude.pdf")


def _plot_combined_spatial(recording, spike_locs, unit_locations, amplitudes,
                           output_dir, verbose=True):
    """Create combined side-by-side panel."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 6.5))

    # Left: Spike density
    ax_density = axes[0]
    si.plot_probe_map(recording, ax=ax_density, with_channel_ids=False)

    x = spike_locs['x']
    y = spike_locs['y']
    total_spikes = len(spike_locs)
    alpha = np.clip(200 / total_spikes, 0.001, 0.1)

    ax_density.scatter(x, y, color='purple', alpha=alpha, s=1, rasterized=True)
    ax_density.invert_yaxis()
    ax_density.set_title(f"Spike Density ({total_spikes:,} spikes)")
    ax_density.set_xlabel("X (µm)")
    ax_density.set_ylabel("Y (µm)")

    # Right: Amplitude
    ax_amp = axes[1]
    si.plot_probe_map(recording, ax=ax_amp, with_channel_ids=False)

    abs_amplitudes = np.abs(amplitudes)
    scatter = ax_amp.scatter(
        unit_locations[:, 0], unit_locations[:, 1],
        s=80, c=abs_amplitudes, cmap='plasma', alpha=0.8,
        edgecolors='white', linewidths=0.5
    )
    ax_amp.invert_yaxis()

    cbar = plt.colorbar(scatter, ax=ax_amp)
    cbar.set_label('|Amplitude| (µV)')

    ax_amp.set_title(f"Unit Amplitude ({len(unit_locations)} units)")
    ax_amp.set_xlabel("X (µm)")
    ax_amp.set_ylabel("Y (µm)")

    fig.tight_layout()
    fig.savefig(output_dir / "neuron_spatial_combined.pdf", dpi=150)
    plt.close(fig)

    if verbose:
        print(f"Saved: neuron_spatial_combined.pdf")

"""
Burst-silence kernel for MEA network burst detection.

This module provides a custom convolution kernel designed to detect burst events
in neural recordings. A burst is characterized by:
1. A period of synchronized firing (multiple channels firing in a short window)
2. Followed by a more silent period marking the end of the burst

The kernel computes: mean(burst_window) - mean(silence_window)
High positive values indicate burst endings (high activity followed by silence).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def burst_silence_kernel1d(
    fs_hz: float,
    burst_ms: float = 100.0,
    silence_ms: float = 200.0,
    *,
    normalize: bool = True,
    dtype=np.float64,
) -> tuple[np.ndarray, int]:
    """
    Build a discrete "burst-then-silence" kernel and the matching ndimage origin.

    Kernel shape (boxcar lobes):
      + (burst window) then - (silence window)

    When applied with ndimage.correlate1d(..., origin=Nb-1), the score at index t is:
      mean(x[t-Nb+1 : t+1]) - mean(x[t+1 : t+1+Ns])

    Parameters
    ----------
    fs_hz : float
        Sampling frequency in Hz (samples per second).
    burst_ms : float, default=100.0
        Duration of the burst window in milliseconds.
    silence_ms : float, default=200.0
        Duration of the silence window in milliseconds.
    normalize : bool, default=True
        If True, normalize each lobe to have unit sum (computes means).
        If False, use raw sums.
    dtype : numpy dtype, default=np.float64
        Data type for the kernel weights.

    Returns
    -------
    weights : np.ndarray
        1D weights of length Nb + Ns
    origin : int
        Use this origin when calling ndimage.correlate1d so "now" is at end of burst window.
    """
    if fs_hz <= 0:
        raise ValueError("fs_hz must be > 0")
    if burst_ms <= 0 or silence_ms <= 0:
        raise ValueError("burst_ms and silence_ms must be > 0")

    Nb = int(round(fs_hz * (burst_ms / 1000.0)))
    Ns = int(round(fs_hz * (silence_ms / 1000.0)))
    Nb = max(Nb, 1)
    Ns = max(Ns, 1)

    if normalize:
        pos = np.full(Nb, 1.0 / Nb, dtype=dtype)
        neg = np.full(Ns, -1.0 / Ns, dtype=dtype)
    else:
        pos = np.ones(Nb, dtype=dtype)
        neg = -np.ones(Ns, dtype=dtype)

    weights = np.concatenate([pos, neg]).astype(dtype, copy=False)

    # Align weights[Nb-1] ("end of burst window") to the current sample.
    origin = Nb - 1
    return weights, origin


def burst_silence_filter1d(
    input,
    fs_hz: float,
    burst_ms: float = 100.0,
    silence_ms: float = 200.0,
    axis: int = -1,
    output=None,
    mode: str = "reflect",
    cval: float = 0.0,
    *,
    normalize: bool = True,
):
    """
    Apply the "burst-then-silence" kernel along one axis, SciPy-gaussian_filter1d style.

    This filter computes at each time point:
        mean(signal over burst_window) - mean(signal over silence_window)

    High positive values indicate transitions from high activity to low activity,
    which marks burst endings.

    Parameters
    ----------
    input : array_like
        Input array to filter.
    fs_hz : float
        Sampling frequency in Hz.
    burst_ms : float, default=100.0
        Duration of the burst window in milliseconds.
    silence_ms : float, default=200.0
        Duration of the silence window in milliseconds.
    axis : int, default=-1
        Axis along which to apply the filter.
    output : array or dtype, optional
        Output array or dtype. If None, a new array is created.
    mode : str, default='reflect'
        Boundary mode: 'reflect', 'constant', 'nearest', 'mirror', 'wrap'.
    cval : float, default=0.0
        Value for 'constant' mode.
    normalize : bool, default=True
        If True, compute means; if False, compute sums.

    Returns
    -------
    filtered : np.ndarray
        Filtered signal with same shape as input.
    """
    weights, origin = burst_silence_kernel1d(
        fs_hz=fs_hz,
        burst_ms=burst_ms,
        silence_ms=silence_ms,
        normalize=normalize,
        dtype=np.float64,
    )

    return ndimage.correlate1d(
        input,
        weights=weights,
        axis=axis,
        output=output,
        mode=mode,
        cval=cval,
        origin=origin,
    )

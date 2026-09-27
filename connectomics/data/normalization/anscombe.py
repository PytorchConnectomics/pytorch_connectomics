"""Noise-stabilizing intensity normalization for shot-noise-limited fluorescence.

Photon-counting microscopy (e.g. ExM expansion light microscopy) has noise
variance that grows linearly with signal, ``variance = gain * (x - offset)``.
The generalized Anscombe transform ``2 * sqrt((x - offset) / gain + 3/8)``
turns that into ~unit noise at every brightness, so dim neuropil and bright
synaptic puncta are weighted alike by pixel losses. A per-volume affine stretch
then maps the transformed intensities to the full uint8 range.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

__all__ = [
    "anscombe_to_uint8",
    "anscombe_transform",
    "estimate_noise_model",
    "fit_anscombe_mapping",
]

# Noise-model fit: per-block noise from the residual after Gaussian smoothing,
# measured only on the flattest blocks of each brightness bin so membranes and
# puncta do not inflate the variance.
_NOISE_BLOCK = 8
_NOISE_SMOOTH_SIGMA = 2.0
_NOISE_GRADIENT_SIGMA = 3.0
_NOISE_FLAT_FRACTION = 0.15
_NOISE_BINS = 32
_NOISE_MIN_BLOCKS_PER_BIN = 50
_NOISE_MIN_BINS = 8
# Variance kept by white noise under (identity - 2D Gaussian of sigma s):
# 1 - 2 * G(0) + sum(G^2) = 1 - 3 / (4 * pi * s^2).
_NOISE_RESIDUAL_VARIANCE = 1.0 - 3.0 / (4.0 * np.pi * _NOISE_SMOOTH_SIGMA**2)


def estimate_noise_model(planes: list[np.ndarray]) -> tuple[float, float]:
    """Fit shot-noise ``variance = gain * (mean - offset)`` to XY planes.

    Returns ``(gain, offset)`` in raw intensity units. Raises when the planes
    hold too little flat signal to fit, or the fit is not shot-noise-like.
    """
    means, variances, gradients = [], [], []
    for plane in planes:
        plane = np.asarray(plane, dtype=np.float64)
        rows = plane.shape[0] // _NOISE_BLOCK * _NOISE_BLOCK
        cols = plane.shape[1] // _NOISE_BLOCK * _NOISE_BLOCK
        if rows == 0 or cols == 0:
            continue
        smooth = ndimage.gaussian_filter(plane, _NOISE_SMOOTH_SIGMA)
        gradient = ndimage.gaussian_gradient_magnitude(plane, _NOISE_GRADIENT_SIGMA)

        def blocks(image: np.ndarray) -> np.ndarray:
            return (
                image[:rows, :cols]
                .reshape(rows // _NOISE_BLOCK, _NOISE_BLOCK, cols // _NOISE_BLOCK, _NOISE_BLOCK)
                .transpose(0, 2, 1, 3)
                .reshape(-1, _NOISE_BLOCK * _NOISE_BLOCK)
            )

        residual = blocks(plane - smooth)
        mad = np.median(np.abs(residual - np.median(residual, axis=1, keepdims=True)), axis=1)
        means.append(blocks(smooth).mean(axis=1))
        variances.append((1.4826 * mad) ** 2 / _NOISE_RESIDUAL_VARIANCE)
        gradients.append(blocks(gradient).mean(axis=1))
    if not means:
        raise ValueError("Planes are too small to estimate a noise model.")
    means, variances, gradients = (np.concatenate(v) for v in (means, variances, gradients))

    edges = np.unique(np.percentile(means, np.linspace(0.5, 99.8, _NOISE_BINS + 1)))
    bin_index = np.digitize(means, edges)
    bin_means, bin_variances = [], []
    for index in range(1, len(edges)):
        in_bin = bin_index == index
        if in_bin.sum() < _NOISE_MIN_BLOCKS_PER_BIN:
            continue
        flat = in_bin & (gradients <= np.percentile(gradients[in_bin], 100 * _NOISE_FLAT_FRACTION))
        bin_means.append(np.median(means[flat]))
        bin_variances.append(np.median(variances[flat]))
    if len(bin_means) < _NOISE_MIN_BINS:
        raise ValueError(
            f"Only {len(bin_means)} usable brightness bins (need {_NOISE_MIN_BINS}) to "
            "estimate the noise model; pass an explicit noise model."
        )
    gain, intercept = np.polyfit(bin_means, bin_variances, 1)
    if not np.isfinite(gain) or gain <= 0.0:
        raise ValueError(
            f"Noise variance does not grow with intensity (fitted gain {gain}); the data "
            "may be denoised or not shot-noise limited. Use a linear clip "
            "or pass an explicit noise model."
        )
    return float(gain), float(-intercept / gain)


def anscombe_transform(values: np.ndarray, gain: float, offset: float) -> np.ndarray:
    """Generalized Anscombe transform: unit noise for ``var = gain * (x - offset)``."""
    photons = (np.asarray(values, dtype=np.float32) - np.float32(offset)) / np.float32(gain)
    return 2.0 * np.sqrt(np.maximum(photons + np.float32(0.375), 0.0))


def anscombe_to_uint8(
    plane: np.ndarray, *, gain: float, offset: float, stretch_range: tuple[float, float]
) -> np.ndarray:
    """Map one raw XY plane to uint8 by the volume's Anscombe transform and stretch."""
    low, high = stretch_range
    scaled = (anscombe_transform(plane, gain, offset) - low) * (255.0 / (high - low))
    return np.clip(np.rint(scaled), 0.0, 255.0).astype(np.uint8)


def fit_anscombe_mapping(
    structural: np.ndarray,
    *,
    stats_planes: int,
    stretch_percentiles: tuple[float, float],
    noise_model: tuple[float, float] | None = None,
) -> dict:
    """Fit the per-volume noise model and the transformed-intensity stretch.

    ``structural`` is any ZYX array indexable by Z (NumPy, HDF5, Zarr, dask).
    Statistics come from ``stats_planes`` evenly spaced Z planes so the mapping
    is one fixed pointwise function for the whole volume. The returned dict
    holds ``gain``, ``offset``, ``noise_model_source``, ``stretch_percentiles``,
    ``stretch_range`` (transformed values mapped to 0/255) and
    ``stats_z_indices``.
    """
    lower, upper = stretch_percentiles
    if not 0.0 <= lower < upper <= 100.0:
        raise ValueError(
            f"Stretch percentiles must satisfy 0 <= low < high <= 100; got {lower}, {upper}."
        )
    if stats_planes < 1:
        raise ValueError(f"stats_planes must be positive, got {stats_planes}.")
    z_indices = np.unique(
        np.linspace(0, structural.shape[0] - 1, min(stats_planes, structural.shape[0])).round()
    ).astype(int)
    planes = [np.asarray(structural[z]) for z in z_indices]
    if noise_model is None:
        gain, offset = estimate_noise_model(planes)
        source = "estimated"
    else:
        gain, offset = (float(value) for value in noise_model)
        if not np.isfinite(gain) or gain <= 0.0 or not np.isfinite(offset):
            raise ValueError(f"Noise model needs finite gain > 0 and offset; got {noise_model}.")
        source = "given"
    transformed = np.concatenate(
        [anscombe_transform(plane, gain, offset).ravel() for plane in planes]
    )
    low, high = (float(v) for v in np.percentile(transformed, (lower, upper)))
    if high <= low:
        raise ValueError(f"Degenerate transformed intensity range [{low}, {high}].")
    return {
        "gain": gain,
        "offset": offset,
        "noise_model_source": source,
        "stretch_percentiles": (float(lower), float(upper)),
        "stretch_range": (low, high),
        "stats_z_indices": z_indices,
    }

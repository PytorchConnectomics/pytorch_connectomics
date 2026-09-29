"""Shot-noise model of photon-counting fluorescence: variance = gain * (x - offset)."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

__all__ = ["estimate_noise_model", "noise_curve"]

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


def noise_curve(planes: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Per-brightness-bin (median mean, median variance) of the flattest blocks.

    These are the points ``estimate_noise_model`` fits a line to.
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
    return np.asarray(bin_means), np.asarray(bin_variances)


def estimate_noise_model(planes: list[np.ndarray]) -> tuple[float, float]:
    """Fit shot-noise ``variance = gain * (mean - offset)`` to XY planes.

    Returns ``(gain, offset)`` in raw intensity units. ``offset`` is the
    empirical intercept ``-b / a`` of ``variance = a * mean + b``; with read
    noise it sits below the true dark level, so prefer a measured dark frame
    when one exists. Raises when the planes hold too little flat signal to fit,
    or the fit is not shot-noise-like.
    """
    bin_means, bin_variances = noise_curve(planes)
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

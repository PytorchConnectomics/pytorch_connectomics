"""Per-volume alignment of raw fluorescence to a shared intensity profile.

``r = (x - offset) / (flat_field(y, x) * tissue_median)``: 0 is no light (the
camera offset from the shot-noise fit), 1 is the median of tissue pixels, and
the smooth illumination falloff across the field of view is divided out. This
is the only per-volume step; the uint8 code table (``codec``) and the display
curve (``display``) are fixed for every volume, so data acquired with different
microscopes, gains, exposures and labeling brightness land on roughly the same
scale. Everything else is left to the model.

All statistics come from a few evenly spaced Z planes, so the mapping is one
fixed pointwise function per volume and can be applied to streamed planes.
"""

from __future__ import annotations

import numpy as np

from .codec import codec_attrs, quantization_report
from .display import display_attrs
from .noise import estimate_noise_model

__all__ = [
    "fit_alignment",
    "flat_field_image",
    "to_aligned",
    "alignment_attrs",
]

# Flat field: exp(quadratic in normalized (y, x)) fitted to per-block medians
# of tissue signal. Vignetting is smooth, so six terms suffice and cell-scale
# structure cannot leak into the correction.
_FLAT_FIELD_MODEL = "exp_quadratic_yx"
_FLAT_MIN_BLOCK = 8
_FLAT_MAX_BLOCK = 64
_FLAT_BLOCKS_PER_SIDE = 16
_FLAT_MIN_TISSUE_BLOCKS = 12
_FLAT_OUTLIER_MADS = 3.0
_MIN_TISSUE_FRACTION = 0.05
_STATS_PIXELS_PER_PLANE = 1_000_000


def _quadratic_terms(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.stack([np.ones_like(u), u, v, u * u, u * v, v * v], axis=-1)


def _normalized_coords(size: int) -> np.ndarray:
    """Pixel-center coordinates mapped to [-1, 1]."""
    return (np.arange(size, dtype=np.float64) + 0.5) / size * 2.0 - 1.0


def _block_size(shape_yx: tuple[int, int]) -> int:
    rows, cols = shape_yx
    block = max(_FLAT_MIN_BLOCK, min(_FLAT_MAX_BLOCK, min(rows, cols) // _FLAT_BLOCKS_PER_SIDE))
    if rows // block < 3 or cols // block < 3:
        raise ValueError(f"Planes {shape_yx} are too small to fit a flat field.")
    return block


def _block_medians(signal: np.ndarray, block: int) -> np.ndarray:
    n_rows, n_cols = signal.shape[0] // block, signal.shape[1] // block
    return np.median(
        signal[: n_rows * block, : n_cols * block]
        .reshape(n_rows, block, n_cols, block)
        .transpose(0, 2, 1, 3)
        .reshape(n_rows, n_cols, -1),
        axis=-1,
    )


def _tissue_threshold(medians: np.ndarray, block: int, gain: float, tissue_snr: float):
    """Block-median signal a block needs to count as tissue.

    ``tissue_snr`` standard errors of the block median under the shot-noise
    model (SD of a median of n pixels ~ 1.2533 * sigma / sqrt(n)), floored at
    one gain unit so errors in the fitted offset do not turn empty regions
    into tissue. Regional, so dim acquisitions are not reduced to their
    brightest pixels.
    """
    sigma = np.sqrt(gain * np.maximum(medians, gain))
    return np.maximum(tissue_snr * 1.2533 * sigma / block, gain)


def _fit_flat_field(block_medians: np.ndarray, tissue: np.ndarray, block: int, shape) -> dict:
    rows, cols = shape
    medians = np.median(block_medians, axis=0)
    n_rows, n_cols = medians.shape
    centers_u = (np.arange(n_rows) + 0.5) * block / rows * 2.0 - 1.0
    centers_v = (np.arange(n_cols) + 0.5) * block / cols * 2.0 - 1.0
    u, v = np.meshgrid(centers_u, centers_v, indexing="ij")
    terms = _quadratic_terms(u, v)
    # A block is used for the fit when it is tissue in most sampled planes.
    valid = (tissue.mean(axis=0) > 0.5) & (medians > 0)
    if valid.sum() < _FLAT_MIN_TISSUE_BLOCKS:
        raise ValueError(
            f"Only {int(valid.sum())} tissue blocks (need {_FLAT_MIN_TISSUE_BLOCKS}) to fit "
            "a flat field; disable the flat-field correction for this volume."
        )
    log_medians = np.log(medians[valid])
    keep = np.ones(log_medians.shape, dtype=bool)
    for _ in range(2):
        coeffs, *_ = np.linalg.lstsq(terms[valid][keep], log_medians[keep], rcond=None)
        residual = log_medians - terms[valid] @ coeffs
        mad = np.median(np.abs(residual - np.median(residual)))
        keep = np.abs(residual - np.median(residual)) <= _FLAT_OUTLIER_MADS * 1.4826 * max(
            mad, 1e-6
        )
    # Unit median over the frame, so the flat field redistributes brightness
    # across the field of view without rescaling the volume.
    coeffs[0] -= np.median(terms @ coeffs)
    return {
        "model": _FLAT_FIELD_MODEL,
        "coefficients": [float(c) for c in coeffs],
        "frame_shape_yx": [int(rows), int(cols)],
        "block_size": int(block),
        "tissue_blocks": int(valid.sum()),
    }


def flat_field_image(alignment: dict, shape_yx: tuple[int, int]) -> np.ndarray:
    """The volume's illumination profile at ``shape_yx`` (all ones if disabled)."""
    flat = alignment.get("flat_field")
    if flat is None:
        return np.ones(shape_yx, dtype=np.float32)
    if flat["model"] != _FLAT_FIELD_MODEL:
        raise ValueError(f"Unknown flat-field model {flat['model']!r}.")
    u, v = np.meshgrid(
        _normalized_coords(shape_yx[0]), _normalized_coords(shape_yx[1]), indexing="ij"
    )
    return np.exp(_quadratic_terms(u, v) @ np.asarray(flat["coefficients"])).astype(np.float32)


def fit_alignment(
    structural,
    *,
    stats_planes: int,
    noise_model: tuple[float, float] | None = None,
    flat_field: bool = True,
    tissue_snr: float = 3.0,
) -> dict:
    """Fit offset, gain, flat field and tissue median for one ZYX volume.

    ``structural`` is any ZYX array indexable by Z (NumPy, HDF5, Zarr, dask).
    Tissue is decided per block of pixels (the flat-field block size): a block
    is tissue when its median signal is at least ``tissue_snr`` standard errors
    of that median above the offset (and at least one gain unit), so empty or
    resin regions do not pull the anchor down while dim acquisitions keep all
    of their tissue. Returns a JSON-ready dict.
    """
    if stats_planes < 1:
        raise ValueError(f"stats_planes must be positive, got {stats_planes}.")
    if tissue_snr <= 0.0:
        raise ValueError(f"tissue_snr must be positive, got {tissue_snr}.")
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

    signals = [plane.astype(np.float32) - np.float32(offset) for plane in planes]
    shape = signals[0].shape
    block = _block_size(shape)
    block_medians = np.stack([_block_medians(signal, block) for signal in signals])
    tissue = block_medians > _tissue_threshold(block_medians, block, gain, tissue_snr)
    alignment = {
        "flat_field": _fit_flat_field(block_medians, tissue, block, shape) if flat_field else None
    }
    flat = flat_field_image(alignment, shape)

    # Tissue median over every pixel of the tissue blocks (subsampled), so
    # dim tissue counts as much as bright tissue.
    step = max(1, int(np.sqrt(signals[0].size / _STATS_PIXELS_PER_PLANE)))
    rows = np.arange(0, tissue.shape[1] * block, step)
    cols = np.arange(0, tissue.shape[2] * block, step)
    flat_sampled = flat[np.ix_(rows, cols)]
    tissue_values, tissue_counts, total = [], 0, 0
    for signal, plane_tissue in zip(signals, tissue):
        mask = plane_tissue[np.ix_(rows // block, cols // block)]
        tissue_counts += int(mask.sum())
        total += mask.size
        tissue_values.append((signal[np.ix_(rows, cols)] / flat_sampled)[mask])
    tissue_fraction = tissue_counts / total
    if tissue_fraction < _MIN_TISSUE_FRACTION:
        raise ValueError(
            f"Only {100 * tissue_fraction:.1f}% of sampled pixels are in tissue blocks "
            f"({tissue_snr} block-median SEs above offset); cannot anchor the intensity profile."
        )
    tissue_median = float(np.median(np.concatenate(tissue_values)))
    alignment.update(
        {
            "gain": float(gain),
            "offset": float(offset),
            "noise_model_source": source,
            "tissue_snr": float(tissue_snr),
            "tissue_rule": "block median(x - offset) > max(tissue_snr * 1.2533 * "
            "sqrt(gain * median) / block_size, gain)",
            "tissue_block_size": int(block),
            "tissue_fraction": float(tissue_fraction),
            "tissue_median_signal": tissue_median,
            # signal / gain equals a photon count only for a calibrated linear
            # detector; for a fitted (empirical) model it is a relative scale.
            "tissue_median_over_gain": tissue_median / float(gain),
            "stats_z_indices": [int(z) for z in z_indices],
        }
    )
    alignment["codec_qa"] = _codec_qa(signals, flat, step, alignment)
    return alignment


def _codec_qa(signals: list[np.ndarray], flat: np.ndarray, step: int, alignment: dict) -> dict:
    """Codec clipping and round-trip error on the sampled planes (all pixels).

    Noise SD in aligned units follows the fitted model, floored at one gain
    unit so the offset region does not divide by zero.
    """
    scale = flat[::step, ::step] * np.float32(alignment["tissue_median_signal"])
    signal = np.stack([s[::step, ::step] for s in signals])
    aligned = signal / scale
    noise_sd = np.sqrt(np.float32(alignment["gain"]) * np.maximum(signal, alignment["gain"]))
    report = quantization_report(aligned, noise_sd / scale)
    report.pop("codec")
    return report


def to_aligned(plane: np.ndarray, alignment: dict, flat: np.ndarray | None = None) -> np.ndarray:
    """Raw XY plane -> aligned intensity (float32). Pass ``flat`` to reuse it."""
    if flat is None:
        flat = flat_field_image(alignment, plane.shape)
    signal = np.asarray(plane, dtype=np.float32) - np.float32(alignment["offset"])
    return signal / (flat * np.float32(alignment["tissue_median_signal"]))


def alignment_attrs(alignment: dict) -> dict:
    """Flat JSON-ready metadata: alignment fit, fixed code table and display."""
    flat = alignment["flat_field"]
    attrs = {
        "intensity_mode": "aligned",
        "aligned_definition": "r = (x - offset) / (flat_field(y, x) * tissue_median_signal)",
        "noise_model": "variance = gain * (x - offset)",
        "noise_gain": alignment["gain"],
        "noise_offset": alignment["offset"],
        "noise_model_source": alignment["noise_model_source"],
        "tissue_snr": alignment["tissue_snr"],
        "tissue_rule": alignment["tissue_rule"],
        "tissue_block_size": alignment["tissue_block_size"],
        "tissue_fraction": alignment["tissue_fraction"],
        "tissue_median_signal": alignment["tissue_median_signal"],
        "tissue_median_over_gain": alignment["tissue_median_over_gain"],
        "stats_z_indices": list(alignment["stats_z_indices"]),
        "flat_field_model": "none" if flat is None else flat["model"],
    }
    for key, value in alignment["codec_qa"].items():
        attrs[f"codec_qa_{key}"] = value
    if flat is not None:
        attrs["flat_field_coefficients"] = list(flat["coefficients"])
        attrs["flat_field_frame_shape_yx"] = list(flat["frame_shape_yx"])
    attrs.update(codec_attrs())
    attrs.update(display_attrs())
    return attrs

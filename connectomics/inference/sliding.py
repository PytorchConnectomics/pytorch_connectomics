"""
Sliding-window inference helpers for PyTorch Connectomics.

This module isolates ROI/overlap resolution and SlidingWindowInferer creation
so the logic can be reused by both the Lightning module and TTA routines.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from monai.data.utils import dense_patch_slices
from monai.inferers import SlidingWindowInferer
from monai.inferers.utils import _get_scan_interval, compute_importance_map

logger = logging.getLogger(__name__)


_DISTANCE_TRANSFORM_BLEND_MODES = {
    "distance",
    "distance_transform",
    "distance-transform",
    "distance_transform_cdt",
    "banis",
    "banis_distance",
}


def _cfg_value(obj, key: str, default=None):
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _normalize_blending_mode(mode: str) -> str:
    return str(mode).strip().lower()


def is_distance_transform_blending(mode: str) -> bool:
    """Return True for BANIS-style distance-transform window blending."""
    return _normalize_blending_mode(mode) in _DISTANCE_TRANSFORM_BLEND_MODES


def build_sliding_importance_map(
    roi_size: Sequence[int],
    *,
    mode: str,
    sigma_scale: Union[float, Sequence[float]],
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build per-window blending weights.

    ``distance_transform`` matches ``lib/banis``:
    ``distance_transform_cdt(np.pad(np.ones(roi), 1))[1:-1]``. For a solid
    rectangular window this is exactly ``min(distance_to_each_face) + 1``.
    """
    normalized_mode = _normalize_blending_mode(mode)
    if normalized_mode not in _DISTANCE_TRANSFORM_BLEND_MODES:
        return compute_importance_map(
            tuple(int(v) for v in roi_size),
            mode=normalized_mode,
            sigma_scale=sigma_scale,
            device=device,
            dtype=dtype,
        )

    spatial_shape = tuple(int(v) for v in roi_size)
    if not spatial_shape or any(v <= 0 for v in spatial_shape):
        raise ValueError(f"roi_size must contain positive values, got {roi_size}.")

    importance_map = None
    for axis, size in enumerate(spatial_shape):
        coord = torch.arange(size, device=device, dtype=dtype)
        dist = torch.minimum(coord + 1, torch.as_tensor(size, device=device, dtype=dtype) - coord)
        view_shape = [1] * len(spatial_shape)
        view_shape[axis] = size
        dist = dist.reshape(view_shape)
        importance_map = dist if importance_map is None else torch.minimum(importance_map, dist)

    return importance_map


def build_sliding_accumulator_weight_maps(
    roi_size: Sequence[int],
    *,
    mode: str,
    sigma_scale: Union[float, Sequence[float]],
    device: torch.device | str,
    value_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build value and weight maps for weighted sliding-window accumulation.

    The value map follows ``inference.model.output_dtype`` to control the large
    channel accumulator. The weight map is always fp32 because it is single
    channel and small, while fp16 can underflow Gaussian tail weights.
    """
    value_map = build_sliding_importance_map(
        roi_size,
        mode=mode,
        sigma_scale=sigma_scale,
        device=device,
        dtype=value_dtype,
    )
    if value_dtype == torch.float32:
        weight_map = value_map
    else:
        weight_map = build_sliding_importance_map(
            roi_size,
            mode=mode,
            sigma_scale=sigma_scale,
            device=device,
            dtype=torch.float32,
        )
    return value_map, weight_map


def normalize_weighted_accumulator(
    value_accumulator: torch.Tensor, weight_accumulator: torch.Tensor
) -> torch.Tensor:
    """Divide value by weight in-place while preserving value dtype."""
    clamp_value = 1.0e-6
    if value_accumulator.dtype == torch.float16:
        clamp_value = max(clamp_value, float(torch.finfo(torch.float16).tiny))
    divisor = torch.clamp_min(weight_accumulator, clamp_value)
    if divisor.dtype != value_accumulator.dtype:
        divisor = divisor.to(value_accumulator.dtype)
    value_accumulator /= divisor
    return value_accumulator


def apply_border_mask(importance_map: torch.Tensor, border_mask: Sequence[int]) -> torch.Tensor:
    """Zero outer ``k`` voxels of each spatial axis in ``importance_map``.

    ``border_mask`` length must equal the number of spatial dimensions; the
    function operates on the trailing spatial dims so leading channel/batch
    singletons are left alone. A no-op when all entries are <= 0.
    """
    if not border_mask or all(int(b) <= 0 for b in border_mask):
        return importance_map
    spatial_dims = len(border_mask)
    spatial_shape = importance_map.shape[-spatial_dims:]
    for axis, k in enumerate(border_mask):
        k = int(k)
        if k <= 0:
            continue
        size = int(spatial_shape[axis])
        if 2 * k >= size:
            raise ValueError(
                f"inference.sliding_window.border_mask[{axis}]={k} is too large "
                f"for window size {size} on that axis."
            )
        idx = [slice(None)] * importance_map.ndim
        trailing = -(spatial_dims - axis)
        idx[trailing] = slice(0, k)
        importance_map[tuple(idx)] = 0
        idx[trailing] = slice(size - k, size)
        importance_map[tuple(idx)] = 0
    return importance_map


_MODEL_OUTPUT_DTYPE_ALIASES: dict[str, "torch.dtype"] = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def resolve_model_output_dtype(cfg) -> torch.dtype:
    """Return the configured inference model-output dtype, defaulting to float32."""
    inference_cfg = _cfg_value(cfg, "inference", None)
    inference_model_cfg = _cfg_value(inference_cfg, "model", None)
    raw = _cfg_value(inference_model_cfg, "output_dtype", None)
    if raw is None:
        return torch.float32
    name = str(raw).strip().lower().removeprefix("torch.")
    if name in _MODEL_OUTPUT_DTYPE_ALIASES:
        return _MODEL_OUTPUT_DTYPE_ALIASES[name]
    raise ValueError(
        "inference.model.output_dtype must be one of "
        f"{sorted(_MODEL_OUTPUT_DTYPE_ALIASES)}, got {raw!r}."
    )


def resolve_border_mask(cfg, spatial_dims: int) -> list[int]:
    """Return the configured per-axis border mask, normalized to length ``spatial_dims``."""
    sliding_cfg = getattr(getattr(cfg, "inference", None), "sliding_window", None)
    raw = getattr(sliding_cfg, "border_mask", None) if sliding_cfg else None
    if not raw:
        return []
    values = [int(v) for v in raw]
    if len(values) == 1:
        values = values * spatial_dims
    if len(values) != spatial_dims:
        raise ValueError(
            f"inference.sliding_window.border_mask must have length 1 or {spatial_dims}, "
            f"got {len(values)}."
        )
    return values


def is_2d_inference_mode(cfg) -> bool:
    """Return True when data config indicates 2D mode."""
    train_cfg = getattr(getattr(cfg, "data", None), "train", None)
    val_cfg = getattr(getattr(cfg, "data", None), "val", None)
    return bool(getattr(train_cfg, "do_2d", False) or getattr(val_cfg, "do_2d", False))


def resolve_inferer_roi_size(cfg) -> Optional[Tuple[int, ...]]:
    """Determine the ROI size for sliding-window inference."""
    if hasattr(cfg, "inference") and hasattr(cfg.inference, "sliding_window"):
        window_size = getattr(cfg.inference.sliding_window, "window_size", None)
        if window_size:
            return tuple(int(v) for v in window_size)

    if hasattr(cfg, "model") and hasattr(cfg.model, "output_size"):
        output_size = getattr(cfg.model, "output_size", None)
        if output_size:
            roi_size = tuple(int(v) for v in output_size)
            if is_2d_inference_mode(cfg) and len(roi_size) == 2:
                roi_size = (1,) + roi_size
            return roi_size

    if hasattr(cfg, "data") and hasattr(cfg.data, "data_transform"):
        patch_size = getattr(cfg.data.data_transform, "patch_size", None)
        if patch_size:
            roi_size = tuple(int(v) for v in patch_size)
            if is_2d_inference_mode(cfg) and len(roi_size) == 2:
                roi_size = (1,) + roi_size
            return roi_size

    return None


def resolve_inferer_overlap(cfg, roi_size: Tuple[int, ...]) -> Union[float, Tuple[float, ...]]:
    """Resolve overlap parameter using inference config."""
    if not hasattr(cfg, "inference") or not hasattr(cfg.inference, "sliding_window"):
        return 0.5

    overlap = getattr(cfg.inference.sliding_window, "overlap", None)
    if overlap is not None:
        if isinstance(overlap, (list, tuple)):
            return tuple(float(max(0.0, min(o, 0.99))) for o in overlap)
        return float(max(0.0, min(overlap, 0.99)))

    return 0.5


def _resolve_sliding_window_runtime(cfg, roi_size: Tuple[int, ...]) -> dict:
    overlap = resolve_inferer_overlap(cfg, roi_size)
    data_cfg = getattr(cfg, "data", None)
    data_loader_cfg = getattr(data_cfg, "dataloader", None) if data_cfg else None
    data_batch_value = getattr(data_loader_cfg, "batch_size", 1) if data_loader_cfg else 1
    sliding_cfg = getattr(getattr(cfg, "inference", None), "sliding_window", None)
    config_sw_batch_size = getattr(sliding_cfg, "sw_batch_size", None) if sliding_cfg else None
    sw_batch_size = max(
        1, int(config_sw_batch_size if config_sw_batch_size is not None else data_batch_value)
    )
    mode = _normalize_blending_mode(
        getattr(sliding_cfg, "blending", "gaussian") if sliding_cfg else "gaussian"
    )
    sigma_scale = float(getattr(sliding_cfg, "sigma_scale", 0.125)) if sliding_cfg else 0.125
    padding_mode = getattr(sliding_cfg, "padding_mode", "constant") if sliding_cfg else "constant"
    cval = float(getattr(sliding_cfg, "cval", 0.0)) if sliding_cfg else 0.0
    keep_input_on_cpu = (
        bool(getattr(sliding_cfg, "keep_input_on_cpu", False)) if sliding_cfg else False
    )
    sw_device = getattr(sliding_cfg, "sw_device", None) if sliding_cfg else None
    output_device = getattr(sliding_cfg, "output_device", None) if sliding_cfg else None

    if isinstance(sw_device, str) and sw_device.lower() in {"", "none", "null"}:
        sw_device = None
    if isinstance(output_device, str) and output_device.lower() in {"", "none", "null"}:
        output_device = None

    if keep_input_on_cpu:
        if sw_device is None and torch.cuda.is_available():
            sw_device = "cuda"
        if output_device is None:
            output_device = "cpu"
        if sw_device is None:
            logger.warning(
                "inference.sliding_window.keep_input_on_cpu=True but no sw_device was set "
                "and CUDA is unavailable. Sliding-window inference will run on CPU."
            )

    return {
        "overlap": overlap,
        "sw_batch_size": sw_batch_size,
        "mode": mode,
        "sigma_scale": sigma_scale,
        "padding_mode": padding_mode,
        "cval": cval,
        "keep_input_on_cpu": keep_input_on_cpu,
        "sw_device": sw_device,
        "output_device": output_device,
    }


def _extract_padded_patch_batch(
    tensor: torch.Tensor,
    patch_slices: Sequence[tuple[slice, ...]],
    *,
    roi_size: tuple[int, ...],
    padding_mode: str,
    cval: float,
) -> tuple[torch.Tensor, list[tuple[int, ...]]]:
    if tensor.shape[0] != 1:
        raise ValueError(
            "Patch-first sliding-window TTA currently expects singleton batches. "
            f"Got batch size {tensor.shape[0]}."
        )

    spatial_dims = len(roi_size)
    image_size = tuple(int(v) for v in tensor.shape[-spatial_dims:])
    patches = []
    locations: list[tuple[int, ...]] = []

    for patch_slice in patch_slices:
        location = tuple(int(s.start) for s in patch_slice)
        end = tuple(location[axis] + int(roi_size[axis]) for axis in range(spatial_dims))

        inner_start = tuple(max(0, location[axis]) for axis in range(spatial_dims))
        inner_end = tuple(min(image_size[axis], end[axis]) for axis in range(spatial_dims))
        inner_slices = [
            slice(int(inner_start[axis]), int(inner_end[axis])) for axis in range(spatial_dims)
        ]
        inner = tensor[(slice(None), slice(None), *inner_slices)]

        pad_pairs = []
        for axis in range(spatial_dims):
            pad_before = max(0, -location[axis])
            pad_after = max(0, end[axis] - image_size[axis])
            pad_pairs.append((pad_before, pad_after))

        if any(before or after for before, after in pad_pairs):
            pad = []
            for before, after in reversed(pad_pairs):
                pad.extend([before, after])
            if padding_mode == "constant":
                inner = F.pad(inner, tuple(pad), mode=padding_mode, value=cval)
            else:
                inner = F.pad(inner, tuple(pad), mode=padding_mode)

        patches.append(inner)
        locations.append(location)

    return torch.cat(patches, dim=0), locations


def build_sliding_inferer(cfg) -> Optional[SlidingWindowInferer]:
    """
    Build a MONAI SlidingWindowInferer if configuration permits.

    BANIS-style boundary handling (`snap_to_edge`, `target_context`) is honored
    by the lazy sliding-window path (`connectomics/inference/lazy.py`) when
    `inference.sliding_window.lazy_load=true`. The eager path here uses MONAI's
    stock inferer; for BANIS-flavored boundary context in eager mode, simply
    set `window_size` larger than the training patch size.
    """
    roi_size = resolve_inferer_roi_size(cfg)
    if roi_size is None:
        logger.warning(
            "Sliding-window inference disabled: unable to determine ROI size. "
            "Set inference.window_size or model.output_size in the config."
        )
        return None

    # The eager MONAI ``SlidingWindowInferer`` is unused when lazy sliding
    # window is configured; skip building it so options that only the lazy
    # path supports (distance_transform blending, border_mask, snap_to_edge,
    # target_context) don't trigger spurious eager-path errors at init.
    sliding_cfg = getattr(getattr(cfg, "inference", None), "sliding_window", None)
    if bool(getattr(sliding_cfg, "lazy_load", False)):
        return None

    runtime = _resolve_sliding_window_runtime(cfg, roi_size)
    if is_distance_transform_blending(runtime["mode"]):
        raise ValueError(
            "inference.window.blending=distance_transform requires lazy_load=true; "
            "MONAI's eager SlidingWindowInferer only supports constant/gaussian blending."
        )
    if resolve_border_mask(cfg, len(roi_size)):
        logger.warning(
            "inference.sliding_window.border_mask is set but the eager MONAI "
            "SlidingWindowInferer ignores it. Enable lazy_load=true to apply "
            "border masking."
        )
    inferer = SlidingWindowInferer(
        roi_size=roi_size,
        sw_batch_size=runtime["sw_batch_size"],
        overlap=runtime["overlap"],
        mode=runtime["mode"],
        sigma_scale=runtime["sigma_scale"],
        padding_mode=runtime["padding_mode"],
        sw_device=runtime["sw_device"],
        device=runtime["output_device"],
        progress=True,
    )

    logger.debug(
        "Sliding-window inference configured: "
        f"roi_size={roi_size}, overlap={runtime['overlap']}, sw_batch={runtime['sw_batch_size']}, "
        f"mode={runtime['mode']}, sigma_scale={runtime['sigma_scale']}, "
        f"padding={runtime['padding_mode']}, keep_input_on_cpu={runtime['keep_input_on_cpu']}, "
        f"sw_device={runtime['sw_device']}, output_device={runtime['output_device']}"
    )

    return inferer

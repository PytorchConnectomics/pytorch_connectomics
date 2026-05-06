"""
Sliding-window inference helpers for PyTorch Connectomics.

This module isolates ROI/overlap resolution and SlidingWindowInferer creation
so the logic can be reused by both the Lightning module and TTA routines.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from monai.data.utils import dense_patch_slices
from monai.inferers import SlidingWindowInferer
from monai.inferers.utils import _get_scan_interval, compute_importance_map

logger = logging.getLogger(__name__)


def apply_border_mask(
    importance_map: torch.Tensor, border_mask: Sequence[int]
) -> torch.Tensor:
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


_ACCUMULATOR_DTYPE_ALIASES: dict[str, "torch.dtype"] = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def resolve_accumulator_dtype(cfg) -> torch.dtype:
    """Return the configured value-accumulator dtype, defaulting to float32."""
    sliding_cfg = getattr(getattr(cfg, "inference", None), "sliding_window", None)
    raw = getattr(sliding_cfg, "accumulator_dtype", None) if sliding_cfg else None
    if raw is None:
        return torch.float32
    name = str(raw).strip().lower().removeprefix("torch.")
    if name in _ACCUMULATOR_DTYPE_ALIASES:
        return _ACCUMULATOR_DTYPE_ALIASES[name]
    raise ValueError(
        "inference.sliding_window.accumulator_dtype must be one of "
        f"{sorted(_ACCUMULATOR_DTYPE_ALIASES)}, got {raw!r}."
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
    mode = getattr(sliding_cfg, "blending", "gaussian") if sliding_cfg else "gaussian"
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

    runtime = _resolve_sliding_window_runtime(cfg, roi_size)
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

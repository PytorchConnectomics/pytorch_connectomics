"""Prediction-space crop helpers for test-time inference."""

from __future__ import annotations

import gc
import logging
from typing import Any, Optional

import numpy as np
import torch

from ...data.processing.affinity import (
    compute_affinity_crop_pad,
    crop_spatial_by_pad,
    resolve_affinity_channel_groups_from_cfg,
    resolve_affinity_mode_from_cfg,
)
from ...data.processing.misc import get_padsize
from ...utils.channel_slices import resolve_channel_indices, resolve_channel_range

logger = logging.getLogger(__name__)


def _resolve_inference_crop_pad(module) -> Optional[tuple[tuple[int, int], ...]]:
    """Return configured symmetric or asymmetric prediction crop, if any."""
    inference_cfg = None
    if hasattr(module, "_get_runtime_inference_config"):
        inference_cfg = module._get_runtime_inference_config()
    elif hasattr(module, "cfg") and hasattr(module.cfg, "inference"):
        inference_cfg = module.cfg.inference
    if inference_cfg is None:
        return None

    crop_pad = getattr(inference_cfg, "crop_pad", None)
    if crop_pad is None:
        return None

    crop_pad_values = tuple(int(v) for v in crop_pad)
    if not crop_pad_values or not any(crop_pad_values):
        return None
    if any(v < 0 for v in crop_pad_values):
        raise ValueError(f"inference.crop_pad must be non-negative, got {crop_pad_values}")

    if len(crop_pad_values) in (2, 4):
        spatial_rank = 2
    elif len(crop_pad_values) in (3, 6):
        spatial_rank = 3
    else:
        raise ValueError(
            "inference.crop_pad must have length 2/3 for symmetric cropping "
            f"or 4/6 for asymmetric cropping, got {crop_pad_values}."
        )

    return tuple(get_padsize(list(crop_pad_values), ndim=spatial_rank))


def _crop_spatial_border(
    data: np.ndarray | torch.Tensor,
    crop_pad: tuple[tuple[int, int], ...],
    *,
    item_name: str,
) -> np.ndarray | torch.Tensor:
    """Crop border padding from the last spatial dimensions."""
    return crop_spatial_by_pad(data, crop_pad, item_name=item_name)


def _apply_prediction_crop_pad_if_needed(
    module,
    data: np.ndarray | torch.Tensor,
    reference_image_shape: tuple[int, ...],
    *,
    item_name: str,
) -> np.ndarray | torch.Tensor:
    """Crop prediction-like tensors back to the pre-context-pad spatial shape."""
    crop_pad = _resolve_inference_crop_pad(module)
    if crop_pad is None:
        return data

    if len(reference_image_shape) < len(crop_pad):
        raise ValueError(
            "reference_image_shape rank must be >= inference.crop_pad rank. "
            f"Got reference_image_shape={reference_image_shape}, crop_pad={crop_pad}"
        )

    spatial_slice = slice(-len(crop_pad), None)
    padded_spatial_shape = tuple(int(v) for v in reference_image_shape[spatial_slice])
    expected_cropped_shape = tuple(
        padded_spatial_shape[i] - crop_pad[i][0] - crop_pad[i][1] for i in range(len(crop_pad))
    )
    if any(size <= 0 for size in expected_cropped_shape):
        raise ValueError(
            "inference.crop_pad is too large for the padded input shape. "
            f"crop_pad={crop_pad}, padded_shape={padded_spatial_shape}"
        )

    data_spatial_shape = tuple(int(v) for v in data.shape[spatial_slice])
    if data_spatial_shape == expected_cropped_shape:
        return data
    if data_spatial_shape != padded_spatial_shape:
        raise ValueError(
            "Cannot apply inference.crop_pad to "
            f"{item_name}: spatial shape {data_spatial_shape} matches neither "
            f"padded input {padded_spatial_shape} nor cropped shape "
            f"{expected_cropped_shape}."
        )

    cropped = _crop_spatial_border(data, crop_pad, item_name=item_name)
    logger.info(f"Cropped {item_name}: {tuple(data.shape)} -> {tuple(cropped.shape)}")
    return cropped


def _resolve_reference_spatial_shape_after_crop_pad(
    module,
    reference_image_shape: tuple[int, ...],
) -> tuple[int, ...]:
    crop_pad = _resolve_inference_crop_pad(module)
    spatial_rank = 3 if len(reference_image_shape) >= 3 else len(reference_image_shape)
    reference_spatial_shape = tuple(int(v) for v in reference_image_shape[-spatial_rank:])
    if crop_pad is None:
        return reference_spatial_shape

    crop_rank = len(crop_pad)
    unchanged_prefix = reference_spatial_shape[:-crop_rank]
    cropped_suffix = tuple(
        reference_spatial_shape[len(unchanged_prefix) + axis]
        - crop_pad[axis][0]
        - crop_pad[axis][1]
        for axis in range(crop_rank)
    )
    return unchanged_prefix + cropped_suffix


def _mapping_get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    if hasattr(obj, "get"):
        try:
            return obj.get(key, default)
        except TypeError:
            pass
    return getattr(obj, key, default)


def _resolve_output_head_target_slice(module, output_head: Optional[str]) -> Any:
    if not output_head:
        return None
    heads = getattr(getattr(module.cfg, "model", None), "heads", None)
    head_cfg = _mapping_get(heads, output_head, None)
    return _mapping_get(head_cfg, "target_slice", None)


def _resolve_affinity_offsets_for_inference_output(
    module,
    *,
    output_head: Optional[str],
) -> list[tuple[int, int, int]]:
    cfg = getattr(module, "cfg", None)
    if cfg is None:
        return []

    affinity_groups = resolve_affinity_channel_groups_from_cfg(cfg)
    if not affinity_groups:
        return []

    label_channels = max(end for (start, end), _offsets in affinity_groups)
    channel_offsets: list[Optional[tuple[int, int, int]]] = [None] * label_channels
    for (start, end), offsets in affinity_groups:
        for channel, offset in zip(range(start, end), offsets):
            channel_offsets[channel] = offset

    target_slice = _resolve_output_head_target_slice(module, output_head)
    if target_slice is not None:
        start_idx, end_idx = resolve_channel_range(
            target_slice,
            num_channels=label_channels,
            context=f"target_slice for output head {output_head!r}",
        )
        channel_offsets = channel_offsets[start_idx:end_idx]

    select_channel = getattr(getattr(module.cfg, "inference", None), "select_channel", None)
    if select_channel is not None:
        selected_indices = resolve_channel_indices(
            select_channel,
            num_channels=len(channel_offsets),
            context="inference.select_channel",
        )
        channel_offsets = [channel_offsets[idx] for idx in selected_indices]

    return [offset for offset in channel_offsets if offset is not None]


def _resolve_affinity_inference_crop(
    module,
    *,
    output_head: Optional[str] = None,
) -> Optional[tuple[tuple[int, int], ...]]:
    cfg = getattr(module, "cfg", None)
    if cfg is None:
        return None
    affinity_mode = resolve_affinity_mode_from_cfg(cfg)
    if affinity_mode is None:
        return None
    if affinity_mode != "deepem":
        return None

    offsets = _resolve_affinity_offsets_for_inference_output(module, output_head=output_head)
    if not offsets:
        return None

    crop_pad = compute_affinity_crop_pad(offsets, affinity_mode=affinity_mode)
    if not crop_pad or not any(before or after for before, after in crop_pad):
        return None
    return crop_pad


def _apply_affinity_inference_crop_if_needed(
    module,
    data: np.ndarray | torch.Tensor,
    *,
    reference_spatial_shape: tuple[int, ...],
    item_name: str,
    output_head: Optional[str] = None,
) -> np.ndarray | torch.Tensor:
    crop_pad = _resolve_affinity_inference_crop(module, output_head=output_head)
    if crop_pad is None:
        return data

    if len(reference_spatial_shape) != len(crop_pad):
        raise ValueError(
            f"Affinity crop rank mismatch for {item_name}: "
            f"reference_spatial_shape={reference_spatial_shape}, crop_pad={crop_pad}"
        )

    expected_cropped_shape = tuple(
        int(reference_spatial_shape[axis]) - crop_pad[axis][0] - crop_pad[axis][1]
        for axis in range(len(crop_pad))
    )
    if any(size <= 0 for size in expected_cropped_shape):
        raise ValueError(
            f"Affinity crop {crop_pad} is too large for {item_name} shape {reference_spatial_shape}"
        )

    spatial_slice = slice(-len(crop_pad), None)
    data_spatial_shape = tuple(int(v) for v in data.shape[spatial_slice])
    if data_spatial_shape == expected_cropped_shape:
        return data
    if data_spatial_shape != reference_spatial_shape:
        return data

    cropped = crop_spatial_by_pad(data, crop_pad, item_name=item_name)
    logger.info(f"Affinity-cropped {item_name}: {tuple(data.shape)} -> {tuple(cropped.shape)}")
    return cropped


def _apply_predecode_prediction_crops(
    module,
    data: np.ndarray | torch.Tensor,
    *,
    reference_image_shape: tuple[int, ...],
    item_name: str,
    output_head: Optional[str] = None,
) -> tuple[np.ndarray | torch.Tensor, tuple[int, ...]]:
    """Apply prediction-space crops before decoding affinities into instances."""
    original_shape = tuple(data.shape)
    data = _apply_prediction_crop_pad_if_needed(
        module,
        data,
        reference_image_shape,
        item_name=item_name,
    )
    reference_spatial_shape = _resolve_reference_spatial_shape_after_crop_pad(
        module, reference_image_shape
    )
    data = _apply_affinity_inference_crop_if_needed(
        module,
        data,
        reference_spatial_shape=reference_spatial_shape,
        item_name=item_name,
        output_head=output_head,
    )
    if isinstance(data, np.ndarray) and tuple(data.shape) != original_shape:
        data = np.array(data, copy=True, order="C")
        logger.info(
            "Compacted cropped %s storage to final shape %s (%.1f MiB)",
            item_name,
            tuple(data.shape),
            data.nbytes / (1024**2),
        )
        gc.collect()
    return data, reference_spatial_shape


__all__ = [
    "_apply_affinity_inference_crop_if_needed",
    "_apply_predecode_prediction_crops",
    "_apply_prediction_crop_pad_if_needed",
    "_resolve_inference_crop_pad",
    "_resolve_reference_spatial_shape_after_crop_pad",
]

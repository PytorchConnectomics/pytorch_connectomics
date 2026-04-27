"""Affinity target generation and valid-region helpers."""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

import numpy as np
import torch

from ...utils.channel_slices import resolve_channel_range

__all__ = [
    "AFFINITY_MODES",
    "compute_affinity_crop_pad",
    "compute_affinity_valid_mask",
    "crop_spatial_by_offsets",
    "crop_spatial_by_pad",
    "normalize_affinity_mode",
    "parse_affinity_offsets",
    "resolve_affinity_channel_groups_from_cfg",
    "resolve_affinity_mode_from_cfg",
    "resolve_affinity_offsets_from_kwargs",
    "resolve_affinity_offsets_for_channel_slice",
    "seg_to_affinity",
]

AFFINITY_MODES = ("deepem", "banis")


def normalize_affinity_mode(mode: Any) -> str:
    """Validate and normalize the configured affinity target convention."""
    if mode is None:
        raise ValueError("Affinity targets require kwargs.affinity_mode: 'deepem' or 'banis'.")
    normalized = str(mode).strip().lower()
    if normalized not in AFFINITY_MODES:
        allowed = ", ".join(AFFINITY_MODES)
        raise ValueError(f"Unsupported affinity_mode {mode!r}. Expected one of: {allowed}.")
    return normalized


def _mapping_get(task: Any, key: str, default: Any = None) -> Any:
    if isinstance(task, dict):
        return task.get(key, default)
    if hasattr(task, "get"):
        try:
            return task.get(key, default)
        except TypeError:
            pass
    return getattr(task, key, default)


def _task_name(task: Any) -> Optional[str]:
    if isinstance(task, str):
        return task
    name = _mapping_get(task, "name", None)
    if name is not None:
        return name
    task_name = _mapping_get(task, "task", None)
    if task_name is not None:
        return task_name
    return _mapping_get(task, "type", None)


def _task_kwargs(task: Any) -> dict[str, Any]:
    raw = _mapping_get(task, "kwargs", None)
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    if hasattr(raw, "items"):
        return {key: value for key, value in raw.items()}
    return dict(raw)


def _task_entries(targets: Any) -> list[Any]:
    if targets is None:
        return []
    if isinstance(targets, str):
        return [targets]
    try:
        return list(targets)
    except TypeError:
        return []


def parse_affinity_offsets(offsets: Sequence[Any]) -> list[tuple[int, int, int]]:
    """Parse affinity offsets from config values."""
    parsed: list[tuple[int, int, int]] = []
    for offset in offsets:
        if isinstance(offset, str):
            parts = offset.split("-")
            if len(parts) != 3:
                raise ValueError(f"Invalid affinity offset {offset!r}. Expected 'z-y-x' format.")
            parsed.append((int(parts[0]), int(parts[1]), int(parts[2])))
            continue
        if isinstance(offset, (list, tuple)) and len(offset) == 3:
            parsed.append((int(offset[0]), int(offset[1]), int(offset[2])))
            continue
        raise ValueError(
            f"Unsupported affinity offset {offset!r}. Expected 'z-y-x' string or length-3 sequence."
        )
    return parsed


def resolve_affinity_offsets_from_kwargs(kwargs: dict[str, Any]) -> list[tuple[int, int, int]]:
    """Resolve configured affinity offsets, including the 6-channel long-range form."""
    long_range = kwargs.get("long_range", None)
    if long_range is not None:
        long_range = int(long_range)
        return [
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
            (long_range, 0, 0),
            (0, long_range, 0),
            (0, 0, long_range),
        ]

    offsets = kwargs.get("offsets", None)
    if offsets is None or len(offsets) == 0:
        offsets = ["1-0-0", "0-1-0", "0-0-1"]
    return parse_affinity_offsets(offsets)


def resolve_affinity_mode_from_cfg(cfg: Any) -> Optional[str]:
    """Return the configured affinity mode, or ``None`` when no affinity target exists."""
    data_cfg = getattr(cfg, "data", None)
    label_cfg = getattr(data_cfg, "label_transform", None) if data_cfg is not None else None
    if label_cfg is None:
        return None

    modes: list[str] = []
    for task in _task_entries(getattr(label_cfg, "targets", None)):
        if _task_name(task) != "affinity":
            continue
        kwargs = _task_kwargs(task)
        modes.append(normalize_affinity_mode(kwargs.get("affinity_mode")))

    if not modes:
        return None
    unique_modes = sorted(set(modes))
    if len(unique_modes) != 1:
        raise ValueError(
            f"Mixed affinity_mode values are not supported in one label stack: {unique_modes}"
        )
    return unique_modes[0]


def resolve_affinity_channel_groups_from_cfg(
    cfg: Any,
) -> list[tuple[tuple[int, int], list[tuple[int, int, int]]]]:
    """Resolve stacked label channel ranges for configured affinity tasks."""
    data_cfg = getattr(cfg, "data", None)
    label_cfg = getattr(data_cfg, "label_transform", None) if data_cfg is not None else None
    if label_cfg is None:
        return []

    targets = getattr(label_cfg, "targets", None)
    stack_outputs = bool(getattr(label_cfg, "stack_outputs", True))

    def _task_channels(name: Optional[str], kwargs: dict[str, Any]) -> int:
        if name == "affinity":
            return len(resolve_affinity_offsets_from_kwargs(kwargs))
        if name == "polarity":
            return 1 if bool(kwargs.get("exclusive", False)) else 3
        return 1

    groups: list[tuple[tuple[int, int], list[tuple[int, int, int]]]] = []
    if targets is None or not stack_outputs:
        return groups

    task_entries = _task_entries(targets)
    if not task_entries:
        return groups

    channel_start = 0
    for task in task_entries:
        name = _task_name(task)
        kwargs = _task_kwargs(task)
        num_channels = _task_channels(name, kwargs)
        if name == "affinity":
            parsed_offsets = resolve_affinity_offsets_from_kwargs(kwargs)
            groups.append(((channel_start, channel_start + num_channels), parsed_offsets))
        channel_start += num_channels

    return groups


def resolve_affinity_offsets_for_channel_slice(
    cfg: Any,
    *,
    num_channels: int,
    channel_slice: Optional[int | str],
) -> Optional[list[tuple[int, int, int]]]:
    """Return affinity offsets for a selected stacked-label channel slice, if applicable."""
    groups = resolve_affinity_channel_groups_from_cfg(cfg)
    if not groups:
        return None

    if channel_slice is None:
        if len(groups) == 1:
            (start, end), offsets = groups[0]
            if start == 0 and end == num_channels:
                return list(offsets)
        return None

    start_idx, end_idx = resolve_channel_range(
        channel_slice,
        num_channels=num_channels,
        context="affinity channel slice",
    )
    for (group_start, group_end), offsets in groups:
        if start_idx < group_start or end_idx > group_end:
            continue
        rel_start = start_idx - group_start
        rel_end = end_idx - group_start
        return list(offsets[rel_start:rel_end])
    return None


def compute_affinity_crop_pad(
    offsets: Sequence[tuple[int, int, int]],
    *,
    affinity_mode: str = "deepem",
) -> tuple[tuple[int, int], ...]:
    """Return asymmetric valid-region crop pads for the given offsets."""
    if not offsets:
        return tuple()

    mode = normalize_affinity_mode(affinity_mode)
    ndim = len(offsets[0])
    leading = [0] * ndim
    trailing = [0] * ndim
    for offset in offsets:
        if len(offset) != ndim:
            raise ValueError(f"Mixed affinity offset dimensions are not supported: {offsets!r}")
        for axis, value in enumerate(offset):
            value = int(value)
            if mode == "deepem":
                leading[axis] = max(leading[axis], max(value, 0))
                trailing[axis] = max(trailing[axis], max(-value, 0))
            else:
                leading[axis] = max(leading[axis], max(-value, 0))
                trailing[axis] = max(trailing[axis], max(value, 0))
    return tuple((leading[axis], trailing[axis]) for axis in range(ndim))


def crop_spatial_by_pad(
    data: np.ndarray | torch.Tensor,
    crop_pad: Sequence[tuple[int, int]],
    *,
    item_name: str = "data",
) -> np.ndarray | torch.Tensor:
    """Crop asymmetric borders from the last spatial axes."""
    if not crop_pad:
        return data
    if data.ndim < len(crop_pad):
        raise ValueError(
            f"Cannot crop {item_name}: rank {data.ndim} is smaller than crop rank {len(crop_pad)}"
        )

    slices = [slice(None)] * data.ndim
    spatial_shape = tuple(int(v) for v in data.shape[-len(crop_pad) :])
    for spatial_idx, (before, after) in enumerate(crop_pad):
        dim_size = spatial_shape[spatial_idx]
        if before < 0 or after < 0:
            raise ValueError(f"Crop pad must be non-negative for {item_name}, got {crop_pad}")
        if before + after >= dim_size:
            raise ValueError(
                f"Cannot crop {item_name}: crop pad {tuple(crop_pad)} "
                f"is too large for shape {tuple(data.shape)}"
            )
        axis = data.ndim - len(crop_pad) + spatial_idx
        end = dim_size - after if after > 0 else dim_size
        slices[axis] = slice(before, end)
    return data[tuple(slices)]


def crop_spatial_by_offsets(
    data: np.ndarray | torch.Tensor,
    offsets: Sequence[tuple[int, int, int]],
    *,
    affinity_mode: str = "deepem",
    item_name: str = "data",
) -> np.ndarray | torch.Tensor:
    """Crop to the common valid spatial region across affinity offsets."""
    crop_pad = compute_affinity_crop_pad(offsets, affinity_mode=affinity_mode)
    return crop_spatial_by_pad(data, crop_pad, item_name=item_name)


def _source_destination_slices(
    offset: Sequence[int],
) -> tuple[tuple[slice, ...], tuple[slice, ...]]:
    src_slice: list[slice] = []
    dst_slice: list[slice] = []
    for value in offset:
        value = int(value)
        if value > 0:
            src_slice.append(slice(None, -value))
            dst_slice.append(slice(value, None))
        elif value < 0:
            src_slice.append(slice(-value, None))
            dst_slice.append(slice(None, value))
        else:
            src_slice.append(slice(None))
            dst_slice.append(slice(None))
    return tuple(src_slice), tuple(dst_slice)


def _storage_slice_for_offset(offset: Sequence[int], affinity_mode: str) -> tuple[slice, ...]:
    src_slice, dst_slice = _source_destination_slices(offset)
    return dst_slice if normalize_affinity_mode(affinity_mode) == "deepem" else src_slice


def compute_affinity_valid_mask(
    offsets: Sequence[tuple[int, int, int]],
    spatial_shape: Sequence[int],
    *,
    affinity_mode: str = "deepem",
    device: torch.device | None = None,
) -> torch.Tensor:
    """Build per-channel valid mask for affinity offsets.

    For each channel *i* with offset ``(dz, dy, dx)``, the valid region is
    where both source and destination voxels exist.  ``deepem`` stores that
    edge at the destination voxel; ``banis`` stores it at the source voxel.
    Voxels outside the selected convention's valid region are set to 0.

    Args:
        offsets: Parsed offsets, each a 3-tuple ``(dz, dy, dx)``.
        spatial_shape: ``(D, H, W)`` of the prediction / target.
        affinity_mode: ``deepem`` for destination-index targets or ``banis``
            for source-index targets.
        device: Torch device for the returned tensor.

    Returns:
        Float tensor of shape ``(len(offsets), D, H, W)`` with 1 in valid
        positions and 0 elsewhere.
    """
    mode = normalize_affinity_mode(affinity_mode)
    num_channels = len(offsets)
    mask = torch.zeros(num_channels, *spatial_shape, device=device)

    for i, offset in enumerate(offsets):
        if all(int(value) == 0 for value in offset):
            mask[i] = 1.0
            continue

        mask[i][_storage_slice_for_offset(offset, mode)] = 1.0

    return mask


def seg_to_affinity(
    seg: np.ndarray,
    offsets: List[str] = None,
    long_range: int = None,
    affinity_mode: str = "deepem",
) -> np.ndarray:
    """
    Compute affinity maps from segmentation.

    ``deepem`` stores each edge at the destination voxel. ``banis`` stores
    each edge at the source voxel and follows ``lib/banis`` target semantics:
    edges outside the valid source/destination region, or touching ``seg == -1``
    unlabeled voxels, are encoded as ``-1`` so the loss can skip them.

    Args:
        seg: The segmentation to compute affinities from. Shape: (z, y, x).
             0 indicates background.
        offsets: List of offset strings in "z-y-x" format (e.g., ["0-0-1", "0-1-0", "1-0-0"]).
                 Each string defines one affinity channel.
        long_range: Offset for 6-channel output:
                    - Channel 0-2: Short-range (offset 1) for z, y, x
                    - Channel 3-5: Long-range (offset long_range) for z, y, x
        affinity_mode: ``deepem`` or ``banis``.

    Returns:
        The affinities. Shape: (num_channels, z, y, x).
    """
    mode = normalize_affinity_mode(affinity_mode)
    parsed_offsets = resolve_affinity_offsets_from_kwargs(
        {"offsets": offsets, "long_range": long_range}
    )
    num_channels = len(parsed_offsets)
    affinities = np.zeros((num_channels, *seg.shape), dtype=np.float32)
    banis_loss_mask = np.zeros_like(affinities, dtype=bool) if mode == "banis" else None
    labeled_mask = seg != -1

    for i, offset in enumerate(parsed_offsets):
        if all(value == 0 for value in offset):
            affinities[i] = (seg > 0).astype(np.float32)
            if banis_loss_mask is not None:
                banis_loss_mask[i] = labeled_mask
            continue

        src_slice, dst_slice = _source_destination_slices(offset)
        storage_slice = dst_slice if mode == "deepem" else src_slice
        affinities[i][storage_slice] = (
            (seg[src_slice] == seg[dst_slice]) & (seg[storage_slice] > 0)
        ).astype(np.float32)
        if banis_loss_mask is not None:
            banis_loss_mask[i][storage_slice] = labeled_mask[src_slice] & labeled_mask[dst_slice]

    if banis_loss_mask is not None:
        affinities[~banis_loss_mask] = -1.0

    return affinities

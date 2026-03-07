"""Affinity target generation and DeepEM-style valid-region helpers."""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

import numpy as np
import torch

from ...utils.channel_slices import resolve_channel_slice_bounds

__all__ = [
    "affinity_deepem_crop_enabled",
    "compute_affinity_crop_pad",
    "crop_spatial_by_offsets",
    "crop_spatial_by_pad",
    "parse_affinity_offsets",
    "resolve_affinity_channel_groups_from_cfg",
    "resolve_affinity_offsets_for_channel_slice",
    "seg_to_affinity",
]


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
                raise ValueError(
                    f"Invalid affinity offset {offset!r}. Expected 'z-y-x' format."
                )
            parsed.append((int(parts[0]), int(parts[1]), int(parts[2])))
            continue
        if isinstance(offset, (list, tuple)) and len(offset) == 3:
            parsed.append((int(offset[0]), int(offset[1]), int(offset[2])))
            continue
        raise ValueError(
            f"Unsupported affinity offset {offset!r}. Expected 'z-y-x' string or length-3 sequence."
        )
    return parsed


def affinity_deepem_crop_enabled(cfg: Any) -> bool:
    """Return whether DeepEM-style affinity valid-region cropping is enabled."""
    data_cfg = getattr(cfg, "data", None)
    label_cfg = getattr(data_cfg, "label_transform", None) if data_cfg is not None else None
    if label_cfg is None:
        return False

    explicit_values: list[bool] = []
    for task in _task_entries(getattr(label_cfg, "targets", None)):
        if _task_name(task) != "affinity":
            continue
        kwargs = _task_kwargs(task)
        if "deepem_crop" in kwargs:
            explicit_values.append(bool(kwargs["deepem_crop"]))
    return any(explicit_values)


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
            offsets = kwargs.get("offsets", None)
            if offsets is None or len(offsets) == 0:
                offsets = ["1-0-0", "0-1-0", "0-0-1"]
            return len(parse_affinity_offsets(offsets))
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
            offsets_cfg = kwargs.get("offsets", None)
            if offsets_cfg is None or len(offsets_cfg) == 0:
                offsets_cfg = ["1-0-0", "0-1-0", "0-0-1"]
            parsed_offsets = parse_affinity_offsets(offsets_cfg)
            groups.append(((channel_start, channel_start + num_channels), parsed_offsets))
        channel_start += num_channels

    return groups


def resolve_affinity_offsets_for_channel_slice(
    cfg: Any,
    *,
    num_channels: int,
    channel_slice: Optional[tuple[int, int]],
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

    start_idx, end_idx = resolve_channel_slice_bounds(
        channel_slice,
        num_channels=num_channels,
        context="affinity channel slice",
        negative_index_offset=0,
        end_minus_one_full_span=False,
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
) -> tuple[tuple[int, int], ...]:
    """Return asymmetric valid-region crop pads for the given offsets."""
    if not offsets:
        return tuple()

    ndim = len(offsets[0])
    leading = [0] * ndim
    trailing = [0] * ndim
    for offset in offsets:
        if len(offset) != ndim:
            raise ValueError(f"Mixed affinity offset dimensions are not supported: {offsets!r}")
        for axis, value in enumerate(offset):
            leading[axis] = max(leading[axis], max(int(value), 0))
            trailing[axis] = max(trailing[axis], max(-int(value), 0))
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
    spatial_shape = tuple(int(v) for v in data.shape[-len(crop_pad):])
    for spatial_idx, (before, after) in enumerate(crop_pad):
        dim_size = spatial_shape[spatial_idx]
        if before < 0 or after < 0:
            raise ValueError(f"Crop pad must be non-negative for {item_name}, got {crop_pad}")
        if before + after >= dim_size:
            raise ValueError(
                f"Cannot crop {item_name}: crop pad {tuple(crop_pad)} is too large for shape {tuple(data.shape)}"
            )
        axis = data.ndim - len(crop_pad) + spatial_idx
        end = dim_size - after if after > 0 else dim_size
        slices[axis] = slice(before, end)
    return data[tuple(slices)]


def crop_spatial_by_offsets(
    data: np.ndarray | torch.Tensor,
    offsets: Sequence[tuple[int, int, int]],
    *,
    item_name: str = "data",
) -> np.ndarray | torch.Tensor:
    """Crop to the common valid spatial region across affinity offsets."""
    crop_pad = compute_affinity_crop_pad(offsets)
    return crop_spatial_by_pad(data, crop_pad, item_name=item_name)


def seg_to_affinity(
    seg: np.ndarray,
    offsets: List[str] = None,
    long_range: int = None,
) -> np.ndarray:
    """
    Compute affinity maps from segmentation.

    Supports two modes:
    1. DeepEM/SNEMI style: Provide `offsets` as list of strings (e.g., ["0-0-1", "0-1-0", "1-0-0"])
    2. BANIS style: Provide `long_range` as int for 6-channel output (3 short + 3 long range)

    Args:
        seg: The segmentation to compute affinities from. Shape: (z, y, x).
             0 indicates background.
        offsets: List of offset strings in "z-y-x" format (e.g., ["0-0-1", "0-1-0", "1-0-0"]).
                 Each string defines one affinity channel.
        long_range: BANIS-style: offset for long-range affinities. Produces 6 channels:
                    - Channel 0-2: Short-range (offset 1) for z, y, x
                    - Channel 3-5: Long-range (offset long_range) for z, y, x

    Returns:
        The affinities. Shape: (num_channels, z, y, x).
    """
    if long_range is not None:
        affinities = np.zeros((6, *seg.shape), dtype=np.float32)

        affinities[0, :-1] = (seg[:-1] == seg[1:]) & (seg[1:] > 0)
        affinities[1, :, :-1] = (seg[:, :-1] == seg[:, 1:]) & (seg[:, 1:] > 0)
        affinities[2, :, :, :-1] = (seg[:, :, :-1] == seg[:, :, 1:]) & (seg[:, :, 1:] > 0)

        affinities[3, :-long_range] = (seg[:-long_range] == seg[long_range:]) & (
            seg[long_range:] > 0
        )
        affinities[4, :, :-long_range] = (seg[:, :-long_range] == seg[:, long_range:]) & (
            seg[:, long_range:] > 0
        )
        affinities[5, :, :, :-long_range] = (seg[:, :, :-long_range] == seg[:, :, long_range:]) & (
            seg[:, :, long_range:] > 0
        )

        return affinities

    if offsets is None:
        offsets = ["1-0-0", "0-1-0", "0-0-1"]

    parsed_offsets = []
    for offset_str in offsets:
        parts = offset_str.split("-")
        if len(parts) == 3:
            parsed_offsets.append([int(parts[0]), int(parts[1]), int(parts[2])])
        else:
            raise ValueError(f"Invalid offset format: {offset_str}. Expected 'z-y-x' format.")

    num_channels = len(parsed_offsets)
    affinities = np.zeros((num_channels, *seg.shape), dtype=np.float32)

    for i, (dz, dy, dx) in enumerate(parsed_offsets):
        if dz == 0 and dy == 0 and dx == 0:
            affinities[i] = (seg > 0).astype(np.float32)
            continue

        if dz > 0:
            z_src = slice(None, -dz)
            z_dst = slice(dz, None)
        elif dz < 0:
            z_src = slice(-dz, None)
            z_dst = slice(None, dz)
        else:
            z_src = slice(None)
            z_dst = slice(None)

        if dy > 0:
            y_src = slice(None, -dy)
            y_dst = slice(dy, None)
        elif dy < 0:
            y_src = slice(-dy, None)
            y_dst = slice(None, dy)
        else:
            y_src = slice(None)
            y_dst = slice(None)

        if dx > 0:
            x_src = slice(None, -dx)
            x_dst = slice(dx, None)
        elif dx < 0:
            x_src = slice(-dx, None)
            x_dst = slice(None, dx)
        else:
            x_src = slice(None)
            x_dst = slice(None)

        src_slice = (z_src, y_src, x_src)
        dst_slice = (z_dst, y_dst, x_dst)
        affinities[i][dst_slice] = (
            (seg[src_slice] == seg[dst_slice]) & (seg[dst_slice] > 0)
        ).astype(np.float32)

    return affinities

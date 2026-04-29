"""TTA augmentation-combination and ensemble-mode resolution."""

from __future__ import annotations

from itertools import combinations
from typing import Any, Optional

import torch

from ..utils.channel_slices import resolve_channel_range

try:
    from omegaconf import ListConfig, OmegaConf

    HAS_OMEGACONF = True
except ImportError:
    HAS_OMEGACONF = False
    ListConfig = list


def _to_plain_list(config_value) -> list:
    """Convert OmegaConf ListConfig or plain list to nested plain Python lists."""
    if HAS_OMEGACONF and isinstance(config_value, ListConfig):
        return OmegaConf.to_container(config_value, resolve=True)
    if isinstance(config_value, (list, tuple)):
        return list(config_value)
    return [config_value]


def _resolve_spatial_dims(ndim: int) -> int:
    if ndim == 5:
        return 3
    if ndim == 4:
        return 2
    raise ValueError(f"Unsupported data dimensions: {ndim}")


def _normalize_spatial_axes(
    axes: Any,
    *,
    spatial_dims: int,
    context: str,
) -> list[int]:
    if isinstance(axes, int):
        axes = [axes]
    if not isinstance(axes, (list, tuple)):
        raise ValueError(f"{context} must be an int or list of ints, got {axes!r}.")

    normalized: list[int] = []
    seen: set[int] = set()
    for raw_axis in axes:
        axis = int(raw_axis)
        if axis < 0 or axis >= spatial_dims:
            raise ValueError(f"{context} axis must be in [0, {spatial_dims - 1}], got {axis}.")
        if axis in seen:
            continue
        normalized.append(axis)
        seen.add(axis)
    return normalized


def _resolve_flip_augmentations(tta_cfg, *, spatial_dims: int) -> list[list[int]]:
    flip_axes_cfg = getattr(tta_cfg, "flip_axes", None)
    if isinstance(flip_axes_cfg, str) and flip_axes_cfg.lower() == "none":
        return [[]]

    if flip_axes_cfg == "all" or flip_axes_cfg == []:
        spatial_axes = list(range(spatial_dims))
        tta_flip_axes = [[]]
        for r in range(1, len(spatial_axes) + 1):
            for combo in combinations(spatial_axes, r):
                tta_flip_axes.append(list(combo))
        return tta_flip_axes

    if flip_axes_cfg is None:
        return [[]]

    tta_flip_axes = [[]]
    for raw_axes in _to_plain_list(flip_axes_cfg):
        tta_flip_axes.append(
            _normalize_spatial_axes(
                raw_axes,
                spatial_dims=spatial_dims,
                context="flip_axes",
            )
        )
    return tta_flip_axes


def _resolve_rotation_planes(tta_cfg, *, spatial_dims: int) -> list[tuple[int, int]]:
    rotation90_axes_cfg = getattr(tta_cfg, "rotation90_axes", None)
    if isinstance(rotation90_axes_cfg, str) and rotation90_axes_cfg.lower() == "none":
        return []

    if rotation90_axes_cfg == "all":
        if spatial_dims == 3:
            return [(0, 1), (0, 2), (1, 2)]
        if spatial_dims == 2:
            return [(0, 1)]
        raise ValueError(f"Unsupported spatial dimensions: {spatial_dims}")

    if rotation90_axes_cfg is None:
        return []

    resolved_planes: list[tuple[int, int]] = []
    for axes in _to_plain_list(rotation90_axes_cfg):
        normalized = _normalize_spatial_axes(
            axes,
            spatial_dims=spatial_dims,
            context="rotation90_axes",
        )
        if len(normalized) != 2:
            raise ValueError(
                f"Invalid rotation plane: {axes}. Each plane must contain exactly 2 axes."
            )
        plane = (normalized[0], normalized[1])
        if plane not in resolved_planes:
            resolved_planes.append(plane)
    return resolved_planes


def _resolve_rotation_k_values(tta_cfg) -> list[int]:
    rotate90_k_cfg = getattr(tta_cfg, "rotate90_k", None)
    if rotate90_k_cfg is None:
        return [0, 1, 2, 3]

    resolved_values: list[int] = []
    seen: set[int] = set()
    for raw_k in _to_plain_list(rotate90_k_cfg):
        k = int(raw_k) % 4
        if k in seen:
            continue
        resolved_values.append(k)
        seen.add(k)
    return resolved_values or [0]


def _augmentation_signature(
    *,
    spatial_dims: int,
    flip_axes: list[int],
    rotation_plane: Optional[tuple[int, int]],
    k_rotations: int,
) -> tuple[int, ...]:
    if spatial_dims == 3:
        base = torch.arange(2 * 3 * 5, dtype=torch.int64).reshape(2, 3, 5)
    elif spatial_dims == 2:
        base = torch.arange(2 * 5, dtype=torch.int64).reshape(2, 5)
    else:
        raise ValueError(f"Unsupported spatial dimensions: {spatial_dims}")

    if flip_axes:
        base = torch.flip(base, dims=flip_axes)
    if rotation_plane is not None and k_rotations % 4:
        base = torch.rot90(base, k=k_rotations, dims=rotation_plane)
    return tuple(int(v) for v in base.reshape(-1).tolist())


def resolve_tta_augmentation_combinations(
    tta_cfg,
    *,
    spatial_dims: int,
) -> list[tuple[list[int], Optional[tuple[int, int]], int]]:
    """Return unique spatial TTA combinations for the configured flips/rotations."""
    flip_variants = _resolve_flip_augmentations(tta_cfg, spatial_dims=spatial_dims)
    rotation_planes = _resolve_rotation_planes(tta_cfg, spatial_dims=spatial_dims)

    if not rotation_planes:
        return [(flip_axes, None, 0) for flip_axes in flip_variants]

    rotation_ks = _resolve_rotation_k_values(tta_cfg)
    combinations_out: list[tuple[list[int], Optional[tuple[int, int]], int]] = []
    seen_signatures: set[tuple[int, ...]] = set()

    for flip_axes in flip_variants:
        for rotation_plane in rotation_planes:
            for k_rotations in rotation_ks:
                signature = _augmentation_signature(
                    spatial_dims=spatial_dims,
                    flip_axes=flip_axes,
                    rotation_plane=rotation_plane,
                    k_rotations=k_rotations,
                )
                if signature in seen_signatures:
                    continue
                seen_signatures.add(signature)
                combinations_out.append((flip_axes, rotation_plane, k_rotations))

    return combinations_out


def _resolve_ensemble_mode_map(
    ensemble_mode: Any,
    num_channels: int,
) -> list[str]:
    """Resolve ``ensemble_mode`` config to a per-channel mode list."""
    if isinstance(ensemble_mode, str):
        return [ensemble_mode] * num_channels

    raw_list = _to_plain_list(ensemble_mode)
    if not isinstance(raw_list, list) or not raw_list:
        raise ValueError(
            f"ensemble_mode must be a string or a list of [channel_selector, mode] pairs, "
            f"got {ensemble_mode!r}."
        )

    if isinstance(raw_list[0], str) and len(raw_list) == 1:
        return [raw_list[0]] * num_channels

    modes: list[str | None] = [None] * num_channels
    for entry in raw_list:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise ValueError(
                f"Each ensemble_mode entry must be [channel_selector, mode], got {entry!r}."
            )
        selector, mode = entry
        if mode not in ("mean", "min", "max"):
            raise ValueError(
                f"Unknown ensemble mode {mode!r} in per-channel spec. Use 'mean', 'min', or 'max'."
            )
        start, stop = resolve_channel_range(
            str(selector),
            num_channels=num_channels,
            context="ensemble_mode channel selector",
        )
        for ch in range(start, stop):
            modes[ch] = mode

    unset = [i for i, m in enumerate(modes) if m is None]
    if unset:
        raise ValueError(
            f"ensemble_mode does not cover channels {unset}. "
            f"Every channel must be assigned a mode."
        )
    return modes  # type: ignore[return-value]


__all__ = [
    "_resolve_ensemble_mode_map",
    "_resolve_spatial_dims",
    "_to_plain_list",
    "resolve_tta_augmentation_combinations",
]

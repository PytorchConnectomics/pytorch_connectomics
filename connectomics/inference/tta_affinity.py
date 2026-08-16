"""Geometric inversion of directional affinity TTA predictions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional, Union

import torch

from ..data.processing.affinity import (
    resolve_affinity_channel_groups_from_cfg,
    resolve_affinity_mode_from_cfg,
    resolve_stacked_label_channel_count,
)
from ..utils.channel_slices import resolve_channel_range

ValidityBox = tuple[slice, ...]
ValidityEntry = Optional[Union[ValidityBox, torch.Tensor]]


@dataclass(frozen=True)
class ChannelMove:
    """Move one raw output channel into canonical position, optionally re-anchoring it."""

    src: int
    dst: int
    shift: Optional[tuple[int, ...]] = None


@dataclass(frozen=True)
class AffinityViewPlan:
    """Affinity correction for one configured augmentation view."""

    moves: tuple[ChannelMove, ...]
    partial_channels: frozenset[int]

    def shift_for_channel(self, channel: int) -> Optional[tuple[int, ...]]:
        for move in self.moves:
            if move.dst == channel:
                return move.shift
        return None


@dataclass(frozen=True)
class AffinityTTAPlan:
    """Resolved correction plans parallel to the configured augmentation views."""

    views: tuple[AffinityViewPlan, ...]
    partial_channels: frozenset[int]
    shifts: frozenset[tuple[int, ...]]
    num_channels: int
    spatial_rank: int


@dataclass(frozen=True)
class ViewValidity:
    """Per-channel validity for one canonicalized TTA view."""

    channels: tuple[ValidityEntry, ...]

    @classmethod
    def all_valid(cls, num_channels: int) -> "ViewValidity":
        return cls((None,) * int(num_channels))

    def select(self, indices: Optional[Sequence[int]]) -> "ViewValidity":
        if indices is None:
            return self
        return ViewValidity(tuple(self.channels[int(index)] for index in indices))


def transform_offset(
    offset: Sequence[int],
    *,
    flip_axes: Sequence[int],
    rotation_plane_spatial: Optional[tuple[int, int]],
    k: int,
) -> tuple[int, ...]:
    """Apply the linear part of inverse rotation followed by inverse flips."""
    transformed = [int(value) for value in offset]
    if rotation_plane_spatial is not None:
        p, q = (int(axis) for axis in rotation_plane_spatial)
        if p == q or min(p, q) < 0 or max(p, q) >= len(transformed):
            raise ValueError(
                f"Rotation plane {rotation_plane_spatial} is invalid for an "
                f"offset with rank {len(transformed)}."
            )
        for _ in range((-int(k)) % 4):
            transformed[p], transformed[q] = -transformed[q], transformed[p]

    for raw_axis in flip_axes:
        axis = int(raw_axis)
        if axis < 0 or axis >= len(transformed):
            raise ValueError(
                f"Flip axis {axis} is invalid for an offset with rank {len(transformed)}."
            )
        transformed[axis] = -transformed[axis]
    return tuple(transformed)


def valid_slices_for_shift(spatial_shape: Sequence[int], shift: Sequence[int]) -> ValidityBox:
    """Return the output box containing non-wrapped values after a roll shift."""
    if len(spatial_shape) != len(shift):
        raise ValueError(
            f"Roll shift rank {len(shift)} does not match spatial rank {len(spatial_shape)}."
        )
    slices: list[slice] = []
    for size, raw_shift in zip(spatial_shape, shift):
        size = int(size)
        component = int(raw_shift)
        if component > 0:
            slices.append(slice(min(component, size), size))
        elif component < 0:
            slices.append(slice(0, max(0, size + component)))
        else:
            slices.append(slice(0, size))
    return tuple(slices)


def _source_slices_for_shift(spatial_shape: Sequence[int], shift: Sequence[int]) -> ValidityBox:
    slices: list[slice] = []
    for size, raw_shift in zip(spatial_shape, shift):
        size = int(size)
        component = int(raw_shift)
        if component > 0:
            slices.append(slice(0, max(0, size - component)))
        elif component < 0:
            slices.append(slice(min(-component, size), size))
        else:
            slices.append(slice(0, size))
    return tuple(slices)


def _mapping_get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _resolve_raw_affinity_groups(
    cfg: Any,
    *,
    num_raw: int,
    requested_head: Optional[str],
) -> list[tuple[tuple[int, int], list[tuple[int, ...]]]]:
    label_groups = resolve_affinity_channel_groups_from_cfg(cfg)
    if not label_groups:
        return []

    total_label = resolve_stacked_label_channel_count(cfg)
    model_cfg = getattr(cfg, "model", None)
    heads = _mapping_get(model_cfg, "heads", {}) or {}
    if not isinstance(heads, Mapping):
        heads = {}

    if not heads:
        declared = _mapping_get(model_cfg, "out_channels", None)
        if declared is None or int(declared) != num_raw or total_label != num_raw:
            raise ValueError(
                "Affinity TTA requires an unambiguous raw-output to stacked-label mapping. "
                f"Got model.out_channels={declared}, raw output channels={num_raw}, and "
                f"stacked label channels={total_label}; all three must match."
            )
        raw_window = (0, num_raw)
    else:
        head_name = requested_head
        if head_name is None and len(heads) == 1:
            head_name = next(iter(heads))
        if head_name is None or head_name not in heads:
            raise ValueError(
                "Affinity TTA cannot map a named raw output to label channels. Select one "
                "model head and declare model.heads.<name>.target_slice."
            )
        head_cfg = heads[head_name]
        target_slice = _mapping_get(head_cfg, "target_slice", None)
        if target_slice is not None:
            start, stop = resolve_channel_range(
                target_slice,
                num_channels=total_label,
                context=f"model.heads.{head_name}.target_slice",
            )
            if stop - start != num_raw:
                raise ValueError(
                    f"model.heads.{head_name}.target_slice resolves to width {stop - start}, "
                    f"but the raw output has {num_raw} channels."
                )
            raw_window = (start, stop)
        else:
            head_width = int(_mapping_get(head_cfg, "out_channels", 0))
            if len(heads) != 1 or head_width != num_raw or total_label != num_raw:
                raise ValueError(
                    f"Affinity TTA cannot prove the label mapping for model head {head_name!r}. "
                    f"Got {len(heads)} configured head(s), head out_channels={head_width}, "
                    f"raw output channels={num_raw}, and stacked label channels={total_label}. "
                    f"Declare model.heads.{head_name}.target_slice."
                )
            raw_window = (0, num_raw)

    raw_groups: list[tuple[tuple[int, int], list[tuple[int, ...]]]] = []
    window_start, window_stop = raw_window
    for (group_start, group_stop), offsets in label_groups:
        overlap_start = max(group_start, window_start)
        overlap_stop = min(group_stop, window_stop)
        if overlap_start >= overlap_stop:
            continue
        if len(offsets) != group_stop - group_start:
            raise ValueError(
                f"Affinity group [{group_start}, {group_stop}) declares {len(offsets)} offsets; "
                "its width and offset count must match."
            )
        # A head may select a contiguous SUB-RANGE of a multi-radius affinity group: a
        # single `affinity` target with offsets r1,r1,r1,r5,r5,r5,r9,r9,r9 is one group
        # [0,9), while the model splits it across heads aff_r1="0:3", aff_r5="3:6",
        # aff_r9="6:9". Offsets are positional within the group, so label channel
        # group_start+i unambiguously carries offsets[i] and the sub-range's offsets are
        # exactly offsets[overlap-group_start]. Restrict to them rather than refusing.
        # A sub-range that is NOT closed under the transform group (e.g. a head holding
        # only one of the three unit offsets) is still rejected -- by the per-view
        # bijection check below, which is the check that can actually see it.
        sub_offsets = [
            tuple(offset)
            for offset in offsets[overlap_start - group_start:overlap_stop - group_start]
        ]
        local_start = overlap_start - window_start
        local_stop = overlap_stop - window_start
        raw_groups.append(((local_start, local_stop), sub_offsets))
    return raw_groups


def build_affinity_tta_plan(
    cfg: Any,
    *,
    augmentation_combinations: Sequence[tuple[Sequence[int], Optional[tuple[int, int]], int]],
    num_raw: int,
    requested_head: Optional[str],
) -> Optional[AffinityTTAPlan]:
    """Build an offset-driven affinity correction plan for raw model outputs."""
    if not resolve_affinity_channel_groups_from_cfg(cfg):
        return None
    raw_groups = _resolve_raw_affinity_groups(
        cfg,
        num_raw=int(num_raw),
        requested_head=requested_head,
    )
    mode = resolve_affinity_mode_from_cfg(cfg)
    if mode is None:
        raise ValueError("Affinity channel groups exist but no affinity_mode could be resolved.")

    offset_ranks = {len(offset) for _channel_range, offsets in raw_groups for offset in offsets}
    if len(offset_ranks) > 1:
        raise ValueError(f"Mixed affinity offset ranks are not supported: {sorted(offset_ranks)}.")
    spatial_rank = next(iter(offset_ranks), 0)

    for channel_range, offsets in raw_groups:
        if len(set(offsets)) != len(offsets):
            raise ValueError(
                f"Affinity group {channel_range} contains duplicate offsets: {offsets!r}."
            )

    views: list[AffinityViewPlan] = []
    all_partial: set[int] = set()
    all_shifts: set[tuple[int, ...]] = set()
    for flip_axes, rotation_plane, k_rotations in augmentation_combinations:
        moves: list[ChannelMove] = []
        view_targets: set[int] = set()
        for (start, stop), offsets in raw_groups:
            if stop - start != len(offsets):
                raise ValueError(
                    f"Affinity group [{start}, {stop}) width does not match its "
                    f"{len(offsets)} configured offsets."
                )
            for source_index, offset in enumerate(offsets):
                directed = transform_offset(
                    offset,
                    flip_axes=flip_axes,
                    rotation_plane_spatial=rotation_plane,
                    k=k_rotations,
                )
                exact = [idx for idx, target in enumerate(offsets) if target == directed]
                reversed_matches = [
                    idx
                    for idx, target in enumerate(offsets)
                    if tuple(-value for value in target) == directed
                ]
                candidates = exact if exact else reversed_matches
                if len(candidates) != 1:
                    preference = "exact" if exact else "sign-reversed"
                    raise ValueError(
                        f"Affinity offset {offset} transforms to {directed}, but group "
                        f"{offsets!r} has {len(candidates)} {preference} counterpart(s)."
                    )
                target_index = candidates[0]
                src = start + source_index
                dst = start + target_index
                if dst in view_targets:
                    raise ValueError(
                        "Affinity TTA channel mapping is not bijective: multiple source "
                        f"channels target raw channel {dst}."
                    )
                view_targets.add(dst)
                shift = None
                if not exact:
                    target_offset = offsets[target_index]
                    sign = -1 if mode == "banis" else 1
                    shift = tuple(sign * int(value) for value in target_offset)
                    if any(shift):
                        all_partial.add(dst)
                        all_shifts.add(shift)
                    else:
                        shift = None
                moves.append(ChannelMove(src=src, dst=dst, shift=shift))

            expected_targets = set(range(start, stop))
            group_targets = {move.dst for move in moves if start <= move.dst < stop}
            if group_targets != expected_targets:
                raise ValueError(
                    f"Affinity TTA mapping for group [{start}, {stop}) is not bijective."
                )

        view_partial = frozenset(move.dst for move in moves if move.shift is not None)
        views.append(AffinityViewPlan(tuple(moves), view_partial))

    return AffinityTTAPlan(
        views=tuple(views),
        partial_channels=frozenset(all_partial),
        shifts=frozenset(all_shifts),
        num_channels=int(num_raw),
        spatial_rank=spatial_rank,
    )


def validate_affinity_output(plan: Optional[AffinityTTAPlan], prediction: torch.Tensor) -> None:
    """Validate tensor-derived channel and spatial ranks against a structural plan."""
    if plan is None:
        return
    actual_channels = int(prediction.shape[1])
    if actual_channels != plan.num_channels:
        raise ValueError(
            f"Affinity TTA plan expects {plan.num_channels} raw output channels, "
            f"but the model produced {actual_channels}."
        )
    actual_spatial_rank = prediction.ndim - 2
    if plan.spatial_rank and plan.spatial_rank != actual_spatial_rank:
        raise ValueError(
            f"Affinity offset rank {plan.spatial_rank} does not match raw output spatial "
            f"rank {actual_spatial_rank}."
        )


def invert_view(
    prediction: torch.Tensor,
    *,
    flip_axes: Sequence[int],
    rotation_plane_spatial: Optional[tuple[int, int]],
    k: int,
    view_plan: Optional[AffinityViewPlan],
    tta_plan: Optional[AffinityTTAPlan],
) -> tuple[torch.Tensor, ViewValidity]:
    """Invert one spatial TTA view and canonicalize configured affinity channels.

    Long-range affinities are displaced by their full configured offset. Wrapped
    outer faces are returned as missing validity so mean/min/max never count them.
    """
    spatial = prediction
    if rotation_plane_spatial is not None and int(k) % 4:
        tensor_plane = tuple(int(axis) + 2 for axis in rotation_plane_spatial)
        spatial = torch.rot90(spatial, k=-int(k), dims=tensor_plane)
    if flip_axes:
        spatial = torch.flip(spatial, dims=[int(axis) + 2 for axis in flip_axes])

    validate_affinity_output(tta_plan, spatial)
    validity: list[ValidityEntry] = [None] * int(spatial.shape[1])
    if view_plan is None or not view_plan.moves:
        return spatial, ViewValidity(tuple(validity))

    corrected = spatial.clone()
    spatial_shape = tuple(int(value) for value in spatial.shape[2:])
    for move in view_plan.moves:
        if move.shift is None:
            corrected[:, move.dst, ...] = spatial[:, move.src, ...]
            continue
        if len(move.shift) != len(spatial_shape):
            raise ValueError(
                f"Affinity roll shift rank {len(move.shift)} does not match raw output "
                f"spatial rank {len(spatial_shape)}."
            )
        destination = valid_slices_for_shift(spatial_shape, move.shift)
        source = _source_slices_for_shift(spatial_shape, move.shift)
        corrected[:, move.dst, ...].zero_()
        corrected[(slice(None), move.dst, *destination)] = spatial[(slice(None), move.src, *source)]
        validity[move.dst] = destination

    return corrected, ViewValidity(tuple(validity))


__all__ = [
    "AffinityTTAPlan",
    "AffinityViewPlan",
    "ChannelMove",
    "ViewValidity",
    "build_affinity_tta_plan",
    "invert_view",
    "transform_offset",
    "valid_slices_for_shift",
    "validate_affinity_output",
]

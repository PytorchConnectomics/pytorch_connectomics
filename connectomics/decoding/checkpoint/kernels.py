"""Numerical kernels lifted from the zebra-finch nucleus repair prototypes."""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed

NEW_ID_BASE = 1 << 60
_SIX_CONNECTED = ndi.generate_binary_structure(3, 1)


@dataclass(frozen=True)
class CompetitiveSplit:
    labels: np.ndarray
    abstention_mask: np.ndarray
    piece_bindings: Mapping[str, tuple[int, ...]]
    assigned_voxels: Mapping[str, int]
    affected_voxels: int


@dataclass(frozen=True)
class Consolidation:
    labels: np.ndarray
    territory_bindings: Mapping[str, tuple[int, ...]]
    remap: Mapping[int, int]
    affected_voxels: int


def minpool(array: np.ndarray, factor: int) -> np.ndarray:
    if factor < 1:
        raise ValueError("pooling factor must be >= 1")
    if factor == 1:
        return np.asarray(array)
    z, y, x = (size // factor * factor for size in array.shape)
    if min(z, y, x) == 0:
        raise ValueError(f"pooling factor {factor} exceeds array shape {array.shape}")
    return (
        array[:z, :y, :x]
        .reshape(
            z // factor,
            factor,
            y // factor,
            factor,
            x // factor,
            factor,
        )
        .min(axis=(1, 3, 5))
    )


def maxpool_bool(array: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return np.asarray(array, dtype=bool)
    z, y, x = (size // factor * factor for size in array.shape)
    if min(z, y, x) == 0:
        raise ValueError(f"pooling factor {factor} exceeds array shape {array.shape}")
    return (
        array[:z, :y, :x]
        .reshape(
            z // factor,
            factor,
            y // factor,
            factor,
            x // factor,
            factor,
        )
        .max(axis=(1, 3, 5))
    )


def _restore_sigmoid(values: np.ndarray, temperature: float | None) -> np.ndarray:
    result = values.astype(np.float32, copy=False)
    if temperature is None:
        return result
    if temperature <= 0:
        raise ValueError("sigmoid restore temperature must be positive")
    clipped = np.clip(result, 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped)) / temperature
    return 1.0 / (1.0 + np.exp(-logits))


def affinity_cost(
    affinity: np.ndarray,
    *,
    channel_indices: Sequence[int],
    channel_axis: int,
    convention: str,
    sigmoid_restore: float | None,
    factor: int,
) -> np.ndarray:
    """Return ``1 - min(channel)`` after the pinned legacy transforms.

    BANIS stores nearest-neighbour edges at the source voxel, so each selected
    channel is rolled one voxel toward its destination-index representation.
    Pooling takes the minimum affinity in each factor cube before conversion
    to cost, matching the measured prototype.  Only complete leading-aligned
    cubes are pooled; nearest upsampling may therefore differ within at most
    ``factor - 1`` trailing voxels per axis.
    """

    if affinity.ndim != 4:
        raise ValueError(f"affinity must be four-dimensional, got {affinity.shape}")
    czyx = np.moveaxis(affinity, channel_axis, 0)
    channels = tuple(int(index) for index in channel_indices)
    if not channels or min(channels) < 0 or max(channels) >= czyx.shape[0]:
        raise ValueError(f"invalid affinity channel indices {channels} for shape {czyx.shape}")
    selected = _restore_sigmoid(czyx[list(channels)], sigmoid_restore)
    if convention not in ("probability", "deepem", "banis"):
        raise ValueError(f"unsupported affinity convention {convention!r}")
    if convention == "banis":
        for channel, axis in enumerate(range(3)):
            if channel < selected.shape[0]:
                selected[channel] = np.roll(selected[channel], 1, axis=axis)
    similarity = selected.min(axis=0)
    return 1.0 - minpool(similarity, factor)


def _upsample_nearest(array: np.ndarray, shape: tuple[int, int, int], factor: int) -> np.ndarray:
    if factor == 1:
        return array[tuple(slice(0, size) for size in shape)]
    up = np.repeat(np.repeat(np.repeat(array, factor, axis=0), factor, axis=1), factor, axis=2)
    padding = [(0, max(0, shape[axis] - up.shape[axis])) for axis in range(3)]
    if any(amount for _, amount in padding):
        up = np.pad(up, padding, mode="edge")
    return up[tuple(slice(0, size) for size in shape)]


def _stable_uint64(component_id: int, anchor_id: int, piece: int) -> int:
    payload = f"{component_id}:{anchor_id}:{piece}".encode()
    offset = int.from_bytes(hashlib.sha256(payload).digest()[:7], "big")
    return NEW_ID_BASE + offset


def _touches_boundary(mask: np.ndarray) -> bool:
    return bool(
        mask[0].any()
        or mask[-1].any()
        or mask[:, 0].any()
        or mask[:, -1].any()
        or mask[:, :, 0].any()
        or mask[:, :, -1].any()
    )


def competitive_split_component(
    segmentation: np.ndarray,
    nuclei: np.ndarray,
    affinity: np.ndarray,
    *,
    component_id: int,
    anchor_ids: Sequence[int],
    channel_indices: Sequence[int],
    channel_axis: int = 0,
    convention: str = "probability",
    sigmoid_restore: float | None = None,
    factor: int = 1,
    contained: bool = True,
) -> CompetitiveSplit:
    """Seeded six-connected flood confined to one existing component.

    Piece IDs are assigned before consolidation so an action record can bind
    every disconnected piece.  Lowest numeric anchor ID wins exact watershed
    ties because markers are assigned in that order and retain that order in
    the watershed queue.
    """

    if segmentation.shape != nuclei.shape:
        raise ValueError("segmentation and nucleus arrays must share a z/y/x shape")
    parent = segmentation == component_id
    if not parent.any():
        raise ValueError(f"component {component_id} is absent from the repair scope")
    anchors = tuple(sorted(set(int(value) for value in anchor_ids)))
    if len(anchors) < 2:
        raise ValueError("competitive split requires at least two anchors")
    cost = affinity_cost(
        affinity,
        channel_indices=channel_indices,
        channel_axis=channel_axis,
        convention=convention,
        sigmoid_restore=sigmoid_restore,
        factor=factor,
    )
    pooled_parent = maxpool_bool(parent, factor)
    markers = np.zeros(pooled_parent.shape, dtype=np.int32)
    marker_to_anchor: dict[int, int] = {}
    for marker, anchor in enumerate(anchors, start=1):
        seed = maxpool_bool((nuclei == anchor) & parent, factor)
        available = seed & pooled_parent & (markers == 0)
        if not available.any():
            raise ValueError(f"anchor {anchor} has no seed in component {component_id}")
        markers[available] = marker
        marker_to_anchor[marker] = anchor
    territory = watershed(cost, markers=markers, mask=pooled_parent, connectivity=_SIX_CONNECTED)
    territory = _upsample_nearest(territory, segmentation.shape, factor)
    territory = np.where(parent, territory, 0)

    labels = np.asarray(segmentation, dtype=np.uint64).copy()
    abstention = parent & (territory == 0)
    piece_bindings: dict[str, tuple[int, ...]] = {}
    assigned: dict[str, int] = {}
    pieces_by_anchor: dict[int, list[tuple[np.ndarray, int]]] = {}
    for marker, anchor in marker_to_anchor.items():
        anchor_territory = parent & (territory == marker)
        components, count = ndi.label(anchor_territory, structure=_SIX_CONNECTED)
        pieces_by_anchor[anchor] = []
        for piece in range(1, count + 1):
            mask = components == piece
            pieces_by_anchor[anchor].append((mask, int(mask.sum())))

    keeper_anchor = min(
        anchors,
        key=lambda anchor: (-sum(size for _, size in pieces_by_anchor[anchor]), anchor),
    )
    for anchor in anchors:
        ids: list[int] = []
        for piece_index, (mask, _size) in enumerate(
            sorted(pieces_by_anchor[anchor], key=lambda item: -item[1])
        ):
            if not contained and anchor != keeper_anchor and _touches_boundary(mask):
                abstention |= mask
                continue
            label_id = (
                component_id
                if anchor == keeper_anchor and piece_index == 0
                else _stable_uint64(component_id, anchor, piece_index)
            )
            labels[mask] = np.asarray(label_id, dtype=labels.dtype)
            ids.append(label_id)
        piece_bindings[str(anchor)] = tuple(ids)
        assigned[str(anchor)] = int(sum((labels == value).sum() for value in ids))
    labels[abstention] = component_id
    return CompetitiveSplit(
        labels=labels,
        abstention_mask=abstention,
        piece_bindings=piece_bindings,
        assigned_voxels=assigned,
        affected_voxels=int((labels != segmentation).sum()),
    )


def consolidate_same_anchor(
    labels: np.ndarray,
    piece_bindings: Mapping[str, tuple[int, ...]],
    *,
    cannot_links: set[tuple[int, int]] | None = None,
) -> Consolidation:
    """Collapse same-anchor pieces without crossing a distinct-anchor exclusion."""

    exclusions = cannot_links or set()
    output = np.array(labels, copy=True)
    remap: dict[int, int] = {}
    territory_bindings: dict[str, tuple[int, ...]] = {}
    owner = {piece: anchor for anchor, pieces in piece_bindings.items() for piece in pieces}
    for anchor in sorted(piece_bindings, key=int):
        pieces = [piece for piece in piece_bindings[anchor] if np.any(output == piece)]
        if not pieces:
            territory_bindings[anchor] = ()
            continue
        keeper = min(pieces, key=lambda piece: (-int((output == piece).sum()), piece))
        for piece in pieces:
            if piece == keeper:
                continue
            if any(
                tuple(sorted((piece, other))) in exclusions and owner.get(other) != anchor
                for other in owner
            ):
                raise ValueError(f"same-anchor consolidation of {piece} crosses a cannot-link")
            output[output == piece] = keeper
            remap[piece] = keeper
        territory_bindings[anchor] = (keeper,)
    return Consolidation(
        labels=output,
        territory_bindings=territory_bindings,
        remap=remap,
        affected_voxels=int(sum((labels == old).sum() for old in remap)),
    )


def pairwise_cannot_links(
    territory_bindings: Mapping[str, tuple[int, ...]],
) -> tuple[tuple[int, int], ...]:
    pairs: set[tuple[int, int]] = set()
    for left, right in itertools.combinations(sorted(territory_bindings, key=int), 2):
        for a in territory_bindings[left]:
            for b in territory_bindings[right]:
                if a != b:
                    pairs.add(tuple(sorted((int(a), int(b)))))
    return tuple(sorted(pairs))


def local_rag(labels: np.ndarray) -> tuple[tuple[int, int], ...]:
    edges: set[tuple[int, int]] = set()
    for axis in range(3):
        first = [slice(None)] * 3
        second = [slice(None)] * 3
        first[axis] = slice(0, -1)
        second[axis] = slice(1, None)
        a = labels[tuple(first)]
        b = labels[tuple(second)]
        changed = (a != b) & (a != 0) & (b != 0)
        for left, right in zip(a[changed].tolist(), b[changed].tolist()):
            edges.add(tuple(sorted((int(left), int(right)))))
    return tuple(sorted(edges))


__all__ = [
    "CompetitiveSplit",
    "Consolidation",
    "NEW_ID_BASE",
    "affinity_cost",
    "competitive_split_component",
    "consolidate_same_anchor",
    "local_rag",
    "maxpool_bool",
    "minpool",
    "pairwise_cannot_links",
]

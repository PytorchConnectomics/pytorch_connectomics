"""Inference-specific chunk-grid helpers.

The shared chunk-grid primitives (``ChunkRef``, ``build_chunk_grid``) live
in :mod:`connectomics.chunked.chunk_grid`. This module owns only the
inference-side helpers that read the structured :class:`Config` (crop_pad,
affinity offsets, save_backend, chunking output mode, h5 chunk sizes).
"""

from __future__ import annotations

from typing import Any, Sequence

from ..data.processing.affinity import (
    compute_affinity_crop_pad,
    resolve_affinity_channel_groups_from_cfg,
    resolve_affinity_mode_from_cfg,
)
from ..utils.channel_slices import resolve_channel_indices
from ..utils.model_outputs import get_inference_select_channel


def normalize_crop_pad(value: Any) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    if value in (None, [], ()):
        return ((0, 0), (0, 0), (0, 0))
    values = [int(v) for v in value]
    if len(values) == 3:
        return tuple((v, v) for v in values)  # type: ignore[return-value]
    if len(values) == 6:
        return ((values[0], values[1]), (values[2], values[3]), (values[4], values[5]))
    raise ValueError(f"inference.model.crop_pad must have length 3 or 6, got {value!r}")


def resolve_selected_affinity_offsets(cfg: Any) -> list[tuple[int, int, int]]:
    groups = resolve_affinity_channel_groups_from_cfg(cfg)
    if not groups:
        return []

    label_channels = max(end for (start, end), _offsets in groups)
    channel_offsets: list[tuple[int, int, int] | None] = [None] * label_channels
    for (start, end), offsets in groups:
        for channel, offset in zip(range(start, end), offsets):
            channel_offsets[channel] = offset

    select_channel = get_inference_select_channel(cfg)
    if select_channel is not None:
        selected = resolve_channel_indices(
            select_channel,
            num_channels=len(channel_offsets),
            context="inference.model.select_channel",
        )
        channel_offsets = [channel_offsets[idx] for idx in selected]

    return [offset for offset in channel_offsets if offset is not None]


def resolve_global_prediction_crop(
    cfg: Any,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    inference_model = getattr(getattr(cfg, "inference", None), "model", None)
    user_crop = normalize_crop_pad(getattr(inference_model, "crop_pad", None))
    affinity_mode = resolve_affinity_mode_from_cfg(cfg)
    if affinity_mode != "deepem":
        affinity_crop = ((0, 0), (0, 0), (0, 0))
    else:
        offsets = resolve_selected_affinity_offsets(cfg)
        affinity_crop = (
            compute_affinity_crop_pad(offsets, affinity_mode=affinity_mode)
            if offsets
            else ((0, 0), (0, 0), (0, 0))
        )
    return tuple(
        (
            int(user_crop[axis][0]) + int(affinity_crop[axis][0]),
            int(user_crop[axis][1]) + int(affinity_crop[axis][1]),
        )
        for axis in range(3)
    )  # type: ignore[return-value]


def validate_chunked_output_format(cfg: Any) -> None:
    inference_cfg = getattr(cfg, "inference", None)
    backend = str(getattr(inference_cfg, "save_backend", "h5")).lower()
    if backend not in {"h5", "hdf5"}:
        raise ValueError(
            "Chunked inference writes a single streamed HDF5 output only; "
            f"unsupported inference.save_backend={backend!r}."
        )


def resolve_chunk_shape(cfg: Any, final_shape: Sequence[int]) -> tuple[int, int, int]:
    chunking_cfg = cfg.inference.chunking
    chunk_size = tuple(int(v) for v in chunking_cfg.chunk_size)
    axes = str(getattr(chunking_cfg, "axes", "all")).lower()
    if axes == "z":
        return (chunk_size[0], int(final_shape[1]), int(final_shape[2]))
    if axes != "all":
        raise ValueError("inference.chunking.axes must be 'all' or 'z'")
    return tuple(min(chunk_size[axis], int(final_shape[axis])) for axis in range(3))


def resolve_h5_spatial_chunks(spatial_shape: Sequence[int]) -> tuple[int, int, int]:
    preferred = (64, 64, 64)
    return tuple(min(int(spatial_shape[axis]), preferred[axis]) for axis in range(3))


def resolve_chunk_output_mode(cfg: Any) -> str:
    chunking_cfg = cfg.inference.chunking
    mode = str(getattr(chunking_cfg, "output_mode", "decoded")).lower()
    if mode not in {"decoded", "raw_prediction"}:
        raise ValueError("inference.chunking.output_mode must be 'decoded' or 'raw_prediction'.")
    return mode


__all__ = [
    "normalize_crop_pad",
    "resolve_selected_affinity_offsets",
    "resolve_global_prediction_crop",
    "validate_chunked_output_format",
    "resolve_chunk_shape",
    "resolve_h5_spatial_chunks",
    "resolve_chunk_output_mode",
]

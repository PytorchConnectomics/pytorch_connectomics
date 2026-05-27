"""Halo coordinate math shared by chunked workflows."""

from __future__ import annotations

from typing import Sequence

from .chunk_grid import ChunkRef

__all__ = ["resolve_halo_region"]


def resolve_halo_region(
    chunk: ChunkRef,
    input_shape: Sequence[int],
    *,
    halo: Sequence[int] = (0, 0, 0),
    crop_before: Sequence[int] = (0, 0, 0),
) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[slice, slice, slice]]:
    """Compute read/core boundaries for a chunk with halo + global crop.

    Returns:
        read_start: per-axis read start in input-volume coordinates (halo-extended,
            clipped to the input volume bounds).
        read_stop: per-axis read stop (halo-extended, clipped).
        local_core_slices: slices into the read crop selecting the inner
            (non-halo) chunk region.
    """
    input_shape = tuple(int(v) for v in input_shape)
    halo = tuple(int(v) for v in halo)
    crop_before = tuple(int(v) for v in crop_before)

    core_start = tuple(chunk.start[axis] + crop_before[axis] for axis in range(3))
    core_stop = tuple(chunk.stop[axis] + crop_before[axis] for axis in range(3))
    read_start = tuple(max(0, core_start[axis] - halo[axis]) for axis in range(3))
    read_stop = tuple(min(input_shape[axis], core_stop[axis] + halo[axis]) for axis in range(3))
    local_core_slices = tuple(
        slice(core_start[axis] - read_start[axis], core_stop[axis] - read_start[axis])
        for axis in range(3)
    )
    return read_start, read_stop, local_core_slices

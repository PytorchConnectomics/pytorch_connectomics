"""Chunk grid types and builder shared across chunked workflows."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Sequence

__all__ = ["ChunkRef", "build_chunk_grid"]


@dataclass(frozen=True)
class ChunkRef:
    index: tuple[int, int, int]
    start: tuple[int, int, int]
    stop: tuple[int, int, int]

    @property
    def key(self) -> str:
        z, y, x = self.index
        return f"z{z}_y{y}_x{x}"

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(self.stop[i] - self.start[i] for i in range(3))

    @property
    def slices(self) -> tuple[slice, slice, slice]:
        return tuple(slice(self.start[axis], self.stop[axis]) for axis in range(3))


def build_chunk_grid(volume_shape: Sequence[int], chunk_shape: Sequence[int]) -> list[ChunkRef]:
    volume = tuple(int(v) for v in volume_shape)
    chunk = tuple(int(v) for v in chunk_shape)
    if len(volume) != 3 or len(chunk) != 3:
        raise ValueError("volume_shape and chunk_shape must both be length-3 tuples.")
    counts = tuple((volume[axis] + chunk[axis] - 1) // chunk[axis] for axis in range(3))
    result: list[ChunkRef] = []
    for index in product(*(range(count) for count in counts)):
        start = tuple(index[axis] * chunk[axis] for axis in range(3))
        stop = tuple(min(start[axis] + chunk[axis], volume[axis]) for axis in range(3))
        result.append(ChunkRef(index=tuple(int(v) for v in index), start=start, stop=stop))
    return result

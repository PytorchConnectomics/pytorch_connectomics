"""Source-indexed BANIS affinity reads for the arm0_96 chunk store.

Channel ``c`` stores the edge from source voxel ``p`` to ``p + OFFSETS[c]`` at
``R[c, p]``.  This module never applies the legacy decoder-side ``2 - axis``
channel transformation during feature extraction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import zarr

AFFINITY_CHUNK = 1008
AFFINITY_OFFSETS_ZYX = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
RESTORE_SIGMOID_SCALE = 0.2
VOLUME_SHAPE_ZYX = (5700, 10912, 10664)


def restore_sigmoid(values: np.ndarray, scale: float = RESTORE_SIGMOID_SCALE) -> np.ndarray:
    """Undo sigmoid compression in float32 and clip the result to ``[0, 1]``."""
    if scale <= 0:
        raise ValueError("restore-sigmoid scale must be positive")
    values_f = np.asarray(values, dtype=np.float32)
    eps = np.finfo(np.float32).eps
    clipped = np.clip(values_f, eps, 1.0 - eps)
    logit = np.log(clipped) - np.log1p(-clipped)
    restored = 1.0 / (1.0 + np.exp(-logit / np.float32(scale)))
    return np.clip(restored, 0.0, 1.0).astype(np.float32, copy=False)


def banis_to_abiss(restored_banis: np.ndarray) -> np.ndarray:
    """Apply the producer's Form-2 source shift and XYZ output channel order.

    For ABISS output index ``q`` and channel ``c'`` (x, y, z), the output is
    ``R[2-c', q-e_(2-c')]``.  The low face without a source is zero padded.
    """
    restored_banis = np.asarray(restored_banis, dtype=np.float32)
    if restored_banis.ndim != 4 or restored_banis.shape[0] != 3:
        raise ValueError("BANIS affinity must have shape (3, Z, Y, X)")
    output = np.zeros_like(restored_banis, dtype=np.float32)
    for output_channel in range(3):
        source_channel = 2 - output_channel
        axis = source_channel
        destination = [slice(None)] * 3
        source = [slice(None)] * 3
        destination[axis] = slice(1, None)
        source[axis] = slice(None, -1)
        output[(output_channel, *destination)] = restored_banis[(source_channel, *source)]
    return output


class _ArrayVolume:
    """Array reference used by the source-indexed affinity truth table.

    The class deliberately exposes both declared forms: ``edge`` reads the
    native source-indexed BANIS field and ``abiss`` performs the producer-side
    channel reversal, one-voxel source shift, and low-face zero padding.
    """

    def __init__(
        self,
        array: np.ndarray,
        *,
        convention: str,
        restore_sigmoid_scale: float,
    ) -> None:
        if convention != "banis":
            raise ValueError("only the verified BANIS convention is accepted")
        source = np.asarray(array)
        if source.ndim != 4 or source.shape[0] != 3:
            raise ValueError("BANIS affinity must have shape (3, Z, Y, X)")
        self.restored = restore_sigmoid(source, restore_sigmoid_scale)

    def edge(self, channel: int, source_zyx: np.ndarray | Sequence[int]) -> np.ndarray:
        return edge_affinity(self.restored, channel, source_zyx)

    def abiss(self) -> np.ndarray:
        return banis_to_abiss(self.restored)


def edge_affinity(
    restored_banis: np.ndarray,
    channel: int,
    source_zyx: np.ndarray | Sequence[int],
    origin_zyx: Sequence[int] = (0, 0, 0),
) -> np.ndarray:
    """Return Form-1 affinity for edges ``(p, p + offsets[channel])``."""
    if channel not in (0, 1, 2):
        raise ValueError(f"invalid channel: {channel}")
    points = np.asarray(source_zyx, dtype=np.int64)
    scalar = points.ndim == 1
    points = np.atleast_2d(points)
    if points.shape[1] != 3:
        raise ValueError("source coordinates must be ZYX triples")
    local = points - np.asarray(origin_zyx, dtype=np.int64)
    spatial = np.asarray(restored_banis.shape[1:], dtype=np.int64)
    destination = points + np.asarray(AFFINITY_OFFSETS_ZYX[channel], dtype=np.int64)
    if np.any(local < 0) or np.any(local >= spatial):
        raise IndexError("source coordinate is outside the loaded affinity block")
    if np.any(destination < 0) or np.any(destination >= np.asarray(VOLUME_SHAPE_ZYX)):
        raise IndexError("edge destination lies outside the global volume")
    result = np.asarray(restored_banis[channel][tuple(local.T)], dtype=np.float32)
    return result[0] if scalar else result


@dataclass(frozen=True)
class AffinitySlab:
    values: np.ndarray
    origin_zyx: tuple[int, int, int]
    requested_lo_zyx: tuple[int, int, int]
    requested_hi_zyx: tuple[int, int, int]


class H5AffinityStore:
    """Coordinate-local reader for the core-only 1008-voxel HDF5 chunk grid."""

    def __init__(
        self,
        root: Path,
        keep_mask: Path | None = None,
        shape_zyx: Sequence[int] = VOLUME_SHAPE_ZYX,
        chunk_size: int = AFFINITY_CHUNK,
        restore_scale: float = RESTORE_SIGMOID_SCALE,
    ):
        self.root = Path(root)
        self.keep_mask_path = Path(keep_mask) if keep_mask is not None else None
        self.shape_zyx = tuple(int(value) for value in shape_zyx)
        self.chunk_size = int(chunk_size)
        self.restore_scale = float(restore_scale)
        self._keep = None

    def _chunk_path(self, index: Sequence[int]) -> Path:
        return self.root / f"chunk_z{index[0]}_y{index[1]}_x{index[2]}.h5"

    def _keep_array(self):
        if self.keep_mask_path is None:
            return None
        if self._keep is None:
            self._keep = zarr.open_array(str(self.keep_mask_path), mode="r")
        return self._keep

    def read_slab(
        self,
        lo_zyx: Sequence[int],
        hi_zyx: Sequence[int],
        *,
        low_halo: bool = True,
    ) -> AffinitySlab:
        """Read ``[lo, hi)`` and optionally one low-side source plane per axis."""
        requested_lo = np.asarray(lo_zyx, dtype=np.int64)
        requested_hi = np.asarray(hi_zyx, dtype=np.int64)
        shape = np.asarray(self.shape_zyx, dtype=np.int64)
        if np.any(requested_lo < 0) or np.any(requested_hi > shape):
            raise IndexError(
                f"requested affinity slab {requested_lo.tolist()}:{requested_hi.tolist()} "
                f"outside {shape.tolist()}"
            )
        if np.any(requested_hi <= requested_lo):
            raise ValueError("affinity slab must be non-empty")
        actual_lo = np.maximum(requested_lo - int(low_halo), 0)
        actual_hi = requested_hi
        out_shape = tuple((actual_hi - actual_lo).tolist())
        raw = np.zeros((3, *out_shape), dtype=np.float16)

        first = actual_lo // self.chunk_size
        last = (actual_hi - 1) // self.chunk_size
        for chunk_z in range(int(first[0]), int(last[0]) + 1):
            for chunk_y in range(int(first[1]), int(last[1]) + 1):
                for chunk_x in range(int(first[2]), int(last[2]) + 1):
                    index = np.asarray((chunk_z, chunk_y, chunk_x), dtype=np.int64)
                    chunk_lo = index * self.chunk_size
                    chunk_hi = np.minimum(chunk_lo + self.chunk_size, shape)
                    overlap_lo = np.maximum(actual_lo, chunk_lo)
                    overlap_hi = np.minimum(actual_hi, chunk_hi)
                    if np.any(overlap_hi <= overlap_lo):
                        continue
                    path = self._chunk_path(index)
                    if not path.is_file():
                        raise FileNotFoundError(f"missing affinity core chunk: {path}")
                    source = tuple(
                        slice(
                            int(overlap_lo[axis] - chunk_lo[axis]),
                            int(overlap_hi[axis] - chunk_lo[axis]),
                        )
                        for axis in range(3)
                    )
                    destination = tuple(
                        slice(
                            int(overlap_lo[axis] - actual_lo[axis]),
                            int(overlap_hi[axis] - actual_lo[axis]),
                        )
                        for axis in range(3)
                    )
                    with h5py.File(path, "r") as handle:
                        dataset = handle["main"]
                        if dataset.shape[0] != 3:
                            raise ValueError(f"{path}: expected 3 channels, got {dataset.shape}")
                        raw[(slice(None), *destination)] = dataset[(slice(None), *source)]

        restored = restore_sigmoid(raw, self.restore_scale)
        keep = self._keep_array()
        if keep is not None:
            mask = np.asarray(
                keep[
                    int(actual_lo[0]) : int(actual_hi[0]),
                    int(actual_lo[1]) : int(actual_hi[1]),
                    int(actual_lo[2]) : int(actual_hi[2]),
                ],
                dtype=bool,
            )
            restored *= mask[np.newaxis]
        return AffinitySlab(
            values=restored,
            origin_zyx=tuple(int(value) for value in actual_lo),
            requested_lo_zyx=tuple(int(value) for value in requested_lo),
            requested_hi_zyx=tuple(int(value) for value in requested_hi),
        )


def expected_chunk_grid(shape_zyx: Sequence[int] = VOLUME_SHAPE_ZYX) -> tuple[int, int, int]:
    return tuple(int(math.ceil(int(value) / AFFINITY_CHUNK)) for value in shape_zyx)

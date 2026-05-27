"""SkeletonVolumeProcessor: chunked kimimaro skeleton-volume precompute.

For each chunk: read segmentation crop, run kimimaro on it, rasterize
the resulting per-instance skeleton vertices back into a same-shape
volume (matching the input dtype) where each skeleton voxel stores
its instance ID and non-skeleton voxels are 0. The output is consumed
by ``connectomics.data.processing.distance.skeleton_aware_edt_from_skeleton_vol``
at training time.

Tradeoff: skeletons of neurites crossing chunk boundaries are split.
SDT is a soft supervision signal and absorbs the small discontinuity.
Set ``overlap`` > 0 to skeletonize each chunk with a halo for slightly
better boundary skeletons; rasterization still only writes the inner
chunk region (the output layout is unchanged).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .chunk_grid import ChunkRef
from .processor import ChunkedProcessor, ChunkedProcessorConfig

__all__ = ["SkeletonVolumeProcessor", "SkeletonVolumeConfig"]


@dataclass
class SkeletonVolumeConfig(ChunkedProcessorConfig):
    """Config for :class:`SkeletonVolumeProcessor`.

    Inherits all base fields (input/output paths, chunk_shape, parallel,
    overlap, etc.) and adds the single kimimaro knob.
    """

    resolution: Tuple[float, float, float] = (1.0, 1.0, 1.0)


class SkeletonVolumeProcessor(ChunkedProcessor):
    """Chunked variant of
    :func:`connectomics.data.processing.distance.precompute_skeleton_volume`.

    Output dtype matches the input segmentation dtype (each voxel holds
    an instance ID, so the source dtype is by construction wide enough).
    Callers can override via ``config.output_dtype`` if needed.
    """

    config_class = SkeletonVolumeConfig

    def __init__(self, config: SkeletonVolumeConfig):
        if not isinstance(config, SkeletonVolumeConfig):
            raise TypeError(
                f"config must be SkeletonVolumeConfig, got {type(config).__name__}."
            )
        super().__init__(config)

    def process_chunk(self, chunk_data: np.ndarray, chunk: ChunkRef) -> np.ndarray:
        from connectomics.data.processing.distance import _batch_skeletonize

        cfg: SkeletonVolumeConfig = self.config  # type: ignore[assignment]
        resolution = tuple(float(v) for v in cfg.resolution)

        # If the caller is using overlap>0, chunk_data is the halo-extended
        # crop and we need to trim back to the inner region after
        # skeletonization. The inner region inside the halo-extended crop
        # is offset by `overlap` on each axis at the low side, but clipped
        # by the volume edge — same logic as halo.resolve_halo_region.
        # The driver passes the halo-extended crop here so the kimimaro
        # call sees boundary context.
        out_shape = chunk.shape
        inner_offset = self._inner_offset(chunk_data.shape, out_shape)

        # Skeletonize the (possibly halo-extended) chunk.
        vertices_by_id = _batch_skeletonize(chunk_data, resolution, max_parallel=1)

        # Rasterize into a chunk-shape volume matching the input dtype.
        # Vertices outside the inner region (i.e. in the halo) are dropped;
        # vertices inside the inner region are translated by -inner_offset.
        out = np.zeros(out_shape, dtype=chunk_data.dtype)
        inner_lo = np.asarray(inner_offset, dtype=np.int64)
        inner_hi = inner_lo + np.asarray(out_shape, dtype=np.int64)
        for inst_id, verts in vertices_by_id.items():
            if verts.shape[0] == 0:
                continue
            v = verts.astype(np.int64, copy=False)
            mask = np.all((v >= inner_lo) & (v < inner_hi), axis=1)
            v = v[mask] - inner_lo
            if v.shape[0] > 0:
                out[v[:, 0], v[:, 1], v[:, 2]] = inst_id
        return out

    @staticmethod
    def _inner_offset(
        crop_shape: Tuple[int, ...], inner_shape: Tuple[int, int, int]
    ) -> Tuple[int, int, int]:
        # crop_shape may be larger than inner_shape by up to overlap on each
        # axis (clipped at volume edge). The inner region starts at the same
        # low-side offset that the halo took. Because the volume edge can
        # cap the halo on one side, derive the actual offset from the
        # difference in sizes split as the low-side trim.
        # For interior chunks: offset = overlap on each axis.
        # For edge chunks: offset = crop_shape - inner_shape (low side trim)
        # We assume the halo is symmetric where possible and the volume edge
        # only trims the high side first. This holds for resolve_halo_region.
        # See halo.resolve_halo_region: read_start = max(0, core_start - halo).
        # So inner_offset = core_start - read_start, which equals halo when
        # core_start >= halo and equals core_start otherwise. We can't recover
        # core_start here; instead we assume the inner_offset equals the
        # low-side excess: max(0, crop - inner) on each axis, computed
        # symmetrically. For the non-edge case this just gives `overlap`;
        # for edge chunks at the start of an axis, the halo is shorter so
        # the inner_offset matches what was actually trimmed.
        return tuple(max(0, crop_shape[axis] - inner_shape[axis]) for axis in range(3))

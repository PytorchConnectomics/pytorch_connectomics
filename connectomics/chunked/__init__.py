"""Chunked map-over-volume workflows.

Public surface (intentionally small):

- :class:`ChunkRef` and :func:`build_chunk_grid` — chunk-grid primitives.
- :func:`resolve_halo_region` — read/core coordinate helper.
- :class:`ResumeManifest` and :class:`ManifestConfigMismatch` — sidecar
  manifest for crash-safe resume.
- :class:`ChunkedProcessor` and :class:`ChunkedProcessorConfig` — parent
  class for chunked processing pipelines (ProcessPool workers +
  ThreadPool(1) writer + bounded backpressure + resume manifest).
- :class:`SkeletonVolumeProcessor` and :class:`SkeletonVolumeConfig` —
  kimimaro skeleton-volume precompute.
"""

from __future__ import annotations

from .chunk_grid import ChunkRef, build_chunk_grid
from .halo import resolve_halo_region
from .manifest import ManifestConfigMismatch, ResumeManifest
from .processor import ChunkedProcessor, ChunkedProcessorConfig
from .skeleton import SkeletonVolumeConfig, SkeletonVolumeProcessor

__all__ = [
    "ChunkRef",
    "build_chunk_grid",
    "resolve_halo_region",
    "ManifestConfigMismatch",
    "ResumeManifest",
    "ChunkedProcessor",
    "ChunkedProcessorConfig",
    "SkeletonVolumeConfig",
    "SkeletonVolumeProcessor",
]

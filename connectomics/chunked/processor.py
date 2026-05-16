"""ChunkedProcessor: parent class for chunked map-over-volume workflows.

Architecture:

- Build a chunk grid over the input volume.
- Dispatch each chunk to a `ProcessPoolExecutor` of CPU workers (one
  per chunk; the per-chunk function returns the output array).
- A single-threaded writer (`ThreadPoolExecutor(max_workers=1)`) writes
  each chunk's result into a preallocated chunked H5/zarr at the chunk's
  spatial slice. This avoids the concurrent-HDF5-write pitfall.
- Backpressure: at most ``2 * parallel`` write-futures in flight.
- Resume: a JSON sidecar ``<output>.chunks.json`` records completed
  chunk keys; restarts skip them.

Subclasses override :meth:`setup` (output allocation hook) and
:meth:`process_chunk` (per-chunk compute).
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np

from .chunk_grid import ChunkRef, build_chunk_grid
from .manifest import ManifestConfigMismatch, ResumeManifest

__all__ = ["ChunkedProcessor", "ChunkedProcessorConfig"]


@dataclass
class ChunkedProcessorConfig:
    """All public knobs for ChunkedProcessor; picklable by design.

    Subclasses extend this with their own fields (resolution, kimimaro
    params, etc.) and override ``ChunkedProcessor.config_class``.
    """

    input_path: str
    output_path: str
    chunk_shape: Tuple[int, int, int]
    parallel: int = 1
    overlap: int = 0
    input_dataset: str = "main"
    output_dataset: str = "main"
    output_dtype: Optional[str] = None  # None => same as input dtype
    compression: Optional[str] = "gzip"
    compression_level: int = 4
    h5_chunks: Optional[Tuple[int, int, int]] = None
    overwrite: bool = False
    extra: dict = field(default_factory=dict)


class ChunkedProcessor:
    """Parent class for chunked map-over-volume workflows.

    Subclassing contract:

    - Override :meth:`process_chunk`.
    - Optionally override :meth:`setup` if the output shape or dtype
      differs from the input (default: same shape, ``config.output_dtype``
      or input dtype).
    - Keep the instance picklable. Pool workers serialize ``self`` via
      pickle; non-picklable attributes (open file handles, GPU tensors)
      must not be set on ``self``.
    """

    #: subclasses may override to specialize the config dataclass
    config_class: type[ChunkedProcessorConfig] = ChunkedProcessorConfig

    def __init__(self, config: ChunkedProcessorConfig):
        if not isinstance(config, ChunkedProcessorConfig):
            raise TypeError(
                f"config must be a ChunkedProcessorConfig (or subclass), got {type(config).__name__}."
            )
        self.config = config
        self._input_shape: Optional[Tuple[int, int, int]] = None
        self._output_shape: Optional[Tuple[int, int, int]] = None
        self._output_dtype: Optional[np.dtype] = None
        self._n_input_axes: int = 3  # number of leading spatial axes; 3 by default

    # ------------------------------------------------------------------ hooks

    def setup(self) -> Tuple[Tuple[int, int, int], np.dtype, int]:
        """Inspect the input and decide on output (shape, dtype, ndim).

        Returns ``(spatial_shape_zyx, output_dtype, input_ndim)``. The
        input array is expected to be 3D ``(Z, Y, X)``; subclasses that
        consume 4D ``(C, Z, Y, X)`` should override this and pull off
        the channel axis themselves in ``process_chunk``.
        """
        from connectomics.data.io.io import get_vol_shape

        shape = tuple(get_vol_shape(self.config.input_path, dataset=self.config.input_dataset))
        if len(shape) == 3:
            spatial = tuple(int(v) for v in shape)
            ndim = 3
        elif len(shape) == 4:
            spatial = tuple(int(v) for v in shape[-3:])
            ndim = 4
        else:
            raise ValueError(
                f"Input volume must be 3D (Z, Y, X) or 4D (C, Z, Y, X); got shape {shape}."
            )

        # Output dtype: explicit override, else infer from a small read.
        if self.config.output_dtype is not None:
            out_dtype = np.dtype(self.config.output_dtype)
        else:
            out_dtype = self._peek_input_dtype()

        return spatial, out_dtype, ndim

    def process_chunk(self, chunk_data: np.ndarray, chunk: ChunkRef) -> np.ndarray:
        """Compute the output chunk from the input chunk crop.

        ``chunk_data`` is the per-chunk crop already trimmed to the
        inner (non-halo) region: shape equals ``chunk.shape``.
        Return value must be the same shape with output dtype.
        """
        raise NotImplementedError

    # ----------------------------------------------------------------- driver

    def run(self) -> str:
        """Execute the chunked workflow. Returns the output path."""
        self._input_shape, self._output_dtype, self._n_input_axes = self.setup()
        self._output_shape = self._input_shape

        grid = build_chunk_grid(self._output_shape, self.config.chunk_shape)
        out_path = Path(self.config.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        manifest_path = out_path.with_suffix(out_path.suffix + ".chunks.json")
        manifest_cfg = {
            "chunk_shape": list(self.config.chunk_shape),
            "overlap": int(self.config.overlap),
            "output_dtype": str(self._output_dtype),
            "output_shape": list(self._output_shape),
        }
        try:
            manifest = ResumeManifest.load_or_create(
                manifest_path, manifest_cfg, overwrite=self.config.overwrite
            )
        except ManifestConfigMismatch:
            raise

        if self.config.overwrite or not self._output_dataset_exists():
            self._allocate_output(self._output_shape, self._output_dtype)

        remaining = [c for c in grid if c.key not in manifest.completed]
        n_done = len(grid) - len(remaining)
        print(
            f"ChunkedProcessor: {len(grid)} chunks total, {n_done} already completed, "
            f"{len(remaining)} to process. chunk_shape={self.config.chunk_shape}, "
            f"parallel={self.config.parallel}, overlap={self.config.overlap}",
            flush=True,
        )
        if not remaining:
            return str(out_path)

        backpressure = max(2 * max(1, self.config.parallel), 2)
        t_start = time.time()
        n_processed = 0

        mp_ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=max(1, self.config.parallel), mp_context=mp_ctx
        ) as worker_pool, ThreadPoolExecutor(max_workers=1) as writer_pool:
            in_flight: list = []
            for chunk in remaining:
                crop = self._read_chunk_with_halo(chunk)
                worker_future = worker_pool.submit(_worker_process_chunk, self, crop, chunk)
                writer_future = writer_pool.submit(
                    self._await_and_write, worker_future, chunk, manifest, len(grid)
                )
                in_flight.append(writer_future)
                if len(in_flight) >= backpressure:
                    # block until at least one write completes, surfacing errors.
                    in_flight = self._drain_one(in_flight)
                n_processed += 1
            # drain remaining
            while in_flight:
                in_flight = self._drain_one(in_flight)

        elapsed = time.time() - t_start
        rate = n_processed / max(elapsed, 1e-9)
        print(
            f"  done in {elapsed:.1f}s ({rate:.2f} chunks/s, {n_processed} processed)",
            flush=True,
        )
        return str(out_path)

    # ----------------------------------------------------------- internals

    def _drain_one(self, futures: list) -> list:
        # Find the first completed future and raise/wait on it.
        first = futures[0]
        first.result()  # surfaces any exception
        return futures[1:]

    def _await_and_write(
        self,
        worker_future,
        chunk: ChunkRef,
        manifest: ResumeManifest,
        total: int,
    ) -> None:
        try:
            arr = worker_future.result()
        except Exception as exc:  # noqa: BLE001 - rethrow with context
            raise RuntimeError(f"Worker failed for chunk {chunk.key}: {exc}") from exc
        if arr.shape != chunk.shape:
            raise ValueError(
                f"Chunk {chunk.key}: process_chunk returned shape {arr.shape}, "
                f"expected {chunk.shape}."
            )
        if arr.dtype != self._output_dtype:
            arr = arr.astype(self._output_dtype, copy=False)
        self._write_chunk(chunk, arr)
        manifest.mark_completed(chunk.key)
        n_done = len(manifest.completed)
        print(f"  chunk {n_done}/{total} {chunk.key}: written", flush=True)

    def _peek_input_dtype(self) -> np.dtype:
        from connectomics.data.io.io import read_volume

        path = self.config.input_path
        if path.endswith(".h5") or ".h5" in path or path.endswith(".hdf5"):
            import h5py

            with h5py.File(path, "r") as f:
                return f[self.config.input_dataset].dtype
        # zarr / fallback: read a tiny slice
        sub = read_volume(path)
        return sub.dtype

    def _read_chunk_with_halo(self, chunk: ChunkRef) -> np.ndarray:
        """Read input crop for the chunk, with halo if configured.

        Returns the **inner** (non-halo) region trimmed to ``chunk.shape``
        for the default 0-halo case. For ``overlap > 0`` the returned crop
        is the halo-extended region; the subclass's ``process_chunk``
        receives this larger region and is responsible for emitting an
        output of ``chunk.shape``.
        """
        from connectomics.data.io.io import read_volume

        halo = (int(self.config.overlap),) * 3
        read_start, read_stop, _ = _resolve_halo_region(
            chunk, self._input_shape, halo=halo
        )
        # Read with read_volume; for h5/zarr it supports lazy slicing through
        # an explicit (Z, Y, X) slice tuple via dataset[...]; fall back to a
        # direct h5py read for HDF5 to avoid loading the full volume.
        path = self.config.input_path
        slc = tuple(slice(read_start[axis], read_stop[axis]) for axis in range(3))
        if path.endswith(".h5") or path.endswith(".hdf5"):
            import h5py

            with h5py.File(path, "r") as f:
                ds = f[self.config.input_dataset]
                if self._n_input_axes == 4:
                    return np.asarray(ds[(slice(None), *slc)])
                return np.asarray(ds[slc])
        if ".zarr" in path:
            from ..data.io.io import _split_zarr_path  # type: ignore[attr-defined]
            import zarr  # type: ignore[import]

            zarr_path, sub = _split_zarr_path(path)
            store = zarr.open(zarr_path, mode="r")
            arr = store[sub] if sub else store
            if self._n_input_axes == 4:
                return np.asarray(arr[(slice(None), *slc)])
            return np.asarray(arr[slc])
        # other formats: read full volume and slice (fine for tests, slow for
        # production volumes — but tests are the only place we hit this).
        vol = read_volume(path)
        if vol.ndim == 4:
            return np.asarray(vol[(slice(None), *slc)])
        return np.asarray(vol[slc])

    def _output_dataset_exists(self) -> bool:
        path = self.config.output_path
        if path.endswith(".h5") or path.endswith(".hdf5"):
            if not Path(path).exists():
                return False
            import h5py

            with h5py.File(path, "r") as f:
                return self.config.output_dataset in f
        if ".zarr" in path:
            from ..data.io.io import _split_zarr_path  # type: ignore[attr-defined]
            import zarr  # type: ignore[import]

            zarr_path, sub = _split_zarr_path(path)
            if not Path(zarr_path).exists():
                return False
            try:
                store = zarr.open(zarr_path, mode="r")
                if sub:
                    return sub in store
                return True
            except Exception:  # noqa: BLE001
                return False
        return Path(path).exists()

    def _allocate_output(self, shape: Tuple[int, int, int], dtype: np.dtype) -> None:
        path = self.config.output_path
        chunks = self.config.h5_chunks or tuple(min(s, 256) for s in shape)
        if path.endswith(".h5") or path.endswith(".hdf5"):
            import h5py

            mode = "w" if self.config.overwrite or not Path(path).exists() else "a"
            with h5py.File(path, mode) as f:
                if self.config.output_dataset in f:
                    del f[self.config.output_dataset]
                kwargs: dict[str, Any] = {"shape": shape, "dtype": dtype, "chunks": chunks}
                if self.config.compression:
                    kwargs["compression"] = self.config.compression
                    if self.config.compression == "gzip":
                        kwargs["compression_opts"] = int(self.config.compression_level)
                f.create_dataset(self.config.output_dataset, **kwargs)
            return
        if ".zarr" in path:
            from ..data.io.io import _split_zarr_path  # type: ignore[attr-defined]
            import zarr  # type: ignore[import]

            zarr_path, sub = _split_zarr_path(path)
            Path(zarr_path).parent.mkdir(parents=True, exist_ok=True)
            if sub:
                root = zarr.open_group(zarr_path, mode="a")
                if sub in root and self.config.overwrite:
                    del root[sub]
                root.create_dataset(
                    sub, shape=shape, dtype=dtype, chunks=chunks, overwrite=self.config.overwrite
                )
            else:
                zarr.open(zarr_path, mode="w", shape=shape, dtype=dtype, chunks=chunks)
            return
        raise ValueError(
            f"Unsupported output format for ChunkedProcessor: {path!r}. Use .h5 or .zarr."
        )

    def _write_chunk(self, chunk: ChunkRef, arr: np.ndarray) -> None:
        path = self.config.output_path
        slc = chunk.slices
        if path.endswith(".h5") or path.endswith(".hdf5"):
            import h5py

            with h5py.File(path, "a") as f:
                ds = f[self.config.output_dataset]
                ds[slc] = arr
                f.flush()
            return
        if ".zarr" in path:
            from ..data.io.io import _split_zarr_path  # type: ignore[attr-defined]
            import zarr  # type: ignore[import]

            zarr_path, sub = _split_zarr_path(path)
            store = zarr.open(zarr_path, mode="a")
            arr_ds = store[sub] if sub else store
            arr_ds[slc] = arr
            return
        raise ValueError(f"Unsupported output format: {path!r}")


def _worker_process_chunk(
    processor: ChunkedProcessor, chunk_data: np.ndarray, chunk: ChunkRef
) -> np.ndarray:
    """Top-level worker function for ProcessPoolExecutor.

    Defined at module top level so it's pickle-able under spawn. The
    processor is sent to the worker as the first arg (also pickled),
    which is why ChunkedProcessor instances must be picklable.
    """
    return processor.process_chunk(chunk_data, chunk)


def _resolve_halo_region(chunk: ChunkRef, input_shape, *, halo):
    # local import to avoid circular-ish imports if subclasses pull halo too
    from .halo import resolve_halo_region

    return resolve_halo_region(chunk, input_shape, halo=halo)

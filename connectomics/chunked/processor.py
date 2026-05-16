"""ChunkedProcessor: parent class for naively-parallel chunked workflows.

Architecture (deliberately flat):

- Build a chunk grid over the input volume.
- Pre-allocate the output H5/zarr once, up front.
- Spawn a ``ProcessPoolExecutor`` of CPU workers. Each worker picks up a
  chunk and runs the **full pipeline** for it: read its own input crop,
  compute, and write its slice into the shared output file.
- Workers are independent: no data flows through the main process except
  the chunk key on completion. Output writes hit disjoint slices, so:

  * For **zarr**, each chunk is its own file on disk -- concurrent writes
    from multiple workers are natively safe.
  * For **HDF5**, libhdf5's built-in file locking (default since 1.10)
    serializes open-for-write across workers. Each worker's open+write+close
    is ~50ms; for hundreds of chunks the total serialization overhead is
    negligible against compute.

- Resume: a JSON sidecar ``<output>.chunks.json`` records completed chunk
  keys; restarts skip them.

Subclasses override :meth:`setup` (output allocation hook) and
:meth:`process_chunk` (pure per-chunk compute).
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np

# libhdf5's built-in file lock returns EAGAIN under contention (instead of
# blocking), which trips concurrent worker writes even when they target
# disjoint storage chunks. Disable it here; we serialize h5 writes ourselves
# via an explicit fcntl.flock on a sidecar lock file (see _write_chunk).
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

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
        manifest = ResumeManifest.load_or_create(
            manifest_path, manifest_cfg, overwrite=self.config.overwrite
        )

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

        t_start = time.time()
        mp_ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=max(1, self.config.parallel), mp_context=mp_ctx
        ) as pool:
            futures = {pool.submit(_worker_pipeline, self, chunk): chunk for chunk in remaining}
            for fut in as_completed(futures):
                chunk = futures[fut]
                try:
                    key = fut.result()
                except Exception as exc:  # noqa: BLE001
                    raise RuntimeError(f"Worker failed for chunk {chunk.key}: {exc}") from exc
                manifest.mark_completed(key)
                done = len(manifest.completed)
                print(f"  chunk {done}/{len(grid)} {key}: written", flush=True)

        elapsed = time.time() - t_start
        rate = len(remaining) / max(elapsed, 1e-9)
        print(
            f"  done in {elapsed:.1f}s ({rate:.2f} chunks/s, {len(remaining)} processed)",
            flush=True,
        )
        return str(out_path)

    # ----------------------------------------------------------- internals

    def _run_one_chunk(self, chunk: ChunkRef) -> str:
        """Worker-side pipeline for one chunk: read input, compute, write output."""
        crop = self._read_chunk_with_halo(chunk)
        arr = self.process_chunk(crop, chunk)
        if arr.shape != chunk.shape:
            raise ValueError(
                f"Chunk {chunk.key}: process_chunk returned shape {arr.shape}, "
                f"expected {chunk.shape}."
            )
        if arr.dtype != self._output_dtype:
            arr = arr.astype(self._output_dtype, copy=False)
        self._write_chunk(chunk, arr)
        return chunk.key

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
        # Output storage chunks must align 1:1 with processor chunks so that
        # concurrent workers writing to disjoint processor chunks never touch
        # the same storage-chunk file (zarr) or storage-chunk page (h5). With
        # output_chunks == chunk_shape, each worker writes exactly one output
        # storage chunk and there is no shared state between writers.
        chunks = self.config.h5_chunks or tuple(
            max(1, min(int(s), int(c))) for s, c in zip(shape, self.config.chunk_shape)
        )
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
            import fcntl

            import h5py

            # External lock: only one worker may open the file in 'a' mode at a
            # time. With output storage chunks aligned to processor chunks the
            # writes themselves go to disjoint storage, but libhdf5 metadata
            # (B-tree, chunk index) is not multi-process safe regardless.
            lock_path = path + ".lock"
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
                with h5py.File(path, "a") as f:
                    ds = f[self.config.output_dataset]
                    ds[slc] = arr
                    f.flush()
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                os.close(lock_fd)
            return
        if ".zarr" in path:
            # Zarr stores each chunk as its own file; with output_chunks
            # aligned to processor chunks, concurrent worker writes target
            # disjoint files and require no locking.
            from ..data.io.io import _split_zarr_path  # type: ignore[attr-defined]
            import zarr  # type: ignore[import]

            zarr_path, sub = _split_zarr_path(path)
            store = zarr.open(zarr_path, mode="a")
            arr_ds = store[sub] if sub else store
            arr_ds[slc] = arr
            return
        raise ValueError(f"Unsupported output format: {path!r}")


def _worker_pipeline(processor: ChunkedProcessor, chunk: ChunkRef) -> str:
    """Top-level worker function for ProcessPoolExecutor.

    Defined at module top level so it's pickle-able under spawn. Workers
    run the full per-chunk pipeline (read input, compute, write output)
    and return only ``chunk.key`` to the main process. The processor
    instance is sent to the worker once via pickle and must be
    picklable; subclass state should live on ``config`` only.
    """
    return processor._run_one_chunk(chunk)


def _resolve_halo_region(chunk: ChunkRef, input_shape, *, halo):
    # local import to avoid circular-ish imports if subclasses pull halo too
    from .halo import resolve_halo_region

    return resolve_halo_region(chunk, input_shape, halo=halo)

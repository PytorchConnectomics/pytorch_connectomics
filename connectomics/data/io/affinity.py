"""Chunk-aligned HDF5 affinity thresholding."""

import numbers
import operator
import os
from collections.abc import Sequence
from typing import Optional

import h5py
import numpy as np

from connectomics.utils.shared_buffer import (
    SharedArray,
    attach,
    process_parallel_available,
    run_process_tasks,
)


def _blocks(x0, x1, step):
    while x0 < x1:
        end = min((x0 // step + 1) * step, x1)
        yield x0, end
        x0 = end


def _read_blocks(filename, dataset, bbox, channels, threshold, blocks, out):
    x0, _, y0, y1, z0, z1 = bbox
    with h5py.File(filename, "r") as stream:
        ds = stream[dataset]
        dtype = ds.dtype.newbyteorder("=")
        lut = (
            np.arange(65536, dtype=np.uint16).view(np.float16) > threshold
            if dtype == np.float16
            else None
        )
        for start, end in blocks:
            raw = np.empty((1, end - start, y1 - y0, z1 - z0), dtype)
            for k, channel in enumerate(channels):
                ds.read_direct(raw, np.s_[channel : channel + 1, start:end, y0:y1, z0:z1])
                target = out[k : k + 1, start - x0 : end - x0]
                if lut is not None:
                    target[:] = lut[raw.view(np.uint16)]
                else:
                    np.greater(raw, threshold, out=target)


def _reader_worker(task):
    filename, dataset, bbox, channels, threshold, blocks, path, shape = task
    buf = attach(path, shape, np.bool_, writable=True)
    try:
        _read_blocks(filename, dataset, bbox, channels, threshold, blocks, buf.array)
    finally:
        buf.close()


def read_hdf5_affinity_threshold(
    filename: str,
    threshold,
    dataset: Optional[str] = None,
    channels: Optional[Sequence[int]] = None,
    bbox: Optional[Sequence[int]] = None,
    num_workers: Optional[int] = None,
    buffer_dir: Optional[str] = None,
) -> np.ndarray:
    """Read affinities and compare with the same native NumPy threshold semantics.

    Channels preserve NumPy indexing order, duplicates and negative indices.
    The bounding box is ``(x0, x1, y0, y1, z0, z1)``. Float16 comparisons
    include all bit patterns, with no range or finiteness restriction.

    Processes avoid h5py's global lock and read chunk-aligned X blocks. Shared
    output storage uses ``buffer_dir`` (default: writable /dev/shm, else the
    temporary directory) and counts toward memory limits. Scripts must guard
    process entry with ``if __name__ == "__main__":`` for forkserver/spawn.
    The returned C-contiguous bool array can feed
    ``connected_components_affinity_3d_slab`` directly (which owns a copy in
    process mode). One-worker and unavailable-process cases use private storage.
    """
    scalar = np.asarray(threshold)
    if (
        scalar.ndim != 0
        or scalar.dtype.kind not in "buif"
        or not isinstance(scalar.item(), numbers.Real)
    ):
        raise TypeError("threshold must be a real scalar")
    workers = num_workers
    if workers is None:
        workers = (
            len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else os.cpu_count() or 1
        )
    if workers < 1:
        raise ValueError("num_workers must be positive")
    with h5py.File(filename, "r") as stream:
        dataset = list(stream.keys())[0] if dataset is None else dataset
        ds = stream[dataset]
        if ds.ndim != 4:
            raise ValueError("Expected a four-dimensional affinity dataset")
        selected = tuple(operator.index(c) for c in ((0, 1, 2) if channels is None else channels))
        if any(c < -ds.shape[0] or c >= ds.shape[0] for c in selected):
            raise IndexError("Channel index out of range")
        selected = tuple(c % ds.shape[0] for c in selected)
        box = (0, ds.shape[1], 0, ds.shape[2], 0, ds.shape[3]) if bbox is None else tuple(bbox)
        if (
            len(box) != 6
            or any(not isinstance(v, numbers.Integral) for v in box)
            or any(
                not 0 <= lo <= hi <= size for lo, hi, size in zip(box[::2], box[1::2], ds.shape[1:])
            )
        ):
            raise ValueError("Invalid bounding box")
        step = ds.chunks[1] if ds.chunks else 32
    shape = (len(selected), *(hi - lo for lo, hi in zip(box[::2], box[1::2])))
    blocks = list(_blocks(box[0], box[1], step))
    if not all(shape) or workers == 1 or len(blocks) <= 1 or not process_parallel_available():
        out = np.empty(shape, np.bool_)
        if all(shape):
            _read_blocks(filename, dataset, box, selected, threshold, blocks, out)
        return out
    workers = min(workers, len(blocks))
    buf = SharedArray.create(shape, np.bool_, buffer_dir)
    try:
        tasks = [
            (str(filename), dataset, box, selected, threshold, blocks[i::workers], buf.path, shape)
            for i in range(workers)
        ]
        run_process_tasks(_reader_worker, tasks, workers, preload=[__name__])
        buf.release_path()
        return buf.array
    except BaseException:
        buf.close()
        raise

"""Slab-wise connected components on directed short-range affinities."""

import mmap
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from queue import Queue
from threading import Lock
from typing import Optional

import cc3d
import fastremap
import numpy as np
from numba import njit, prange

from connectomics.data.processing import get_seg_type
from connectomics.decoding.utils import cast2dtype
from connectomics.utils.shared_buffer import (
    SharedArray,
    attach,
    process_parallel_available,
    run_process_tasks,
)

from .segmentation_kernels import _affinity_foreground_mask

UINT32_LIMIT = 1 << 32
SLAB_VOXEL_LIMIT = (1 << 32) - 1
_fork_launch_lock = Lock()
_fork_state: tuple  # Assigned by the initializer only in child processes.


@njit(nogil=True, cache=True)
def _build_graph(hard, x0, x1, eo):
    graph = np.zeros((x1 - x0, hard.shape[2], hard.shape[3]), np.uint8)
    for x in range(x1 - x0):
        for y in range(graph.shape[1]):
            for z in range(graph.shape[2]):
                if x + 1 < graph.shape[0] and hard[0, x0 + x + eo, y, z]:
                    graph[x, y, z] |= 1
                    graph[x + 1, y, z] |= 2
                if y + 1 < graph.shape[1] and hard[1, x0 + x, y + eo, z]:
                    graph[x, y, z] |= 4
                    graph[x, y + 1, z] |= 8
                if z + 1 < graph.shape[2] and hard[2, x0 + x, y, z + eo]:
                    graph[x, y, z] |= 16
                    graph[x, y, z + 1] |= 32
    return graph


@njit(nogil=True, cache=True)
def _compact_first_appearance(local, out, table):
    count = 0
    source = local.reshape(-1)
    target = out.reshape(-1)
    max_local = 0
    for i in range(source.size):
        v = np.int64(source[i])
        max_local = max(max_local, v)
        assert 0 <= v < len(table)
        if v:
            if table[v] == 0:
                count += 1
                table[v] = count
            target[i] = table[v]
        else:
            target[i] = 0
    # cc3d ids are dense; clear the used table prefix, not the volume.
    for i in range(max_local + 1):
        table[i] = 0
    return count


@njit(parallel=True, cache=True)
def _add_offset(seg, offset):
    flat = seg.reshape(-1)
    for i in prange(flat.size):
        if flat[i]:
            flat[i] += offset


@njit(nogil=True, cache=True)
def _index(value, base, size):
    idx = np.int64(value) - np.int64(base)
    assert 0 <= idx < size, "Label outside table (possible truncation)"
    return idx


@njit(nogil=True, cache=True)
def _root(parent, value, base):
    value = np.int64(value)
    while True:
        idx = _index(value, base, len(parent))
        p = np.int64(parent[idx])
        if p == 0 or p == value:
            return value
        pi = _index(p, base, len(parent))
        pp = np.int64(parent[pi])
        if pp:
            parent[idx] = pp
        value = p


@njit(nogil=True, cache=True)
def _union_plane(left, right, edges, parent, base=0):
    for y in range(edges.shape[0]):
        for z in range(edges.shape[1]):
            if edges[y, z]:
                a, b = left[y, z], right[y, z]
                assert a != 0 and b != 0
                ra, rb = _root(parent, a, base), _root(parent, b, base)
                if ra != rb:
                    parent[_index(max(ra, rb), base, len(parent))] = min(ra, rb)


@njit(nogil=True, cache=True)
def _compress_roots(parent, base=0):
    for i in range(len(parent)):
        parent[i] = _root(parent, np.int64(i) + np.int64(base), base)


@njit(nogil=True, cache=True)
def _renumber_by_roots(seg, parent, base=0):
    newid = np.zeros(len(parent), dtype=parent.dtype)
    count = 0
    flat = seg.reshape(-1)
    for i in range(flat.size):
        v = flat[i]
        if v:
            root = parent[_index(v, base, len(parent))]
            ri = _index(root, base, len(parent))
            if newid[ri] == 0:
                count += 1
                newid[ri] = count
            flat[i] = newid[ri]
    return count


@njit(nogil=True, cache=True)
def _final_ids_from_roots(parent):
    """After _compress_roots: parent[i] = root of local id i; rewrite to final ids.

    Local ids follow global first-C-order appearance (slabs are X-contiguous and
    compacted in C order, offsets increase with X), and unions keep the smaller
    root, so each root is its component's earliest local id. Ranking roots by id
    therefore equals first-appearance numbering (fastremap.renumber order).
    """
    rank = np.zeros(len(parent), parent.dtype)
    count = 0
    for i in range(1, len(parent)):
        if parent[i] == i:
            count += 1
            rank[i] = count
    for i in range(1, len(parent)):
        parent[i] = rank[parent[i]]
    return count


@njit(parallel=True, cache=True)
def _apply_final_ids(seg, final, out):
    """out[v] = final[seg[v]] (final[0] == 0); out may alias seg."""
    for i in prange(seg.size):
        out[i] = final[seg[i]]


def _slab_foreground(hard, x0, x1, eo):
    """Rows x0:x1 of _affinity_foreground_mask(hard, eo), from a 1-row halo."""
    lo, hi = max(0, x0 - 1), min(hard.shape[1], x1 + 1)
    fg = _affinity_foreground_mask(hard[:, lo:hi], eo)
    return fg[x0 - lo : x1 - lo]


def _local_slab(hard, seg, eo, bounds, table):
    x0, x1 = bounds
    fg = _slab_foreground(hard, x0, x1, eo)
    graph = _build_graph(hard, x0, x1, eo)
    labels = np.ascontiguousarray(cc3d.color_connectivity_graph(graph, connectivity=6))
    del graph
    labels *= fg
    return int(_compact_first_appearance(labels, seg[x0:x1], table))


def _slab_worker(task):
    hard_path, hard_shape, seg_path, seg_shape, eo, bounds = task
    hard = attach(hard_path, hard_shape, np.bool_)
    try:
        seg = attach(seg_path, seg_shape, np.uint32, writable=True)
        try:
            process = mp.current_process()
            cache = getattr(process, "_pytc_compaction_tables", None)
            if cache is None:
                cache = {}
                process._pytc_compaction_tables = cache
            length = (bounds[1] - bounds[0]) * seg_shape[1] * seg_shape[2] + 1
            if length not in cache:
                cache[length] = np.zeros(length, np.uint32)
            table = cache[length]
            return _local_slab(hard.array, seg.array, eo, bounds, table)
        finally:
            seg.close()
    finally:
        hard.close()


def _init_fork_worker(hard, seg, eo, table_size):
    # Written only in children: simultaneous parent calls never share call state.
    global _fork_state
    _fork_state = (hard, seg, eo, np.zeros(table_size, np.uint32))


def _fork_worker(bounds):
    hard, seg, eo, table = _fork_state
    return _local_slab(hard, seg, eo, bounds, table)


def _precompile_workers(hard):
    """Exercise every worker kernel signature before inheriting Numba state."""
    if hard.flags.c_contiguous:
        tiny = np.zeros((3, 2, 2, 2), bool)
    elif hard.flags.f_contiguous:
        tiny = np.zeros((3, 2, 2, 2), bool, order="F")
    else:
        tiny = np.zeros((3, 2, 2, 4), bool)[:, :, :, ::2]
    tiny.flags.writeable = hard.flags.writeable
    _build_graph(tiny, 0, 2, 0)
    out = np.zeros((2, 2, 2), np.uint32)
    table = np.zeros(9, np.uint32)
    # cc3d chooses its unsigned label width from the graph size.
    for dtype in (np.uint8, np.uint16, np.uint32, np.uint64):
        _compact_first_appearance(np.zeros(out.shape, dtype), out, table)


def _run_fork_tasks(hard, seg, eo, ranges, workers):
    table_size = (ranges[0][1] - ranges[0][0]) * seg.shape[1] * seg.shape[2] + 1
    executor = ProcessPoolExecutor(
        max_workers=workers,
        mp_context=mp.get_context("fork"),
        initializer=_init_fork_worker,
        initargs=(hard, seg, eo, table_size),
    )
    futures = []
    try:
        # Serialize warm-up and fork launches, not worker execution. Concurrent
        # launches can overlap library at-fork hooks or inherit an active compiler.
        with _fork_launch_lock:
            _precompile_workers(hard)
            futures = [executor.submit(_fork_worker, bounds) for bounds in ranges]
        return [future.result() for future in futures]
    finally:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)


def _resolve_parallel(parallel, workers):
    if parallel not in ("auto", "processes", "fork", "forkserver", "threads"):
        raise ValueError("Invalid parallel mode")
    if parallel == "auto":
        parallel = "processes" if process_parallel_available() and workers > 1 else "threads"
    fork_available = sys.platform.startswith("linux") and "fork" in mp.get_all_start_methods()
    if parallel == "processes":
        parallel = "fork" if fork_available else "forkserver"
    if parallel in ("fork", "forkserver") and not process_parallel_available():
        raise RuntimeError("Process parallelism is unavailable in this process")
    if parallel == "fork" and not fork_available:
        raise RuntimeError("Fork parallelism requires Linux with the fork start method")
    return parallel


def _finish(seg):
    seg = fastremap.refit(seg)
    if seg.size and not np.any(seg == 0):
        seg[0, 0, 0] = 0
    return cast2dtype(seg)


def _layout(shape, edge_offset, num_workers, slab_size, parallel):
    if len(shape) != 4 or shape[0] < 3:
        raise ValueError("Expected (>=3, X, Y, Z) affinities")
    if edge_offset not in (0, 1):
        raise ValueError("Invalid edge_offset or parallel mode")
    workers = num_workers
    if workers is None:
        workers = (
            len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else os.cpu_count() or 1
        )
    if workers < 1 or (slab_size is not None and slab_size < 1):
        raise ValueError("num_workers and slab_size must be positive")
    x, y, z = shape[1:]
    slab = (
        slab_size
        if slab_size is not None
        else min(max(1, x), max(1, min(SLAB_VOXEL_LIMIT, 2**28) // max(1, y * z)))
    )
    if slab * y * z > SLAB_VOXEL_LIMIT:
        raise ValueError("Slab exceeds the voxel limit")
    ranges = [(a, min(x, a + slab)) for a in range(0, x, slab)]
    workers = min(workers, max(1, len(ranges)))
    return ranges, workers, _resolve_parallel(parallel, workers)


def connected_components_affinity_3d_slab(
    hard_aff: np.ndarray,
    edge_offset: int = 0,
    num_workers: Optional[int] = None,
    slab_size: Optional[int] = None,
    parallel: str = "auto",
    finalize: bool = True,
    buffer_dir: Optional[str] = None,
) -> np.ndarray:
    """Label hard affinities in C-order, matching converged parallel-numba labels.

    Only the first three channels are used. "auto" selects processes when available
    with more than one effective worker, otherwise threads. "processes" uses "fork"
    on Linux where available, otherwise "forkserver" (with spawn fallback).
    Explicit "fork" requires Linux and a non-daemonic caller.

    On Linux the default process mode forks the calling process. Forking a process
    whose other threads hold locks can deadlock children. Children only run
    precompiled nogil Numba kernels and cc3d for labeling; use parallel="forkserver"
    or "threads" when that is a concern. Fork inherits hard affinities without a
    copy and stores uint32 labels in an anonymous shared mapping. Forkserver/spawn
    copies hard affinities into shared files in buffer_dir, alongside label storage;
    scripts using that mode need a main-module guard.
    With ``finalize=False``, return labels before refit and the background-corner rule.
    """
    return _connected_components_slab(
        hard_aff, edge_offset, num_workers, slab_size, parallel, finalize, buffer_dir
    )


def _connected_components_slab(
    hard,
    edge_offset=0,
    num_workers=None,
    slab_size=None,
    parallel="auto",
    finalize=True,
    buffer_dir=None,
):
    """Consume an array or transfer ownership of a SharedArray on every exit path."""
    hard_buffer = hard if isinstance(hard, SharedArray) else None
    label_buffer = None
    anonymous = None
    try:
        source = hard.array if hard_buffer is not None else hard
        ranges, workers, mode = _layout(source.shape, edge_offset, num_workers, slab_size, parallel)
        shape = source.shape[1:]
        if not all(shape):
            empty = np.zeros(shape, np.uint32)
            return _finish(empty) if finalize else empty
        if mode == "fork":
            hard = np.asarray(source[:3], dtype=np.bool_)
            anonymous = mmap.mmap(-1, int(np.prod(shape)) * np.dtype(np.uint32).itemsize)
            seg = np.frombuffer(anonymous, np.uint32).reshape(shape)
            counts = _run_fork_tasks(hard, seg, edge_offset, ranges, workers)
        elif mode == "forkserver":
            if hard_buffer is None:
                hard_buffer = SharedArray.create((3, *shape), np.bool_, buffer_dir)
                hard_buffer.array[:] = source[:3]
            hard = hard_buffer.array
            label_buffer = SharedArray.create(shape, np.uint32, buffer_dir)
            seg = label_buffer.array
            tasks = [
                (hard_buffer.path, hard.shape, label_buffer.path, shape, edge_offset, b)
                for b in ranges
            ]
            counts = run_process_tasks(_slab_worker, tasks, workers, preload=[__name__])
            hard_buffer.release_path()
            label_buffer.release_path()
        else:
            hard = np.asarray(source[:3], dtype=np.bool_)
            seg = np.empty(shape, np.uint32)
            tables = Queue()
            for _ in range(workers):
                tables.put(
                    np.zeros((ranges[0][1] - ranges[0][0]) * shape[1] * shape[2] + 1, np.uint32)
                )

            def work(bounds):
                table = tables.get()
                try:
                    return _local_slab(hard, seg, edge_offset, bounds, table)
                finally:
                    tables.put(table)

            if workers == 1:
                counts = list(map(work, ranges))
            else:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    counts = list(pool.map(work, ranges))
        total = sum(counts)
        dtype = np.dtype(np.uint64 if total >= UINT32_LIMIT else np.uint32)
        if seg.dtype != dtype:
            seg = seg.astype(dtype)
        offset = 0
        for bounds, count in zip(ranges, counts):
            _add_offset(seg[bounds[0] : bounds[1]], np.int64(offset))
            offset += count
        parent = np.zeros(total + 1, dtype)
        for b, _ in ranges[1:]:
            _union_plane(seg[b - 1], seg[b], hard[0, b - 1 + edge_offset], parent)
        hard = None
        source = None
        if hard_buffer is not None:
            hard_buffer.close()
        _compress_roots(parent)
        components = int(_final_ids_from_roots(parent))
        has_background = any(np.any(seg[a:b] == 0) for a, b in ranges)
        target = np.dtype(get_seg_type(components)) if finalize and has_background else dtype
        out = seg if target == seg.dtype else np.empty(shape, target)
        _apply_final_ids(seg.reshape(-1), parent, out.reshape(-1))
        result = _finish(out) if finalize and not has_background else out
        if label_buffer is not None and np.shares_memory(result, label_buffer.array):
            label_buffer.release_path()
            label_buffer = None  # The returned array now owns the mapping lifetime.
        return result
    finally:
        if hard_buffer is not None:
            hard_buffer.close()
        if label_buffer is not None:
            label_buffer.close()
        if anonymous is not None:
            # frombuffer keeps the mapping alive while any returned view exists.
            try:
                anonymous.close()
            except BufferError:
                pass

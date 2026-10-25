"""Exact-output gates for the standalone affinity CC experiment."""

import sys
import warnings

import cc3d
import fastremap
import numpy as np
import pytest

from connectomics.decoding.decoders import affinity_cc_slab as fc
from connectomics.decoding.decoders.segmentation import decode_affinity_cc
from connectomics.utils.shared_buffer import process_parallel_available

PROCESS_MODES = [
    pytest.param(
        "fork",
        marks=pytest.mark.skipif(
            not sys.platform.startswith("linux") or "fork" not in fc.mp.get_all_start_methods(),
            reason="Linux fork unavailable",
        ),
    ),
    "forkserver",
]


def assert_partition_equal(actual, expected):
    # Canonicalize by first C-order appearance, preserving background.
    def canonical(labels):
        flat = labels.ravel()
        _, first, inverse = np.unique(flat, return_index=True, return_inverse=True)
        order = np.argsort(first)
        ranks = np.empty(order.size, dtype=np.int64)
        ranks[order] = np.arange(1, order.size + 1)
        ranks[np.unique(flat) == 0] = 0
        return ranks[inverse]

    np.testing.assert_array_equal(actual == 0, expected == 0)
    np.testing.assert_array_equal(canonical(actual), canonical(expected))


def numba_reference(data, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        return decode_affinity_cc(data, backend="numba", **kwargs)


def check(hard, eo, slab, workers, parallel="threads"):
    actual = fc.connected_components_affinity_3d_slab(
        hard, eo, slab_size=slab, num_workers=workers, parallel=parallel
    )
    ref = numba_reference(hard.astype(np.float32), threshold=0.5, edge_offset=eo)
    assert actual.dtype == ref.dtype
    np.testing.assert_array_equal(actual, ref)
    serial = decode_affinity_cc(
        hard.astype(np.float32), threshold=0.5, backend="numba_serial", edge_offset=eo
    )
    assert_partition_equal(actual, serial)


def edge(hard, a, b, eo):
    axis = next(i for i in range(3) if a[i] != b[i])
    lo = list(min(a, b))
    lo[axis] += eo
    hard[(axis,) + tuple(lo)] = True


def fixture(kind, eo):
    hard = np.zeros((3, 8, 7, 6), bool)
    if kind == "full":
        hard[:] = True
    elif kind == "singleton":
        hard[0, -1 if eo == 0 else 0, 2, 2] = True
    elif kind == "faces":
        for axis in range(3):
            index = [axis, slice(None), slice(None), slice(None)]
            index[axis + 1] = -1 if eo == 0 else 0
            hard[tuple(index)] = True
    elif kind == "snake":
        points = [
            (x, y, 2) for y in range(7) for x in (range(8) if y % 2 == 0 else range(7, -1, -1))
        ]
        for a, b in zip(points, points[1:]):
            edge(hard, a, b, eo)
    elif kind in ("ring", "u"):
        for x in range(7):
            for y in (1, 5):
                edge(hard, (x, y, 2), (x + 1, y, 2), eo)
        for y in range(1, 5):
            edge(hard, (7, y, 2), (7, y + 1, 2), eo)
            if kind == "ring":
                edge(hard, (0, y, 2), (0, y + 1, 2), eo)
    return hard


@pytest.mark.parametrize("eo", [0, 1])
@pytest.mark.parametrize("slab", [1, 2, 3, 8])
@pytest.mark.parametrize("workers", [1, 3])
@pytest.mark.parametrize("kind", ["empty", "full", "singleton", "faces", "snake", "ring", "u"])
def test_structured(kind, workers, slab, eo):
    check(fixture(kind, eo), eo, slab, workers)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("shape", [(3, 17, 9, 11), (3, 40, 23, 7)])
@pytest.mark.parametrize("p", [0.3, 0.55, 0.7])
@pytest.mark.parametrize("eo", [0, 1])
def test_random(seed, shape, p, eo):
    hard = np.random.default_rng(seed).random(shape) < p
    # Full matrix on the random fixtures, including repeated table reuse.
    for slab in (1, 2, 3, shape[1]):
        for workers in (1, 3):
            check(hard, eo, slab, workers)


@pytest.mark.parametrize("eo", [0, 1])
@pytest.mark.parametrize("p", [0.05, 0.5, 0.95])
def test_slab_foreground_matches_global_mask(eo, p):
    from connectomics.decoding.decoders.segmentation_kernels import _affinity_foreground_mask

    hard = np.random.default_rng(3).random((3, 11, 7, 5)) < p
    full = _affinity_foreground_mask(hard, eo)
    for slab in (1, 2, 4, 11):
        for x0 in range(0, 11, slab):
            x1 = min(x0 + slab, 11)
            np.testing.assert_array_equal(fc._slab_foreground(hard, x0, x1, eo), full[x0:x1])


@pytest.mark.parametrize("axis", range(3))
@pytest.mark.parametrize("eo", [0, 1])
def test_bits(axis, eo):
    hard = np.zeros((3, 4, 4, 4), bool)
    a, b = [1, 1, 1], [1, 1, 1]
    b[axis] += 1
    edge(hard, tuple(a), tuple(b), eo)
    graph = fc._build_graph(hard, 0, 4, eo)
    assert graph[tuple(a)] == 1 << (2 * axis)
    assert graph[tuple(b)] == 1 << (2 * axis + 1)
    labels = cc3d.color_connectivity_graph(graph, connectivity=6)
    assert labels[tuple(a)] == labels[tuple(b)]
    for slab in (2, 4):
        check(hard, eo, slab, 1)


def test_above_uint32_roots():
    base = 2**32 - 3
    seg = np.array([0, 1, 8], np.uint64).reshape(1, 1, 3)
    fc._add_offset(seg, np.int64(2**32 + 5))
    np.testing.assert_array_equal(seg.ravel(), [0, 2**32 + 6, 2**32 + 13])
    parent = np.zeros(64, np.uint64)
    left = np.array([[base + 1]], np.uint64)
    right = np.array([[base + 40]], np.uint64)
    fc._union_plane(left, right, np.ones((1, 1), bool), parent, base)
    fc._compress_roots(parent, base)
    assert parent[1] == parent[40] == base + 1
    assert parent[2] == base + 2
    original = np.array([base + 40, base + 1, base + 2], np.uint64).reshape(1, 1, 3)
    resolved = parent[(original - np.uint64(base)).astype(np.int64)]
    expected, _ = fastremap.renumber(resolved, preserve_zero=True)
    fc._renumber_by_roots(original, parent, base)
    np.testing.assert_array_equal(original, expected)
    np.testing.assert_array_equal(original.ravel(), [1, 1, 2])
    bad = np.array([[[base + 64]]], np.uint64)
    with pytest.raises(AssertionError):
        fc._renumber_by_roots(bad, parent, base)
    with pytest.raises(AssertionError):
        fc._union_plane(
            left, np.array([[base + 64]], np.uint64), np.ones((1, 1), bool), parent, base
        )


def test_first_appearance():
    seg = np.array([0, 7, 7, 3, 0, 9, 3, 1], np.uint32).reshape(2, 2, 2)
    expected, _ = fastremap.renumber(seg, preserve_zero=True)
    parent = np.zeros(10, np.uint32)
    fc._compress_roots(parent)
    fc._renumber_by_roots(seg, parent)
    np.testing.assert_array_equal(seg, expected)


@pytest.mark.parametrize("seed", range(5))
def test_final_ids_match_first_appearance_renumber(seed):
    # Labels in first-appearance order (as produced by slab compaction + offsets),
    # merged with min-root unions; ranking roots must equal fastremap.renumber.
    rng = np.random.default_rng(seed)
    n_labels = 60
    seg = np.zeros(400, np.uint32)
    seg[rng.random(400) < 0.8] = 1
    nz = np.flatnonzero(seg)
    seg[nz] = np.sort(rng.integers(1, n_labels + 1, nz.size))
    seg[nz[:n_labels]] = np.arange(1, n_labels + 1)  # every label appears, in order
    seg[nz] = np.maximum.accumulate(seg[nz])
    parent = np.zeros(n_labels + 1, np.uint32)
    for _ in range(25):
        a, b = rng.integers(1, n_labels + 1, 2)
        fc._union_plane(
            np.array([[a]], np.uint32), np.array([[b]], np.uint32), np.ones((1, 1), bool), parent
        )
    fc._compress_roots(parent)
    roots = parent[seg]
    count = fc._final_ids_from_roots(parent)
    out = np.empty_like(seg)
    fc._apply_final_ids(seg, parent, out)
    expected, _ = fastremap.renumber(roots.copy(), preserve_zero=True)
    np.testing.assert_array_equal(out, expected)
    assert count == len(np.unique(roots[roots != 0]))


@pytest.mark.parametrize("shape", [(3, 0, 4, 5), (3, 4, 0, 5), (3, 4, 5, 0)])
def test_empty_axes(shape):
    hard = np.zeros(shape, np.float32)
    with pytest.raises(ValueError) as reference:
        decode_affinity_cc(hard, backend="numba")
    with pytest.raises(type(reference.value)):
        decode_affinity_cc(hard, backend="slab", parallel="threads")
    actual = fc.connected_components_affinity_3d_slab(hard, parallel="threads", finalize=False)
    assert actual.shape == shape[1:]
    assert actual.size == 0 and actual.dtype == np.uint32


@pytest.mark.parametrize("eo", [0, 1])
@pytest.mark.parametrize("orphan", [False, True])
@pytest.mark.parametrize("full", [False, True])
def test_decoder_channels(eo, orphan, full):
    data = (
        np.ones((6, 8, 7, 6), np.float32)
        if full
        else np.random.default_rng(2).random((6, 8, 7, 6), dtype=np.float32)
    )
    kwargs = {"edge_offset": eo, "orphan_fill": orphan, "affinity_channels": [5, 3, 1]}
    actual = decode_affinity_cc(data, backend="slab", parallel="threads", num_workers=3, **kwargs)
    expected = numba_reference(data, **kwargs)
    assert_partition_equal(actual, decode_affinity_cc(data, backend="numba_serial", **kwargs))
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)


def test_unfinalized_and_promotion(monkeypatch):
    from connectomics.decoding.decoders.segmentation_kernels import (
        connected_components_affinity_3d_numba_parallel,
    )

    hard = np.random.default_rng(5).random((3, 8, 7, 6)) > 0.7
    monkeypatch.setattr(fc, "UINT32_LIMIT", 5)
    result = fc.connected_components_affinity_3d_slab(
        hard, slab_size=3, parallel="threads", finalize=False
    )
    assert result.dtype == np.uint64
    np.testing.assert_array_equal(result, connected_components_affinity_3d_numba_parallel(hard))
    check(hard, 0, 3, 3)
    full = np.ones_like(hard)
    result = fc.connected_components_affinity_3d_slab(
        full, slab_size=3, parallel="threads", finalize=False
    )
    assert result.dtype == np.uint32
    assert np.all(result == 1)


@pytest.mark.parametrize(
    "kwargs", [{"edge_offset": 2}, {"parallel": "bad"}, {"num_workers": 0}, {"slab_size": 0}]
)
def test_invalid(kwargs):
    with pytest.raises(ValueError):
        fc.connected_components_affinity_3d_slab(np.ones((3, 4, 4, 4)), **kwargs)


def test_shape_and_voxel_guards(monkeypatch):
    for shape in [(2, 3, 4, 5), (3, 4, 5)]:
        with pytest.raises(ValueError):
            fc.connected_components_affinity_3d_slab(np.ones(shape))
    monkeypatch.setattr(fc, "SLAB_VOXEL_LIMIT", 10)
    with pytest.raises(ValueError):
        fc.connected_components_affinity_3d_slab(np.ones((3, 4, 4, 4)), slab_size=1)


def test_serpentine():
    hard = np.zeros((3, 24, 24, 24), bool)
    points = []
    for z in range(24):
        plane = [
            (x, y, z) for y in range(24) for x in (range(24) if y % 2 == 0 else range(23, -1, -1))
        ]
        points.extend(plane if z % 2 == 0 else plane[::-1])
    for a, b in zip(points, points[1:]):
        edge(hard, a, b, 0)
    actual = fc.connected_components_affinity_3d_slab(
        hard, slab_size=3, num_workers=3, parallel="threads"
    )
    assert_partition_equal(actual, decode_affinity_cc(hard, backend="numba_serial"))


def test_no_numba(monkeypatch):
    from connectomics.decoding.decoders import segmentation

    data = np.ones((3, 4, 4, 4))
    monkeypatch.setattr(segmentation, "NUMBA_AVAILABLE", False)
    with pytest.warns(UserWarning, match="falling back"):
        actual = decode_affinity_cc(data, backend="slab")
    np.testing.assert_array_equal(actual, decode_affinity_cc(data, backend="cc3d"))


@pytest.mark.skipif(not process_parallel_available(), reason="Processes unavailable")
@pytest.mark.parametrize("eo", [0, 1])
@pytest.mark.parametrize("parallel", PROCESS_MODES)
def test_process_owned_and_transferred(tmp_path, eo, parallel):
    buf = tmp_path / "buf"
    buf.mkdir()
    hard = np.random.default_rng(7).random((3, 40, 23, 7)) < 0.55
    expected = numba_reference(hard, edge_offset=eo)
    actual = fc.connected_components_affinity_3d_slab(
        hard, edge_offset=eo, slab_size=8, num_workers=2, parallel=parallel, buffer_dir=str(buf)
    )
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == expected.dtype
    assert_partition_equal(actual, decode_affinity_cc(hard, backend="numba_serial", edge_offset=eo))
    data = np.ones_like(hard, dtype=np.float32)
    actual = decode_affinity_cc(
        data,
        backend="slab",
        edge_offset=eo,
        orphan_fill=bool(eo),
        num_workers=2,
        parallel=parallel,
        buffer_dir=str(buf),
    )
    expected = numba_reference(data, edge_offset=eo, orphan_fill=bool(eo))
    assert_partition_equal(
        actual,
        decode_affinity_cc(data, backend="numba_serial", edge_offset=eo, orphan_fill=bool(eo)),
    )
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == expected.dtype
    assert not list(buf.iterdir())


def test_transferred_cleanup_on_validation(tmp_path):
    from connectomics.utils.shared_buffer import SharedArray

    buf = SharedArray.create((3, 2, 2, 2), bool, tmp_path)
    with pytest.raises(ValueError):
        fc._connected_components_slab(buf, parallel="invalid")
    assert not list(tmp_path.iterdir())


@pytest.mark.skipif(not process_parallel_available(), reason="Processes unavailable")
@pytest.mark.parametrize("parallel", PROCESS_MODES)
def test_process_safety_subprocess(tmp_path, parallel):
    from tests.unit.test_shared_buffer import _run_child

    _run_child(
        tmp_path,
        """
import os
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from connectomics.decoding.decoders.segmentation import decode_affinity_cc
from connectomics.utils.shared_buffer import process_pool_context
from connectomics.decoding.decoders import affinity_cc_slab as fc
fc.SLAB_VOXEL_LIMIT = 4 * 9 * 7
MODE = MODE_VALUE
original_initializer = fc._init_fork_worker

def forbid_compile(*args, **kwargs):
    raise AssertionError("Worker attempted Numba compilation")

def checked_initializer(*args):
    original_initializer(*args)
    fc._build_graph._compile_for_args = forbid_compile
    fc._compact_first_appearance._compile_for_args = forbid_compile

def daemon_check(data):
    expected = decode_affinity_cc(data, backend="numba")
    actual = decode_affinity_cc(data, backend="slab", parallel="auto", num_workers=2)
    np.testing.assert_array_equal(actual, expected)
    try:
        decode_affinity_cc(data, backend="slab", parallel=MODE, num_workers=2)
    except RuntimeError:
        return True
    raise AssertionError("Explicit processes accepted inside daemon")

if __name__ == "__main__":
    data = np.random.default_rng(5).random((3, 12, 9, 7), dtype=np.float32)
    expected = decode_affinity_cc(data, backend="numba")
    if MODE == "fork":
        fc._init_fork_worker = checked_initializer
    def decode(mode, value=data):
        return decode_affinity_cc(value, backend="slab", parallel=mode, num_workers=2)
    lock = threading.Lock()
    held = threading.Event()
    release = threading.Event()
    def hold_lock():
        with lock:
            held.set()
            release.wait()
    holder = threading.Thread(target=hold_lock)
    holder.start()
    held.wait()
    try:
        for _ in range(3):
            np.testing.assert_array_equal(decode(MODE), expected)
        with ThreadPoolExecutor(1) as pool:
            np.testing.assert_array_equal(pool.submit(decode, MODE).result(), expected)
    finally:
        release.set()
        holder.join()
    other = data * .8
    other_expected = decode("threads", other)
    for second_mode in (MODE, "forkserver"):
        with ThreadPoolExecutor(2) as pool:
            first = pool.submit(decode, MODE)
            second = pool.submit(decode, second_mode, other)
            np.testing.assert_array_equal(first.result(), expected)
            np.testing.assert_array_equal(second.result(), other_expected)
    with process_pool_context().Pool(1) as pool:
        assert pool.apply(daemon_check, (data,))
""".replace("MODE_VALUE", repr(parallel)),
    )


@pytest.mark.skipif(not process_parallel_available(), reason="Processes unavailable")
@pytest.mark.parametrize("parallel", PROCESS_MODES)
def test_slab_worker_failure_subprocess(tmp_path, parallel):
    from tests.unit.test_shared_buffer import _run_child

    buf = tmp_path / "buf"
    buf.mkdir()
    _run_child(
        tmp_path,
        f"""
import os
import numpy as np
from concurrent.futures.process import BrokenProcessPool
from connectomics.decoding.decoders import affinity_cc_slab as fc
from connectomics.utils.shared_buffer import SharedArray

def fail(task):
    raise ValueError("worker failure")

def die(task):
    os._exit(1)

if __name__ == "__main__":
    mode = {parallel!r}
    worker_name = "_fork_worker" if mode == "fork" else "_slab_worker"
    original_worker = getattr(fc, worker_name)
    for worker, error in [(fail, ValueError), (die, BrokenProcessPool)]:
        setattr(fc, worker_name, worker)
        buf = SharedArray.create((3, 4, 4, 4), bool, {str(buf)!r})
        buf.array[:] = True
        try:
            fc._connected_components_slab(buf, parallel=mode, num_workers=2,
                                          slab_size=2, buffer_dir={str(buf)!r})
        except error:
            pass
        else:
            raise AssertionError("Worker failure did not propagate")
        assert not os.listdir({str(buf)!r})
    setattr(fc, worker_name, original_worker)
    hard = np.ones((3, 4, 4, 4), bool)
    actual = fc.connected_components_affinity_3d_slab(
        hard, parallel=mode, num_workers=2, slab_size=2, buffer_dir={str(buf)!r})
    expected = fc.connected_components_affinity_3d_slab(hard, parallel="threads")
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == expected.dtype
    assert not os.listdir({str(buf)!r})
""",
    )


def test_process_buffer_lifetime_without_pool(tmp_path, monkeypatch):
    from connectomics.utils.shared_buffer import SharedArray

    monkeypatch.setattr(fc, "process_parallel_available", lambda: True)
    monkeypatch.setattr(
        fc, "run_process_tasks", lambda fn, tasks, *args, **kwargs: list(map(fn, tasks))
    )
    hard = np.ones((3, 4, 4, 4), bool)
    expected = numba_reference(hard)
    actual = fc.connected_components_affinity_3d_slab(
        hard, parallel="forkserver", num_workers=2, slab_size=2, buffer_dir=str(tmp_path)
    )
    np.testing.assert_array_equal(actual, expected)
    assert not list(tmp_path.iterdir())

    labels = fc.connected_components_affinity_3d_slab(
        hard,
        parallel="forkserver",
        num_workers=2,
        slab_size=2,
        finalize=False,
        buffer_dir=str(tmp_path),
    )
    assert labels.dtype == np.uint32
    np.testing.assert_array_equal(labels, np.ones(hard.shape[1:], np.uint32))
    assert not list(tmp_path.iterdir())

    def fail(*args, **kwargs):
        raise ValueError("injected finalization failure")

    monkeypatch.setattr(fc, "_finish", fail)
    buf = SharedArray.create(hard.shape, bool, tmp_path)
    buf.array[:] = hard
    with pytest.raises(ValueError, match="finalization"):
        fc._connected_components_slab(
            buf, parallel="forkserver", num_workers=2, slab_size=2, buffer_dir=str(tmp_path)
        )
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "platform,methods,expected",
    [
        ("linux", ["fork", "forkserver", "spawn"], "fork"),
        ("linux", ["spawn"], "forkserver"),
        ("darwin", ["fork", "spawn"], "forkserver"),
        ("win32", ["spawn"], "forkserver"),
    ],
)
def test_resolve_parallel(monkeypatch, platform, methods, expected):
    monkeypatch.setattr(fc.sys, "platform", platform)
    monkeypatch.setattr(fc.mp, "get_all_start_methods", lambda: methods)
    monkeypatch.setattr(fc, "process_parallel_available", lambda: True)
    for mode in ("auto", "processes"):
        assert fc._resolve_parallel(mode, 2) == expected
    assert fc._resolve_parallel("auto", 1) == "threads"
    assert fc._resolve_parallel("forkserver", 2) == "forkserver"
    if expected == "forkserver":
        with pytest.raises(RuntimeError):
            fc._resolve_parallel("fork", 2)
    monkeypatch.setattr(fc, "process_parallel_available", lambda: False)
    assert fc._resolve_parallel("auto", 2) == "threads"
    for mode in ("fork", "forkserver", "processes"):
        with pytest.raises(RuntimeError):
            fc._resolve_parallel(mode, 2)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux fork")
def test_fork_inherits_without_shared_files(monkeypatch):
    from connectomics.utils.shared_buffer import SharedArray

    def forbidden(*args, **kwargs):
        raise AssertionError("Fork must not allocate shared files")

    monkeypatch.setattr(SharedArray, "create", forbidden)
    hard = np.random.default_rng(9).random((3, 8, 7, 12)) > 0.6
    for value in (hard, np.asfortranarray(hard), hard[:, :, :, ::2]):
        value.flags.writeable = False
        actual = fc.connected_components_affinity_3d_slab(
            value, parallel="fork", num_workers=2, slab_size=3, finalize=False
        )
        expected = fc.connected_components_affinity_3d_slab(
            value, parallel="threads", num_workers=2, slab_size=3, finalize=False
        )
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == expected.dtype
    decode_affinity_cc(hard, backend="slab", parallel="fork", num_workers=2)
    assert not hasattr(fc, "_fork_state")

"""Exact native NumPy comparisons for HDF5 affinities."""

import h5py
import numpy as np
import pytest

from connectomics.data.io import read_hdf5_affinity_threshold as read
from connectomics.utils.shared_buffer import process_parallel_available

THRESHOLDS = [
    -np.inf,
    -1.0,
    -0.0,
    0.0,
    1e-8,
    0.3,
    0.5,
    0.66,
    0.7,
    0.66015625,
    0.999,
    1.0,
    2.0,
    np.inf,
    np.nan,
    np.float32(0.66),
    np.float16(0.66),
]


def write(path, data, chunked=True):
    with h5py.File(path, "w") as f:
        f.create_dataset(
            "main",
            data=data,
            chunks=(1, min(4, data.shape[1]), data.shape[2], data.shape[3]) if chunked else None,
            compression="lzf" if chunked else None,
        )


@pytest.mark.parametrize("threshold", THRESHOLDS)
@pytest.mark.parametrize("dtype", ["f2", ">f2"])
def test_all_float16_patterns(tmp_path, threshold, dtype):
    data = np.arange(65536, dtype=np.uint16).view(np.float16).reshape(1, 16, 64, 64).astype(dtype)
    path = tmp_path / "a.h5"
    write(path, data)
    actual = read(path, threshold, channels=[0], num_workers=1)
    np.testing.assert_array_equal(actual, data > threshold)
    assert actual.dtype == bool and actual.flags.c_contiguous


@pytest.mark.parametrize("dtype", ["f2", ">f2", "f4", "u1", "bool"])
@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize("channels", [(2, 0, 1), (1, 1, 0), (-1, 0, 1)])
@pytest.mark.parametrize("bbox", [None, (3, 17, 2, 7, 1, 5)])
def test_dtypes_channels_bbox(tmp_path, dtype, chunked, channels, bbox):
    data = (np.random.default_rng(0).random((6, 18, 8, 6)) * 2).astype(dtype)
    path = tmp_path / "a.h5"
    write(path, data, chunked)
    actual = read(path, 0.66, channels=channels, bbox=bbox, num_workers=1)
    expected = data[list(channels)]
    if bbox is not None:
        a, b, c, d, e, f = bbox
        expected = expected[:, a:b, c:d, e:f]
    np.testing.assert_array_equal(actual, expected > 0.66)


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"channels": [6]}, IndexError),
        ({"channels": [-7]}, IndexError),
        ({"bbox": [0, 19, 0, 8, 0, 6]}, ValueError),
        ({"bbox": [1]}, ValueError),
        ({"threshold": [0.5]}, TypeError),
        ({"threshold": 1j}, TypeError),
        ({"num_workers": 0}, ValueError),
    ],
)
def test_invalid(tmp_path, kwargs, error):
    path = tmp_path / "a.h5"
    write(path, np.ones((6, 18, 8, 6), np.float32))
    args = {"threshold": 0.5, "num_workers": 1}
    args.update(kwargs)
    with pytest.raises(error):
        read(path, **args)


@pytest.mark.skipif(not process_parallel_available(), reason="Processes unavailable")
def test_process_reader_and_decode(tmp_path):
    from connectomics.decoding.decoders.affinity_cc_slab import (
        connected_components_affinity_3d_slab,
    )
    from connectomics.decoding.decoders.segmentation import decode_affinity_cc

    data = np.random.default_rng(0).choice(
        np.array([0.659, 0.66, 0.661, 0, 1], np.float16), (3, 18, 8, 6)
    )
    path = tmp_path / "a.h5"
    buf = tmp_path / "buf"
    buf.mkdir()
    write(path, data)
    kwargs = {"channels": [2, 0, 1], "bbox": (3, 17, 2, 7, 1, 5)}
    actual = read(path, 0.66, num_workers=3, buffer_dir=str(buf), **kwargs)
    np.testing.assert_array_equal(actual, read(path, 0.66, num_workers=1, **kwargs))
    hard = read(path, 0.66, num_workers=3, buffer_dir=str(buf))
    actual = connected_components_affinity_3d_slab(hard, num_workers=2, parallel="threads")
    from tests.unit.test_affinity_cc_slab import assert_partition_equal, numba_reference

    expected = numba_reference(data, threshold=0.66)
    assert_partition_equal(actual, decode_affinity_cc(data, 0.66, backend="numba_serial"))
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == expected.dtype
    assert not list(buf.iterdir())


@pytest.mark.skipif(not process_parallel_available(), reason="Processes unavailable")
def test_corrupt_chunk_cleanup(tmp_path):
    path = tmp_path / "a.h5"
    buf = tmp_path / "buf"
    buf.mkdir()
    write(path, np.zeros((3, 18, 8, 6), np.float16))
    with h5py.File(path, "r") as f:
        chunk = f["main"].id.get_chunk_info(0)
    with open(path, "r+b") as f:
        f.seek(chunk.byte_offset)
        f.write(b"\xff" * chunk.size)
    with pytest.raises(OSError, match="read|decompress|filter|inflate"):
        read(path, 0.66, num_workers=3, buffer_dir=str(buf))
    assert not list(buf.iterdir())

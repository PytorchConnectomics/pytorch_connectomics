"""Chunked inference -> precomputed affinity layer.

Covers the two things that are silent when wrong: the ABISS affinity convention
(edge shift + channel order) and the storage-chunk alignment that makes concurrent
chunk writes safe.
"""

import numpy as np
import pytest

from connectomics.inference.chunked import (
    _to_abiss_affinity_convention,
    _validate_precomputed_alignment,
)


def _reference_convention(a: np.ndarray) -> np.ndarray:
    """Independent port of dev/zebrafinch/upload_affinity_full_masked.py.

    ``_load_affinity_neg_offset`` (dst[c, v] = src[c, v-1], global low face zeroed)
    followed by the ``a[::-1]`` channel reversal, which is how the reference affinity
    layer that ABISS consumed was actually produced.
    """
    out = np.empty_like(a)
    for c in range(3):
        src = a[c]
        dst = np.empty_like(src)
        hi = [slice(None)] * 3
        lo = [slice(None)] * 3
        hi[c] = slice(1, None)
        lo[c] = slice(0, -1)
        dst[tuple(hi)] = src[tuple(lo)]
        face = [slice(None)] * 3
        face[c] = 0
        dst[tuple(face)] = 0
        out[c] = dst
    return out[::-1]


def test_matches_reference_implementation():
    vol = np.random.default_rng(0).random((3, 12, 10, 9)).astype("float32")
    assert np.array_equal(_to_abiss_affinity_convention(vol), _reference_convention(vol))


def test_channel_zero_is_x_affinity():
    """ABISS reads channel 0 as x-affinity; the model emits channel 0 as z."""
    probe = np.zeros((3, 4, 4, 4), dtype="float32")
    probe[2] = 1.0  # model's x-affinity channel
    out = _to_abiss_affinity_convention(probe)
    assert out[0].max() == 1.0
    assert out[1].max() == 0.0 and out[2].max() == 0.0


def test_halo_supplies_core_low_face():
    """The shift must run on the haloed array, not per cropped chunk.

    A chunk-local conversion zero-fills each chunk's low face, corrupting every
    internal chunk boundary; converting the haloed block and then cropping matches
    the whole-volume answer.
    """
    vol = np.random.default_rng(1).random((3, 16, 6, 6)).astype("float32")
    halo, z0, z1 = 2, 6, 12
    truth = _to_abiss_affinity_convention(vol)[:, z0:z1]

    haloed = _to_abiss_affinity_convention(vol[:, z0 - halo : z1 + halo])[:, halo:-halo]
    assert np.allclose(haloed, truth)

    chunk_local = _to_abiss_affinity_convention(vol[:, z0:z1])
    assert not np.allclose(chunk_local, truth)


def test_rejects_non_three_channel():
    with pytest.raises(ValueError, match="3-channel"):
        _to_abiss_affinity_convention(np.zeros((2, 4, 4, 4), dtype="float32"))


@pytest.mark.parametrize(
    "chunk_shape_zyx, storage_xyz, ok",
    [
        ((1008, 1008, 1008), [144, 144, 72], True),  # 1008 = 7*144 = 14*72
        ((1008, 1008, 1008), [128, 128, 64], False),  # 1008 % 128 != 0
        ((256, 256, 256), [256, 256, 256], True),
        ((256, 256, 256), [0, 64, 64], False),  # degenerate
    ],
)
def test_alignment_guard(chunk_shape_zyx, storage_xyz, ok):
    """Storage chunks must tile the inference chunk, else two ranks race on one."""
    if ok:
        _validate_precomputed_alignment(chunk_shape_zyx, storage_xyz)
    else:
        with pytest.raises(ValueError, match="precomputed_chunk_size"):
            _validate_precomputed_alignment(chunk_shape_zyx, storage_xyz)

from __future__ import annotations

import numpy as np

from connectomics.decoding.decoders.shape_smooth import label_opening, shape_smooth

# ABISS emits chunked-graph ids in this range; they do not fit in uint32.
ABISS_BASE = 2**56


def _abiss_volume() -> np.ndarray:
    seg = np.zeros((16, 32, 32), np.uint64)
    seg[2:14, 5:13, 5:13] = ABISS_BASE + 5
    seg[2:14, 19:27, 19:27] = ABISS_BASE + 9
    return seg


def test_opening_preserves_large_uint64_ids():
    seg = _abiss_volume()
    out = label_opening(seg)
    assert out.dtype == np.uint64
    assert set(np.unique(out)) <= {0, ABISS_BASE + 5, ABISS_BASE + 9}
    assert (out == ABISS_BASE + 5).any() and (out == ABISS_BASE + 9).any()


def test_shape_smooth_does_not_wrap_abiss_ids_into_one_label():
    seg = _abiss_volume()
    out = shape_smooth(seg, split=False)
    # Truncating 2**56 + n to uint32 would collapse both blocks onto label 0.
    assert len(np.unique(out[out > 0])) == 2
    assert int((out > 0).sum()) > 0

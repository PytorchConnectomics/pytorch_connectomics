#!/usr/bin/env python3
"""Axis resolution and the CLAHE shim.

`nd2_source.output_axes_and_shape` is a port of
`connectomics/data/io/io.py::_nd2_output_axes_and_shape`. The port exists to
keep torch out of a CPU image, so the risk it introduces is drift -- these
cases are taken from the original's own error paths.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import nd2_source as ns  # noqa: E402


def test_channel_first_order_regardless_of_file_order():
    axes, shape = ns.output_axes_and_shape("v.nd2", {"Z": 585, "C": 2, "Y": 2304, "X": 2304})
    assert axes == ["C", "Z", "Y", "X"]
    assert shape == (2, 585, 2304, 2304)


def test_single_channel_file_has_no_c_axis():
    axes, shape = ns.output_axes_and_shape("v.nd2", {"Z": 10, "Y": 4, "X": 4})
    assert axes == ["Z", "Y", "X"] and shape == (10, 4, 4)


def test_multi_position_is_refused_by_name():
    """A mosaic must not be silently ingested as one tile."""
    try:
        ns.output_axes_and_shape("v.nd2", {"P": 4, "Z": 10, "Y": 4, "X": 4})
    except ValueError as e:
        assert "one XY position" in str(e) and "P=4" in str(e)
    else:
        raise AssertionError("expected ValueError for a multi-position ND2")


def test_other_unsupported_axis_is_named():
    try:
        ns.output_axes_and_shape("v.nd2", {"T": 3, "Z": 10, "Y": 4, "X": 4})
    except ValueError as e:
        assert "T=3" in str(e)
    else:
        raise AssertionError("expected ValueError for a time axis")


def test_missing_xy_is_refused():
    try:
        ns.output_axes_and_shape("v.nd2", {"Z": 10, "Y": 4})
    except ValueError as e:
        assert "Y and X" in str(e)
    else:
        raise AssertionError("expected ValueError when X is absent")


def test_preprocess_module_loads_without_connectomics():
    """The shim must give us the repo's CLAHE, not a copy of it."""
    pre = ns.load_preprocess_module()
    assert hasattr(pre, "preprocess_xy_plane")
    assert hasattr(pre, "volume_average_uint8")
    plane = np.linspace(0, 4000, 64 * 64, dtype=np.float32).reshape(64, 64)
    out = pre.preprocess_xy_plane(plane, clip_intensity_range=None,
                                  lower_percentile=1.0, upper_percentile=99.0,
                                  clip_limit=0.03)
    assert out.dtype == np.uint8 and out.shape == (64, 64)
    assert out.min() < out.max(), "CLAHE collapsed a ramp to a constant"


def test_percentile_clip_is_per_plane():
    """Streaming is only equivalent to a whole-volume read because the
    statistic is per plane. If this ever becomes global, the streaming design
    in zarr_writer silently changes the pixels."""
    pre = ns.load_preprocess_module()
    kw = dict(clip_intensity_range=None, lower_percentile=1.0,
              upper_percentile=99.0, clip_limit=0.03)
    rng = np.random.default_rng(0)
    plane = rng.integers(100, 400, size=(64, 64)).astype(np.float32)
    alone = pre.preprocess_xy_plane(plane, **kw)
    # Same plane, processed while a far brighter plane exists elsewhere in the
    # stack: a global statistic would move; a per-plane one cannot.
    _bright = pre.preprocess_xy_plane(plane * 10.0, **kw)
    again = pre.preprocess_xy_plane(plane, **kw)
    assert np.array_equal(alone, again)


def test_flat_plane_returns_zeros_not_an_error():
    pre = ns.load_preprocess_module()
    out = pre.preprocess_xy_plane(np.full((8, 8), 7.0, dtype=np.float32),
                                  clip_intensity_range=None, clip_limit=0.03)
    assert out.dtype == np.uint8 and not out.any()

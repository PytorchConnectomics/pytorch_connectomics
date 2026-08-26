"""`run_abiss_volume --edge-storage` must shift source-stored affinity by one voxel.

An affinity value describes the edge between two adjacent voxels but is stored in
a voxel-indexed array, so one of the two endpoints has to hold it. ABISS `ws`
reads the destination endpoint -- ``basic_watershed.hpp`` uses
``negx = aff[x][y][z][0]``, ``posx = aff[x+1][y][z][0]`` -- while this repo's
``banis`` affinity target (``seg_to_affinity``, ``affinity_mode="banis"``) writes
the source endpoint. Feeding one to the other puts every boundary decision a
voxel off along every axis, which is silent: the segmentation still looks
plausible, it is just wrong.

The default stays ``destination`` so existing configs are unaffected.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "run_abiss_volume.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("run_abiss_volume", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script():
    return _load_script()


def _ramp_czyx(shape=(3, 4, 5, 6)) -> np.ndarray:
    """Affinity whose value encodes its own index, so a shift is visible."""
    return np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + 1.0


def test_destination_is_the_default_and_does_not_shift(script):
    aff = _ramp_czyx()
    default = script._to_abiss_affinity(aff, channels=None)
    explicit = script._to_abiss_affinity(aff, channels=None, edge_storage="destination")
    # (C, Z, Y, X) -> (X, Y, Z, C)
    np.testing.assert_array_equal(default, np.transpose(aff[:3], (3, 2, 1, 0)))
    np.testing.assert_array_equal(default, explicit)


def test_source_shifts_each_channel_along_its_own_axis(script):
    aff = _ramp_czyx()
    shifted = script._to_abiss_affinity(aff, channels=None, edge_storage="source")
    unshifted = script._to_abiss_affinity(aff, channels=None, edge_storage="destination")

    for k in range(3):
        # value at index i must now be the one that was at i-1 along dim k
        moved = np.take(shifted[..., k], range(1, shifted.shape[k]), axis=k)
        original = np.take(unshifted[..., k], range(0, shifted.shape[k] - 1), axis=k)
        np.testing.assert_array_equal(moved, original)

        # the exposed face is not a real edge and must be zeroed, not wrapped
        face = np.take(shifted[..., k], [0], axis=k)
        assert np.all(face == 0.0), f"channel {k} face was not zeroed"


def test_shift_is_applied_after_channel_selection(script):
    """Channel k of the ABISS layout is the dim-k edge whatever order was asked for."""
    aff = _ramp_czyx()
    reversed_sel = script._to_abiss_affinity(aff, channels=[2, 1, 0], edge_storage="source")
    per_channel = script._to_abiss_affinity(aff, channels=None, edge_storage="destination")

    for k, c in enumerate([2, 1, 0]):
        moved = np.take(reversed_sel[..., k], range(1, reversed_sel.shape[k]), axis=k)
        original = np.take(per_channel[..., c], range(0, reversed_sel.shape[k] - 1), axis=k)
        np.testing.assert_array_equal(moved, original)


def test_single_channel_probability_input_ignores_edge_storage(script):
    """The pmap path builds min(p[i-1], p[i]), which is destination-stored already."""
    pmap = _ramp_czyx(shape=(1, 4, 5, 6))
    as_source = script._to_abiss_affinity(pmap, channels=None, edge_storage="source")
    as_destination = script._to_abiss_affinity(pmap, channels=None, edge_storage="destination")
    np.testing.assert_array_equal(as_source, as_destination)


def test_unknown_edge_storage_raises(script):
    with pytest.raises(ValueError, match="edge_storage"):
        script._to_abiss_affinity(_ramp_czyx(), channels=None, edge_storage="banis")

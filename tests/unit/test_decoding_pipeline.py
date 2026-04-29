"""Unit tests for decoding registry and decode pipeline."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from connectomics.decoding import (
    DecoderRegistry,
    apply_decode_pipeline,
    decode_affinity_cc,
    list_decoders,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


class _Mode:
    """Simple object-form decode mode for compatibility checks."""

    def __init__(self, name: str, kwargs: dict | None = None):
        self.name = name
        self.kwargs = kwargs or {}


@pytest.fixture
def affinity_with_redundant_channels() -> np.ndarray:
    """Create 6-channel affinity with meaningful short-range and noisy extra channels."""
    aff = np.zeros((6, 16, 16, 16), dtype=np.float32)
    aff[:3, 2:10, 2:10, 2:10] = 0.9
    aff[:3, 11:15, 11:15, 11:15] = 0.9

    # Extra channels are intentionally noisy/redundant and should be ignored.
    rng = np.random.default_rng(42)
    aff[3:] = rng.random((3, 16, 16, 16), dtype=np.float32)
    return aff


def test_builtin_decoders_are_registered():
    names = set(list_decoders())
    assert "decode_affinity_cc" in names
    assert "decode_distance_watershed" in names
    assert "decode_instance_binary_contour_distance" in names
    assert "decode_abiss" in names


def test_import_decoding_does_not_eagerly_import_decoder_modules():
    code = (
        "import sys\n"
        "import connectomics.decoding as decoding\n"
        "print('connectomics.decoding.decoders.waterz' in sys.modules)\n"
        "print('connectomics.decoding.decoders.segmentation' in sys.modules)\n"
        "_ = decoding.decode_affinity_cc\n"
        "print('connectomics.decoding.decoders.segmentation' in sys.modules)\n"
        "print('connectomics.decoding.decoders.waterz' in sys.modules)\n"
    )
    output = subprocess.check_output(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        text=True,
    ).splitlines()

    assert output == ["False", "False", "True", "False"]


def test_decode_pipeline_dict_mode_matches_direct_decoder(affinity_with_redundant_channels):
    decode_modes = [{"name": "decode_affinity_cc", "kwargs": {"threshold": 0.5}}]

    seg_pipeline = apply_decode_pipeline(affinity_with_redundant_channels, decode_modes)
    seg_direct = decode_affinity_cc(affinity_with_redundant_channels, threshold=0.5)

    np.testing.assert_array_equal(seg_pipeline, seg_direct)


def test_decode_pipeline_object_mode_matches_direct_decoder(affinity_with_redundant_channels):
    decode_modes = [_Mode("decode_affinity_cc", {"threshold": 0.5})]

    seg_pipeline = apply_decode_pipeline(affinity_with_redundant_channels, decode_modes)
    seg_direct_short = decode_affinity_cc(affinity_with_redundant_channels[:3], threshold=0.5)

    np.testing.assert_array_equal(seg_pipeline, seg_direct_short)


def test_decode_pipeline_unknown_decoder_raises(affinity_with_redundant_channels):
    decode_modes = [{"name": "decode_not_exists", "kwargs": {}}]
    with pytest.raises(ValueError, match="Unknown decode function"):
        apply_decode_pipeline(affinity_with_redundant_channels, decode_modes)


def test_decode_pipeline_resolves_python_style_channel_selectors():
    registry = DecoderRegistry()

    def capture_channels(data: np.ndarray, distance_channels=None):
        return np.asarray(distance_channels, dtype=np.int64)

    registry.register("capture_channels", capture_channels)
    data = np.zeros((4, 2, 2, 2), dtype=np.float32)
    decode_modes = [
        {
            "name": "capture_channels",
            "kwargs": {"distance_channels": "1:-1"},
        }
    ]

    resolved = apply_decode_pipeline(data, decode_modes, registry=registry)

    np.testing.assert_array_equal(resolved, np.array([1, 2], dtype=np.int64))

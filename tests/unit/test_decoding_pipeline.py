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
    aff: np.ndarray = np.zeros((6, 16, 16, 16), dtype=np.float32)
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
    assert "decode_binary_contour_distance_watershed" in names
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


def test_binary_contour_distance_alias_matches_canonical_on_banis_channels():
    predictions = np.zeros((7, 8, 8, 8), dtype=np.float32)
    predictions[:3, 2:6, 2:6, 2:6] = 1.0
    predictions[2, 3, 3, 3] = 0.6
    predictions[6, 2:6, 2:6, 2:6] = 1.0
    kwargs = {
        "binary_channels": [0, 1, 2],
        "binary_channel_reduction": "min",
        "contour_channels": None,
        "distance_channels": [6],
        "binary_threshold": [0.9, 0.85],
        "contour_threshold": None,
        "distance_threshold": [0.5, -0.5],
        "min_instance_size": 0,
        "min_seed_size": 0,
        "prediction_scale": 1,
    }

    seg_alias = apply_decode_pipeline(
        predictions,
        [{"name": "decode_binary_contour_distance_watershed", "kwargs": kwargs}],
    )
    seg_canonical = apply_decode_pipeline(
        predictions,
        [{"name": "decode_instance_binary_contour_distance", "kwargs": kwargs}],
    )

    np.testing.assert_array_equal(seg_alias, seg_canonical)
    assert seg_alias[3, 3, 3] == 0


def test_decode_pipeline_on_step_complete_called_per_step():
    """Per-step callback fires once per (batch, step) with the step's output array."""
    registry = DecoderRegistry()
    registry.register("step_a", lambda data, **kw: data + 1)
    registry.register("step_b", lambda data, **kw: data * 2)

    data = np.zeros((1, 2, 2, 2), dtype=np.float32)
    captured: list[tuple[int, str, np.ndarray]] = []

    def on_step(batch_idx, step, sample):
        name = step.name if hasattr(step, "name") else step["name"]
        captured.append((batch_idx, name, sample.copy()))

    decode_modes = [{"name": "step_a", "kwargs": {}}, {"name": "step_b", "kwargs": {}}]
    final = apply_decode_pipeline(data, decode_modes, registry=registry, on_step_complete=on_step)

    assert [(b, n) for b, n, _ in captured] == [(0, "step_a"), (0, "step_b")]
    np.testing.assert_array_equal(captured[0][2], data + 1)
    np.testing.assert_array_equal(captured[1][2], (data + 1) * 2)
    np.testing.assert_array_equal(final, (data + 1) * 2)


def test_decode_pipeline_on_step_complete_single_step_matches_final():
    """For a one-step pipeline the callback's array equals the final output."""
    registry = DecoderRegistry()
    registry.register("only", lambda data, **kw: data + 7)

    data = np.zeros((1, 2, 2, 2), dtype=np.float32)
    captured = []

    final = apply_decode_pipeline(
        data,
        [{"name": "only", "kwargs": {}}],
        registry=registry,
        on_step_complete=lambda b, s, x: captured.append(x.copy()),
    )

    assert len(captured) == 1
    np.testing.assert_array_equal(captured[0], final)


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

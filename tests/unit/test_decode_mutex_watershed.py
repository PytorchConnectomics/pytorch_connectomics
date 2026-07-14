"""Unit tests for the Mutex Watershed decoder wrapper.

The real ``affogato`` (C++/xtensor, conda-forge only) is not required: the
tests stub ``compute_mws_segmentation`` so wrapper behavior — offset defaulting,
channel/attractive validation, repulsive-channel inversion, stride forwarding,
and registration — is exercised deterministically. One test also covers the
graceful ImportError path when affogato is absent.
"""

from __future__ import annotations

import numpy as np
import pytest

from connectomics.decoding.decoders import mutex_watershed as mws


def _install_fake(monkeypatch):
    """Patch in a recording fake for affogato's compute_mws_segmentation."""
    calls = {}

    def fake_compute(affs, offsets, num_attractive, **kwargs):
        calls["affs"] = np.asarray(affs).copy()
        calls["offsets"] = list(offsets)
        calls["num_attractive"] = num_attractive
        calls["kwargs"] = dict(kwargs)
        return np.zeros(np.asarray(affs).shape[1:], dtype=np.uint32)

    monkeypatch.setattr(mws, "AFFOGATO_AVAILABLE", True)
    monkeypatch.setattr(mws, "compute_mws_segmentation", fake_compute, raising=False)
    return calls


def test_raises_without_affogato(monkeypatch):
    monkeypatch.setattr(mws, "AFFOGATO_AVAILABLE", False)
    with pytest.raises(ImportError, match="conda-forge"):
        mws.decode_mutex_watershed(np.zeros((6, 2, 4, 4), dtype=np.float32))


def test_requires_4d_input(monkeypatch):
    _install_fake(monkeypatch)
    with pytest.raises(ValueError, match=r"\(C, Z, Y, X\)"):
        mws.decode_mutex_watershed(np.zeros((6, 4, 4), dtype=np.float32))


def test_default_offsets_only_for_six_channels(monkeypatch):
    # 6 channels -> default banis layout is used.
    calls = _install_fake(monkeypatch)
    mws.decode_mutex_watershed(np.zeros((6, 2, 4, 4), dtype=np.float32))
    assert len(calls["offsets"]) == 6
    assert calls["offsets"][0] == (1, 0, 0)
    assert calls["offsets"][3] == (10, 0, 0)
    # Non-6 channel count with no offsets -> explicit error.
    with pytest.raises(ValueError, match="offsets must be provided"):
        mws.decode_mutex_watershed(np.zeros((4, 2, 4, 4), dtype=np.float32))


def test_offset_channel_mismatch(monkeypatch):
    _install_fake(monkeypatch)
    with pytest.raises(ValueError, match="must match affinity"):
        mws.decode_mutex_watershed(
            np.zeros((6, 2, 4, 4), dtype=np.float32),
            offsets=["1-0-0", "0-1-0", "0-0-1"],  # only 3 for 6 channels
        )


def test_num_attractive_bounds(monkeypatch):
    _install_fake(monkeypatch)
    with pytest.raises(ValueError, match="num_attractive"):
        mws.decode_mutex_watershed(
            np.zeros((6, 2, 4, 4), dtype=np.float32), num_attractive=6
        )


def test_repulsive_channels_inverted_by_default(monkeypatch):
    calls = _install_fake(monkeypatch)
    pred = np.zeros((6, 2, 4, 4), dtype=np.float32)
    pred[:3] = 0.9  # attractive: high = merge
    pred[3:] = 0.8  # long-range: net predicts P(same); MWS needs repulsion
    mws.decode_mutex_watershed(pred)
    passed = calls["affs"]
    # Attractive channels untouched, repulsive inverted (1 - 0.8 = 0.2).
    assert np.allclose(passed[:3], 0.9)
    assert np.allclose(passed[3:], 0.2)
    assert calls["num_attractive"] == 3


def test_inversion_flags_can_be_disabled(monkeypatch):
    calls = _install_fake(monkeypatch)
    pred = np.full((6, 2, 4, 4), 0.7, dtype=np.float32)
    mws.decode_mutex_watershed(pred, invert_repulsive=False)
    assert np.allclose(calls["affs"], 0.7)  # nothing inverted


def test_strides_forwarded(monkeypatch):
    calls = _install_fake(monkeypatch)
    mws.decode_mutex_watershed(
        np.zeros((6, 2, 8, 8), dtype=np.float32),
        strides=[1, 4, 4],
        randomize_strides=False,
    )
    assert calls["kwargs"]["strides"] == [1, 4, 4]
    assert calls["kwargs"]["randomize_strides"] is False


def test_registered_in_registry():
    from connectomics.decoding.registry import get_decoder, list_decoders

    assert "decode_mutex_watershed" in list_decoders()
    assert callable(get_decoder("decode_mutex_watershed"))

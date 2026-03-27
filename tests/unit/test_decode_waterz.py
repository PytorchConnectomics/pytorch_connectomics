"""Unit tests for the WaterZ decoder wrapper."""

from __future__ import annotations

import numpy as np

from connectomics.decoding.decoders import waterz as waterz_decoder


class _FakeWaterzModule:
    """Minimal waterz stub for testing wrapper behavior."""

    def __init__(self):
        self.merge_dust_calls = []
        self.waterz_calls = []

    def waterz(self, affs, thresholds, **kwargs):
        self.waterz_calls.append(kwargs.copy())
        seg = np.zeros(affs.shape[1:], dtype=np.uint64)
        seg[:, :, :2] = 1
        seg[:, :, 2:] = 2
        return [seg.copy() for _ in thresholds]

    def merge_dust(self, seg, affs, size_th, weight_th, dust_th,
                   scoring_function, channels="all"):
        self.merge_dust_calls.append(
            {
                "seg_shape": seg.shape,
                "aff_shape": affs.shape,
                "size_th": size_th,
                "weight_th": weight_th,
                "dust_th": dust_th,
                "scoring_function": scoring_function,
            }
        )


def test_decode_waterz_skips_dust_postprocessing_when_disabled(monkeypatch):
    fake_waterz = _FakeWaterzModule()
    monkeypatch.setattr(waterz_decoder, "waterz", fake_waterz)
    monkeypatch.setattr(waterz_decoder, "WATERZ_AVAILABLE", True)

    predictions = np.ones((3, 4, 4, 4), dtype=np.float32)

    seg = waterz_decoder.decode_waterz(
        predictions,
        thresholds=0.4,
        dust_merge=False,
        dust_merge_size=100,
        dust_merge_affinity=0.3,
        dust_remove_size=50,
    )

    assert seg.shape == (4, 4, 4)
    assert fake_waterz.waterz_calls == [
        {
            "scoring_function": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256>>",
            "aff_threshold_low": 0.0001,
            "aff_threshold_high": 0.9999,
        }
    ]
    assert fake_waterz.merge_dust_calls == []


def test_decode_waterz_dust_merge_uses_same_scoring_function(monkeypatch):
    """Dust merge rebuilds graph with same scoring as agglomeration (OneMinus stripped)."""
    fake_waterz = _FakeWaterzModule()
    monkeypatch.setattr(waterz_decoder, "waterz", fake_waterz)
    monkeypatch.setattr(waterz_decoder, "WATERZ_AVAILABLE", True)

    predictions = np.ones((3, 4, 4, 4), dtype=np.float32)

    waterz_decoder.decode_waterz(
        predictions,
        thresholds=0.4,
        merge_function="aff85_his256",
        dust_merge=True,
        dust_merge_size=100,
        dust_merge_affinity=0.3,
        dust_remove_size=50,
    )

    assert fake_waterz.waterz_calls == [
        {
            "scoring_function": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 85, ScoreValue, 256>>",
            "aff_threshold_low": 0.0001,
            "aff_threshold_high": 0.9999,
        }
    ]
    assert fake_waterz.merge_dust_calls == [
        {
            "seg_shape": (4, 4, 4),
            "aff_shape": (3, 4, 4, 4),
            "size_th": 100,
            "weight_th": 0.3,
            "dust_th": 50,
            "scoring_function": "HistogramQuantileAffinity<RegionGraphType, 85, ScoreValue, 256>",
        }
    ]


def test_decode_waterz_dust_merge_strips_one255minus(monkeypatch):
    """One255Minus wrapper is also stripped for dust merge scoring."""
    fake_waterz = _FakeWaterzModule()
    monkeypatch.setattr(waterz_decoder, "waterz", fake_waterz)
    monkeypatch.setattr(waterz_decoder, "WATERZ_AVAILABLE", True)

    predictions = np.ones((3, 4, 4, 4), dtype=np.float32)

    waterz_decoder.decode_waterz(
        predictions,
        thresholds=0.4,
        merge_function="aff50_his256_ran255",
        dust_merge=True,
        dust_merge_size=100,
        dust_merge_affinity=0.3,
        dust_remove_size=50,
    )

    call = fake_waterz.merge_dust_calls[0]
    assert call["scoring_function"] == "HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256>"

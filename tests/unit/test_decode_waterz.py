"""Unit tests for the WaterZ decoder wrapper."""

from __future__ import annotations

import numpy as np

from connectomics.decoding.decoders import waterz as waterz_decoder


class _FakeWaterzModule:
    """Minimal waterz stub for testing wrapper behavior."""

    def __init__(self):
        self.merge_dust_calls = []
        self.merge_segments_calls = []
        self.waterz_calls = []

    def waterz(self, affs, thresholds, **kwargs):
        self.waterz_calls.append(kwargs.copy())
        seg = np.zeros(affs.shape[1:], dtype=np.uint64)
        seg[:, :, :2] = 1
        seg[:, :, 2:] = 2
        if kwargs.get("return_region_graph", False):
            # ScoredEdge dicts: score=0.2 → affinity = 1.0 - 0.2 = 0.8
            rg = [{"u": 1, "v": 2, "score": 0.2}]
            return [(seg.copy(), list(rg)) for _ in thresholds]
        return [seg.copy() for _ in thresholds]

    def merge_dust(self, seg, affs, size_th, weight_th, dust_th):
        self.merge_dust_calls.append(
            {
                "seg_shape": seg.shape,
                "aff_shape": affs.shape,
                "size_th": size_th,
                "weight_th": weight_th,
                "dust_th": dust_th,
            }
        )

    def merge_segments(
        self,
        seg,
        rg_affs,
        id1,
        id2,
        counts,
        size_th,
        weight_th,
        dust_th,
    ):
        self.merge_segments_calls.append(
            {
                "seg_shape": seg.shape,
                "rg_affs": rg_affs.tolist(),
                "id1": id1.tolist(),
                "id2": id2.tolist(),
                "counts": counts.tolist(),
                "size_th": size_th,
                "weight_th": weight_th,
                "dust_th": dust_th,
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
            "return_region_graph": False,
        }
    ]
    assert fake_waterz.merge_dust_calls == []
    assert fake_waterz.merge_segments_calls == []


def test_decode_waterz_reuses_region_graph_for_dust_when_scores_are_compatible(
    monkeypatch,
):
    fake_waterz = _FakeWaterzModule()
    monkeypatch.setattr(waterz_decoder, "waterz", fake_waterz)
    monkeypatch.setattr(waterz_decoder, "WATERZ_AVAILABLE", True)

    predictions = np.ones((3, 4, 4, 4), dtype=np.float32)

    waterz_decoder.decode_waterz(
        predictions,
        thresholds=0.4,
        dust_merge=True,
        dust_merge_size=100,
        dust_merge_affinity=0.3,
        dust_remove_size=50,
    )

    assert fake_waterz.waterz_calls == [
        {
            "scoring_function": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256>>",
            "aff_threshold_low": 0.0001,
            "aff_threshold_high": 0.9999,
            "return_region_graph": True,
        }
    ]
    assert fake_waterz.merge_dust_calls == []
    assert fake_waterz.merge_segments_calls == [
        {
            "seg_shape": (4, 4, 4),
            "rg_affs": [0.800000011920929],
            "id1": [1],
            "id2": [2],
            "counts": [0, 32, 32],
            "size_th": 100,
            "weight_th": 0.3,
            "dust_th": 50,
        }
    ]


def test_decode_waterz_reuses_region_graph_for_any_scoring_function(
    monkeypatch,
):
    """Region graph reuse works for any scoring function, not just OneMinus."""
    fake_waterz = _FakeWaterzModule()
    monkeypatch.setattr(waterz_decoder, "waterz", fake_waterz)
    monkeypatch.setattr(waterz_decoder, "WATERZ_AVAILABLE", True)

    predictions = np.ones((3, 4, 4, 4), dtype=np.float32)
    scoring_function = "Invert<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256>>"

    waterz_decoder.decode_waterz(
        predictions,
        thresholds=0.4,
        merge_function=scoring_function,
        dust_merge=True,
        dust_merge_size=100,
        dust_merge_affinity=0.3,
        dust_remove_size=50,
    )

    assert fake_waterz.waterz_calls == [
        {
            "scoring_function": scoring_function,
            "aff_threshold_low": 0.0001,
            "aff_threshold_high": 0.9999,
            "return_region_graph": True,
        }
    ]
    assert fake_waterz.merge_dust_calls == []
    assert len(fake_waterz.merge_segments_calls) == 1
    call = fake_waterz.merge_segments_calls[0]
    assert call["seg_shape"] == (4, 4, 4)
    assert call["size_th"] == 100
    assert call["weight_th"] == 0.3
    assert call["dust_th"] == 50

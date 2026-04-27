"""Unit tests for the WaterZ decoder wrapper."""

from __future__ import annotations

import numpy as np

from connectomics.decoding.decoders import waterz as waterz_decoder


class _FakeWaterzModule:
    """Minimal waterz stub for testing wrapper behavior."""

    def __init__(self):
        self.merge_segments_calls = []
        self.waterz_calls = []

    def waterz(self, affs, thresholds, **kwargs):
        self.waterz_calls.append(kwargs.copy())
        seg = np.zeros(affs.shape[1:], dtype=np.uint64)
        seg[:, :, :2] = 1
        seg[:, :, 2:] = 2
        if kwargs.get("return_region_graph", False):
            # ScoredEdge dicts from extractRegionGraph (OneMinus scores).
            # score=0.2 → affinity = 1.0 - 0.2 = 0.8
            rg = [{"u": 1, "v": 2, "score": 0.2}]
            return [(seg.copy(), list(rg)) for _ in thresholds]
        return [seg.copy() for _ in thresholds]

    def merge_segments(self, seg, rg_affs, id1, id2, counts,
                       size_th, weight_th, dust_th):
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


def test_decode_waterz_skips_dust_when_disabled(monkeypatch):
    fake_waterz = _FakeWaterzModule()
    monkeypatch.setattr(waterz_decoder, "waterz", fake_waterz)
    monkeypatch.setattr(waterz_decoder, "WATERZ_AVAILABLE", True)

    seg = waterz_decoder.decode_waterz(
        np.ones((3, 4, 4, 4), dtype=np.float32),
        thresholds=0.4,
        dust_merge=False,
        dust_merge_size=100,
    )

    assert seg.shape == (4, 4, 4)
    assert fake_waterz.waterz_calls[0]["return_region_graph"] is False
    assert fake_waterz.merge_segments_calls == []


def test_decode_waterz_reuses_agglomeration_graph_for_dust(monkeypatch):
    """Dust merge reuses agglomeration's region graph with inverted OneMinus scores."""
    fake_waterz = _FakeWaterzModule()
    monkeypatch.setattr(waterz_decoder, "waterz", fake_waterz)
    monkeypatch.setattr(waterz_decoder, "WATERZ_AVAILABLE", True)

    dust_calls = []

    def _fake_dust_merge(seg, rg, *, is_uint8, size_th, weight_th, dust_th):
        dust_calls.append({
            "seg_shape": seg.shape,
            "rg": rg,
            "is_uint8": is_uint8,
            "size_th": size_th,
            "weight_th": weight_th,
            "dust_th": dust_th,
        })

    monkeypatch.setattr(waterz_decoder, "dust_merge_from_region_graph", _fake_dust_merge)

    waterz_decoder.decode_waterz(
        np.ones((3, 4, 4, 4), dtype=np.float32),
        thresholds=0.4,
        dust_merge=True,
        dust_merge_size=100,
        dust_merge_affinity=0.3,
        dust_remove_size=50,
    )

    assert fake_waterz.waterz_calls[0]["return_region_graph"] is True
    assert len(dust_calls) == 1
    call = dust_calls[0]
    assert call["seg_shape"] == (4, 4, 4)
    assert call["rg"] == [{"u": 1, "v": 2, "score": 0.2}]
    assert call["is_uint8"] is False
    assert call["size_th"] == 100
    assert call["weight_th"] == 0.3
    assert call["dust_th"] == 50

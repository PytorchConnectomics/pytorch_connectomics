"""Smoke tests for connectomics.decoding.qc.affinity."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from connectomics.config import load_config
from connectomics.decoding.qc.affinity import (
    AffinityQCAccumulator,
    AffinityQCParams,
    begin_streaming_qc,
    finish_streaming_qc,
    render_markdown_report,
    run_affinity_qc,
    scan_prediction,
)
from connectomics.runtime.preflight import validate_runtime_coherence


def _synthetic_pred(shape=(3, 16, 16, 40), bad_z=(0, 1, 38, 39)):
    """Affinity-like (C, X, Y, Z) float32 where edge slices are intentionally low."""
    pred = np.full(shape, 0.6, dtype=np.float32)
    for z in bad_z:
        pred[..., z] = 0.0
    return pred


def test_scan_prediction_finds_edge_outliers():
    pred = _synthetic_pred()
    params = AffinityQCParams(z_stride=1, k_edge=4, refine_window=8, drift_thresh=0.05)
    report = scan_prediction(pred, img=None, params=params)

    assert report.low_z == 2
    assert report.high_z == 38
    assert report.nan_count == 0 and report.inf_count == 0
    assert report.interior_mean.shape == (pred.shape[0],)
    assert pytest.approx(float(report.interior_mean.mean()), abs=1e-4) == 0.6


def test_render_markdown_has_frontmatter():
    pred = _synthetic_pred()
    params = AffinityQCParams(z_stride=1, k_edge=4, refine_window=8)
    report = scan_prediction(pred, img=None, params=params)
    md = render_markdown_report(
        report, params,
        pred_desc="synthetic", img_desc="",
        mask_path="/tmp/mask.h5", image_path="",
    )
    assert md.startswith("---\n")
    assert f"low_z: {report.low_z}" in md
    assert f"high_z: {report.high_z}" in md


def test_run_affinity_qc_writes_mask_and_wires_path(tmp_path):
    from types import SimpleNamespace

    pred = _synthetic_pred()
    mask_p = tmp_path / "affinity_mask.h5"
    report_p = tmp_path / "affinity_qc_report.md"
    qc = SimpleNamespace(
        enabled=True, image_path="", mask_path=str(mask_p), report_path=str(report_p),
        z_stride=1, k_edge=4, refine_window=8, drift_thresh=0.05,
        border_width=0, bg_thresh=30, n_z_border=4,
    )
    decoding = SimpleNamespace(
        affinity_qc=qc, affinity_mask_path="", save_path=str(tmp_path),
    )
    cfg = SimpleNamespace(decoding=decoding, inference=None)

    out = run_affinity_qc(cfg, pred)
    assert out == str(mask_p)
    assert mask_p.exists()
    assert report_p.exists()
    assert cfg.decoding.affinity_mask_path == str(mask_p)

    with h5py.File(str(mask_p), "r") as f:
        mask = f["main"][...]
        assert mask.shape == pred.shape[1:]
        # Edge z bands zeroed; interior kept.
        assert mask[..., 0].sum() == 0
        assert mask[..., 1].sum() == 0
        assert mask[..., 5].all()


def test_streaming_accumulator_matches_one_shot_scan():
    """The streaming accumulator should derive the same z-cuts as scan_prediction."""
    pred = _synthetic_pred()  # (C=3, X=16, Y=16, Z=40), bad z at edges
    params = AffinityQCParams(z_stride=1, k_edge=4, refine_window=8, drift_thresh=0.05)
    ref = scan_prediction(pred, img=None, params=params)

    # Stream as (C, Zslab, Y, X) slabs (matches chunked stitcher layout).
    czyx = np.moveaxis(pred, -1, 1)  # (C, Z, Y, X) ≡ (C, Zslab, Y, X) when sliced
    acc = AffinityQCAccumulator(channel_count=pred.shape[0], z_extent=pred.shape[-1],
                                params=params)
    slab_size = 7
    for z0 in range(0, czyx.shape[1], slab_size):
        z1 = min(z0 + slab_size, czyx.shape[1])
        slab = czyx[:, z0:z1]  # (C, Zslab, Y, X)
        acc.update(slab, z_offset=z0, z_axis=1)
    streamed = acc.finalize()
    assert streamed.low_z == ref.low_z
    assert streamed.high_z == ref.high_z
    assert streamed.nan_count == ref.nan_count


def test_streaming_interior_mean_matches_post_save():
    rng = np.random.default_rng(0)
    pred = (rng.random((3, 8, 8, 100)).astype(np.float32) * 0.4 + 0.3)
    params = AffinityQCParams(
        z_stride=5,
        k_edge=4,
        refine_window=30,
        drift_thresh=0.05,
    )
    ref = scan_prediction(pred, img=None, params=params)

    acc = AffinityQCAccumulator(channel_count=3, z_extent=100, params=params)
    acc.update(np.moveaxis(pred, -1, 1), z_offset=0, z_axis=1)
    streamed = acc.finalize()

    np.testing.assert_allclose(streamed.interior_mean, ref.interior_mean, atol=1e-5)
    assert streamed.low_z == ref.low_z
    assert streamed.high_z == ref.high_z
    assert streamed.z_idx.shape == (100,)
    assert ref.z_idx.shape == (20,)


def test_streaming_finalize_matches_post_save_when_head_window_fully_bad():
    pred = np.full((3, 16, 16, 60), 0.6, dtype=np.float32)
    pred[..., 0:13] = 0.0
    params = AffinityQCParams(
        z_stride=1,
        k_edge=4,
        refine_window=10,
        drift_thresh=0.05,
    )
    ref = scan_prediction(pred, img=None, params=params)

    acc = AffinityQCAccumulator(channel_count=3, z_extent=60, params=params)
    acc.update(np.moveaxis(pred, -1, 1), z_offset=0, z_axis=1)
    streamed = acc.finalize()

    assert ref.low_z == 10
    assert ref.high_z == 60
    assert streamed.low_z == 10
    assert streamed.high_z == 60


def test_streaming_accumulator_finalize_all_zero_message():
    params = AffinityQCParams()
    acc = AffinityQCAccumulator(channel_count=3, z_extent=10, params=params)

    with pytest.raises(
        ValueError,
        match=(
            "no slabs were folded in\\. In distributed runs, "
            "finish_streaming_qc must only be called on rank 0\\."
        ),
    ):
        acc.finalize()


def test_streaming_accumulator_rejects_oob_offsets():
    params = AffinityQCParams()
    acc = AffinityQCAccumulator(channel_count=3, z_extent=10, params=params)
    slab = np.ones((3, 5, 4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="out of bounds"):
        acc.update(slab, z_offset=7, z_axis=1)  # 7+5 > 10


def test_begin_streaming_qc_only_fires_in_streaming_mode():
    from types import SimpleNamespace

    qc = SimpleNamespace(enabled=True, mode="post_save", z_stride=1, k_edge=4,
                         refine_window=8, drift_thresh=0.05, border_width=0,
                         bg_thresh=30, n_z_border=4)
    cfg = SimpleNamespace(decoding=SimpleNamespace(affinity_qc=qc), inference=None)
    assert begin_streaming_qc(cfg, channel_count=3, z_extent=40) is None
    qc.mode = "streaming"
    assert begin_streaming_qc(cfg, channel_count=3, z_extent=40) is not None
    qc.enabled = False
    assert begin_streaming_qc(cfg, channel_count=3, z_extent=40) is None


def test_begin_streaming_qc_skips_when_mask_already_wired():
    from types import SimpleNamespace

    qc = SimpleNamespace(
        enabled=True,
        mode="streaming",
        z_stride=1,
        k_edge=4,
        refine_window=8,
        drift_thresh=0.05,
        border_width=0,
        bg_thresh=30,
        n_z_border=4,
    )
    decoding = SimpleNamespace(
        affinity_qc=qc,
        affinity_mask_path="/tmp/dummy.h5",
    )
    cfg = SimpleNamespace(decoding=decoding, inference=None)

    assert begin_streaming_qc(cfg, channel_count=3, z_extent=40) is None


def test_finish_streaming_qc_writes_mask_and_wires_path(tmp_path):
    from types import SimpleNamespace

    pred = _synthetic_pred()
    params = AffinityQCParams(z_stride=1, k_edge=4, refine_window=8, drift_thresh=0.05)
    acc = AffinityQCAccumulator(channel_count=pred.shape[0], z_extent=pred.shape[-1],
                                params=params)
    czyx = np.moveaxis(pred, -1, 1)
    acc.update(czyx, z_offset=0, z_axis=1)

    qc = SimpleNamespace(
        enabled=True, mode="streaming", image_path="",
        mask_path=str(tmp_path / "mask.h5"),
        report_path=str(tmp_path / "report.md"),
        z_stride=1, k_edge=4, refine_window=8, drift_thresh=0.05,
        border_width=0, bg_thresh=30, n_z_border=4,
    )
    decoding = SimpleNamespace(
        affinity_qc=qc, affinity_mask_path="", save_path=str(tmp_path),
    )
    cfg = SimpleNamespace(decoding=decoding, inference=None)

    # No image path → finish should raise (streaming requires image for mask build).
    with pytest.raises(ValueError, match="image_path"):
        finish_streaming_qc(cfg, acc)


def test_preflight_streaming_image_path_rules():
    cfg = load_config("tutorials/neuron_nisb/liconn_banis_v3_erosion2.yaml")
    cfg.decoding.affinity_qc.enabled = True
    cfg.decoding.affinity_qc.mode = "streaming"
    cfg.decoding.affinity_qc.image_path = ""
    cfg.decoding.affinity_mask_path = ""
    cfg.inference.strategy = "chunked"

    with pytest.raises(ValueError, match="image_path|affinity_mask_path"):
        validate_runtime_coherence(cfg)

    cfg.decoding.affinity_mask_path = "/tmp/preexisting.h5"
    validate_runtime_coherence(cfg)

    cfg.decoding.affinity_mask_path = ""
    cfg.decoding.affinity_qc.image_path = "/tmp/dummy.zarr"
    validate_runtime_coherence(cfg)


def test_run_affinity_qc_streaming_short_circuit_when_mask_already_wired(tmp_path):
    """If chunked inference already populated the mask, decoding-stage QC is a no-op."""
    from types import SimpleNamespace

    pred = _synthetic_pred()
    existing_mask = tmp_path / "existing_mask.h5"
    qc = SimpleNamespace(enabled=True, mode="streaming")
    decoding = SimpleNamespace(
        affinity_qc=qc, affinity_mask_path=str(existing_mask),
    )
    cfg = SimpleNamespace(decoding=decoding, inference=None)
    assert run_affinity_qc(cfg, pred) == str(existing_mask)


def test_run_affinity_qc_disabled_returns_none():
    from types import SimpleNamespace

    pred = _synthetic_pred()
    cfg = SimpleNamespace(
        decoding=SimpleNamespace(
            affinity_qc=SimpleNamespace(enabled=False),
            affinity_mask_path="",
        ),
        inference=None,
    )
    assert run_affinity_qc(cfg, pred) is None

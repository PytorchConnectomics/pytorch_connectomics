from __future__ import annotations

import numpy as np
import pytest
import torch

from connectomics.config import Config, validate_config
from connectomics.data.io import write_hdf5
from connectomics.decoding.streamed_chunked import UnionFind, _union_face_pairs
from connectomics.inference.chunked import (
    _build_chunk_grid,
    _validate_chunked_output_contract,
    run_chunked_prediction_inference,
)
from connectomics.inference.lazy import lazy_predict_volume


def _patch_mean_forward(x: torch.Tensor) -> torch.Tensor:
    return x + x.mean(dim=(2, 3, 4), keepdim=True)


def test_build_chunk_grid_covers_volume_without_overlap():
    chunks = _build_chunk_grid((5, 7, 9), (2, 3, 4))

    coverage = np.zeros((5, 7, 9), dtype=np.uint8)
    for chunk in chunks:
        assert all(0 <= start < stop for start, stop in zip(chunk.start, chunk.stop))
        assert all(stop <= bound for stop, bound in zip(chunk.stop, (5, 7, 9)))
        coverage[chunk.slices] += 1

    assert len(chunks) == 27
    assert np.all(coverage == 1)


def test_union_face_pairs_respects_affinity_mask_and_min_contact():
    uf = UnionFind()
    src_face = np.array([[1, 1], [2, 2]], dtype=np.uint32)
    dst_face = np.array([[3, 3], [4, 5]], dtype=np.uint32)
    seam_affinity = np.ones((2, 2), dtype=bool)

    assert _union_face_pairs(uf, src_face, dst_face, seam_affinity, min_contact=2) == 1
    assert uf.find(1) == uf.find(3)
    assert uf.find(2) != uf.find(4)
    assert uf.find(2) != uf.find(5)


def test_validate_config_accepts_chunked_inference():
    cfg = Config()
    cfg.data.dataloader.patch_size = [128, 128, 128]
    cfg.inference.strategy = "chunked"
    cfg.inference.chunking.enabled = True
    cfg.inference.chunking.output_mode = "raw_prediction"
    cfg.inference.chunking.chunk_size = [64, 512, 512]
    cfg.inference.chunking.halo = [16, 64, 64]

    validate_config(cfg)


def test_validate_config_rejects_unknown_chunk_output_mode():
    cfg = Config()
    cfg.data.dataloader.patch_size = [128, 128, 128]
    cfg.inference.strategy = "chunked"
    cfg.inference.chunking.enabled = True
    cfg.inference.chunking.output_mode = "labels"
    cfg.inference.chunking.chunk_size = [64, 512, 512]
    cfg.inference.chunking.halo = [16, 64, 64]

    with pytest.raises(ValueError, match="output_mode"):
        validate_config(cfg)


def test_chunked_output_contract_allows_later_decode_postprocessing():
    cfg = Config()
    cfg.decoding.postprocessing.enabled = True
    cfg.decoding.postprocessing.output_transpose = [2, 1, 0]

    _validate_chunked_output_contract(cfg)


def test_chunked_raw_prediction_matches_full_lazy_prediction(tmp_path):
    cfg = Config()
    cfg.data.image_transform.normalize = "none"
    cfg.data.dataloader.patch_size = [3, 3, 3]
    cfg.data.dataloader.batch_size = 2
    cfg.model.output_size = [3, 3, 3]
    cfg.inference.strategy = "chunked"
    cfg.inference.sliding_window.window_size = [3, 3, 3]
    cfg.inference.sliding_window.overlap = 0.5
    cfg.inference.sliding_window.blending = "constant"
    cfg.inference.sliding_window.snap_to_edge = True
    cfg.inference.chunking.enabled = True
    cfg.inference.chunking.output_mode = "raw_prediction"
    cfg.inference.chunking.chunk_size = [2, 4, 7]
    cfg.inference.chunking.halo = [0, 0, 0]

    image_path = tmp_path / "chunked_raw_input.h5"
    output_path = tmp_path / "chunked_raw_prediction.h5"
    volume = np.arange(5 * 6 * 7, dtype=np.float32).reshape(5, 6, 7)
    write_hdf5(str(image_path), volume, dataset="main")

    full = lazy_predict_volume(cfg, _patch_mean_forward, str(image_path), device="cpu")
    run_chunked_prediction_inference(
        cfg,
        _patch_mean_forward,
        str(image_path),
        output_path=output_path,
        device="cpu",
        checkpoint_path="checkpoint.ckpt",
    )

    import h5py

    with h5py.File(output_path, "r") as handle:
        raw = np.asarray(handle["main"])
        assert handle["main"].attrs["checkpoint_path"] == "checkpoint.ckpt"

    assert raw.shape == tuple(full.shape[1:])
    assert np.allclose(raw, full.numpy()[0], atol=1.0e-5)

from __future__ import annotations

import numpy as np
import pytest

from connectomics.config import Config, validate_config
from connectomics.inference.chunked import (
    UnionFind,
    _build_chunk_grid,
    _union_face_pairs,
    _validate_chunked_output_contract,
)


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
    cfg.inference.chunking.chunk_size = [64, 512, 512]
    cfg.inference.chunking.halo = [16, 64, 64]

    validate_config(cfg)


def test_chunked_output_contract_rejects_transpose_postprocessing():
    cfg = Config()
    cfg.inference.postprocessing.enabled = True
    cfg.inference.postprocessing.output_transpose = [2, 1, 0]

    with pytest.raises(ValueError, match="output_transpose"):
        _validate_chunked_output_contract(cfg)

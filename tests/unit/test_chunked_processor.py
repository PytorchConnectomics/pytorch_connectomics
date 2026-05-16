"""Unit tests for connectomics.chunked.ChunkedProcessor.

Cover the parent class's plumbing in isolation: chunk-grid drive, async
writer, resume manifest. The kimimaro subclass is tested separately in
tests/unit/test_chunked_skeleton.py.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from connectomics.chunked import (
    ChunkedProcessor,
    ChunkedProcessorConfig,
    ManifestConfigMismatch,
    ResumeManifest,
    build_chunk_grid,
)
from connectomics.chunked.chunk_grid import ChunkRef


class _DoubleProcessor(ChunkedProcessor):
    """Toy subclass: output = input * 2 (cast to uint16)."""

    def process_chunk(self, chunk_data: np.ndarray, chunk) -> np.ndarray:
        return (chunk_data.astype(np.int32) * 2).astype(np.uint16)


def _write_seg(path, shape, seed=0):
    rng = np.random.default_rng(seed)
    vol = rng.integers(0, 100, size=shape, dtype=np.uint32)
    with h5py.File(path, "w") as f:
        f.create_dataset("main", data=vol, chunks=tuple(min(s, 16) for s in shape))
    return vol


def test_double_processor_parallel_writes_correct_volume(tmp_path):
    in_path = tmp_path / "in.h5"
    out_path = tmp_path / "out.h5"
    shape = (32, 32, 32)
    vol = _write_seg(in_path, shape, seed=1)

    cfg = ChunkedProcessorConfig(
        input_path=str(in_path),
        output_path=str(out_path),
        chunk_shape=(16, 16, 16),
        parallel=2,
        output_dtype="uint16",
    )
    proc = _DoubleProcessor(cfg)
    proc.run()

    with h5py.File(out_path, "r") as f:
        out = f["main"][...]
    np.testing.assert_array_equal(out, (vol.astype(np.int32) * 2).astype(np.uint16))

    # Manifest records every chunk.
    manifest_path = out_path.with_suffix(out_path.suffix + ".chunks.json")
    assert manifest_path.exists()
    grid = build_chunk_grid(shape, (16, 16, 16))
    expected_keys = {c.key for c in grid}
    import json

    payload = json.loads(manifest_path.read_text())
    assert set(payload["completed"]) == expected_keys


def test_resume_skips_completed_chunks(tmp_path):
    in_path = tmp_path / "in.h5"
    out_path = tmp_path / "out.h5"
    shape = (24, 24, 24)
    vol = _write_seg(in_path, shape, seed=2)
    chunk_shape = (12, 12, 12)
    cfg = ChunkedProcessorConfig(
        input_path=str(in_path),
        output_path=str(out_path),
        chunk_shape=chunk_shape,
        parallel=1,
        output_dtype="uint16",
    )

    # First run completes everything.
    _DoubleProcessor(cfg).run()
    with h5py.File(out_path, "r") as f:
        out_first = f["main"][...]

    # Hand-pollute one chunk on disk to make sure a resume call does NOT
    # overwrite already-completed chunks.
    grid = build_chunk_grid(shape, chunk_shape)
    target = grid[0]
    with h5py.File(out_path, "a") as f:
        f["main"][target.slices] = np.zeros(target.shape, dtype=np.uint16)

    # Re-run with the same config (overwrite=False). Manifest says all chunks
    # are done, so nothing should re-process and the pollution stays.
    _DoubleProcessor(cfg).run()
    with h5py.File(out_path, "r") as f:
        out_second = f["main"][...]
    assert np.all(out_second[target.slices] == 0), "Resume re-wrote a completed chunk."

    # Sanity: the non-polluted chunks still match the original double-output.
    for c in grid[1:]:
        np.testing.assert_array_equal(
            out_second[c.slices], (vol[c.slices].astype(np.int32) * 2).astype(np.uint16)
        )


def test_overwrite_resets_manifest_and_recomputes(tmp_path):
    in_path = tmp_path / "in.h5"
    out_path = tmp_path / "out.h5"
    shape = (16, 16, 16)
    vol = _write_seg(in_path, shape, seed=3)

    cfg = ChunkedProcessorConfig(
        input_path=str(in_path),
        output_path=str(out_path),
        chunk_shape=(8, 8, 8),
        parallel=1,
        output_dtype="uint16",
    )
    _DoubleProcessor(cfg).run()

    # Mutate input, run with overwrite=True; output should reflect new input.
    vol2 = vol + 1
    with h5py.File(in_path, "w") as f:
        f.create_dataset("main", data=vol2, chunks=(8, 8, 8))
    cfg.overwrite = True
    _DoubleProcessor(cfg).run()
    with h5py.File(out_path, "r") as f:
        out = f["main"][...]
    np.testing.assert_array_equal(out, (vol2.astype(np.int32) * 2).astype(np.uint16))


def test_manifest_config_mismatch_raises(tmp_path):
    path = tmp_path / "out.h5.chunks.json"
    ResumeManifest.load_or_create(
        path, {"chunk_shape": [16, 16, 16], "overlap": 0, "output_dtype": "uint16", "output_shape": [32, 32, 32]}
    ).mark_completed("z0_y0_x0")

    with pytest.raises(ManifestConfigMismatch):
        ResumeManifest.load_or_create(
            path,
            {"chunk_shape": [8, 8, 8], "overlap": 0, "output_dtype": "uint16", "output_shape": [32, 32, 32]},
        )


def test_chunk_ref_and_grid_basic():
    grid = build_chunk_grid((10, 8, 6), (4, 4, 4))
    assert all(isinstance(c, ChunkRef) for c in grid)
    # last chunk along x is truncated 4 -> 2
    assert grid[-1].stop == (10, 8, 6)
    assert grid[0].key == "z0_y0_x0"
    keys = {c.key for c in grid}
    # 3 * 2 * 2 = 12 chunks
    assert len(keys) == 12

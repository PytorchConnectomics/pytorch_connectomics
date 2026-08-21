from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

DEV = Path(__file__).resolve().parents[2] / "dev" / "zebrafinch"
sys.path.insert(0, str(DEV))

import upload_affinity_full_masked as affinity  # noqa: E402


def test_missing_low_neighbors_fail_closed_unless_isolated_gate(tmp_path, monkeypatch):
    chunks = tmp_path / "chunks"
    chunks.mkdir()
    source = np.arange(3 * 2 * 2 * 2, dtype=np.float32).reshape(3, 2, 2, 2)
    with h5py.File(chunks / "chunk_z1_y1_x1.h5", "w") as handle:
        handle.create_dataset("main", data=source)
    monkeypatch.setattr(affinity, "CHUNKS", chunks)

    with pytest.raises((FileNotFoundError, OSError)):
        affinity._load_affinity_neg_offset("z1_y1_x1", (1, 1, 1))

    shifted = affinity._load_affinity_neg_offset("z1_y1_x1", (1, 1, 1), zero_missing_low_faces=True)
    for channel in range(3):
        low_face = [slice(None)] * 3
        low_face[channel] = 0
        assert np.all(shifted[channel][tuple(low_face)] == 0)
        high = [slice(None)] * 3
        high[channel] = slice(1, None)
        low = [slice(None)] * 3
        low[channel] = slice(0, -1)
        np.testing.assert_array_equal(shifted[channel][tuple(high)], source[channel][tuple(low)])

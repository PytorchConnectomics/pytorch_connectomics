"""Smoke benchmark for chunked raw-prediction writes.

The assertion is intentionally broad: this is a regression tripwire for the V3
chunked artifact refactor, not a statistically rigorous benchmark.
"""

from __future__ import annotations

import time

import h5py
import numpy as np
import torch

from connectomics.config import Config
from connectomics.data.io import write_hdf5
from connectomics.inference.chunked import run_chunked_prediction_inference


def _identity_forward(x: torch.Tensor) -> torch.Tensor:
    return x


def test_chunked_prediction_inference_smoke_throughput(tmp_path):
    cfg = Config()
    cfg.data.image_transform.normalize = "none"
    cfg.data.dataloader.patch_size = [3, 3, 3]
    cfg.data.dataloader.batch_size = 1
    cfg.model.output_size = [3, 3, 3]
    cfg.inference.strategy = "chunked"
    cfg.inference.sliding_window.window_size = [3, 3, 3]
    cfg.inference.sliding_window.overlap = 0.0
    cfg.inference.sliding_window.blending = "constant"
    cfg.inference.sliding_window.snap_to_edge = True
    cfg.inference.chunking.enabled = True
    cfg.inference.chunking.output_mode = "raw_prediction"
    cfg.inference.chunking.chunk_size = [2, 4, 4]
    cfg.inference.chunking.halo = [0, 0, 0]

    image_path = tmp_path / "input.h5"
    output_path = tmp_path / "prediction.h5"
    volume = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
    write_hdf5(str(image_path), volume, dataset="main")

    started = time.perf_counter()
    run_chunked_prediction_inference(
        cfg,
        _identity_forward,
        str(image_path),
        output_path=output_path,
        device="cpu",
    )
    elapsed_s = time.perf_counter() - started

    assert elapsed_s < 10.0
    with h5py.File(output_path, "r") as handle:
        assert handle["main"].shape == (1, 4, 5, 6)
        assert handle["main"].chunks == (1, 4, 5, 6)
        assert handle["main"].compression == "gzip"

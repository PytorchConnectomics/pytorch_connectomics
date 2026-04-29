from __future__ import annotations

import numpy as np
import torch

from connectomics.config import Config
from connectomics.data.augmentation.build import build_test_transforms
from connectomics.data.io import write_hdf5
from connectomics.inference import InferenceManager
from connectomics.inference.lazy import lazy_predict_region, lazy_predict_volume


def _identity_forward(x: torch.Tensor) -> torch.Tensor:
    return x


def _patch_mean_forward(x: torch.Tensor) -> torch.Tensor:
    return x + x.mean(dim=(2, 3, 4), keepdim=True)


def _target_context_forward(x: torch.Tensor) -> torch.Tensor:
    shifted = torch.zeros_like(x)
    shifted[..., 1:, :, :] = x[..., :-1, :, :]
    return x + shifted


def _make_cfg() -> Config:
    cfg = Config()
    cfg.data.image_transform.normalize = "none"
    cfg.inference.test_time_augmentation.enabled = False
    cfg.system.num_workers = 0
    return cfg


def _run_eager_prediction(cfg: Config, image_path: str) -> torch.Tensor:
    transforms = build_test_transforms(cfg, keys=["image"], mode="test")
    batch = transforms({"image": image_path})
    image = batch["image"].unsqueeze(0)
    manager = InferenceManager(cfg=cfg, model=torch.nn.Identity(), forward_fn=_identity_forward)
    return manager.predict_with_tta(image)


def test_lazy_sliding_window_matches_eager_inference(tmp_path):
    cfg = _make_cfg()
    cfg.data.dataloader.patch_size = [2, 3, 3]
    cfg.model.output_size = [2, 3, 3]
    cfg.inference.sliding_window.window_size = [2, 3, 3]
    cfg.inference.sliding_window.overlap = 0.5
    cfg.inference.sliding_window.blending = "gaussian"

    image_path = tmp_path / "lazy_eager_match.h5"
    volume = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
    write_hdf5(str(image_path), volume, dataset="main")

    eager = _run_eager_prediction(cfg, str(image_path))
    lazy = lazy_predict_volume(cfg, _identity_forward, str(image_path), device="cpu")

    assert lazy.shape == eager.shape
    assert torch.allclose(lazy, eager, atol=1.0e-5)


def test_lazy_region_matches_full_volume_global_window_grid(tmp_path):
    cfg = _make_cfg()
    cfg.data.dataloader.patch_size = [3, 3, 3]
    cfg.model.output_size = [3, 3, 3]
    cfg.inference.sliding_window.window_size = [3, 3, 3]
    cfg.inference.sliding_window.overlap = 0.5
    cfg.inference.sliding_window.blending = "constant"
    cfg.inference.sliding_window.snap_to_edge = True

    image_path = tmp_path / "lazy_region_global_grid.h5"
    volume = np.arange(5 * 6 * 7, dtype=np.float32).reshape(5, 6, 7)
    write_hdf5(str(image_path), volume, dataset="main")

    full = lazy_predict_volume(cfg, _patch_mean_forward, str(image_path), device="cpu")
    region = lazy_predict_region(
        cfg,
        _patch_mean_forward,
        str(image_path),
        region_start=(1, 1, 2),
        region_stop=(5, 6, 7),
        device="cpu",
    )

    assert torch.allclose(region, full[..., 1:5, 1:6, 2:7], atol=1.0e-5)


def test_lazy_region_keeps_sub_roi_edge_shape(tmp_path):
    cfg = _make_cfg()
    cfg.data.dataloader.patch_size = [3, 3, 3]
    cfg.model.output_size = [3, 3, 3]
    cfg.inference.sliding_window.window_size = [3, 3, 3]
    cfg.inference.sliding_window.overlap = 0.5
    cfg.inference.sliding_window.blending = "constant"
    cfg.inference.sliding_window.snap_to_edge = True

    image_path = tmp_path / "lazy_region_sub_roi_edge.h5"
    volume = np.arange(5 * 5 * 5, dtype=np.float32).reshape(5, 5, 5)
    write_hdf5(str(image_path), volume, dataset="main")

    full = lazy_predict_volume(cfg, _patch_mean_forward, str(image_path), device="cpu")
    region = lazy_predict_region(
        cfg,
        _patch_mean_forward,
        str(image_path),
        region_start=(4, 3, 2),
        region_stop=(5, 5, 5),
        device="cpu",
    )

    assert region.shape == (1, 1, 1, 2, 3)
    assert torch.allclose(region, full[..., 4:5, 3:5, 2:5], atol=1.0e-5)


def test_lazy_region_honors_target_context(tmp_path):
    cfg = _make_cfg()
    cfg.data.dataloader.patch_size = [3, 3, 3]
    cfg.model.output_size = [3, 3, 3]
    cfg.inference.sliding_window.window_size = [3, 3, 3]
    cfg.inference.sliding_window.overlap = 0.5
    cfg.inference.sliding_window.blending = "constant"
    cfg.inference.sliding_window.target_context = [1, 1, 1]

    image_path = tmp_path / "lazy_region_target_context.h5"
    volume = np.arange(6 * 6 * 6, dtype=np.float32).reshape(6, 6, 6)
    write_hdf5(str(image_path), volume, dataset="main")

    full = lazy_predict_volume(cfg, _target_context_forward, str(image_path), device="cpu")
    region = lazy_predict_region(
        cfg,
        _target_context_forward,
        str(image_path),
        region_start=(1, 1, 1),
        region_stop=(5, 5, 5),
        device="cpu",
    )

    assert torch.allclose(region, full[..., 1:5, 1:5, 1:5], atol=1.0e-5)


def test_lazy_sliding_window_matches_eager_with_x2_resize(tmp_path):
    cfg = _make_cfg()
    cfg.data.dataloader.patch_size = [2, 2, 2]
    cfg.data.data_transform.resize = [4, 4, 4]
    cfg.model.output_size = [4, 4, 4]
    cfg.inference.sliding_window.window_size = [4, 4, 4]
    cfg.inference.sliding_window.overlap = 0.5
    cfg.inference.sliding_window.blending = "gaussian"

    image_path = tmp_path / "lazy_resize_match.h5"
    volume = np.linspace(0.0, 1.0, num=4 * 4 * 4, dtype=np.float32).reshape(4, 4, 4)
    write_hdf5(str(image_path), volume, dataset="main")

    eager = _run_eager_prediction(cfg, str(image_path))
    lazy = lazy_predict_volume(cfg, _identity_forward, str(image_path), device="cpu")

    assert lazy.shape == eager.shape
    assert torch.allclose(lazy, eager, atol=1.0e-5)

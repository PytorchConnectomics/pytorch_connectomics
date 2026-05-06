from __future__ import annotations

import numpy as np
import torch

from connectomics.config import Config
from connectomics.data.augmentation.build import build_test_transforms
from connectomics.data.io import write_hdf5
from connectomics.inference import InferenceManager
from connectomics.inference.lazy import (
    _build_intersecting_window_slices,
    _build_window_slices,
    lazy_predict_region,
    lazy_predict_volume,
)
from connectomics.inference.sliding import build_sliding_importance_map


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


def test_distance_transform_blending_matches_banis_weight_map():
    importance = build_sliding_importance_map(
        (5, 5, 5),
        mode="distance_transform",
        sigma_scale=0.125,
        device="cpu",
        dtype=torch.float32,
    )

    assert importance[0, 0, 0] == 1
    assert importance[1, 1, 1] == 2
    assert importance[2, 2, 2] == 3
    assert torch.equal(importance[0], torch.ones(5, 5))


def test_lazy_model_output_dtype_controls_accumulators(tmp_path):
    cfg = _make_cfg()
    cfg.data.dataloader.patch_size = [2, 2, 2]
    cfg.model.output_size = [2, 2, 2]
    cfg.inference.model.output_dtype = "float16"
    cfg.inference.sliding_window.window_size = [2, 2, 2]
    cfg.inference.sliding_window.overlap = 0.5
    cfg.inference.sliding_window.blending = "constant"

    image_path = tmp_path / "lazy_output_dtype.h5"
    volume = np.linspace(0.0, 1.0, num=3 * 3 * 3, dtype=np.float32).reshape(3, 3, 3)
    write_hdf5(str(image_path), volume, dataset="main")

    lazy = lazy_predict_volume(cfg, _identity_forward, str(image_path), device="cpu")

    assert lazy.dtype == torch.float16
    assert torch.allclose(
        lazy.float(),
        torch.from_numpy(volume).unsqueeze(0).unsqueeze(0),
        atol=5.0e-4,
    )


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


def test_lazy_region_window_builder_only_enumerates_intersecting_global_grid_windows():
    image_size = (17, 19, 23)
    roi_size = (5, 6, 7)
    overlap = (0.5, 0.25, 0.5)
    region_start = (8, 8, 10)
    region_stop = (9, 10, 11)

    for snap_to_edge in (False, True):
        full_grid = _build_window_slices(
            image_size,
            roi_size,
            overlap,
            snap_to_edge=snap_to_edge,
        )
        expected = [
            patch
            for patch in full_grid
            if all(
                int(patch[axis].start) < region_stop[axis]
                and int(patch[axis].stop) > region_start[axis]
                for axis in range(3)
            )
        ]

        region_grid = _build_intersecting_window_slices(
            image_size,
            roi_size,
            overlap,
            region_start=region_start,
            region_stop=region_stop,
            snap_to_edge=snap_to_edge,
        )

        assert region_grid == expected
        assert len(region_grid) < len(full_grid)


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

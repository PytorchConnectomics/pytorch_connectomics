"""Tests for nnU-Net style preprocessing and output restoration."""

from __future__ import annotations

import numpy as np

from connectomics.config import Config
from connectomics.data.io import NNUNetPreprocessd, read_hdf5
from connectomics.inference.io import _restore_prediction_to_input_space, write_outputs


def test_nnunet_preprocess_crops_and_tracks_metadata():
    image = np.zeros((1, 6, 8, 10), dtype=np.float32)
    image[:, 2:5, 3:7, 4:9] = 10.0
    data = {
        "image": image,
        "image_meta_dict": {"filename_or_obj": "toy_volume.h5", "transpose_axes": None},
    }

    transform = NNUNetPreprocessd(
        keys=["image"],
        enabled=True,
        crop_to_nonzero=True,
        normalization="zscore",
        normalization_use_nonzero_mask=True,
    )
    result = transform(data)

    out = result["image"]
    assert out.shape == (1, 3, 4, 5)

    meta = result["image_meta_dict"]["nnunet_preprocess"]
    assert meta["applied_crop"] is True
    assert meta["crop_bbox"] == [[2, 5], [3, 7], [4, 9]]
    assert meta["original_spatial_shape"] == [6, 8, 10]
    assert meta["cropped_spatial_shape"] == [3, 4, 5]


def test_restore_prediction_inverts_resampling():
    image = np.random.RandomState(0).rand(1, 4, 6, 8).astype(np.float32)
    data = {
        "image": image,
        "image_meta_dict": {"filename_or_obj": "toy_volume.h5", "transpose_axes": None},
    }

    transform = NNUNetPreprocessd(
        keys=["image"],
        enabled=True,
        crop_to_nonzero=False,
        source_spacing=[2.0, 1.0, 1.0],
        target_spacing=[1.0, 1.0, 1.0],
        normalization="none",
    )
    result = transform(data)
    pre_shape = result["image"].shape[1:]
    assert pre_shape == (8, 6, 8)

    pred = np.random.RandomState(1).rand(2, *pre_shape).astype(np.float32)
    restored = _restore_prediction_to_input_space(pred, result["image_meta_dict"])
    assert restored.shape == (2, 4, 6, 8)


def test_write_outputs_restores_to_input_space(tmp_path):
    image = np.zeros((1, 5, 7, 9), dtype=np.float32)
    image[:, 1:4, 2:6, 3:8] = 1.0
    data = {
        "image": image,
        "image_meta_dict": {"filename_or_obj": "toy_volume.h5", "transpose_axes": None},
    }

    transform = NNUNetPreprocessd(
        keys=["image"],
        enabled=True,
        crop_to_nonzero=True,
        normalization="none",
    )
    result = transform(data)

    pre_shape = result["image"].shape[1:]
    predictions = np.ones((1, 1, *pre_shape), dtype=np.float32)

    from connectomics.config.hydra_config import TestConfig as HydraTestConfig

    cfg = Config()
    cfg.test = HydraTestConfig()
    cfg.test.data.output_path = str(tmp_path)
    cfg.test.data.nnunet_preprocessing.enabled = True
    cfg.test.data.nnunet_preprocessing.restore_to_input_space = True

    write_outputs(
        cfg=cfg,
        predictions=predictions,
        filenames=["toy_volume"],
        suffix="prediction",
        mode="test",
        batch_meta=[result["image_meta_dict"]],
    )

    saved = read_hdf5(str(tmp_path / "toy_volume_prediction.h5"), dataset="main")
    assert saved.shape == (5, 7, 9)

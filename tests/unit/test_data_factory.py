from __future__ import annotations

from types import SimpleNamespace

import pytest
from monai.transforms import Resized

from connectomics.config import Config
from connectomics.data.augment.build import build_test_transforms, build_val_transforms
from connectomics.data.augment.transforms import ResizeByFactord
from connectomics.training.lightning.data_factory import create_datamodule


def test_create_datamodule_tune_requires_val_image(monkeypatch):
    dummy_transforms = SimpleNamespace(transforms=[])

    monkeypatch.setattr(
        "connectomics.training.lightning.data_factory.build_train_transforms",
        lambda cfg: dummy_transforms,
    )
    monkeypatch.setattr(
        "connectomics.training.lightning.data_factory.build_val_transforms",
        lambda cfg: dummy_transforms,
    )
    monkeypatch.setattr(
        "connectomics.training.lightning.data_factory.build_test_transforms",
        lambda cfg, mode="tune": dummy_transforms,
    )

    cfg = Config()
    cfg.data.test.image = "dummy_test_image.h5"

    with pytest.raises(ValueError, match="Tune mode requires data.val.image to be set"):
        create_datamodule(cfg, mode="tune")


def test_build_val_transforms_uses_data_transform_resize_with_expected_modes():
    cfg = Config()
    cfg.data.data_transform.resize = [64, 256, 256]

    transforms = build_val_transforms(cfg, keys=["image", "label", "mask"], skip_loading=True)
    resize_transforms = [t for t in transforms.transforms if isinstance(t, Resized)]

    assert len(resize_transforms) == 2
    assert resize_transforms[0].keys == ("image",)
    assert resize_transforms[0].mode == ("bilinear",)
    assert resize_transforms[1].keys == ("label", "mask")
    assert resize_transforms[1].mode == ("nearest", "nearest")


def test_build_test_transforms_derives_scale_factors_from_patch_resize():
    cfg = Config()
    cfg.data.dataloader.patch_size = [32, 128, 128]
    cfg.data.data_transform.resize = [64, 256, 256]
    cfg.data.test.image = "dummy_test_image.h5"
    cfg.data.test.label = "dummy_test_label.h5"
    cfg.data.test.mask = "dummy_test_mask.h5"

    transforms = build_test_transforms(cfg, keys=["image", "label", "mask"], mode="test")
    resize_transforms = [t for t in transforms.transforms if isinstance(t, ResizeByFactord)]

    assert len(resize_transforms) == 2
    assert resize_transforms[0].keys == ("image",)
    assert resize_transforms[0].mode == "bilinear"
    assert resize_transforms[0].scale_factors == [2.0, 2.0, 2.0]
    assert resize_transforms[1].keys == ("label", "mask")
    assert resize_transforms[1].mode == "nearest"
    assert resize_transforms[1].scale_factors == [2.0, 2.0, 2.0]


def test_create_datamodule_test_lazy_load_keeps_paths_unloaded(tmp_path):
    image_path = tmp_path / "lazy_input.h5"
    image_path.touch()

    cfg = Config()
    cfg.inference.sliding_window.lazy_load = True
    cfg.data.test.image = str(image_path)
    cfg.system.num_workers = 0
    cfg.data.dataloader.batch_size = 1

    datamodule = create_datamodule(cfg, mode="test")
    batch = next(iter(datamodule.test_dataloader()))

    assert batch["image"] == [str(image_path)]

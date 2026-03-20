from __future__ import annotations

from types import SimpleNamespace

import pytest
from monai.transforms import Compose, Resized

from connectomics.config import Config
from connectomics.data.augmentation.build import build_test_transforms, build_val_transforms
from connectomics.data.augmentation.transforms import ResizeByFactord
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


def test_build_val_transforms_auto_includes_label_aux_keys():
    cfg = Config()
    cfg.data.train.label_aux = "train_aux.h5"

    transforms = build_val_transforms(cfg, skip_loading=True)

    assert "label_aux" in transforms.transforms[-1].keys

    cfg = Config()
    cfg.data.val.label_aux = "val_aux.h5"

    transforms = build_val_transforms(cfg, skip_loading=True)

    assert "label_aux" in transforms.transforms[-1].keys


def test_create_datamodule_auto_populates_val_label_aux(monkeypatch, tmp_path):
    train_image = tmp_path / "train_image.h5"
    train_label = tmp_path / "train_label.h5"
    val_image = tmp_path / "val_image.h5"
    val_label = tmp_path / "val_label.h5"
    for path in (train_image, train_label, val_image, val_label):
        path.touch()

    cfg = Config()
    cfg.system.num_workers = 0
    cfg.data.dataloader.batch_size = 1
    cfg.data.dataloader.persistent_workers = False
    cfg.data.dataloader.use_preloaded_cache_train = False
    cfg.data.dataloader.use_preloaded_cache_val = False
    cfg.data.dataloader.use_lazy_zarr = False
    cfg.data.train.image = str(train_image)
    cfg.data.train.label = str(train_label)
    cfg.data.val.image = str(val_image)
    cfg.data.val.label = str(val_label)
    cfg.optimization.n_steps_per_epoch = 1

    captured = []

    def fake_precompute(cfg_obj, split_cfg, label_paths, *, split_name):
        return [str(tmp_path / f"{split_name}_aux.h5")]

    def fake_create_data_dicts(
        image_paths, label_paths=None, label_aux_paths=None, mask_paths=None
    ):
        captured.append((image_paths, label_paths, label_aux_paths, mask_paths))
        sample = {"image": image_paths[0]}
        if label_paths is not None:
            sample["label"] = label_paths[0]
        if label_aux_paths is not None:
            sample["label_aux"] = label_aux_paths[0]
        if mask_paths is not None:
            sample["mask"] = mask_paths[0]
        return [sample]

    monkeypatch.setattr(
        "connectomics.training.lightning.data_factory._maybe_precompute_label_aux",
        fake_precompute,
    )
    monkeypatch.setattr(
        "connectomics.training.lightning.data_factory.create_data_dicts_from_paths",
        fake_create_data_dicts,
    )
    monkeypatch.setattr(
        "connectomics.training.lightning.data_factory.build_train_transforms",
        lambda cfg: Compose([]),
    )
    monkeypatch.setattr(
        "connectomics.training.lightning.data_factory.build_val_transforms",
        lambda cfg: Compose([]),
    )

    create_datamodule(cfg, mode="train")

    assert cfg.data.train.label_aux == str(tmp_path / "train_aux.h5")
    assert cfg.data.val.label_aux == str(tmp_path / "val_aux.h5")
    assert captured[0][2] == [str(tmp_path / "train_aux.h5")]
    assert captured[1][2] == [str(tmp_path / "val_aux.h5")]

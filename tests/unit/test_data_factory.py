from __future__ import annotations

from types import SimpleNamespace

import pytest

from connectomics.config import Config
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

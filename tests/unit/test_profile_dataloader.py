"""The profiler resolves staged dataset paths before constructing its loader."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_SPEC = importlib.util.spec_from_file_location(
    "profile_script", Path(__file__).resolve().parents[2] / "scripts/profile_dataloader.py"
)
assert _SPEC is not None and _SPEC.loader is not None
profile_script = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(profile_script)


def test_profiler_resolves_train_stage_before_creating_loader(monkeypatch, tmp_path):
    config = tmp_path / "staged.yaml"
    config.write_text(
        f"train:\n  data:\n    root_path: {tmp_path}\n"
        "    train:\n      image: image.h5\n      label: labels.h5\n"
        "    dataloader:\n      batch_size: 2\n"
    )
    observed = []

    def create_datamodule(cfg):
        observed.append(cfg)
        batch = {"image": np.zeros((2, 1, 2, 2, 2)), "label": np.zeros((2, 1, 2, 2, 2))}
        return SimpleNamespace(train_dataloader=lambda: [batch])

    monkeypatch.setattr(profile_script, "create_datamodule", create_datamodule)
    profile_script.profile_dataloader(str(config), num_batches=1)
    assert len(observed) == 1
    assert observed[0].data.train.image == str(tmp_path / "image.h5")
    assert observed[0].data.train.label == str(tmp_path / "labels.h5")
    assert observed[0].data.dataloader.batch_size == 2

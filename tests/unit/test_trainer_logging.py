from pathlib import Path

import pytest

from connectomics.config import from_dict
from connectomics.training.lightning.trainer import create_trainer


@pytest.mark.parametrize("mode", ["test", "tune", "tune-test"])
def test_create_trainer_disables_logger_for_non_train_modes(tmp_path: Path, mode: str):
    cfg = from_dict(
        {
            "system": {"num_gpus": 0},
            "optimization": {"max_epochs": 1},
        }
    )

    trainer = create_trainer(cfg, run_dir=tmp_path, mode=mode)

    assert trainer.logger is None
    assert not (tmp_path / "logs").exists()


def test_create_trainer_disables_lightning_distributed_sampler_replacement_for_test(
    tmp_path: Path, monkeypatch
):
    cfg = from_dict(
        {
            "system": {"num_gpus": 0},
            "optimization": {"max_epochs": 1},
        }
    )
    captured = {}

    class _FakeTrainer:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.logger = kwargs.get("logger")

    monkeypatch.setattr("connectomics.training.lightning.trainer.pl.Trainer", _FakeTrainer)

    trainer = create_trainer(cfg, run_dir=tmp_path, mode="test")

    assert isinstance(trainer, _FakeTrainer)
    assert captured["use_distributed_sampler"] is False


def test_create_trainer_adds_periodic_step_checkpoint_callback(tmp_path: Path, monkeypatch):
    cfg = from_dict(
        {
            "system": {"num_gpus": 0},
            "optimization": {"max_epochs": 1},
            "monitor": {
                "checkpoint": {
                    "save_top_k": 3,
                    "save_every_n_steps": 50000,
                },
                "logging": {"images": {"enabled": False}},
            },
        }
    )
    checkpoint_kwargs = []
    trainer_kwargs = {}

    class _FakeModelCheckpoint:
        def __init__(self, **kwargs):
            checkpoint_kwargs.append(kwargs)

    class _FakeTrainer:
        def __init__(self, **kwargs):
            trainer_kwargs.update(kwargs)
            self.logger = kwargs.get("logger")

    monkeypatch.setattr(
        "connectomics.training.lightning.trainer.ModelCheckpoint",
        _FakeModelCheckpoint,
    )
    monkeypatch.setattr("connectomics.training.lightning.trainer.pl.Trainer", _FakeTrainer)

    trainer = create_trainer(cfg, run_dir=tmp_path, mode="train")

    assert isinstance(trainer, _FakeTrainer)
    assert len(checkpoint_kwargs) == 2
    assert checkpoint_kwargs[0]["save_top_k"] == 3
    assert checkpoint_kwargs[0]["monitor"] == cfg.monitor.checkpoint.monitor
    assert checkpoint_kwargs[1]["monitor"] is None
    assert checkpoint_kwargs[1]["save_top_k"] == -1
    assert checkpoint_kwargs[1]["every_n_train_steps"] == 50000
    assert checkpoint_kwargs[1]["every_n_epochs"] == 0
    assert sum(isinstance(cb, _FakeModelCheckpoint) for cb in trainer_kwargs["callbacks"]) == 2

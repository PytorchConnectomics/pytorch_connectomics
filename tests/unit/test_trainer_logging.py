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

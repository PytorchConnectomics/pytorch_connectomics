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

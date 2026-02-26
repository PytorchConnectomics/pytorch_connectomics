import re
from pathlib import Path

import torch

from connectomics.config import Config
from connectomics.training.lightning.config import (
    cleanup_run_directory,
    modify_checkpoint_state,
    setup_run_directory,
)
from connectomics.training.lightning.runtime import (
    modify_checkpoint_state as runtime_modify_checkpoint_state,
)


def test_setup_run_directory_train_creates_timestamped_layout(tmp_path):
    cfg = Config()
    checkpoint_dir = tmp_path / "outputs" / "exp" / "checkpoints"

    run_dir = setup_run_directory("train", cfg, str(checkpoint_dir))

    assert run_dir.parent == checkpoint_dir.parent
    assert re.fullmatch(r"\d{8}_\d{6}", run_dir.name)
    assert (run_dir / "checkpoints").exists()
    assert (run_dir / "config.yaml").exists()

    timestamp_file = checkpoint_dir.parent / ".latest_timestamp"
    assert timestamp_file.exists()
    assert timestamp_file.read_text().strip() == run_dir.name
    assert Path(cfg.monitor.checkpoint.dirpath) == run_dir / "checkpoints"


def test_setup_run_directory_train_ddp_reuses_timestamp_file(tmp_path, monkeypatch):
    cfg = Config()
    checkpoint_dir = tmp_path / "outputs" / "exp" / "checkpoints"
    output_base = checkpoint_dir.parent
    output_base.mkdir(parents=True, exist_ok=True)

    timestamp = "20250208_112233"
    (output_base / ".latest_timestamp").write_text(timestamp)
    monkeypatch.setenv("LOCAL_RANK", "1")

    run_dir = setup_run_directory("train", cfg, str(checkpoint_dir))
    assert run_dir == output_base / timestamp
    assert Path(cfg.monitor.checkpoint.dirpath) == run_dir / "checkpoints"


def test_setup_run_directory_non_train_modes_create_requested_directory(tmp_path):
    for mode in ["test", "tune", "tune-test"]:
        cfg = Config()
        target = tmp_path / mode / "artifacts"
        run_dir = setup_run_directory(mode, cfg, str(target))
        assert run_dir == target
        assert target.exists()


def test_cleanup_run_directory_removes_timestamp_file(tmp_path):
    output_base = tmp_path / "outputs" / "exp"
    output_base.mkdir(parents=True, exist_ok=True)
    timestamp_file = output_base / ".latest_timestamp"
    timestamp_file.write_text("20250208_112233")

    cleanup_run_directory(output_base)
    assert not timestamp_file.exists()


def test_modify_checkpoint_state_returns_original_when_no_resets_requested(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = tmp_path / "checkpoint.ckpt"
    torch.save({"epoch": 5, "global_step": 42}, checkpoint)

    assert modify_checkpoint_state(None, run_dir) is None
    assert modify_checkpoint_state(str(checkpoint), run_dir) == str(checkpoint)
    assert runtime_modify_checkpoint_state(None, run_dir) is None


def test_modify_checkpoint_state_applies_selected_resets(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = tmp_path / "checkpoint.ckpt"
    torch.save(
        {
            "epoch": 7,
            "global_step": 128,
            "optimizer_states": [{"state": 1}],
            "lr_schedulers": [{"state": 2}],
            "callbacks": {
                "EarlyStopping": {
                    "wait_count": 3,
                    "best_score": torch.tensor(0.123),
                }
            },
        },
        checkpoint,
    )

    modified = modify_checkpoint_state(
        str(checkpoint),
        run_dir,
        reset_optimizer=True,
        reset_scheduler=True,
        reset_epoch=True,
        reset_early_stopping=True,
    )

    assert modified is not None
    modified_path = Path(modified)
    assert modified_path.exists()
    assert modified_path.parent == run_dir
    assert modified_path.name == "temp_modified_checkpoint.ckpt"

    loaded = torch.load(modified_path, map_location="cpu", weights_only=False)
    assert "optimizer_states" not in loaded
    assert "lr_schedulers" not in loaded
    assert loaded["epoch"] == 0
    assert loaded["global_step"] == 0
    assert loaded["callbacks"]["EarlyStopping"]["wait_count"] == 0
    assert loaded["callbacks"]["EarlyStopping"]["best_score"] is None

import argparse
from pathlib import Path

from connectomics.config import Config, save_config
from connectomics.training.lit.utils import (
    expand_file_paths,
    extract_best_score_from_checkpoint,
    setup_config,
)


def _make_args(config_path: Path, overrides=None, fast_dev_run: int = 0, mode: str = "train"):
    return argparse.Namespace(
        config=str(config_path),
        demo=False,
        mode=mode,
        checkpoint=None,
        reset_optimizer=False,
        reset_scheduler=False,
        reset_epoch=False,
        reset_max_epochs=5,
        fast_dev_run=fast_dev_run,
        external_prefix=None,
        params=None,
        param_source=None,
        tune_trials=None,
        overrides=overrides or [],
    )


def test_setup_config_applies_overrides_and_fast_dev_run(tmp_path):
    cfg = Config()
    cfg.system.training.num_workers = 2  # ensure override happens

    cfg_path = tmp_path / "config.yaml"
    save_config(cfg, cfg_path)

    args = _make_args(
        cfg_path,
        overrides=["optimization.optimizer.lr=0.01", "system.training.batch_size=2"],
        fast_dev_run=1,
    )

    updated = setup_config(args)

    expected_dir = f"outputs/{cfg_path.stem}_{cfg.model.architecture}/checkpoints"
    assert updated.monitor.checkpoint.dirpath == expected_dir
    assert updated.optimization.optimizer.lr == 0.01
    assert updated.optimization.max_epochs == 5
    assert updated.system.training.batch_size == 2
    assert updated.system.training.num_workers == 0  # forced by fast-dev-run


def test_expand_file_paths_handles_globs_and_lists(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "a.txt").write_text("a")
    (data_dir / "b.txt").write_text("b")

    # Glob expansion returns sorted list
    expanded = expand_file_paths(str(data_dir / "*.txt"))
    assert expanded == [str(data_dir / "a.txt"), str(data_dir / "b.txt")]

    # Passing a list should be returned unchanged
    assert expand_file_paths([str(data_dir / "a.txt")]) == [str(data_dir / "a.txt")]


def test_extract_best_score_from_checkpoint():
    score = extract_best_score_from_checkpoint(
        "outputs/run/train_loss_total_epoch=0.1234.ckpt", "train_loss_total_epoch"
    )
    assert score == 0.1234

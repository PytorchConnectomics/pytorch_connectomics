"""Train-mode output layout: outputs/<yaml_stem>/<timestamp>/checkpoints/."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from connectomics.config import Config, save_config
from connectomics.runtime.cli import setup_config
from connectomics.training.lightning.runtime import setup_run_directory


def _make_args(config_path: Path, mode: str = "train"):
    return argparse.Namespace(
        config=str(config_path),
        demo=False,
        debug_config=False,
        mode=mode,
        checkpoint=None,
        reset_optimizer=False,
        reset_scheduler=False,
        reset_epoch=False,
        reset_early_stopping=False,
        reset_max_epochs=5,
        fast_dev_run=0,
        external_prefix=None,
        nnunet_preprocess=False,
        overrides=[],
        params=None,
        param_source=None,
        tune_trials=None,
        tune_timeout=None,
        tune_trial_timeout=None,
        shard_id=None,
        num_shards=None,
    )


def test_setup_config_default_save_path_is_yaml_stem(tmp_path):
    """Default save_path is ``outputs/<yaml_stem>`` (base, not leaf)."""
    cfg = Config()
    cfg_path = tmp_path / "lucchi.yaml"
    save_config(cfg, cfg_path)

    cfg = setup_config(_make_args(cfg_path))

    assert Path(cfg.monitor.checkpoint.save_path).as_posix() == "outputs/lucchi"
    # Train mode does not auto-set inference.save_path.
    assert cfg.inference.save_path == ""


def test_train_layout_uses_yaml_stem_and_timestamp(tmp_path, monkeypatch):
    """Walk the chain explicitly (addresses plan_v0_review F3)."""
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.chdir(tmp_path)

    cfg = Config()
    cfg_path = tmp_path / "lucchi.yaml"
    save_config(cfg, cfg_path)

    cfg = setup_config(_make_args(cfg_path))
    setup_run_directory("train", cfg, cfg.monitor.checkpoint.save_path)

    cp = Path(cfg.monitor.checkpoint.save_path)
    assert cp.name == "checkpoints"
    assert re.fullmatch(r"\d{8}_\d{6}", cp.parent.name)
    assert cp.parent.parent.name == "lucchi"
    assert cp.parent.parent.parent.name == "outputs"

    assert (cp.parent / "config.yaml").exists()
    # No `results/` subdir under <run_dir>/.
    assert not (cp.parent / "results").exists()


def test_decode_only_test_mode_defaults_inference_save_path(tmp_path):
    """Decode-only flow (--mode test, no --checkpoint) gets a YAML-stem default
    so decoded outputs / evaluation reports do not silently fall back to cwd
    or to the input prediction's parent directory.
    """
    cfg = Config()
    cfg_path = tmp_path / "decode_only.yaml"
    save_config(cfg, cfg_path)

    cfg = setup_config(_make_args(cfg_path, mode="test"))

    assert cfg.inference.save_path == "outputs/decode_only"
    # Sanity: monitor.checkpoint.save_path also follows the YAML-stem rule.
    assert cfg.monitor.checkpoint.save_path == "outputs/decode_only"

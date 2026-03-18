import argparse
from pathlib import Path

import torch

from connectomics.config import Config, save_config
from connectomics.config.schema.inference import EvaluationConfig
from connectomics.training.lightning.utils import setup_config
from scripts.main import (
    _is_test_evaluation_enabled,
    has_assigned_test_shard,
    maybe_enable_independent_test_sharding,
    maybe_limit_test_devices,
    resolve_test_stage_runtime,
)


def _make_args(config_path: Path, mode: str = "test"):
    return argparse.Namespace(
        config=str(config_path),
        demo=False,
        debug_config=False,
        mode=mode,
        checkpoint=None,
        reset_optimizer=False,
        reset_scheduler=False,
        reset_epoch=False,
        reset_max_epochs=5,
        reset_early_stopping=False,
        fast_dev_run=0,
        external_prefix=None,
        params=None,
        param_source=None,
        tune_trials=None,
        tune_timeout=None,
        tune_trial_timeout=None,
        nnunet_preprocess=False,
        overrides=[],
        shard_id=None,
        num_shards=None,
    )


def test_resolve_test_stage_runtime_reapplies_resource_sentinels(tmp_path):
    cfg = Config()
    cfg.default.system.num_workers = -1
    cfg.default.system.num_gpus = -1

    cfg_path = tmp_path / "config.yaml"
    save_config(cfg, cfg_path)

    args = _make_args(cfg_path, mode="test")
    resolved_cfg = setup_config(args)
    switched_cfg = resolve_test_stage_runtime(resolved_cfg)

    assert switched_cfg.system.num_workers >= 0
    assert switched_cfg.system.num_workers != -1
    assert switched_cfg.system.num_gpus >= 0


def test_is_test_evaluation_enabled_uses_runtime_inference_config():
    cfg = Config()
    cfg.inference.evaluation.enabled = True

    assert _is_test_evaluation_enabled(cfg) is True

    cfg.inference.evaluation.enabled = False
    assert _is_test_evaluation_enabled(cfg) is False


def test_is_test_evaluation_enabled_supports_mapping_or_dataclass_config():
    cfg = Config()
    cfg.inference.evaluation = {"enabled": False}
    assert _is_test_evaluation_enabled(cfg) is False

    cfg.inference.evaluation = {"enabled": True}
    assert _is_test_evaluation_enabled(cfg) is True

    cfg.inference.evaluation = EvaluationConfig(enabled=False)
    assert _is_test_evaluation_enabled(cfg) is False

    cfg.inference.evaluation.enabled = True
    assert _is_test_evaluation_enabled(cfg) is True


class _DummyTestDataModule:
    def __init__(self, volume_count: int):
        self.test_data_dicts = [{} for _ in range(volume_count)]


def test_maybe_limit_test_devices_disables_distributed_tta_sharding_for_multi_volume_tests():
    cfg = Config()
    cfg.system.num_gpus = 4
    cfg.inference.test_time_augmentation.enabled = True
    cfg.inference.test_time_augmentation.distributed_sharding = True

    changed = maybe_limit_test_devices(cfg, _DummyTestDataModule(volume_count=2))

    assert changed is True
    assert cfg.system.num_gpus == 2
    assert cfg.inference.test_time_augmentation.distributed_sharding is False


def test_maybe_limit_test_devices_keeps_distributed_tta_sharding_for_single_volume_tests():
    cfg = Config()
    cfg.system.num_gpus = 4
    cfg.inference.test_time_augmentation.enabled = True
    cfg.inference.test_time_augmentation.distributed_sharding = True
    cfg.inference.test_time_augmentation.flip_axes = [1, 2]
    cfg.inference.test_time_augmentation.rotation90_axes = [[1, 2]]

    changed = maybe_limit_test_devices(cfg, _DummyTestDataModule(volume_count=1))

    assert changed is False
    assert cfg.system.num_gpus == 4
    assert cfg.inference.test_time_augmentation.distributed_sharding is True


def test_maybe_limit_test_devices_uses_deduplicated_tta_pass_count_for_single_volume_tests():
    cfg = Config()
    cfg.system.num_gpus = 32
    cfg.inference.test_time_augmentation.enabled = True
    cfg.inference.test_time_augmentation.distributed_sharding = True
    cfg.inference.test_time_augmentation.flip_axes = "all"
    cfg.inference.test_time_augmentation.rotation90_axes = [[1, 2]]

    changed = maybe_limit_test_devices(cfg, _DummyTestDataModule(volume_count=1))

    assert changed is True
    assert cfg.system.num_gpus == 16
    assert cfg.inference.test_time_augmentation.distributed_sharding is True


def test_maybe_enable_independent_test_sharding_uses_rank_env_for_multi_volume_tests(
    tmp_path, monkeypatch
):
    cfg = Config()
    cfg.system.num_gpus = 4
    cfg.inference.test_time_augmentation.distributed_sharding = True
    args = _make_args(tmp_path / "config.yaml")

    monkeypatch.setenv("SLURM_PROCID", "2")
    monkeypatch.setenv("SLURM_NTASKS", "4")
    monkeypatch.setattr("scripts.main.resolve_test_image_paths", lambda _cfg: ["a", "b", "c", "d"])

    changed = maybe_enable_independent_test_sharding(args, cfg)

    assert changed is True
    assert args.shard_id == 2
    assert args.num_shards == 4
    assert cfg.system.num_gpus == (1 if torch.cuda.is_available() else 0)
    assert cfg.inference.test_time_augmentation.distributed_sharding is False


def test_maybe_enable_independent_test_sharding_uses_explicit_shard_args(tmp_path):
    cfg = Config()
    cfg.system.num_gpus = 4
    args = _make_args(tmp_path / "config.yaml")
    args.shard_id = 1
    args.num_shards = 4

    changed = maybe_enable_independent_test_sharding(args, cfg)

    assert changed is True
    assert cfg.system.num_gpus == (1 if torch.cuda.is_available() else 0)


def test_maybe_enable_independent_test_sharding_skips_single_volume_tests(tmp_path, monkeypatch):
    cfg = Config()
    cfg.system.num_gpus = 4
    args = _make_args(tmp_path / "config.yaml")

    monkeypatch.setenv("SLURM_PROCID", "0")
    monkeypatch.setenv("SLURM_NTASKS", "4")
    monkeypatch.setattr("scripts.main.resolve_test_image_paths", lambda _cfg: ["only_one"])

    changed = maybe_enable_independent_test_sharding(args, cfg)

    assert changed is False
    assert args.shard_id is None
    assert args.num_shards is None
    assert cfg.system.num_gpus == 4


def test_has_assigned_test_shard_returns_false_for_empty_slice(tmp_path, monkeypatch):
    args = _make_args(tmp_path / "config.yaml")
    cfg = Config()
    args.shard_id = 3
    args.num_shards = 4

    monkeypatch.setattr("scripts.main.resolve_test_image_paths", lambda _cfg: ["vol0"])

    assert has_assigned_test_shard(cfg, args) is False

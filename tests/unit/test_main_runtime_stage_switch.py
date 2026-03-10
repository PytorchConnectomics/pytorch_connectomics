import argparse
from pathlib import Path

from connectomics.config import Config, save_config
from connectomics.config.schema.inference import EvaluationConfig
from connectomics.training.lightning.utils import setup_config
from scripts.main import (
    _is_test_evaluation_enabled,
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
        nnunet_preprocess=False,
        overrides=[],
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

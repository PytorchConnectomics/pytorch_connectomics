import argparse
from pathlib import Path

from connectomics.config import Config, save_config
from connectomics.training.lightning.utils import setup_config
from scripts.main import resolve_test_stage_runtime


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

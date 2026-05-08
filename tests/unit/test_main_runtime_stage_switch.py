import argparse
from pathlib import Path

import numpy as np
import torch

from connectomics.config import Config, save_config
from connectomics.config.schema.evaluation import EvaluationConfig
from connectomics.config.schema.stages import TuneConfig
from connectomics.data.io import write_hdf5
from connectomics.runtime.cache_resolver import (
    has_cached_predictions_in_output_dir,
)
from connectomics.runtime.cache_resolver import (
    is_test_evaluation_enabled as _is_test_evaluation_enabled,
)
from connectomics.runtime.checkpoint_dispatch import configure_checkpoint_output_paths
from connectomics.runtime.cli import setup_config
from connectomics.runtime.dispatch import dispatch_runtime
from connectomics.runtime.sharding import (
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
    cfg.evaluation.enabled = True

    assert _is_test_evaluation_enabled(cfg) is True

    cfg.evaluation.enabled = False
    assert _is_test_evaluation_enabled(cfg) is False


def test_is_test_evaluation_enabled_supports_mapping_or_dataclass_config():
    cfg = Config()
    cfg.evaluation = {"enabled": False}
    assert _is_test_evaluation_enabled(cfg) is False

    cfg.evaluation = {"enabled": True}
    assert _is_test_evaluation_enabled(cfg) is True

    cfg.evaluation = EvaluationConfig(enabled=False)
    assert _is_test_evaluation_enabled(cfg) is False

    cfg.evaluation.enabled = True
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


def test_maybe_limit_test_devices_keeps_distributed_window_sharding_for_single_volume_tests():
    cfg = Config()
    cfg.system.num_gpus = 4
    cfg.data.dataloader.use_lazy_zarr = True
    cfg.inference.sliding_window.distributed_sharding = True

    changed = maybe_limit_test_devices(cfg, _DummyTestDataModule(volume_count=1))

    assert changed is False
    assert cfg.system.num_gpus == 4
    assert cfg.inference.sliding_window.distributed_sharding is True


def test_maybe_limit_test_devices_keeps_chunked_raw_for_single_volume_tests():
    cfg = Config()
    cfg.system.num_gpus = 4
    cfg.inference.strategy = "chunked"
    cfg.inference.chunking.enabled = True
    cfg.inference.chunking.output_mode = "raw_prediction"

    changed = maybe_limit_test_devices(cfg, _DummyTestDataModule(volume_count=1))

    assert changed is False
    assert cfg.system.num_gpus == 4


def test_maybe_limit_test_devices_disables_distributed_window_sharding_for_multi_volume_tests():
    cfg = Config()
    cfg.system.num_gpus = 4
    cfg.data.dataloader.use_lazy_zarr = True
    cfg.inference.sliding_window.distributed_sharding = True

    changed = maybe_limit_test_devices(cfg, _DummyTestDataModule(volume_count=2))

    assert changed is True
    assert cfg.system.num_gpus == 2
    assert cfg.inference.sliding_window.distributed_sharding is False


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
    cfg.inference.test_time_augmentation.enabled = True
    cfg.inference.test_time_augmentation.distributed_sharding = True
    args = _make_args(tmp_path / "config.yaml")

    monkeypatch.setenv("SLURM_PROCID", "2")
    monkeypatch.setenv("SLURM_NTASKS", "4")
    monkeypatch.setattr(
        "connectomics.runtime.sharding.resolve_test_image_paths",
        lambda _cfg: ["a", "b", "c", "d"],
    )

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
    monkeypatch.setattr(
        "connectomics.runtime.sharding.resolve_test_image_paths",
        lambda _cfg: ["only_one"],
    )

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

    monkeypatch.setattr(
        "connectomics.runtime.sharding.resolve_test_image_paths",
        lambda _cfg: ["vol0"],
    )

    assert has_assigned_test_shard(cfg, args) is False


def test_tune_cache_only_preserves_checkpoint_tag_for_tuning_suffix(tmp_path, monkeypatch):
    cfg = Config()
    cfg.tune = TuneConfig()
    cfg.inference.save_path = str(tmp_path / "results")

    args = _make_args(tmp_path / "config.yaml", mode="tune")
    args.checkpoint = "outputs/run/checkpoints/step-step=00050000.ckpt"

    captured = {}

    monkeypatch.setattr(
        "connectomics.runtime.dispatch.setup_runtime_directories",
        lambda _args, _cfg: (tmp_path / "tuning", tmp_path),
    )
    monkeypatch.setattr(
        "connectomics.runtime.dispatch.try_cache_only_test_execution",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        "connectomics.runtime.tune_runner.try_skip_tune_with_cached_results",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        "connectomics.runtime.dispatch.has_tta_prediction_file",
        lambda _cfg: False,
    )
    monkeypatch.setattr(
        "connectomics.runtime.dispatch.has_cached_predictions_in_output_dir",
        lambda *_args, **_kwargs: True,
    )

    def _unexpected_model_build(*_args, **_kwargs):
        raise AssertionError("cache-only tuning should not build a Lightning module")

    monkeypatch.setattr(
        "connectomics.runtime.dispatch._create_runtime_model", _unexpected_model_build
    )

    def _fake_run_tuning(model, trainer_or_factory, runtime_cfg, checkpoint_path=None):
        captured["model"] = model
        captured["trainer_or_factory"] = trainer_or_factory
        captured["cfg"] = runtime_cfg
        captured["checkpoint_path"] = checkpoint_path

    monkeypatch.setattr("connectomics.runtime.tune_runner.run_tuning", _fake_run_tuning)

    dispatch_runtime(args, cfg)

    assert captured["model"] is None
    assert captured["cfg"] is cfg
    assert captured["checkpoint_path"] == args.checkpoint


def test_checkpoint_tune_uses_tuning_prediction_folder(tmp_path):
    cfg = Config()
    cfg.tune = TuneConfig()
    args = _make_args(tmp_path / "config.yaml", mode="tune")
    args.checkpoint = str(
        tmp_path
        / "outputs"
        / "nisb_base_banis"
        / "20260427_095218"
        / "checkpoints"
        / "step-step=00050000.ckpt"
    )

    output_base, tuning_dir = configure_checkpoint_output_paths(args, cfg)

    expected_output_base = tmp_path / "outputs" / "nisb_base_banis" / "20260427_095218"
    expected_tuning_dir = expected_output_base / "tune_step=00050000"
    expected_predictions_dir = expected_tuning_dir / "predictions"
    assert output_base == expected_output_base
    assert tuning_dir == str(expected_tuning_dir)
    assert cfg.tune.save_path == str(expected_tuning_dir)
    assert cfg.tune.save_predictions_path == str(expected_predictions_dir)
    # Pure tune mode: cached intermediate predictions live under the tune dir.
    assert cfg.inference.save_path == str(expected_predictions_dir)


def test_test_stage_sync_preserves_checkpoint_prediction_output_path(tmp_path):
    cfg = Config()
    cfg.inference.save_results = True
    cfg.inference.save_backend = "h5"
    cfg.inference.save_dtype = "float16"
    args = _make_args(tmp_path / "config.yaml", mode="test")
    args.checkpoint = str(
        tmp_path
        / "outputs"
        / "nisb_base_banis"
        / "20260427_095218"
        / "checkpoints"
        / "step=00200000.ckpt"
    )

    configure_checkpoint_output_paths(args, cfg)
    expected_test_dir = (
        tmp_path / "outputs" / "nisb_base_banis" / "20260427_095218" / "test_step=00200000"
    )

    assert cfg.inference.save_path == str(expected_test_dir)

    cfg = resolve_test_stage_runtime(cfg)

    assert cfg.inference.save_results is True
    assert cfg.inference.save_backend == "h5"
    assert cfg.inference.save_dtype == "float16"
    assert cfg.inference.save_path == str(expected_test_dir)


def test_tune_cache_detection_uses_tuning_folder_then_result_fallback(tmp_path):
    cfg = Config()
    cfg.tune = TuneConfig()
    cfg.inference.save_path = str(tmp_path / "test_step=00050000")
    cfg.tune.save_predictions_path = str(tmp_path / "tune_step=00050000" / "predictions")
    cfg.data.val.image = str(tmp_path / "images" / "volume_a.h5")

    image_path = Path(cfg.data.val.image)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.touch()

    assert (
        has_cached_predictions_in_output_dir(
            cfg,
            mode="tune",
            checkpoint_path="step-step=00050000.ckpt",
        )
        is False
    )

    # Per-volume layout: <save_path>/<volume_stem>/raw_x1.h5
    results_pred = Path(cfg.inference.save_path) / "volume_a" / "raw_x1.h5"
    results_pred.parent.mkdir(parents=True, exist_ok=True)
    write_hdf5(str(results_pred), np.zeros((1, 1, 1), dtype=np.float32), dataset="main")

    assert (
        has_cached_predictions_in_output_dir(
            cfg,
            mode="tune",
            checkpoint_path="step-step=00050000.ckpt",
        )
        is True
    )

    tuning_pred = Path(cfg.tune.save_predictions_path) / "volume_a" / "raw_x1.h5"
    tuning_pred.parent.mkdir(parents=True, exist_ok=True)
    write_hdf5(str(tuning_pred), np.zeros((1, 1, 1), dtype=np.float32), dataset="main")

    assert (
        has_cached_predictions_in_output_dir(
            cfg,
            mode="tune",
            checkpoint_path="step-step=00050000.ckpt",
        )
        is True
    )

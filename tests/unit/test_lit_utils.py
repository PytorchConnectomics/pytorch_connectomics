import argparse
from pathlib import Path

import numpy as np
import pytest

from connectomics.config import Config, save_config
from connectomics.config.schema.stages import TuneConfig
from connectomics.training.lightning.data_factory import _calculate_validation_steps_per_epoch
from connectomics.training.lightning.path_utils import (
    expand_file_paths as canonical_expand_file_paths,
)
from connectomics.training.lightning.utils import (
    expand_file_paths,
    extract_best_score_from_checkpoint,
    format_decode_tag,
    resolve_prediction_cache_suffix,
    setup_config,
    tta_cache_suffix,
    tta_cache_suffix_candidates,
    tuning_best_params_filename,
    tuning_best_params_filename_candidates,
)


def _make_args(
    config_path: Path,
    overrides=None,
    fast_dev_run: int = 0,
    mode: str = "train",
    nnunet_preprocess: bool = False,
    tune_timeout: int | None = None,
    tune_trial_timeout: int | None = None,
    tune_trials: int | None = None,
):
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
        fast_dev_run=fast_dev_run,
        external_prefix=None,
        params=None,
        param_source=None,
        tune_trials=tune_trials,
        tune_timeout=tune_timeout,
        tune_trial_timeout=tune_trial_timeout,
        nnunet_preprocess=nnunet_preprocess,
        overrides=overrides or [],
    )


def test_setup_config_applies_overrides_and_fast_dev_run(tmp_path):
    cfg = Config()
    cfg.system.num_workers = 2  # ensure override happens

    cfg_path = tmp_path / "config.yaml"
    save_config(cfg, cfg_path)

    args = _make_args(
        cfg_path,
        overrides=["optimization.optimizer.lr=0.01", "data.dataloader.batch_size=2"],
        fast_dev_run=1,
    )

    updated = setup_config(args)

    expected_dir = f"outputs/{cfg_path.stem}/checkpoints"
    assert Path(updated.monitor.checkpoint.dirpath).as_posix() == expected_dir
    assert updated.optimization.optimizer.lr == 0.01
    assert updated.optimization.max_epochs == 5
    assert updated.data.dataloader.batch_size == 2
    assert updated.system.num_workers == 0  # forced by fast-dev-run


def test_setup_config_enables_nnunet_preprocess_from_cli_switch(tmp_path):
    cfg = Config()
    assert cfg.data.nnunet_preprocessing.enabled is False

    cfg_path = tmp_path / "config.yaml"
    save_config(cfg, cfg_path)

    args = _make_args(cfg_path, nnunet_preprocess=True)
    updated = setup_config(args)

    assert updated.data.nnunet_preprocessing.enabled is True


def test_setup_config_applies_tune_timeout_cli_overrides(tmp_path):
    cfg = Config()
    cfg.tune = TuneConfig()

    cfg_path = tmp_path / "config.yaml"
    save_config(cfg, cfg_path)

    args = _make_args(
        cfg_path,
        mode="tune",
        tune_trials=17,
        tune_timeout=3600,
        tune_trial_timeout=300,
    )
    updated = setup_config(args)

    assert updated.tune.n_trials == 17
    assert updated.tune.timeout == 3600
    assert updated.tune.trial_timeout == 300


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

    # Utility helper should match canonical helper behavior
    assert canonical_expand_file_paths(str(data_dir / "*.txt")) == expanded

    with pytest.raises(FileNotFoundError):
        canonical_expand_file_paths(str(data_dir / "*.missing"))


def test_calculate_validation_steps_per_epoch_unknown_suffix_defaults():
    val_data_dicts = [{"image": "dummy.unknown"}]

    assert (
        _calculate_validation_steps_per_epoch(
            val_data_dicts=val_data_dicts,
            patch_size=(16, 16, 16),
        )
        == 100
    )


def test_calculate_validation_steps_per_epoch_uses_fallback_shape_without_clamp():
    val_data_dicts = [{"image": "dummy.unknown"}]

    val_steps = _calculate_validation_steps_per_epoch(
        val_data_dicts=val_data_dicts,
        patch_size=(32, 32, 32),
        min_steps=1,
        max_steps=None,
        fallback_volume_shape=(100, 4096, 4096),
        return_default_on_error=False,
    )

    assert val_steps == 24384


def test_calculate_validation_steps_per_epoch_handles_multipage_tiff(tmp_path):
    tifffile = pytest.importorskip("tifffile")

    tiff_path = tmp_path / "stack_pages.tif"
    volume = np.random.randint(0, 255, size=(8, 20, 20), dtype=np.uint8)

    # Write as separate pages/series to reproduce the real-world failure mode.
    with tifffile.TiffWriter(tiff_path) as writer:
        for z in range(volume.shape[0]):
            writer.write(volume[z])

    val_steps = _calculate_validation_steps_per_epoch(
        val_data_dicts=[{"image": str(tiff_path)}],
        patch_size=(2, 4, 4),
        min_steps=1,
        max_steps=None,
        return_default_on_error=False,
    )

    # Expected from shape (8, 20, 20):
    # stride=(1,2,2), patches=(7,9,9), total=567, 7.5%=42
    assert val_steps == 42


def test_extract_best_score_from_checkpoint():
    score = extract_best_score_from_checkpoint(
        "outputs/run/train_loss_total_epoch=0.1234.ckpt", "train_loss_total_epoch"
    )
    assert score == 0.1234


def test_resolve_prediction_cache_suffix_uses_current_tta_plan_for_test_mode():
    cfg = Config()
    cfg.inference.save_prediction.cache_suffix = "_x1_prediction.h5"
    cfg.inference.test_time_augmentation.enabled = True
    cfg.inference.test_time_augmentation.flip_axes = [1, 2]
    cfg.inference.test_time_augmentation.rotation90_axes = [[1, 2]]

    assert resolve_prediction_cache_suffix(cfg, mode="test") == "_tta_x8_prediction.h5"


def test_resolve_prediction_cache_suffix_preserves_non_tta_test_suffix_when_tta_disabled():
    cfg = Config()
    cfg.inference.save_prediction.cache_suffix = "_x1_prediction.h5"
    cfg.inference.test_time_augmentation.enabled = False

    assert resolve_prediction_cache_suffix(cfg, mode="test") == "_x1_prediction.h5"


def test_resolve_prediction_cache_suffix_includes_checkpoint_name_for_tta_test_mode():
    cfg = Config()
    cfg.inference.save_prediction.cache_suffix = "_x1_prediction.h5"
    cfg.inference.test_time_augmentation.enabled = True
    cfg.inference.test_time_augmentation.flip_axes = [1, 2]
    cfg.inference.test_time_augmentation.rotation90_axes = [[1, 2]]

    assert (
        resolve_prediction_cache_suffix(
            cfg,
            mode="test",
            checkpoint_path="/tmp/checkpoints/epoch=4-step=99.ckpt",
        )
        == "_tta_x8_ckpt-epoch=4-step=99_prediction.h5"
    )


def test_resolve_prediction_cache_suffix_includes_output_head_for_multi_head_tta():
    cfg = Config()
    cfg.model.out_channels = 3
    cfg.model.primary_head = "affinity"
    cfg.model.heads = {
        "affinity": {"out_channels": 2, "num_blocks": 0},
        "sdt": {"out_channels": 1, "num_blocks": 0},
    }
    cfg.inference.head = "sdt"
    cfg.inference.test_time_augmentation.enabled = True
    cfg.inference.test_time_augmentation.flip_axes = None
    cfg.inference.test_time_augmentation.rotation90_axes = None

    assert resolve_prediction_cache_suffix(cfg, mode="test") == "_tta_x1_head-sdt_prediction.h5"


def test_tta_cache_suffix_accepts_explicit_output_head_override():
    cfg = Config()
    cfg.model.out_channels = 3
    cfg.model.primary_head = "affinity"
    cfg.model.heads = {
        "affinity": {"out_channels": 2, "num_blocks": 0},
        "sdt": {"out_channels": 1, "num_blocks": 0},
    }
    cfg.inference.head = "affinity"
    cfg.inference.test_time_augmentation.enabled = True
    cfg.inference.test_time_augmentation.flip_axes = None
    cfg.inference.test_time_augmentation.rotation90_axes = None

    assert tta_cache_suffix(cfg, output_head="sdt") == "_tta_x1_head-sdt_prediction.h5"
    assert (
        resolve_prediction_cache_suffix(cfg, mode="test", output_head="sdt")
        == "_tta_x1_head-sdt_prediction.h5"
    )


def test_tta_cache_suffix_candidates_do_not_fall_back_to_legacy_suffix_with_checkpoint():
    cfg = Config()
    cfg.inference.test_time_augmentation.enabled = True
    cfg.inference.test_time_augmentation.flip_axes = [1, 2]
    cfg.inference.test_time_augmentation.rotation90_axes = [[1, 2]]

    assert tta_cache_suffix_candidates(
        cfg,
        checkpoint_path="/tmp/checkpoints/epoch=4-step=99.ckpt",
    ) == ["_tta_x8_ckpt-epoch=4-step=99_prediction.h5"]


def test_tuning_best_params_filename_matches_tta_prediction_identity():
    cfg = Config()
    cfg.inference.test_time_augmentation.select_channel = [4, 6, 9]

    assert (
        tuning_best_params_filename(
            cfg,
            checkpoint_path="/tmp/checkpoints/epoch=4-step=99.ckpt",
        )
        == "best_params_tta_x1_ch4-6-9_ckpt-epoch=4-step=99_prediction.yaml"
    )
    assert tuning_best_params_filename_candidates(
        cfg,
        checkpoint_path="/tmp/checkpoints/epoch=4-step=99.ckpt",
    ) == [
        "best_params_tta_x1_ch4-6-9_ckpt-epoch=4-step=99_prediction.yaml",
        "best_params.yaml",
    ]


def test_tuning_best_params_filename_includes_output_head_identity():
    cfg = Config()
    cfg.model.out_channels = 3
    cfg.model.primary_head = "affinity"
    cfg.model.heads = {
        "affinity": {"out_channels": 2, "num_blocks": 0},
        "sdt": {"out_channels": 1, "num_blocks": 0},
    }
    cfg.inference.head = "sdt"

    assert (
        tuning_best_params_filename(
            cfg,
            checkpoint_path="/tmp/checkpoints/epoch=4-step=99.ckpt",
        )
        == "best_params_tta_x1_head-sdt_ckpt-epoch=4-step=99_prediction.yaml"
    )


def test_format_decode_tag_includes_all_decoding_parameters():
    cfg = Config()
    cfg.inference.decoding = [
        {
            "name": "decode_waterz",
            "kwargs": {
                "discretize_queue": 256,
                "fragments": "watershed",
                "merge_function": "aff50_his256",
                "return_seg": True,
                "thresholds": [0.1, 0.2, 0.4],
            },
        }
    ]

    assert format_decode_tag(cfg) == "_waterz_256-watershed-aff50_his256-true-0.1-0.2-0.4"


def test_format_decode_tag_gates_dust_and_branch_parameter_groups():
    cfg = Config()
    cfg.inference.decoding = [
        {
            "name": "decode_waterz",
            "kwargs": {
                "branch_merge": True,
                "branch_iou_threshold": 0.5,
                "branch_best_buddy": True,
                "branch_one_sided_threshold": 0.8,
                "branch_one_sided_min_size": 100,
                "branch_affinity_threshold": 0.0,
                "dust_merge": True,
                "dust_merge_size": 800,
                "dust_merge_affinity": 0.3,
                "dust_remove_size": 600,
                "merge_function": "aff85_his256",
                "thresholds": 0.5,
            },
        }
    ]

    assert format_decode_tag(cfg) == "_waterz_0.5-true-0.8-100-0-800-0.3-600-aff85_his256-0.5"


def test_format_decode_tag_skips_disabled_dust_and_branch_parameter_groups():
    cfg = Config()
    cfg.inference.decoding = [
        {
            "name": "decode_waterz",
            "kwargs": {
                "branch_merge": False,
                "branch_iou_threshold": 0.5,
                "branch_best_buddy": True,
                "branch_one_sided_threshold": 0.8,
                "branch_one_sided_min_size": 100,
                "branch_affinity_threshold": 0.0,
                "dust_merge": False,
                "dust_merge_size": 800,
                "dust_merge_affinity": 0.3,
                "dust_remove_size": 600,
                "merge_function": "aff85_his256",
                "thresholds": 0.5,
            },
        }
    ]

    assert format_decode_tag(cfg) == "_waterz_aff85_his256-0.5"

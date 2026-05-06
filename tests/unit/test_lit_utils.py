import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from connectomics.config import Config, save_config
from connectomics.config.schema.stages import TuneConfig
from connectomics.runtime.cache_resolver import resolve_cached_prediction_files
from connectomics.runtime.cli import setup_config
from connectomics.runtime.output_naming import (
    final_prediction_decoded_glob_suffix,
    final_prediction_output_tag,
    format_checkpoint_name_tag,
    format_decode_tag,
    intermediate_prediction_cache_suffix,
    intermediate_prediction_cache_suffix_candidates,
    is_tta_cache_suffix,
    resolve_prediction_cache_suffix,
    tta_cache_suffix,
    tta_cache_suffix_candidates,
    tuning_best_params_filename,
    tuning_best_params_filename_candidates,
)
from connectomics.training.lightning import utils as lightning_utils
from connectomics.training.lightning.data_factory import _calculate_validation_steps_per_epoch
from connectomics.training.lightning.path_utils import expand_file_paths
from connectomics.training.lightning.utils import extract_best_score_from_checkpoint

REPO_ROOT = Path(__file__).resolve().parents[2]


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


def test_output_naming_import_does_not_import_lightning():
    code = (
        "import sys\n"
        "import connectomics.runtime.output_naming\n"
        "print(any(name.startswith('connectomics.training.lightning') for name in sys.modules))\n"
    )

    output = subprocess.check_output(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        text=True,
    ).strip()

    assert output == "False"


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

    expanded = expand_file_paths(str(data_dir / "*.txt"))
    assert expanded == [str(data_dir / "a.txt"), str(data_dir / "b.txt")]

    assert expand_file_paths([str(data_dir / "a.txt")]) == [str(data_dir / "a.txt")]

    assert not hasattr(lightning_utils, "expand_file_paths")

    with pytest.raises(FileNotFoundError):
        expand_file_paths(str(data_dir / "*.missing"))


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


def test_resolve_prediction_cache_suffix_includes_channel_for_non_tta_checkpoint():
    cfg = Config()
    cfg.inference.select_channel = [0, 1, 2]
    cfg.inference.test_time_augmentation.enabled = False

    assert (
        resolve_prediction_cache_suffix(
            cfg,
            mode="test",
            checkpoint_path="/tmp/checkpoints/epoch=4-step=99.ckpt",
        )
        == "_x1_ch0-1-2_ckpt-epoch=4-step=99_prediction.h5"
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


def test_format_checkpoint_name_tag_canonicalizes_lightning_inserted_metric_names():
    assert (
        format_checkpoint_name_tag("/tmp/checkpoints/step-step=00050000.ckpt")
        == "_ckpt-step=00050000"
    )
    assert (
        format_checkpoint_name_tag("/tmp/checkpoints/epoch-epoch=004-step-step=00050000.ckpt")
        == "_ckpt-epoch=004-step=00050000"
    )
    assert (
        format_checkpoint_name_tag("/tmp/checkpoints/epoch=4-step=99.ckpt")
        == "_ckpt-epoch=4-step=99"
    )


def test_chunked_raw_intermediate_suffix_does_not_collide_with_whole_volume_cache():
    cfg = Config()
    cfg.inference.select_channel = [0, 1, 2]
    cfg.inference.strategy = "chunked"
    cfg.inference.chunking.enabled = True
    cfg.inference.chunking.output_mode = "raw_prediction"
    cfg.inference.chunking.chunk_size = [1000, 1000, 1350]
    cfg.inference.chunking.halo = [0, 0, 0]
    cfg.decoding.output_suffix = "chunk_raw_v1"
    checkpoint = "/tmp/checkpoints/step-step=00050000.ckpt"

    suffix = intermediate_prediction_cache_suffix(cfg, checkpoint_path=checkpoint)

    assert suffix == (
        "_tta_x1_ch0-1-2_ckpt-step=00050000"
        "_chunked-raw_cs1000x1000x1350_chunk_raw_v1_prediction.h5"
    )
    assert suffix != tta_cache_suffix(cfg, checkpoint_path=checkpoint)
    assert is_tta_cache_suffix(suffix)
    assert intermediate_prediction_cache_suffix_candidates(cfg, checkpoint_path=checkpoint) == [
        suffix
    ]


def test_cache_resolver_ignores_whole_volume_raw_for_chunked_raw_config(tmp_path):
    h5py = pytest.importorskip("h5py")
    cfg = Config()
    cfg.inference.select_channel = [0, 1, 2]
    cfg.inference.strategy = "chunked"
    cfg.inference.chunking.enabled = True
    cfg.inference.chunking.output_mode = "raw_prediction"
    cfg.inference.chunking.chunk_size = [1000, 1000, 1350]
    cfg.decoding.output_suffix = "chunk_raw_v1"
    checkpoint = "/tmp/checkpoints/step-step=00050000.ckpt"
    filename = "img"

    old_whole_volume_raw = (
        tmp_path / f"{filename}{tta_cache_suffix(cfg, checkpoint_path=checkpoint)}"
    )
    with h5py.File(old_whole_volume_raw, "w") as handle:
        handle.create_dataset("main", data=np.zeros((3, 2, 2, 2), dtype=np.float32))

    cache_hit, loaded_suffix, resolved_files = resolve_cached_prediction_files(
        tmp_path,
        [filename],
        resolve_prediction_cache_suffix(cfg, mode="test", checkpoint_path=checkpoint),
        fallback_tta_suffixes=intermediate_prediction_cache_suffix_candidates(
            cfg, checkpoint_path=checkpoint
        ),
        preferred_decoded_suffix=(
            "_" + final_prediction_output_tag(cfg, checkpoint_path=checkpoint) + ".h5"
        ),
        decoded_glob_suffix=final_prediction_decoded_glob_suffix(cfg, checkpoint_path=checkpoint),
    )

    assert cache_hit is False
    assert loaded_suffix is None
    assert resolved_files == []


def test_cache_resolver_prefers_decoded_final_over_large_raw_intermediate(tmp_path):
    h5py = pytest.importorskip("h5py")
    cfg = Config()
    cfg.inference.select_channel = [0, 1, 2]
    cfg.decoding.steps = [
        {
            "name": "decode_affinity_cc",
            "kwargs": {"threshold": 0.7, "backend": "numba", "edge_offset": 0},
        }
    ]
    checkpoint = "/tmp/checkpoints/step-step=00050000.ckpt"
    filename = "img"
    raw_suffix = tta_cache_suffix(cfg, checkpoint_path=checkpoint)
    final_suffix = "_" + final_prediction_output_tag(cfg, checkpoint_path=checkpoint) + ".h5"
    variant_suffix = final_suffix.removesuffix(".h5") + "_crop1.h5"

    with h5py.File(tmp_path / f"{filename}{raw_suffix}", "w") as handle:
        handle.create_dataset("main", data=np.zeros((3, 2, 2, 2), dtype=np.float32))
    with h5py.File(tmp_path / f"{filename}{final_suffix}", "w") as handle:
        handle.create_dataset("main", data=np.zeros((2, 2, 2), dtype=np.uint32))
    with h5py.File(tmp_path / f"{filename}{variant_suffix}", "w") as handle:
        handle.create_dataset("main", data=np.ones((2, 2, 2), dtype=np.uint32))

    cache_hit, loaded_suffix, resolved_files = resolve_cached_prediction_files(
        tmp_path,
        [filename],
        resolve_prediction_cache_suffix(cfg, mode="test", checkpoint_path=checkpoint),
        fallback_tta_suffixes=intermediate_prediction_cache_suffix_candidates(
            cfg, checkpoint_path=checkpoint
        ),
        preferred_decoded_suffix=final_suffix,
        decoded_glob_suffix=final_prediction_decoded_glob_suffix(cfg, checkpoint_path=checkpoint),
    )

    assert cache_hit is True
    assert loaded_suffix == final_suffix
    assert resolved_files == [tmp_path / f"{filename}{final_suffix}"]


def test_tuning_best_params_filename_matches_tta_prediction_identity():
    cfg = Config()
    cfg.inference.select_channel = [4, 6, 9]

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
    cfg.decoding.steps = [
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


def test_decoding_output_suffix_disambiguates_final_prediction_cache_glob():
    cfg = Config()
    cfg.inference.select_channel = [0, 1, 2]
    cfg.decoding.output_suffix = "chunk raw/v1"
    cfg.decoding.steps = [
        {
            "name": "decode_affinity_cc",
            "kwargs": {
                "threshold": 0.7,
                "backend": "numba",
                "edge_offset": 0,
            },
        }
    ]

    assert (
        final_prediction_output_tag(
            cfg,
            checkpoint_path="/tmp/checkpoints/step-step=00050000.ckpt",
        )
        == "x1_ch0-1-2_ckpt-step=00050000_decoding_affinity_cc_numba-0-0.7_chunk-raw-v1"
    )
    assert (
        final_prediction_decoded_glob_suffix(
            cfg,
            checkpoint_path="/tmp/checkpoints/step-step=00050000.ckpt",
        )
        == "_x1_ch0-1-2_ckpt-step=00050000_decoding_affinity_cc_numba-0-0.7*_chunk-raw-v1.h5"
    )


def test_format_decode_tag_gates_dust_and_branch_parameter_groups():
    cfg = Config()
    cfg.decoding.steps = [
        {
            "name": "decode_waterz",
            "kwargs": {
                "branch_merge": True,
                "iou_threshold": 0.5,
                "best_buddy": True,
                "one_sided_threshold": 0.8,
                "one_sided_min_size": 100,
                "affinity_threshold": 0.0,
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
    cfg.decoding.steps = [
        {
            "name": "decode_waterz",
            "kwargs": {
                "branch_merge": False,
                "iou_threshold": 0.5,
                "best_buddy": True,
                "one_sided_threshold": 0.8,
                "one_sided_min_size": 100,
                "affinity_threshold": 0.0,
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

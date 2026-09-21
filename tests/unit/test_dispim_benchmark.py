"""Contracts for the paired, staged diSPIM cloud benchmark."""

import argparse
import importlib.util
import json
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
import torch

from connectomics.config import load_config, resolve_default_profiles
from connectomics.runtime.dispim_benchmark import checkpoint_record, run_stage
from connectomics.runtime import dispim_benchmark
from connectomics.runtime.checkpoint_dispatch import get_checkpoint_test_output_dir

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "tutorials/dispim_gcloud/dispim_snemi_20epoch.yaml"
spec = importlib.util.spec_from_file_location(
    "dispim_prepare", ROOT / "tutorials/dispim_gcloud/prepare_data.py"
)
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def test_cloud_config_keeps_pairs_and_isolates_test_labels(monkeypatch):
    train = resolve_default_profiles(load_config(CONFIG), mode="train")
    test = resolve_default_profiles(load_config(CONFIG), mode="test")
    assert train.model.arch.type == "mednext_dual_view"
    assert train.model.dual_view.design == "partial_gated"
    assert train.model.in_channels == 2 and train.model.out_channels == 9
    assert train.system.num_gpus == 4 and test.system.num_gpus == 1
    from connectomics.config.hardware.gpu_utils import resolve_accelerator_type

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_accelerator_type(train.system.accelerator) == "cuda"
    assert train.data.image_transform.channelwise
    assert train.optimization.max_epochs == 20
    assert train.optimization.scheduler.params["max_iter"] == 20
    assert train.optimization.n_steps_per_epoch == 200
    assert train.monitor.checkpoint.save_top_k == -1
    assert train.data.train.label == "train-labels.h5"
    assert train.data.val.label == "val-labels.h5"
    assert test.data.test.label == "test-labels.h5"
    assert test.tune is None
    assert not test.inference.test_time_augmentation.enabled
    assert [s.name for s in test.decoding.steps] == ["decode_abiss"]


def test_voxel_center_mapping_preserves_large_integer_ids():
    labels = np.arange(4 * 6 * 8, dtype=np.uint64).reshape(4, 6, 8) + 2**55
    mapped = prepare.label_grid(labels, (2, 3, 4))
    np.testing.assert_array_equal(mapped, labels[1::2, 1::2, 1::2])
    upsampled = prepare.label_grid(labels, (8, 12, 16))
    np.testing.assert_array_equal(upsampled, labels.repeat(2, 0).repeat(2, 1).repeat(2, 2))


@pytest.mark.parametrize("spacing", [20, 10, 5])
def test_streamed_preparation_preserves_pairs_labels_and_hashes(tmp_path, monkeypatch, spacing):
    import h5py

    class Group(dict):
        pass

    labels = np.arange(4 * 6 * 8, dtype=np.uint64).reshape(4, 6, 8) + 2**50
    a = np.arange(labels.size, dtype=np.float32).reshape(labels.shape)
    b = a + 100
    suffix = "" if spacing == 20 else f"-{spacing}nm"
    groups = {"labels": {"segments": labels, "test_segments": labels}}
    for split, key in (("train", "segments"), ("test", "test_segments")):
        group = Group(image_blur_A=a, image_blur_B=b)
        group.attrs = {
            "simulation_mode": "whole_volume", "axes": "ZYX",
            "expansion_deformation_applied": False, "sectioning": {"active": False},
            "tiling": {"active": False}, "source": f"source.zarr::/{key}",
            "experiment_params": json.dumps({"expansion_factor": 400 / spacing}),
            "voxel_size_um_zyx": [0.4] * 3,
            "source_voxel_size_um_zyx": [spacing / 1000] * 3,
        }
        groups[f"snemi-{split}{suffix}.zarr"] = group
    monkeypatch.setitem(sys.modules, "zarr", SimpleNamespace(
        open_group=lambda path, mode: groups[Path(path).name]
    ))
    output = tmp_path / "prepared"
    record = prepare.prepare(tmp_path, tmp_path / "labels", output, split=0.5,
                             spacing_nm=spacing, slab_depth=3)
    for part, interval in (("train", slice(0, 2)), ("val", slice(2, 4)), ("test", slice(None))):
        with h5py.File(output / f"{part}-views.h5") as f:
            np.testing.assert_array_equal(f["main"][:], np.stack([a, b])[:, interval])
        with h5py.File(output / f"{part}-labels.h5") as f:
            np.testing.assert_array_equal(f["main"][:], labels[interval])
    assert record["sources"]["test"]["mapped_label_sha256"] == prepare.array_hash(labels)
    assert record["sources"]["test"]["view_sha256"] == [prepare.array_hash(a), prepare.array_hash(b)]
    assert record["sources"]["test"]["gcs"].endswith(f"snemi-test{suffix}.zarr")
    assert len(record["files"]) == 6


@pytest.mark.parametrize("spacing", [10, 5])
def test_finer_recipes_keep_training_budget_and_change_spacing(spacing):
    path = CONFIG.with_name(f"dispim_snemi_{spacing}nm_20epoch.yaml")
    train = resolve_default_profiles(load_config(path), mode="train")
    test = resolve_default_profiles(load_config(path), mode="test")
    assert train.optimization.max_epochs == 20
    assert train.optimization.n_steps_per_epoch == 200
    assert train.system.num_gpus == 4
    assert train.model.input_size == [64, 64, 64]
    assert train.data.label_transform.resolution == [spacing] * 3
    assert train.data.train.resolution == [spacing] * 3
    assert test.data.test.resolution == [spacing] * 3
    assert train.monitor.checkpoint.save_top_k == -1


@pytest.mark.parametrize("spacing,shape,disk,cpu", [
    (10, [300, 614, 614], 200, "e2-standard-16"),
    (5, [600, 1229, 1229], 500, "n2-highmem-32"),
])
@pytest.mark.parametrize("stage", ["gpu", "cpu"])
def test_cloud_resource_sizing_for_finer_grids(monkeypatch, spacing, shape, disk, cpu, stage):
    spec = importlib.util.spec_from_file_location(
        "dispim_launch", ROOT / "tutorials/dispim_gcloud/launch_stage.py"
    )
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    commands = []
    prepared = {
        "status": "complete", "files": {str(i): "hash" for i in range(6)},
        "sources": {"test": {"shape": shape}},
    }
    monkeypatch.setattr(launcher.subprocess, "check_output", lambda *a, **k: json.dumps(prepared))
    monkeypatch.setattr(launcher.subprocess, "run", lambda command, **k: commands.append(command))
    launcher.launch(stage, f"gs://bucket/{spacing}nm", "project", "zone", "vm", "service-account")
    create = commands[-1]
    assert f"--boot-disk-size={disk}GB" in create
    assert f"--machine-type={'g2-standard-48' if stage == 'gpu' else cpu}" in create
    assert "--instance-termination-action=DELETE" in create


@pytest.mark.parametrize("epoch,accepted", [(2, False), (19, True)])
def test_final_checkpoint_epoch_guard(tmp_path, epoch, accepted):
    path = tmp_path / "last.ckpt"
    torch.save({"epoch": epoch, "global_step": (epoch + 1) * 200}, path)
    if accepted:
        assert checkpoint_record(path, 20)["global_step"] == 4000
    else:
        with pytest.raises(ValueError, match="Expected 20 completed epochs"):
            checkpoint_record(path, 20)


def test_no_cuda_cannot_silently_train_on_cpu(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    args = argparse.Namespace(stage="train", config=CONFIG, output=tmp_path / "train")
    with pytest.raises(RuntimeError, match="CPU fallback forbidden"):
        run_stage(args)
    assert not args.output.exists()


def test_train_to_inference_artifact_handoff(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    train_out, infer_out = tmp_path / "train", tmp_path / "infer"

    def execute(command, **kwargs):
        from connectomics.runtime.cli import parse_args

        monkeypatch.setattr(sys, "argv", command[1:])
        assert parse_args().overrides[-1] == f"data.root_path={ROOT / 'datasets/diSPIM_SNEMI'}"
        assert f"data.root_path={ROOT / 'datasets/diSPIM_SNEMI'}" in command
        if command[command.index("--mode") + 1] == "train":
            path = train_out / "training/20260910_000000/checkpoints/last.ckpt"
            path.parent.mkdir(parents=True)
            torch.save({"epoch": 19, "global_step": 4000}, path)
        else:
            path = Path(command[command.index("--checkpoint") + 1])
            assert path != train_out / "final.ckpt"
            assert "decoding.enabled=false" in command
            assert "evaluation.enabled=false" in command
            directory = get_checkpoint_test_output_dir(path) / "test-views"
            directory.mkdir(parents=True)
            (directory / "raw_x1_ch0-1-2.h5").write_bytes(b"raw affinity artifact")

    monkeypatch.setattr(dispim_benchmark.subprocess, "run", execute)
    manifest = run_stage(argparse.Namespace(
        stage="train", config=CONFIG, output=train_out, checkpoint=None
    ))
    assert manifest["global_step"] == 4000
    run_stage(argparse.Namespace(
        stage="infer", config=CONFIG, output=infer_out, checkpoint=train_out / "final.ckpt"
    ))
    assert (infer_out / "affinities.h5").read_bytes() == b"raw affinity artifact"
    result = json.loads((infer_out / "manifest.json").read_text())
    assert result["status"] == "complete"
    assert result["checkpoint_sha256"] == manifest["checkpoint_sha256"]


def test_resume_preserves_optimizer_progress_and_requires_final_epoch(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    source = tmp_path / "old-last.ckpt"
    torch.save({"epoch": 16, "global_step": 3400}, source)

    def execute(command, **kwargs):
        from connectomics.runtime.cli import parse_args

        monkeypatch.setattr(sys, "argv", command[1:])
        assert parse_args().reset_max_epochs == 20
        copied = Path(command[command.index("--checkpoint") + 1])
        assert copied.name == "resume.ckpt"
        assert copied.read_bytes() == source.read_bytes()
        assert command[command.index("--reset-max-epochs") + 1] == "20"
        torch.save({"epoch": 19, "global_step": 4000}, copied.with_name("last.ckpt"))

    monkeypatch.setattr(dispim_benchmark.subprocess, "run", execute)
    manifest = run_stage(argparse.Namespace(
        stage="train", config=CONFIG, output=tmp_path / "resumed", checkpoint=source
    ))
    assert manifest["resumed_from_completed_epochs"] == 17
    assert manifest["resumed_from_global_step"] == 3400
    assert manifest["global_step"] == 4000


def test_inference_dispatch_preserves_cli_stage_flags_and_resolved_paths(monkeypatch, tmp_path):
    from connectomics.config.pipeline import resolve_data_paths
    from connectomics.runtime import dispatch

    cfg = resolve_data_paths(resolve_default_profiles(load_config(CONFIG), mode="test"))
    cfg.decoding.enabled = False
    cfg.evaluation.enabled = False
    expected_image = cfg.data.test.image
    monkeypatch.setattr(dispatch, "resolve_prediction_cache_suffix", lambda *a, **k: "")
    monkeypatch.setattr(dispatch, "maybe_enable_naive_chunk_sharding", lambda *a: False)
    monkeypatch.setattr(dispatch, "maybe_enable_independent_test_sharding", lambda *a: False)

    def check_runtime(actual, args):
        assert actual is cfg
        assert not actual.decoding.enabled and not actual.evaluation.enabled
        assert actual.data.test.image == expected_image
        return False

    monkeypatch.setattr(dispatch, "has_assigned_test_shard", check_runtime)
    dispatch._run_test(
        argparse.Namespace(mode="test", checkpoint=None), cfg, None, object(), tmp_path, None,
        has_saved_prediction=False, saved_prediction_path="",
    )

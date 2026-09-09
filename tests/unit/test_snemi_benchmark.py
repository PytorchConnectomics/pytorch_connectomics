"""SNEMI tutorial contracts; GPU and C++ execution are isolated from the real scorer."""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from connectomics.config import load_config, resolve_default_profiles
from connectomics.data.io import write_hdf5
from connectomics.runtime import snemi_benchmark as benchmark
from connectomics.runtime.checkpoint_dispatch import get_checkpoint_test_output_dir

TUTORIAL = benchmark.REPO_ROOT / "tutorials/neuron_snemi_gcloud/neuron_snemi_gcloud.yaml"


@pytest.fixture
def case(tmp_path, monkeypatch):
    labels = np.ones((12, 24, 24), dtype=np.uint32)
    labels[:, :, 12:] = 2
    label_path = tmp_path / "labels.h5"
    raw_path = tmp_path / "raw_x16_ch0-1-2.h5"
    write_hdf5(str(label_path), labels)
    write_hdf5(str(raw_path), np.full((3, *labels.shape), 0.8, dtype=np.float32))
    config_path = tmp_path / "config.yaml"
    config = yaml.safe_load(TUTORIAL.read_text())
    config["_base_"] = str(benchmark.REPO_ROOT / "tutorials/neuron_snemi/neuron_snemi_v1.yaml")
    config["test"] = {"data": {"test": {"path": "", "label": str(label_path)}}}
    config["default"]["decoding"]["steps"][0]["kwargs"]["cli_args"]["ws_binary"] = sys.executable
    config_path.write_text(yaml.safe_dump(config))
    # Exercise real HDF5 -> stage invocation -> HDF5 -> challenge evaluator.
    # The external ABISS binary has separate wrapper tests and Docker smoke validation.
    monkeypatch.setattr(
        benchmark, "run_decoding_stage", lambda cfg, raw: SimpleNamespace(postprocessed=labels)
    )
    args = argparse.Namespace(
        config=config_path,
        output=tmp_path / "run",
        prediction=raw_path,
        checkpoint=None,
        resume_to=None,
    )
    return args, labels


def test_tutorial_replaces_waterz_and_keeps_checkpoint_training_contract():
    train = resolve_default_profiles(load_config(TUTORIAL), mode="train")
    test = resolve_default_profiles(load_config(TUTORIAL), mode="test")
    assert train.optimization.max_epochs == 3
    assert train.optimization.scheduler.params["max_iter"] == 100
    assert train.optimization.n_steps_per_epoch == 200
    assert train.data.dataloader.batch_size == 12
    assert train.data.split_train_range == [0.0, 0.8]
    assert train.data.split_val_range == [0.8, 1.0]
    assert test.model.out_channels == 12
    assert test.inference.model.select_channel == [0, 1, 2]
    assert test.inference.test_time_augmentation.enabled is False
    assert [step.name for step in test.decoding.steps] == ["decode_abiss"]
    assert test.decoding.steps[0].kwargs["cli_args"]["edge_storage"] == "destination"
    assert test.tune is None


def test_saved_affinities_reach_real_challenge_metric_without_cuda(case):
    args, _ = case
    assert benchmark.run_benchmark(args) == 0.0
    assert float((args.output / "metric.txt").read_text()) == 0.0
    score = json.loads((args.output / "metrics.json").read_text())
    assert score["evaluation_support"] == "snemi3d_gc_challenge_crop"
    assert score["crop_shape"] == "6x19x19"
    manifest = json.loads((args.output / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["raw_prediction"] == str(args.prediction)
    assert len(manifest["commands"]) == 1  # Only the real evaluator, no model execution.
    with pytest.raises(FileExistsError):
        benchmark.run_benchmark(args)


@pytest.mark.parametrize("bad", ["shape", "nan", "logits"])
def test_bad_affinities_fail_without_publishing_a_metric(case, bad):
    args, labels = case
    raw = np.full((3, *labels.shape), 0.8, dtype=np.float32)
    if bad == "shape":
        raw = raw[:, :-1]
    elif bad == "nan":
        raw[0, 0, 0, 0] = np.nan
    else:
        raw[0, 0, 0, 0] = 2.0
    write_hdf5(str(args.prediction), raw)
    with pytest.raises(ValueError):
        benchmark.run_benchmark(args)
    assert not (args.output / "metric.txt").exists()
    assert json.loads((args.output / "manifest.json").read_text())["status"] == "failed"


def test_failed_decoder_does_not_publish_a_metric(case, monkeypatch):
    args, _ = case

    def fail(cfg, raw):
        raise RuntimeError("ABISS process failed")

    monkeypatch.setattr(benchmark, "run_decoding_stage", fail)
    with pytest.raises(RuntimeError, match="ABISS process failed"):
        benchmark.run_benchmark(args)
    assert not (args.output / "metrics.json").exists()
    assert "ABISS process failed" in (args.output / "manifest.json").read_text()


def test_resume_copies_checkpoint_and_preserves_state_flags(case, monkeypatch):
    import torch

    args, labels = case
    args.prediction = None
    args.checkpoint = args.output.parent / "original.ckpt"
    args.checkpoint.write_bytes(b"original optimizer scheduler epoch state")
    args.resume_to = 20
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    real_run = benchmark._run
    commands = []

    def fake_gpu(command, log, manifest):
        if log.stem == "evaluate":
            return real_run(command, log, manifest)
        commands.append(command)
        checkpoint = Path(command[command.index("--checkpoint") + 1])
        assert checkpoint != args.checkpoint
        if log.stem == "train":
            assert checkpoint.read_bytes() == args.checkpoint.read_bytes()
            checkpoint.write_bytes(b"resumed state")
        else:
            assert checkpoint.read_bytes() == b"resumed state"
            prediction_dir = get_checkpoint_test_output_dir(checkpoint) / "test-input"
            prediction_dir.mkdir(parents=True)
            write_hdf5(
                str(prediction_dir / "raw_x1_ch0-1-2.h5"),
                np.full((3, *labels.shape), 0.8, dtype=np.float32),
            )

    monkeypatch.setattr(benchmark, "_run", fake_gpu)
    assert benchmark.run_benchmark(args) == 0.0
    assert args.checkpoint.read_bytes() == b"original optimizer scheduler epoch state"
    train, infer = commands
    assert train[train.index("--reset-max-epochs") + 1] == "20"
    assert [arg for arg in train if arg.startswith("--reset-")] == ["--reset-max-epochs"]
    assert "decoding.enabled=false" in infer
    assert "evaluation.enabled=false" in infer


@pytest.mark.parametrize("use_checkpoint", [False, True])
def test_fresh_training_and_checkpoint_only_paths(case, monkeypatch, use_checkpoint):
    import torch

    args, labels = case
    args.prediction = None
    if use_checkpoint:
        args.checkpoint = args.output.parent / "source.ckpt"
        args.checkpoint.write_bytes(b"source weights")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    real_run = benchmark._run
    modes = []

    def fake_gpu(command, log, manifest):
        if log.stem == "evaluate":
            return real_run(command, log, manifest)
        modes.append(log.stem)
        if log.stem == "train":
            assert "--checkpoint" not in command
            checkpoint = args.output / "training/20260908_120000/checkpoints/last.ckpt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"fresh weights")
        else:
            checkpoint = Path(command[command.index("--checkpoint") + 1])
            assert checkpoint.is_file()
            prediction_dir = get_checkpoint_test_output_dir(checkpoint) / "test-input"
            prediction_dir.mkdir(parents=True)
            write_hdf5(
                str(prediction_dir / "raw_x1_ch0-1-2.h5"),
                np.full((3, *labels.shape), 0.8, dtype=np.float32),
            )

    monkeypatch.setattr(benchmark, "_run", fake_gpu)
    assert benchmark.run_benchmark(args) == 0.0
    assert modes == (["infer"] if use_checkpoint else ["train", "infer"])


@pytest.mark.parametrize("value", ["nan", "inf", "-0.1", "1.1"])
def test_nonfinite_or_out_of_range_score_is_rejected(tmp_path, value):
    path = tmp_path / "metrics.tsv"
    path.write_text(f"status\tadapted_rand_error\nok\t{value}\n")
    with pytest.raises(ValueError, match="Invalid adapted Rand error"):
        benchmark._read_score(path)

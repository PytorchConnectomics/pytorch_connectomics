from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from connectomics.config import Config
from connectomics.config.schema.stages import TuneConfig
from connectomics.decoding.tuning.optuna_tuner import (
    OptunaDecodingTuner,
    TrialEvaluationTimeoutError,
    _get_trial_process_context,
    load_and_apply_best_params,
    run_tuning,
)


class _DummyModel:
    def __init__(self, cfg: Config):
        self.cfg = cfg


class _DummyTrainer:
    def __init__(self, on_test=None):
        self.observed = {}
        self.on_test = on_test

    def test(self, model, datamodule=None, ckpt_path=None):
        inference_cfg = model.cfg.inference
        self.observed = {
            "datamodule": datamodule,
            "ckpt_path": ckpt_path,
            "save_prediction_enabled": inference_cfg.save_prediction.enabled,
            "cache_suffix": inference_cfg.save_prediction.cache_suffix,
            "decoding": inference_cfg.decoding,
            "evaluation_enabled": inference_cfg.evaluation.enabled,
        }
        if self.on_test is not None:
            self.on_test()
        return [{"status": "ok"}]


class _FakeStudy:
    best_value = 0.1234
    best_params = {"binary_threshold": 0.5}


class _DummyTrial:
    def __init__(self):
        self.user_attrs = {}

    def set_user_attr(self, key, value):
        self.user_attrs[key] = value


def test_run_tuning_uses_intermediate_only_inference_overrides(monkeypatch, tmp_path):
    cfg = Config()
    cfg.tune = TuneConfig()
    cfg.inference.save_prediction.output_path = str(tmp_path / "results")
    cfg.inference.save_prediction.enabled = False
    cfg.inference.save_prediction.cache_suffix = "_x1_prediction.h5"
    cfg.inference.decoding = [{"name": "decode_semantic", "kwargs": {"threshold": 0.8}}]
    cfg.inference.evaluation.enabled = True
    cfg.data.val.image = str(tmp_path / "images" / "volume_0_input.h5")
    cfg.data.val.label = str(tmp_path / "labels" / "volume_0_label.h5")

    model = _DummyModel(cfg)

    image_file = tmp_path / "images" / "volume_0_input.h5"
    image_file.parent.mkdir(parents=True, exist_ok=True)
    image_file.touch()

    prediction_file = str(
        tmp_path / "results" / "volume_0_input_tta_x1_ckpt-checkpoint_prediction.h5"
    )
    label_file = str(tmp_path / "labels" / "volume_0_label.h5")
    Path(label_file).parent.mkdir(parents=True, exist_ok=True)
    Path(label_file).touch()
    loaded_arrays = {
        prediction_file: np.zeros((3, 4, 4, 4), dtype=np.float32),
        label_file: np.zeros((4, 4, 4), dtype=np.uint16),
    }
    captured = {}
    trainer = _DummyTrainer(
        on_test=lambda: (
            Path(prediction_file).parent.mkdir(parents=True, exist_ok=True),
            Path(prediction_file).touch(),
        )
    )

    class _FakeTuner:
        def __init__(self, cfg, predictions, ground_truth, mask=None):
            captured["cfg"] = cfg
            captured["predictions"] = predictions
            captured["ground_truth"] = ground_truth
            captured["mask"] = mask

        def optimize(self):
            return _FakeStudy()

    monkeypatch.setattr("connectomics.decoding.tuning.optuna_tuner.OPTUNA_AVAILABLE", True)
    monkeypatch.setattr(
        "connectomics.training.lightning.create_datamodule",
        lambda cfg, mode="tune": {"cfg": cfg, "mode": mode},
    )
    monkeypatch.setattr("connectomics.data.io.read_volume", lambda path: loaded_arrays[path])
    monkeypatch.setattr("connectomics.decoding.tuning.optuna_tuner.OptunaDecodingTuner", _FakeTuner)

    run_tuning(model, trainer, cfg, checkpoint_path="checkpoint.ckpt")

    assert trainer.observed["datamodule"]["mode"] == "tune"
    assert trainer.observed["ckpt_path"] == "checkpoint.ckpt"
    assert trainer.observed["save_prediction_enabled"] is True
    assert trainer.observed["cache_suffix"] == "_tta_x1_ckpt-checkpoint_prediction.h5"
    assert trainer.observed["decoding"] is None
    assert trainer.observed["evaluation_enabled"] is False

    assert cfg.inference.save_prediction.enabled is False
    assert cfg.inference.save_prediction.cache_suffix == "_x1_prediction.h5"
    assert cfg.inference.decoding == [{"name": "decode_semantic", "kwargs": {"threshold": 0.8}}]
    assert cfg.inference.evaluation.enabled is True

    assert len(captured["predictions"]) == 1
    assert len(captured["ground_truth"]) == 1
    assert captured["mask"] is None


def test_run_tuning_ignores_stale_test_prediction_cache_when_tuning(monkeypatch, tmp_path):
    cfg = Config()
    cfg.tune = TuneConfig()
    cfg.inference.save_prediction.output_path = str(tmp_path / "results")
    cfg.data.val.image = str(tmp_path / "images" / "train-input.tif")
    cfg.data.val.label = str(tmp_path / "labels" / "train-labels.h5")

    model = _DummyModel(cfg)

    image_file = tmp_path / "images" / "train-input.tif"
    image_file.parent.mkdir(parents=True, exist_ok=True)
    image_file.touch()

    stale_prediction_file = tmp_path / "results" / "test-input_z29_tta_x1_prediction.h5"
    stale_prediction_file.parent.mkdir(parents=True, exist_ok=True)
    stale_prediction_file.touch()

    expected_prediction_file = (
        tmp_path / "results" / "train-input_tta_x1_ckpt-checkpoint_prediction.h5"
    )
    label_file = tmp_path / "labels" / "train-labels.h5"
    label_file.parent.mkdir(parents=True, exist_ok=True)
    label_file.touch()

    stale_prediction = np.ones((3, 4, 4, 4), dtype=np.float32)
    expected_prediction = np.full((3, 4, 4, 4), 7.0, dtype=np.float32)
    loaded_arrays = {
        str(stale_prediction_file): stale_prediction,
        str(expected_prediction_file): expected_prediction,
        str(label_file): np.zeros((4, 4, 4), dtype=np.uint16),
    }
    captured = {}
    trainer = _DummyTrainer(on_test=lambda: expected_prediction_file.touch())

    class _FakeTuner:
        def __init__(self, cfg, predictions, ground_truth, mask=None):
            captured["predictions"] = predictions
            captured["ground_truth"] = ground_truth

        def optimize(self):
            return _FakeStudy()

    monkeypatch.setattr("connectomics.decoding.tuning.optuna_tuner.OPTUNA_AVAILABLE", True)
    monkeypatch.setattr(
        "connectomics.training.lightning.create_datamodule",
        lambda cfg, mode="tune": {"cfg": cfg, "mode": mode},
    )
    monkeypatch.setattr("connectomics.data.io.read_volume", lambda path: loaded_arrays[path])
    monkeypatch.setattr("connectomics.decoding.tuning.optuna_tuner.OptunaDecodingTuner", _FakeTuner)

    run_tuning(model, trainer, cfg, checkpoint_path="checkpoint.ckpt")

    assert trainer.observed["datamodule"]["mode"] == "tune"
    assert len(captured["predictions"]) == 1
    assert np.array_equal(captured["predictions"][0], expected_prediction)
    assert len(captured["ground_truth"]) == 1


def test_run_tuning_requires_val_labels_in_tune_mode(monkeypatch, tmp_path):
    cfg = Config()
    cfg.tune = TuneConfig()
    cfg.inference.save_prediction.output_path = str(tmp_path / "results")
    cfg.data.val.image = str(tmp_path / "images" / "val_input.h5")
    cfg.data.test.label = str(tmp_path / "labels" / "test_*.h5")

    image_file = tmp_path / "images" / "val_input.h5"
    image_file.parent.mkdir(parents=True, exist_ok=True)
    image_file.touch()
    expected_prediction_file = (
        tmp_path / "results" / "val_input_tta_x1_ckpt-checkpoint_prediction.h5"
    )

    model = _DummyModel(cfg)
    trainer = _DummyTrainer(
        on_test=lambda: (
            expected_prediction_file.parent.mkdir(parents=True, exist_ok=True),
            expected_prediction_file.touch(),
        )
    )

    monkeypatch.setattr("connectomics.decoding.tuning.optuna_tuner.OPTUNA_AVAILABLE", True)
    monkeypatch.setattr(
        "connectomics.training.lightning.create_datamodule",
        lambda cfg, mode="tune": {"cfg": cfg, "mode": mode},
    )
    monkeypatch.setattr(
        "connectomics.data.io.read_volume",
        lambda path: np.zeros((3, 4, 4, 4), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="Missing data.val.label in configuration"):
        run_tuning(model, trainer, cfg, checkpoint_path="checkpoint.ckpt")


def test_run_tuning_prints_existing_best_params_yaml(monkeypatch, tmp_path, capsys):
    cfg = Config()
    cfg.tune = TuneConfig()
    cfg.inference.save_prediction.output_path = str(tmp_path / "results")

    tuning_dir = tmp_path / "tuning"
    tuning_dir.mkdir(parents=True, exist_ok=True)
    best_params_file = tuning_dir / "best_params_tta_x1_ckpt-checkpoint_prediction.yaml"
    best_params_file.write_text(
        "best_trial: 7\nbest_value: 0.1234\ndecoding_function: decode_waterz\n"
    )

    model = _DummyModel(cfg)
    trainer = _DummyTrainer()

    monkeypatch.setattr("connectomics.decoding.tuning.optuna_tuner.OPTUNA_AVAILABLE", True)

    run_tuning(model, trainer, cfg, checkpoint_path="checkpoint.ckpt")

    stdout = capsys.readouterr().out
    assert "BEST PARAMETERS" in stdout
    assert str(best_params_file) in stdout
    assert "best_trial: 7" in stdout
    assert "decode_waterz" in stdout
    assert trainer.observed == {}


def test_load_and_apply_best_params_prefers_checkpoint_aware_file(tmp_path):
    cfg = Config()
    cfg.inference.save_prediction.output_path = str(tmp_path / "results")
    cfg.inference.decoding = [{"name": "decode_waterz", "kwargs": {"thresholds": 0.4}}]

    tuning_dir = tmp_path / "tuning"
    tuning_dir.mkdir(parents=True, exist_ok=True)
    best_params_file = tuning_dir / "best_params_tta_x1_ckpt-checkpoint_prediction.yaml"
    best_params_file.write_text(
        "\n".join(
            [
                "best_trial: 1",
                "best_value: 0.12",
                "decoding_function: decode_waterz",
                "decoding_params:",
                "  thresholds: 0.5",
                "  dust_merge: false",
            ]
        )
    )

    updated = load_and_apply_best_params(cfg, checkpoint_path="checkpoint.ckpt")

    assert updated.inference.decoding[0]["kwargs"]["thresholds"] == 0.5
    assert updated.inference.decoding[0]["kwargs"]["dust_merge"] is False


def test_load_and_apply_best_params_falls_back_to_legacy_filename(tmp_path):
    cfg = Config()
    cfg.inference.save_prediction.output_path = str(tmp_path / "results")
    cfg.inference.decoding = [{"name": "decode_waterz", "kwargs": {"thresholds": 0.4}}]

    tuning_dir = tmp_path / "tuning"
    tuning_dir.mkdir(parents=True, exist_ok=True)
    legacy_file = tuning_dir / "best_params.yaml"
    legacy_file.write_text(
        "\n".join(
            [
                "best_trial: 2",
                "best_value: 0.08",
                "decoding_function: decode_waterz",
                "decoding_params:",
                "  thresholds: 0.6",
            ]
        )
    )

    updated = load_and_apply_best_params(cfg, checkpoint_path="checkpoint.ckpt")

    assert updated.inference.decoding[0]["kwargs"]["thresholds"] == 0.6


def test_objective_returns_bad_value_when_standard_trial_times_out(monkeypatch):
    cfg = Config()
    cfg.tune = TuneConfig()
    cfg.tune.trial_timeout = 12
    cfg.tune.parameter_space.decoding.function_name = "decode_instance_binary_contour_distance"

    tuner = OptunaDecodingTuner(
        cfg=cfg,
        predictions=np.zeros((3, 4, 4, 4), dtype=np.float32),
        ground_truth=np.zeros((4, 4, 4), dtype=np.uint16),
    )

    def _raise_timeout(_evaluation_kind, _payload):
        raise TrialEvaluationTimeoutError("standard evaluation exceeded timeout")

    monkeypatch.setattr(tuner, "_execute_evaluation", _raise_timeout)

    trial = _DummyTrial()
    result = tuner._objective(trial)

    assert result == float("inf")
    assert trial.user_attrs["timed_out"] is True
    assert trial.user_attrs["timeout_stage"] == "standard"
    assert trial.user_attrs["trial_timeout"] == 12.0


def test_objective_returns_bad_value_when_waterz_batch_trial_times_out(monkeypatch):
    cfg = Config()
    cfg.tune = TuneConfig()
    cfg.tune.trial_timeout = 30
    cfg.tune.parameter_space.decoding.function_name = "decode_waterz"
    cfg.tune.parameter_space.decoding.defaults = {
        "thresholds": 0.4,
        "merge_function": "aff85_his256",
        "aff_threshold": [0.001, 0.999],
    }
    cfg.tune.parameter_space.decoding.parameters = {
        "thresholds": {
            "type": "float",
            "range": [0.1, 0.2],
            "step": 0.1,
        }
    }

    tuner = OptunaDecodingTuner(
        cfg=cfg,
        predictions=np.zeros((3, 4, 4, 4), dtype=np.float32),
        ground_truth=np.zeros((4, 4, 4), dtype=np.uint16),
    )
    assert tuner._waterz_batch_enabled is True

    def _raise_timeout(_evaluation_kind, _payload):
        raise TrialEvaluationTimeoutError("waterz batch exceeded timeout")

    monkeypatch.setattr(tuner, "_execute_evaluation", _raise_timeout)

    trial = _DummyTrial()
    result = tuner._objective(trial)

    assert result == float("inf")
    assert trial.user_attrs["timed_out"] is True
    assert trial.user_attrs["timeout_stage"] == "waterz_batch"
    assert trial.user_attrs["trial_timeout"] == 30.0


def test_get_trial_process_context_prefers_spawn_after_cuda_init(monkeypatch):
    observed = []

    class _DummyContext:
        pass

    def _fake_get_context(method=None):
        observed.append(method)
        if method == "spawn":
            return _DummyContext()
        raise ValueError(f"unsupported: {method}")

    monkeypatch.setattr(
        "connectomics.decoding.tuning.optuna_tuner.torch.cuda.is_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "connectomics.decoding.tuning.optuna_tuner.torch.cuda.is_initialized",
        lambda: True,
    )
    monkeypatch.setattr(
        "connectomics.decoding.tuning.optuna_tuner.mp.get_context", _fake_get_context
    )

    ctx = _get_trial_process_context()

    assert isinstance(ctx, _DummyContext)
    assert observed == ["spawn"]


def test_get_trial_process_context_prefers_fork_without_cuda_init(monkeypatch):
    observed = []

    class _DummyContext:
        pass

    def _fake_get_context(method=None):
        observed.append(method)
        if method == "fork":
            return _DummyContext()
        raise ValueError(f"unsupported: {method}")

    monkeypatch.setattr(
        "connectomics.decoding.tuning.optuna_tuner.torch.cuda.is_available",
        lambda: False,
    )
    monkeypatch.setattr(
        "connectomics.decoding.tuning.optuna_tuner.torch.cuda.is_initialized",
        lambda: False,
    )
    monkeypatch.setattr(
        "connectomics.decoding.tuning.optuna_tuner.mp.get_context", _fake_get_context
    )

    ctx = _get_trial_process_context()

    assert isinstance(ctx, _DummyContext)
    assert observed == ["fork"]

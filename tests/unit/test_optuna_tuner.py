from __future__ import annotations

import glob

import numpy as np

from connectomics.config import Config
from connectomics.decoding.tuning.optuna_tuner import run_tuning


class _DummyModel:
    def __init__(self, cfg: Config):
        self.cfg = cfg


class _DummyTrainer:
    def __init__(self):
        self.observed = {}

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
        return [{"status": "ok"}]


class _FakeStudy:
    best_value = 0.1234
    best_params = {"binary_threshold": 0.5}


def test_run_tuning_uses_intermediate_only_inference_overrides(monkeypatch, tmp_path):
    cfg = Config()
    cfg.inference.save_prediction.output_path = str(tmp_path / "results")
    cfg.inference.save_prediction.enabled = False
    cfg.inference.save_prediction.cache_suffix = "_prediction.h5"
    cfg.inference.decoding = [{"name": "decode_semantic", "kwargs": {"threshold": 0.8}}]
    cfg.inference.evaluation.enabled = True
    cfg.data.val.label = str(tmp_path / "labels" / "*.h5")

    model = _DummyModel(cfg)
    trainer = _DummyTrainer()

    prediction_file = str(tmp_path / "results" / "volume_0_tta_prediction.h5")
    label_file = str(tmp_path / "labels" / "volume_0_label.h5")
    loaded_arrays = {
        prediction_file: np.zeros((3, 4, 4, 4), dtype=np.float32),
        label_file: np.zeros((4, 4, 4), dtype=np.uint16),
    }
    glob_patterns = []
    captured = {}

    def _fake_glob(pattern):
        glob_patterns.append(pattern)
        if pattern.endswith("_tta_prediction.h5"):
            return [prediction_file]
        if pattern == cfg.data.val.label:
            return [label_file]
        return []

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
    monkeypatch.setattr(glob, "glob", _fake_glob)
    monkeypatch.setattr("connectomics.decoding.tuning.optuna_tuner.OptunaDecodingTuner", _FakeTuner)

    run_tuning(model, trainer, cfg, checkpoint_path="checkpoint.ckpt")

    assert trainer.observed["datamodule"]["mode"] == "tune"
    assert trainer.observed["ckpt_path"] == "checkpoint.ckpt"
    assert trainer.observed["save_prediction_enabled"] is True
    assert trainer.observed["cache_suffix"] == "_tta_prediction.h5"
    assert trainer.observed["decoding"] is None
    assert trainer.observed["evaluation_enabled"] is False

    assert cfg.inference.save_prediction.enabled is False
    assert cfg.inference.save_prediction.cache_suffix == "_prediction.h5"
    assert cfg.inference.decoding == [{"name": "decode_semantic", "kwargs": {"threshold": 0.8}}]
    assert cfg.inference.evaluation.enabled is True

    assert any(pattern.endswith("*_tta_prediction.h5") for pattern in glob_patterns)
    assert len(captured["predictions"]) == 1
    assert len(captured["ground_truth"]) == 1
    assert captured["mask"] is None

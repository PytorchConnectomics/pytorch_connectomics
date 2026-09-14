"""Inference must load the requested checkpoint tensors and keep their caches separate."""

from types import SimpleNamespace

import pytest
import pytorch_lightning as pl
import torch
from pytorch_lightning.trainer.states import TrainerFn
from torch import nn
from torch.utils.data import DataLoader

from connectomics.config import Config, load_config, resolve_default_profiles, validate_config
from connectomics.runtime.dispatch import _create_runtime_model, dispatch_runtime
from connectomics.runtime.output_naming import (
    final_prediction_output_tag,
    intermediate_decode_step_output_tag,
    intermediate_prediction_cache_suffix,
    raw_cache_suffix_candidates,
    tuning_best_params_filename,
    tuning_best_params_filename_candidates,
    tuning_study_db_filename,
)
from connectomics.training.lightning.callbacks import EMAWeightsCallback
from connectomics.training.lightning.model import ConnectomicsModule


class _CheckpointProbe(ConnectomicsModule):
    def __init__(self, cfg):
        super().__init__(cfg, model=nn.Linear(1, 1, bias=False), skip_loss=True)
        self.register_buffer("outside_model", torch.tensor(-9.0))
        self.observed = []

    def on_test_start(self):
        pass

    def test_step(self, batch, batch_idx):
        self.observed.append(self(batch).detach().cpu())

    def on_test_epoch_end(self):
        pass

    def on_validation_start(self):
        pass

    def on_validation_epoch_start(self):
        pass

    def validation_step(self, batch, batch_idx):
        self.observed.append(self(batch).detach().cpu())

    def on_validation_epoch_end(self):
        pass


def _checkpoint():
    return {
        "pytorch-lightning_version": pl.__version__,
        "epoch": 0,
        "global_step": 4,
        "state_dict": {
            "model.weight": torch.tensor([[2.0]]),
            "outside_model": torch.tensor(7.0),
        },
        "callbacks": {"EMAWeightsCallback": {"ema_state": {"weight": torch.tensor([[5.0]])}}},
    }


@pytest.mark.parametrize("source, expected", [("raw", 2.0), ("ema", 5.0)])
@pytest.mark.parametrize("stage", ["test", "validate"])
@pytest.mark.parametrize("stale_ema", [False, True])
def test_lightning_evaluation_loads_selected_checkpoint_weights(
    tmp_path, source, expected, stage, stale_ema
):
    cfg = Config()
    cfg.inference.checkpoint_weights = source
    module = _CheckpointProbe(cfg)
    callbacks = []
    if stale_ema:
        callback = EMAWeightsCallback()
        with torch.no_grad():
            module.model.weight.fill_(99.0)
        callback.on_train_start(SimpleNamespace(global_step=0), module)
        callbacks.append(callback)
    path = tmp_path / "step=4.ckpt"
    torch.save(_checkpoint(), path)
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        callbacks=callbacks,
    )

    getattr(trainer, stage)(module, DataLoader(torch.ones(1, 1)), ckpt_path=str(path))

    assert module.observed[0].item() == expected
    assert module.outside_model.item() == 7.0
    assert module._loaded_checkpoint_weights == source


def test_direct_load_from_checkpoint_honors_explicit_ema(tmp_path):
    cfg = Config()
    cfg.inference.checkpoint_weights = "ema"
    path = tmp_path / "checkpoint.ckpt"
    torch.save(_checkpoint(), path)

    module = _CheckpointProbe.load_from_checkpoint(path, cfg=cfg)

    assert module.model.weight.item() == 5.0
    assert module.outside_model.item() == 7.0


def test_training_resume_keeps_raw_weights_and_ema_callback_state():
    cfg = Config()
    cfg.inference.checkpoint_weights = "ema"
    module = _CheckpointProbe(cfg)
    module.trainer = SimpleNamespace(state=SimpleNamespace(fn=TrainerFn.FITTING))
    checkpoint = _checkpoint()

    module.on_load_checkpoint(checkpoint)
    module.load_state_dict(checkpoint["state_dict"])

    assert module.model.weight.item() == 2.0
    assert module._loaded_checkpoint_weights == "raw"
    assert checkpoint["callbacks"]["EMAWeightsCallback"]["ema_state"]["weight"].item() == 5.0


def test_requested_ema_cannot_fall_back_to_raw():
    cfg = Config()
    cfg.inference.checkpoint_weights = "ema"
    checkpoint = _checkpoint()
    checkpoint.pop("callbacks")

    with pytest.raises(ValueError, match="checkpoint has no saved EMA state"):
        _CheckpointProbe(cfg).on_load_checkpoint(checkpoint)


@pytest.mark.parametrize(
    "ema_state",
    [{"unexpected": torch.ones(1, 1)}, {"weight": torch.ones(2, 2)}],
)
def test_ema_model_mismatch_is_rejected_before_restore(ema_state):
    cfg = Config()
    cfg.inference.checkpoint_weights = "ema"
    checkpoint = _checkpoint()
    checkpoint["callbacks"]["EMAWeightsCallback"]["ema_state"] = ema_state

    with pytest.raises(RuntimeError, match="does not match the model"):
        _CheckpointProbe(cfg).on_load_checkpoint(checkpoint)
    assert checkpoint["state_dict"]["model.weight"].item() == 2.0


@pytest.mark.parametrize("mode", ["test", "tune"])
def test_checkpoint_weight_selection_resolves_from_default_stage(tmp_path, mode):
    path = tmp_path / "config.yaml"
    path.write_text("default:\n  inference:\n    checkpoint_weights: ema\n")

    cfg = resolve_default_profiles(load_config(path), mode=mode)

    assert cfg.inference.checkpoint_weights == "ema"


def test_unknown_checkpoint_weight_selection_is_rejected():
    cfg = Config()
    cfg.inference.checkpoint_weights = "automatic"
    with pytest.raises(ValueError, match="inference.checkpoint_weights"):
        validate_config(cfg)


def test_ema_cache_decode_and_tuning_names_cannot_reuse_raw_results():
    raw = Config()
    ema = Config()
    ema.inference.checkpoint_weights = "ema"
    step = SimpleNamespace(name="decode_affinity_cc", kwargs={"threshold": 0.66})
    for filename in (
        intermediate_prediction_cache_suffix,
        final_prediction_output_tag,
        tuning_best_params_filename,
        lambda cfg: intermediate_decode_step_output_tag(cfg, step),
        lambda cfg: tuning_study_db_filename(cfg, "study"),
    ):
        assert filename(raw) != filename(ema)
        assert "_weights-ema" in filename(ema)
    assert raw_cache_suffix_candidates(raw) == ["raw_x1.h5"]
    assert raw_cache_suffix_candidates(ema) == ["raw_x1_weights-ema.h5"]
    assert "best_params.yaml" not in tuning_best_params_filename_candidates(ema)


@pytest.mark.parametrize("external_prefix, external_path", [("model.", None), (None, "raw.pt")])
def test_ema_external_weights_fail_before_runtime_side_effects(external_prefix, external_path):
    cfg = Config()
    cfg.inference.checkpoint_weights = "ema"
    cfg.model.external_weights_path = external_path
    args = SimpleNamespace(mode="test", external_prefix=external_prefix)

    with pytest.raises(ValueError, match="external-prefix/external_weights_path"):
        dispatch_runtime(args, cfg)


def test_ema_without_checkpoint_fails_before_model_construction(tmp_path):
    cfg = Config()
    cfg.inference.checkpoint_weights = "ema"
    args = SimpleNamespace(mode="test", checkpoint=None)

    with pytest.raises(ValueError, match="requires a Lightning --checkpoint"):
        _create_runtime_model(
            args,
            cfg,
            tmp_path,
            has_saved_prediction=False,
            saved_prediction_path="",
            tta_cached=False,
        )

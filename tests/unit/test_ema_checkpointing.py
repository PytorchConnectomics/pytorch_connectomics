"""EMAWeightsCallback must persist its average across a checkpoint.

Without `state_dict`/`load_state_dict` the callback re-seeds from the live weights
in `on_train_start`, so every resume silently restarts the average and the
configured decay horizon is a fiction. Worse, with `validate_with_ema=True` the
logged `val_*` metrics -- which is what `ModelCheckpoint` ranks on -- describe the
EMA weights while the saved `state_dict` holds the raw ones, so nothing on disk
reproduces the number that picked the checkpoint.
"""

from types import SimpleNamespace

import pytest
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from connectomics.training.lightning.callbacks import EMAWeightsCallback, load_ema_state_dict


class _Module(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.model = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.model.weight.fill_(value)

    @property
    def device(self):
        return torch.device("cpu")


def _train_steps(cb, module, n, value):
    trainer = SimpleNamespace(global_step=cb._last_global_step)
    for _ in range(n):
        with torch.no_grad():
            module.model.weight.fill_(value)
        trainer.global_step += 1
        cb.on_train_batch_end(trainer, module, {}, {}, 0)


def _start(cb, module, step=0):
    cb.on_train_start(SimpleNamespace(global_step=step), module)


def test_state_dict_roundtrip_preserves_the_average():
    module = _Module(0.0)
    cb = EMAWeightsCallback(decay=0.9, warmup_steps=0)
    _start(cb, module)
    _train_steps(cb, module, 5, value=1.0)

    saved = cb.state_dict()
    ema_before = saved[EMAWeightsCallback.EMA_STATE_KEY]["weight"].clone()
    assert cb._updates == 5
    # 5 steps of decay=0.9 from 0 toward 1 -> 1 - 0.9**5
    assert torch.allclose(ema_before, torch.full((2, 2), 1 - 0.9**5), atol=1e-6)

    # A fresh callback + fresh module = what a resume looks like.
    resumed_module = _Module(1.0)
    resumed = EMAWeightsCallback(decay=0.9, warmup_steps=0)
    resumed.load_state_dict(saved)
    _start(resumed, resumed_module, step=5)

    assert resumed._updates == 5
    assert torch.allclose(resumed._ema_state["weight"], ema_before, atol=1e-6)


def test_without_restore_the_average_restarts_from_the_live_weights():
    """The pre-fix behaviour, pinned so a regression is visible."""
    module = _Module(1.0)
    cb = EMAWeightsCallback(decay=0.9, warmup_steps=0)
    _start(cb, module)  # no load_state_dict first
    assert cb._updates == 0
    assert torch.allclose(cb._ema_state["weight"], torch.ones(2, 2))


def test_restored_state_must_match_the_model():
    module = _Module(0.0)
    cb = EMAWeightsCallback(decay=0.9)
    _start(cb, module)
    _train_steps(cb, module, 2, value=1.0)
    saved = cb.state_dict()
    del saved[EMAWeightsCallback.EMA_STATE_KEY]["weight"]
    saved[EMAWeightsCallback.EMA_STATE_KEY]["not_a_real_key"] = torch.zeros(1)

    resumed = EMAWeightsCallback(decay=0.9)
    resumed.load_state_dict(saved)
    try:
        _start(resumed, _Module(0.0))
    except RuntimeError as exc:
        assert "does not match the model" in str(exc)
    else:  # pragma: no cover - the assertion above is the point of the test
        raise AssertionError("a mismatched EMA state must raise, not load partially")


def test_load_ema_state_dict_finds_the_callback_entry():
    module = _Module(0.0)
    cb = EMAWeightsCallback(decay=0.9)
    _start(cb, module)
    _train_steps(cb, module, 3, value=1.0)

    checkpoint = {
        "callbacks": {
            "ModelCheckpoint{'monitor': 'val_loss_total'}": {"best_model_score": 0.5},
            "EMAWeightsCallback": cb.state_dict(),
        }
    }
    ema = load_ema_state_dict(checkpoint)
    assert ema is not None
    assert torch.allclose(ema["weight"], torch.full((2, 2), 1 - 0.9**3), atol=1e-6)

    # Checkpoints written before this fix carry no EMA at all -- say so, do not
    # hand back the raw weights as if they were the average.
    assert load_ema_state_dict({"callbacks": {"ModelCheckpoint": {}}}) is None
    assert load_ema_state_dict({}) is None


def test_state_dict_is_empty_before_training_starts():
    assert EMAWeightsCallback().state_dict() == {}


class _TrainingProbe(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = nn.Linear(1, 1, bias=False)
        nn.init.zeros_(self.model.weight)
        self.validation_weights = []

    def training_step(self, batch, batch_idx):
        return self.model(batch).mean()

    def validation_step(self, batch, batch_idx):
        self.validation_weights.append(self.model.weight.item())

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


def _trainer(callback, max_steps, accumulation, checkpoint=None):
    return pl.Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=checkpoint is not None,
        enable_model_summary=False,
        enable_progress_bar=False,
        callbacks=[callback, checkpoint] if checkpoint is not None else [callback],
        max_epochs=-1,
        max_steps=max_steps,
        accumulate_grad_batches=accumulation,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=None,
        val_check_interval=2 * accumulation,
    )


@pytest.mark.parametrize("accumulation", [1, 4])
def test_lightning_ema_tracks_optimizer_updates_and_validates_with_ema(accumulation):
    module = _TrainingProbe()
    callback = EMAWeightsCallback(decay=0.5, warmup_steps=1)
    trainer = _trainer(callback, max_steps=4, accumulation=accumulation)

    trainer.fit(
        module,
        DataLoader(torch.ones(4 * accumulation, 1)),
        DataLoader(torch.ones(1, 1)),
    )

    assert trainer.global_step == callback._updates == 4
    assert module.model.weight.item() == pytest.approx(-0.4)
    assert callback._ema_state["weight"].item() == pytest.approx(-0.3125)
    assert module.validation_weights == pytest.approx([-0.15, -0.3125])


def test_lightning_resume_preserves_ema_and_skips_accumulation_microbatches(tmp_path):
    accumulation = 4
    loader = DataLoader(torch.ones(4 * accumulation, 1))
    module = _TrainingProbe()
    callback = EMAWeightsCallback(decay=0.5, warmup_steps=1)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=tmp_path,
        filename="{step}",
        every_n_train_steps=2,
        save_top_k=-1,
        save_on_train_epoch_end=False,
    )
    trainer = _trainer(
        callback, max_steps=4, accumulation=accumulation, checkpoint=checkpoint_callback
    )
    trainer.fit(module, loader)
    checkpoint = tmp_path / "step=2.ckpt"

    resumed_module = _TrainingProbe()
    resumed_callback = EMAWeightsCallback(decay=0.5, warmup_steps=1)
    resumed = _trainer(resumed_callback, max_steps=4, accumulation=accumulation)
    resumed.fit(resumed_module, loader, ckpt_path=checkpoint)

    assert resumed.global_step == resumed_callback._updates == 4
    assert resumed_module.model.weight.item() == pytest.approx(-0.4)
    assert resumed_callback._ema_state["weight"].item() == pytest.approx(-0.3125)

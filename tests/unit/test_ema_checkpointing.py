"""EMAWeightsCallback must persist its average across a checkpoint.

Without `state_dict`/`load_state_dict` the callback re-seeds from the live weights
in `on_fit_start`, so every resume silently restarts the average and the
configured decay horizon is a fiction. Worse, with `validate_with_ema=True` the
logged `val_*` metrics -- which is what `ModelCheckpoint` ranks on -- describe the
EMA weights while the saved `state_dict` holds the raw ones, so nothing on disk
reproduces the number that picked the checkpoint.
"""

import torch
import torch.nn as nn

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
    for _ in range(n):
        with torch.no_grad():
            module.model.weight.fill_(value)
        cb.on_train_batch_end(None, module, {}, {}, 0)


def test_state_dict_roundtrip_preserves_the_average():
    module = _Module(0.0)
    cb = EMAWeightsCallback(decay=0.9, warmup_steps=0)
    cb.on_fit_start(None, module)
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
    resumed.on_fit_start(None, resumed_module)

    assert resumed._updates == 5
    assert torch.allclose(resumed._ema_state["weight"], ema_before, atol=1e-6)


def test_without_restore_the_average_restarts_from_the_live_weights():
    """The pre-fix behaviour, pinned so a regression is visible."""
    module = _Module(1.0)
    cb = EMAWeightsCallback(decay=0.9, warmup_steps=0)
    cb.on_fit_start(None, module)  # no load_state_dict first
    assert cb._updates == 0
    assert torch.allclose(cb._ema_state["weight"], torch.ones(2, 2))


def test_restored_state_must_match_the_model():
    module = _Module(0.0)
    cb = EMAWeightsCallback(decay=0.9)
    cb.on_fit_start(None, module)
    _train_steps(cb, module, 2, value=1.0)
    saved = cb.state_dict()
    del saved[EMAWeightsCallback.EMA_STATE_KEY]["weight"]
    saved[EMAWeightsCallback.EMA_STATE_KEY]["not_a_real_key"] = torch.zeros(1)

    resumed = EMAWeightsCallback(decay=0.9)
    resumed.load_state_dict(saved)
    try:
        resumed.on_fit_start(None, _Module(0.0))
    except RuntimeError as exc:
        assert "does not match the model" in str(exc)
    else:  # pragma: no cover - the assertion above is the point of the test
        raise AssertionError("a mismatched EMA state must raise, not load partially")


def test_load_ema_state_dict_finds_the_callback_entry():
    module = _Module(0.0)
    cb = EMAWeightsCallback(decay=0.9)
    cb.on_fit_start(None, module)
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


def test_state_dict_is_empty_before_fit_start():
    assert EMAWeightsCallback().state_dict() == {}

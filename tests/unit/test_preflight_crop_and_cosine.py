"""Two cross-section checks whose failure modes are otherwise silent.

`data.<split>.crop` smaller than the read patch: `_ensure_minimum_size` pads it
back up, so the run trains on reflect-padded filler and only shows as a loss that
never drops. `CosineAnnealingLR` with `t_max != max_steps`: the LR never reaches
its minimum, so the run misses the anneal that does most of the fitting — the
first SegEM run stopped at 60k of a 200k `t_max` with LR still at 8e-4, and
nothing in the logs said so.
"""

import pytest

from connectomics.config import Config
from connectomics.runtime.preflight import validate_runtime_coherence


def _cfg(patch=(48, 96, 96), context=(0, 0, 0, 10, 10, 10)):
    cfg = Config()
    cfg.model.input_size = list(patch)
    cfg.model.output_size = list(patch)
    cfg.data.dataloader.patch_size = list(patch)
    cfg.data.dataloader.target_context = list(context)
    # keep the scheduler out of the way unless a test is about it
    cfg.optimization.scheduler.name = "MultiStepLR"
    return cfg


# --------------------------------------------------------------------- crop


def test_crop_at_least_the_read_patch_passes():
    cfg = _cfg()
    # read patch is 48+0 x 96+10 x 96+10 = (58, 106, 106); this crop is 124^3
    cfg.data.train.crop = [13, 137, 38, 162, 38, 162]
    cfg.data.val.crop = [13, 137, 38, 162, 38, 162]
    validate_runtime_coherence(cfg)


def test_crop_smaller_than_read_patch_raises():
    cfg = _cfg()
    # 100^3 -- the bare annotated centre -- is 6 voxels short in Y and X
    cfg.data.train.crop = [25, 125, 50, 150, 50, 150]
    with pytest.raises(ValueError, match="target_context"):
        validate_runtime_coherence(cfg)


def test_crop_short_only_on_z_still_raises():
    cfg = _cfg()
    cfg.data.train.crop = [13, 60, 38, 162, 38, 162]  # z spans 47 < 58
    with pytest.raises(ValueError, match="axis 0"):
        validate_runtime_coherence(cfg)


def test_crop_wrong_length_raises():
    cfg = _cfg()
    cfg.data.train.crop = [13, 137, 38, 162]
    with pytest.raises(ValueError, match="must hold 6 values"):
        validate_runtime_coherence(cfg)


def test_crop_reversed_bounds_raise():
    cfg = _cfg()
    cfg.data.val.crop = [137, 13, 38, 162, 38, 162]
    with pytest.raises(ValueError, match="0 <= start < stop"):
        validate_runtime_coherence(cfg)


def test_no_crop_is_fine():
    validate_runtime_coherence(_cfg())


# ------------------------------------------------------------------- cosine


def _cosine(cfg, max_steps, t_max, interval="step"):
    cfg.optimization.scheduler.name = "CosineAnnealingLR"
    cfg.optimization.scheduler.interval = interval
    cfg.optimization.scheduler.params = {"t_max": t_max}
    cfg.optimization.max_steps = max_steps
    return cfg


def test_cosine_horizon_matching_passes():
    validate_runtime_coherence(_cosine(_cfg(), 200000, 200000))


def test_cosine_horizon_mismatch_raises():
    """The exact shape of the first SegEM run: 60k of a 200k cosine."""
    with pytest.raises(ValueError, match="would not reach its minimum"):
        validate_runtime_coherence(_cosine(_cfg(), 60000, 200000))


def test_cosine_shorter_than_run_also_raises():
    """t_max < max_steps restarts the cosine partway through -- also unintended."""
    with pytest.raises(ValueError, match="t_max"):
        validate_runtime_coherence(_cosine(_cfg(), 200000, 50000))


def test_step_unlimited_max_steps_is_not_checked():
    """max_steps=-1 means 'bounded by max_epochs'; t_max is not comparable."""
    validate_runtime_coherence(_cosine(_cfg(), -1, 200000))


def test_epoch_interval_cosine_is_not_checked():
    """An epoch-interval t_max counts epochs, so comparing it to steps is wrong."""
    validate_runtime_coherence(_cosine(_cfg(), 60000, 200, interval="epoch"))


def test_other_schedulers_are_untouched():
    cfg = _cfg()
    cfg.optimization.scheduler.name = "MultiStepLR"
    cfg.optimization.scheduler.params = {"t_max": 999}
    cfg.optimization.max_steps = 60000
    validate_runtime_coherence(cfg)

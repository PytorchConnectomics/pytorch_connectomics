import pytest
import torch

from connectomics.config import Config
from connectomics.inference.tta import TTAPredictor


def _forward_constant(x: torch.Tensor) -> torch.Tensor:
    # Return deterministic 2-channel logits for masking assertions.
    return torch.ones((x.shape[0], 2, *x.shape[2:]), device=x.device, dtype=x.dtype)


def _forward_expand_width(x: torch.Tensor) -> torch.Tensor:
    # Simulate a model/output pipeline that changes spatial size (e.g., 514 -> 516).
    out = torch.ones((x.shape[0], 2, *x.shape[2:]), device=x.device, dtype=x.dtype)
    return torch.nn.functional.pad(out, (1, 1, 0, 0, 0, 0))


def test_tta_applies_mask_to_predictions_by_default():
    cfg = Config()
    cfg.inference.test_time_augmentation.enabled = False
    # Default should apply mask when provided.
    assert cfg.inference.test_time_augmentation.apply_mask is True

    predictor = TTAPredictor(cfg=cfg, sliding_inferer=None, forward_fn=_forward_constant)

    images = torch.zeros((1, 1, 4, 4, 4), dtype=torch.float32)
    mask = torch.zeros((1, 1, 4, 4, 4), dtype=torch.float32)
    mask[:, :, 1:3, 1:3, 1:3] = 1.0

    pred = predictor.predict(images, mask=mask)

    assert pred.shape == (1, 2, 4, 4, 4)
    # Outside mask must be zero after masking.
    assert torch.all(pred[:, :, 0, :, :] == 0)
    # Inside mask region should retain the original value of 1.
    assert torch.all(pred[:, :, 1:3, 1:3, 1:3] == 1)


def test_tta_masking_can_be_disabled_explicitly():
    cfg = Config()
    cfg.inference.test_time_augmentation.enabled = False
    cfg.inference.test_time_augmentation.apply_mask = False

    predictor = TTAPredictor(cfg=cfg, sliding_inferer=None, forward_fn=_forward_constant)

    images = torch.zeros((1, 1, 4, 4, 4), dtype=torch.float32)
    mask = torch.zeros((1, 1, 4, 4, 4), dtype=torch.float32)

    pred = predictor.predict(images, mask=mask)
    # With masking disabled, output stays unmasked.
    assert torch.all(pred == 1)


def test_tta_raises_on_mismatched_mask_shape():
    cfg = Config()
    cfg.inference.test_time_augmentation.enabled = False
    cfg.inference.test_time_augmentation.apply_mask = True

    predictor = TTAPredictor(cfg=cfg, sliding_inferer=None, forward_fn=_forward_expand_width)

    images = torch.zeros((1, 1, 133, 516, 514), dtype=torch.float32)
    # All-zero mask with original input width; after alignment it should still zero everything.
    mask = torch.zeros((1, 1, 133, 516, 514), dtype=torch.float32)

    with pytest.raises(ValueError, match="Mask spatial shape must exactly match"):
        predictor.predict(images, mask=mask)


def test_tta_allows_minor_mask_alignment_when_enabled():
    cfg = Config()
    cfg.inference.test_time_augmentation.enabled = False
    cfg.inference.test_time_augmentation.apply_mask = True

    predictor = TTAPredictor(cfg=cfg, sliding_inferer=None, forward_fn=_forward_expand_width)

    images = torch.zeros((1, 1, 133, 516, 514), dtype=torch.float32)
    mask = torch.zeros((1, 1, 133, 516, 514), dtype=torch.float32)

    pred = predictor.predict(
        images,
        mask=mask,
        mask_align_to_image=True,
    )
    assert pred.shape == (1, 2, 133, 516, 516)
    assert torch.all(pred == 0)

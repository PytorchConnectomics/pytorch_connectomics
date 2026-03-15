import pytest
import torch

from connectomics.config import Config
from connectomics.inference.tta import TTAPredictor
from connectomics.training.lightning.utils import compute_tta_passes


def _forward_constant(x: torch.Tensor) -> torch.Tensor:
    # Return deterministic 2-channel logits for masking assertions.
    return torch.ones((x.shape[0], 2, *x.shape[2:]), device=x.device, dtype=x.dtype)


def _forward_expand_width(x: torch.Tensor) -> torch.Tensor:
    # Simulate a model/output pipeline that changes spatial size (e.g., 514 -> 516).
    out = torch.ones((x.shape[0], 2, *x.shape[2:]), device=x.device, dtype=x.dtype)
    return torch.nn.functional.pad(out, (1, 1, 0, 0, 0, 0))


def _forward_three_channel_logits(x: torch.Tensor) -> torch.Tensor:
    shape = (x.shape[0], 1, *x.shape[2:])
    ch0 = torch.full(shape, -2.0, device=x.device, dtype=x.dtype)
    ch1 = torch.zeros(shape, device=x.device, dtype=x.dtype)
    ch2 = torch.full(shape, 2.0, device=x.device, dtype=x.dtype)
    return torch.cat([ch0, ch1, ch2], dim=1)


def test_tta_applies_mask_to_predictions_by_default():
    cfg = Config()
    cfg.inference.test_time_augmentation.enabled = False
    # apply_mask is optional in schema; predictor defaults to True when absent.
    assert getattr(cfg.inference.test_time_augmentation, "apply_mask", True) is True

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
    setattr(cfg.inference.test_time_augmentation, "apply_mask", False)

    predictor = TTAPredictor(cfg=cfg, sliding_inferer=None, forward_fn=_forward_constant)

    images = torch.zeros((1, 1, 4, 4, 4), dtype=torch.float32)
    mask = torch.zeros((1, 1, 4, 4, 4), dtype=torch.float32)

    pred = predictor.predict(images, mask=mask)
    # With masking disabled, output stays unmasked.
    assert torch.all(pred == 1)


def test_tta_raises_on_mismatched_mask_shape():
    cfg = Config()
    cfg.inference.test_time_augmentation.enabled = False
    setattr(cfg.inference.test_time_augmentation, "apply_mask", True)

    predictor = TTAPredictor(cfg=cfg, sliding_inferer=None, forward_fn=_forward_expand_width)

    images = torch.zeros((1, 1, 133, 516, 514), dtype=torch.float32)
    # All-zero mask with original input width; after alignment it should still zero everything.
    mask = torch.zeros((1, 1, 133, 516, 514), dtype=torch.float32)

    with pytest.raises(ValueError, match="Mask spatial shape must exactly match"):
        predictor.predict(images, mask=mask)


def test_tta_allows_minor_mask_alignment_when_enabled():
    cfg = Config()
    cfg.inference.test_time_augmentation.enabled = False
    setattr(cfg.inference.test_time_augmentation, "apply_mask", True)

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


def test_tta_accepts_nested_singleton_list_masks():
    cfg = Config()
    cfg.inference.test_time_augmentation.enabled = False
    setattr(cfg.inference.test_time_augmentation, "apply_mask", True)

    predictor = TTAPredictor(cfg=cfg, sliding_inferer=None, forward_fn=_forward_constant)

    images = torch.zeros((1, 1, 4, 4, 4), dtype=torch.float32)
    nested_mask = [[torch.zeros((1, 4, 4, 4), dtype=torch.float32)]]

    pred = predictor.predict(images, mask=nested_mask)

    assert pred.shape == (1, 2, 4, 4, 4)
    assert torch.all(pred == 0)


def test_tta_channel_activations_colon_applies_to_all_channels():
    cfg = Config()
    cfg.model.out_channels = 3
    cfg.inference.test_time_augmentation.enabled = False
    cfg.inference.test_time_augmentation.channel_activations = [
        {"channels": ":", "activation": "sigmoid"}
    ]

    predictor = TTAPredictor(
        cfg=cfg, sliding_inferer=None, forward_fn=_forward_three_channel_logits
    )

    images = torch.zeros((1, 1, 2, 2, 2), dtype=torch.float32)
    pred = predictor.predict(images)

    assert pred.shape == (1, 3, 2, 2, 2)
    expected = torch.sigmoid(_forward_three_channel_logits(images))
    assert torch.allclose(pred, expected)


def test_tta_channel_activations_follow_python_slice_semantics():
    cfg = Config()
    cfg.model.out_channels = 3
    cfg.inference.test_time_augmentation.enabled = False
    cfg.inference.test_time_augmentation.channel_activations = [
        {"channels": "0:-1", "activation": "sigmoid"},
        {"channels": "-1:", "activation": "tanh"},
    ]

    predictor = TTAPredictor(
        cfg=cfg, sliding_inferer=None, forward_fn=_forward_three_channel_logits
    )

    images = torch.zeros((1, 1, 2, 2, 2), dtype=torch.float32)
    pred = predictor.predict(images)

    assert pred.shape == (1, 3, 2, 2, 2)
    expected = _forward_three_channel_logits(images)
    expected[:, 0:2, ...] = torch.sigmoid(expected[:, 0:2, ...])
    expected[:, 2:3, ...] = torch.tanh(expected[:, 2:3, ...])
    assert torch.allclose(pred, expected)


def test_distributed_tta_reduction_raises_on_mismatched_rank_shapes(monkeypatch):
    cfg = Config()
    predictor = TTAPredictor(cfg=cfg, sliding_inferer=None, forward_fn=_forward_constant)

    monkeypatch.setattr(predictor, "_distributed_context", lambda: (True, 0, 2))

    def _fake_all_gather(output_tensors, _input_tensor):
        output_tensors[0].copy_(torch.tensor([5, 1, 2, 4, 4, 4, 0], dtype=torch.int64))
        output_tensors[1].copy_(torch.tensor([5, 1, 2, 6, 4, 4, 0], dtype=torch.int64))

    monkeypatch.setattr(torch.distributed, "all_gather", _fake_all_gather)

    with pytest.raises(RuntimeError, match="same shape"):
        predictor._validate_distributed_reduction_shape(
            torch.zeros((1, 2, 4, 4, 4), dtype=torch.float32),
            reduction_device=torch.device("cpu"),
        )


def test_tta_rotate90_k_subset_is_honored():
    cfg = Config()
    cfg.inference.test_time_augmentation.enabled = True
    cfg.inference.test_time_augmentation.flip_axes = None
    cfg.inference.test_time_augmentation.rotation90_axes = [[1, 2]]
    cfg.inference.test_time_augmentation.rotate90_k = [1, 3]

    predictor = TTAPredictor(cfg=cfg, sliding_inferer=None, forward_fn=_forward_constant)
    combinations = predictor._build_augmentation_combinations(
        cfg.inference.test_time_augmentation,
        ndim=5,
    )

    assert combinations == [
        ([], (3, 4), 1),
        ([], (3, 4), 3),
    ]
    assert compute_tta_passes(cfg, spatial_dims=3) == 2


def test_tta_deduplicates_redundant_xy_flip_rotation_combinations():
    cfg = Config()
    cfg.inference.test_time_augmentation.enabled = True
    cfg.inference.test_time_augmentation.flip_axes = "all"
    cfg.inference.test_time_augmentation.rotation90_axes = [[1, 2]]

    predictor = TTAPredictor(cfg=cfg, sliding_inferer=None, forward_fn=_forward_constant)
    combinations = predictor._build_augmentation_combinations(
        cfg.inference.test_time_augmentation,
        ndim=5,
    )
    unique_combinations = {(tuple(flip_axes), rotation_plane, k) for flip_axes, rotation_plane, k in combinations}

    assert len(combinations) == 16
    assert len(unique_combinations) == 16
    assert compute_tta_passes(cfg, spatial_dims=3) == 16

import copy

import pytest
import torch

import connectomics.inference.tta as tta_module
from connectomics.config import Config
from connectomics.inference.sliding import build_sliding_inferer
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


def _forward_affine_logits(x: torch.Tensor) -> torch.Tensor:
    return torch.cat([x + 0.5, x * 2.0 - 0.25], dim=1)


def _forward_multi_head_logits(x: torch.Tensor):
    affinity = torch.cat([x + 0.5, x * 2.0 - 0.25], dim=1)
    sdt = x - 0.75
    return {"output": {"affinity": affinity, "sdt": sdt}}


class _TrackingSlidingInferer:
    def __init__(self, inferer):
        self.inferer = inferer
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self.inferer(*args, **kwargs)


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


def test_tta_channel_activations_sigmoid_is_in_place_for_all_channels():
    cfg = Config()
    cfg.model.out_channels = 3
    cfg.inference.test_time_augmentation.enabled = False
    cfg.inference.test_time_augmentation.channel_activations = [
        {"channels": ":", "activation": "sigmoid"}
    ]

    predictor = TTAPredictor(
        cfg=cfg, sliding_inferer=None, forward_fn=_forward_three_channel_logits
    )

    tensor = _forward_three_channel_logits(torch.zeros((1, 1, 2, 2, 2), dtype=torch.float32))
    original_ptr = tensor.data_ptr()
    out = predictor.apply_preprocessing(tensor)

    assert out.data_ptr() == original_ptr
    assert torch.allclose(out, torch.sigmoid(_forward_three_channel_logits(tensor[:, :1])))


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


def test_tta_selects_named_output_head_before_channel_preprocessing():
    cfg = Config()
    cfg.model.out_channels = 3
    cfg.model.primary_head = "affinity"
    cfg.model.heads = {
        "affinity": {"out_channels": 2, "num_blocks": 0},
        "sdt": {"out_channels": 1, "num_blocks": 0},
    }
    cfg.inference.head = "affinity"
    cfg.inference.test_time_augmentation.enabled = False
    cfg.inference.test_time_augmentation.channel_activations = [
        {"channels": ":", "activation": "sigmoid"}
    ]
    cfg.inference.select_channel = 1

    predictor = TTAPredictor(cfg=cfg, sliding_inferer=None, forward_fn=_forward_multi_head_logits)

    images = torch.zeros((1, 1, 2, 2, 2), dtype=torch.float32)
    pred = predictor.predict(images)

    assert pred.shape == (1, 1, 2, 2, 2)
    expected_affinity = _forward_multi_head_logits(images)["output"]["affinity"]
    expected = torch.sigmoid(expected_affinity[:, 1:2, ...])
    assert torch.allclose(pred, expected)


def test_tta_requested_head_override_selects_alternate_named_head():
    cfg = Config()
    cfg.model.out_channels = 3
    cfg.model.primary_head = "affinity"
    cfg.model.heads = {
        "affinity": {"out_channels": 2, "num_blocks": 0},
        "sdt": {"out_channels": 1, "num_blocks": 0},
    }
    cfg.inference.head = "affinity"
    cfg.inference.test_time_augmentation.enabled = False

    predictor = TTAPredictor(cfg=cfg, sliding_inferer=None, forward_fn=_forward_multi_head_logits)

    images = torch.zeros((1, 1, 2, 2, 2), dtype=torch.float32)
    pred = predictor.predict(images, requested_head="sdt")

    assert pred.shape == (1, 1, 2, 2, 2)
    expected = _forward_multi_head_logits(images)["output"]["sdt"]
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
    unique_combinations = {
        (tuple(flip_axes), rotation_plane, k) for flip_axes, rotation_plane, k in combinations
    }

    assert len(combinations) == 16
    assert len(unique_combinations) == 16
    assert compute_tta_passes(cfg, spatial_dims=3) == 16


def test_patch_first_local_tta_matches_standard_sliding_output():
    cfg = Config()
    cfg.inference.test_time_augmentation.enabled = True
    cfg.inference.test_time_augmentation.flip_axes = [0]
    cfg.inference.test_time_augmentation.rotation90_axes = [[1, 2]]
    cfg.inference.test_time_augmentation.rotate90_k = [0, 1]
    cfg.inference.sliding_window.window_size = [2, 2, 2]
    cfg.inference.sliding_window.sw_batch_size = 2
    cfg.inference.sliding_window.overlap = 0.5
    cfg.inference.sliding_window.padding_mode = "constant"
    cfg.inference.output_act = "sigmoid"

    images = torch.arange(1, 1 + 4 * 4 * 4, dtype=torch.float32).reshape(1, 1, 4, 4, 4) / 64.0
    mask = torch.ones((1, 1, 4, 4, 4), dtype=torch.float32)
    mask[:, :, 0, :, :] = 0.0

    standard_cfg = copy.deepcopy(cfg)
    standard_inferer = _TrackingSlidingInferer(build_sliding_inferer(standard_cfg))
    standard_predictor = TTAPredictor(
        cfg=standard_cfg,
        sliding_inferer=standard_inferer,
        forward_fn=_forward_affine_logits,
    )
    standard_pred = standard_predictor.predict(images, mask=mask)

    patch_cfg = copy.deepcopy(cfg)
    patch_cfg.inference.test_time_augmentation.patch_first_local = True
    patch_inferer = _TrackingSlidingInferer(build_sliding_inferer(patch_cfg))
    patch_predictor = TTAPredictor(
        cfg=patch_cfg,
        sliding_inferer=patch_inferer,
        forward_fn=_forward_affine_logits,
    )
    patch_pred = patch_predictor.predict(images, mask=mask)

    assert standard_inferer.calls == compute_tta_passes(standard_cfg, spatial_dims=3)
    assert patch_inferer.calls == 0
    assert torch.allclose(patch_pred, standard_pred, atol=1e-6, rtol=1e-6)


def test_patch_first_local_tta_rejects_odd_rotations_on_unequal_axes():
    cfg = Config()
    cfg.inference.test_time_augmentation.enabled = True
    cfg.inference.test_time_augmentation.patch_first_local = True
    cfg.inference.test_time_augmentation.flip_axes = None
    cfg.inference.test_time_augmentation.rotation90_axes = [[0, 1]]
    cfg.inference.test_time_augmentation.rotate90_k = [1]
    cfg.inference.sliding_window.window_size = [2, 2, 2]

    predictor = TTAPredictor(
        cfg=cfg,
        sliding_inferer=object(),
        forward_fn=_forward_constant,
    )

    with pytest.raises(ValueError, match="equal image and ROI sizes"):
        predictor.predict(torch.zeros((1, 1, 2, 3, 4), dtype=torch.float32))


def test_patch_first_local_tta_reports_progress(monkeypatch):
    progress_instances = []

    class _FakeProgressBar:
        def __init__(self, *, total, desc, leave):
            self.total = total
            self.desc = desc
            self.leave = leave
            self.updated = 0
            self.closed = False

        def update(self, n=1):
            self.updated += n

        def close(self):
            self.closed = True

    def _fake_tqdm(*, total, desc, leave):
        bar = _FakeProgressBar(total=total, desc=desc, leave=leave)
        progress_instances.append(bar)
        return bar

    monkeypatch.setattr(tta_module, "tqdm", _fake_tqdm)

    cfg = Config()
    cfg.inference.test_time_augmentation.enabled = True
    cfg.inference.test_time_augmentation.patch_first_local = True
    cfg.inference.test_time_augmentation.flip_axes = [0]
    cfg.inference.test_time_augmentation.rotation90_axes = None
    cfg.inference.sliding_window.window_size = [2, 2, 2]
    cfg.inference.sliding_window.sw_batch_size = 2

    predictor = TTAPredictor(
        cfg=cfg,
        sliding_inferer=object(),
        forward_fn=_forward_constant,
    )

    pred = predictor.predict(torch.zeros((1, 1, 4, 4, 4), dtype=torch.float32))

    assert pred.shape == (1, 2, 4, 4, 4)
    assert len(progress_instances) == 1
    assert progress_instances[0].total == 14
    assert progress_instances[0].updated == 14
    assert progress_instances[0].closed is True
    assert progress_instances[0].desc == "Patch-first TTA x2"

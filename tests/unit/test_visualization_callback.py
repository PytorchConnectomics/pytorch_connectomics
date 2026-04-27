import torch

from connectomics.config import Config
from connectomics.training.lightning.callbacks import (
    VisualizationCallback,
    _apply_affinity_visualization_crop_if_needed,
)


def test_affinity_visualization_crop_matches_full_affinity_training_crop():
    cfg = Config()
    cfg.data.label_transform.targets = [
        {
            "name": "affinity",
            "kwargs": {
                "offsets": ["0-0-1", "0-1-0", "1-0-0", "2-0-0", "3-0-0", "4-0-0"],
                "affinity_mode": "deepem",
            },
        }
    ]

    image = torch.zeros((1, 1, 16, 32, 32), dtype=torch.float32)
    label = torch.zeros((1, 6, 16, 32, 32), dtype=torch.float32)
    pred = torch.zeros((1, 6, 16, 32, 32), dtype=torch.float32)
    mask = torch.ones((1, 1, 16, 32, 32), dtype=torch.float32)

    image_c, label_c, pred_c, mask_c = _apply_affinity_visualization_crop_if_needed(
        cfg,
        image=image,
        label=label,
        pred=pred,
        mask=mask,
    )

    assert tuple(image_c.shape) == (1, 1, 12, 31, 31)
    assert tuple(label_c.shape) == (1, 6, 12, 31, 31)
    assert tuple(pred_c.shape) == (1, 6, 12, 31, 31)
    assert mask_c is not None
    assert tuple(mask_c.shape) == (1, 1, 12, 31, 31)


def test_affinity_visualization_crop_skips_mixed_task_tensors():
    cfg = Config()
    cfg.data.label_transform.targets = [
        {
            "name": "affinity",
            "kwargs": {
                "offsets": [
                    "0-0-1",
                    "0-1-0",
                    "1-0-0",
                    "0-0-5",
                    "0-5-0",
                    "5-0-0",
                    "0-0-17",
                    "0-17-0",
                    "17-0-0",
                ],
                "affinity_mode": "deepem",
            },
        },
        {
            "name": "skeleton_aware_edt",
            "kwargs": {
                "alpha": 0.8,
                "bg_value": -1.0,
                "relabel": True,
            },
        },
    ]

    image = torch.zeros((1, 1, 16, 32, 32), dtype=torch.float32)
    label = torch.ones((1, 10, 16, 32, 32), dtype=torch.float32)
    pred = torch.ones((1, 10, 16, 32, 32), dtype=torch.float32)

    image_c, label_c, pred_c, mask_c = _apply_affinity_visualization_crop_if_needed(
        cfg,
        image=image,
        label=label,
        pred=pred,
        mask=None,
    )

    assert image_c is image
    assert tuple(label_c.shape) == tuple(label.shape)
    assert tuple(pred_c.shape) == tuple(pred.shape)
    assert torch.equal(label_c[:, 9:], torch.ones_like(label_c[:, 9:]))
    assert torch.equal(pred_c[:, 9:], torch.ones_like(pred_c[:, 9:]))
    assert torch.count_nonzero(label_c[:, 8]) == 0
    assert torch.count_nonzero(pred_c[:, 8]) == 0
    assert torch.count_nonzero(label_c[:, 2]) > 0
    assert torch.count_nonzero(pred_c[:, 2]) > 0
    assert mask_c is None


def test_affinity_visualization_crop_skips_partial_affinity_groups():
    cfg = Config()
    cfg.data.label_transform.targets = [
        {
            "name": "affinity",
            "kwargs": {
                "offsets": [
                    "0-0-1",
                    "0-1-0",
                    "1-0-0",
                    "0-0-5",
                    "0-5-0",
                    "5-0-0",
                    "0-0-17",
                    "0-17-0",
                    "17-0-0",
                ],
                "affinity_mode": "deepem",
            },
        },
        {
            "name": "skeleton_aware_edt",
            "kwargs": {
                "alpha": 0.8,
                "bg_value": -1.0,
                "relabel": True,
            },
        },
    ]

    image = torch.zeros((1, 1, 16, 32, 32), dtype=torch.float32)
    label = torch.ones((1, 1, 16, 32, 32), dtype=torch.float32)
    pred = torch.ones((1, 1, 16, 32, 32), dtype=torch.float32)

    image_c, label_c, pred_c, mask_c = _apply_affinity_visualization_crop_if_needed(
        cfg,
        image=image,
        label=label,
        pred=pred,
        mask=None,
    )

    assert image_c is image
    assert label_c is not label
    assert pred_c is not pred
    assert tuple(label_c.shape) == tuple(label.shape)
    assert tuple(pred_c.shape) == tuple(pred.shape)
    assert torch.equal(label_c, torch.ones_like(label_c))
    assert torch.equal(pred_c, torch.ones_like(pred_c))
    assert mask_c is None


class _DummyVisualizationModule:
    @staticmethod
    def _resolve_validation_target_slice(head_name: str):
        assert head_name == "sdt"
        return "2:3"

    @staticmethod
    def _slice_tensor_channels(tensor: torch.Tensor, channel_selector, *, context: str):
        assert channel_selector == "2:3"
        assert "sdt" in context
        return tensor[:, 2:3, ...]


def test_visualization_callback_selects_named_head_and_label_slice():
    cfg = Config()
    cfg.model.out_channels = 3
    cfg.model.primary_head = "affinity"
    cfg.model.heads = {
        "affinity": {"out_channels": 2, "num_blocks": 0},
        "sdt": {"out_channels": 1, "num_blocks": 0},
    }
    cfg.monitor.logging.images.head = "sdt"
    callback = VisualizationCallback(cfg)

    label = torch.cat(
        [
            torch.zeros((1, 2, 4, 4, 4), dtype=torch.float32),
            torch.full((1, 1, 4, 4, 4), 7.0, dtype=torch.float32),
        ],
        dim=1,
    )
    pred = {
        "output": {
            "affinity": torch.ones((1, 2, 4, 4, 4), dtype=torch.float32),
            "sdt": torch.full((1, 1, 4, 4, 4), 3.0, dtype=torch.float32),
        }
    }

    label_sel, pred_sel, resolved_head = callback._select_visualization_tensors(
        _DummyVisualizationModule(),
        label,
        pred,
    )

    assert resolved_head == "sdt"
    assert tuple(label_sel.shape) == (1, 1, 4, 4, 4)
    assert tuple(pred_sel.shape) == (1, 1, 4, 4, 4)
    assert torch.equal(label_sel, torch.full_like(label_sel, 7.0))
    assert torch.equal(pred_sel, torch.full_like(pred_sel, 3.0))

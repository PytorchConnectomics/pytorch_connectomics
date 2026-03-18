import torch

from connectomics.config import Config
from connectomics.training.lightning.callbacks import (
    _apply_affinity_visualization_crop_if_needed,
)


def test_affinity_visualization_crop_matches_full_affinity_training_crop():
    cfg = Config()
    cfg.data.label_transform.targets = [
        {
            "name": "affinity",
            "kwargs": {
                "offsets": ["0-0-1", "0-1-0", "1-0-0", "2-0-0", "3-0-0", "4-0-0"],
                "deepem_crop": True,
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
                "offsets": ["0-0-1", "0-1-0", "1-0-0", "0-0-5", "0-5-0", "5-0-0", "0-0-17", "0-17-0", "17-0-0"],
                "deepem_crop": True,
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

from __future__ import annotations

import numpy as np
import pytest
import torch

from connectomics.config import Config
from connectomics.data.augmentation.build import _build_augmentations, build_train_transforms
from connectomics.data.processing.transforms import RelabelConnectedComponentsd


def test_target_context_generates_banis_affinity_before_leading_crop():
    cfg = Config()
    cfg.data.dataloader.patch_size = [4, 4, 4]
    cfg.data.dataloader.target_context = [2, 2, 2]
    cfg.data.image_transform.normalize = "none"
    cfg.data.augmentation = None
    cfg.data.label_transform.targets = [
        {
            "name": "affinity",
            "kwargs": {
                "offsets": ["2-0-0"],
                "affinity_mode": "banis",
            },
        }
    ]

    transforms = build_train_transforms(cfg, keys=["image", "label"], skip_loading=True)
    sample = {
        "image": np.ones((1, 6, 6, 6), dtype=np.float32),
        "label": np.ones((1, 6, 6, 6), dtype=np.int32),
    }

    out = transforms(sample)

    assert tuple(out["image"].shape) == (1, 4, 4, 4)
    assert tuple(out["label"].shape) == (1, 4, 4, 4)
    assert torch.all(out["label"] == 1)
    assert tuple(out["label_mask"].shape) == (1, 4, 4, 4)
    assert torch.all(out["label_mask"])


def test_target_context_supports_asymmetric_pre_post_extension():
    cfg = Config()
    cfg.data.dataloader.patch_size = [4, 4, 4]
    cfg.data.dataloader.target_context = [1, 0, 0, 1, 2, 2]
    cfg.data.image_transform.normalize = "none"
    cfg.data.augmentation = None
    cfg.data.label_transform.targets = [
        {
            "name": "affinity",
            "kwargs": {
                "offsets": ["2-0-0"],
                "affinity_mode": "banis",
            },
        }
    ]

    transforms = build_train_transforms(cfg, keys=["image", "label"], skip_loading=True)
    leading = np.arange(6 * 4 * 6, dtype=np.float32).reshape(6, 4, 6) * 0 + 1
    sample = {
        "image": np.ones((1, 6, 4, 6), dtype=np.float32),
        "label": np.ones((1, 6, 4, 6), dtype=np.int32),
    }

    out = transforms(sample)

    assert tuple(out["image"].shape) == (1, 4, 4, 4)
    assert tuple(out["label"].shape) == (1, 4, 4, 4)


def test_affinity_pipeline_emits_paired_label_and_mask_keys():
    """Train pipeline ends with bool ``label_mask`` paired with ``label``."""
    cfg = Config()
    cfg.data.dataloader.patch_size = [4, 4, 4]
    cfg.data.dataloader.target_context = [0, 0, 0, 0, 0, 0]
    cfg.data.image_transform.normalize = "none"
    cfg.data.augmentation = None
    cfg.data.label_transform.targets = [
        {
            "name": "affinity",
            "kwargs": {
                "offsets": ["1-0-0", "0-1-0", "0-0-1"],
                "affinity_mode": "banis",
            },
        }
    ]

    transforms = build_train_transforms(cfg, keys=["image", "label"], skip_loading=True)

    seg = np.ones((1, 4, 4, 4), dtype=np.int32)
    seg[0, 0, 0, 0] = -1
    sample = {
        "image": np.ones((1, 4, 4, 4), dtype=np.float32),
        "label": seg,
    }

    out = transforms(sample)

    assert "label" in out and "label_mask" in out
    assert out["label"].dtype == torch.float32
    assert out["label_mask"].dtype == torch.bool
    assert out["label"].shape == out["label_mask"].shape == (3, 4, 4, 4)
    # Voxels touching seg == -1 must be masked out for any affinity offset
    # whose source or destination lands on (0, 0, 0).
    assert not bool(out["label_mask"][0, 0, 0, 0])


def test_binary_only_pipeline_skips_label_mask_emission():
    """Targets without affinity set ``use_mask=False`` and emit no mask key."""
    cfg = Config()
    cfg.data.dataloader.patch_size = [4, 4, 4]
    cfg.data.dataloader.target_context = [0, 0, 0, 0, 0, 0]
    cfg.data.image_transform.normalize = "none"
    cfg.data.augmentation = None
    cfg.data.label_transform.targets = [{"name": "binary"}]

    transforms = build_train_transforms(cfg, keys=["image", "label"], skip_loading=True)

    seg = np.ones((1, 4, 4, 4), dtype=np.int32)
    seg[0, 0, 0, 0] = -1
    sample = {
        "image": np.ones((1, 4, 4, 4), dtype=np.float32),
        "label": seg,
    }

    out = transforms(sample)

    assert "label" in out
    assert "label_mask" not in out


def test_multi_task_label_transform_use_mask_attribute_reflects_config():
    """``MultiTaskLabelTransformd.use_mask`` should agree with task list."""
    from connectomics.data.processing.transforms import MultiTaskLabelTransformd

    binary_only = MultiTaskLabelTransformd(
        keys=["label"],
        tasks=[{"name": "binary"}],
    )
    assert binary_only.use_mask is False

    aff_plus_binary = MultiTaskLabelTransformd(
        keys=["label"],
        tasks=[
            {"name": "binary"},
            {
                "name": "affinity",
                "kwargs": {"offsets": ["1-0-0"], "affinity_mode": "banis"},
            },
        ],
    )
    assert aff_plus_binary.use_mask is True


def test_relabel_connected_components_splits_same_id_components_and_preserves_unlabeled():
    pytest.importorskip("cc3d")
    seg = np.zeros((1, 5, 5, 5), dtype=np.int32)
    seg[0, 0, 0, 0] = 7
    seg[0, 4, 4, 4] = 7
    seg[0, 2, 2, 2] = -1

    out = RelabelConnectedComponentsd(keys=["label"])({"label": seg})
    relabeled = out["label"][0]

    positive_ids = set(np.unique(relabeled[relabeled > 0]).tolist())
    assert len(positive_ids) == 2
    assert relabeled[2, 2, 2] == -1
    assert relabeled[0, 0, 1] == 0


def test_banis_style_intensity_order_is_mul_add_then_noise():
    cfg = Config().data.augmentation
    cfg.intensity.enabled = True
    cfg.intensity.banis_style = True
    cfg.intensity.mul_add_prob = 0.5
    cfg.intensity.gaussian_noise_prob = 0.5

    transforms = _build_augmentations(cfg, keys=["image", "label"])

    assert [type(t).__name__ for t in transforms] == [
        "RandMulAddIntensityd",
        "RandGaussianNoised",
    ]

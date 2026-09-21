from pathlib import Path

import numpy as np
import torch

from connectomics.config import load_config, resolve_default_profiles
from connectomics.data.augmentation.build import build_train_transforms
from connectomics.data.processing.segment import seg_erosion_instance
from connectomics.models.losses.embedding import EmbeddingMeanLoss


def test_real_config_label_pipeline_preserves_thin_identity():
    configs = []
    for name in ("base_banis+.yaml", "base_banis+_embed12.yaml"):
        cfg = resolve_default_profiles(
            load_config(Path(__file__).parents[2] / "tutorials/neuron_nisb" / name)
        )
        cfg.data.augmentation = None
        cfg.data.image_transform.normalize = "none"
        cfg.data.dataloader.patch_size = [32, 32, 32]
        cfg.data.dataloader.target_context = [10, 10, 10]
        configs.append(cfg)
    embedding_term = configs[1].model.loss.losses[1]
    assert embedding_term["function"] == "EmbeddingMeanLoss"
    assert embedding_term["weight"] == 0.3
    seg = np.zeros((1, 42, 42, 42), dtype=np.int32)
    seg[:, 4:28, 4:16, 4:24] = 1
    seg[:, 4:28, 16:28, 4:24] = 2
    seg[:, 10:24, 12:14, 24:30] = 1  # Two-voxel process adjacent to instance 2.
    seg[:, 10:24, 14:20, 24:30] = 2
    seg[:, 29:31] = -1
    thin = np.zeros((32, 32, 32), dtype=bool)
    thin[10:24, 12:14, 25:29] = True
    eroded = seg_erosion_instance(seg[0], tsz_h=2)
    assert np.all(eroded[:32, :32, :32][thin] == 0)
    outputs = [
        build_train_transforms(cfg, skip_loading=True)(
            {"image": np.ones_like(seg, dtype=np.float32), "label": seg.copy()}
        )
        for cfg in configs
    ]
    baseline, embed = outputs
    gt = torch.as_tensor(embed["gt_seg"])
    torch.testing.assert_close(gt, torch.from_numpy(seg[:, :32, :32, :32]), rtol=0, atol=0)
    for key in ("label", "label_mask"):
        torch.testing.assert_close(embed[key], baseline[key], rtol=0, atol=0)
    thin = torch.from_numpy(thin)
    assert (gt[0][thin] > 0).all()
    assert embed["label_mask"][:3, thin].all()
    ignored = gt[0] < 0
    assert not embed["label_mask"][:, ignored].any()
    torch.manual_seed(5)
    pred = torch.randn(1, 12, 32, 32, 32, requires_grad=True)
    loss = EmbeddingMeanLoss()(pred, None, gt_seg=gt[None], mask=embed["label_mask"][None, :3])
    loss.backward()
    assert (pred.grad[0, :, thin].abs().sum(0) > 0).all()
    assert torch.count_nonzero(pred.grad[0, :, ignored]) == 0

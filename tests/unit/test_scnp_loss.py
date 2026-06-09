"""Tests for ScnpLoss (Same-Class Neighbor Penalization)."""

import unittest

import torch

from connectomics.models.losses import create_loss
from connectomics.models.losses.losses import PerChannelBCEWithLogitsLoss, ScnpLoss


class TestScnpLoss(unittest.TestCase):
    def test_registry_and_metadata(self):
        loss = create_loss("ScnpLoss", neighborhood_size=3)
        self.assertIsInstance(loss, ScnpLoss)
        meta = getattr(loss, "_connectomics_loss_metadata")
        self.assertEqual(meta.name, "ScnpLoss")
        self.assertEqual(meta.call_kind, "pred_target")
        self.assertEqual(meta.spatial_weight_arg, "weight")

    def test_even_neighborhood_rejected(self):
        with self.assertRaises(ValueError):
            ScnpLoss(neighborhood_size=2)

    def test_identity_when_logits_uniform_per_class(self):
        # When every same-class neighbor shares the logit, SCNP pooling is a
        # no-op and the loss equals plain per-channel BCE on the raw logits.
        torch.manual_seed(0)
        target = (torch.rand(2, 3, 8, 8, 8) > 0.5).float()
        logits = torch.where(target > 0.5, 4.0, -4.0)

        scnp = ScnpLoss(neighborhood_size=3, auto_pos_weight=False)
        bce = PerChannelBCEWithLogitsLoss(auto_pos_weight=False)
        self.assertTrue(
            torch.allclose(scnp(logits, target), bce(logits, target), atol=1e-5)
        )

    def test_penalizes_worst_neighbor(self):
        # A single foreground voxel with a high logit surrounded by foreground
        # voxels with low logits: SCNP replaces the high logit with the low
        # neighbor min, so its loss exceeds plain BCE (the confident voxel is
        # dragged down to its worst neighbor).
        target = torch.ones(1, 1, 3, 3, 3)
        logits = torch.full((1, 1, 3, 3, 3), -2.0)
        logits[0, 0, 1, 1, 1] = 6.0  # one confident-correct fg voxel

        scnp = ScnpLoss(neighborhood_size=3, auto_pos_weight=False)
        bce = PerChannelBCEWithLogitsLoss(auto_pos_weight=False)
        self.assertGreater(scnp(logits, target).item(), bce(logits, target).item())

    def test_gradient_routes_to_worst_neighbor(self):
        target = torch.ones(1, 1, 3, 3, 3)
        logits = torch.full((1, 1, 3, 3, 3), -2.0, requires_grad=True)
        loss = ScnpLoss(neighborhood_size=3, auto_pos_weight=False)(logits, target)
        loss.backward()
        # The worst (lowest) foreground logit governs z_tilde for the whole
        # window, so it must receive non-trivial gradient.
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(logits.grad.abs().sum().item(), 0.0)

    def test_weight_masks_invalid_voxels(self):
        torch.manual_seed(1)
        target = (torch.rand(1, 3, 6, 6, 6) > 0.5).float()
        logits = torch.randn(1, 3, 6, 6, 6)
        weight = torch.ones_like(target)
        weight[..., :2] = 0.0  # mask out a slab

        loss = ScnpLoss(neighborhood_size=3)
        out = loss(logits, target, weight=weight)
        self.assertTrue(torch.isfinite(out))

    def test_2d_supported(self):
        torch.manual_seed(2)
        target = (torch.rand(2, 1, 16, 16) > 0.5).float()
        logits = torch.randn(2, 1, 16, 16)
        loss = ScnpLoss(neighborhood_size=3)(logits, target)
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()

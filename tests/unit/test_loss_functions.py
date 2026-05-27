"""Test loss functions (migrated to MONAI)."""

import unittest

import torch
import torch.nn.functional as F

from connectomics.models.losses import create_loss


class TestLossFunctions(unittest.TestCase):
    """Test MONAI-based loss functions."""

    def test_dice_loss(self):
        """Test Dice loss."""
        loss_fn = create_loss("DiceLoss")
        pred = torch.rand(2, 2, 4, 8, 8)
        target = torch.randint(0, 2, (2, 2, 4, 8, 8)).float()

        loss = loss_fn(pred, target)
        self.assertTrue(loss >= 0.0)

    def test_focal_loss(self):
        """Test Focal loss."""
        loss_fn = create_loss("FocalLoss")
        pred = torch.rand(2, 2, 4, 8, 8)
        target = torch.randint(0, 2, (2, 2, 4, 8, 8)).float()

        loss = loss_fn(pred, target)
        self.assertTrue(loss >= 0.0)

    def test_tversky_loss(self):
        """Test Tversky loss."""
        loss_fn = create_loss("TverskyLoss")
        pred = torch.rand(2, 2, 4, 8, 8)
        target = torch.randint(0, 2, (2, 2, 4, 8, 8)).float()

        loss = loss_fn(pred, target)
        self.assertTrue(loss >= 0.0)

    def test_soft_cldice_binary_probabilities(self):
        """Soft clDice in binary mode should accept probability tensors."""
        loss_fn = create_loss("SoftClDiceLoss", mode="binary", num_iters=3)
        pred = torch.rand(2, 1, 4, 8, 8)
        target = (torch.rand(2, 1, 4, 8, 8) > 0.5).float()

        loss = loss_fn(pred, target)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(loss >= 0.0)

    def test_soft_cldice_multi_excludes_background(self):
        """Soft clDice in multi mode should use foreground classes only."""
        loss_fn = create_loss("SoftClDiceLoss", mode="multi", num_iters=2)

        logits = torch.randn(1, 3, 4, 8, 8)
        pred = torch.softmax(logits, dim=1)
        labels = torch.randint(0, 3, (1, 4, 8, 8))
        target = F.one_hot(labels, num_classes=3).movedim(-1, 1).float()

        loss = loss_fn(pred, target)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(loss >= 0.0)

    def test_soft_cldice_rejects_logits_without_activation(self):
        """Soft clDice should fail fast on logits when no activation is requested."""
        loss_fn = create_loss("SoftClDiceLoss", mode="binary", num_iters=1, validate_inputs=True)
        logits = torch.randn(1, 1, 4, 8, 8)
        target = (torch.rand(1, 1, 4, 8, 8) > 0.5).float()

        with self.assertRaisesRegex(ValueError, "must be probabilities in \\[0, 1\\]"):
            _ = loss_fn(logits, target)

    def test_soft_cldice_accepts_logits_with_sigmoid(self):
        """Soft clDice should support logits when sigmoid=True."""
        loss_fn = create_loss("SoftClDiceLoss", mode="binary", num_iters=2, sigmoid=True)
        logits = torch.randn(1, 1, 4, 8, 8)
        target = (torch.rand(1, 1, 4, 8, 8) > 0.5).float()

        loss = loss_fn(logits, target)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(loss >= 0.0)

    def test_soft_cldice_clamp_probabilities_allows_out_of_range_inputs(self):
        """Clamping should tolerate out-of-range values when validate_inputs is enabled."""
        loss_fn = create_loss(
            "SoftClDiceLoss",
            mode="binary",
            num_iters=1,
            clamp_probabilities=True,
            validate_inputs=True,
        )
        pred = torch.randn(1, 1, 4, 8, 8) * 3.0  # outside [0, 1]
        target = (torch.randn(1, 1, 4, 8, 8) * 2.0) + 0.5

        loss = loss_fn(pred, target)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(loss >= 0.0)

    def test_soft_cldice_backward_produces_finite_gradients(self):
        """Soft clDice should backpropagate finite gradients."""
        loss_fn = create_loss("SoftClDiceLoss", mode="binary", num_iters=2, sigmoid=True)
        logits = torch.randn(2, 1, 4, 8, 8, requires_grad=True)
        target = (torch.rand(2, 1, 4, 8, 8) > 0.5).float()

        loss = loss_fn(logits, target)
        loss.backward()

        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.all(torch.isfinite(logits.grad)))

    def test_soft_cldice_weight_channel_routing(self):
        """Soft clDice should handle weight maps with full channel count in multi mode."""
        loss_fn = create_loss("SoftClDiceLoss", mode="multi", num_iters=1)

        pred = torch.softmax(torch.randn(1, 3, 4, 8, 8), dim=1)
        labels = torch.randint(0, 3, (1, 4, 8, 8))
        target = F.one_hot(labels, num_classes=3).movedim(-1, 1).float()
        weight = torch.rand(1, 3, 4, 8, 8)

        loss = loss_fn(pred, target, weight=weight)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(loss >= 0.0)

    def test_soft_cldice_weight_single_channel_broadcast(self):
        """Soft clDice should broadcast a single-channel weight map."""
        loss_fn = create_loss("SoftClDiceLoss", mode="multi", num_iters=1)

        pred = torch.softmax(torch.randn(1, 3, 4, 8, 8), dim=1)
        labels = torch.randint(0, 3, (1, 4, 8, 8))
        target = F.one_hot(labels, num_classes=3).movedim(-1, 1).float()
        weight = torch.rand(1, 1, 4, 8, 8)

        loss = loss_fn(pred, target, weight=weight)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(loss >= 0.0)


if __name__ == "__main__":
    unittest.main()

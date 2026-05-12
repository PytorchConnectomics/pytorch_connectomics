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


if __name__ == "__main__":
    unittest.main()

"""Test loss functions (migrated to MONAI)."""
import unittest
import torch
import pytest

from connectomics.models.loss import create_loss


class TestLossFunctions(unittest.TestCase):
    """Test MONAI-based loss functions."""

    def test_dice_loss(self):
        """Test Dice loss."""
        loss_fn = create_loss('DiceLoss')
        pred = torch.rand(2, 2, 4, 8, 8)
        target = torch.randint(0, 2, (2, 2, 4, 8, 8)).float()

        loss = loss_fn(pred, target)
        self.assertTrue(loss >= 0.0)

    def test_focal_loss(self):
        """Test Focal loss."""
        loss_fn = create_loss('FocalLoss')
        pred = torch.rand(2, 2, 4, 8, 8)
        target = torch.randint(0, 2, (2, 2, 4, 8, 8)).float()

        loss = loss_fn(pred, target)
        self.assertTrue(loss >= 0.0)

    def test_tversky_loss(self):
        """Test Tversky loss."""
        loss_fn = create_loss('TverskyLoss')
        pred = torch.rand(2, 2, 4, 8, 8)
        target = torch.randint(0, 2, (2, 2, 4, 8, 8)).float()

        loss = loss_fn(pred, target)
        self.assertTrue(loss >= 0.0)


if __name__ == '__main__':
    unittest.main()

"""
Connectomics-specific loss functions.

Custom losses that provide functionality not available in MONAI.
Only includes losses that are truly unique to connectomics use cases.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedMSELoss(nn.Module):
    """
    Weighted mean-squared error loss.

    Useful for regression tasks with spatial importance weighting.

    Args:
        reduction: Reduction method ('mean', 'sum', 'none')
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute weighted MSE loss.

        Args:
            pred: Predictions
            target: Ground truth
            weight: Optional spatial weights

        Returns:
            Loss value
        """
        mse = (pred - target) ** 2

        if weight is not None:
            mse = mse * weight

        if self.reduction == 'mean':
            return mse.mean()
        elif self.reduction == 'sum':
            return mse.sum()
        else:
            return mse


class WeightedMAELoss(nn.Module):
    """
    Weighted mean absolute error loss.

    Useful for regression tasks with spatial importance weighting.

    Args:
        reduction: Reduction method ('mean', 'sum', 'none')
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute weighted MAE loss.

        Args:
            pred: Predictions
            target: Ground truth
            weight: Optional spatial weights

        Returns:
            Loss value
        """
        mae = torch.abs(pred - target)

        if weight is not None:
            mae = mae * weight

        if self.reduction == 'mean':
            return mae.mean()
        elif self.reduction == 'sum':
            return mae.sum()
        else:
            return mae


class GANLoss(nn.Module):
    """
    GAN loss for adversarial training.

    Supports vanilla, LSGAN, and WGAN-GP objectives.
    Based on https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

    Args:
        gan_mode: GAN objective type ('vanilla', 'lsgan', 'wgangp')
        target_real_label: Label for real images
        target_fake_label: Label for fake images

    Note:
        Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. Vanilla GANs will handle it with BCEWithLogitsLoss.
    """

    def __init__(
        self,
        gan_mode: str = 'lsgan',
        target_real_label: float = 1.0,
        target_fake_label: float = 0.0,
    ):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode

        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError(f'GAN mode {gan_mode} not implemented')

    def get_target_tensor(
        self,
        prediction: torch.Tensor,
        target_is_real: bool,
    ) -> torch.Tensor:
        """
        Create label tensors with the same size as the input.

        Args:
            prediction: Discriminator prediction
            target_is_real: Whether the ground truth is real or fake

        Returns:
            Label tensor filled with ground truth labels
        """
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label

        return target_tensor.expand_as(prediction)

    def forward(
        self,
        prediction: torch.Tensor,
        target_is_real: bool,
    ) -> torch.Tensor:
        """
        Calculate GAN loss.

        Args:
            prediction: Discriminator output
            target_is_real: Whether ground truth labels are for real or fake images

        Returns:
            Calculated loss
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()

        return loss


__all__ = [
    'WeightedMSELoss',
    'WeightedMAELoss',
    'GANLoss',
]
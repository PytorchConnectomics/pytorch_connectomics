"""
Connectomics-specific loss functions.

Custom losses that provide functionality not available in MONAI.
Only includes losses that are truly unique to connectomics use cases.
"""

from __future__ import annotations

from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def _reduce_weighted_tensor(
    loss_tensor: torch.Tensor,
    weight: torch.Tensor | None,
    reduction: str,
) -> torch.Tensor:
    """Reduce weighted loss values while excluding invalid (weight<=0) voxels for mean."""
    if reduction == "none":
        return loss_tensor

    if reduction == "sum":
        return loss_tensor.sum()

    # reduction == "mean"
    if weight is None:
        return loss_tensor.mean()

    valid = weight > 0
    if not torch.any(valid):
        return loss_tensor.new_tensor(0.0)
    return loss_tensor[valid].mean()


class CrossEntropyLossWrapper(nn.Module):
    """
    Wrapper for CrossEntropyLoss that handles shape conversion.

    Expects labels in format [B, 1, D, H, W] and converts to [B, D, H, W]
    for compatibility with PyTorch's CrossEntropyLoss.

    Args:
        weight: Class weights
        ignore_index: Index to ignore
        reduction: Reduction method
        label_smoothing: Label smoothing factor (0.0 = no smoothing, 0.1 = 10% smoothing)
    """

    def __init__(self, weight=None, ignore_index=-100, reduction="mean", label_smoothing=0.0):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_index,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute cross-entropy loss with automatic shape handling.

        Args:
            input: Model output
                   2D: [B, C, H, W]
                   3D: [B, C, D, H, W]
            target: Ground truth
                    2D: [B, 1, H, W] or [B, H, W]
                    3D: [B, 1, D, H, W] or [B, D, H, W]

        Returns:
            Loss value
        """
        # Squeeze channel dimension if present
        # 2D: [B, 1, H, W] -> [B, H, W]
        # 3D: [B, 1, D, H, W] -> [B, D, H, W]
        if target.dim() == 4 and target.shape[1] == 1:
            target = target.squeeze(1)  # 2D case
        elif target.dim() == 5 and target.shape[1] == 1:
            target = target.squeeze(1)  # 3D case

        # Convert to long type for cross-entropy
        target = target.long()

        return self.ce_loss(input, target)


class WeightedMSELoss(nn.Module):
    """
    Weighted mean-squared error loss.

    Useful for regression tasks with spatial importance weighting.
    Supports optional tanh activation for distance transform predictions.

    Args:
        reduction: Reduction method ('mean', 'sum', 'none')
        tanh: If True, apply tanh activation to predictions before computing loss.
              Useful for distance transform targets in range [-1, 1].
              With both pred and target in [-1, 1], MSE should be < 4.
    """

    def __init__(self, reduction: str = "mean", tanh: bool = False):
        super().__init__()
        self.reduction = reduction
        self.tanh = tanh

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute weighted MSE loss.

        Args:
            pred: Predictions (logits if tanh=True, otherwise predictions)
            target: Ground truth (range [-1, 1] for SDT)
            weight: Optional spatial weights

        Returns:
            Loss value (should be < 4 for range [-1, 1])
        """
        # Apply tanh activation if enabled (constrains pred to [-1, 1])
        if self.tanh:
            pred = torch.tanh(pred)

        # Compute MSE (for range [-1,1], max error is (1-(-1))^2 = 4)
        mse = (pred - target) ** 2

        if weight is not None:
            mse = mse * weight

        loss_value = _reduce_weighted_tensor(mse, weight, self.reduction)
        return loss_value


class WeightedBCEWithLogitsLoss(nn.Module):
    """
    Wrapper for BCEWithLogitsLoss with optional class and spatial weighting.

    Supports static ``pos_weight`` (configured once) and optional per-call
    ``pos_weight`` override (used by auto class-ratio mode in orchestration).

    Args:
        pos_weight: Optional positive-class weight (scalar or tensor)
        reduction: Reduction method ('mean', 'sum', 'none')
    """

    def __init__(
        self,
        pos_weight: Union[float, torch.Tensor, None] = None,
        reduction: str = "mean",
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            unexpected = ", ".join(sorted(kwargs.keys()))
            raise TypeError(f"Unexpected argument(s) for WeightedBCEWithLogitsLoss: {unexpected}")
        self.reduction = reduction
        if pos_weight is not None and isinstance(pos_weight, (int, float)):
            if float(pos_weight) <= 0:
                raise ValueError(f"pos_weight must be > 0, got {float(pos_weight)}")
            self.register_buffer("pos_weight", torch.tensor([float(pos_weight)]))
        elif pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor = None,
        pos_weight: Union[float, torch.Tensor, None] = None,
    ) -> torch.Tensor:
        """
        Compute weighted BCE with logits loss.

        Args:
            input: Model output (logits) [B, C, ...]
            target: Ground truth [B, C, ...]
            weight: Optional spatial weights/mask.
            pos_weight: Optional per-call positive-class weight override.

        Returns:
            Loss value
        """
        if pos_weight is not None and isinstance(pos_weight, (int, float)):
            if float(pos_weight) <= 0:
                raise ValueError(f"pos_weight must be > 0, got {float(pos_weight)}")
            effective_pos_weight = torch.tensor(
                [float(pos_weight)],
                device=input.device,
                dtype=input.dtype,
            )
        elif pos_weight is not None:
            effective_pos_weight = pos_weight.to(device=input.device, dtype=input.dtype)
        elif self.pos_weight is not None:
            effective_pos_weight = self.pos_weight.to(device=input.device, dtype=input.dtype)
        else:
            effective_pos_weight = None

        bce = F.binary_cross_entropy_with_logits(
            input,
            target,
            pos_weight=effective_pos_weight,
            reduction="none",
        )

        if weight is not None:
            bce = bce * weight

        return _reduce_weighted_tensor(bce, weight, self.reduction)


class PerChannelBCEWithLogitsLoss(nn.Module):
    """BCE loss computed independently per channel with per-channel class balancing.

    Equivalent to summing C separate WeightedBCEWithLogitsLoss instances (one
    per output channel) but expressed as a single loss entry for config brevity.
    Matches the DeepEM per-edge loss structure where each affinity channel has
    its own class-balanced BCE.

    Args:
        auto_pos_weight: If True, compute per-channel pos_weight as
            min(n_neg / n_pos, max_pos_weight) from the current batch.
        max_pos_weight: Cap for auto pos_weight to avoid extreme values.
        reduction: Per-channel reduction before summing ('mean' or 'sum').
    """

    def __init__(
        self,
        auto_pos_weight: bool = True,
        max_pos_weight: float = 10.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.auto_pos_weight = auto_pos_weight
        self.max_pos_weight = max_pos_weight
        self.reduction = reduction

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute per-channel BCE loss.

        Args:
            input: Logits [B, C, ...].
            target: Ground truth [B, C, ...].
            weight: Optional spatial mask [B, C, ...] (e.g. affinity valid mask).

        Returns:
            Sum of per-channel reduced BCE losses.
        """
        C = input.shape[1]

        # Vectorised per-channel pos_weight computation.
        pos_weight_tensor = None
        if self.auto_pos_weight:
            valid = weight > 0 if weight is not None else torch.ones_like(target, dtype=torch.bool)
            pos = (target > 0) & valid
            neg = (target <= 0) & valid

            # Reduce over batch + spatial dims → (C,)
            reduce_dims = (0,) + tuple(range(2, target.ndim))
            pos_counts = pos.sum(dim=reduce_dims).float()
            neg_counts = neg.sum(dim=reduce_dims).float()

            pw = torch.ones(C, device=input.device, dtype=torch.float32)
            nonzero = pos_counts > 0
            pw[nonzero] = torch.clamp(
                neg_counts[nonzero] / pos_counts[nonzero],
                max=self.max_pos_weight,
            )
            # (1, C, 1, ..., 1) for broadcasting; cast to input dtype for AMP
            pos_weight_tensor = pw.reshape(1, C, *([1] * (target.ndim - 2))).to(input.dtype)

        bce = F.binary_cross_entropy_with_logits(
            input,
            target,
            pos_weight=pos_weight_tensor,
            reduction="none",
        )

        if weight is not None:
            bce = bce * weight

        # Per-channel reduction, then sum across channels.
        total = input.new_tensor(0.0)
        for c in range(C):
            ch_bce = bce[:, c : c + 1, ...]
            ch_w = weight[:, c : c + 1, ...] if weight is not None else None
            total = total + _reduce_weighted_tensor(ch_bce, ch_w, self.reduction)

        return total


class WeightedMAELoss(nn.Module):
    """
    Weighted mean absolute error loss.

    Useful for regression tasks with spatial importance weighting.
    Supports optional tanh activation for distance transform predictions.

    Args:
        reduction: Reduction method ('mean', 'sum', 'none')
        tanh: If True, apply tanh activation to predictions before computing loss.
              Useful for distance transform targets in range [-1, 1].
    """

    def __init__(self, reduction: str = "mean", tanh: bool = False):
        super().__init__()
        self.reduction = reduction
        self.tanh = tanh

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute weighted MAE loss.

        Args:
            pred: Predictions (logits if tanh=True, otherwise predictions)
            target: Ground truth
            weight: Optional spatial weights

        Returns:
            Loss value
        """
        # Apply tanh activation if enabled
        if self.tanh:
            pred = torch.tanh(pred)

        mae = torch.abs(pred - target)

        if weight is not None:
            mae = mae * weight

        return _reduce_weighted_tensor(mae, weight, self.reduction)


class SmoothL1Loss(nn.Module):
    """
    Smooth L1 (Huber) loss with optional tanh activation and spatial weighting.

    Useful for distance transform regression where large outliers should be
    down-weighted relative to MSE.
    """

    def __init__(self, beta: float = 1.0, reduction: str = "mean", tanh: bool = False):
        super().__init__()
        self.beta = beta
        self.reduction = reduction
        self.tanh = tanh

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor = None,
    ) -> torch.Tensor:
        if self.tanh:
            pred = torch.tanh(pred)

        loss = F.smooth_l1_loss(pred, target, beta=self.beta, reduction="none")

        if weight is not None:
            loss = loss * weight

        return _reduce_weighted_tensor(loss, weight, self.reduction)


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
        gan_mode: str = "lsgan",
        target_real_label: float = 1.0,
        target_fake_label: float = 0.0,
    ):
        super().__init__()
        self.register_buffer("real_label", torch.tensor(target_real_label))
        self.register_buffer("fake_label", torch.tensor(target_fake_label))
        self.gan_mode = gan_mode

        if gan_mode == "lsgan":
            self.loss = nn.MSELoss()
        elif gan_mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ["wgangp"]:
            self.loss = None
        else:
            raise NotImplementedError(f"GAN mode {gan_mode} not implemented")

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
        if self.gan_mode in ["lsgan", "vanilla"]:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == "wgangp":
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()

        return loss


__all__ = [
    "CrossEntropyLossWrapper",
    "WeightedBCEWithLogitsLoss",
    "PerChannelBCEWithLogitsLoss",
    "WeightedMSELoss",
    "WeightedMAELoss",
    "SmoothL1Loss",
    "GANLoss",
]

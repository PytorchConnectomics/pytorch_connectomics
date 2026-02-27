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
        from ...utils.debug_utils import DEBUG_NORM, print_tensor_stats

        # DEBUG: Print input to loss function (before tanh)
        if DEBUG_NORM and not hasattr(self, "_debug_loss_printed"):
            self._debug_loss_printed = True
            print_tensor_stats(
                pred,
                stage_name="STAGE 7: LOSS FUNCTION INPUT (before tanh)",
                tensor_name="pred",
                print_once=False,
                extra_info={
                    "tanh_enabled": self.tanh,
                    "reduction": self.reduction,
                    "weight_provided": weight is not None,
                },
            )
            print_tensor_stats(
                target,
                stage_name="STAGE 7: LOSS FUNCTION INPUT (target)",
                tensor_name="target",
                print_once=False,
                extra_info={"expected_range": "[-1, 1] for SDT"},
            )

        # Apply tanh activation if enabled (constrains pred to [-1, 1])
        if self.tanh:
            pred = torch.tanh(pred)

            # DEBUG: Print after tanh activation
            if DEBUG_NORM and not hasattr(self, "_debug_tanh_printed"):
                self._debug_tanh_printed = True
                print_tensor_stats(
                    pred,
                    stage_name="STAGE 6: PREDICTION AFTER TANH",
                    tensor_name="pred_after_tanh",
                    print_once=False,
                    extra_info={
                        "activation_applied": "tanh",
                        "expected_range": "[-1, 1]",
                        "note": "Should now match target range",
                    },
                )

        # Compute MSE (for range [-1,1], max error is (1-(-1))^2 = 4)
        mse = (pred - target) ** 2

        if weight is not None:
            mse = mse * weight

        loss_value = _reduce_weighted_tensor(mse, weight, self.reduction)

        # DEBUG: Print loss output
        if DEBUG_NORM and not hasattr(self, "_debug_loss_output_printed"):
            self._debug_loss_output_printed = True
            print(f"\n{'=' * 80}")
            print("[DEBUG NORM] STAGE 8: LOSS FUNCTION OUTPUT")
            print(f"{'=' * 80}")
            print(f"LOSS VALUE: {loss_value.item():.6f}")
            print("  Expected range: [0, 4] for MSE with values in [-1, 1]")
            if loss_value.item() > 4:
                print("  ⚠️  WARNING: Loss > 4 suggests tanh might not be working!")
            else:
                print("  ✅ Loss is reasonable")
            print(f"{'=' * 80}\n")

        return loss_value


class WeightedBCEWithLogitsLoss(nn.Module):
    """
    Wrapper for BCEWithLogitsLoss with support for pos_weight (class rebalancing).

    Useful for handling class imbalance in binary segmentation, especially
    for affinity prediction in connectomics where foreground is often sparse.

    Args:
        pos_weight: Weight for positive class (tensor or float)
                   Set to neg_samples/pos_samples for balanced loss
                   If None, no rebalancing is applied
        reduction: Reduction method ('mean', 'sum', 'none')

    Examples:
        # Auto-calculate from data
        pos_weight = (target == 0).sum() / (target == 1).sum()
        loss_fn = WeightedBCEWithLogitsLoss(pos_weight=pos_weight)

        # Manual weight (e.g., 10x weight for positive class)
        loss_fn = WeightedBCEWithLogitsLoss(pos_weight=10.0)
    """

    def __init__(
        self, pos_weight: Union[float, torch.Tensor, None] = None, reduction: str = "mean"
    ):
        super().__init__()
        self.reduction = reduction

        # Convert float to tensor if needed
        if pos_weight is not None and isinstance(pos_weight, (int, float)):
            self.register_buffer("pos_weight", torch.tensor([pos_weight]))
        elif pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute weighted BCE with logits loss.

        Args:
            input: Model output (logits) [B, C, ...]
            target: Ground truth [B, C, ...]
            weight: Optional spatial weights/mask.

        Returns:
            Loss value
        """
        bce = F.binary_cross_entropy_with_logits(
            input,
            target,
            pos_weight=self.pos_weight,
            reduction="none",
        )

        if weight is not None:
            bce = bce * weight

        return _reduce_weighted_tensor(bce, weight, self.reduction)


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
    "WeightedMSELoss",
    "WeightedMAELoss",
    "GANLoss",
]

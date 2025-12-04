"""
Regularization losses for connectomics.

These losses encourage specific properties in the predictions, such as:
- Binary outputs
- Consistency between related prediction tasks
- Non-overlapping predictions

All losses are implemented as nn.Module for consistency with MONAI.
"""

from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryRegularization(nn.Module):
    """
    Regularization encouraging outputs to be binary (close to 0 or 1).

    Penalizes predictions that are close to 0.5 (uncertain).

    Args:
        min_threshold: Minimum threshold for clamping (default: 1e-2)

    Example:
        >>> reg = BinaryRegularization()
        >>> pred = torch.sigmoid(torch.randn(1, 1, 64, 64, 64))
        >>> loss = reg(pred)
    """

    def __init__(self, min_threshold: float = 1e-2):
        super().__init__()
        self.min_threshold = min_threshold

    def forward(
        self,
        pred: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute binary regularization loss.

        Args:
            pred: Predictions (logits or probabilities)
            mask: Optional spatial weight mask

        Returns:
            Regularization loss
        """
        # Convert logits to probabilities if needed
        if pred.min() < 0 or pred.max() > 1:
            pred = torch.sigmoid(pred)

        # Distance from 0.5 (most uncertain)
        diff = torch.abs(pred - 0.5)
        diff = torch.clamp(diff, min=self.min_threshold)

        # Penalize being close to 0.5
        loss = 1.0 / diff

        if mask is not None:
            loss = loss * mask

        return loss.mean()


class ForegroundDistanceConsistency(nn.Module):
    """
    Consistency regularization between binary foreground mask and signed distance transform.

    Encourages foreground predictions to be consistent with distance transform predictions.

    Example:
        >>> reg = ForegroundDistanceConsistency()
        >>> fg_logits = torch.randn(1, 1, 64, 64, 64)
        >>> dt_pred = torch.randn(1, 1, 64, 64, 64)
        >>> loss = reg(fg_logits, dt_pred)
    """

    def forward(
        self,
        foreground_logits: torch.Tensor,
        distance_transform: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute consistency loss between foreground and distance transform.

        Args:
            foreground_logits: Binary foreground logits
            distance_transform: Signed distance transform predictions
            mask: Optional spatial weight mask

        Returns:
            Consistency loss
        """
        # Log probabilities for numerical stability
        log_prob_pos = F.logsigmoid(foreground_logits)
        log_prob_neg = F.logsigmoid(-foreground_logits)

        # Distance transform (normalized with tanh)
        distance = torch.tanh(distance_transform)
        dist_pos = torch.clamp(distance, min=0.0)  # Positive distances (inside)
        dist_neg = -torch.clamp(distance, max=0.0)  # Negative distances (outside)

        # Consistency: high positive prob should match positive distances
        loss_pos = -log_prob_pos * dist_pos
        loss_neg = -log_prob_neg * dist_neg
        loss = loss_pos + loss_neg

        if mask is not None:
            loss = loss * mask

        return loss.mean()


class ContourDistanceConsistency(nn.Module):
    """
    Consistency regularization between instance contour map and signed distance transform.

    Encourages contour predictions (high at boundaries) to be consistent with
    distance transform predictions (low magnitude at boundaries).

    Example:
        >>> reg = ContourDistanceConsistency()
        >>> contour_logits = torch.randn(1, 1, 64, 64, 64)
        >>> dt_pred = torch.randn(1, 1, 64, 64, 64)
        >>> loss = reg(contour_logits, dt_pred)
    """

    def forward(
        self,
        contour_logits: torch.Tensor,
        distance_transform: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute consistency loss between contour and distance transform.

        Args:
            contour_logits: Instance contour logits
            distance_transform: Signed distance transform predictions
            mask: Optional spatial weight mask

        Returns:
            Consistency loss
        """
        contour_prob = torch.sigmoid(contour_logits)
        distance_abs = torch.abs(torch.tanh(distance_transform))

        assert contour_prob.shape == distance_abs.shape, (
            f"Shape mismatch: {contour_prob.shape} vs {distance_abs.shape}"
        )

        # Penalize: high contour prob should match low distance
        loss = contour_prob * distance_abs
        loss = loss**2

        if mask is not None:
            loss = loss * mask

        return loss.mean()


class ForegroundContourConsistency(nn.Module):
    """
    Consistency regularization between binary foreground and instance contour maps.

    Encourages contour predictions to align with foreground edges detected via Sobel filters.

    Args:
        kernel_half_size: Half-size of edge detection kernel (default: 1)
        eps: Small epsilon for numerical stability (default: 1e-7)

    Example:
        >>> reg = ForegroundContourConsistency()
        >>> fg_logits = torch.randn(1, 1, 64, 64, 64)
        >>> contour_logits = torch.randn(1, 1, 64, 64, 64)
        >>> loss = reg(fg_logits, contour_logits)
    """

    def __init__(self, kernel_half_size: int = 1, eps: float = 1e-7):
        super().__init__()
        self.kernel_size = 2 * kernel_half_size + 1
        self.eps = eps

        # Sobel filters for edge detection
        sobel = torch.tensor([1.0, 0.0, -1.0])
        self.register_buffer("sobel_x", sobel.view(1, 1, 1, 1, 3))
        self.register_buffer("sobel_y", sobel.view(1, 1, 1, 3, 1))

    def forward(
        self,
        foreground_logits: torch.Tensor,
        contour_logits: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute consistency loss between foreground edges and contours.

        Args:
            foreground_logits: Binary foreground logits
            contour_logits: Instance contour logits
            mask: Optional spatial weight mask

        Returns:
            Consistency loss
        """
        fg_prob = torch.sigmoid(foreground_logits)
        contour_prob = torch.sigmoid(contour_logits)

        # Detect edges in foreground using Sobel filters
        edge_x = F.conv3d(fg_prob, self.sobel_x, padding=(0, 0, 1))
        edge_y = F.conv3d(fg_prob, self.sobel_y, padding=(0, 1, 0))

        # Compute edge magnitude
        edge = torch.sqrt(edge_x**2 + edge_y**2 + self.eps)
        edge = torch.clamp(edge, min=self.eps, max=1.0 - self.eps)

        # Max pooling to expand edge regions
        edge = F.pad(edge, (1, 1, 1, 1, 0, 0))
        edge = F.max_pool3d(edge, kernel_size=(1, self.kernel_size, self.kernel_size), stride=1)

        assert edge.shape == contour_prob.shape, (
            f"Shape mismatch: {edge.shape} vs {contour_prob.shape}"
        )

        # MSE between detected edges and predicted contours
        loss = F.mse_loss(edge, contour_prob, reduction="none")

        if mask is not None:
            loss = loss * mask

        return loss.mean()


class NonOverlapRegularization(nn.Module):
    """
    Regularization preventing overlapping predictions.

    Specifically designed for synaptic polarity prediction where pre- and post-synaptic
    masks should not overlap. Optionally masks the regularization by the cleft prediction.

    Args:
        cleft_masked: Whether to mask regularization by cleft prediction (default: True)

    Example:
        >>> reg = NonOverlapRegularization()
        >>> # pred has shape (B, 3, Z, Y, X) with channels: [pre, post, cleft]
        >>> pred = torch.randn(2, 3, 32, 64, 64)
        >>> loss = reg(pred)
    """

    def __init__(self, cleft_masked: bool = True):
        super().__init__()
        self.cleft_masked = cleft_masked

    def forward(self, pred: torch.Tensor) -> torch.Tensor:
        """
        Compute non-overlap regularization loss.

        Args:
            pred: Predictions with shape (B, C, Z, Y, X) where:
                  - Channel 0: Pre-synaptic logits
                  - Channel 1: Post-synaptic logits
                  - Channel 2: Cleft logits (optional, used if cleft_masked=True)

        Returns:
            Non-overlap regularization loss
        """
        if pred.shape[1] < 2:
            raise ValueError(
                f"Expected at least 2 channels for pre/post predictions, got {pred.shape[1]}"
            )

        # Pre- and post-synaptic probabilities
        pre_prob = torch.sigmoid(pred[:, 0])
        post_prob = torch.sigmoid(pred[:, 1])

        # Penalize overlap
        loss = pre_prob * post_prob

        if self.cleft_masked and pred.shape[1] >= 3:
            # Mask by cleft prediction (detached to avoid decreasing cleft prob)
            cleft_prob = torch.sigmoid(pred[:, 2].detach())
            loss = loss * cleft_prob

        return loss.mean()


__all__ = [
    "BinaryRegularization",
    "ForegroundDistanceConsistency",
    "ContourDistanceConsistency",
    "ForegroundContourConsistency",
    "NonOverlapRegularization",
]

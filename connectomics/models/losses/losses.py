"""
Connectomics-specific loss functions.

Custom losses that provide functionality not available in MONAI.
Only includes losses that are truly unique to connectomics use cases.
"""

from __future__ import annotations

from typing import List, Union

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
    if valid.shape != loss_tensor.shape:
        try:
            valid = torch.broadcast_to(valid, loss_tensor.shape)
        except RuntimeError as e:
            raise ValueError(
                "Weight mask shape is not broadcastable to loss tensor shape: "
                f"weight={tuple(weight.shape)}, loss={tuple(loss_tensor.shape)}"
            ) from e
    if not torch.any(valid):
        return loss_tensor.new_tensor(0.0)
    return loss_tensor[valid].mean()


def _soft_erode_pool(prob: torch.Tensor) -> torch.Tensor:
    """Differentiable morphological erosion using min-pool via max-pool."""
    if prob.ndim == 5:
        # Use an axis-aligned cross (3x1x1, 1x3x1, 1x1x3), matching clDice-style soft morphology.
        p1 = -F.max_pool3d(-prob, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0))
        p2 = -F.max_pool3d(-prob, kernel_size=(1, 3, 1), stride=1, padding=(0, 1, 0))
        p3 = -F.max_pool3d(-prob, kernel_size=(1, 1, 3), stride=1, padding=(0, 0, 1))
        return torch.minimum(torch.minimum(p1, p2), p3)
    if prob.ndim == 4:
        p1 = -F.max_pool2d(-prob, kernel_size=(3, 1), stride=1, padding=(1, 0))
        p2 = -F.max_pool2d(-prob, kernel_size=(1, 3), stride=1, padding=(0, 1))
        return torch.minimum(p1, p2)
    raise ValueError(f"Expected 4D/5D tensor for soft erosion, got shape {tuple(prob.shape)}")


def _soft_dilate_pool(prob: torch.Tensor) -> torch.Tensor:
    """Differentiable morphological dilation."""
    if prob.ndim == 5:
        return F.max_pool3d(prob, kernel_size=3, stride=1, padding=1)
    if prob.ndim == 4:
        return F.max_pool2d(prob, kernel_size=3, stride=1, padding=1)
    raise ValueError(f"Expected 4D/5D tensor for soft dilation, got shape {tuple(prob.shape)}")


def _soft_open_pool(prob: torch.Tensor) -> torch.Tensor:
    """Differentiable opening (erode followed by dilate)."""
    return _soft_dilate_pool(_soft_erode_pool(prob))


def _soft_skeletonize_pool(prob: torch.Tensor, num_iters: int) -> torch.Tensor:
    """Iterative soft skeletonization from clDice-style morphology."""
    opened = _soft_open_pool(prob)
    skeleton = F.relu(prob - opened)
    for _ in range(num_iters):
        prob = _soft_erode_pool(prob)
        opened = _soft_open_pool(prob)
        delta = F.relu(prob - opened)
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


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


class ScnpLoss(nn.Module):
    """Same-Class Neighbor Penalization (SCNP) loss for binary/affinity channels.

    Implements the logit modification from Lipman et al., "Towards High-Quality
    Image Segmentation: Improving Topology Accuracy by Penalizing Neighbor
    Pixels" (CVPR 2026, https://github.com/jmlipman/SCNP-SameClassNeighborPenalization).
    Each channel is treated as an independent binary problem, matching PyTC's
    canonical affinity layout. Before scoring, the per-voxel logit is replaced by
    its worst same-class neighbor inside an ``N^3`` window:

    - foreground voxels (target==1): the *minimum* logit over foreground
      neighbors (via negated max-pool), so BCE pushes even the least-confident
      foreground neighbor up;
    - background voxels (target==0): the *maximum* logit over background
      neighbors (via max-pool), so BCE pushes the most-confident false positive
      down.

    The modified logits ``z_tilde`` are then scored with the same per-channel
    class-balanced BCE as :class:`PerChannelBCEWithLogitsLoss`. This discourages
    topological errors (isolated false-positive clusters, holes/breaks) without
    a dedicated topology metric. Max-pooling is differentiable, so the gradient
    routes to the worst same-class neighbor.

    Note: the pooling treats every target==0 voxel as background. Within
    ``neighborhood_size // 2`` voxels of a masked-out (invalid) region a valid
    background voxel may pool over an invalid neighbor's logit; this boundary
    effect is bounded by the kernel radius and the invalid centers themselves are
    excluded from the loss via ``weight``.

    Args:
        neighborhood_size: Side length ``N`` of the cubic max-pool window
            (odd, stride 1, ``N//2`` padding so spatial size is preserved). The
            paper notes the optimal size tracks structure size.
        auto_pos_weight: Per-channel class balancing, as in
            :class:`PerChannelBCEWithLogitsLoss`.
        max_pos_weight: Cap for the auto pos_weight.
        reduction: Per-channel reduction before summing ('mean' or 'sum').
    """

    _LARGE = 9999.0

    def __init__(
        self,
        neighborhood_size: int = 3,
        auto_pos_weight: bool = True,
        max_pos_weight: float = 10.0,
        reduction: str = "mean",
    ):
        super().__init__()
        if neighborhood_size < 1 or neighborhood_size % 2 == 0:
            raise ValueError(
                f"neighborhood_size must be a positive odd int, got {neighborhood_size}."
            )
        self.neighborhood_size = int(neighborhood_size)
        self.bce = PerChannelBCEWithLogitsLoss(
            auto_pos_weight=auto_pos_weight,
            max_pos_weight=max_pos_weight,
            reduction=reduction,
        )

    def _scnp_logits(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ns = self.neighborhood_size
        if logits.ndim == 5:
            mp = F.max_pool3d
            kernel = (ns, ns, ns)
            pad = (ns // 2, ns // 2, ns // 2)
        elif logits.ndim == 4:
            mp = F.max_pool2d
            kernel = (ns, ns)
            pad = (ns // 2, ns // 2)
        else:
            raise ValueError(
                f"ScnpLoss expects 4D [B,C,H,W] or 5D [B,C,Z,Y,X] logits, got {logits.ndim}D."
            )

        # Binary same-class gate; affinity targets are 0/1 but tolerate soft GT.
        fg = (target > 0.5).to(logits.dtype)
        bg = 1.0 - fg
        large = self._LARGE
        # Min logit over foreground neighbors (negate to reuse max-pool).
        t1 = -mp(-(logits * fg + large * bg), kernel, 1, pad)
        # Max logit over background neighbors.
        t2 = mp(logits * bg - large * fg, kernel, 1, pad)
        return t1 * fg + t2 * bg

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute SCNP-penalized per-channel BCE.

        Args:
            input: Logits [B, C, ...].
            target: Binary ground truth [B, C, ...].
            weight: Optional spatial mask [B, C, ...] (e.g. affinity valid mask).
        """
        z_tilde = self._scnp_logits(input, target)
        return self.bce(z_tilde, target, weight=weight)


class SoftClDiceLoss(nn.Module):
    """
    Soft clDice loss using differentiable skeletonization.

    Supports optional activation on logits (`sigmoid=True` or `softmax=True`),
    MONAI-style, before topology computation.
    Targets must be dense maps (one-hot or soft labels); single-channel class-index
    targets are accepted and converted to one-hot for multi-class predictions.

    Args:
        num_iters: Number of soft-skeleton erosion iterations.
        mode: ``"binary"`` (single foreground channel) or ``"multi"`` (all classes except
            ``background_index``).
        reduction: ``"none"``, ``"mean"``, or ``"sum"``.
        smooth: Smoothing constant in topology precision/sensitivity fractions.
        foreground_channel: Foreground channel index used in ``mode="binary"`` when C>1.
        background_index: Background channel index excluded in ``mode="multi"``.
        sigmoid: Apply sigmoid activation to predictions inside ``forward``.
        softmax: Apply softmax activation to predictions inside ``forward``.
        clamp_probabilities: Clamp predictions/targets to ``[0, 1]`` before skeletonization.
        validate_inputs: When True, enforce probability-range and minimal spatial-size checks.
        validation_tolerance: Tolerance used by probability-range checks.
    """

    def __init__(
        self,
        num_iters: int = 5,
        mode: str = "binary",
        reduction: str = "mean",
        smooth: float = 1.0,
        foreground_channel: int = 1,
        background_index: int = 0,
        sigmoid: bool = False,
        softmax: bool = False,
        clamp_probabilities: bool = False,
        validate_inputs: bool = True,
        validation_tolerance: float = 1e-5,
    ):
        super().__init__()
        if num_iters < 0:
            raise ValueError(f"num_iters must be >= 0, got {num_iters}")
        if mode not in {"binary", "multi"}:
            raise ValueError(f"mode must be 'binary' or 'multi', got {mode!r}")
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(f"reduction must be 'none', 'mean', or 'sum', got {reduction!r}")
        if smooth <= 0:
            raise ValueError(f"smooth must be > 0, got {smooth}")
        if sigmoid and softmax:
            raise ValueError("sigmoid and softmax are mutually exclusive")
        if validation_tolerance < 0:
            raise ValueError(f"validation_tolerance must be >= 0, got {validation_tolerance}")

        self.num_iters = int(num_iters)
        self.mode = mode
        self.reduction = reduction
        self.smooth = float(smooth)
        self.foreground_channel = int(foreground_channel)
        self.background_index = int(background_index)
        self.sigmoid = bool(sigmoid)
        self.softmax = bool(softmax)
        self.clamp_probabilities = bool(clamp_probabilities)
        self.validate_inputs = bool(validate_inputs)
        self.validation_tolerance = float(validation_tolerance)

    def _prepare_target(self, target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        if target.ndim == pred.ndim - 1:
            target = target.unsqueeze(1)
        if target.ndim != pred.ndim:
            raise ValueError(
                f"Target ndim ({target.ndim}) does not match prediction ndim ({pred.ndim})"
            )
        if target.shape[0] != pred.shape[0] or target.shape[2:] != pred.shape[2:]:
            raise ValueError(
                "Target shape must match prediction shape except for channel dimension: "
                f"target={tuple(target.shape)}, pred={tuple(pred.shape)}"
            )

        if target.shape[1] == pred.shape[1]:
            return target.to(device=pred.device, dtype=pred.dtype)

        if target.shape[1] == 1 and pred.shape[1] > 1:
            class_index = target.squeeze(1).long()
            min_label = int(class_index.min().item())
            max_label = int(class_index.max().item())
            if min_label < 0 or max_label >= pred.shape[1]:
                raise ValueError(
                    f"Class-index targets must be in [0, {pred.shape[1] - 1}], "
                    f"got min={min_label}, max={max_label}"
                )
            one_hot = F.one_hot(class_index, num_classes=pred.shape[1]).movedim(-1, 1)
            return one_hot.to(device=pred.device, dtype=pred.dtype)

        raise ValueError(
            "Target channel count is incompatible with prediction: "
            f"target_channels={target.shape[1]}, pred_channels={pred.shape[1]}"
        )

    def _apply_activation(self, pred: torch.Tensor) -> torch.Tensor:
        if self.sigmoid:
            return torch.sigmoid(pred)
        if self.softmax:
            if pred.shape[1] < 2:
                raise ValueError("softmax=True requires prediction with at least 2 channels")
            return F.softmax(pred, dim=1)
        return pred

    def _validate_probability_range(self, tensor: torch.Tensor, name: str) -> None:
        if not self.validate_inputs:
            return
        tol = self.validation_tolerance
        min_val = float(tensor.min().item())
        max_val = float(tensor.max().item())
        if min_val < -tol or max_val > (1.0 + tol):
            raise ValueError(
                f"{name} must be probabilities in [0, 1] (tolerance={tol}), "
                f"got min={min_val:.6f}, max={max_val:.6f}. "
                "Pass sigmoid=True/softmax=True for logits."
            )

    def _validate_spatial_shape(self, pred: torch.Tensor) -> None:
        if not self.validate_inputs:
            return
        spatial_shape = tuple(pred.shape[2:])
        if any(dim < 3 for dim in spatial_shape):
            raise ValueError(
                "SoftClDiceLoss expects each spatial dimension >= 3 for stable morphology, "
                f"got spatial shape {spatial_shape}"
            )

    def _select_foreground_channels(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, List[int]]:
        channels = pred.shape[1]
        if self.mode == "binary":
            fg_idx = 0 if channels == 1 else self.foreground_channel
            if fg_idx < 0 or fg_idx >= channels:
                raise ValueError(
                    f"foreground_channel={self.foreground_channel} is invalid "
                    f"for {channels} channels"
                )
            return pred[:, fg_idx : fg_idx + 1], target[:, fg_idx : fg_idx + 1], [fg_idx]

        if channels == 1:
            return pred, target, [0]

        background_index = self.background_index
        if background_index < 0:
            background_index += channels
        if background_index < 0 or background_index >= channels:
            raise ValueError(
                f"background_index={self.background_index} is invalid for {channels} channels"
            )

        foreground_indices = [idx for idx in range(channels) if idx != background_index]
        if not foreground_indices:
            raise ValueError(
                f"No foreground classes available: channels={channels}, "
                f"background_index={self.background_index}"
            )
        index_tensor = torch.tensor(foreground_indices, device=pred.device, dtype=torch.long)
        return (
            torch.index_select(pred, dim=1, index=index_tensor),
            torch.index_select(target, dim=1, index=index_tensor),
            foreground_indices,
        )

    def _prepare_weight(
        self,
        weight: torch.Tensor,
        pred: torch.Tensor,
        foreground_indices: List[int],
        num_fg_channels: int,
    ) -> torch.Tensor:
        if weight.ndim == pred.ndim - 1:
            weight = weight.unsqueeze(1)
        if weight.ndim != pred.ndim:
            raise ValueError(f"Weight ndim ({weight.ndim}) must match pred ndim ({pred.ndim})")
        if weight.shape[0] != pred.shape[0] or weight.shape[2:] != pred.shape[2:]:
            raise ValueError(
                "Weight shape must match prediction shape except for channel dimension: "
                f"weight={tuple(weight.shape)}, pred={tuple(pred.shape)}"
            )

        weight = weight.to(device=pred.device, dtype=pred.dtype)
        if weight.shape[1] == num_fg_channels:
            return weight
        if weight.shape[1] == 1:
            return weight.expand(weight.shape[0], num_fg_channels, *weight.shape[2:])
        if weight.shape[1] == pred.shape[1]:
            if num_fg_channels == pred.shape[1]:
                return weight
            index_tensor = torch.tensor(foreground_indices, device=pred.device, dtype=torch.long)
            return torch.index_select(weight, dim=1, index=index_tensor)

        raise ValueError(
            "Weight channel count must be 1, foreground-channel count, "
            "or prediction-channel count; "
            f"got {weight.shape[1]}"
        )

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if pred.ndim not in {4, 5}:
            raise ValueError(f"SoftClDiceLoss expects 4D/5D tensors, got {tuple(pred.shape)}")

        pred = self._apply_activation(pred)
        target = self._prepare_target(target, pred)

        if self.clamp_probabilities:
            pred = pred.clamp(0.0, 1.0)
            target = target.clamp(0.0, 1.0)

        self._validate_spatial_shape(pred)
        self._validate_probability_range(pred, "pred")
        self._validate_probability_range(target, "target")

        pred_fg, target_fg, foreground_indices = self._select_foreground_channels(pred, target)

        if self.clamp_probabilities:
            pred_fg = pred_fg.clamp(0.0, 1.0)
            target_fg = target_fg.clamp(0.0, 1.0)

        fg_weight = None
        if weight is not None:
            fg_weight = self._prepare_weight(weight, pred, foreground_indices, pred_fg.shape[1])
            if self.clamp_probabilities:
                fg_weight = fg_weight.clamp_min(0.0)

        pred_skeleton = _soft_skeletonize_pool(pred_fg, self.num_iters)
        target_skeleton = _soft_skeletonize_pool(target_fg, self.num_iters)

        if fg_weight is not None:
            pred_eval = pred_fg * fg_weight
            target_eval = target_fg * fg_weight
            pred_skeleton_eval = pred_skeleton * fg_weight
            target_skeleton_eval = target_skeleton * fg_weight
        else:
            pred_eval = pred_fg
            target_eval = target_fg
            pred_skeleton_eval = pred_skeleton
            target_skeleton_eval = target_skeleton

        spatial_dims = tuple(range(2, pred_fg.ndim))
        topology_precision = (
            (pred_skeleton_eval * target_eval).sum(dim=spatial_dims) + self.smooth
        ) / (pred_skeleton_eval.sum(dim=spatial_dims) + self.smooth)
        topology_sensitivity = (
            (target_skeleton_eval * pred_eval).sum(dim=spatial_dims) + self.smooth
        ) / (target_skeleton_eval.sum(dim=spatial_dims) + self.smooth)

        cl_dice = (
            2.0
            * topology_precision
            * topology_sensitivity
            / (topology_precision + topology_sensitivity + self.smooth)
        )
        loss = 1.0 - cl_dice

        if self.reduction == "none":
            return loss
        if self.reduction == "sum":
            return loss.sum()
        return loss.mean()


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
    "GANLoss",
    "PerChannelBCEWithLogitsLoss",
    "ScnpLoss",
    "SmoothL1Loss",
    "SoftClDiceLoss",
    "WeightedBCEWithLogitsLoss",
    "WeightedMAELoss",
    "WeightedMSELoss",
]

"""
Deep supervision and multi-task learning utilities for PyTorch Connectomics.

This module implements multi-scale loss computation with deep supervision
and multi-task learning support.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import warnings
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from ..config import Config


class DeepSupervisionHandler:
    """
    Handler for deep supervision and multi-task learning.

    This class manages:
    - Multi-scale loss computation with deep supervision
    - Multi-task learning with task-specific losses
    - Target resizing and interpolation for different scales
    - NaN/Inf detection and debugging

    Args:
        cfg: Hydra Config object or OmegaConf DictConfig
        loss_functions: List of loss functions (nn.ModuleList)
        loss_weights: List of weights for each loss function
        enable_nan_detection: Enable NaN/Inf detection (default: True)
        debug_on_nan: Enter debugger on NaN/Inf (default: True)
    """

    def __init__(
        self,
        cfg: Config | DictConfig,
        loss_functions: nn.ModuleList,
        loss_weights: List[float],
        enable_nan_detection: bool = True,
        debug_on_nan: bool = True,
        loss_weighter: Optional[nn.Module] = None,
    ):
        self.cfg = cfg
        self.loss_functions = loss_functions
        self.loss_weights = loss_weights
        self.enable_nan_detection = enable_nan_detection
        self.debug_on_nan = debug_on_nan
        self.loss_weighter = loss_weighter

        # Deep supervision configuration
        self.clamp_min = getattr(cfg.model, 'deep_supervision_clamp_min', -20.0)
        self.clamp_max = getattr(cfg.model, 'deep_supervision_clamp_max', 20.0)

        # Multi-task configuration
        self.is_multi_task = (
            hasattr(cfg.model, 'multi_task_config') and
            cfg.model.multi_task_config is not None
        )

    def _apply_task_weighting(
        self,
        task_losses: List[torch.Tensor],
        task_names: List[str],
        stage: str,
    ) -> tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        Apply optional adaptive weighting across task-level losses.

        Returns:
            total_loss: Combined loss for all tasks
            weights: Tensor of weights applied to each task (detach for logging)
            log_dict: Additional logs from the weighting strategy
        """
        if self.loss_weighter is None or len(task_losses) == 0:
            total_loss = sum(task_losses)
            weights = torch.ones(len(task_losses), device=task_losses[0].device)
            return total_loss, weights, {}

        total_loss, weights, log_dict = self.loss_weighter.combine(
            task_losses, task_names, stage
        )
        return total_loss, weights, log_dict

    def compute_multitask_loss(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        stage: str = "train",
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute losses for multi-task learning where different output channels
        correspond to different targets with specific losses.

        Args:
            outputs: Model outputs (B, C, D, H, W) where C contains all task channels
            labels: Ground truth labels (B, C, D, H, W) where C contains all target channels

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        loss_dict = {}
        task_losses: List[torch.Tensor] = []
        task_names: List[str] = []

        # Parse multi-task configuration
        # Format: [[start_ch, end_ch, "task_name", [loss_indices]], ...]
        # Track label channel offset (starts at 0)
        label_ch_offset = 0

        for task_idx, task_config in enumerate(self.cfg.model.multi_task_config):
            start_ch, end_ch, task_name, loss_indices = task_config

            # Extract channels for this task from outputs
            task_output = outputs[:, start_ch:end_ch, ...]
            end_ch - start_ch

            # Determine number of label channels needed
            # For softmax-based losses (2+ output channels), label has 1 channel
            # For sigmoid-based losses (1 output channel), label has 1 channel
            # So labels always use 1 channel per task
            num_label_channels = 1

            # Extract label channels
            task_label = labels[:, label_ch_offset:label_ch_offset + num_label_channels, ...]
            label_ch_offset += num_label_channels

            # Apply specified losses for this task
            task_loss_components: List[torch.Tensor] = []
            for loss_idx in loss_indices:
                loss_fn = self.loss_functions[loss_idx]
                weight = self.loss_weights[loss_idx]

                loss = loss_fn(task_output, task_label)

                # Check for NaN/Inf
                if self.enable_nan_detection and (torch.isnan(loss) or torch.isinf(loss)):
                    print(f"\n{'='*80}")
                    print(f"⚠️  NaN/Inf detected in multi-task loss!")
                    print(f"{'='*80}")
                    print(f"Task: {task_name} (channels {start_ch}:{end_ch})")
                    print(f"Loss function: {loss_fn.__class__.__name__} (index {loss_idx})")
                    print(f"Loss value: {loss.item()}")
                    print(f"Output shape: {task_output.shape}, range: [{task_output.min():.4f}, {task_output.max():.4f}]")
                    print(f"Label shape: {task_label.shape}, range: [{task_label.min():.4f}, {task_label.max():.4f}]")
                    print(f"Output contains NaN: {torch.isnan(task_output).any()}")
                    print(f"Label contains NaN: {torch.isnan(task_label).any()}")
                    if self.debug_on_nan:
                        print(f"\nEntering debugger...")
                        pdb.set_trace()
                    raise ValueError(f"NaN/Inf in loss for task '{task_name}' with loss index {loss_idx}")

                weighted_loss = loss * weight
                task_loss_components.append(weighted_loss)

                # Log individual loss
                loss_dict[f'{stage}_loss_{task_name}_loss{loss_idx}'] = loss.item()

            # Aggregate per-task losses before adaptive weighting
            task_loss = sum(task_loss_components)
            task_losses.append(task_loss)
            task_names.append(task_name)
            loss_dict[f'{stage}_loss_{task_name}_unweighted'] = task_loss.item()

        total_loss, weights, weighting_logs = self._apply_task_weighting(
            task_losses, task_names, stage=stage
        )

        for task_name, task_loss, weight in zip(task_names, task_losses, weights):
            loss_dict[f'{stage}_loss_{task_name}_weight'] = float(weight)
            loss_dict[f'{stage}_loss_{task_name}_total'] = (task_loss * weight).item()

        loss_dict.update(weighting_logs)
        loss_dict[f'{stage}_loss_total'] = total_loss.item()
        return total_loss, loss_dict

    def compute_loss_for_scale(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        scale_idx: int,
        stage: str = "train"
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute loss for a single scale with multi-task or standard loss.

        Args:
            output: Model output at this scale (B, C, D, H, W)
            target: Target labels (B, C, D, H, W)
            scale_idx: Scale index for logging (0 = full resolution)
            stage: 'train' or 'val' for logging prefix

        Returns:
            Tuple of (scale_loss, loss_dict) where loss_dict contains individual loss components
        """
        scale_loss = 0.0
        loss_dict = {}

        if self.is_multi_task:
            # Multi-task learning with deep supervision:
            # Apply specific losses to specific channels at each scale
            task_losses: List[torch.Tensor] = []
            task_names: List[str] = []

            for task_idx, task_config in enumerate(self.cfg.model.multi_task_config):
                start_ch, end_ch, task_name, loss_indices = task_config

                # Extract channels for this task
                task_output = output[:, start_ch:end_ch, ...]
                task_target = target[:, start_ch:end_ch, ...]

                # CRITICAL: Clamp outputs to prevent numerical instability
                # At coarser scales (especially with mixed precision), logits can explode
                # BCEWithLogitsLoss: clamp to [-20, 20] (sigmoid maps to [2e-9, 1-2e-9])
                # MSELoss with tanh: clamp to [-10, 10] (tanh maps to [-0.9999, 0.9999])
                task_output = torch.clamp(task_output, min=self.clamp_min, max=self.clamp_max)

                # Apply specified losses for this task
                task_loss_components: List[torch.Tensor] = []
                for loss_idx in loss_indices:
                    loss_fn = self.loss_functions[loss_idx]
                    weight = self.loss_weights[loss_idx]

                    loss = loss_fn(task_output, task_target)

                    # Check for NaN/Inf (only in training mode)
                    if stage == "train" and self.enable_nan_detection and (torch.isnan(loss) or torch.isinf(loss)):
                        print(f"\n{'='*80}")
                        print(f"⚠️  NaN/Inf detected in deep supervision multi-task loss!")
                        print(f"{'='*80}")
                        print(f"Scale: {scale_idx}, Task: {task_name} (channels {start_ch}:{end_ch})")
                        print(f"Loss function: {loss_fn.__class__.__name__} (index {loss_idx})")
                        print(f"Loss value: {loss.item()}")
                        print(f"Output shape: {task_output.shape}, range: [{task_output.min():.4f}, {task_output.max():.4f}]")
                        print(f"Target shape: {task_target.shape}, range: [{task_target.min():.4f}, {task_target.max():.4f}]")
                        if self.debug_on_nan:
                            print(f"\nEntering debugger...")
                            pdb.set_trace()
                        raise ValueError(f"NaN/Inf in deep supervision loss at scale {scale_idx}, task {task_name}")

                    task_loss_components.append(loss * weight)

                task_losses.append(sum(task_loss_components))
                task_names.append(task_name)

            scale_loss, _, _ = self._apply_task_weighting(task_losses, task_names, stage)
        else:
            # Standard deep supervision: apply all losses to all outputs
            # Clamp outputs to prevent numerical instability at coarser scales
            output_clamped = torch.clamp(output, min=self.clamp_min, max=self.clamp_max)

            for loss_fn, weight in zip(self.loss_functions, self.loss_weights):
                loss = loss_fn(output_clamped, target)

                # Check for NaN/Inf (only in training mode)
                if stage == "train" and self.enable_nan_detection and (torch.isnan(loss) or torch.isinf(loss)):
                    print(f"\n{'='*80}")
                    print(f"⚠️  NaN/Inf detected in loss computation!")
                    print(f"{'='*80}")
                    print(f"Loss function: {loss_fn.__class__.__name__}")
                    print(f"Loss value: {loss.item()}")
                    print(f"Scale: {scale_idx}, Weight: {weight}")
                    print(f"Output shape: {output.shape}, range: [{output.min():.4f}, {output.max():.4f}]")
                    print(f"Target shape: {target.shape}, range: [{target.min():.4f}, {target.max():.4f}]")
                    print(f"Output contains NaN: {torch.isnan(output).any()}")
                    print(f"Target contains NaN: {torch.isnan(target).any()}")
                    if self.debug_on_nan:
                        print(f"\nEntering debugger...")
                        pdb.set_trace()
                    raise ValueError(f"NaN/Inf in loss at scale {scale_idx}")

                scale_loss += loss * weight

        loss_dict[f'{stage}_loss_scale_{scale_idx}'] = scale_loss.item()
        return scale_loss, loss_dict

    def compute_deep_supervision_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        stage: str = "train"
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute multi-scale loss with deep supervision.

        Args:
            outputs: Dictionary with 'output' and 'ds_i' keys for deep supervision
            labels: Ground truth labels
            stage: 'train' or 'val' for logging prefix

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Multi-scale loss with deep supervision
        # Weights decrease for smaller scales: [1.0, 0.5, 0.25, 0.125, 0.0625]
        main_output = outputs['output']
        ds_outputs = [outputs[f'ds_{i}'] for i in range(1, 5) if f'ds_{i}' in outputs]

        # Use configured weights or default exponential decay
        if hasattr(self.cfg.model, 'deep_supervision_weights') and self.cfg.model.deep_supervision_weights is not None:
            ds_weights = self.cfg.model.deep_supervision_weights
            # Ensure we have enough weights for all outputs
            if len(ds_weights) < len(ds_outputs) + 1:
                warnings.warn(
                    f"deep_supervision_weights has {len(ds_weights)} weights but "
                    f"{len(ds_outputs) + 1} outputs. Using exponential decay for missing weights."
                )
                ds_weights = [1.0] + [0.5 ** i for i in range(1, len(ds_outputs) + 1)]
        else:
            ds_weights = [1.0] + [0.5 ** i for i in range(1, len(ds_outputs) + 1)]

        all_outputs = [main_output] + ds_outputs

        total_loss = 0.0
        loss_dict = {}

        for scale_idx, (output, ds_weight) in enumerate(zip(all_outputs, ds_weights)):
            # Match target to output size
            target = match_target_to_output(labels, output)

            # Compute loss for this scale
            scale_loss, scale_loss_dict = self.compute_loss_for_scale(
                output, target, scale_idx, stage
            )

            # Accumulate with deep supervision weight
            total_loss += scale_loss * ds_weight
            loss_dict.update(scale_loss_dict)

        loss_dict[f'{stage}_loss_total'] = total_loss.item()
        return total_loss, loss_dict

    def compute_standard_loss(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        stage: str = "train"
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute standard single-scale loss.

        Args:
            outputs: Model outputs (B, C, D, H, W)
            labels: Ground truth labels (B, C, D, H, W)
            stage: 'train' or 'val' for logging prefix

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        total_loss = 0.0
        loss_dict = {}

        # Check if multi-task learning is configured
        if self.is_multi_task:
            # Multi-task learning: apply specific losses to specific channels
            total_loss, loss_dict = self.compute_multitask_loss(outputs, labels, stage=stage)
        else:
            # Standard single-scale loss: apply all losses to all outputs
            for i, (loss_fn, weight) in enumerate(zip(self.loss_functions, self.loss_weights)):
                loss = loss_fn(outputs, labels)

                # Check for NaN/Inf (only in training mode)
                if stage == "train" and self.enable_nan_detection and (torch.isnan(loss) or torch.isinf(loss)):
                    print(f"\n{'='*80}")
                    print(f"⚠️  NaN/Inf detected in loss computation!")
                    print(f"{'='*80}")
                    print(f"Loss function: {loss_fn.__class__.__name__}")
                    print(f"Loss value: {loss.item()}")
                    print(f"Loss index: {i}, Weight: {weight}")
                    print(f"Output shape: {outputs.shape}, range: [{outputs.min():.4f}, {outputs.max():.4f}]")
                    print(f"Label shape: {labels.shape}, range: [{labels.min():.4f}, {labels.max():.4f}]")
                    print(f"Output contains NaN: {torch.isnan(outputs).any()}")
                    print(f"Label contains NaN: {torch.isnan(labels).any()}")
                    if self.debug_on_nan:
                        print(f"\nEntering debugger...")
                        pdb.set_trace()
                    raise ValueError(f"NaN/Inf in loss at index {i}")

                weighted_loss = loss * weight
                total_loss += weighted_loss

                loss_dict[f'{stage}_loss_{i}'] = loss.item()

            loss_dict[f'{stage}_loss_total'] = total_loss.item()

        return total_loss, loss_dict


def match_target_to_output(
    target: torch.Tensor,
    output: torch.Tensor
) -> torch.Tensor:
    """
    Match target size to output size for deep supervision.

    Uses interpolation to downsample labels to match output resolution.
    For segmentation masks, uses nearest-neighbor interpolation to preserve labels.
    For continuous targets, uses trilinear interpolation.

    IMPORTANT: For continuous targets in range [-1, 1] (e.g., tanh-normalized SDT),
    trilinear interpolation can cause overshooting beyond bounds. We clamp the
    resized targets back to [-1, 1] to prevent loss explosion.

    Args:
        target: Target tensor of shape (B, C, D, H, W)
        output: Output tensor of shape (B, C, D', H', W')

    Returns:
        Resized target tensor matching output shape
    """
    if target.shape == output.shape:
        return target

    # Determine interpolation mode based on data type
    if target.dtype in [torch.long, torch.int, torch.int32, torch.int64, torch.uint8, torch.ByteTensor]:
        # Integer labels (including Byte/uint8): use nearest-neighbor
        mode = 'nearest'
        target_resized = F.interpolate(
            target.float(),
            size=output.shape[2:],
            mode=mode,
        ).long()
    else:
        # Continuous values: use trilinear
        mode = 'trilinear'
        target_resized = F.interpolate(
            target,
            size=output.shape[2:],
            mode=mode,
            align_corners=False,
        )

        # CRITICAL FIX: Clamp resized targets to prevent interpolation overshooting
        # For targets in range [-1, 1] (e.g., tanh-normalized SDT), trilinear interpolation
        # can produce values outside this range (e.g., -1.2, 1.3) which causes loss explosion
        # when used with tanh-activated predictions.
        # Check if targets are in typical normalized ranges:
        if target.min() >= -1.5 and target.max() <= 1.5:
            # Likely normalized to [-1, 1] (with some tolerance for existing overshoots)
            target_resized = torch.clamp(target_resized, -1.0, 1.0)
        elif target.min() >= 0.0 and target.max() <= 1.5:
            # Likely normalized to [0, 1]
            target_resized = torch.clamp(target_resized, 0.0, 1.0)

    return target_resized

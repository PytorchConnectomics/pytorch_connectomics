"""
Loss orchestration utilities for PyTorch Connectomics.

This module coordinates single-scale and deep-supervision loss computation,
including multi-task target routing and optional task weighting.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import inspect
import warnings
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from ..config import Config


def _loss_supports_weight(loss_fn: nn.Module) -> bool:
    """Check if a loss function's forward method accepts a 'weight' keyword argument.

    This is used to conditionally pass per-voxel weight masks only to loss
    functions that support them (e.g., WeightedMSELoss, WeightedMAELoss,
    SmoothL1Loss) while skipping the argument for standard losses that do
    not (e.g., MONAI DiceLoss, BCEWithLogitsLoss).
    """
    try:
        sig = inspect.signature(loss_fn.forward)
        return "weight" in sig.parameters
    except (ValueError, TypeError):
        return False


def _is_class_index_loss(loss_fn: nn.Module) -> bool:
    """Return True if loss expects class-index labels (1 channel target).

    Cross-entropy style losses consume dense logits [B, C, ...] and class-index
    targets [B, 1, ...] or [B, ...], unlike BCE/MSE-style losses that require
    channel-aligned dense targets [B, C, ...].
    """
    return loss_fn.__class__.__name__ in {"CrossEntropyLoss", "CrossEntropyLossWrapper"}


class LossOrchestrator:
    """Orchestrates single-scale, multi-scale, and multi-task supervised losses."""

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

        self.clamp_min = getattr(cfg.model, "deep_supervision_clamp_min", -20.0)
        self.clamp_max = getattr(cfg.model, "deep_supervision_clamp_max", 20.0)
        self.is_multi_task = (
            hasattr(cfg.model, "multi_task_config") and cfg.model.multi_task_config is not None
        )

    def _apply_task_weighting(
        self,
        task_losses: List[torch.Tensor],
        task_names: List[str],
        stage: str,
    ) -> tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        if len(task_losses) == 0:
            raise ValueError("No task losses were produced for multi-task loss computation")

        if self.loss_weighter is None:
            total_loss = sum(task_losses)
            weights = torch.ones(len(task_losses), device=task_losses[0].device)
            return total_loss, weights, {}

        total_loss, weights, log_dict = self.loss_weighter.combine(task_losses, task_names, stage)
        return total_loss, weights, log_dict

    def _build_foreground_weight_tensor(self, target: torch.Tensor) -> torch.Tensor:
        fg_weight = 2.0
        loss_weight_mask = torch.ones_like(target)
        loss_weight_mask[target > 0] = fg_weight
        return loss_weight_mask

    def _call_supervised_loss(
        self,
        loss_fn: nn.Module,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Call a standard supervised loss, injecting foreground weighting if supported."""
        if _loss_supports_weight(loss_fn):
            return loss_fn(pred, target, weight=self._build_foreground_weight_tensor(target))
        return loss_fn(pred, target)

    def _check_loss_is_finite(
        self,
        loss: torch.Tensor,
        *,
        stage: str,
        train_only: bool,
        title: str,
        error_message: str,
        info_lines: List[str],
        tensor_map: Dict[str, torch.Tensor],
    ) -> None:
        if train_only and stage != "train":
            return
        if not self.enable_nan_detection:
            return
        if not (torch.isnan(loss) or torch.isinf(loss)):
            return

        print(f"\n{'=' * 80}", flush=True)
        print(title, flush=True)
        print(f"{'=' * 80}", flush=True)
        print(f"Loss value: {loss.item()}", flush=True)
        for line in info_lines:
            print(line, flush=True)
        for name, tensor in tensor_map.items():
            tensor_range = f"[{tensor.min():.4f}, {tensor.max():.4f}]"
            print(f"{name} shape: {tensor.shape}, range: {tensor_range}", flush=True)
            print(f"{name} contains NaN: {torch.isnan(tensor).any()}", flush=True)
        if self.debug_on_nan:
            print("\nEntering debugger...", flush=True)
            pdb.set_trace()
        raise ValueError(error_message)

    def _resize_class_index_target_to_output(
        self, target: torch.Tensor, output: torch.Tensor
    ) -> torch.Tensor:
        """Resize class-index labels with nearest interpolation regardless of dtype."""
        if target.shape[2:] == output.shape[2:]:
            return target

        target_resized = F.interpolate(
            target.float(),
            size=output.shape[2:],
            mode="nearest",
        )
        if target.dtype.is_floating_point:
            return target_resized.to(dtype=target.dtype)
        return target_resized.long()

    def _iter_multitask_views(
        self,
        output: torch.Tensor,
        labels: torch.Tensor,
        *,
        resize_targets_to_output: bool,
    ):
        """Yield per-task slices with consistent target routing for deep/non-deep paths."""
        label_ch_offset = 0

        for task_config in self.cfg.model.multi_task_config:
            start_ch, end_ch, task_name, loss_indices = task_config
            task_output = output[:, start_ch:end_ch, ...]
            task_output_channels = end_ch - start_ch

            task_loss_fns = [self.loss_functions[idx] for idx in loss_indices]
            uses_class_index_targets = any(_is_class_index_loss(fn) for fn in task_loss_fns)
            uses_dense_targets = any(not _is_class_index_loss(fn) for fn in task_loss_fns)

            if uses_class_index_targets and uses_dense_targets:
                raise ValueError(
                    f"Task '{task_name}' mixes class-index and dense target losses. "
                    "Use either CE-style losses only, or dense losses only, per task."
                )

            num_label_channels = 1 if uses_class_index_targets else task_output_channels
            if label_ch_offset + num_label_channels > labels.shape[1]:
                raise ValueError(
                    f"Label channel mismatch for task '{task_name}': expected "
                    f"{num_label_channels} channel(s) at offset {label_ch_offset}, "
                    f"but label tensor has {labels.shape[1]} total channels. "
                    f"Task output slice is [{start_ch}:{end_ch}] "
                    f"({task_output_channels} channel(s))."
                )

            task_target = labels[:, label_ch_offset:label_ch_offset + num_label_channels, ...]
            label_ch_offset += num_label_channels

            if resize_targets_to_output:
                if uses_class_index_targets:
                    task_target = self._resize_class_index_target_to_output(task_target, task_output)
                else:
                    task_target = match_target_to_output(task_target, task_output)

            yield {
                "task_name": task_name,
                "start_ch": start_ch,
                "end_ch": end_ch,
                "loss_indices": loss_indices,
                "task_output": task_output,
                "task_target": task_target,
            }

    def _compute_multitask_task_losses(
        self,
        output: torch.Tensor,
        labels: torch.Tensor,
        *,
        stage: str,
        resize_targets_to_output: bool,
        scale_idx: Optional[int] = None,
        log_components: bool = False,
    ) -> tuple[List[torch.Tensor], List[str], Dict[str, float]]:
        loss_dict: Dict[str, float] = {}
        task_losses: List[torch.Tensor] = []
        task_names: List[str] = []

        for task_view in self._iter_multitask_views(
            output,
            labels,
            resize_targets_to_output=resize_targets_to_output,
        ):
            task_name = task_view["task_name"]
            start_ch = task_view["start_ch"]
            end_ch = task_view["end_ch"]
            loss_indices = task_view["loss_indices"]
            task_output = torch.clamp(
                task_view["task_output"],
                min=self.clamp_min,
                max=self.clamp_max,
            )
            task_target = task_view["task_target"]

            task_loss_components: List[torch.Tensor] = []
            for loss_idx in loss_indices:
                loss_fn = self.loss_functions[loss_idx]
                weight = self.loss_weights[loss_idx]
                loss = self._call_supervised_loss(loss_fn, task_output, task_target)

                if scale_idx is None:
                    title = "⚠️  NaN/Inf detected in multi-task loss!"
                    error_message = (
                        f"NaN/Inf in loss for task '{task_name}' with loss index {loss_idx}"
                    )
                    info_lines = [
                        f"Task: {task_name} (channels {start_ch}:{end_ch})",
                        f"Loss function: {loss_fn.__class__.__name__} (index {loss_idx})",
                    ]
                    train_only = False
                else:
                    title = "⚠️  NaN/Inf detected in deep supervision multi-task loss!"
                    error_message = (
                        f"NaN/Inf in deep supervision loss at scale {scale_idx}, task {task_name}"
                    )
                    info_lines = [
                        f"Scale: {scale_idx}, Task: {task_name} (channels {start_ch}:{end_ch})",
                        f"Loss function: {loss_fn.__class__.__name__} (index {loss_idx})",
                    ]
                    train_only = True

                self._check_loss_is_finite(
                    loss,
                    stage=stage,
                    train_only=train_only,
                    title=title,
                    error_message=error_message,
                    info_lines=info_lines,
                    tensor_map={"Output": task_output, "Target": task_target},
                )

                task_loss_components.append(loss * weight)
                if log_components:
                    loss_dict[f"{stage}_loss_{task_name}_loss{loss_idx}"] = loss.item()

            task_loss = sum(task_loss_components)
            task_losses.append(task_loss)
            task_names.append(task_name)
            if log_components:
                loss_dict[f"{stage}_loss_{task_name}_unweighted"] = task_loss.item()

        return task_losses, task_names, loss_dict

    def compute_multitask_loss(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        stage: str = "train",
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        task_losses, task_names, loss_dict = self._compute_multitask_task_losses(
            outputs,
            labels,
            stage=stage,
            resize_targets_to_output=False,
            log_components=True,
        )

        total_loss, weights, weighting_logs = self._apply_task_weighting(
            task_losses, task_names, stage=stage
        )
        for task_name, task_loss, weight in zip(task_names, task_losses, weights):
            loss_dict[f"{stage}_loss_{task_name}_weight"] = float(weight)
            loss_dict[f"{stage}_loss_{task_name}_total"] = (task_loss * weight).item()
        loss_dict.update(weighting_logs)
        loss_dict[f"{stage}_loss_total"] = total_loss.item()
        return total_loss, loss_dict

    def compute_loss_for_scale(
        self, output: torch.Tensor, target: torch.Tensor, scale_idx: int, stage: str = "train"
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        scale_loss = 0.0
        loss_dict: Dict[str, float] = {}

        if self.is_multi_task:
            task_losses, task_names, _ = self._compute_multitask_task_losses(
                output,
                target,
                stage=stage,
                resize_targets_to_output=(target.shape[2:] != output.shape[2:]),
                scale_idx=scale_idx,
                log_components=False,
            )
            scale_loss, _, _ = self._apply_task_weighting(task_losses, task_names, stage)
        else:
            output_clamped = torch.clamp(output, min=self.clamp_min, max=self.clamp_max)
            for loss_fn, weight in zip(self.loss_functions, self.loss_weights):
                loss = self._call_supervised_loss(loss_fn, output_clamped, target)
                self._check_loss_is_finite(
                    loss,
                    stage=stage,
                    train_only=True,
                    title="⚠️  NaN/Inf detected in loss computation!",
                    error_message=f"NaN/Inf in loss at scale {scale_idx}",
                    info_lines=[
                        f"Loss function: {loss_fn.__class__.__name__}",
                        f"Scale: {scale_idx}, Weight: {weight}",
                    ],
                    tensor_map={"Output": output, "Target": target},
                )
                scale_loss += loss * weight

        loss_dict[f"{stage}_loss_scale_{scale_idx}"] = scale_loss.item()
        return scale_loss, loss_dict

    def compute_deep_supervision_loss(
        self, outputs: Dict[str, torch.Tensor], labels: torch.Tensor, stage: str = "train"
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        main_output = outputs["output"]
        ds_outputs = [outputs[f"ds_{i}"] for i in range(1, 5) if f"ds_{i}" in outputs]

        if (
            hasattr(self.cfg.model, "deep_supervision_weights")
            and self.cfg.model.deep_supervision_weights is not None
        ):
            ds_weights = self.cfg.model.deep_supervision_weights
            if len(ds_weights) < len(ds_outputs) + 1:
                warnings.warn(
                    f"deep_supervision_weights has {len(ds_weights)} weights but "
                    f"{len(ds_outputs) + 1} outputs. Using exponential decay for missing weights."
                )
                ds_weights = [1.0] + [0.5**i for i in range(1, len(ds_outputs) + 1)]
        else:
            ds_weights = [1.0] + [0.5**i for i in range(1, len(ds_outputs) + 1)]

        all_outputs = [main_output] + ds_outputs

        total_loss = 0.0
        loss_dict: Dict[str, float] = {}
        for scale_idx, (output, ds_weight) in enumerate(zip(all_outputs, ds_weights)):
            # For multitask, resize per task to preserve CE-vs-dense target semantics.
            target_for_scale = labels if self.is_multi_task else match_target_to_output(labels, output)
            scale_loss, scale_loss_dict = self.compute_loss_for_scale(
                output, target_for_scale, scale_idx, stage
            )
            total_loss += scale_loss * ds_weight
            loss_dict.update(scale_loss_dict)

        loss_dict[f"{stage}_loss_total"] = total_loss.item()
        return total_loss, loss_dict

    def compute_standard_loss(
        self, outputs: torch.Tensor, labels: torch.Tensor, stage: str = "train"
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        total_loss = 0.0
        loss_dict: Dict[str, float] = {}

        if self.is_multi_task:
            return self.compute_multitask_loss(outputs, labels, stage=stage)

        for i, (loss_fn, weight) in enumerate(zip(self.loss_functions, self.loss_weights)):
            loss = self._call_supervised_loss(loss_fn, outputs, labels)
            self._check_loss_is_finite(
                loss,
                stage=stage,
                train_only=True,
                title="⚠️  NaN/Inf detected in loss computation!",
                error_message=f"NaN/Inf in loss at index {i}",
                info_lines=[
                    f"Loss function: {loss_fn.__class__.__name__}",
                    f"Loss index: {i}, Weight: {weight}",
                ],
                tensor_map={"Output": outputs, "Label": labels},
            )

            total_loss += loss * weight
            loss_dict[f"{stage}_loss_{i}"] = loss.item()

        loss_dict[f"{stage}_loss_total"] = total_loss.item()
        return total_loss, loss_dict


# Backward-compatible name while imports migrate to LossOrchestrator.
DeepSupervisionHandler = LossOrchestrator


def match_target_to_output(target: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
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
    if target.dtype in [
        torch.long,
        torch.int,
        torch.int32,
        torch.int64,
        torch.uint8,
        torch.ByteTensor,
    ]:
        # Integer labels (including Byte/uint8): use nearest-neighbor
        mode = "nearest"
        target_resized = F.interpolate(
            target.float(),
            size=output.shape[2:],
            mode=mode,
        ).long()
    else:
        # Continuous values: use trilinear
        mode = "trilinear"
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

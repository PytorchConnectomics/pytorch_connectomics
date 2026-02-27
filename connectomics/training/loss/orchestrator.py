"""
Loss orchestration utilities for PyTorch Connectomics.

This module coordinates single-scale and deep-supervision loss computation,
including multi-task target routing and optional task weighting.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import warnings
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from ...config import Config
from ...models.loss import (
    LossMetadata,
    get_loss_metadata_for_module,
)
from .plan import LossTermSpec, compile_loss_terms_from_config


class LossOrchestrator:
    """Orchestrates single-scale and deep-supervision losses from explicit loss terms."""

    def __init__(
        self,
        cfg: Config | DictConfig,
        loss_functions: nn.ModuleList,
        loss_weights: List[float],
        enable_nan_detection: bool = True,
        debug_on_nan: bool = True,
        loss_weighter: Optional[nn.Module] = None,
        loss_metadata: Optional[List[LossMetadata]] = None,
    ):
        self.cfg = cfg
        self.loss_functions = loss_functions
        self.loss_weights = loss_weights
        self.enable_nan_detection = enable_nan_detection
        self.debug_on_nan = debug_on_nan
        self.loss_weighter = loss_weighter
        self.loss_metadata = (
            list(loss_metadata)
            if loss_metadata is not None
            else [get_loss_metadata_for_module(loss_fn) for loss_fn in self.loss_functions]
        )

        self.clamp_min = getattr(cfg.model, "deep_supervision_clamp_min", -20.0)
        self.clamp_max = getattr(cfg.model, "deep_supervision_clamp_max", 20.0)
        self.loss_term_specs = compile_loss_terms_from_config(
            cfg,
            self.loss_functions,
            self.loss_weights,
            loss_metadata=self.loss_metadata,
        )
        if not self.loss_term_specs:
            raise ValueError(
                "No loss terms were compiled. Configure model.losses."
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

    def _call_with_optional_spatial_weight(
        self,
        loss_fn: nn.Module,
        *args: torch.Tensor,
        spatial_weight_arg: Optional[str] = None,
        spatial_weight_tensor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if spatial_weight_arg is None or spatial_weight_tensor is None:
            return loss_fn(*args)
        if spatial_weight_arg == "weight":
            return loss_fn(*args, weight=spatial_weight_tensor)
        if spatial_weight_arg == "mask":
            return loss_fn(*args, mask=spatial_weight_tensor)
        raise ValueError(f"Unsupported spatial weight arg: {spatial_weight_arg}")

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


    def _slice_channels(
        self,
        tensor: torch.Tensor,
        channel_slice: tuple[int, int],
    ) -> torch.Tensor:
        start_ch, end_ch = channel_slice
        return tensor[:, start_ch:end_ch, ...]

    def _resize_tensor_for_output(
        self,
        tensor: torch.Tensor,
        output_like: torch.Tensor,
        *,
        target_kind: str = "dense",
    ) -> torch.Tensor:
        if tensor.shape[2:] == output_like.shape[2:]:
            return tensor
        if target_kind == "class_index":
            return self._resize_class_index_target_to_output(tensor, output_like)
        return match_target_to_output(tensor, output_like)

    def _compute_explicit_terms_for_output(
        self,
        output: torch.Tensor,
        labels: torch.Tensor,
        *,
        stage: str,
        scale_idx: Optional[int],
        is_main_scale: bool,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        if not self.loss_term_specs:
            raise ValueError("Explicit loss term path requested but no loss terms are configured")

        loss_dict: Dict[str, float] = {}
        task_losses_map: Dict[str, torch.Tensor] = {}
        task_names_in_order: List[str] = []

        for term in self.loss_term_specs:
            if not is_main_scale and not term.apply_deep_supervision:
                continue

            loss_fn = self.loss_functions[term.loss_index]
            meta = self.loss_metadata[term.loss_index]
            call_kind = term.call_kind
            spatial_weight_arg = term.spatial_weight_arg

            pred = output if term.pred_slice is None else self._slice_channels(output, term.pred_slice)
            pred2 = (
                output if term.pred2_slice is None else self._slice_channels(output, term.pred2_slice)
            )
            if call_kind == "pred_target":
                target = labels if term.target_slice is None else self._slice_channels(labels, term.target_slice)
            else:
                target = None
            term_mask_tensor = (
                self._slice_channels(labels, term.mask_slice)
                if term.mask_slice is not None
                else None
            )
            batch_mask_tensor = mask

            if pred is not None:
                pred = torch.clamp(pred, min=self.clamp_min, max=self.clamp_max)
            if pred2 is not None:
                pred2 = torch.clamp(pred2, min=self.clamp_min, max=self.clamp_max)

            # Resize labels/masks per term for deep supervision scales.
            if pred is not None:
                if target is not None:
                    target = self._resize_tensor_for_output(
                        target,
                        pred,
                        target_kind=term.target_kind,
                    )
                if term_mask_tensor is not None:
                    term_mask_tensor = self._resize_tensor_for_output(
                        term_mask_tensor,
                        pred,
                        target_kind="dense",
                    )
                if batch_mask_tensor is not None:
                    batch_mask_tensor = self._resize_tensor_for_output(
                        batch_mask_tensor,
                        pred,
                        target_kind="class_index",
                    )

            combined_mask_tensor: Optional[torch.Tensor] = None
            if term_mask_tensor is not None and batch_mask_tensor is not None:
                combined_mask_tensor = term_mask_tensor * batch_mask_tensor
            elif term_mask_tensor is not None:
                combined_mask_tensor = term_mask_tensor
            elif batch_mask_tensor is not None:
                combined_mask_tensor = batch_mask_tensor

            if call_kind == "pred_target":
                if pred is None or target is None:
                    raise ValueError(f"Loss term '{term.name}' is missing pred/target tensors")
                spatial_weight_tensor = term_mask_tensor
                if spatial_weight_arg == "weight" and spatial_weight_tensor is None:
                    spatial_weight_tensor = self._build_foreground_weight_tensor(target)
                if batch_mask_tensor is not None:
                    if spatial_weight_tensor is None:
                        spatial_weight_tensor = batch_mask_tensor
                    else:
                        spatial_weight_tensor = spatial_weight_tensor * batch_mask_tensor

                pred_for_loss = pred
                target_for_loss = target
                if spatial_weight_arg is None and combined_mask_tensor is not None:
                    valid = combined_mask_tensor > 0
                    if valid.shape != pred_for_loss.shape:
                        valid = valid.expand_as(pred_for_loss)
                    pred_for_loss = pred_for_loss.masked_fill(~valid, self.clamp_min)
                    target_for_loss = target_for_loss * (combined_mask_tensor > 0).to(
                        dtype=target_for_loss.dtype
                    )
                raw_loss = self._call_with_optional_spatial_weight(
                    loss_fn,
                    pred_for_loss,
                    target_for_loss,
                    spatial_weight_arg=spatial_weight_arg,
                    spatial_weight_tensor=spatial_weight_tensor,
                )
                tensor_map = {"Pred": pred_for_loss, "Target": target_for_loss}
                if spatial_weight_tensor is not None:
                    tensor_map["Mask"] = spatial_weight_tensor
            elif call_kind == "pred_only":
                if pred is None:
                    raise ValueError(f"Loss term '{term.name}' is missing pred tensor")
                spatial_weight_tensor = combined_mask_tensor
                raw_loss = self._call_with_optional_spatial_weight(
                    loss_fn,
                    pred,
                    spatial_weight_arg=spatial_weight_arg,
                    spatial_weight_tensor=spatial_weight_tensor,
                )
                tensor_map = {"Pred": pred}
                if spatial_weight_tensor is not None:
                    tensor_map["Mask"] = spatial_weight_tensor
            elif call_kind == "pred_pred":
                if pred2 is None:
                    raise ValueError(f"Loss term '{term.name}' is missing pred/pred2 tensors")
                spatial_weight_tensor = combined_mask_tensor
                raw_loss = self._call_with_optional_spatial_weight(
                    loss_fn,
                    pred,
                    pred2,
                    spatial_weight_arg=spatial_weight_arg,
                    spatial_weight_tensor=spatial_weight_tensor,
                )
                tensor_map = {"Pred": pred, "Pred2": pred2}
                if spatial_weight_tensor is not None:
                    tensor_map["Mask"] = spatial_weight_tensor
            else:
                raise ValueError(
                    f"Unsupported call_kind {call_kind!r} for loss term '{term.name}' "
                    f"(loss metadata: {meta.name})"
                )

            if scale_idx is None:
                title = "⚠️  NaN/Inf detected in explicit loss term!"
                error_message = f"NaN/Inf in explicit loss term '{term.name}'"
                info_lines = [
                    f"Term: {term.name}",
                    f"Loss function: {loss_fn.__class__.__name__} (index {term.loss_index})",
                    f"Call kind: {call_kind}",
                ]
                train_only = False
            else:
                title = "⚠️  NaN/Inf detected in explicit deep supervision loss term!"
                error_message = f"NaN/Inf in explicit loss term '{term.name}' at scale {scale_idx}"
                info_lines = [
                    f"Scale: {scale_idx}, Term: {term.name}",
                    f"Loss function: {loss_fn.__class__.__name__} (index {term.loss_index})",
                    f"Call kind: {call_kind}",
                ]
                train_only = True

            self._check_loss_is_finite(
                raw_loss,
                stage=stage,
                train_only=train_only,
                title=title,
                error_message=error_message,
                info_lines=info_lines,
                tensor_map=tensor_map,
            )

            weighted_term_loss = raw_loss * term.coefficient
            task_losses_map[term.name] = weighted_term_loss
            task_names_in_order.append(term.name)

            term_prefix = (
                f"{stage}_loss_term_{term.name}"
                if scale_idx is None
                else f"{stage}_loss_scale_{scale_idx}_term_{term.name}"
            )
            loss_dict[f"{term_prefix}_raw"] = raw_loss.item()
            loss_dict[f"{term_prefix}_coef"] = float(term.coefficient)
            loss_dict[f"{term_prefix}_weighted"] = weighted_term_loss.item()

        task_losses = [task_losses_map[name] for name in task_names_in_order]
        if len(task_losses) == 0:
            return torch.tensor(0.0, device=output.device), loss_dict
        total_loss, task_weights, weighting_logs = self._apply_task_weighting(
            task_losses, task_names_in_order, stage=stage
        )
        for task_name, task_loss, task_weight in zip(task_names_in_order, task_losses, task_weights):
            task_prefix = (
                f"{stage}_loss_task_{task_name}"
                if scale_idx is None
                else f"{stage}_loss_scale_{scale_idx}_task_{task_name}"
            )
            loss_dict[f"{task_prefix}_weight"] = float(task_weight)
            loss_dict[f"{task_prefix}_total"] = (task_loss * task_weight).item()

        loss_dict.update(weighting_logs)
        return total_loss, loss_dict

    def compute_loss_for_scale(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        scale_idx: int,
        stage: str = "train",
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute explicit loss terms for one output scale."""
        loss_dict: Dict[str, float] = {}
        scale_loss, explicit_loss_dict = self._compute_explicit_terms_for_output(
            output,
            target,
            stage=stage,
            scale_idx=scale_idx,
            is_main_scale=(scale_idx == 0),
            mask=mask,
        )
        loss_dict.update(explicit_loss_dict)

        loss_dict[f"{stage}_loss_scale_{scale_idx}"] = scale_loss.item()
        return scale_loss, loss_dict

    def compute_deep_supervision_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        stage: str = "train",
        mask: Optional[torch.Tensor] = None,
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
            scale_loss, scale_loss_dict = self.compute_loss_for_scale(
                output, labels, scale_idx, stage, mask=mask
            )
            total_loss += scale_loss * ds_weight
            loss_dict.update(scale_loss_dict)

        loss_dict[f"{stage}_loss_total"] = total_loss.item()
        return total_loss, loss_dict

    def compute_standard_loss(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        stage: str = "train",
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        total_loss, loss_dict = self._compute_explicit_terms_for_output(
            outputs,
            labels,
            stage=stage,
            scale_idx=None,
            is_main_scale=True,
            mask=mask,
        )
        loss_dict[f"{stage}_loss_total"] = total_loss.item()
        return total_loss, loss_dict


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

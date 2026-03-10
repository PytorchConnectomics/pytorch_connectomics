"""
Loss orchestration utilities for PyTorch Connectomics.

This module coordinates single-scale and deep-supervision loss computation,
including multi-task target routing and optional task weighting.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from ...config import Config
from ...models.loss import (
    LossMetadata,
    get_loss_metadata_for_module,
)
from ...utils.channel_slices import resolve_channel_range
from .plan import LossTermSpec, compile_loss_terms_from_config

logger = logging.getLogger(__name__)


def _default_affinity_deepem_crop_enabled(cfg) -> bool:
    """Lazy-import bridge to data.process.affinity."""
    from ...data.process.affinity import affinity_deepem_crop_enabled

    return affinity_deepem_crop_enabled(cfg)


def _default_crop_spatial_fn(
    tensor: torch.Tensor, offsets: list, *, item_name: str = ""
) -> torch.Tensor:
    """Lazy-import bridge to data.process.affinity."""
    from ...data.process.affinity import crop_spatial_by_offsets

    return crop_spatial_by_offsets(tensor, offsets, item_name=item_name)


def _default_compute_valid_mask(
    offsets: list,
    spatial_shape: tuple,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Lazy-import bridge to data.process.affinity."""
    from ...data.process.affinity import compute_affinity_valid_mask

    return compute_affinity_valid_mask(offsets, spatial_shape, device=device)


def _default_resolve_affinity_offsets(cfg, *, num_channels: int, channel_slice):
    """Lazy-import bridge to data.process.affinity."""
    from ...data.process.affinity import resolve_affinity_offsets_for_channel_slice

    return resolve_affinity_offsets_for_channel_slice(
        cfg,
        num_channels=num_channels,
        channel_slice=channel_slice,
    )


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
        *,
        affinity_crop_enabled_fn: Optional[Callable] = None,
        crop_spatial_fn: Optional[Callable] = None,
        resolve_affinity_offsets_fn: Optional[Callable] = None,
        compute_valid_mask_fn: Optional[Callable] = None,
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
        self.loss_cfg = getattr(cfg.model, "loss", None)

        # Injected affinity functions (decoupled from data.process.affinity)
        self._affinity_crop_enabled_fn = (
            affinity_crop_enabled_fn or _default_affinity_deepem_crop_enabled
        )
        self._crop_spatial_fn = crop_spatial_fn or _default_crop_spatial_fn
        self._resolve_affinity_offsets_fn = (
            resolve_affinity_offsets_fn or _default_resolve_affinity_offsets
        )
        self._compute_valid_mask_fn = compute_valid_mask_fn or _default_compute_valid_mask

        if self.loss_cfg is None:
            raise ValueError("cfg.model.loss is required for loss orchestration")
        self.clamp_min = float(self.loss_cfg.deep_supervision_clamp_min)
        self.clamp_max = float(self.loss_cfg.deep_supervision_clamp_max)
        self.affinity_deepem_crop = self._affinity_crop_enabled_fn(cfg)
        self.loss_term_specs = compile_loss_terms_from_config(
            cfg,
            self.loss_functions,
            self.loss_weights,
            loss_metadata=self.loss_metadata,
        )
        if not self.loss_term_specs:
            raise ValueError("No loss terms were compiled. Configure model.loss.losses.")

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

    def _build_pos_weight_tensor(
        self,
        target: torch.Tensor,
        *,
        pos_weight: Optional[float | str] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if pos_weight is None:
            pos_weight = "auto"

        if isinstance(pos_weight, str):
            if pos_weight != "auto":
                raise ValueError(
                    f"Unsupported pos_weight mode: {pos_weight!r}. "
                    "Expected a positive number or 'auto'."
                )
            if valid_mask is None:
                valid = torch.ones_like(target, dtype=torch.bool)
            else:
                valid = valid_mask > 0

            pos = (target > 0) & valid
            neg = (target <= 0) & valid
            pos_count = torch.count_nonzero(pos).item()
            neg_count = torch.count_nonzero(neg).item()

            loss_weight_mask = torch.ones_like(target)
            if pos_count == 0 or neg_count == 0:
                return loss_weight_mask

            # Class-balanced weights normalized to keep mean(valid weight)=1:
            # w_fg / w_bg = neg/pos and (pos*w_fg + neg*w_bg)/(pos+neg) = 1.
            # Then cap extreme class-imbalance amplification.
            valid_count = float(pos_count + neg_count)
            bg_weight = valid_count / (2.0 * float(neg_count))
            fg_weight = valid_count / (2.0 * float(pos_count))
            max_ratio_weight = 10.0
            bg_weight = min(bg_weight, max_ratio_weight)
            fg_weight = min(fg_weight, max_ratio_weight)
            loss_weight_mask.fill_(bg_weight)
            loss_weight_mask[target > 0] = fg_weight
            return loss_weight_mask

        fg_weight = float(pos_weight)
        if fg_weight <= 0:
            raise ValueError(f"pos_weight must be > 0, got {fg_weight}")

        loss_weight_mask = torch.ones_like(target)
        loss_weight_mask[target > 0] = fg_weight
        return loss_weight_mask

    def _compute_auto_pos_weight_scalar(
        self,
        target: torch.Tensor,
        *,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> float:
        if valid_mask is None:
            valid = torch.ones_like(target, dtype=torch.bool)
        else:
            valid = valid_mask > 0

        pos = (target > 0) & valid
        neg = (target <= 0) & valid
        pos_count = torch.count_nonzero(pos).item()
        neg_count = torch.count_nonzero(neg).item()
        if pos_count == 0 or neg_count == 0:
            return 1.0
        return min(float(neg_count) / float(pos_count), 10.0)

    def _call_with_optional_spatial_weight(
        self,
        loss_fn: nn.Module,
        *args: torch.Tensor,
        spatial_weight_arg: Optional[str] = None,
        spatial_weight_tensor: Optional[torch.Tensor] = None,
        extra_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        kwargs: Dict[str, torch.Tensor] = dict(extra_kwargs or {})
        if spatial_weight_arg is None or spatial_weight_tensor is None:
            return loss_fn(*args, **kwargs)
        if spatial_weight_arg == "weight":
            return loss_fn(*args, weight=spatial_weight_tensor, **kwargs)
        if spatial_weight_arg == "mask":
            return loss_fn(*args, mask=spatial_weight_tensor, **kwargs)
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

        logger.warning(f"\n{'=' * 80}")
        logger.warning(title)
        logger.warning(f"{'=' * 80}")
        logger.warning(f"Loss value: {loss.item()}")
        for line in info_lines:
            logger.warning(line)
        for name, tensor in tensor_map.items():
            tensor_range = f"[{tensor.min():.4f}, {tensor.max():.4f}]"
            logger.warning(f"{name} shape: {tensor.shape}, range: {tensor_range}")
            logger.warning(f"{name} contains NaN: {torch.isnan(tensor).any()}")
        if self.debug_on_nan:
            logger.warning(
                "debug_on_nan=True is set, but interactive breakpoints are disabled. "
                "Raising ValueError with diagnostics instead."
            )
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

    @staticmethod
    def _resolve_channel_selector_range(
        channel_slice,
        *,
        num_channels: int,
    ) -> tuple[int, int]:
        """Resolve a channel selector to absolute half-open indices."""
        return resolve_channel_range(
            channel_slice,
            num_channels=num_channels,
            context="channel slice",
        )

    def _slice_channels(
        self,
        tensor: torch.Tensor,
        channel_slice,
    ) -> torch.Tensor:
        start_idx, end_idx = self._resolve_channel_selector_range(
            channel_slice,
            num_channels=int(tensor.shape[1]),
        )
        return tensor[:, start_idx:end_idx, ...]

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

    def _resolve_affinity_offsets_for_term(
        self,
        term: LossTermSpec,
        *,
        labels_num_channels: int,
        is_main_scale: bool,
    ) -> Optional[list[tuple[int, int, int]]]:
        if not self.affinity_deepem_crop or not is_main_scale:
            return None
        return self._resolve_affinity_offsets_fn(
            self.cfg,
            num_channels=labels_num_channels,
            channel_slice=term.target_slice,
        )

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
        task_losses: List[torch.Tensor] = []
        task_names_in_order: List[str] = []
        term_prefix_by_name: Dict[str, str] = {}

        for term in self.loss_term_specs:
            if not is_main_scale and not term.apply_deep_supervision:
                continue

            loss_fn = self.loss_functions[term.loss_index]
            meta = self.loss_metadata[term.loss_index]
            call_kind = term.call_kind
            spatial_weight_arg = term.spatial_weight_arg
            affinity_offsets = self._resolve_affinity_offsets_for_term(
                term,
                labels_num_channels=int(labels.shape[1]),
                is_main_scale=is_main_scale,
            )

            pred = (
                output if term.pred_slice is None else self._slice_channels(output, term.pred_slice)
            )
            pred2 = (
                output
                if term.pred2_slice is None
                else self._slice_channels(output, term.pred2_slice)
            )
            if call_kind == "pred_target":
                target = (
                    labels
                    if term.target_slice is None
                    else self._slice_channels(labels, term.target_slice)
                )
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
                # DeepEM get_pair logic: per-channel valid mask instead of
                # uniform spatial crop.  Each channel is masked to its own
                # valid region so (a) border zeros don't train the model and
                # (b) reflection-padding artifacts from augmentation are
                # excluded for long-range channels.
                affinity_valid_mask: Optional[torch.Tensor] = None
                if affinity_offsets:
                    spatial_shape = tuple(int(v) for v in pred.shape[2:])
                    affinity_valid_mask = self._compute_valid_mask_fn(
                        affinity_offsets,
                        spatial_shape,
                        device=pred.device,
                    )
                    # (C, D, H, W) -> (1, C, D, H, W); match pred dtype for AMP
                    affinity_valid_mask = affinity_valid_mask.unsqueeze(0).to(pred.dtype)

            # Merge all masks into combined_mask_tensor (for masking invalid
            # regions in the loss).  The affinity valid mask is kept separate
            # from term_mask_tensor so that the pos_weight / class-balancing
            # computation below works correctly.
            combined_mask_tensor: Optional[torch.Tensor] = None
            for m in (term_mask_tensor, batch_mask_tensor, affinity_valid_mask):
                if m is not None:
                    combined_mask_tensor = (
                        m if combined_mask_tensor is None else combined_mask_tensor * m
                    )

            if call_kind == "pred_target":
                if pred is None or target is None:
                    raise ValueError(f"Loss term '{term.name}' is missing pred/target tensors")
                spatial_weight_tensor = term_mask_tensor
                extra_loss_kwargs: Dict[str, torch.Tensor] = {}
                is_weighted_bce = meta.name == "WeightedBCEWithLogitsLoss"
                if spatial_weight_arg == "weight" and spatial_weight_tensor is None:
                    if not is_weighted_bce:
                        spatial_weight_tensor = self._build_pos_weight_tensor(
                            target,
                            pos_weight=term.pos_weight,
                            valid_mask=combined_mask_tensor,
                        )
                if batch_mask_tensor is not None:
                    if spatial_weight_tensor is None:
                        spatial_weight_tensor = batch_mask_tensor
                    else:
                        spatial_weight_tensor = spatial_weight_tensor * batch_mask_tensor
                # Fold affinity valid mask into the spatial weight so border
                # voxels get zero weight (no gradient).  This is applied AFTER
                # pos_weight computation so class balancing uses the correct
                # valid-region statistics.
                if affinity_valid_mask is not None:
                    if spatial_weight_tensor is None:
                        spatial_weight_tensor = affinity_valid_mask
                    else:
                        spatial_weight_tensor = spatial_weight_tensor * affinity_valid_mask

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
                if is_weighted_bce and term.pos_weight is not None:
                    if isinstance(term.pos_weight, str):
                        auto_pos_weight = self._compute_auto_pos_weight_scalar(
                            target_for_loss,
                            valid_mask=combined_mask_tensor,
                        )
                        extra_loss_kwargs["pos_weight"] = torch.tensor(
                            [auto_pos_weight],
                            device=pred_for_loss.device,
                            dtype=pred_for_loss.dtype,
                        )
                    elif (
                        float(term.pos_weight) != 1.0
                        and getattr(loss_fn, "pos_weight", None) is None
                    ):
                        extra_loss_kwargs["pos_weight"] = torch.tensor(
                            [float(term.pos_weight)],
                            device=pred_for_loss.device,
                            dtype=pred_for_loss.dtype,
                        )
                raw_loss = self._call_with_optional_spatial_weight(
                    loss_fn,
                    pred_for_loss,
                    target_for_loss,
                    spatial_weight_arg=spatial_weight_arg,
                    spatial_weight_tensor=spatial_weight_tensor,
                    extra_kwargs=extra_loss_kwargs,
                )
                tensor_map = {"Pred": pred_for_loss, "Target": target_for_loss}
                if spatial_weight_tensor is not None:
                    tensor_map["Mask"] = spatial_weight_tensor
                if "pos_weight" in extra_loss_kwargs:
                    tensor_map["PosWeight"] = extra_loss_kwargs["pos_weight"]
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
                title = "WARNING: NaN/Inf detected in explicit loss term!"
                error_message = f"NaN/Inf in explicit loss term '{term.name}'"
                info_lines = [
                    f"Term: {term.name}",
                    f"Loss function: {loss_fn.__class__.__name__} (index {term.loss_index})",
                    f"Call kind: {call_kind}",
                ]
                train_only = False
            else:
                title = "WARNING: NaN/Inf detected in explicit deep supervision loss term!"
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
            task_losses.append(weighted_term_loss)
            task_names_in_order.append(term.name)

            term_prefix = (
                f"{stage}_loss_{term.name}"
                if scale_idx is None
                else f"{stage}_loss_scale_{scale_idx}_{term.name}"
            )
            term_prefix_by_name[term.name] = term_prefix
            loss_dict[f"{term_prefix}_raw"] = raw_loss.item()
            loss_dict[f"{term_prefix}_coef"] = float(term.coefficient)
            loss_dict[f"{term_prefix}_weighted"] = weighted_term_loss.item()

        if len(task_losses) == 0:
            return torch.tensor(0.0, device=output.device), loss_dict
        total_loss, task_weights, weighting_logs = self._apply_task_weighting(
            task_losses, task_names_in_order, stage=stage
        )
        if self.loss_weighter is not None:
            for task_name, task_loss, task_weight in zip(
                task_names_in_order, task_losses, task_weights
            ):
                term_prefix = term_prefix_by_name[task_name]
                loss_dict[f"{term_prefix}_balance_weight"] = float(task_weight)
                loss_dict[f"{term_prefix}_balanced"] = (task_loss * task_weight).item()

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
            self.loss_cfg is not None
            and hasattr(self.loss_cfg, "deep_supervision_weights")
            and self.loss_cfg.deep_supervision_weights is not None
        ):
            ds_weights = self.loss_cfg.deep_supervision_weights
            if len(ds_weights) < len(ds_outputs) + 1:
                logger.warning(
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

        # Clamp resized targets to prevent interpolation overshooting.
        # Trilinear interpolation can produce values outside the original range
        # (e.g., -1.2 from a [-1, 1] SDT target), causing loss explosion with
        # tanh-activated predictions.
        #
        # Heuristic: use 0.5 tolerance bands to detect the original range:
        #   - [-1.5, 1.5] → target was in [-1, 1] (e.g., tanh-normalized SDT)
        #   - [0.0, 1.5]  → target was in [0, 1]  (e.g., sigmoid targets / masks)
        if target.min() >= -1.5 and target.max() <= 1.5:
            target_resized = torch.clamp(target_resized, -1.0, 1.0)
        elif target.min() >= 0.0 and target.max() <= 1.5:
            target_resized = torch.clamp(target_resized, 0.0, 1.0)

    return target_resized

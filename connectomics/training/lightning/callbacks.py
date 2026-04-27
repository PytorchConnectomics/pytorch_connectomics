"""
PyTorch Lightning callbacks for PyTorch Connectomics.

Provides callbacks for visualization, checkpointing, and monitoring.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pytorch_lightning as pl
import torch
import torchvision.utils as vutils
from pytorch_lightning import Callback
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

try:
    from torch.utils.data import ChainDataset

    _HAS_CHAIN_DATASET = True
except ImportError:
    _HAS_CHAIN_DATASET = False

from ...data.processing.affinity import (
    compute_affinity_valid_mask,
    crop_spatial_by_offsets,
    resolve_affinity_channel_groups_from_cfg,
    resolve_affinity_mode_from_cfg,
    resolve_affinity_offsets_for_channel_slice,
)
from ...utils import get_model_head_names, resolve_configured_output_head, select_output_tensor
from .visualizer import Visualizer, get_visualization_mask

logger = logging.getLogger(__name__)

__all__ = [
    "VisualizationCallback",
    "NaNDetectionCallback",
    "EMAWeightsCallback",
    "ValidationReseedingCallback",
]


def _apply_affinity_visualization_crop_if_needed(
    cfg,
    *,
    image: torch.Tensor,
    label: torch.Tensor,
    pred: torch.Tensor,
    mask: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Mirror training-time affinity target handling in TensorBoard visualization.

    For mixed-task stacks (for example affinity + SDT), training applies a
    per-channel affinity valid-region mask rather than a global crop. The
    visualization path applies the same masking to labels and predictions so
    invalid affinity borders are hidden. Pure-affinity tensors still receive
    the common valid-region crop for a cleaner spatial view.
    """
    if cfg is None:
        return image, label, pred, mask
    affinity_mode = resolve_affinity_mode_from_cfg(cfg)
    if affinity_mode is None:
        return image, label, pred, mask
    if image.ndim < 5 or label.ndim < 5 or pred.ndim < 5:
        return image, label, pred, mask

    num_channels = int(label.shape[1])
    if num_channels != int(pred.shape[1]):
        return image, label, pred, mask

    spatial_shape = tuple(int(v) for v in label.shape[-3:])
    affinity_groups = resolve_affinity_channel_groups_from_cfg(cfg)
    if affinity_groups:
        label = label.clone()
        pred = pred.clone()

    for (start, end), offsets in affinity_groups:
        start_idx = max(0, int(start))
        end_idx = min(int(end), num_channels)
        if start_idx >= end_idx:
            continue
        if end_idx - start_idx != len(offsets):
            continue

        group_mask = compute_affinity_valid_mask(
            list(offsets[: end_idx - start_idx]),
            spatial_shape,
            affinity_mode=affinity_mode,
            device=label.device,
        ).unsqueeze(0)
        group_mask = group_mask.to(device=label.device, dtype=label.dtype)
        target_valid = (label[:, start_idx:end_idx] >= 0).to(dtype=label.dtype)
        group_valid = group_mask * target_valid
        label[:, start_idx:end_idx] = torch.where(
            group_valid > 0,
            label[:, start_idx:end_idx],
            torch.zeros_like(label[:, start_idx:end_idx]),
        )
        pred[:, start_idx:end_idx] = pred[:, start_idx:end_idx] * group_valid.to(
            device=pred.device,
            dtype=pred.dtype,
        )

    offsets = resolve_affinity_offsets_for_channel_slice(
        cfg,
        num_channels=num_channels,
        channel_slice=None,
    )
    if not offsets:
        return image, label, pred, mask

    image = crop_spatial_by_offsets(
        image,
        offsets,
        affinity_mode=affinity_mode,
        item_name="visualization image",
    )
    label = crop_spatial_by_offsets(
        label,
        offsets,
        affinity_mode=affinity_mode,
        item_name="visualization label",
    )
    pred = crop_spatial_by_offsets(
        pred,
        offsets,
        affinity_mode=affinity_mode,
        item_name="visualization prediction",
    )
    if mask is not None:
        mask = crop_spatial_by_offsets(
            mask,
            offsets,
            affinity_mode=affinity_mode,
            item_name="visualization mask",
        )
    return image, label, pred, mask


class VisualizationCallback(Callback):
    """
    Lightning callback for TensorBoard visualization.

    Visualizes input images, ground truth, and predictions at the end of each epoch.
    """

    def __init__(
        self,
        cfg,
        max_images: int = 8,
        num_slices: int = 8,
        slice_sampling: str = "uniform",
        log_every_n_epochs: int = 1,
    ):
        """
        Args:
            cfg: Hydra config object
            max_images: Maximum number of images to visualize per batch
            num_slices: Number of slices to show for 3D volumes
            slice_sampling: Slice selection mode ("uniform" or "consecutive")
            log_every_n_epochs: Log visualization every N epochs (default: 1)
        """
        super().__init__()
        self.visualizer = Visualizer(cfg, max_images=max_images)
        self.num_slices = num_slices
        self.slice_sampling = slice_sampling
        self.log_every_n_epochs = log_every_n_epochs
        self.cfg = cfg
        images_cfg = cfg.monitor.logging.images
        self.channel_mode = getattr(images_cfg, "channel_mode", "all") or "all"
        self.selected_channels = getattr(images_cfg, "selected_channels", None)
        self.output_head = getattr(images_cfg, "head", None)

        # Store batch for end-of-epoch visualization
        self._last_train_batch = None
        self._last_val_batch = None
        # Next epoch at which to log per prefix; advances by log_every_n_epochs
        # after each log so val (which runs every val_check_interval epochs)
        # still fires at the first val epoch >= each 10-epoch boundary.
        self._next_log_epoch: Dict[str, int] = {"train": 0, "val": 0}

    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs: Dict[str, Any],
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ):
        """Store first batch for epoch-end visualization."""
        if batch_idx == 0:
            self._last_train_batch = self._build_cached_batch(batch)
            # Log image+label on the very first batch (no prediction) for data sanity check.
            if trainer.current_epoch == 0 and trainer.logger is not None:
                self._log_data_check(trainer, batch)

    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs: Dict[str, Any],
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        """Store first batch for epoch-end visualization."""
        if batch_idx == 0:
            self._last_val_batch = self._build_cached_batch(batch)

    def on_train_epoch_end(self, trainer, pl_module):
        """Visualize at end of training epoch based on log_every_n_epochs."""
        self._log_cached_batch_visualization(
            trainer=trainer,
            pl_module=pl_module,
            cached_batch=self._last_train_batch,
            prefix="train",
            restore_train_mode=True,
        )

    def on_validation_epoch_end(self, trainer, pl_module):
        """Visualize at end of validation epoch based on log_every_n_epochs."""
        self._log_cached_batch_visualization(
            trainer=trainer,
            pl_module=pl_module,
            cached_batch=self._last_val_batch,
            prefix="val",
            restore_train_mode=False,
        )

    @staticmethod
    def _build_cached_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        cached = {
            "image": batch["image"].detach(),
            "label": batch["label"].detach(),
        }
        mask = get_visualization_mask(batch)
        if mask is not None:
            cached["mask"] = mask.detach()
        return cached

    def _log_cached_batch_visualization(
        self,
        trainer,
        pl_module,
        cached_batch: Optional[Dict[str, torch.Tensor]],
        prefix: str,
        restore_train_mode: bool,
    ) -> None:
        """Run inference on cached batch and log visualization images."""
        if cached_batch is None or trainer.logger is None:
            return
        if prefix == "val" and getattr(trainer, "sanity_checking", False):
            return
        next_log = self._next_log_epoch.get(prefix, 0)
        if trainer.current_epoch < next_log:
            return
        step = max(1, int(self.log_every_n_epochs))
        while next_log <= trainer.current_epoch:
            next_log += step
        self._next_log_epoch[prefix] = next_log

        try:
            writer = trainer.logger.experiment
            with torch.no_grad():
                if restore_train_mode:
                    pl_module.eval()

                image = cached_batch["image"].to(pl_module.device)
                pred = pl_module(image)

                if restore_train_mode:
                    pl_module.train()

            image_cpu = image.cpu()
            mask_cpu = cached_batch.get("mask", None)
            if mask_cpu is not None:
                mask_cpu = mask_cpu.cpu()

            # Determine which heads to visualize.
            head_names = self._get_visualization_heads(pl_module, pred)

            for head_name in head_names:
                # Temporarily override output_head for this iteration.
                saved_head = self.output_head
                self.output_head = head_name
                try:
                    label_tensor, pred_tensor, resolved = self._select_visualization_tensors(
                        pl_module,
                        cached_batch["label"],
                        pred,
                    )
                finally:
                    self.output_head = saved_head

                label_cpu = label_tensor.cpu()
                pred_cpu = pred_tensor.cpu()
                img_viz, lbl_viz, pred_viz, mask_viz = _apply_affinity_visualization_crop_if_needed(
                    self.cfg,
                    image=image_cpu,
                    label=label_cpu,
                    pred=pred_cpu,
                    mask=mask_cpu,
                )

                head_prefix = f"{prefix}_{head_name}" if head_name else prefix
                self._log_visualization(
                    image=img_viz,
                    label=lbl_viz,
                    mask=mask_viz,
                    pred=pred_viz,
                    writer=writer,
                    iteration=trainer.current_epoch,
                    prefix=head_prefix,
                )
            if prefix == "train":
                logger.info("Saved visualization for epoch %s", trainer.current_epoch)
        except Exception as e:
            import traceback

            logger.error("%s epoch-end visualization failed: %s", prefix, e)
            logger.error("Error type: %s", type(e).__name__)
            if hasattr(e, "__traceback__"):
                logger.error(
                    "Traceback:\n%s",
                    "".join(traceback.format_exception(type(e), e, e.__traceback__)),
                )

    def _log_visualization(
        self,
        image: torch.Tensor,
        label: torch.Tensor,
        mask: Optional[torch.Tensor],
        pred: torch.Tensor,
        writer,
        iteration: int,
        prefix: str,
    ) -> None:
        """Log either consecutive-slice or single-slice visualization."""
        if image.ndim == 5 and self.num_slices > 1:
            self.visualizer.visualize_consecutive_slices(
                volume=image,
                label=label,
                mask=mask,
                output=pred,
                writer=writer,
                iteration=iteration,
                prefix=prefix,
                num_slices=self.num_slices,
                slice_sampling=self.slice_sampling,
                channel_mode=self.channel_mode,
                selected_channels=self.selected_channels,
            )
        else:
            self.visualizer.visualize(
                volume=image,
                label=label,
                mask=mask,
                output=pred,
                iteration=iteration,
                writer=writer,
                prefix=prefix,
                channel_mode=self.channel_mode,
                selected_channels=self.selected_channels,
            )

    def _log_data_check(self, trainer, batch: Dict[str, torch.Tensor]) -> None:
        """Log image + label from the first training batch (no prediction).

        Runs once at the start of training so the user can visually verify
        data loading, augmentation, and label transforms before waiting for
        the first epoch to finish.  Only label channels are shown (no output).
        """
        try:
            writer = trainer.logger.experiment
            image = self.visualizer._prepare_volume(batch["image"].cpu())
            label = self.visualizer._prepare_volume(batch["label"].cpu())

            image = self.visualizer._normalize(image)
            label = self.visualizer._normalize(label)

            # Limit samples
            image = image[: self.visualizer.max_images]
            label = label[: self.visualizer.max_images]

            # Log input image
            writer.add_image(
                "data_check/input",
                vutils.make_grid(
                    image, nrow=min(8, self.visualizer.max_images), normalize=True, scale_each=True
                ),
                0,
            )

            # Log each label channel
            for i in range(min(label.shape[1], self.visualizer.max_channels)):
                ch = label[:, i : i + 1].repeat(1, 3, 1, 1)
                writer.add_image(
                    f"data_check/label_channel_{i}",
                    vutils.make_grid(
                        ch, nrow=min(8, self.visualizer.max_images), normalize=True, scale_each=True
                    ),
                    0,
                )

            logger.info(
                "Logged data check visualization (image + %d label channels)", label.shape[1]
            )
        except Exception as e:
            logger.warning("Data check visualization failed: %s", e)

    @staticmethod
    def _to_tensor(pred):
        """Extract a tensor from possible deep-supervision dict outputs."""
        if isinstance(pred, torch.Tensor):
            return pred
        if isinstance(pred, dict):
            for key in ("out", "pred", "logits"):
                if key in pred and isinstance(pred[key], torch.Tensor):
                    return pred[key]
            for value in pred.values():
                if isinstance(value, torch.Tensor):
                    return value
        raise TypeError(f"Unexpected prediction type for visualization: {type(pred)}")

    def _get_visualization_heads(self, pl_module, pred) -> list:
        """Return list of head names to visualize.

        If ``head: all``, returns all head names from model config (same source
        the inference/test pipeline uses for multi-head enumeration).
        Otherwise returns a single-element list with the configured head.
        """
        if isinstance(self.output_head, str) and self.output_head.strip().lower() == "all":
            cfg = getattr(pl_module, "cfg", self.cfg)
            head_names = get_model_head_names(cfg)
            if head_names:
                return head_names
        return [
            (
                self.output_head
                if isinstance(self.output_head, str) and self.output_head.strip()
                else None
            )
        ]

    def _resolve_requested_output_head(self) -> Optional[str]:
        if isinstance(self.output_head, str) and self.output_head.strip():
            return self.output_head.strip()
        return resolve_configured_output_head(
            self.cfg,
            purpose="visualization selection",
            allow_none=True,
        )

    def _select_visualization_tensors(
        self,
        pl_module,
        label: torch.Tensor,
        pred,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[str]]:
        requested_head = self._resolve_requested_output_head()
        primary_head = getattr(getattr(pl_module, "cfg", self.cfg).model, "primary_head", None)

        try:
            pred_tensor, resolved_head = select_output_tensor(
                pred,
                requested_head=requested_head,
                primary_head=primary_head,
                purpose="visualization selection",
            )
        except (TypeError, ValueError):
            if requested_head is not None:
                raise
            return label, self._to_tensor(pred), None

        if resolved_head is None:
            return label, pred_tensor, None

        target_slice = pl_module._resolve_validation_target_slice(resolved_head)
        label_tensor = (
            label
            if target_slice is None
            else pl_module._slice_tensor_channels(
                label,
                target_slice,
                context=f"visualization target slice for head '{resolved_head}'",
            )
        )
        return label_tensor, pred_tensor, resolved_head


class NaNDetectionCallback(Callback):
    """
    Lightning callback to detect NaN/Inf values in training loss and trigger debugger.

    This callback monitors the loss value after each training step and:
    - Checks for NaN or Inf values
    - Prints diagnostic information (loss value, batch statistics, gradient norms)
    - Optionally raises immediately when configured
    - Can terminate training or continue with a warning

    Args:
        check_grads: If True, also check for NaN/Inf in model gradients
        check_inputs: If True, also check for NaN/Inf in batch inputs
        debug_on_nan: If True, emit extra diagnostics when NaN is detected
        terminate_on_nan: If True, raise exception to stop training when NaN is detected
        print_diagnostics: If True, print detailed diagnostics when NaN is detected
    """

    def __init__(
        self,
        check_grads: bool = True,
        check_inputs: bool = True,
        debug_on_nan: bool = True,
        terminate_on_nan: bool = False,
        print_diagnostics: bool = True,
    ):
        super().__init__()
        self.check_grads = check_grads
        self.check_inputs = check_inputs
        self.debug_on_nan = debug_on_nan
        self.terminate_on_nan = terminate_on_nan
        self.print_diagnostics = print_diagnostics
        self._last_batch = None  # Store last batch for debugging

    def on_train_batch_start(
        self, trainer, pl_module, batch: Dict[str, torch.Tensor], batch_idx: int
    ):
        """Store batch for later debugging."""
        self._last_batch = batch

    def on_after_backward(self, trainer, pl_module):
        """Check for NaN/Inf right after backward pass (earliest point to catch it)."""
        # This runs BEFORE on_train_batch_end, giving us the earliest detection
        if self._last_batch is None:
            return

        # Check logged metrics (this is where NaN appears in train_loss_0_step)
        logged_metrics = trainer.callback_metrics

        is_nan = False
        is_inf = False
        loss_value = None
        nan_metric_keys = []

        # Check all loss metrics
        for key, value in logged_metrics.items():
            if "loss" in key.lower() or "train" in key.lower():
                if isinstance(value, torch.Tensor):
                    val = value.item() if value.numel() == 1 else None
                else:
                    val = value

                if val is not None and (val != val or abs(val) == float("inf")):
                    is_nan = is_nan or (val != val)
                    is_inf = is_inf or (abs(val) == float("inf"))
                    nan_metric_keys.append(f"{key}={val}")
                    if loss_value is None:
                        loss_value = val

        if is_nan or is_inf:
            self._handle_nan_detection(
                trainer, pl_module, self._last_batch, is_nan, is_inf, loss_value, nan_metric_keys
            )

    def _handle_nan_detection(
        self,
        trainer,
        pl_module,
        batch: Dict[str, torch.Tensor],
        is_nan: bool,
        is_inf: bool,
        loss_value: float,
        nan_metric_keys: list,
    ):
        """Handle NaN/Inf detection with diagnostics and debugging."""
        issue_type = "NaN" if is_nan else "Inf"
        logger.warning("\n%s", "=" * 80)
        logger.warning("%s DETECTED IN TRAINING LOSS!", issue_type)
        logger.warning("=" * 80)
        logger.warning("Epoch: %s, Global Step: %s", trainer.current_epoch, trainer.global_step)
        logger.warning("Loss value: %s", loss_value)
        if nan_metric_keys:
            logger.warning("Affected metrics: %s", ", ".join(nan_metric_keys))

        if self.print_diagnostics:
            self._print_diagnostics(trainer, pl_module, batch, None)

        if self.debug_on_nan:
            logger.warning("Interactive debugger disabled in production path.")
            logger.warning(
                "   Set monitor.nan_detection.debug_on_nan=false to suppress this notice."
            )

        if self.terminate_on_nan:
            raise ValueError(
                f"{issue_type} detected in training loss at epoch {trainer.current_epoch}"
            )

    def _print_diagnostics(self, trainer, pl_module, batch, outputs):
        """Print detailed diagnostic information."""
        logger.warning("\n%s", "─" * 80)
        logger.warning("DIAGNOSTIC INFORMATION:")
        logger.warning("─" * 80)

        # Batch statistics
        if "image" in batch:
            images = batch["image"]
            logger.warning("\nInput Image Stats:")
            logger.warning("   Shape: %s", images.shape)
            logger.warning("   Min: %.6f, Max: %.6f", images.min().item(), images.max().item())
            logger.warning("   Mean: %.6f, Std: %.6f", images.mean().item(), images.std().item())
            logger.warning("   Contains NaN: %s", torch.isnan(images).any().item())
            logger.warning("   Contains Inf: %s", torch.isinf(images).any().item())

        if "label" in batch:
            labels = batch["label"]
            logger.warning("\nLabel Stats:")
            logger.warning("   Shape: %s", labels.shape)
            logger.warning("   Min: %.6f, Max: %.6f", labels.min().item(), labels.max().item())
            logger.warning("   Unique values: %s", torch.unique(labels).tolist())
            logger.warning("   Contains NaN: %s", torch.isnan(labels).any().item())
            logger.warning("   Contains Inf: %s", torch.isinf(labels).any().item())

        # Check gradients
        if self.check_grads:
            logger.warning("\nGradient Stats:")
            nan_grads = []
            inf_grads = []
            grad_norms = []

            for name, param in pl_module.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    grad_norms.append((name, grad_norm))

                    if torch.isnan(param.grad).any():
                        nan_grads.append(name)
                    if torch.isinf(param.grad).any():
                        inf_grads.append(name)

            if nan_grads:
                logger.warning("   Parameters with NaN gradients: %s", nan_grads[:5])
            if inf_grads:
                logger.warning("   Parameters with Inf gradients: %s", inf_grads[:5])

            # Show largest gradient norms
            grad_norms.sort(key=lambda x: x[1], reverse=True)
            logger.warning("   Top 5 gradient norms:")
            for name, norm in grad_norms[:5]:
                logger.warning("      %s: %.6f", name, norm)

        # Check model parameters
        logger.warning("\nModel Parameter Stats:")
        nan_params = []
        inf_params = []
        for name, param in pl_module.named_parameters():
            if torch.isnan(param).any():
                nan_params.append(name)
            if torch.isinf(param).any():
                inf_params.append(name)

        if nan_params:
            logger.warning("   Parameters with NaN: %s", nan_params)
        if inf_params:
            logger.warning("   Parameters with Inf: %s", inf_params)
        if not nan_params and not inf_params:
            logger.warning("   No NaN/Inf in parameters")

        # Learning rate
        optimizer = trainer.optimizers[0] if trainer.optimizers else None
        lr = None
        if optimizer:
            lr = optimizer.param_groups[0]["lr"]
            logger.warning("\nOptimizer:")
        if lr is not None:
            logger.warning("   Learning rate: %.2e", lr)
        else:
            logger.warning("   Learning rate: N/A (optimizer not initialized)")

        logger.warning("─" * 80)


class EMAWeightsCallback(Callback):
    """
    Maintain exponential moving average (EMA) weights and swap them in for evaluation.
    """

    def __init__(
        self,
        decay: float = 0.999,
        warmup_steps: int = 0,
        validate_with_ema: bool = True,
        device: Optional[str] = None,
        copy_buffers: bool = True,
    ):
        super().__init__()
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.validate_with_ema = validate_with_ema
        self.device = device
        self.copy_buffers = copy_buffers

        self._ema_state: Optional[Dict[str, torch.Tensor]] = None
        self._backup_state: Optional[Dict[str, torch.Tensor]] = None
        self._ema_device: Optional[torch.device] = None
        self._updates: int = 0
        self._using_ema: bool = False

    def on_fit_start(self, trainer, pl_module):
        self._initialize_ema(pl_module)

    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs: Dict[str, Any],
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ):
        if self._ema_state is None:
            self._initialize_ema(pl_module)
        if self._ema_state is None:
            return

        self._updates += 1
        decay = 0.0 if self._updates <= self.warmup_steps else self.decay
        self._update_ema(pl_module, decay)

    def on_validation_epoch_start(self, trainer, pl_module):
        if trainer.sanity_checking or not self.validate_with_ema:
            return
        self._apply_ema_weights(pl_module)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        self._restore_original_weights(pl_module)

    def on_test_epoch_start(self, trainer, pl_module):
        if not self.validate_with_ema:
            return
        self._apply_ema_weights(pl_module)

    def on_test_epoch_end(self, trainer, pl_module):
        self._restore_original_weights(pl_module)

    def on_fit_end(self, trainer, pl_module):
        self._restore_original_weights(pl_module)

    def _initialize_ema(self, pl_module):
        """Create EMA storage on the desired device."""
        device = torch.device(self.device) if self.device is not None else pl_module.device
        self._ema_device = device

        with torch.no_grad():
            self._ema_state = {
                name: tensor.detach().clone().to(device)
                for name, tensor in pl_module.model.state_dict().items()
            }

    def _update_ema(self, pl_module, decay: float):
        """Update EMA weights using the current model parameters."""
        if self._ema_state is None:
            return

        device = self._ema_device or pl_module.device
        with torch.no_grad():
            for name, param in pl_module.model.named_parameters():
                if not param.requires_grad:
                    continue
                self._update_single(name, param.detach(), decay, device, use_decay=True)

            for name, buffer in pl_module.model.named_buffers():
                if not self.copy_buffers:
                    continue
                self._update_single(name, buffer.detach(), decay, device, use_decay=False)

    def _update_single(
        self, name: str, tensor: torch.Tensor, decay: float, device: torch.device, use_decay: bool
    ):
        if self._ema_state is None:
            return

        tensor = tensor.to(device)
        ema_tensor = self._ema_state.get(name)

        if ema_tensor is None:
            self._ema_state[name] = tensor.clone()
            return

        if ema_tensor.device != device:
            ema_tensor = ema_tensor.to(device)
            self._ema_state[name] = ema_tensor

        if not torch.is_floating_point(ema_tensor) or not use_decay:
            self._ema_state[name] = tensor.clone()
            return

        ema_tensor.mul_(decay).add_(tensor, alpha=1.0 - decay)

    def _apply_ema_weights(self, pl_module):
        """Swap EMA weights into the model for evaluation."""
        if self._ema_state is None or self._using_ema:
            return

        with torch.no_grad():
            self._backup_state = {
                name: tensor.detach().clone()
                for name, tensor in pl_module.model.state_dict().items()
            }

            ema_on_device = {
                name: tensor.to(pl_module.device) for name, tensor in self._ema_state.items()
            }
            pl_module.model.load_state_dict(ema_on_device, strict=True)
            self._using_ema = True

    def _restore_original_weights(self, pl_module):
        """Restore the training weights after evaluation."""
        if not self._using_ema or self._backup_state is None:
            return

        with torch.no_grad():
            pl_module.model.load_state_dict(self._backup_state, strict=True)

        self._backup_state = None
        self._using_ema = False


class ValidationReseedingCallback(Callback):
    """
    Callback to reseed validation datasets at the start of each validation epoch.

    Ensures validation samples different patches each epoch while maintaining
    determinism, preventing the model from memorizing fixed validation patches.

    Args:
        base_seed: Base random seed (typically from cfg.system.seed)
        log_fingerprint: If True, log a sampling fingerprint for verification
        log_all_ranks: If True, log from all DDP ranks (otherwise rank 0 only)
        verbose: If True, log detailed information about dataset reseeding
    """

    def __init__(
        self,
        base_seed: int = 0,
        log_fingerprint: bool = True,
        log_all_ranks: bool = False,
        verbose: bool = True,
    ):
        super().__init__()
        self.base_seed = base_seed
        self.log_fingerprint = log_fingerprint
        self.log_all_ranks = log_all_ranks
        self.verbose = verbose

        self._reseeded_count = 0
        self._skipped_count = 0
        self._last_epoch = -1

    def on_validation_epoch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Reseed validation datasets before each validation epoch."""
        if not self.log_all_ranks and trainer.global_rank != 0:
            return

        current_epoch = trainer.current_epoch
        is_sanity_check = trainer.sanity_checking

        if current_epoch == self._last_epoch and not is_sanity_check:
            return

        self._last_epoch = current_epoch

        epoch_type = "SANITY CHECK" if is_sanity_check else f"EPOCH {current_epoch}"
        if self.verbose:
            logger.info(
                "[VAL RESEED] %s | Step %s | Rank %s",
                epoch_type,
                trainer.global_step,
                trainer.global_rank,
            )

        val_dataloaders = self._get_validation_dataloaders(trainer)
        if not val_dataloaders:
            logger.warning("[VAL RESEED SKIPPED] %s | No validation dataloaders", epoch_type)
            return

        total_reseeded = 0
        total_skipped = 0
        skipped_reasons: list[str] = []

        for dl_idx, dataloader in enumerate(val_dataloaders):
            if self.verbose:
                logger.info("[VAL RESEED] Processing DataLoader %s", dl_idx)

            datasets = self._find_all_datasets(dataloader)
            for ds_idx, dataset in enumerate(datasets):
                dataset_info = f"{type(dataset).__name__}@{id(dataset)}"

                if hasattr(dataset, "set_epoch"):
                    try:
                        seed_epoch = -1 if is_sanity_check else current_epoch
                        dataset.set_epoch(seed_epoch, self.base_seed)

                        if self.verbose:
                            logger.info("[VAL RESEED]  Dataset %s: %s", ds_idx, dataset_info)
                            logger.info(
                                "[VAL RESEED]    set_epoch(epoch=%s, base_seed=%s)",
                                seed_epoch,
                                self.base_seed,
                            )

                        total_reseeded += 1

                        if self.log_fingerprint and hasattr(dataset, "get_sampling_fingerprint"):
                            logger.info(
                                "[VAL RESEED]    Fingerprint: %s",
                                dataset.get_sampling_fingerprint(),
                            )

                    except Exception as e:
                        logger.warning(
                            "[VAL RESEED SKIPPED] Dataset %s: %s | Exception: %s",
                            ds_idx,
                            dataset_info,
                            e,
                        )
                        total_skipped += 1
                        skipped_reasons.append(f"Exception: {e}")
                else:
                    logger.warning(
                        "[VAL RESEED SKIPPED] Dataset %s: %s | no set_epoch() method",
                        ds_idx,
                        dataset_info,
                    )
                    total_skipped += 1
                    skipped_reasons.append("no set_epoch() method")

        self._reseeded_count += total_reseeded
        self._skipped_count += total_skipped

        logger.info("[VAL RESEED] Summary for %s:", epoch_type)
        logger.info("[VAL RESEED]   Datasets reseeded: %s", total_reseeded)
        logger.info("[VAL RESEED]   Datasets skipped:  %s", total_skipped)
        if skipped_reasons:
            logger.info("[VAL RESEED]   Skip reasons: %s", ", ".join(set(skipped_reasons)))

    @staticmethod
    def _get_validation_dataloaders(trainer: pl.Trainer) -> List[DataLoader]:
        """Get all validation dataloaders from the trainer."""
        val_dataloaders = trainer.val_dataloaders
        if val_dataloaders is None:
            return []
        if isinstance(val_dataloaders, DataLoader):
            return [val_dataloaders]
        if isinstance(val_dataloaders, list):
            return val_dataloaders
        if hasattr(val_dataloaders, "loaders"):
            loaders = val_dataloaders.loaders
            if isinstance(loaders, dict):
                return list(loaders.values())
            if isinstance(loaders, list):
                return loaders
        return [val_dataloaders]

    def _find_all_datasets(self, dataloader: DataLoader) -> List[Dataset]:
        """Recursively find all leaf datasets in a dataloader."""
        datasets: list[Dataset] = []
        if not hasattr(dataloader, "dataset"):
            return datasets
        self._extract_datasets_recursive(dataloader.dataset, datasets)
        return datasets

    def _extract_datasets_recursive(self, dataset: Any, result: List[Dataset]) -> None:
        """Recursively extract all leaf datasets."""
        if isinstance(dataset, Subset):
            self._extract_datasets_recursive(dataset.dataset, result)
            return
        if isinstance(dataset, ConcatDataset):
            for sub_dataset in dataset.datasets:
                self._extract_datasets_recursive(sub_dataset, result)
            return
        if _HAS_CHAIN_DATASET and isinstance(dataset, ChainDataset):
            for sub_dataset in dataset.datasets:
                self._extract_datasets_recursive(sub_dataset, result)
            return
        result.append(dataset)

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Log final statistics at end of training."""
        if trainer.global_rank != 0:
            return
        logger.info("[VAL RESEED] Final statistics")
        logger.info("[VAL RESEED]   Total datasets reseeded: %s", self._reseeded_count)
        logger.info("[VAL RESEED]   Total datasets skipped:  %s", self._skipped_count)

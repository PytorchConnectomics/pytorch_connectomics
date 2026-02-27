"""
PyTorch Lightning callbacks for PyTorch Connectomics.

Provides callbacks for visualization, checkpointing, and monitoring.
"""

from __future__ import annotations
from typing import Dict, Any, Optional
import pdb
import torch
from pytorch_lightning import Callback

from ...utils.visualizer import Visualizer, get_visualization_mask

__all__ = [
    "VisualizationCallback",
    "NaNDetectionCallback",
    "EMAWeightsCallback",
]


class VisualizationCallback(Callback):
    """
    Lightning callback for TensorBoard visualization.

    Visualizes input images, ground truth, and predictions at the end of each epoch.
    """

    def __init__(self, cfg, max_images: int = 8, num_slices: int = 8, log_every_n_epochs: int = 1):
        """
        Args:
            cfg: Hydra config object
            max_images: Maximum number of images to visualize per batch
            num_slices: Number of consecutive slices to show for 3D volumes
            log_every_n_epochs: Log visualization every N epochs (default: 1)
        """
        super().__init__()
        self.visualizer = Visualizer(cfg, max_images=max_images)
        self.num_slices = num_slices
        self.log_every_n_epochs = log_every_n_epochs
        self.cfg = cfg

        # Store batch for end-of-epoch visualization
        self._last_train_batch = None
        self._last_val_batch = None

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
        if trainer.current_epoch % self.log_every_n_epochs != 0:
            return

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
                label_cpu = cached_batch["label"].cpu()
                pred_cpu = self._to_tensor(pred).cpu()
                mask_cpu = cached_batch.get("mask", None)
                if mask_cpu is not None:
                    mask_cpu = mask_cpu.cpu()

            self._log_visualization(
                image=image_cpu,
                label=label_cpu,
                mask=mask_cpu,
                pred=pred_cpu,
                writer=writer,
                iteration=trainer.current_epoch,
                prefix=prefix,
            )
            if prefix == "train":
                print(f"✓ Saved visualization for epoch {trainer.current_epoch}")
        except Exception as e:
            import traceback

            print(f"{prefix} epoch-end visualization failed: {e}")
            print(f"Error type: {type(e).__name__}")
            if hasattr(e, "__traceback__"):
                print("Traceback:")
                traceback.print_exception(type(e), e, e.__traceback__)

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
            )

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


class NaNDetectionCallback(Callback):
    """
    Lightning callback to detect NaN/Inf values in training loss and trigger debugger.

    This callback monitors the loss value after each training step and:
    - Checks for NaN or Inf values
    - Prints diagnostic information (loss value, batch statistics, gradient norms)
    - Optionally triggers pdb.set_trace() to pause training for debugging
    - Can terminate training or continue with a warning

    Args:
        check_grads: If True, also check for NaN/Inf in model gradients
        check_inputs: If True, also check for NaN/Inf in batch inputs
        debug_on_nan: If True, trigger pdb.set_trace() when NaN is detected
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
        self._last_outputs = None  # Store last outputs for debugging

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

    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs: Dict[str, Any],
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ):
        """Check for NaN/Inf after each training step (backup check)."""
        # This is a backup check - on_after_backward should catch it first

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
        print(f"\n{'=' * 80}")
        print(f"⚠️  {issue_type} DETECTED IN TRAINING LOSS!")
        print(f"{'=' * 80}")
        print(f"Epoch: {trainer.current_epoch}, Global Step: {trainer.global_step}")
        print(f"Loss value: {loss_value}")
        if nan_metric_keys:
            print(f"Affected metrics: {', '.join(nan_metric_keys)}")

        if self.print_diagnostics:
            self._print_diagnostics(trainer, pl_module, batch, None)

        if self.debug_on_nan:
            print("\n🔍 Entering debugger (pdb)...")
            print("Available variables:")
            print("  - trainer: PyTorch Lightning trainer")
            print("  - pl_module: LightningModule (model)")
            print("  - batch: Current batch data")
            print("  - loss_value: The NaN/Inf loss value")
            print("  - nan_metric_keys: List of affected metrics")
            print("\nUseful commands:")
            print(
                "  - Check gradients: [p for n, p in pl_module.named_parameters() "
                "if p.grad is not None]"
            )
            print("  - Check inputs: batch['image'].min(), batch['image'].max()")
            print("  - Continue: 'c' or quit: 'q'")
            print()
            pdb.set_trace()

        if self.terminate_on_nan:
            raise ValueError(
                f"{issue_type} detected in training loss at epoch {trainer.current_epoch}"
            )

    def _print_diagnostics(self, trainer, pl_module, batch, outputs):
        """Print detailed diagnostic information."""
        print(f"\n{'─' * 80}")
        print("📊 DIAGNOSTIC INFORMATION:")
        print(f"{'─' * 80}")

        # Batch statistics
        if "image" in batch:
            images = batch["image"]
            print("\n🖼️  Input Image Stats:")
            print(f"   Shape: {images.shape}")
            print(f"   Min: {images.min().item():.6f}, Max: {images.max().item():.6f}")
            print(f"   Mean: {images.mean().item():.6f}, Std: {images.std().item():.6f}")
            print(f"   Contains NaN: {torch.isnan(images).any().item()}")
            print(f"   Contains Inf: {torch.isinf(images).any().item()}")

        if "label" in batch:
            labels = batch["label"]
            print("\n🎯 Label Stats:")
            print(f"   Shape: {labels.shape}")
            print(f"   Min: {labels.min().item():.6f}, Max: {labels.max().item():.6f}")
            print(f"   Unique values: {torch.unique(labels).tolist()}")
            print(f"   Contains NaN: {torch.isnan(labels).any().item()}")
            print(f"   Contains Inf: {torch.isinf(labels).any().item()}")

        # Check gradients
        if self.check_grads:
            print("\n📉 Gradient Stats:")
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
                print(f"   ⚠️  Parameters with NaN gradients: {nan_grads[:5]}")
            if inf_grads:
                print(f"   ⚠️  Parameters with Inf gradients: {inf_grads[:5]}")

            # Show largest gradient norms
            grad_norms.sort(key=lambda x: x[1], reverse=True)
            print("   Top 5 gradient norms:")
            for name, norm in grad_norms[:5]:
                print(f"      {name}: {norm:.6f}")

        # Check model parameters
        print("\n⚙️  Model Parameter Stats:")
        nan_params = []
        inf_params = []
        for name, param in pl_module.named_parameters():
            if torch.isnan(param).any():
                nan_params.append(name)
            if torch.isinf(param).any():
                inf_params.append(name)

        if nan_params:
            print(f"   ⚠️  Parameters with NaN: {nan_params}")
        if inf_params:
            print(f"   ⚠️  Parameters with Inf: {inf_params}")
        if not nan_params and not inf_params:
            print("   ✓ No NaN/Inf in parameters")

        # Learning rate
        optimizer = trainer.optimizers[0] if trainer.optimizers else None
        if optimizer:
            lr = optimizer.param_groups[0]["lr"]
            print("\n📚 Optimizer:")
        print(f"   Learning rate: {lr:.2e}")

        print(f"{'─' * 80}\n")


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

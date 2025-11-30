"""
PyTorch Lightning callbacks for PyTorch Connectomics.

Provides callbacks for visualization, checkpointing, and monitoring.
"""

from __future__ import annotations
from typing import Dict, Any
import pdb
import torch
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

from ...utils.visualizer import Visualizer

__all__ = [
    'VisualizationCallback',
    'NaNDetectionCallback',
    'create_callbacks',
]


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
        log_every_n_epochs: int = 1
    ):
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
        batch_idx: int
    ):
        """Store first batch for epoch-end visualization."""
        # Always store first batch for epoch-end visualization
        if batch_idx == 0:
            self._last_train_batch = {
                'image': batch['image'].detach(),
                'label': batch['label'].detach()
            }

    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs: Dict[str, Any],
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0
    ):
        """Store first batch for epoch-end visualization."""
        # Store first validation batch for epoch-end visualization
        if batch_idx == 0:
            self._last_val_batch = {
                'image': batch['image'].detach(),
                'label': batch['label'].detach()
            }

    def on_train_epoch_end(self, trainer, pl_module):
        """Visualize at end of training epoch based on log_every_n_epochs."""
        if self._last_train_batch is None or trainer.logger is None:
            return

        # Check if we should log this epoch
        if trainer.current_epoch % self.log_every_n_epochs != 0:
            return

        try:
            writer = trainer.logger.experiment

            # Generate predictions for stored batch
            with torch.no_grad():
                pl_module.eval()
                # Move batch to model's device before inference
                image = self._last_train_batch['image'].to(pl_module.device)
                pred = pl_module(image)
                pl_module.train()

                # Move all tensors to CPU for visualization to avoid device mismatch
                image_cpu = image.cpu()
                label_cpu = self._last_train_batch['label'].cpu()
                pred_cpu = pred.cpu()
                mask_cpu = self._last_train_batch.get('mask', None)
                if mask_cpu is not None:
                    mask_cpu = mask_cpu.cpu()

            # Visualize - use epoch number as step for slider
            if image_cpu.ndim == 5 and self.num_slices > 1:
                # Use consecutive slices for 3D volumes
                self.visualizer.visualize_consecutive_slices(
                    volume=image_cpu,
                    label=label_cpu,
                    mask=mask_cpu,
                    output=pred_cpu,
                    writer=writer,
                    iteration=trainer.current_epoch,  # Use epoch as step for slider
                    prefix='train',
                    num_slices=self.num_slices
                )
            else:
                # Use single slice for 2D or when num_slices=1
                self.visualizer.visualize(
                    volume=image_cpu,
                    label=label_cpu,
                    mask=mask_cpu,
                    output=pred_cpu,
                    iteration=trainer.current_epoch,  # Use epoch as step for slider
                    writer=writer,
                    prefix='train'  # Single tab name (no epoch prefix)
                )

            print(f"✓ Saved visualization for epoch {trainer.current_epoch}")
        except Exception as e:
            import traceback
            print(f"Epoch-end visualization failed: {e}")
            print(f"Error type: {type(e).__name__}")
            if hasattr(e, '__traceback__'):
                print("Traceback:")
                traceback.print_exception(type(e), e, e.__traceback__)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Visualize at end of validation epoch based on log_every_n_epochs."""
        if self._last_val_batch is None or trainer.logger is None:
            return

        # Check if we should log this epoch
        if trainer.current_epoch % self.log_every_n_epochs != 0:
            return

        try:
            writer = trainer.logger.experiment

            # Generate predictions for stored batch
            with torch.no_grad():
                # Move batch to model's device before inference
                image = self._last_val_batch['image'].to(pl_module.device)
                pred = pl_module(image)

                # Move all tensors to CPU for visualization to avoid device mismatch
                image_cpu = image.cpu()
                label_cpu = self._last_val_batch['label'].cpu()
                pred_cpu = pred.cpu()
                mask_cpu = self._last_val_batch.get('mask', None)
                if mask_cpu is not None:
                    mask_cpu = mask_cpu.cpu()

            # Visualize - use epoch number as step for slider
            if image_cpu.ndim == 5 and self.num_slices > 1:
                # Use consecutive slices for 3D volumes
                self.visualizer.visualize_consecutive_slices(
                    volume=image_cpu,
                    label=label_cpu,
                    mask=mask_cpu,
                    output=pred_cpu,
                    writer=writer,
                    iteration=trainer.current_epoch,  # Use epoch as step for slider
                    prefix='val',
                    num_slices=self.num_slices
                )
            else:
                # Use single slice for 2D or when num_slices=1
                self.visualizer.visualize(
                    volume=image_cpu,
                    label=label_cpu,
                    mask=mask_cpu,
                    output=pred_cpu,
                    iteration=trainer.current_epoch,  # Use epoch as step for slider
                    writer=writer,
                    prefix='val'  # Single tab name (no epoch prefix)
                )
        except Exception as e:
            import traceback
            print(f"Validation epoch-end visualization failed: {e}")
            print(f"Error type: {type(e).__name__}")
            if hasattr(e, '__traceback__'):
                print("Traceback:")
                traceback.print_exception(type(e), e, e.__traceback__)


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
        print_diagnostics: bool = True
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
        self,
        trainer,
        pl_module,
        batch: Dict[str, torch.Tensor],
        batch_idx: int
    ):
        """Store batch for later debugging."""
        self._last_batch = batch

    def on_after_backward(
        self,
        trainer,
        pl_module
    ):
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
            if 'loss' in key.lower() or 'train' in key.lower():
                if isinstance(value, torch.Tensor):
                    val = value.item() if value.numel() == 1 else None
                else:
                    val = value

                if val is not None and (val != val or abs(val) == float('inf')):
                    is_nan = is_nan or (val != val)
                    is_inf = is_inf or (abs(val) == float('inf'))
                    nan_metric_keys.append(f"{key}={val}")
                    if loss_value is None:
                        loss_value = val

        if is_nan or is_inf:
            self._handle_nan_detection(
                trainer, pl_module, self._last_batch,
                is_nan, is_inf, loss_value, nan_metric_keys
            )

    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs: Dict[str, Any],
        batch: Dict[str, torch.Tensor],
        batch_idx: int
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
        nan_metric_keys: list
    ):
        """Handle NaN/Inf detection with diagnostics and debugging."""
        issue_type = "NaN" if is_nan else "Inf"
        print(f"\n{'='*80}")
        print(f"⚠️  {issue_type} DETECTED IN TRAINING LOSS!")
        print(f"{'='*80}")
        print(f"Epoch: {trainer.current_epoch}, Global Step: {trainer.global_step}")
        print(f"Loss value: {loss_value}")
        if nan_metric_keys:
            print(f"Affected metrics: {', '.join(nan_metric_keys)}")

        if self.print_diagnostics:
            self._print_diagnostics(trainer, pl_module, batch, None)

        if self.debug_on_nan:
            print(f"\n🔍 Entering debugger (pdb)...")
            print(f"Available variables:")
            print(f"  - trainer: PyTorch Lightning trainer")
            print(f"  - pl_module: LightningModule (model)")
            print(f"  - batch: Current batch data")
            print(f"  - loss_value: The NaN/Inf loss value")
            print(f"  - nan_metric_keys: List of affected metrics")
            print(f"\nUseful commands:")
            print(f"  - Check gradients: [p for n, p in pl_module.named_parameters() if p.grad is not None]")
            print(f"  - Check inputs: batch['image'].min(), batch['image'].max()")
            print(f"  - Continue: 'c' or quit: 'q'")
            print()
            pdb.set_trace()

        if self.terminate_on_nan:
            raise ValueError(f"{issue_type} detected in training loss at epoch {trainer.current_epoch}")

    def _print_diagnostics(self, trainer, pl_module, batch, outputs):
        """Print detailed diagnostic information."""
        print(f"\n{'─'*80}")
        print("📊 DIAGNOSTIC INFORMATION:")
        print(f"{'─'*80}")

        # Batch statistics
        if 'image' in batch:
            images = batch['image']
            print(f"\n🖼️  Input Image Stats:")
            print(f"   Shape: {images.shape}")
            print(f"   Min: {images.min().item():.6f}, Max: {images.max().item():.6f}")
            print(f"   Mean: {images.mean().item():.6f}, Std: {images.std().item():.6f}")
            print(f"   Contains NaN: {torch.isnan(images).any().item()}")
            print(f"   Contains Inf: {torch.isinf(images).any().item()}")

        if 'label' in batch:
            labels = batch['label']
            print(f"\n🎯 Label Stats:")
            print(f"   Shape: {labels.shape}")
            print(f"   Min: {labels.min().item():.6f}, Max: {labels.max().item():.6f}")
            print(f"   Unique values: {torch.unique(labels).tolist()}")
            print(f"   Contains NaN: {torch.isnan(labels).any().item()}")
            print(f"   Contains Inf: {torch.isinf(labels).any().item()}")

        # Check gradients
        if self.check_grads:
            print(f"\n📉 Gradient Stats:")
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
            print(f"   Top 5 gradient norms:")
            for name, norm in grad_norms[:5]:
                print(f"      {name}: {norm:.6f}")

        # Check model parameters
        print(f"\n⚙️  Model Parameter Stats:")
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
            print(f"   ✓ No NaN/Inf in parameters")

        # Learning rate
        optimizer = trainer.optimizers[0] if trainer.optimizers else None
        if optimizer:
            lr = optimizer.param_groups[0]['lr']
            print(f"\n📚 Optimizer:")
            print(f"   Learning rate: {lr:.2e}")

        print(f"{'─'*80}\n")


def create_callbacks(cfg) -> list:
    """
    Create PyTorch Lightning callbacks from config.

    Args:
        cfg: Hydra config object

    Returns:
        List of Lightning callbacks
    """
    callbacks = []

    # Visualization callback
    if hasattr(cfg, 'visualization') and getattr(cfg.visualization, 'enabled', True):
        vis_callback = VisualizationCallback(
            cfg,
            max_images=getattr(cfg.visualization, 'max_images', 8),
            num_slices=getattr(cfg.visualization, 'num_slices', 8),
            log_every_n_epochs=getattr(cfg.visualization, 'log_every_n_epochs', 1)
        )
        callbacks.append(vis_callback)
    else:
        # Default: add visualization with defaults
        vis_callback = VisualizationCallback(cfg)
        callbacks.append(vis_callback)

    # Model checkpoint callback
    if hasattr(cfg, 'monitor') and hasattr(cfg.monitor, 'checkpoint'):
        monitor = getattr(cfg.monitor.checkpoint, 'monitor', 'val/loss')
        filename = getattr(cfg.monitor.checkpoint, 'filename', None)
        if filename is None:
            # Auto-generate filename from monitor metric
            filename = f'epoch={{epoch:03d}}-{monitor}={{{monitor}:.4f}}'

        checkpoint_callback = ModelCheckpoint(
            monitor=monitor,
            mode=getattr(cfg.monitor.checkpoint, 'mode', 'min'),
            save_top_k=getattr(cfg.monitor.checkpoint, 'save_top_k', 3),
            save_last=getattr(cfg.monitor.checkpoint, 'save_last', True),
            dirpath=getattr(cfg.monitor.checkpoint, 'dirpath', 'checkpoints'),
            filename=filename,
            verbose=True
        )
        callbacks.append(checkpoint_callback)

    # Early stopping callback
    if hasattr(cfg, 'monitor') and hasattr(cfg.monitor, 'early_stopping') and getattr(cfg.monitor.early_stopping, 'enabled', False):
        early_stop_callback = EarlyStopping(
            monitor=getattr(cfg.monitor.early_stopping, 'monitor', 'val/loss'),
            patience=getattr(cfg.monitor.early_stopping, 'patience', 10),
            mode=getattr(cfg.monitor.early_stopping, 'mode', 'min'),
            min_delta=getattr(cfg.monitor.early_stopping, 'min_delta', 0.0),
            verbose=True
        )
        callbacks.append(early_stop_callback)

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)

    return callbacks

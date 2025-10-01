"""
PyTorch Lightning callbacks for PyTorch Connectomics.

Provides callbacks for visualization, checkpointing, and monitoring.
"""

from __future__ import annotations
from typing import Optional, Dict, Any
import torch
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

from ..utils.visualizer import Visualizer

__all__ = [
    'VisualizationCallback',
    'create_callbacks',
]


class VisualizationCallback(Callback):
    """
    Lightning callback for TensorBoard visualization.

    Visualizes input images, ground truth, and predictions during training/validation.
    """

    def __init__(self, cfg, max_images: int = 8, log_every_n_steps: int = 100):
        """
        Args:
            cfg: Hydra config object
            max_images: Maximum number of images to visualize per batch
            log_every_n_steps: Visualize every N training steps
        """
        super().__init__()
        self.visualizer = Visualizer(cfg, max_images=max_images)
        self.log_every_n_steps = log_every_n_steps
        self.cfg = cfg

    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs: Dict[str, Any],
        batch: Dict[str, torch.Tensor],
        batch_idx: int
    ):
        """Visualize training batch."""
        # Only visualize every N steps and only first batch of the step
        if trainer.global_step % self.log_every_n_steps != 0:
            return

        if batch_idx != 0:
            return

        if trainer.logger is None:
            return

        try:
            writer = trainer.logger.experiment

            # Get predictions from outputs
            if isinstance(outputs, dict):
                pred = outputs.get('pred', outputs.get('logits', outputs.get('loss')))
            else:
                pred = None

            if pred is None or not isinstance(pred, torch.Tensor):
                return

            # Visualize
            self.visualizer.visualize(
                volume=batch['image'],
                label=batch['label'],
                output=pred,
                iteration=trainer.global_step,
                writer=writer,
                prefix='train'
            )
        except Exception as e:
            # Don't break training if visualization fails
            print(f"Visualization failed: {e}")

    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs: Dict[str, Any],
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0
    ):
        """Visualize validation batch."""
        # Only visualize first validation batch
        if batch_idx != 0:
            return

        if trainer.logger is None:
            return

        try:
            writer = trainer.logger.experiment

            # Get predictions
            if isinstance(outputs, dict):
                pred = outputs.get('pred', outputs.get('logits'))
            else:
                pred = None

            if pred is None or not isinstance(pred, torch.Tensor):
                return

            # Visualize
            self.visualizer.visualize(
                volume=batch['image'],
                label=batch['label'],
                output=pred,
                iteration=trainer.global_step,
                writer=writer,
                prefix='val'
            )
        except Exception as e:
            print(f"Validation visualization failed: {e}")


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
            log_every_n_steps=getattr(cfg.visualization, 'log_every_n_steps', 100)
        )
        callbacks.append(vis_callback)
    else:
        # Default: add visualization with defaults
        vis_callback = VisualizationCallback(cfg)
        callbacks.append(vis_callback)

    # Model checkpoint callback
    if hasattr(cfg, 'checkpoint'):
        checkpoint_callback = ModelCheckpoint(
            monitor=getattr(cfg.checkpoint, 'monitor', 'val/loss'),
            mode=getattr(cfg.checkpoint, 'mode', 'min'),
            save_top_k=getattr(cfg.checkpoint, 'save_top_k', 3),
            save_last=getattr(cfg.checkpoint, 'save_last', True),
            dirpath=getattr(cfg.checkpoint, 'dirpath', 'checkpoints'),
            filename=getattr(cfg.checkpoint, 'filename', 'model-{epoch:02d}-{val_loss:.4f}'),
            verbose=True
        )
        callbacks.append(checkpoint_callback)

    # Early stopping callback
    if hasattr(cfg, 'early_stopping') and cfg.early_stopping.enabled:
        early_stop_callback = EarlyStopping(
            monitor=getattr(cfg.early_stopping, 'monitor', 'val/loss'),
            patience=getattr(cfg.early_stopping, 'patience', 10),
            mode=getattr(cfg.early_stopping, 'mode', 'min'),
            min_delta=getattr(cfg.early_stopping, 'min_delta', 0.0),
            verbose=True
        )
        callbacks.append(early_stop_callback)

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)

    return callbacks

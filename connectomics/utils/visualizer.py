"""
Visualization utilities for PyTorch Connectomics.

Updated for PyTorch Lightning + Hydra config system.
Provides TensorBoard visualization of training progress, predictions, and metrics.
"""

from __future__ import annotations
from typing import Optional, Dict, Any, List
import torch
import torch.nn as nn
import torchvision.utils as vutils
import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

__all__ = ['Visualizer', 'LightningVisualizer']


class Visualizer:
    """
    TensorBoard visualizer for displaying predictions during training.

    Compatible with both legacy YACS configs and modern Hydra configs.
    """

    def __init__(self, cfg=None, max_images: int = 16):
        """
        Args:
            cfg: Config object (Hydra Config or YACS CfgNode)
            max_images: Maximum number of images to show
        """
        self.cfg = cfg
        self.max_images = max_images
        self.semantic_colors = {}

        # Initialize color maps for semantic segmentation
        self._init_color_maps()

    def _init_color_maps(self):
        """Initialize random colors for semantic segmentation visualization."""
        # Default color map for binary segmentation
        self.semantic_colors['default'] = torch.tensor([
            [0.0, 0.0, 0.0],  # Background (black)
            [1.0, 1.0, 1.0],  # Foreground (white)
        ])

    def visualize(
        self,
        volume: torch.Tensor,
        label: torch.Tensor,
        output: torch.Tensor,
        iteration: int,
        writer: SummaryWriter,
        prefix: str = 'train',
        **kwargs
    ):
        """
        Visualize input, target, and prediction.

        Args:
            volume: Input image (B, C, D, H, W) or (B, C, H, W)
            label: Ground truth (B, C, D, H, W) or (B, C, H, W)
            output: Model prediction (B, C, D, H, W) or (B, C, H, W)
            iteration: Current iteration/step
            writer: TensorBoard SummaryWriter
            prefix: Prefix for logging (train/val/test)
        """
        if not HAS_TENSORBOARD:
            return

        # Prepare data
        volume = self._prepare_volume(volume)
        label = self._prepare_volume(label)
        output = self._prepare_volume(output)

        # Create visualization grid
        self._visualize_grid(
            volume, label, output,
            writer, iteration, prefix
        )

    def _prepare_volume(self, vol: torch.Tensor) -> torch.Tensor:
        """Prepare volume for visualization (handle 3D -> 2D conversion)."""
        if vol.ndim == 5:  # 3D: (B, C, D, H, W)
            # Take middle slice
            mid_slice = vol.shape[2] // 2
            vol = vol[:, :, mid_slice, :, :]

        # Ensure (B, C, H, W)
        return vol[:self.max_images]

    def _visualize_grid(
        self,
        volume: torch.Tensor,
        label: torch.Tensor,
        output: torch.Tensor,
        writer: SummaryWriter,
        iteration: int,
        prefix: str
    ):
        """Create and log visualization grid."""
        # Normalize to [0, 1]
        volume = self._normalize(volume)
        label = self._normalize(label)
        output = torch.sigmoid(output)  # Apply sigmoid for visualization

        # Handle multi-channel predictions
        if output.shape[1] > 1:
            # For multi-class, take argmax
            output = torch.argmax(output, dim=1, keepdim=True).float()

        if label.shape[1] > 1:
            label = torch.argmax(label, dim=1, keepdim=True).float()

        # Expand single channel to RGB
        if volume.shape[1] == 1:
            volume = volume.repeat(1, 3, 1, 1)
        if label.shape[1] == 1:
            label = label.repeat(1, 3, 1, 1)
        if output.shape[1] == 1:
            output = output.repeat(1, 3, 1, 1)

        # Stack: [volume, prediction, ground_truth]
        grid = torch.cat([volume, output, label], dim=0)

        # Create grid visualization
        grid_img = vutils.make_grid(
            grid, nrow=self.max_images, normalize=True, scale_each=True
        )

        # Log to tensorboard
        writer.add_image(f'{prefix}/visualization', grid_img, iteration)

    def _normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalize tensor to [0, 1] range."""
        tensor = tensor.detach().cpu()
        min_val = tensor.min()
        max_val = tensor.max()
        if max_val > min_val:
            tensor = (tensor - min_val) / (max_val - min_val)
        return tensor

    def visualize_consecutive_slices(
        self,
        volume: torch.Tensor,
        label: torch.Tensor,
        output: torch.Tensor,
        writer: SummaryWriter,
        iteration: int,
        prefix: str = 'train',
        num_slices: int = 8
    ):
        """
        Visualize consecutive slices from 3D volume.

        Args:
            volume: Input volume (B, C, D, H, W)
            label: Ground truth (B, C, D, H, W)
            output: Prediction (B, C, D, H, W)
            writer: TensorBoard writer
            iteration: Current iteration
            prefix: Logging prefix
            num_slices: Number of consecutive slices to show
        """
        if volume.ndim != 5:
            return  # Not 3D

        # Take first batch item
        volume = volume[0]  # (C, D, H, W)
        label = label[0]
        output = output[0]

        # Select middle slices
        depth = volume.shape[1]
        start_idx = max(0, depth // 2 - num_slices // 2)
        end_idx = min(depth, start_idx + num_slices)

        # Extract slices
        vol_slices = volume[:, start_idx:end_idx, :, :]  # (C, num_slices, H, W)
        lab_slices = label[:, start_idx:end_idx, :, :]
        out_slices = output[:, start_idx:end_idx, :, :]

        # Reshape to (num_slices, C, H, W)
        vol_slices = vol_slices.permute(1, 0, 2, 3)
        lab_slices = lab_slices.permute(1, 0, 2, 3)
        out_slices = out_slices.permute(1, 0, 2, 3)

        # Visualize
        self._visualize_grid(
            vol_slices, lab_slices, out_slices,
            writer, iteration, f'{prefix}_slices'
        )


class LightningVisualizer:
    """
    Lightning-compatible visualizer.

    Designed to work with PyTorch Lightning callbacks.
    """

    def __init__(self, cfg, max_images: int = 16):
        """
        Args:
            cfg: Hydra Config object
            max_images: Maximum number of images to visualize
        """
        self.visualizer = Visualizer(cfg, max_images)
        self.cfg = cfg

    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs: Dict[str, Any],
        batch: Dict[str, torch.Tensor],
        batch_idx: int
    ):
        """Called at the end of training batch."""
        if not self._should_visualize(trainer, batch_idx):
            return

        if trainer.logger is None:
            return

        # Get tensorboard writer
        writer = trainer.logger.experiment

        # Visualize
        self.visualizer.visualize(
            volume=batch['image'],
            label=batch['label'],
            output=outputs.get('pred', outputs.get('logits')),
            iteration=trainer.global_step,
            writer=writer,
            prefix='train'
        )

    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs: Dict[str, Any],
        batch: Dict[str, torch.Tensor],
        batch_idx: int
    ):
        """Called at the end of validation batch."""
        if batch_idx != 0:  # Only visualize first batch
            return

        if trainer.logger is None:
            return

        writer = trainer.logger.experiment

        self.visualizer.visualize(
            volume=batch['image'],
            label=batch['label'],
            output=outputs.get('pred', outputs.get('logits')),
            iteration=trainer.global_step,
            writer=writer,
            prefix='val'
        )

    def _should_visualize(self, trainer, batch_idx: int) -> bool:
        """Determine if should visualize this batch."""
        # Visualize every N steps
        log_every_n_steps = getattr(self.cfg.training, 'vis_every_n_steps', 100)
        return trainer.global_step % log_every_n_steps == 0 and batch_idx == 0


# Legacy compatibility
def create_visualizer(cfg, **kwargs):
    """Factory function for creating visualizer (backward compatible)."""
    return Visualizer(cfg, **kwargs)

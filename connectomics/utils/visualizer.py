"""
Visualization utilities for PyTorch Connectomics.

Updated for PyTorch Lightning + Hydra config system.
Provides TensorBoard visualization of training progress, predictions, and metrics.
"""

from __future__ import annotations
from typing import Optional, Dict, Any, List
import torch
import torchvision.utils as vutils
import numpy as np
from skimage.color import label2rgb

try:
    from torch.utils.tensorboard import SummaryWriter

    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

__all__ = ["Visualizer", "LightningVisualizer"]


class Visualizer:
    """
    TensorBoard visualizer for displaying predictions during training.

    Compatible with Hydra configs.
    """

    def __init__(self, cfg=None, max_images: int = 16):
        """
        Args:
            cfg: Hydra Config object
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
        self.semantic_colors["default"] = torch.tensor(
            [
                [0.0, 0.0, 0.0],  # Background (black)
                [1.0, 1.0, 1.0],  # Foreground (white)
            ]
        )

    def visualize(
        self,
        volume: torch.Tensor,
        label: torch.Tensor,
        mask: Optional[torch.Tensor],
        output: torch.Tensor,
        iteration: int,
        writer: SummaryWriter,
        prefix: str = "train",
        channel_mode: str = "argmax",
        selected_channels: Optional[List[int]] = None,
        **kwargs,
    ):
        """
        Visualize input, target, and prediction.

        Args:
            volume: Input image (B, C, D, H, W) or (B, C, H, W)
            label: Ground truth (B, C, D, H, W) or (B, C, H, W)
            mask: Optional mask (B, C, D, H, W) or (B, C, H, W), shown in cyan if provided
            output: Model prediction (B, C, D, H, W) or (B, C, H, W)
            iteration: Current iteration/step
            writer: TensorBoard SummaryWriter
            prefix: Prefix for logging (train/val/test)
            channel_mode: How to handle multi-channel output ('argmax', 'all', 'selected')
            selected_channels: List of channel indices to show (only used when channel_mode='selected')
        """
        if not HAS_TENSORBOARD:
            return

        # Prepare data
        volume = self._prepare_volume(volume)
        label = self._prepare_volume(label)
        mask = self._prepare_volume(mask)
        output = self._prepare_volume(output)

        # Create visualization grid
        self._visualize_grid(
            volume,
            label,
            mask,
            output,
            writer,
            iteration,
            prefix,
            channel_mode,
            selected_channels,
        )

    def _prepare_volume(self, vol: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Prepare volume for visualization (handle 3D -> 2D conversion)."""
        # Handle None volumes (e.g., when mask is not provided)
        if vol is None:
            return None

        # Move to CPU first to avoid device mismatches
        vol = vol.detach().cpu()

        if vol.ndim == 5:  # 3D: (B, C, D, H, W)
            # Take middle slice
            mid_slice = vol.shape[2] // 2
            vol = vol[:, :, mid_slice, :, :]

        # Ensure (B, C, H, W)
        return vol[: self.max_images]

    def _visualize_grid(
        self,
        volume: torch.Tensor,
        label: torch.Tensor,
        mask: Optional[torch.Tensor],
        output: torch.Tensor,
        writer: SummaryWriter,
        iteration: int,
        prefix: str,
        channel_mode: str = "argmax",
        selected_channels: Optional[List[int]] = None,
    ):
        """Create and log visualization grid."""
        # Normalize to [0, 1]
        volume = self._normalize(volume)
        label = self._normalize(label)
        mask = self._normalize(mask)

        # Process output based on channel mode
        if channel_mode == "selected" and selected_channels is not None:
            output_viz = self._process_output_channels(output, channel_mode, selected_channels)
        else:
            output_viz = output  # Show all output channels as-is
        # output_viz = self._normalize(output_viz)

        # For labels, only apply channel selection if in 'selected' mode
        # Otherwise show all channels as-is for proper comparison
        if channel_mode == "selected" and selected_channels is not None:
            label_viz = self._process_output_channels(label, channel_mode, selected_channels)
        else:
            label_viz = label  # Show all label channels as-is
        label_viz = self._normalize(label_viz)

        # Create visualizations
        self._log_visualization(volume, label_viz, mask, output_viz, writer, iteration, prefix)

    def _process_output_channels(
        self,
        tensor: torch.Tensor,
        channel_mode: str,
        selected_channels: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """Process multi-channel output based on visualization mode."""
        if tensor.shape[1] == 1:
            # Single channel - apply sigmoid for binary
            return torch.sigmoid(tensor)

        if channel_mode == "argmax":
            # Take argmax for multi-class
            if tensor.shape[1] > 1:
                # Apply softmax then argmax
                tensor = torch.softmax(tensor, dim=1)
                tensor = torch.argmax(tensor, dim=1, keepdim=True).float()
            return tensor

        elif channel_mode == "all":
            # Show all channels separately
            return tensor

        elif channel_mode == "selected":
            # Show only selected channels
            if selected_channels is None:
                selected_channels = list(
                    range(min(3, tensor.shape[1]))
                )  # Default to first 3 channels

            # Validate channel indices
            valid_channels = [i for i in selected_channels if 0 <= i < tensor.shape[1]]
            if not valid_channels:
                valid_channels = [0]  # Fallback to first channel

            return tensor[:, valid_channels]

        else:
            raise ValueError(
                f"Unknown channel_mode: {channel_mode}. Must be 'argmax', 'all', or 'selected'"
            )

    def _log_visualization(
        self,
        volume: torch.Tensor,
        label: torch.Tensor,
        mask: Optional[torch.Tensor],
        output: torch.Tensor,
        writer: SummaryWriter,
        iteration: int,
        prefix: str,
    ):
        """Log different types of visualizations based on channel count."""
        # Input volume (always show as grayscale/RGB)
        if volume.shape[1] == 1:
            volume_rgb = volume.repeat(1, 3, 1, 1)
        else:
            volume_rgb = volume[:, :3]  # Take first 3 channels

        # Single channel visualization (argmax mode)
        if output.shape[1] == 1:
            self._log_single_channel_viz(volume_rgb, label, mask, output, writer, iteration, prefix)
        else:
            # Multi-channel visualization
            self._log_multi_channel_viz(volume_rgb, label, mask, output, writer, iteration, prefix)

    def _log_single_channel_viz(
        self,
        volume: torch.Tensor,
        label: torch.Tensor,
        mask: Optional[torch.Tensor],
        output: torch.Tensor,
        writer: SummaryWriter,
        iteration: int,
        prefix: str,
    ):
        """Log single channel visualization (argmax mode)."""
        # For single-channel label, show as grayscale; for multi-channel, use label2rgb
        if label.shape[1] == 1:
            # Show single-channel label as grayscale
            label_rgb = torch.cat([label, label, label], dim=1)
        else:
            # Multi-channel label: use label2rgb conversion
            label_rgb = self._label2rgb(label)

        # For single-channel output, show as grayscale (not red)
        if output.shape[1] == 1:
            # Show single-channel prediction as grayscale
            output_rgb = torch.cat([output, output, output], dim=1)
        else:
            output_rgb = output[:, :3]

        # Prepare mask visualization if present
        if mask is not None and mask.numel() > 0:
            # Convert mask to RGB (show as cyan/blue for visibility)
            if mask.shape[1] == 1:
                # Single-channel mask: show as cyan (0, 1, 1) where mask is active
                mask_rgb = torch.cat(
                    [torch.zeros_like(mask), mask, mask],  # R=0  # G=mask  # B=mask
                    dim=1,
                )
            else:
                mask_rgb = mask[:, :3]

            # Stack: [input, prediction, ground_truth, mask]
            grid = torch.cat([volume, output_rgb, label_rgb, mask_rgb], dim=0)
        else:
            # Stack: [input, prediction, ground_truth]
            # This makes it easy to see: white=FP, colored=FN, mixed=TP
            grid = torch.cat([volume, output_rgb, label_rgb], dim=0)

        # Create grid visualization
        grid_img = vutils.make_grid(
            grid, nrow=min(8, self.max_images), normalize=True, scale_each=True
        )

        # Log to tensorboard
        writer.add_image(f"{prefix}/visualization", grid_img, iteration)

    def _log_multi_channel_viz(
        self,
        volume: torch.Tensor,
        label: torch.Tensor,
        mask: Optional[torch.Tensor],
        output: torch.Tensor,
        writer: SummaryWriter,
        iteration: int,
        prefix: str,
    ):
        """Log multi-channel visualization."""
        # Limit to max_images for all tensors
        volume = volume[: self.max_images]
        label = label[: self.max_images]
        output = output[: self.max_images]
        if mask is not None and mask.numel() > 0:
            mask = mask[: self.max_images]

        # Show input
        writer.add_image(
            f"{prefix}/input",
            vutils.make_grid(volume, nrow=min(8, self.max_images), normalize=True, scale_each=True),
            iteration,
        )

        # Show each output channel
        for i in range(min(output.shape[1], 12)):  # Increased limit to 12 channels
            channel_img = output[:, i : i + 1].repeat(1, 3, 1, 1)  # Convert to RGB
            writer.add_image(
                f"{prefix}/output_channel_{i}",
                vutils.make_grid(
                    channel_img,
                    nrow=min(8, self.max_images),
                    normalize=True,
                    scale_each=True,
                ),
                iteration,
            )

        # Show each label channel
        for i in range(min(label.shape[1], 12)):  # Increased limit to 12 channels
            channel_img = label[:, i : i + 1].repeat(1, 3, 1, 1)  # Convert to RGB
            writer.add_image(
                f"{prefix}/label_channel_{i}",
                vutils.make_grid(
                    channel_img,
                    nrow=min(8, self.max_images),
                    normalize=True,
                    scale_each=True,
                ),
                iteration,
            )

        # Show mask if present
        if mask is not None and mask.numel() > 0:
            for i in range(min(mask.shape[1], 12)):  # Show up to 12 mask channels
                # Show mask in cyan for better visibility
                mask_channel = mask[:, i : i + 1]
                mask_rgb = torch.cat(
                    [
                        torch.zeros_like(mask_channel),  # R=0
                        mask_channel,  # G=mask
                        mask_channel,  # B=mask
                    ],
                    dim=1,
                )
                writer.add_image(
                    f"{prefix}/mask_channel_{i}",
                    vutils.make_grid(
                        mask_rgb,
                        nrow=min(8, self.max_images),
                        normalize=True,
                        scale_each=True,
                    ),
                    iteration,
                )

    def _label2rgb(self, label: torch.Tensor) -> torch.Tensor:
        """Convert multi-channel label to RGB visualization using skimage.label2rgb."""
        # Convert to numpy for skimage processing
        label_np = label.detach().cpu().numpy()
        batch_size, channels, *spatial_dims = label_np.shape

        # Process each sample in the batch
        rgb_batch = []
        for b in range(batch_size):
            # For multi-channel labels, we need to convert to a single label image
            # Take the argmax across channels to get the most prominent label per pixel
            if channels > 1:
                label_single = np.argmax(label_np[b], axis=0)
            else:
                label_single = label_np[b, 0]

            # Use skimage.label2rgb for robust color mapping
            rgb_single = label2rgb(label_single, bg_label=0, kind="overlay")

            # Convert back to torch tensor and transpose to (C, H, W)
            rgb_tensor = torch.from_numpy(rgb_single).permute(2, 0, 1).float()
            rgb_batch.append(rgb_tensor)

        # Stack batch and keep on CPU (all visualization tensors should be on CPU)
        rgb = torch.stack(rgb_batch, dim=0)
        return rgb

    def _normalize(self, tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Normalize tensor to [0, 1] range per-channel."""
        # Handle None tensors (e.g., when mask is not provided)
        if tensor is None:
            return None

        tensor = tensor.detach().cpu()

        # Normalize each channel independently to preserve relative values
        if tensor.ndim >= 2 and tensor.shape[1] > 1:
            # Multi-channel: normalize each channel independently
            normalized = []
            for c in range(tensor.shape[1]):
                channel = tensor[:, c : c + 1]
                min_val = channel.min()
                max_val = channel.max()
                if max_val > min_val:
                    channel = (channel - min_val) / (max_val - min_val)
                normalized.append(channel)
            tensor = torch.cat(normalized, dim=1)
        else:
            # Single channel or no channel dimension: normalize globally
            min_val = tensor.min()
            max_val = tensor.max()
            if max_val > min_val:
                tensor = (tensor - min_val) / (max_val - min_val)

        return tensor

    def visualize_consecutive_slices(
        self,
        volume: torch.Tensor,
        label: torch.Tensor,
        mask: Optional[torch.Tensor],
        output: torch.Tensor,
        writer: SummaryWriter,
        iteration: int,
        prefix: str = "train",
        num_slices: int = 8,
        channel_mode: str = "argmax",
        selected_channels: Optional[List[int]] = None,
    ):
        """
        Visualize consecutive slices from 3D volume.

        Args:
            volume: Input volume (B, C, D, H, W)
            label: Ground truth (B, C, D, H, W)
            mask: Optional mask (B, C, D, H, W), shown in cyan if provided
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
        if mask is not None:
            mask = mask[0]
        else:
            mask = None

        output = output[0]

        # Select middle slices
        depth = volume.shape[1]
        start_idx = max(0, depth // 2 - num_slices // 2)
        end_idx = min(depth, start_idx + num_slices)

        # Extract slices
        vol_slices = volume[:, start_idx:end_idx, :, :]  # (C, num_slices, H, W)
        lab_slices = label[:, start_idx:end_idx, :, :]
        out_slices = output[:, start_idx:end_idx, :, :]
        mask_slices = mask[:, start_idx:end_idx, :, :] if mask is not None else None
        # Reshape to (num_slices, C, H, W)
        vol_slices = vol_slices.permute(1, 0, 2, 3)
        lab_slices = lab_slices.permute(1, 0, 2, 3)
        out_slices = out_slices.permute(1, 0, 2, 3)
        mask_slices = mask_slices.permute(1, 0, 2, 3) if mask_slices is not None else None
        # Visualize
        self._visualize_grid(
            vol_slices,
            lab_slices,
            mask_slices,
            out_slices,
            writer,
            iteration,
            f"{prefix}_slices",
            channel_mode,
            selected_channels,
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
        batch_idx: int,
    ):
        """Called at the end of training batch."""
        if not self._should_visualize(trainer, batch_idx):
            return

        if trainer.logger is None:
            return

        # Get tensorboard writer
        writer = trainer.logger.experiment

        # Get visualization options from config
        channel_mode = getattr(self.cfg.monitor.logging.images, "channel_mode", "argmax")
        selected_channels = getattr(self.cfg.monitor.logging.images, "selected_channels", None)

        # Visualize
        self.visualizer.visualize(
            volume=batch["image"],
            label=batch["label"],
            mask=batch.get("mask", None),  # Include mask if present
            output=outputs.get("pred", outputs.get("logits")),
            iteration=trainer.global_step,
            writer=writer,
            prefix="train",
            channel_mode=channel_mode,
            selected_channels=selected_channels,
        )

    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs: Dict[str, Any],
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ):
        """Called at the end of validation batch."""
        if batch_idx != 0:  # Only visualize first batch
            return

        if trainer.logger is None:
            return

        writer = trainer.logger.experiment

        # Get visualization options from config
        channel_mode = getattr(self.cfg.monitor.logging.images, "channel_mode", "argmax")
        selected_channels = getattr(self.cfg.monitor.logging.images, "selected_channels", None)

        self.visualizer.visualize(
            volume=batch["image"],
            label=batch["label"],
            mask=batch.get("mask", None),  # Include mask if present
            output=outputs.get("pred", outputs.get("logits")),
            iteration=trainer.global_step,
            writer=writer,
            prefix="val",
            channel_mode=channel_mode,
            selected_channels=selected_channels,
        )

    def _should_visualize(self, trainer, batch_idx: int) -> bool:
        """Determine if should visualize this batch."""
        # Check if images are enabled
        if not getattr(self.cfg.monitor.logging.images, "enabled", True):
            return False

        # Visualize every N steps (check optimization config)
        log_every_n_steps = getattr(self.cfg.optimization, "vis_every_n_steps", 100)

        return trainer.global_step % log_every_n_steps == 0 and batch_idx == 0

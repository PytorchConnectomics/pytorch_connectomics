"""
PyTorch Lightning module for PyTorch Connectomics.

This module implements the Lightning interface with:
- Hydra/OmegaConf configuration
- MONAI native models
- Modern loss functions
- Automatic distributed training, mixed precision, checkpointing

The implementation delegates to specialized modules:
- deep_supervision.py: Deep supervision and multi-task learning
- inference.py: Sliding window inference and test-time augmentation
- debugging.py: NaN detection and debugging utilities
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional, Union
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from omegaconf import DictConfig
import torchmetrics

# Import existing components
from ..models import build_model
from ..models.loss import create_loss
from ..models.solver import build_optimizer, build_lr_scheduler
from ..config import Config

# Import new modular components
from .deep_supervision import DeepSupervisionHandler, match_target_to_output
from .inference import (
    InferenceManager,
    apply_postprocessing,
    apply_decode_mode,
    resolve_output_filenames,
    write_outputs,
)
from .debugging import DebugManager


class ConnectomicsModule(pl.LightningModule):
    """
    PyTorch Lightning module for connectomics tasks.

    This module provides automatic training features including:
    - Distributed training
    - Mixed precision
    - Gradient accumulation
    - Checkpointing
    - Logging
    - Learning rate scheduling

    Args:
        cfg: Hydra Config object or OmegaConf DictConfig
        model: Optional pre-built model (if None, builds from config)
    """

    def __init__(
        self,
        cfg: Union[Config, DictConfig],
        model: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(ignore=['model'])

        # Build model
        self.model = model if model is not None else self._build_model(cfg)

        # Build loss functions
        self.loss_functions = self._build_losses(cfg)
        self.loss_weights = cfg.model.loss_weights if hasattr(cfg.model, 'loss_weights') else [1.0] * len(self.loss_functions)

        # Enable inline NaN detection (can be disabled via config)
        self.enable_nan_detection = getattr(cfg.model, 'enable_nan_detection', True)
        self.debug_on_nan = getattr(cfg.model, 'debug_on_nan', True)

        # Activation clamping to prevent inf (can be configured)
        self.clamp_activations = getattr(cfg.model, 'clamp_activations', False)
        self.clamp_min = getattr(cfg.model, 'clamp_min', -10.0)
        self.clamp_max = getattr(cfg.model, 'clamp_max', 10.0)

        # Initialize specialized handlers
        self.deep_supervision_handler = DeepSupervisionHandler(
            cfg=cfg,
            loss_functions=self.loss_functions,
            loss_weights=self.loss_weights,
            enable_nan_detection=self.enable_nan_detection,
            debug_on_nan=self.debug_on_nan,
        )

        self.inference_manager = InferenceManager(
            cfg=cfg,
            model=self.model,
            forward_fn=self.forward,
        )

        self.debug_manager = DebugManager(model=self.model)

        # Test metrics (initialized lazily during test mode if specified in config)
        self.test_jaccard = None
        self.test_dice = None
        self.test_accuracy = None
        self.test_adapted_rand = None  # Adapted Rand error (instance segmentation metric)
        self.test_adapted_rand_results = []  # Store per-batch results for averaging

        # Prediction saving state
        self._prediction_save_counter = 0  # Track number of samples saved

    def _build_model(self, cfg) -> nn.Module:
        """Build model from configuration."""
        return build_model(cfg)

    def _build_losses(self, cfg) -> nn.ModuleList:
        """Build loss functions from configuration."""
        loss_names = cfg.model.loss_functions if hasattr(cfg.model, 'loss_functions') else ['DiceLoss']
        loss_kwargs_list = cfg.model.loss_kwargs if hasattr(cfg.model, 'loss_kwargs') else [{}] * len(loss_names)

        losses = nn.ModuleList()
        for loss_name, kwargs in zip(loss_names, loss_kwargs_list):
            loss_fn = create_loss(loss_name, **kwargs)
            losses.append(loss_fn)

        return losses

    def _setup_test_metrics(self):
        """Initialize test metrics based on inference config."""
        if not hasattr(self.cfg, 'inference') or not hasattr(self.cfg.inference, 'evaluation'):
            return

        # Check if evaluation is enabled
        if not getattr(self.cfg.inference.evaluation, 'enabled', False):
            return

        metrics = getattr(self.cfg.inference.evaluation, 'metrics', None)
        if metrics is None:
            return

        num_classes = self.cfg.model.out_channels if hasattr(self.cfg.model, 'out_channels') else 2

        # Create only the specified metrics
        if 'jaccard' in metrics:
            if num_classes == 1:
                # Binary segmentation - use binary metrics
                self.test_jaccard = torchmetrics.JaccardIndex(task='binary').to(self.device)
            else:
                # Multi-class segmentation
                self.test_jaccard = torchmetrics.JaccardIndex(task='multiclass', num_classes=num_classes).to(self.device)
        if 'dice' in metrics:
            if num_classes == 1:
                # Binary segmentation - use binary metrics
                self.test_dice = torchmetrics.Dice(task='binary').to(self.device)
            else:
                # Multi-class segmentation
                self.test_dice = torchmetrics.Dice(num_classes=num_classes, average='macro').to(self.device)
        if 'accuracy' in metrics:
            if num_classes == 1:
                # Binary segmentation - use binary metrics
                self.test_accuracy = torchmetrics.Accuracy(task='binary').to(self.device)
            else:
                # Multi-class segmentation
                self.test_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes).to(self.device)
        if 'adapted_rand' in metrics:
            # Adapted Rand error for instance segmentation
            # This is a custom metric that needs manual computation
            self.test_adapted_rand = True  # Flag to enable adapted_rand computation
            self.test_adapted_rand_results = []  # Store per-batch results

    def _setup_sliding_window_inferer(self):
        """Initialize MONAI's SlidingWindowInferer based on config."""
        self.sliding_inferer = None

        if not hasattr(self.cfg, 'inference'):
            return

        # For 2D models with do_2d=True, disable sliding window inference
        if getattr(self.cfg.data, 'do_2d', False):
            warnings.warn(
                "Sliding-window inference disabled for 2D models with do_2d=True. "
                "Using direct inference instead.",
                UserWarning,
            )
            return

        roi_size = self._resolve_inferer_roi_size()
        if roi_size is None:
            warnings.warn(
                "Sliding-window inference disabled: unable to determine ROI size. "
                "Set inference.window_size or model.output_size in the config.",
                UserWarning,
            )
            return

        overlap = self._resolve_inferer_overlap(roi_size)
        # Use system.inference.batch_size as default, fall back to sliding_window.sw_batch_size if specified
        system_batch_size = getattr(self.cfg.system.inference, 'batch_size', 1)
        config_sw_batch_size = getattr(self.cfg.inference.sliding_window, 'sw_batch_size', None)
        sw_batch_size = max(1, int(config_sw_batch_size if config_sw_batch_size is not None else system_batch_size))
        mode = getattr(self.cfg.inference.sliding_window, 'blending', 'gaussian')
        sigma_scale = float(getattr(self.cfg.inference.sliding_window, 'sigma_scale', 0.125))
        padding_mode = getattr(self.cfg.inference.sliding_window, 'padding_mode', 'constant')

        self.sliding_inferer = SlidingWindowInferer(
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            overlap=overlap,
            mode=mode,
            sigma_scale=sigma_scale,
            padding_mode=padding_mode,
            progress=True,
        )

        print(
            "  Sliding-window inference configured: "
            f"roi_size={roi_size}, overlap={overlap}, sw_batch={sw_batch_size}, "
            f"mode={mode}, sigma_scale={sigma_scale}, padding={padding_mode}"
        )

    def _resolve_inferer_roi_size(self) -> Optional[Tuple[int, ...]]:
        """Determine the ROI size for sliding-window inference."""
        if hasattr(self.cfg, 'inference') and hasattr(self.cfg.inference, 'sliding_window'):
            window_size = getattr(self.cfg.inference.sliding_window, 'window_size', None)
            if window_size:
                return tuple(int(v) for v in window_size)

        if hasattr(self.cfg, 'model') and hasattr(self.cfg.model, 'output_size'):
            output_size = getattr(self.cfg.model, 'output_size', None)
            if output_size:
                roi_size = tuple(int(v) for v in output_size)
                # For 2D models with do_2d=True, convert to 3D ROI size
                if getattr(self.cfg.data, 'do_2d', False) and len(roi_size) == 2:
                    roi_size = (1,) + roi_size  # Add depth dimension
                return roi_size

        if hasattr(self.cfg, 'data') and hasattr(self.cfg.data, 'patch_size'):
            patch_size = getattr(self.cfg.data, 'patch_size', None)
            if patch_size:
                roi_size = tuple(int(v) for v in patch_size)
                # For 2D models with do_2d=True, convert to 3D ROI size
                if getattr(self.cfg.data, 'do_2d', False) and len(roi_size) == 2:
                    roi_size = (1,) + roi_size  # Add depth dimension
                return roi_size

        return None

    def _resolve_inferer_overlap(self, roi_size: Tuple[int, ...]) -> Union[float, Tuple[float, ...]]:
        """Resolve overlap parameter using inference config."""
        if not hasattr(self.cfg, 'inference') or not hasattr(self.cfg.inference, 'sliding_window'):
            return 0.5

        overlap = getattr(self.cfg.inference.sliding_window, 'overlap', None)
        if overlap is not None:
            if isinstance(overlap, (list, tuple)):
                return tuple(float(max(0.0, min(o, 0.99))) for o in overlap)
            return float(max(0.0, min(overlap, 0.99)))

        stride = getattr(self.cfg.inference, 'stride', None)
        if stride:
            values: List[float] = []
            for size, step in zip(roi_size, stride):
                if size <= 0:
                    values.append(0.0)
                    continue
                ratio = 1.0 - float(step) / float(size)
                values.append(float(max(0.0, min(ratio, 0.99))))
            if len(set(values)) == 1:
                return values[0]
            return tuple(values)

        return 0.5

    def _extract_main_output(self, outputs: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Extract the primary segmentation logits from model outputs."""
        if isinstance(outputs, dict):
            if 'output' not in outputs:
                raise KeyError("Expected key 'output' in model outputs for deep supervision.")
            return outputs['output']
        return outputs

    def _sliding_window_predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """Wrapper used by MONAI inferer to obtain primary model predictions."""
        outputs = self.forward(inputs)
        result = self._extract_main_output(outputs)

        # Filter channels during inference to reduce memory during accumulation
        # This is applied before the sliding window aggregates results
        if hasattr(self.cfg, 'inference') and hasattr(self.cfg.inference, 'sliding_window'):
            save_channels = getattr(self.cfg.inference.sliding_window, 'save_channels', None)
            if save_channels is not None:
                # Select only specified channels (e.g., [0, 6] for foreground + SDT)
                result = result[:, save_channels, ...]

        return result

    def _apply_tta_preprocessing(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply activation and channel selection before TTA ensemble.

        Supports per-channel activations via channel_activations config:
        Format: [[start_ch, end_ch, 'activation'], ...]
        Examples:
          - [[0, 2, 'softmax'], [2, 3, 'sigmoid']]  # Softmax over channels 0-1, sigmoid for channel 2
          - [[0, 1, 'sigmoid'], [1, 2, 'sigmoid']]  # Sigmoid for channels 0 and 1 separately

        Args:
            tensor: Raw predictions (B, C, D, H, W)

        Returns:
            Preprocessed tensor for TTA ensembling
        """
        if not hasattr(self.cfg, 'inference'):
            return tensor

        # Check for per-channel activations first (new approach)
        channel_activations = getattr(self.cfg.inference.test_time_augmentation, 'channel_activations', None)

        if channel_activations is not None:
            # Apply different activations to different channels
            activated_channels = []
            for config_entry in channel_activations:
                # Only support new format: [start_ch, end_ch, activation]
                if len(config_entry) != 3:
                    raise ValueError(
                        f"Invalid channel_activations entry: {config_entry}. "
                        f"Expected [start_ch, end_ch, activation] with 3 elements. "
                        f"Example: [[0, 2, 'softmax'], [2, 3, 'sigmoid']]"
                    )

                start_ch, end_ch, act = config_entry
                channel_tensor = tensor[:, start_ch:end_ch, ...]

                if act == 'sigmoid':
                    channel_tensor = torch.sigmoid(channel_tensor)
                elif act == 'scale_sigmoid':
                    # Scaled sigmoid for BANIS: sigmoid(0.2 * x)
                    # This avoids numerical issues with high-confidence fp16 predictions
                    channel_tensor = torch.sigmoid(0.2 * channel_tensor)
                elif act == 'tanh':
                    channel_tensor = torch.tanh(channel_tensor)
                elif act == 'softmax':
                    # Apply softmax across the channel dimension
                    if end_ch - start_ch > 1:
                        channel_tensor = torch.softmax(channel_tensor, dim=1)
                    else:
                        warnings.warn(
                            f"Softmax activation for single channel ({start_ch}:{end_ch}) is not meaningful. Skipping.",
                            UserWarning,
                        )
                elif act is None or (isinstance(act, str) and act.lower() == 'none'):
                    # No activation (keep as is)
                    pass
                else:
                    raise ValueError(
                        f"Unknown activation '{act}' for channels {start_ch}:{end_ch}. "
                        f"Supported: 'sigmoid', 'scale_sigmoid', 'softmax', 'tanh', None"
                    )

                activated_channels.append(channel_tensor)

            # Concatenate all channels back together
            tensor = torch.cat(activated_channels, dim=1)

        # Get TTA-specific channel selection or fall back to output_channel
        tta_channel = getattr(self.cfg.inference.test_time_augmentation, 'select_channel', None)
        if tta_channel is None:
            tta_channel = getattr(self.cfg.inference, 'output_channel', None)

        # Apply channel selection
        if tta_channel is not None:
            if isinstance(tta_channel, int):
                if tta_channel == -1:
                    # -1 means all channels
                    pass
                else:
                    # Single channel selection
                    tensor = tensor[:, tta_channel:tta_channel+1, ...]
            elif isinstance(tta_channel, (list, tuple)):
                # Multiple channel selection
                tensor = tensor[:, list(tta_channel), ...]

        return tensor

    def _predict_with_tta(self, images: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Perform test-time augmentation using flips and ensemble predictions.

        Args:
            images: Input volume (B, C, D, H, W) or (B, D, H, W) or (D, H, W)
            mask: Optional mask to multiply with predictions after ensemble (B, C, D, H, W) or (B, 1, D, H, W)

        Returns:
            Averaged predictions from all TTA variants, optionally masked
        """
        # Normalize input to 5D tensor (B, C, D, H, W)
        if images.ndim == 3:
            # (D, H, W) -> (1, 1, D, H, W)
            images = images.unsqueeze(0).unsqueeze(0)
            warnings.warn(
                f"Input shape {images.shape} (D, H, W) automatically expanded to (1, 1, D, H, W)",
                UserWarning,
            )
        elif images.ndim == 4:
            # (B, D, H, W) -> (B, 1, D, H, W)
            images = images.unsqueeze(1)
            warnings.warn(
                f"Input shape (B, D, H, W) automatically expanded to (B, 1, D, H, W)",
                UserWarning,
            )
        elif images.ndim != 5:
            raise ValueError(
                f"TTA requires 3D, 4D, or 5D input tensor. Got {images.ndim}D tensor with shape {images.shape}. "
                f"Expected shapes: (D, H, W), (B, D, H, W), or (B, C, D, H, W)"
            )

        # For 2D models with do_2d=True, squeeze the depth dimension if present
        if getattr(self.cfg.data, 'do_2d', False) and images.size(2) == 1:  # [B, C, 1, H, W] -> [B, C, H, W]
            images = images.squeeze(2)

        # Get TTA configuration (default to no augmentation if not configured)
        if hasattr(self.cfg, 'inference') and hasattr(self.cfg.inference, 'test_time_augmentation'):
            tta_flip_axes_config = getattr(self.cfg.inference.test_time_augmentation, 'flip_axes', None)
        else:
            tta_flip_axes_config = None  # No config = no augmentation, just forward pass

        # Handle different tta_flip_axes configurations
        if tta_flip_axes_config is None:
            # null: No augmentation, but still apply tta_act and tta_channel (no ensemble)
            if self.sliding_inferer is not None:
                pred = self.sliding_inferer(inputs=images, network=self._sliding_window_predict)
            else:
                pred = self._sliding_window_predict(images)
            
            # Apply TTA preprocessing (activation + channel selection) even without augmentation
            ensemble_result = self._apply_tta_preprocessing(pred)
        else:
            if tta_flip_axes_config == 'all' or tta_flip_axes_config == []:
                # "all" or []: All flips (all combinations of spatial axes)
                # Determine spatial axes based on data dimensions
                if images.dim() == 5:  # 3D data: [B, C, D, H, W]
                    spatial_axes = [1, 2, 3]  # [D, H, W]
                elif images.dim() == 4:  # 2D data: [B, C, H, W]
                    spatial_axes = [1, 2]  # [H, W]
                else:
                    raise ValueError(f"Unsupported data dimensions: {images.dim()}")
                
                # Generate all combinations of spatial axes
                tta_flip_axes = [[]]  # No flip baseline
                for r in range(1, len(spatial_axes) + 1):
                    from itertools import combinations
                    for combo in combinations(spatial_axes, r):
                        tta_flip_axes.append(list(combo))
            elif isinstance(tta_flip_axes_config, (list, tuple)):
                # Custom list: Add no-flip baseline + user-specified flips
                tta_flip_axes = [[]] + list(tta_flip_axes_config)
            else:
                raise ValueError(
                    f"Invalid tta_flip_axes: {tta_flip_axes_config}. "
                    f"Expected 'all' (8 flips), null (no aug), or list of flip axes."
                )

            # Apply TTA with flips, preprocessing, and ensembling
            predictions = []

            for flip_axes in tta_flip_axes:
                # Apply flip augmentation
                if flip_axes:
                    x_aug = Flip(spatial_axis=flip_axes)(images)
                else:
                    x_aug = images

                # Inference with sliding window
                if self.sliding_inferer is not None:
                    pred = self.sliding_inferer(
                        inputs=x_aug,
                        network=self._sliding_window_predict,
                    )
                else:
                    pred = self._sliding_window_predict(x_aug)

                # Invert flip for prediction
                if flip_axes:
                    pred = Flip(spatial_axis=flip_axes)(pred)

                # Apply TTA preprocessing (activation + channel selection) if configured
                # Note: This is applied BEFORE ensembling for probability-space averaging
                pred_processed = self._apply_tta_preprocessing(pred)

                predictions.append(pred_processed)

            # Ensemble predictions based on configured mode
            ensemble_mode = getattr(self.cfg.inference.test_time_augmentation, 'ensemble_mode', 'mean')
            stacked_preds = torch.stack(predictions, dim=0)

            if ensemble_mode == 'mean':
                ensemble_result = stacked_preds.mean(dim=0)
            elif ensemble_mode == 'min':
                ensemble_result = stacked_preds.min(dim=0)[0]  # min returns (values, indices)
            elif ensemble_mode == 'max':
                ensemble_result = stacked_preds.max(dim=0)[0]  # max returns (values, indices)
            else:
                raise ValueError(f"Unknown TTA ensemble mode: {ensemble_mode}. Use 'mean', 'min', or 'max'.")
        
        # Apply mask after ensemble if requested
        apply_mask = getattr(self.cfg.inference.test_time_augmentation, 'apply_mask', False)
        if apply_mask and mask is not None:
            # Ensure mask has the same shape as ensemble_result
            # mask can be (B, C, D, H, W) with C matching channels, or (B, 1, D, H, W) to broadcast
            if mask.shape != ensemble_result.shape:
                # If mask is (B, 1, D, H, W), it will broadcast across channels
                if mask.shape[1] == 1 and mask.shape[0] == ensemble_result.shape[0]:
                    # Broadcast across all channels
                    pass
                elif mask.shape[1] != ensemble_result.shape[1]:
                    warnings.warn(
                        f"Mask shape {mask.shape} does not match ensemble result shape {ensemble_result.shape}. "
                        f"Expected mask with C={ensemble_result.shape[1]} or C=1 channels. Skipping mask application.",
                        UserWarning,
                    )
                    return ensemble_result

            # Multiply each channel by corresponding mask channel (or broadcast if mask has 1 channel)
            ensemble_result = ensemble_result * mask

        return ensemble_result

    def _apply_postprocessing(self, data: np.ndarray) -> np.ndarray:
        """
        Apply postprocessing transformations to predictions.

        This method applies (in order):
        1. Binary postprocessing (morphological operations, connected components filtering)
        2. Scaling (intensity_scale or output_scale): Multiply predictions by scale factor
        3. Dtype conversion (intensity_dtype or output_dtype): Convert to target dtype with clamping

        Args:
            data: Numpy array of predictions (B, C, D, H, W) or (B, D, H, W)

        Returns:
            Postprocessed predictions with applied transformations
        """
        if not hasattr(self.cfg, 'inference') or not hasattr(self.cfg.inference, 'postprocessing'):
            return data

        postprocessing = self.cfg.inference.postprocessing

        # Step 1: Apply binary postprocessing if configured
        binary_config = getattr(postprocessing, 'binary', None)
        if binary_config is not None and getattr(binary_config, 'enabled', False):
            from connectomics.decoding.postprocess import apply_binary_postprocessing

            # Process each sample in batch
            # Handle both 4D (B, C, H, W) for 2D data and 5D (B, C, D, H, W) for 3D data
            if data.ndim == 4:
                # 2D data: (B, C, H, W)
                batch_size = data.shape[0]
            elif data.ndim == 5:
                # 3D data: (B, C, D, H, W)
                batch_size = data.shape[0]
            elif data.ndim == 3:
                # Single 3D volume: (C, D, H, W) or (D, H, W) - add batch dimension
                batch_size = 1
                data = data[np.newaxis, ...]  # (1, C, D, H, W) or (1, D, H, W)
            elif data.ndim == 2:
                # Single 2D image: (H, W) - add batch and channel dimensions
                batch_size = 1
                data = data[np.newaxis, np.newaxis, ...]  # (1, 1, H, W)
            else:
                batch_size = 1

            # Ensure we have at least 4D: (B, ...) where ... can be (C, H, W) for 2D or (C, D, H, W) for 3D
            results = []
            for batch_idx in range(batch_size):
                sample = data[batch_idx]  # (C, H, W) for 2D or (C, D, H, W) for 3D

                # Extract foreground probability (always use first channel if channel dimension exists)
                if sample.ndim == 4:  # (C, D, H, W) - 3D with channel
                    foreground_prob = sample[0]  # Use first channel -> (D, H, W)
                elif sample.ndim == 3:
                    # Could be (C, H, W) for 2D or (D, H, W) for 3D without channel
                    # If first dim is small (<=4), assume it's channel (2D), otherwise depth (3D)
                    if sample.shape[0] <= 4:
                        foreground_prob = sample[0]  # (C, H, W) -> use first channel -> (H, W)
                    else:
                        foreground_prob = sample  # (D, H, W) - already single channel
                elif sample.ndim == 2:  # (H, W) - 2D single channel
                    foreground_prob = sample
                else:
                    foreground_prob = sample

                # Apply binary postprocessing
                processed = apply_binary_postprocessing(foreground_prob, binary_config)

                # Expand dims to maintain shape consistency with original sample structure
                if sample.ndim == 4:  # (C, D, H, W) -> processed is (D, H, W)
                    processed = processed[np.newaxis, ...]  # (1, D, H, W)
                elif sample.ndim == 3 and sample.shape[0] <= 4:  # (C, H, W) -> processed is (H, W)
                    processed = processed[np.newaxis, ...]  # (1, H, W) 
                # else: processed is already correct shape (D, H, W) or (H, W)

                results.append(processed)

            # Stack results back into batch
            data = np.stack(results, axis=0)

        # Step 2: Apply scaling if configured
        intensity_scale = getattr(postprocessing, 'intensity_scale', None)
        if intensity_scale is not None:
            data = data * intensity_scale

        # Step 3: Apply dtype conversion if configured
        target_dtype_str = getattr(postprocessing, 'intensity_dtype', None)

        if target_dtype_str is not None and target_dtype_str != 'float32':
            # Map string dtype to numpy dtype
            dtype_map = {
                'uint8': np.uint8,
                'int8': np.int8,
                'uint16': np.uint16,
                'int16': np.int16,
                'uint32': np.uint32,
                'int32': np.int32,
                'float16': np.float16,
                'float32': np.float32,
                'float64': np.float64,
            }

            if target_dtype_str not in dtype_map:
                warnings.warn(
                    f"Unknown dtype '{target_dtype_str}'. Supported: {list(dtype_map.keys())}. "
                    f"Keeping float32.",
                    UserWarning,
                )
                return data

            target_dtype = dtype_map[target_dtype_str]

            # Clamp to valid range before conversion for integer types
            if target_dtype_str == 'uint8':
                data = np.clip(data, 0, 255)
            elif target_dtype_str == 'int8':
                data = np.clip(data, -128, 127)
            elif target_dtype_str == 'uint16':
                data = np.clip(data, 0, 65535)
            elif target_dtype_str == 'int16':
                data = np.clip(data, -32768, 32767)
            elif target_dtype_str == 'uint32':
                data = np.clip(data, 0, 4294967295)
            elif target_dtype_str == 'int32':
                data = np.clip(data, -2147483648, 2147483647)

            data = data.astype(target_dtype)

        return data

    def _apply_decode_mode(self, data: np.ndarray) -> np.ndarray:
        """
        Apply decode mode transformations to convert probability maps to instance segmentation.

        Args:
            data: Numpy array of predictions (B, C, D, H, W) or (C, D, H, W)

        Returns:
            Decoded segmentation mask(s)
        """
        if not hasattr(self.cfg, 'inference'):
            return data

        # Access decoding config directly from inference
        decode_modes = getattr(self.cfg.inference, 'decoding', None)

        if not decode_modes:
            return data

        # Import decoding functions
        from connectomics.decoding import (
            decode_binary_thresholding,
            decode_binary_cc,
            decode_binary_watershed,
            decode_binary_contour_cc,
            decode_binary_contour_watershed,
            decode_binary_contour_distance_watershed,
            decode_affinity_cc,
        )

        # Map function names to actual functions
        decode_fn_map = {
            'binary_thresholding': decode_binary_thresholding,
            'decode_binary_thresholding': decode_binary_thresholding,
            'decode_binary_cc': decode_binary_cc,
            'decode_binary_watershed': decode_binary_watershed,
            'decode_binary_contour_cc': decode_binary_contour_cc,
            'decode_binary_contour_watershed': decode_binary_contour_watershed,
            'decode_binary_contour_distance_watershed': decode_binary_contour_distance_watershed,
            'decode_affinity_cc': decode_affinity_cc,
        }

        # Process each sample in batch
        # Handle both 4D (B, C, H, W) for 2D data and 5D (B, C, D, H, W) for 3D data
        if data.ndim == 4:
            # 2D data: (B, C, H, W)
            batch_size = data.shape[0]
        elif data.ndim == 5:
            # 3D data: (B, C, D, H, W)
            batch_size = data.shape[0]
        else:
            # Single sample: add batch dimension
            batch_size = 1
            if data.ndim == 3:
                data = data[np.newaxis, ...]  # (C, H, W) -> (1, C, H, W)
            elif data.ndim == 2:
                data = data[np.newaxis, np.newaxis, ...]  # (H, W) -> (1, 1, H, W)

        results = []
        for batch_idx in range(batch_size):
            sample = data[batch_idx]  # (C, H, W) for 2D or (C, D, H, W) for 3D

            # Apply each decode mode sequentially
            for decode_cfg in decode_modes:
                fn_name = decode_cfg.name if hasattr(decode_cfg, 'name') else decode_cfg.get('name')
                kwargs = decode_cfg.kwargs if hasattr(decode_cfg, 'kwargs') else decode_cfg.get('kwargs', {})
                
                # Ensure kwargs is a mutable dict (convert from OmegaConf if needed)
                if hasattr(kwargs, 'items'):
                    kwargs = dict(kwargs)
                else:
                    kwargs = {}

                if fn_name not in decode_fn_map:
                    warnings.warn(
                        f"Unknown decode function '{fn_name}'. Available: {list(decode_fn_map.keys())}. "
                        f"Skipping this decode mode.",
                        UserWarning,
                    )
                    continue

                decode_fn = decode_fn_map[fn_name]

                try:
                    # Apply decoding function
                    sample = decode_fn(sample, **kwargs)
                    # Ensure output has channel dimension for potential chaining
                    if sample.ndim == 3:
                        sample = sample[np.newaxis, ...]  # (1, D, H, W)
                except Exception as e:
                    warnings.warn(
                        f"Error applying decode function '{fn_name}': {e}. "
                        f"Skipping this decode mode.",
                        UserWarning,
                    )
                    continue

            results.append(sample)

        # Stack results back into batch
        # Always preserve batch dimension, even for batch_size=1
        decoded = np.stack(results, axis=0)
        return decoded

    def _resolve_output_filenames(self, batch: Dict[str, Any]) -> List[str]:
        """
        Extract and resolve filenames from batch metadata.

        Args:
            batch: Batch dictionary containing metadata and images

        Returns:
            List of resolved filenames (without extension)
        """
        # Determine batch size from images
        images = batch.get('image')
        if images is not None:
            batch_size = images.shape[0]
        else:
            # Fallback: try to infer from metadata
            batch_size = 1

        meta = batch.get('image_meta_dict')
        filenames: List[Optional[str]] = []

        # Handle different metadata structures
        if isinstance(meta, list):
            # Multiple metadata dicts (one per sample in batch)
            for idx, meta_item in enumerate(meta):
                if isinstance(meta_item, dict):
                    filename = meta_item.get('filename_or_obj')
                    if filename is not None:
                        filenames.append(filename)
            # Update batch_size from metadata if we have a list
            batch_size = max(batch_size, len(filenames))
        elif isinstance(meta, dict):
            # Single metadata dict
            meta_filenames = meta.get('filename_or_obj')
            if isinstance(meta_filenames, (list, tuple)):
                filenames = [f for f in meta_filenames if f is not None]
            elif meta_filenames is not None:
                filenames = [meta_filenames]
            # Update batch_size from metadata
            if len(filenames) > 0:
                batch_size = max(batch_size, len(filenames))

        resolved_names: List[str] = []
        for idx in range(batch_size):
            if idx < len(filenames) and filenames[idx]:
                resolved_names.append(Path(str(filenames[idx])).stem)
            else:
                # Generate fallback filename - this shouldn't happen if metadata is preserved correctly
                resolved_names.append(f"volume_{self.global_step}_{idx}")
        
        # Always return exactly batch_size filenames
        if len(resolved_names) < batch_size:
            print(f"  WARNING: _resolve_output_filenames - Only {len(resolved_names)} filenames but batch_size is {batch_size}, padding with fallback names")
            while len(resolved_names) < batch_size:
                resolved_names.append(f"volume_{self.global_step}_{len(resolved_names)}")

        return resolved_names

    def _write_outputs(
        self,
        predictions: np.ndarray,
        filenames: List[str],
        suffix: str = "prediction"
    ) -> None:
        """
        Persist predictions to disk.

        Args:
            predictions: Numpy array of predictions to save (B, C, D, H, W) or (B, D, H, W)
            filenames: List of filenames (without extension) for each sample in batch
            suffix: Suffix for output filename (default: "prediction")
        """
        if not hasattr(self.cfg, 'inference'):
            return

        # Access output_path from nested data config
        output_dir_value = None
        if hasattr(self.cfg.inference, 'data') and hasattr(self.cfg.inference.data, 'output_path'):
            output_dir_value = self.cfg.inference.data.output_path
        if not output_dir_value:
            return

        output_dir = Path(output_dir_value)
        output_dir.mkdir(parents=True, exist_ok=True)

        from connectomics.data.io import write_hdf5

        # Get output transpose from postprocessing config
        output_transpose = []
        if hasattr(self.cfg.inference, 'postprocessing'):
            output_transpose = getattr(self.cfg.inference.postprocessing, 'output_transpose', [])

        # Determine actual batch size from predictions
        # Handle both batched (B, ...) and unbatched (...) predictions
        if predictions.ndim >= 4:
            # Has batch dimension: (B, C, D, H, W) or (B, C, H, W) or (B, D, H, W)
            actual_batch_size = predictions.shape[0]
        elif predictions.ndim == 3:
            # Could be batched 2D data (B, H, W) or single 3D volume (D, H, W)
            # Check if first dimension matches number of filenames -> it's batched 2D data
            if len(filenames) > 0 and predictions.shape[0] == len(filenames):
                # Batched 2D data: (B, H, W) where B matches number of filenames
                actual_batch_size = predictions.shape[0]
            else:
                # Single 3D volume: (D, H, W) - treat as batch_size=1
                actual_batch_size = 1
                predictions = predictions[np.newaxis, ...]  # Add batch dimension
        elif predictions.ndim == 2:
            # Single 2D image: (H, W) - treat as batch_size=1
            actual_batch_size = 1
            predictions = predictions[np.newaxis, ...]  # Add batch dimension
        else:
            # Unexpected shape, default to batch_size=1
            actual_batch_size = 1
            if predictions.ndim < 2:
                predictions = predictions[np.newaxis, ...]  # Add batch dimension

        # Ensure we don't exceed the actual batch size
        batch_size = min(actual_batch_size, len(filenames))

        # Save predictions
        for idx in range(batch_size):
            name = filenames[idx]
            prediction = predictions[idx]

            # Apply output transpose if specified
            if output_transpose:
                if prediction.ndim == 3:
                    # 3D volume (D, H, W): transpose spatial dimensions
                    prediction = np.transpose(prediction, output_transpose)
                elif prediction.ndim == 4:
                    # 4D volume (C, D, H, W): keep channel first, transpose spatial dims
                    spatial_transpose = [i + 1 for i in output_transpose]
                    prediction = np.transpose(prediction, [0] + spatial_transpose)

            # Squeeze singleton dimensions (e.g., (1, 1, D, H, W) -> (D, H, W))
            prediction = np.squeeze(prediction)

            destination = output_dir / f"{name}_{suffix}.h5"
            write_hdf5(str(destination), prediction, dataset="prediction")
            print(f"  Saved {suffix}: {destination}")

    def _compute_adapted_rand(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        filenames: List[str]
    ) -> None:
        """
        Compute Adapted Rand Error for instance segmentation evaluation.

        Args:
            predictions: Decoded instance segmentation predictions (B, C, D, H, W) or (B, D, H, W)
            labels: Ground truth labels (B, C, D, H, W) or (B, D, H, W)
            filenames: List of filenames for each sample in batch
        """
        if not self.test_adapted_rand:
            return

        from connectomics.metrics import adapted_rand

        batch_size = predictions.shape[0]

        # Compare decoded predictions with ground truth
        for idx in range(batch_size):
            pred = predictions[idx]
            # Remove channel dimension if present
            if pred.ndim == 4:
                pred = pred[0]  # Take first channel

            # Handle label indexing - labels might be a dict (multi-task) or ndarray
            try:
                if isinstance(labels, dict):
                    # Multi-task labels - use the first task (typically 'label')
                    label_key = list(labels.keys())[0]
                    gt = labels[label_key][idx]
                else:
                    gt = labels[idx]

                gt = gt.astype(np.uint32)
                if gt.ndim == 4:
                    gt = gt[0]  # Remove channel dimension

                are, prec, rec = adapted_rand(pred, gt, all_stats=True)
                self.test_adapted_rand_results.append({
                    'are': are,
                    'precision': prec,
                    'recall': rec,
                    'filename': filenames[idx]
                })
            except Exception as e:
                warnings.warn(
                    f"Error computing adapted_rand for {filenames[idx]}: {e}",
                    UserWarning
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        output = self.model(x)

        # Optionally clamp activations to prevent inf/nan
        if self.clamp_activations:
            if isinstance(output, dict):
                # Deep supervision - clamp all outputs
                output = {k: torch.clamp(v, self.clamp_min, self.clamp_max) for k, v in output.items()}
            else:
                output = torch.clamp(output, self.clamp_min, self.clamp_max)

        return output

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> STEP_OUTPUT:
        """Training step with deep supervision support."""
        images = batch['image']
        labels = batch['label']

        # Forward pass
        outputs = self(images)

        # Check if model outputs deep supervision
        is_deep_supervision = isinstance(outputs, dict) and any(k.startswith('ds_') for k in outputs.keys())

        # Compute loss using deep supervision handler
        if is_deep_supervision:
            total_loss, loss_dict = self.deep_supervision_handler.compute_deep_supervision_loss(
                outputs, labels, stage="train"
            )
        else:
            total_loss, loss_dict = self.deep_supervision_handler.compute_standard_loss(
                outputs, labels, stage="train"
            )

        # Log losses (sync across GPUs for distributed training)
        self.log_dict(loss_dict, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return total_loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> STEP_OUTPUT:
        """Validation step with deep supervision support."""
        images = batch['image']
        labels = batch['label']

        # Forward pass
        outputs = self(images)

        # Check if model outputs deep supervision
        is_deep_supervision = isinstance(outputs, dict) and any(k.startswith('ds_') for k in outputs.keys())

        # Compute loss using deep supervision handler
        if is_deep_supervision:
            total_loss, loss_dict = self.deep_supervision_handler.compute_deep_supervision_loss(
                outputs, labels, stage="val"
            )
        else:
            total_loss, loss_dict = self.deep_supervision_handler.compute_standard_loss(
                outputs, labels, stage="val"
            )

        # Compute evaluation metrics if enabled
        if hasattr(self.cfg, 'inference') and hasattr(self.cfg.inference, 'evaluation'):
            if getattr(self.cfg.inference.evaluation, 'enabled', False):
                metrics = getattr(self.cfg.inference.evaluation, 'metrics', None)
                if metrics is not None:
                    # Get the main output for metric computation
                    if is_deep_supervision:
                        main_output = outputs['output']
                    else:
                        main_output = outputs

                    # Check if this is multi-task learning
                    is_multi_task = hasattr(self.cfg.model, 'multi_task_config') and self.cfg.model.multi_task_config is not None

                    # Convert logits/probabilities to predictions
                    if is_multi_task:
                        # Multi-task learning: use first channel (usually binary segmentation)
                        # Extract first channel for both output and target
                        binary_output = main_output[:, 0:1, ...]  # (B, 1, H, W)
                        binary_target = labels[:, 0:1, ...]  # (B, 1, H, W)
                        preds = (binary_output.squeeze(1) > 0.5).long()  # (B, H, W)
                        targets = binary_target.squeeze(1).long()  # (B, H, W)
                    elif main_output.shape[1] > 1:
                        # Multi-class segmentation: use argmax
                        preds = torch.argmax(main_output, dim=1)  # (B, D, H, W)
                        targets = labels.squeeze(1).long()  # (B, D, H, W)
                    else:
                        # Single channel output (already predicted class or probability)
                        preds = (main_output.squeeze(1) > 0.5).long()  # (B, D, H, W)
                        targets = labels.squeeze(1).long()  # (B, D, H, W)

                    # Compute and log metrics
                    if 'jaccard' in metrics:
                        if not hasattr(self, 'val_jaccard'):
                            num_classes = self.cfg.model.out_channels if hasattr(self.cfg.model, 'out_channels') else 2
                            if num_classes == 1:
                                # Binary segmentation - use binary metrics
                                self.val_jaccard = torchmetrics.JaccardIndex(task='binary').to(self.device)
                            else:
                                # Multi-class segmentation
                                self.val_jaccard = torchmetrics.JaccardIndex(task='multiclass', num_classes=num_classes).to(self.device)
                        self.val_jaccard(preds, targets)
                        self.log('val_jaccard', self.val_jaccard, on_step=False, on_epoch=True, prog_bar=True)

                    if 'dice' in metrics:
                        if not hasattr(self, 'val_dice'):
                            num_classes = self.cfg.model.out_channels if hasattr(self.cfg.model, 'out_channels') else 2
                            if num_classes == 1:
                                # Binary segmentation - use binary metrics
                                self.val_dice = torchmetrics.Dice(task='binary').to(self.device)
                            else:
                                # Multi-class segmentation
                                self.val_dice = torchmetrics.Dice(num_classes=num_classes, average='macro').to(self.device)
                        self.val_dice(preds, targets)
                        self.log('val_dice', self.val_dice, on_step=False, on_epoch=True, prog_bar=True)

                    if 'accuracy' in metrics:
                        if not hasattr(self, 'val_accuracy'):
                            num_classes = self.cfg.model.out_channels if hasattr(self.cfg.model, 'out_channels') else 2
                            if num_classes == 1:
                                # Binary segmentation - use binary metrics
                                self.val_accuracy = torchmetrics.Accuracy(task='binary').to(self.device)
                            else:
                                # Multi-class segmentation
                                self.val_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes).to(self.device)
                        self.val_accuracy(preds, targets)
                        self.log('val_accuracy', self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True)

        # Log losses (sync across GPUs for distributed training)
        self.log_dict(loss_dict, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return total_loss

    def on_test_start(self):
        """Called at the beginning of testing to initialize metrics and inferer."""
        self._setup_test_metrics()

        # Explicitly set eval mode if configured (Lightning does this by default, but be explicit)
        if hasattr(self.cfg, 'inference') and getattr(self.cfg.inference, 'do_eval', True):
            self.eval()
        else:
            # Keep in training mode (e.g., for Monte Carlo Dropout uncertainty estimation)
            self.train()

    def on_test_end(self):
        """Called at the end of testing to compute and log final metrics."""
        # Compute average adapted_rand if results were collected
        if self.test_adapted_rand and len(self.test_adapted_rand_results) > 0:
            # Compute average metrics
            avg_are = np.mean([r['are'] for r in self.test_adapted_rand_results])
            avg_prec = np.mean([r['precision'] for r in self.test_adapted_rand_results])
            avg_rec = np.mean([r['recall'] for r in self.test_adapted_rand_results])

            # Log to console
            print(f"\n{'='*60}")
            print(f"Adapted Rand Error (ARE) Results:")
            print(f"{'='*60}")
            print(f"Average ARE:       {avg_are:.6f}")
            print(f"Average Precision: {avg_prec:.6f}")
            print(f"Average Recall:    {avg_rec:.6f}")
            print(f"Number of samples: {len(self.test_adapted_rand_results)}")
            print(f"{'='*60}\n")

            # Log individual results
            print("Per-sample results:")
            for r in self.test_adapted_rand_results:
                print(f"  {r['filename']}: ARE={r['are']:.6f}, Prec={r['precision']:.6f}, Rec={r['recall']:.6f}")
            print()

            # Log to tensorboard/wandb directly (can't use self.log in on_test_end)
            if self.logger is not None:
                try:
                    # For TensorBoardLogger
                    if hasattr(self.logger, 'experiment'):
                        self.logger.experiment.add_scalar('test_adapted_rand_mean', avg_are, self.global_step)
                        self.logger.experiment.add_scalar('test_adapted_rand_precision', avg_prec, self.global_step)
                        self.logger.experiment.add_scalar('test_adapted_rand_recall', avg_rec, self.global_step)
                except Exception as e:
                    warnings.warn(f"Could not log metrics to logger: {e}", UserWarning)

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> STEP_OUTPUT:
        """Test step with optional sliding-window inference and metrics computation."""
        images = batch['image']
        labels = batch.get('label')
        mask = batch.get('mask')  # Get test mask if available

        # Get batch size from images
        actual_batch_size = images.shape[0]

        # Resolve filenames once for all operations
        filenames = resolve_output_filenames(self.cfg, batch, self.global_step)

        # Ensure filenames list matches actual batch size
        # If we don't have enough filenames, generate default ones
        while len(filenames) < actual_batch_size:
            filenames.append(f"volume_{self.global_step}_{len(filenames)}")

        # Check if prediction files already exist
        from connectomics.data.io import read_hdf5
        from pathlib import Path
        
        output_dir_value = None
        if hasattr(self.cfg, 'inference') and hasattr(self.cfg.inference, 'data') and hasattr(self.cfg.inference.data, 'output_path'):
            output_dir_value = self.cfg.inference.data.output_path
        
        predictions_np = None
        if output_dir_value:
            output_dir = Path(output_dir_value)
            # Check if all prediction files exist
            all_exist = True
            existing_predictions = []
            
            for filename in filenames[:actual_batch_size]:
                pred_file = output_dir / f"{filename}_prediction.h5"
                if pred_file.exists():
                    print(f"  ✓ Found existing prediction: {pred_file}")
                    try:
                        pred = read_hdf5(str(pred_file), dataset="main")
                        existing_predictions.append(pred)
                    except Exception as e:
                        print(f"  ⚠️  Failed to load {pred_file}: {e}, will re-run inference")
                        all_exist = False
                        break
                else:
                    print(f"  ✗ Prediction file not found: {pred_file}, will run inference")
                    all_exist = False
                    break
            
            if all_exist and len(existing_predictions) == actual_batch_size:
                print(f"  ✅ All prediction files exist! Loading {len(existing_predictions)} predictions and skipping inference.")
                # Stack predictions into batch format
                if actual_batch_size == 1:
                    predictions_np = existing_predictions[0]
                    # Add batch dimension if needed
                    if predictions_np.ndim < 4:
                        predictions_np = predictions_np[np.newaxis, ...]
                else:
                    # Stack multiple predictions
                    predictions_np = np.stack([p[np.newaxis, ...] if p.ndim < 4 else p for p in existing_predictions], axis=0)
                
                # Reverse postprocessing to get back to [0,1] range for evaluation
                # The saved predictions are postprocessed (scaled, dtype converted)
                if hasattr(self.cfg, 'inference') and hasattr(self.cfg.inference, 'postprocessing'):
                    postprocessing = self.cfg.inference.postprocessing
                    # Reverse intensity scaling
                    intensity_scale = getattr(postprocessing, 'intensity_scale', None)
                    output_scale = getattr(postprocessing, 'output_scale', None)
                    scale = intensity_scale if intensity_scale is not None else output_scale
                    if scale is not None:
                        predictions_np = predictions_np.astype(np.float32) / float(scale)
                    else:
                        # Convert to float if it was converted to uint8/int
                        if predictions_np.dtype in [np.uint8, np.int8, np.uint16, np.int16]:
                            predictions_np = predictions_np.astype(np.float32) / 255.0

        # Run inference only if predictions don't exist
        if predictions_np is None:
            print(f"  🔄 Running inference (predictions not found or incomplete)")
            # Always use TTA (handles no-transform case) + sliding window
            # TTA preprocessing (activation, masking) is applied regardless of flip augmentation
            # Note: TTA always returns a simple tensor, not a dict (deep supervision not supported in test mode)
            predictions = self.inference_manager.predict_with_tta(images, mask=mask)

            # Convert predictions to numpy for saving/decoding
            predictions_np = predictions.detach().cpu().float().numpy()

        # Track if we loaded existing predictions (skip decode/postprocess if already done)
        loaded_from_file = (predictions_np is not None and output_dir_value and all_exist)
        
        # Check if we should save intermediate predictions (before decoding)
        save_intermediate = False
        if hasattr(self.cfg, 'inference') and hasattr(self.cfg.inference, 'test_time_augmentation'):
            save_intermediate = getattr(self.cfg.inference.test_time_augmentation, 'save_predictions', False)

        # Save intermediate TTA predictions (before decoding) if requested (only if we ran inference)
        if save_intermediate and not loaded_from_file:
            write_outputs(self.cfg, predictions_np, filenames, suffix="tta_prediction")

        # Apply decode mode (instance segmentation decoding) - skip if loaded from file
        if loaded_from_file:
            decoded_predictions = predictions_np  # Already decoded and reversed from postprocessing
        else:
            decoded_predictions = apply_decode_mode(self.cfg, predictions_np)

            # Apply postprocessing (scaling and dtype conversion) if configured
            postprocessed_predictions = apply_postprocessing(self.cfg, decoded_predictions)

            # Save final decoded and postprocessed predictions
            write_outputs(self.cfg, postprocessed_predictions, filenames, suffix="prediction")

        # Compute adapted_rand if enabled and labels available
        if labels is not None:
            labels_np = labels.detach().cpu().numpy()
            self._compute_adapted_rand(decoded_predictions, labels_np, filenames)

        # Compute evaluation metrics if enabled and labels are available
        if labels is not None and hasattr(self.cfg, 'inference') and hasattr(self.cfg.inference, 'evaluation'):
            evaluation_enabled = getattr(self.cfg.inference.evaluation, 'enabled', False)
            if evaluation_enabled:
                # Convert predictions back to torch tensor for metrics (before postprocessing)
                # Use decoded_predictions (after decoding, before postprocessing scaling)
                pred_tensor = torch.from_numpy(decoded_predictions).float().to(self.device)
                labels_tensor = labels.float()
                
                # Squeeze singleton dimensions for metrics
                pred_tensor = pred_tensor.squeeze()
                labels_tensor = labels_tensor.squeeze()
                
                # Ensure same number of dimensions
                if pred_tensor.ndim != labels_tensor.ndim:
                    # Add channel dimension if needed
                    if pred_tensor.ndim == labels_tensor.ndim - 1:
                        pred_tensor = pred_tensor.unsqueeze(0)
                    elif labels_tensor.ndim == pred_tensor.ndim - 1:
                        labels_tensor = labels_tensor.unsqueeze(0)
                
                # Binarize predictions for binary metrics (threshold at 0.5)
                if pred_tensor.max() <= 1.0:  # Already in [0, 1] range
                    pred_binary = (pred_tensor > 0.5).long()
                else:
                    # Normalize first if needed
                    pred_binary = (torch.sigmoid(pred_tensor) > 0.5).long()
                
                labels_binary = (labels_tensor > 0.5).long() if labels_tensor.max() <= 1.0 else labels_tensor.long()
                
                # Update metrics
                if hasattr(self, 'test_jaccard') and self.test_jaccard is not None:
                    self.test_jaccard.update(pred_binary, labels_binary)
                    self.log('test_jaccard', self.test_jaccard, on_step=False, on_epoch=True, prog_bar=True, logger=True)
                
                if hasattr(self, 'test_dice') and self.test_dice is not None:
                    self.test_dice.update(pred_binary, labels_binary)
                    self.log('test_dice', self.test_dice, on_step=False, on_epoch=True, prog_bar=True, logger=True)
                
                if hasattr(self, 'test_accuracy') and self.test_accuracy is not None:
                    self.test_accuracy.update(pred_binary, labels_binary)
                    self.log('test_accuracy', self.test_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
                
                # Return dummy value (metrics are logged separately)
                return torch.tensor(0.0, device=self.device)

        # Determine if we should skip loss computation
        skip_loss = False
        if labels is None:
            skip_loss = True
        elif hasattr(self.cfg, 'inference') and hasattr(self.cfg.inference, 'evaluation'):
            # Skip loss when any evaluation metric is enabled (labels are not transformed)
            evaluation_enabled = getattr(self.cfg.inference.evaluation, 'enabled', False)
            if evaluation_enabled:
                skip_loss = True

        # Skip loss computation if needed
        if skip_loss:
            return torch.tensor(0.0, device=self.device)

        # Compute test loss if labels are available and evaluation is not enabled
        outputs = self(images)
        is_deep_supervision = isinstance(outputs, dict) and any(k.startswith('ds_') for k in outputs.keys())

        if is_deep_supervision:
            total_loss, loss_dict = self.deep_supervision_handler.compute_deep_supervision_loss(
                outputs, labels, stage="test"
            )
        else:
            total_loss, loss_dict = self.deep_supervision_handler.compute_standard_loss(
                outputs, labels, stage="test"
            )

        self.log_dict(loss_dict, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return total_loss

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers and learning rate schedulers."""
        optimizer = build_optimizer(self.cfg, self.model)

        # Build scheduler if configured
        if hasattr(self.cfg, 'scheduler') and self.cfg.scheduler is not None:
            scheduler = build_lr_scheduler(self.cfg, optimizer)

            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1,
                },
            }
        else:
            return optimizer

    def on_train_epoch_end(self) -> None:
        """Called at the end of training epoch."""
        # Log learning rate
        if self.optimizers():
            optimizer = self.optimizers()
            if isinstance(optimizer, list):
                optimizer = optimizer[0]
            lr = optimizer.param_groups[0]['lr']
            self.log('lr', lr, on_step=False, on_epoch=True, prog_bar=True, logger=True)


def create_lightning_module(
    cfg: Union[Config, DictConfig],
    model: Optional[nn.Module] = None,
) -> ConnectomicsModule:
    """
    Factory function to create ConnectomicsModule.

    Args:
        cfg: Hydra Config object or OmegaConf DictConfig
        model: Optional pre-built model

    Returns:
        ConnectomicsModule instance
    """
    return ConnectomicsModule(cfg, model)

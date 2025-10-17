"""
PyTorch Lightning module for PyTorch Connectomics.

This module implements the Lightning interface with:
- Hydra/OmegaConf configuration
- MONAI native models
- Modern loss functions
- Automatic distributed training, mixed precision, checkpointing
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional, Union, Tuple
import warnings
import pdb
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from omegaconf import DictConfig
import torchmetrics
from monai.inferers import SlidingWindowInferer
from monai.transforms import Flip

# Import existing components
from ..models import build_model
from ..models.loss import create_loss
from ..models.solver import build_optimizer, build_lr_scheduler
from ..config import Config
from ..utils.debug_hooks import NaNDetectionHookManager


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

        # Hook manager for intermediate layer debugging
        self._hook_manager: Optional[NaNDetectionHookManager] = None

        # Test metrics (initialized lazily during test mode if specified in config)
        self.test_jaccard = None
        self.test_dice = None
        self.test_accuracy = None
        self.test_adapted_rand = None  # Adapted Rand error (instance segmentation metric)
        self.test_adapted_rand_results = []  # Store per-batch results for averaging
        self.sliding_inferer: Optional[SlidingWindowInferer] = None

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
        for i, loss_name in enumerate(loss_names):
            # Get kwargs for this loss (default to empty dict if not specified)
            kwargs = loss_kwargs_list[i] if i < len(loss_kwargs_list) else {}
            loss = create_loss(loss_name=loss_name, **kwargs)
            losses.append(loss)

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
                return tuple(int(v) for v in output_size)

        if hasattr(self.cfg, 'data') and hasattr(self.cfg.data, 'patch_size'):
            patch_size = getattr(self.cfg.data, 'patch_size', None)
            if patch_size:
                return tuple(int(v) for v in patch_size)

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
        return self._extract_main_output(outputs)

    def _apply_tta_preprocessing(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply activation and channel selection before TTA ensemble.
        
        Supports per-channel activations via channel_activations config:
        channel_activations: [[0, 'sigmoid'], [1, 'sigmoid'], [2, None]]

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
            for channel_idx, act in channel_activations:
                channel_tensor = tensor[:, channel_idx:channel_idx+1, ...]
                
                if act == 'sigmoid':
                    channel_tensor = torch.sigmoid(channel_tensor)
                elif act == 'tanh':
                    channel_tensor = torch.tanh(channel_tensor)
                elif act == 'softmax':
                    # Softmax doesn't make sense for single channel, skip
                    warnings.warn(
                        f"Softmax activation for single channel {channel_idx} is not supported. Skipping.",
                        UserWarning,
                    )
                elif act is None or (isinstance(act, str) and act.lower() == 'none'):
                    # No activation (keep as is)
                    pass
                else:
                    warnings.warn(
                        f"Unknown activation '{act}' for channel {channel_idx}. Supported: 'sigmoid', 'tanh', None",
                        UserWarning,
                    )
                
                activated_channels.append(channel_tensor)
            
            # Concatenate all channels back together
            tensor = torch.cat(activated_channels, dim=1)
        else:
            # Fall back to single activation for all channels (old approach)
            tta_act = getattr(self.cfg.inference.test_time_augmentation, 'act', None)
            if tta_act is None:
                tta_act = getattr(self.cfg.inference, 'output_act', None)

            # Apply activation function
            if tta_act == 'softmax':
                tensor = torch.softmax(tensor, dim=1)
            elif tta_act == 'sigmoid':
                tensor = torch.sigmoid(tensor)
            elif tta_act == 'tanh':
                tensor = torch.tanh(tensor)
            elif tta_act is not None and tta_act.lower() != 'none':
                warnings.warn(
                    f"Unknown TTA activation function '{tta_act}'. Supported: 'softmax', 'sigmoid', 'tanh', None",
                    UserWarning,
                )

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
                # "all" or []: All 8 flips (all combinations of Z, Y, X)
                # IMPORTANT: MONAI Flip spatial_axis behavior for (B, C, D, H, W) tensors:
                #   spatial_axis=[0] flips C (channel) - WRONG for TTA!
                #   spatial_axis=[1] flips D (depth/Z) - CORRECT
                #   spatial_axis=[2] flips H (height/Y) - CORRECT
                #   spatial_axis=[3] flips W (width/X) - CORRECT
                # Must use [1, 2, 3] for [D, H, W] flips, NOT [0, 1, 2]!
                tta_flip_axes = [
                    [],           # No flip
                    [1],          # Flip Z (depth)
                    [2],          # Flip Y (height)
                    [3],          # Flip X (width)
                    [1, 2],       # Flip Z+Y
                    [1, 3],       # Flip Z+X
                    [2, 3],       # Flip Y+X
                    [1, 2, 3],    # Flip Z+Y+X
                ]
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

    def _apply_postprocessing(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply postprocessing to predictions: activation and channel selection.

        Args:
            tensor: Raw predictions (B, C, D, H, W)

        Returns:
            Postprocessed tensor with activation and/or channel selection applied
        """
        if not hasattr(self.cfg, 'inference'):
            return tensor

        # Apply activation function
        output_act = getattr(self.cfg.inference, 'output_act', None)
        if output_act == 'softmax':
            tensor = torch.softmax(tensor, dim=1)
        elif output_act == 'sigmoid':
            tensor = torch.sigmoid(tensor)
        elif output_act is not None and output_act.lower() != 'none':
            warnings.warn(
                f"Unknown activation function '{output_act}'. Supported: 'softmax', 'sigmoid', None",
                UserWarning,
            )

        # Apply channel selection
        output_channel = getattr(self.cfg.inference, 'output_channel', None)
        if output_channel is not None:
            if isinstance(output_channel, int):
                if output_channel == -1:
                    # -1 means all channels
                    pass
                else:
                    # Single channel selection
                    tensor = tensor[:, output_channel:output_channel+1, ...]
            elif isinstance(output_channel, (list, tuple)):
                # Multiple channel selection
                tensor = tensor[:, list(output_channel), ...]

        return tensor

    def _apply_output_scale(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply optional per-channel scaling before saving."""
        if not hasattr(self.cfg, 'inference'):
            return tensor

        # Access output_scale from nested postprocessing config
        scale = None
        if hasattr(self.cfg.inference, 'postprocessing'):
            scale = getattr(self.cfg.inference.postprocessing, 'output_scale', None)
        if not scale:
            return tensor

        scale_tensor = torch.tensor(scale, dtype=tensor.dtype, device=tensor.device)
        if scale_tensor.numel() == 1:
            return tensor * scale_tensor.view(1, 1, 1, 1, 1)

        if scale_tensor.numel() == tensor.shape[1]:
            return tensor * scale_tensor.view(1, -1, 1, 1, 1)

        warnings.warn(
            "inference.output_scale length does not match output channels; skipping scaling.",
            UserWarning,
        )
        return tensor

    def _apply_output_dtype(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply output dtype conversion."""
        if not hasattr(self.cfg, 'inference'):
            return tensor

        # Access output_dtype from nested postprocessing config
        output_dtype = None
        if hasattr(self.cfg.inference, 'postprocessing'):
            output_dtype = getattr(self.cfg.inference.postprocessing, 'output_dtype', None)
        if not output_dtype or output_dtype == 'float32':
            return tensor

        # Map string dtype to numpy dtype
        dtype_map = {
            'uint8': torch.uint8,
            'int8': torch.int8,
            'int16': torch.int16,
            'int32': torch.int32,
            'int64': torch.int64,
            'float16': torch.float16,
            'float32': torch.float32,
            'float64': torch.float64,
        }

        if output_dtype not in dtype_map:
            warnings.warn(
                f"Unknown output_dtype '{output_dtype}'. Supported: {list(dtype_map.keys())}. "
                f"Keeping float32.",
                UserWarning,
            )
            return tensor

        target_dtype = dtype_map[output_dtype]

        # Clamp to valid range before conversion for integer types
        if output_dtype == 'uint8':
            tensor = torch.clamp(tensor, 0, 255)
        elif output_dtype == 'int8':
            tensor = torch.clamp(tensor, -128, 127)
        elif output_dtype == 'int16':
            tensor = torch.clamp(tensor, -32768, 32767)

        return tensor.to(target_dtype)

    def _apply_postprocessing(self, data: np.ndarray) -> np.ndarray:
        """
        Apply postprocessing transformations (scaling and dtype conversion) to predictions.

        This method applies:
        1. Scaling (output_scale): Multiply predictions by scale factor
        2. Dtype conversion (output_dtype): Convert to target dtype with clamping

        Args:
            data: Numpy array of predictions (B, C, D, H, W) or (B, D, H, W)

        Returns:
            Postprocessed predictions with applied scaling and dtype conversion
        """
        if not hasattr(self.cfg, 'inference') or not hasattr(self.cfg.inference, 'postprocessing'):
            return data

        postprocessing = self.cfg.inference.postprocessing

        # Apply scaling if configured
        output_scale = getattr(postprocessing, 'output_scale', None)
        if output_scale is not None:
            data = data * output_scale

        # Apply dtype conversion if configured
        output_dtype = getattr(postprocessing, 'output_dtype', None)
        if output_dtype is not None and output_dtype != 'float32':
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

            if output_dtype not in dtype_map:
                warnings.warn(
                    f"Unknown output_dtype '{output_dtype}'. Supported: {list(dtype_map.keys())}. "
                    f"Keeping current dtype.",
                    UserWarning,
                )
                return data

            target_dtype = dtype_map[output_dtype]

            # Clamp to valid range before conversion for integer types
            if output_dtype == 'uint8':
                data = np.clip(data, 0, 255)
            elif output_dtype == 'int8':
                data = np.clip(data, -128, 127)
            elif output_dtype == 'uint16':
                data = np.clip(data, 0, 65535)
            elif output_dtype == 'int16':
                data = np.clip(data, -32768, 32767)
            elif output_dtype == 'uint32':
                data = np.clip(data, 0, 4294967295)
            elif output_dtype == 'int32':
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
            decode_binary_cc,
            decode_binary_watershed,
            decode_binary_contour_cc,
            decode_binary_contour_watershed,
            decode_binary_contour_distance_watershed,
            decode_affinity_cc,
        )

        # Map function names to actual functions
        decode_fn_map = {
            'decode_binary_cc': decode_binary_cc,
            'decode_binary_watershed': decode_binary_watershed,
            'decode_binary_contour_cc': decode_binary_contour_cc,
            'decode_binary_contour_watershed': decode_binary_contour_watershed,
            'decode_binary_contour_distance_watershed': decode_binary_contour_distance_watershed,
            'decode_affinity_cc': decode_affinity_cc,
        }

        # Process each sample in batch
        batch_size = data.shape[0] if data.ndim == 5 else 1
        if data.ndim == 4:
            data = data[np.newaxis, ...]  # Add batch dimension

        results = []
        for batch_idx in range(batch_size):
            sample = data[batch_idx]  # (C, D, H, W)

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

                # Backward compatibility: convert old parameter format to new tuple format
                # for decode_binary_contour_distance_watershed
                if fn_name == 'decode_binary_contour_distance_watershed':
                    if 'seed_threshold' in kwargs or 'foreground_threshold' in kwargs:
                        warnings.warn(
                            "Detected legacy parameters (seed_threshold, contour_threshold, foreground_threshold) "
                            "for decode_binary_contour_distance_watershed. Converting to new tuple format "
                            "(binary_threshold, contour_threshold, distance_threshold). "
                            "Please update your config files to use the new format.",
                            DeprecationWarning,
                        )
                        # Convert old parameters to new tuple format
                        seed_thresh = kwargs.pop('seed_threshold', 0.9)
                        contour_thresh = kwargs.pop('contour_threshold', 0.8)
                        foreground_thresh = kwargs.pop('foreground_threshold', 0.85)
                        
                        # Map old parameters to new tuple format
                        kwargs['binary_threshold'] = (seed_thresh, foreground_thresh)
                        kwargs['contour_threshold'] = (contour_thresh, 1.1)
                        kwargs['distance_threshold'] = (0.5, -0.5)

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
        decoded = np.stack(results, axis=0) if len(results) > 1 else results[0]

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

        if isinstance(meta, dict):
            meta_filenames = meta.get('filename_or_obj')
            if isinstance(meta_filenames, (list, tuple)):
                filenames = list(meta_filenames)
            elif meta_filenames is not None:
                filenames = [meta_filenames]
        elif isinstance(meta, list):
            for meta_item in meta:
                if isinstance(meta_item, dict):
                    filenames.append(meta_item.get('filename_or_obj'))
            # Update batch_size from metadata if we have a list
            batch_size = max(batch_size, len(filenames))

        resolved_names: List[str] = []
        for idx in range(batch_size):
            if idx < len(filenames) and filenames[idx]:
                resolved_names.append(Path(str(filenames[idx])).stem)
            else:
                resolved_names.append(f"volume_{self.global_step}_{idx}")

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

        # Save predictions
        for idx, name in enumerate(filenames):
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

        from connectomics.engine.metrics import adapted_rand

        batch_size = predictions.shape[0]

        # Compare decoded predictions with ground truth
        for idx in range(batch_size):
            pred = predictions[idx]
            # Remove channel dimension if present
            if pred.ndim == 4:
                pred = pred[0]  # Take first channel
            gt = labels[idx].astype(np.uint32)
            if gt.ndim == 4:
                gt = gt[0]  # Remove channel dimension

            try:
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

    def _compute_multitask_loss(self, outputs: torch.Tensor, labels: torch.Tensor) -> tuple:
        """
        Compute losses for multi-task learning where different output channels
        correspond to different targets with specific losses.
        
        Args:
            outputs: Model outputs (B, C, D, H, W) where C contains all task channels
            labels: Ground truth labels (B, C, D, H, W) where C contains all target channels
            
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        total_loss = 0.0
        loss_dict = {}
        
        # Parse multi-task configuration
        # Format: [[start_ch, end_ch, "task_name", [loss_indices]], ...]
        for task_idx, task_config in enumerate(self.cfg.model.multi_task_config):
            start_ch, end_ch, task_name, loss_indices = task_config
            
            # Extract channels for this task
            task_output = outputs[:, start_ch:end_ch, ...]
            task_label = labels[:, start_ch:end_ch, ...]
            
            # Apply specified losses for this task
            task_loss = 0.0
            for loss_idx in loss_indices:
                loss_fn = self.loss_functions[loss_idx]
                weight = self.loss_weights[loss_idx]
                
                loss = loss_fn(task_output, task_label)
                
                # Check for NaN/Inf
                if self.enable_nan_detection and (torch.isnan(loss) or torch.isinf(loss)):
                    print(f"\n{'='*80}")
                    print(f"⚠️  NaN/Inf detected in multi-task loss!")
                    print(f"{'='*80}")
                    print(f"Task: {task_name} (channels {start_ch}:{end_ch})")
                    print(f"Loss function: {loss_fn.__class__.__name__} (index {loss_idx})")
                    print(f"Loss value: {loss.item()}")
                    print(f"Output shape: {task_output.shape}, range: [{task_output.min():.4f}, {task_output.max():.4f}]")
                    print(f"Label shape: {task_label.shape}, range: [{task_label.min():.4f}, {task_label.max():.4f}]")
                    print(f"Output contains NaN: {torch.isnan(task_output).any()}")
                    print(f"Label contains NaN: {torch.isnan(task_label).any()}")
                    if self.debug_on_nan:
                        print(f"\nEntering debugger...")
                        import pdb
                        pdb.set_trace()
                    raise ValueError(f"NaN/Inf in loss for task '{task_name}' with loss index {loss_idx}")
                
                weighted_loss = loss * weight
                task_loss += weighted_loss
                
                # Log individual loss
                loss_dict[f'train_loss_{task_name}_loss{loss_idx}'] = loss.item()
            
            # Log task total
            loss_dict[f'train_loss_{task_name}_total'] = task_loss.item()
            total_loss += task_loss
        
        loss_dict['train_loss_total'] = total_loss.item()
        return total_loss, loss_dict

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

        # Compute loss
        total_loss = 0.0
        loss_dict = {}

        if is_deep_supervision:
            # Multi-scale loss with deep supervision
            # Weights decrease for smaller scales: [1.0, 0.5, 0.25, 0.125, 0.0625]
            main_output = outputs['output']
            ds_outputs = [outputs[f'ds_{i}'] for i in range(1, 5) if f'ds_{i}' in outputs]

            ds_weights = [1.0] + [0.5 ** i for i in range(1, len(ds_outputs) + 1)]
            all_outputs = [main_output] + ds_outputs

            for scale_idx, (output, ds_weight) in enumerate(zip(all_outputs, ds_weights)):
                # Match target to output size
                target = self._match_target_to_output(labels, output)

                # Compute loss for this scale
                scale_loss = 0.0
                for loss_fn, weight in zip(self.loss_functions, self.loss_weights):
                    loss = loss_fn(output, target)

                    # Check for NaN/Inf immediately after computing loss
                    if self.enable_nan_detection and (torch.isnan(loss) or torch.isinf(loss)):
                        print(f"\n{'='*80}")
                        print(f"⚠️  NaN/Inf detected in loss computation!")
                        print(f"{'='*80}")
                        print(f"Loss function: {loss_fn.__class__.__name__}")
                        print(f"Loss value: {loss.item()}")
                        print(f"Scale: {scale_idx}, Weight: {weight}")
                        print(f"Output shape: {output.shape}, range: [{output.min():.4f}, {output.max():.4f}]")
                        print(f"Target shape: {target.shape}, range: [{target.min():.4f}, {target.max():.4f}]")
                        print(f"Output contains NaN: {torch.isnan(output).any()}")
                        print(f"Target contains NaN: {torch.isnan(target).any()}")
                        if self.debug_on_nan:
                            print(f"\nEntering debugger...")
                            pdb.set_trace()
                        raise ValueError(f"NaN/Inf in loss at scale {scale_idx}")

                    scale_loss += loss * weight

                total_loss += scale_loss * ds_weight
                loss_dict[f'train_loss_scale_{scale_idx}'] = scale_loss.item()

            loss_dict['train_loss_total'] = total_loss.item()

        else:
            # Check if multi-task learning is configured
            if hasattr(self.cfg.model, 'multi_task_config') and self.cfg.model.multi_task_config is not None:
                # Multi-task learning: apply specific losses to specific channels
                total_loss, loss_dict = self._compute_multitask_loss(outputs, labels)
            else:
                # Standard single-scale loss: apply all losses to all outputs
                for i, (loss_fn, weight) in enumerate(zip(self.loss_functions, self.loss_weights)):
                    loss = loss_fn(outputs, labels)

                    # Check for NaN/Inf immediately after computing loss
                    if self.enable_nan_detection and (torch.isnan(loss) or torch.isinf(loss)):
                        print(f"\n{'='*80}")
                        print(f"⚠️  NaN/Inf detected in loss computation!")
                        print(f"{'='*80}")
                        print(f"Loss function: {loss_fn.__class__.__name__}")
                        print(f"Loss value: {loss.item()}")
                        print(f"Loss index: {i}, Weight: {weight}")
                        print(f"Output shape: {outputs.shape}, range: [{outputs.min():.4f}, {outputs.max():.4f}]")
                        print(f"Label shape: {labels.shape}, range: [{labels.min():.4f}, {labels.max():.4f}]")
                        print(f"Output contains NaN: {torch.isnan(outputs).any()}")
                        print(f"Label contains NaN: {torch.isnan(labels).any()}")
                        if self.debug_on_nan:
                            print(f"\nEntering debugger...")
                            pdb.set_trace()
                        raise ValueError(f"NaN/Inf in loss at index {i}")

                    weighted_loss = loss * weight
                    total_loss += weighted_loss

                    loss_dict[f'train_loss_{i}'] = loss.item()

                loss_dict['train_loss_total'] = total_loss.item()

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

        # Compute loss
        total_loss = 0.0
        loss_dict = {}

        if is_deep_supervision:
            # Multi-scale loss with deep supervision
            main_output = outputs['output']
            ds_outputs = [outputs[f'ds_{i}'] for i in range(1, 5) if f'ds_{i}' in outputs]

            ds_weights = [1.0] + [0.5 ** i for i in range(1, len(ds_outputs) + 1)]
            all_outputs = [main_output] + ds_outputs

            for scale_idx, (output, ds_weight) in enumerate(zip(all_outputs, ds_weights)):
                # Match target to output size
                target = self._match_target_to_output(labels, output)

                # Compute loss for this scale
                scale_loss = 0.0
                for loss_fn, weight in zip(self.loss_functions, self.loss_weights):
                    loss = loss_fn(output, target)
                    scale_loss += loss * weight

                total_loss += scale_loss * ds_weight
                loss_dict[f'val_loss_scale_{scale_idx}'] = scale_loss.item()

            loss_dict['val_loss_total'] = total_loss.item()

        else:
            # Standard single-scale loss
            for i, (loss_fn, weight) in enumerate(zip(self.loss_functions, self.loss_weights)):
                loss = loss_fn(outputs, labels)
                weighted_loss = loss * weight
                total_loss += weighted_loss

                loss_dict[f'val_loss_{i}'] = loss.item()

            loss_dict['val_loss_total'] = total_loss.item()

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
        self._setup_sliding_window_inferer()

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

        # Always use TTA (handles no-transform case) + sliding window
        # TTA preprocessing (activation, masking) is applied regardless of flip augmentation
        # Note: TTA always returns a simple tensor, not a dict (deep supervision not supported in test mode)
        predictions = self._predict_with_tta(images, mask=mask)

        # Convert predictions to numpy for saving/decoding
        predictions_np = predictions.detach().cpu().float().numpy()

        # Resolve filenames once for all saving operations
        filenames = self._resolve_output_filenames(batch)

        # Check if we should save intermediate predictions (before decoding)
        save_intermediate = False
        if hasattr(self.cfg, 'inference') and hasattr(self.cfg.inference, 'test_time_augmentation'):
            save_intermediate = getattr(self.cfg.inference.test_time_augmentation, 'save_predictions', False)

        # Save intermediate TTA predictions (before decoding) if requested
        if save_intermediate:
            self._write_outputs(predictions_np, filenames, suffix="tta_prediction")

        # Apply decode mode (instance segmentation decoding)
        decoded_predictions = self._apply_decode_mode(predictions_np)

        # Apply postprocessing (scaling and dtype conversion) if configured
        postprocessed_predictions = self._apply_postprocessing(decoded_predictions)

        # Save final decoded and postprocessed predictions
        self._write_outputs(postprocessed_predictions, filenames, suffix="prediction")

        # Compute adapted_rand if enabled and labels available
        if labels is not None:
            labels_np = labels.detach().cpu().numpy()
            self._compute_adapted_rand(decoded_predictions, labels_np, filenames)

        # Determine if we should skip loss computation
        skip_loss = False
        if labels is None:
            skip_loss = True
        elif hasattr(self.cfg, 'inference') and hasattr(self.cfg.inference, 'evaluation'):
            # Skip loss when any evaluation metric is enabled (labels are not transformed)
            evaluation_enabled = getattr(self.cfg.inference.evaluation, 'enabled', False)
            metrics = getattr(self.cfg.inference.evaluation, 'metrics', [])
            if evaluation_enabled and metrics:
                skip_loss = True
        elif hasattr(self.cfg, 'inference') and hasattr(self.cfg.inference, 'test_time_augmentation'):
            # Skip loss computation if TTA preprocessing changed the output shape
            # (e.g., channel selection makes it incompatible with loss functions)
            tta_channel = getattr(self.cfg.inference.test_time_augmentation, 'select_channel', None)
            if tta_channel is None:
                tta_channel = getattr(self.cfg.inference, 'output_channel', None)

            # If channel selection was applied, skip loss computation but keep metrics
            if tta_channel is not None and tta_channel != -1:
                skip_loss = True
                print("⚠️  Skipping loss computation (TTA channel selection enabled). Metrics will still be computed.")

        total_loss = 0.0
        loss_dict = {}

        # Compute loss only if not skipped
        # Note: Loss computation in test mode is typically skipped when evaluation metrics are enabled
        # or when TTA channel selection changes output shape
        if not skip_loss:
            for i, (loss_fn, weight) in enumerate(zip(self.loss_functions, self.loss_weights)):
                loss = loss_fn(predictions, labels)
                weighted_loss = loss * weight
                total_loss += weighted_loss
                loss_dict[f'test_loss_{i}'] = loss.item()

            loss_dict['test_loss_total'] = total_loss.item()

        # Compute metrics (only if metrics are configured and labels available)
        # Metrics can be computed even if loss is skipped (e.g., with TTA channel selection)
        if labels is not None and (self.test_jaccard is not None or self.test_dice is not None or self.test_accuracy is not None):
            # Convert logits/probabilities to predictions
            # Check if predictions has multiple channels (need argmax) or single channel
            if predictions.shape[1] > 1:
                preds = torch.argmax(predictions, dim=1)  # (B, D, H, W)
            else:
                # Single channel output (already predicted class or probability)
                preds = (predictions.squeeze(1) > 0.5).long()  # (B, D, H, W)

            targets = labels.squeeze(1).long()  # (B, D, H, W)

            # Update and log metrics
            if self.test_jaccard is not None:
                self.test_jaccard(preds, targets)
                self.log('test_jaccard', self.test_jaccard, on_step=False, on_epoch=True, prog_bar=True)
            if self.test_dice is not None:
                self.test_dice(preds, targets)
                self.log('test_dice', self.test_dice, on_step=False, on_epoch=True, prog_bar=True)
            if self.test_accuracy is not None:
                self.test_accuracy(preds, targets)
                self.log('test_accuracy', self.test_accuracy, on_step=False, on_epoch=True, prog_bar=True)

        # Log losses (only if loss was computed, sync across GPUs for distributed training)
        if not skip_loss and loss_dict:
            self.log_dict(loss_dict, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        # Return loss if computed, otherwise None
        return total_loss if not skip_loss else None

    def _match_target_to_output(
        self,
        target: torch.Tensor,
        output: torch.Tensor
    ) -> torch.Tensor:
        """
        Match target size to output size for deep supervision.

        Uses interpolation to downsample labels to match output resolution.
        For segmentation masks, uses nearest-neighbor interpolation to preserve labels.
        For continuous targets, uses trilinear interpolation.

        Args:
            target: Target tensor of shape (B, C, D, H, W)
            output: Output tensor of shape (B, C, D', H', W')

        Returns:
            Resized target tensor matching output shape
        """
        if target.shape == output.shape:
            return target

        # Determine interpolation mode based on data type
        if target.dtype in [torch.long, torch.int, torch.int32, torch.int64, torch.uint8, torch.ByteTensor]:
            # Integer labels (including Byte/uint8): use nearest-neighbor
            mode = 'nearest'
            target_resized = nn.functional.interpolate(
                target.float(),
                size=output.shape[2:],
                mode=mode,
            ).long()
        else:
            # Continuous values: use trilinear
            mode = 'trilinear'
            target_resized = nn.functional.interpolate(
                target,
                size=output.shape[2:],
                mode=mode,
                align_corners=False,
            )

        return target_resized

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers and learning rate schedulers."""
        # Use the unified builder for Hydra configs
        optimizer = build_optimizer(self.cfg, self.model)
        scheduler = build_lr_scheduler(self.cfg, optimizer)
        
        # Determine monitor metric - check config first, then use fallbacks
        monitor_metric = 'train_loss_total_epoch'  # Default fallback
        
        if hasattr(self.cfg, 'optimization') and hasattr(self.cfg.optimization, 'scheduler'):
            monitor_metric = getattr(self.cfg.optimization.scheduler, 'monitor', monitor_metric)
        elif hasattr(self.cfg, 'optimization') and hasattr(self.cfg.optimization, 'scheduler') and hasattr(self.cfg.optimization.scheduler, 'monitor'):
            monitor_metric = self.cfg.optimization.scheduler.monitor
        
        # For ReduceLROnPlateau, use train loss if val loss not available
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if monitor_metric == 'val_loss_total' and not hasattr(self, '_val_metrics_available'):
                # Check if validation is configured
                val_check_interval = getattr(self.cfg, 'val_check_interval', 1.0)
                if val_check_interval <= 0:
                    monitor_metric = 'train_loss_total_epoch'
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': monitor_metric,
                'interval': 'epoch',
                'frequency': 1,
            }
        }

    def enable_nan_hooks(
        self,
        debug_on_nan: bool = True,
        verbose: bool = False,
        layer_types: Optional[Tuple] = None,
    ) -> NaNDetectionHookManager:
        """
        Enable forward hooks to detect NaN in intermediate layer outputs.

        This attaches hooks to all layers in the model that will check for NaN/Inf
        in layer outputs during forward pass. When NaN is detected, it will print
        diagnostics and optionally enter the debugger.

        Useful for debugging in pdb:
            (Pdb) pl_module.enable_nan_hooks()
            (Pdb) outputs = pl_module(batch['image'])
            # Will stop at first layer producing NaN

        Args:
            debug_on_nan: If True, enter pdb when NaN detected (default: True)
            verbose: If True, print stats for every layer (slow, default: False)
            layer_types: Tuple of layer types to hook (default: all common layers)

        Returns:
            NaNDetectionHookManager instance
        """
        if self._hook_manager is not None:
            print("⚠️  Hooks already enabled. Call disable_nan_hooks() first.")
            return self._hook_manager

        self._hook_manager = NaNDetectionHookManager(
            model=self.model,
            debug_on_nan=debug_on_nan,
            verbose=verbose,
            collect_stats=True,
            layer_types=layer_types,
        )

        return self._hook_manager

    def disable_nan_hooks(self):
        """
        Disable forward hooks for NaN detection.

        Removes all hooks that were attached by enable_nan_hooks().
        """
        if self._hook_manager is not None:
            self._hook_manager.remove_hooks()
            self._hook_manager = None
        else:
            print("⚠️  No hooks to remove.")

    def get_hook_stats(self) -> Optional[Dict[str, Dict[str, Any]]]:
        """
        Get statistics from NaN detection hooks.

        Returns:
            Dictionary mapping layer names to their statistics, or None if hooks not enabled
        """
        if self._hook_manager is not None:
            return self._hook_manager.get_stats()
        else:
            print("⚠️  Hooks not enabled. Call enable_nan_hooks() first.")
            return None

    def print_hook_summary(self):
        """
        Print summary of NaN detection hook statistics.

        Shows which layers detected NaN/Inf and how many times.
        """
        if self._hook_manager is not None:
            self._hook_manager.print_summary()
        else:
            print("⚠️  Hooks not enabled. Call enable_nan_hooks() first.")

    def check_for_nan(self, check_grads: bool = True, verbose: bool = True) -> dict:
        """
        Debug utility to check for NaN/Inf in model parameters and gradients.

        Useful when debugging in pdb. Call as: pl_module.check_for_nan()

        Args:
            check_grads: Also check gradients
            verbose: Print detailed information

        Returns:
            Dictionary with NaN/Inf information
        """
        nan_params = []
        inf_params = []
        nan_grads = []
        inf_grads = []

        for name, param in self.named_parameters():
            # Check parameters
            if torch.isnan(param).any():
                nan_params.append((name, param.shape))
                if verbose:
                    print(f"⚠️  NaN in parameter: {name}, shape={param.shape}")
            if torch.isinf(param).any():
                inf_params.append((name, param.shape))
                if verbose:
                    print(f"⚠️  Inf in parameter: {name}, shape={param.shape}")

            # Check gradients
            if check_grads and param.grad is not None:
                if torch.isnan(param.grad).any():
                    nan_grads.append((name, param.grad.shape))
                    if verbose:
                        print(f"⚠️  NaN in gradient: {name}, shape={param.grad.shape}")
                if torch.isinf(param.grad).any():
                    inf_grads.append((name, param.grad.shape))
                    if verbose:
                        print(f"⚠️  Inf in gradient: {name}, shape={param.grad.shape}")

        result = {
            'nan_params': nan_params,
            'inf_params': inf_params,
            'nan_grads': nan_grads,
            'inf_grads': inf_grads,
            'has_nan': len(nan_params) > 0 or len(nan_grads) > 0,
            'has_inf': len(inf_params) > 0 or len(inf_grads) > 0,
        }

        if verbose:
            if not result['has_nan'] and not result['has_inf']:
                print("✅ No NaN/Inf found in parameters or gradients")
            else:
                print(f"\n📊 Summary:")
                print(f"   NaN parameters: {len(nan_params)}")
                print(f"   Inf parameters: {len(inf_params)}")
                print(f"   NaN gradients: {len(nan_grads)}")
                print(f"   Inf gradients: {len(inf_grads)}")

        return result

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
    Factory function to create a Lightning module from configuration.
    
    Args:
        cfg: Hydra Config object or OmegaConf DictConfig
        model: Optional pre-built model
        
    Returns:
        ConnectomicsModule instance
    """
    return ConnectomicsModule(cfg=cfg, model=model)


__all__ = [
    'ConnectomicsModule',
    'create_lightning_module',
]

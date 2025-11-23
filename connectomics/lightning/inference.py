"""
Inference utilities for PyTorch Connectomics.

This module implements:
- Sliding window inference with MONAI
- Test-time augmentation (TTA)
- Post-processing transformations
- Instance segmentation decoding
- Output file writing
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from monai.inferers import SlidingWindowInferer
from monai.transforms import Flip

from ..config import Config


class InferenceManager:
    """
    Manager for inference operations including sliding window and TTA.

    This class handles:
    - Sliding window inference configuration
    - Test-time augmentation (TTA) with flips
    - Activation functions and channel selection
    - Post-processing transformations
    - Instance segmentation decoding
    - Output file writing

    Args:
        cfg: Hydra Config object or OmegaConf DictConfig
        model: PyTorch model (nn.Module)
        forward_fn: Forward function to use for predictions
    """

    def __init__(
        self,
        cfg: Config | DictConfig,
        model: nn.Module,
        forward_fn: callable,
    ):
        self.cfg = cfg
        self.model = model
        self.forward_fn = forward_fn
        self.sliding_inferer: Optional[SlidingWindowInferer] = None

        # Setup sliding window inferer if configured
        self.setup_sliding_window_inferer()

    def setup_sliding_window_inferer(self):
        """Initialize MONAI's SlidingWindowInferer based on config."""
        self.sliding_inferer = None

        if not hasattr(self.cfg, "inference"):
            return

        # For 2D models with do_2d=True, disable sliding window inference
        if getattr(self.cfg.data, "do_2d", False):
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
        system_batch_size = getattr(self.cfg.system.inference, "batch_size", 1)
        config_sw_batch_size = getattr(self.cfg.inference.sliding_window, "sw_batch_size", None)
        sw_batch_size = max(
            1, int(config_sw_batch_size if config_sw_batch_size is not None else system_batch_size)
        )
        mode = getattr(self.cfg.inference.sliding_window, "blending", "gaussian")
        sigma_scale = float(getattr(self.cfg.inference.sliding_window, "sigma_scale", 0.125))
        padding_mode = getattr(self.cfg.inference.sliding_window, "padding_mode", "constant")

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
        if hasattr(self.cfg, "inference") and hasattr(self.cfg.inference, "sliding_window"):
            window_size = getattr(self.cfg.inference.sliding_window, "window_size", None)
            if window_size:
                return tuple(int(v) for v in window_size)

        if hasattr(self.cfg, "model") and hasattr(self.cfg.model, "output_size"):
            output_size = getattr(self.cfg.model, "output_size", None)
            if output_size:
                roi_size = tuple(int(v) for v in output_size)
                # For 2D models with do_2d=True, convert to 3D ROI size
                if getattr(self.cfg.data, "do_2d", False) and len(roi_size) == 2:
                    roi_size = (1,) + roi_size  # Add depth dimension
                return roi_size

        if hasattr(self.cfg, "data") and hasattr(self.cfg.data, "patch_size"):
            patch_size = getattr(self.cfg.data, "patch_size", None)
            if patch_size:
                roi_size = tuple(int(v) for v in patch_size)
                # For 2D models with do_2d=True, convert to 3D ROI size
                if getattr(self.cfg.data, "do_2d", False) and len(roi_size) == 2:
                    roi_size = (1,) + roi_size  # Add depth dimension
                return roi_size

        return None

    def _resolve_inferer_overlap(
        self, roi_size: Tuple[int, ...]
    ) -> Union[float, Tuple[float, ...]]:
        """Resolve overlap parameter using inference config."""
        if not hasattr(self.cfg, "inference") or not hasattr(self.cfg.inference, "sliding_window"):
            return 0.5

        overlap = getattr(self.cfg.inference.sliding_window, "overlap", None)
        if overlap is not None:
            if isinstance(overlap, (list, tuple)):
                return tuple(float(max(0.0, min(o, 0.99))) for o in overlap)
            return float(max(0.0, min(overlap, 0.99)))

        stride = getattr(self.cfg.inference, "stride", None)
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

    def extract_main_output(
        self, outputs: Union[torch.Tensor, Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """Extract the primary segmentation logits from model outputs."""
        if isinstance(outputs, dict):
            if "output" not in outputs:
                raise KeyError("Expected key 'output' in model outputs for deep supervision.")
            return outputs["output"]
        return outputs

    def sliding_window_predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """Wrapper used by MONAI inferer to obtain primary model predictions."""
        outputs = self.forward_fn(inputs)
        return self.extract_main_output(outputs)

    def apply_tta_preprocessing(self, tensor: torch.Tensor) -> torch.Tensor:
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
        if not hasattr(self.cfg, "inference"):
            return tensor

        # Check for per-channel activations first (new approach)
        channel_activations = getattr(
            self.cfg.inference.test_time_augmentation, "channel_activations", None
        )

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

                if act == "sigmoid":
                    channel_tensor = torch.sigmoid(channel_tensor)
                elif act == "tanh":
                    channel_tensor = torch.tanh(channel_tensor)
                elif act == "softmax":
                    # Apply softmax across the channel dimension
                    if end_ch - start_ch > 1:
                        channel_tensor = torch.softmax(channel_tensor, dim=1)
                    else:
                        warnings.warn(
                            f"Softmax activation for single channel ({start_ch}:{end_ch}) is not meaningful. Skipping.",
                            UserWarning,
                        )
                elif act is None or (isinstance(act, str) and act.lower() == "none"):
                    # No activation (keep as is)
                    pass
                else:
                    raise ValueError(
                        f"Unknown activation '{act}' for channels {start_ch}:{end_ch}. "
                        f"Supported: 'sigmoid', 'softmax', 'tanh', None"
                    )

                activated_channels.append(channel_tensor)

            # Concatenate all channels back together
            tensor = torch.cat(activated_channels, dim=1)
        else:
            # Fall back to single activation for all channels (old approach)
            tta_act = getattr(self.cfg.inference.test_time_augmentation, "act", None)
            if tta_act is None:
                tta_act = getattr(self.cfg.inference, "output_act", None)

            # Apply activation function
            if tta_act == "softmax":
                tensor = torch.softmax(tensor, dim=1)
            elif tta_act == "sigmoid":
                tensor = torch.sigmoid(tensor)
            elif tta_act == "tanh":
                tensor = torch.tanh(tensor)
            elif tta_act is not None and tta_act.lower() != "none":
                warnings.warn(
                    f"Unknown TTA activation function '{tta_act}'. Supported: 'softmax', 'sigmoid', 'tanh', None",
                    UserWarning,
                )

        # Get TTA-specific channel selection or fall back to output_channel
        tta_channel = getattr(self.cfg.inference.test_time_augmentation, "select_channel", None)
        if tta_channel is None:
            tta_channel = getattr(self.cfg.inference, "output_channel", None)

        # Apply channel selection
        if tta_channel is not None:
            if isinstance(tta_channel, int):
                if tta_channel == -1:
                    # -1 means all channels
                    pass
                else:
                    # Single channel selection
                    tensor = tensor[:, tta_channel : tta_channel + 1, ...]
            elif isinstance(tta_channel, (list, tuple)):
                # Multiple channel selection
                tensor = tensor[:, list(tta_channel), ...]

        return tensor

    def predict_with_tta(
        self, images: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
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
        if (
            getattr(self.cfg.data, "do_2d", False) and images.size(2) == 1
        ):  # [B, C, 1, H, W] -> [B, C, H, W]
            images = images.squeeze(2)

        # Get TTA configuration (default to no augmentation if not configured)
        if hasattr(self.cfg, "inference") and hasattr(self.cfg.inference, "test_time_augmentation"):
            tta_flip_axes_config = getattr(
                self.cfg.inference.test_time_augmentation, "flip_axes", None
            )
        else:
            tta_flip_axes_config = None  # No config = no augmentation, just forward pass

        # Handle different tta_flip_axes configurations
        if tta_flip_axes_config is None:
            # null: No augmentation, but still apply tta_act and tta_channel (no ensemble)
            if self.sliding_inferer is not None:
                pred = self.sliding_inferer(inputs=images, network=self.sliding_window_predict)
            else:
                pred = self.sliding_window_predict(images)

            # Apply TTA preprocessing (activation + channel selection) even without augmentation
            ensemble_result = self.apply_tta_preprocessing(pred)
        else:
            if tta_flip_axes_config == "all" or tta_flip_axes_config == []:
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
                        network=self.sliding_window_predict,
                    )
                else:
                    pred = self.sliding_window_predict(x_aug)

                # Invert flip for prediction
                if flip_axes:
                    pred = Flip(spatial_axis=flip_axes)(pred)

                # Apply TTA preprocessing (activation + channel selection) if configured
                # Note: This is applied BEFORE ensembling for probability-space averaging
                pred_processed = self.apply_tta_preprocessing(pred)

                predictions.append(pred_processed)

            # Ensemble predictions based on configured mode
            ensemble_mode = getattr(
                self.cfg.inference.test_time_augmentation, "ensemble_mode", "mean"
            )
            stacked_preds = torch.stack(predictions, dim=0)

            if ensemble_mode == "mean":
                ensemble_result = stacked_preds.mean(dim=0)
            elif ensemble_mode == "min":
                ensemble_result = stacked_preds.min(dim=0)[0]  # min returns (values, indices)
            elif ensemble_mode == "max":
                ensemble_result = stacked_preds.max(dim=0)[0]  # max returns (values, indices)
            else:
                raise ValueError(
                    f"Unknown TTA ensemble mode: {ensemble_mode}. Use 'mean', 'min', or 'max'."
                )

        # Apply mask after ensemble if requested
        apply_mask = getattr(self.cfg.inference.test_time_augmentation, "apply_mask", False)
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


def apply_postprocessing(cfg: Config | DictConfig, data: np.ndarray) -> np.ndarray:
    """
    Apply postprocessing transformations to predictions.

    This method applies (in order):
    1. Binary postprocessing (morphological operations, connected components filtering)
    2. Scaling (intensity_scale or output_scale): Multiply predictions by scale factor
    3. Dtype conversion (intensity_dtype or output_dtype): Convert to target dtype with clamping

    Args:
        cfg: Configuration object
        data: Numpy array of predictions (B, C, D, H, W) or (B, D, H, W)

    Returns:
        Postprocessed predictions with applied transformations
    """
    if not hasattr(cfg, "inference") or not hasattr(cfg.inference, "postprocessing"):
        return data

    postprocessing = cfg.inference.postprocessing

    # Step 1: Apply binary postprocessing if configured
    binary_config = getattr(postprocessing, "binary", None)
    if binary_config is not None and getattr(binary_config, "enabled", False):
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

    # Step 2: Apply scaling if configured (support both new and legacy names)
    intensity_scale = getattr(postprocessing, "intensity_scale", None)
    output_scale = getattr(postprocessing, "output_scale", None)
    scale = intensity_scale if intensity_scale is not None else output_scale
    if scale is not None:
        data = data * scale

    # Step 3: Apply dtype conversion if configured (support both new and legacy names)
    intensity_dtype = getattr(postprocessing, "intensity_dtype", None)
    output_dtype = getattr(postprocessing, "output_dtype", None)
    target_dtype_str = intensity_dtype if intensity_dtype is not None else output_dtype

    if target_dtype_str is not None and target_dtype_str != "float32":
        # Map string dtype to numpy dtype
        dtype_map = {
            "uint8": np.uint8,
            "int8": np.int8,
            "uint16": np.uint16,
            "int16": np.int16,
            "uint32": np.uint32,
            "int32": np.int32,
            "float16": np.float16,
            "float32": np.float32,
            "float64": np.float64,
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
        if target_dtype_str == "uint8":
            data = np.clip(data, 0, 255)
        elif target_dtype_str == "int8":
            data = np.clip(data, -128, 127)
        elif target_dtype_str == "uint16":
            data = np.clip(data, 0, 65535)
        elif target_dtype_str == "int16":
            data = np.clip(data, -32768, 32767)
        elif target_dtype_str == "uint32":
            data = np.clip(data, 0, 4294967295)
        elif target_dtype_str == "int32":
            data = np.clip(data, -2147483648, 2147483647)

        data = data.astype(target_dtype)

    return data


def apply_decode_mode(cfg: Config | DictConfig, data: np.ndarray) -> np.ndarray:
    """
    Apply decode mode transformations to convert probability maps to instance segmentation.

    Args:
        cfg: Configuration object
        data: Numpy array of predictions (B, C, D, H, W) or (C, D, H, W)

    Returns:
        Decoded segmentation mask(s)
    """
    if not hasattr(cfg, "inference"):
        return data

    # Access decoding config directly from inference
    decode_modes = getattr(cfg.inference, "decoding", None)

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
        "binary_thresholding": decode_binary_thresholding,
        "decode_binary_thresholding": decode_binary_thresholding,
        "decode_binary_cc": decode_binary_cc,
        "decode_binary_watershed": decode_binary_watershed,
        "decode_binary_contour_cc": decode_binary_contour_cc,
        "decode_binary_contour_watershed": decode_binary_contour_watershed,
        "decode_binary_contour_distance_watershed": decode_binary_contour_distance_watershed,
        "decode_affinity_cc": decode_affinity_cc,
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
            fn_name = decode_cfg.name if hasattr(decode_cfg, "name") else decode_cfg.get("name")
            kwargs = (
                decode_cfg.kwargs if hasattr(decode_cfg, "kwargs") else decode_cfg.get("kwargs", {})
            )

            # Ensure kwargs is a mutable dict (convert from OmegaConf if needed)
            if hasattr(kwargs, "items"):
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
            if fn_name == "decode_binary_contour_distance_watershed":
                if "seed_threshold" in kwargs or "foreground_threshold" in kwargs:
                    warnings.warn(
                        "Detected legacy parameters (seed_threshold, contour_threshold, foreground_threshold) "
                        "for decode_binary_contour_distance_watershed. Converting to new tuple format "
                        "(binary_threshold, contour_threshold, distance_threshold). "
                        "Please update your config files to use the new format.",
                        DeprecationWarning,
                    )
                    # Convert old parameters to new tuple format
                    seed_thresh = kwargs.pop("seed_threshold", 0.9)
                    contour_thresh = kwargs.pop("contour_threshold", 0.8)
                    foreground_thresh = kwargs.pop("foreground_threshold", 0.85)

                    # Map old parameters to new tuple format
                    kwargs["binary_threshold"] = (seed_thresh, foreground_thresh)
                    kwargs["contour_threshold"] = (contour_thresh, 1.1)
                    kwargs["distance_threshold"] = (0.5, -0.5)

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


def resolve_output_filenames(
    cfg: Config | DictConfig, batch: Dict[str, Any], global_step: int = 0
) -> List[str]:
    """
    Extract and resolve filenames from batch metadata.

    Args:
        cfg: Configuration object
        batch: Batch dictionary containing metadata and images
        global_step: Current global step for fallback filename generation

    Returns:
        List of resolved filenames (without extension)
    """
    # Determine batch size from images
    images = batch.get("image")
    if images is not None:
        batch_size = images.shape[0]
    else:
        # Fallback: try to infer from metadata
        batch_size = 1

    meta = batch.get("image_meta_dict")
    filenames: List[Optional[str]] = []

    # Handle different metadata structures
    if isinstance(meta, list):
        # Multiple metadata dicts (one per sample in batch)
        for idx, meta_item in enumerate(meta):
            if isinstance(meta_item, dict):
                filename = meta_item.get("filename_or_obj")
                if filename is not None:
                    filenames.append(filename)
        # Update batch_size from metadata if we have a list
        batch_size = max(batch_size, len(filenames))
    elif isinstance(meta, dict):
        # Single metadata dict
        meta_filenames = meta.get("filename_or_obj")
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
            resolved_names.append(f"volume_{global_step}_{idx}")

    # Always return exactly batch_size filenames
    if len(resolved_names) < batch_size:
        print(
            f"  WARNING: resolve_output_filenames - Only {len(resolved_names)} filenames but batch_size is {batch_size}, padding with fallback names"
        )
        while len(resolved_names) < batch_size:
            resolved_names.append(f"volume_{global_step}_{len(resolved_names)}")

    return resolved_names


def write_outputs(
    cfg: Config | DictConfig,
    predictions: np.ndarray,
    filenames: List[str],
    suffix: str = "prediction",
) -> None:
    """
    Persist predictions to disk.

    Args:
        cfg: Configuration object
        predictions: Numpy array of predictions to save (B, C, D, H, W) or (B, D, H, W)
        filenames: List of filenames (without extension) for each sample in batch
        suffix: Suffix for output filename (default: "prediction")
    """
    if not hasattr(cfg, "inference"):
        return

    # Access output_path from nested data config
    output_dir_value = None
    if hasattr(cfg.inference, "data") and hasattr(cfg.inference.data, "output_path"):
        output_dir_value = cfg.inference.data.output_path
    if not output_dir_value:
        return

    output_dir = Path(output_dir_value)
    output_dir.mkdir(parents=True, exist_ok=True)

    from connectomics.data.io import write_hdf5

    # Get output transpose from postprocessing config
    output_transpose = []
    if hasattr(cfg.inference, "postprocessing"):
        output_transpose = getattr(cfg.inference.postprocessing, "output_transpose", [])

    # Get save_channels from sliding_window config (to reduce memory/storage)
    save_channels = None
    if hasattr(cfg.inference, "sliding_window"):
        save_channels = getattr(cfg.inference.sliding_window, "save_channels", None)

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
    else:
        # Single 2D image: (H, W) - treat as batch_size=1
        actual_batch_size = 1
        predictions = predictions[np.newaxis, ...]  # Add batch dimension

    # Verify filenames match actual batch size
    if len(filenames) != actual_batch_size:
        print(
            f"  WARNING: write_outputs - filename count ({len(filenames)}) does not match "
            f"batch size ({actual_batch_size}). Using first {min(len(filenames), actual_batch_size)} filenames."
        )

    # Write each sample in batch
    for idx in range(actual_batch_size):
        if idx >= len(filenames):
            print(f"  WARNING: write_outputs - no filename for batch index {idx}, skipping")
            continue

        sample = predictions[idx]
        filename = filenames[idx]
        output_path = output_dir / f"{filename}_{suffix}.h5"

        # Select specific channels if save_channels is specified
        # save_channels can be a list of indices like [0, 6] to save only channels 0 and 6
        # Skip if channels were already filtered during inference (check if num_channels matches)
        if save_channels is not None and sample.ndim >= 4:
            channel_indices = list(save_channels)
            num_channels = sample.shape[0]
            # Only filter if not already filtered (channels > len(save_channels))
            if num_channels > len(channel_indices):
                try:
                    sample = sample[channel_indices]
                    print(f"  Selected channels {channel_indices} from {predictions[idx].shape[0]} channels")
                except Exception as e:
                    print(f"  WARNING: write_outputs - channel selection failed: {e}, keeping all channels")

        # Transpose if needed (output_transpose: list of axis permutation)
        if output_transpose and len(output_transpose) > 0:
            try:
                sample = np.transpose(sample, axes=output_transpose)
            except Exception as e:
                print(f"  WARNING: write_outputs - transpose failed: {e}, keeping original shape")

        # Squeeze singleton dimensions (e.g., (1, 1, D, H, W) -> (D, H, W))
        sample = np.squeeze(sample)

        # Write HDF5 file
        try:
            write_hdf5(str(output_path), sample, dataset="main")
            print(f"  Saved prediction: {output_path} (shape: {sample.shape})")
        except Exception as e:
            print(f"  ERROR: write_outputs - failed to write {output_path}: {e}")

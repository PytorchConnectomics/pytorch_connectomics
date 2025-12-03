"""
Test-time augmentation utilities for PyTorch Connectomics.

This module isolates activation/channel preprocessing and flip-based ensembling
so it can be reused independent of the LightningModule implementation.
"""

from __future__ import annotations

from typing import Optional, Sequence
import warnings

import torch
from monai.transforms import Flip

try:
    from omegaconf import ListConfig
    HAS_OMEGACONF = True
except ImportError:
    HAS_OMEGACONF = False
    ListConfig = list  # Fallback


class TTAPredictor:
    """Encapsulates TTA preprocessing and flip ensemble logic."""

    def __init__(self, cfg, sliding_inferer, forward_fn):
        self.cfg = cfg
        self.sliding_inferer = sliding_inferer
        self.forward_fn = forward_fn

    def apply_preprocessing(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply activation and channel selection before TTA ensemble.

        Supports per-channel activations via channel_activations config:
        Format: [[start_ch, end_ch, 'activation'], ...]
        Examples:
          - [[0, 2, 'softmax'], [2, 3, 'sigmoid']]  # Softmax over channels 0-1, sigmoid for channel 2
          - [[0, 1, 'sigmoid'], [1, 2, 'sigmoid']]  # Sigmoid for channels 0 and 1 separately
        """
        if not hasattr(self.cfg, "inference") or not hasattr(
            self.cfg.inference, "test_time_augmentation"
        ):
            return tensor

        channel_activations = getattr(
            self.cfg.inference.test_time_augmentation, "channel_activations", None
        )

        if channel_activations is not None:
            activated_channels = []
            for config_entry in channel_activations:
                if len(config_entry) != 3:
                    raise ValueError(
                        f"Invalid channel_activations entry: {config_entry}. "
                        f"Expected [start_ch, end_ch, activation]."
                    )

                start_ch, end_ch, act = config_entry
                channel_tensor = tensor[:, start_ch:end_ch, ...]

                if act == "sigmoid":
                    channel_tensor = torch.sigmoid(channel_tensor)
                elif act == "scale_sigmoid":
                    channel_tensor = torch.sigmoid(0.2 * channel_tensor)
                elif act == "tanh":
                    channel_tensor = torch.tanh(channel_tensor)
                elif act == "softmax":
                    if end_ch - start_ch > 1:
                        channel_tensor = torch.softmax(channel_tensor, dim=1)
                    else:
                        warnings.warn(
                            f"Softmax activation for single channel ({start_ch}:{end_ch}) is not meaningful. Skipping.",
                            UserWarning,
                        )
                elif act is None or (isinstance(act, str) and act.lower() == "none"):
                    pass
                else:
                    raise ValueError(
                        f"Unknown activation '{act}' for channels {start_ch}:{end_ch}. "
                        f"Supported: 'sigmoid', 'scale_sigmoid', 'softmax', 'tanh', None"
                    )

                activated_channels.append(channel_tensor)

            tensor = torch.cat(activated_channels, dim=1)
        else:
            tta_act = getattr(self.cfg.inference.test_time_augmentation, "act", None)
            if tta_act is None:
                tta_act = getattr(self.cfg.inference, "output_act", None)

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

        tta_channel = getattr(self.cfg.inference.test_time_augmentation, "select_channel", None)
        if tta_channel is None:
            tta_channel = getattr(self.cfg.inference, "output_channel", None)

        if tta_channel is not None:
            if isinstance(tta_channel, int):
                if tta_channel != -1:
                    tensor = tensor[:, tta_channel : tta_channel + 1, ...]
            elif isinstance(tta_channel, (list, tuple, Sequence)):
                tensor = tensor[:, list(tta_channel), ...]

        return tensor

    def _run_network(self, images: torch.Tensor) -> torch.Tensor:
        """Run network with optional sliding window."""
        with torch.no_grad():
            if self.sliding_inferer is not None:
                return self.sliding_inferer(inputs=images, network=self._sliding_window_predict)
            return self._sliding_window_predict(images)

    def _sliding_window_predict(self, inputs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.forward_fn(inputs)
            if isinstance(outputs, dict):
                if "output" not in outputs:
                    raise KeyError("Expected key 'output' in model outputs for deep supervision.")
                return outputs["output"]
            return outputs

    def predict(self, images: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Perform test-time augmentation using flips, rotations, and ensemble predictions.

        Args:
            images: Input volume (B, C, D, H, W) or (B, D, H, W) or (D, H, W)
            mask: Optional mask to multiply with predictions after ensemble (B, C, D, H, W) or (B, 1, D, H, W)
        """
        if images.ndim == 3:
            images = images.unsqueeze(0).unsqueeze(0)
            warnings.warn(
                f"Input shape {images.shape} (D, H, W) automatically expanded to (1, 1, D, H, W)",
                UserWarning,
            )
        elif images.ndim == 4:
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

        if getattr(self.cfg.data, "do_2d", False) and images.size(2) == 1:
            images = images.squeeze(2)

        # Get TTA configuration
        if hasattr(self.cfg, "inference") and hasattr(self.cfg.inference, "test_time_augmentation"):
            tta_flip_axes_config = getattr(
                self.cfg.inference.test_time_augmentation, "flip_axes", None
            )
            tta_rotation90_axes_config = getattr(
                self.cfg.inference.test_time_augmentation, "rotation90_axes", None
            )
        else:
            tta_flip_axes_config = None
            tta_rotation90_axes_config = None

        # If no augmentation configured, run network once
        if tta_flip_axes_config is None and tta_rotation90_axes_config is None:
            pred = self._run_network(images)
            ensemble_result = self.apply_preprocessing(pred)
        else:
            # Parse flip axes configuration
            if tta_flip_axes_config == "all" or tta_flip_axes_config == []:
                if images.dim() == 5:
                    spatial_axes = [1, 2, 3]
                elif images.dim() == 4:
                    spatial_axes = [1, 2]
                else:
                    raise ValueError(f"Unsupported data dimensions: {images.dim()}")

                tta_flip_axes = [[]]
                for r in range(1, len(spatial_axes) + 1):
                    from itertools import combinations

                    for combo in combinations(spatial_axes, r):
                        tta_flip_axes.append(list(combo))
            elif HAS_OMEGACONF and isinstance(tta_flip_axes_config, ListConfig):
                # OmegaConf ListConfig - convert to regular list
                tta_flip_axes_config = [
                    list(item) if isinstance(item, ListConfig) else item
                    for item in tta_flip_axes_config
                ]
                tta_flip_axes = [[]] + tta_flip_axes_config
            elif isinstance(tta_flip_axes_config, (list, tuple)):
                tta_flip_axes = [[]] + list(tta_flip_axes_config)
            elif tta_flip_axes_config is None:
                tta_flip_axes = [[]]  # No flip augmentation
            else:
                raise ValueError(
                    f"Invalid tta_flip_axes: {tta_flip_axes_config}. "
                    f"Expected 'all' (8 flips), null (no aug), or list of flip axes."
                )

            # Parse rotation90 axes configuration
            # NOTE: We use torch.rot90 which expects full tensor axes
            # For 5D tensor (B, C, D, H, W): D=2, H=3, W=4
            # For 4D tensor (B, C, H, W): H=2, W=3
            # Spatial axes from config (0=D, 1=H, 2=W) need to be converted
            spatial_offset = 2  # Offset for batch and channel dimensions

            if tta_rotation90_axes_config == "all":
                if images.dim() == 5:
                    # For 3D data (B, C, D, H, W), all possible rotation planes
                    tta_rotation90_axes = [
                        (2, 3),  # D-H plane
                        (2, 4),  # D-W plane
                        (3, 4),  # H-W plane
                    ]
                elif images.dim() == 4:
                    # For 2D data (B, C, H, W), only one rotation plane
                    tta_rotation90_axes = [(2, 3)]  # H-W plane
                else:
                    raise ValueError(f"Unsupported data dimensions: {images.dim()}")
            elif HAS_OMEGACONF and isinstance(tta_rotation90_axes_config, ListConfig):
                # OmegaConf ListConfig - convert to list and process
                tta_rotation90_axes_config = list(tta_rotation90_axes_config)
                if len(tta_rotation90_axes_config) > 0:
                    tta_rotation90_axes = []
                    for axes in tta_rotation90_axes_config:
                        if HAS_OMEGACONF and isinstance(axes, ListConfig):
                            axes = list(axes)
                        if not isinstance(axes, (list, tuple)) or len(axes) != 2:
                            raise ValueError(
                                f"Invalid rotation plane: {axes}. Each plane must be a list/tuple of 2 axes."
                            )
                        # Convert spatial axes to full tensor axes
                        full_axes = tuple(a + spatial_offset for a in axes)
                        tta_rotation90_axes.append(full_axes)
                else:
                    tta_rotation90_axes = []
            elif isinstance(tta_rotation90_axes_config, (list, tuple)) and len(tta_rotation90_axes_config) > 0:
                # User-specified rotation planes: e.g., [[1, 2], [2, 3]]
                # Validate that each entry is a list/tuple of length 2
                tta_rotation90_axes = []
                for axes in tta_rotation90_axes_config:
                    if not isinstance(axes, (list, tuple)) or len(axes) != 2:
                        raise ValueError(
                            f"Invalid rotation plane: {axes}. Each plane must be a list/tuple of 2 axes."
                        )
                    # Convert spatial axes to full tensor axes
                    full_axes = tuple(a + spatial_offset for a in axes)
                    tta_rotation90_axes.append(full_axes)
            elif tta_rotation90_axes_config is None:
                tta_rotation90_axes = []  # No rotation augmentation
            else:
                raise ValueError(
                    f"Invalid tta_rotation90_axes: {tta_rotation90_axes_config}. "
                    f"Expected 'all', null (no rotation), or list of rotation planes like [[1, 2]]."
                )

            ensemble_mode = getattr(
                self.cfg.inference.test_time_augmentation, "ensemble_mode", "mean"
            )

            ensemble_result = None
            num_predictions = 0

            # Generate all combinations of (flip_axes, rotation_plane, k_rotations)
            # For each rotation plane, we try k=0,1,2,3 (0°, 90°, 180°, 270°)
            augmentation_combinations = []

            for flip_axes in tta_flip_axes:
                if not tta_rotation90_axes:
                    # No rotation: just add flip augmentation
                    augmentation_combinations.append((flip_axes, None, 0))
                else:
                    # Add all rotation combinations for this flip
                    for rotation_plane in tta_rotation90_axes:
                        for k in range(4):  # 0, 1, 2, 3 rotations (0°, 90°, 180°, 270°)
                            augmentation_combinations.append((flip_axes, rotation_plane, k))

            # Apply each augmentation combination
            for flip_axes, rotation_plane, k_rotations in augmentation_combinations:
                x_aug = images

                # Apply flip augmentation
                if flip_axes:
                    x_aug = Flip(spatial_axis=flip_axes)(x_aug)

                # Apply rotation augmentation using torch.rot90
                if rotation_plane is not None and k_rotations > 0:
                    x_aug = torch.rot90(x_aug, k=k_rotations, dims=rotation_plane)

                # Run network
                pred = self._run_network(x_aug)

                # Reverse rotation augmentation
                if rotation_plane is not None and k_rotations > 0:
                    pred = torch.rot90(pred, k=-k_rotations, dims=rotation_plane)

                # Reverse flip augmentation
                if flip_axes:
                    pred = Flip(spatial_axis=flip_axes)(pred)

                pred_processed = self.apply_preprocessing(pred)

                # Ensemble predictions
                if ensemble_result is None:
                    ensemble_result = pred_processed.clone()
                else:
                    if ensemble_mode == "mean":
                        ensemble_result = ensemble_result + (
                            pred_processed - ensemble_result
                        ) / (num_predictions + 1)
                    elif ensemble_mode == "min":
                        ensemble_result = torch.minimum(ensemble_result, pred_processed)
                    elif ensemble_mode == "max":
                        ensemble_result = torch.maximum(ensemble_result, pred_processed)
                    else:
                        raise ValueError(
                            f"Unknown TTA ensemble mode: {ensemble_mode}. Use 'mean', 'min', or 'max'."
                        )

                num_predictions += 1

                if torch.cuda.is_available() and num_predictions % 4 == 0:
                    torch.cuda.empty_cache()

        apply_mask = getattr(self.cfg.inference.test_time_augmentation, "apply_mask", False) if hasattr(self.cfg, "inference") and hasattr(self.cfg.inference, "test_time_augmentation") else False
        if apply_mask and mask is not None:
            if mask.shape != ensemble_result.shape:
                if not (mask.shape[1] == 1 and mask.shape[0] == ensemble_result.shape[0]):
                    warnings.warn(
                        f"Mask shape {mask.shape} does not match ensemble result shape {ensemble_result.shape}. "
                        f"Expected mask with C={ensemble_result.shape[1]} or C=1 channels. Skipping mask application.",
                        UserWarning,
                    )
                    return ensemble_result
            ensemble_result = ensemble_result * mask

        return ensemble_result

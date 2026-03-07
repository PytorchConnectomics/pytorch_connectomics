"""
Test-time augmentation utilities for PyTorch Connectomics.

This module isolates activation/channel preprocessing and flip-based ensembling
so it can be reused independent of the LightningModule implementation.
"""

from __future__ import annotations

from typing import Optional, Sequence
import warnings

import torch
import torch.nn.functional as F
from monai.transforms import Flip

from ..utils.channel_slices import resolve_channel_slice_bounds

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
        # Track activation types per channel for proper masking
        self.channel_activation_types = None
        self._last_distributed_sharding_active = False
        self._last_skip_postprocess_on_rank = False
        self._parse_channel_activations()

    @staticmethod
    def _resolve_channel_range(
        start_ch: int,
        end_ch: int,
        num_channels: int,
    ) -> tuple[int, int]:
        """Resolve TTA channel activation ranges using shared slice resolver.

        TTA keeps legacy boundary-offset behavior:
        - negative bounds resolve with ``+1`` offset
        - ``end=-1`` means "all remaining channels"
        """
        return resolve_channel_slice_bounds(
            (int(start_ch), int(end_ch)),
            num_channels=num_channels,
            context="channel_activations range",
            negative_index_offset=1,
            end_minus_one_full_span=True,
        )

    def _parse_channel_activations(self):
        """Parse channel_activations config to determine activation type per channel."""
        if not hasattr(self.cfg, "inference") or not hasattr(
            self.cfg.inference, "test_time_augmentation"
        ):
            return

        channel_activations = getattr(
            self.cfg.inference.test_time_augmentation, "channel_activations", None
        )

        if channel_activations is not None:
            channel_count_hint = int(getattr(getattr(self.cfg, "model", None), "out_channels", 0))
            if channel_count_hint <= 0:
                self.channel_activation_types = None
                return

            # Build a list mapping each output channel to its activation type
            self.channel_activation_types = []
            for config_entry in channel_activations:
                if len(config_entry) != 3:
                    continue
                start_ch, end_ch, act = config_entry
                start_ch, end_ch = self._resolve_channel_range(
                    start_ch, end_ch, channel_count_hint
                )
                # Add activation type for each channel in this range
                for _ in range(start_ch, end_ch):
                    self.channel_activation_types.append(act)

    def _get_tta_cfg(self):
        if not hasattr(self.cfg, "inference") or not hasattr(
            self.cfg.inference, "test_time_augmentation"
        ):
            return None
        return self.cfg.inference.test_time_augmentation

    @staticmethod
    def _distributed_context() -> tuple[bool, int, int]:
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return False, 0, 1
        return True, torch.distributed.get_rank(), torch.distributed.get_world_size()

    def is_distributed_sharding_enabled(self) -> bool:
        """Return whether distributed TTA sharding is active for the current process."""
        tta_cfg = self._get_tta_cfg()
        is_dist, _rank, world_size = self._distributed_context()
        return bool(
            tta_cfg is not None
            and getattr(tta_cfg, "enabled", False)
            and getattr(tta_cfg, "distributed_sharding", False)
            and is_dist
            and world_size > 1
        )

    def should_skip_postprocess_on_rank(self) -> bool:
        """Return True on nonzero DDP ranks after distributed TTA reduction."""
        return self._last_distributed_sharding_active and self._last_skip_postprocess_on_rank

    def _reduce_cpu_tensor_to_rank_zero(
        self,
        tensor: torch.Tensor,
        *,
        op,
        reduction_device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Reduce a large CPU tensor to rank 0 in manageable GPU chunks."""
        is_dist, rank, _world_size = self._distributed_context()
        if not is_dist:
            return tensor

        tta_cfg = self._get_tta_cfg()
        chunk_mb = int(getattr(tta_cfg, "distributed_reduce_chunk_mb", 128) or 128)
        chunk_bytes = max(1, chunk_mb) * 1024 * 1024

        flat_tensor = tensor.contiguous().view(-1)
        elems_per_chunk = max(1, chunk_bytes // max(1, flat_tensor.element_size()))
        reduced_flat = torch.empty_like(flat_tensor) if rank == 0 else None

        for start in range(0, flat_tensor.numel(), elems_per_chunk):
            end = min(start + elems_per_chunk, flat_tensor.numel())
            reduced_chunk = flat_tensor[start:end].to(device=reduction_device, non_blocking=False)
            torch.distributed.reduce(reduced_chunk, dst=0, op=op)
            if rank == 0:
                reduced_flat[start:end].copy_(reduced_chunk.cpu())

        if rank == 0:
            return reduced_flat.view_as(tensor)
        return None

    def _reduce_prediction_count_to_rank_zero(
        self,
        count: int,
        *,
        reduction_device: torch.device,
    ) -> int:
        """Reduce local TTA prediction counts to rank 0."""
        is_dist, rank, _world_size = self._distributed_context()
        if not is_dist:
            return int(count)

        count_tensor = torch.tensor([int(count)], device=reduction_device, dtype=torch.int64)
        torch.distributed.reduce(count_tensor, dst=0, op=torch.distributed.ReduceOp.SUM)
        if rank == 0:
            return int(count_tensor.item())
        return 0

    def apply_preprocessing(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply activation and channel selection before TTA ensemble.

        Supports per-channel activations via channel_activations config:
        Format: [[start_ch, end_ch, 'activation'], ...]
        Examples:
          - [[0, 2, 'softmax'], [2, 3, 'sigmoid']]  # Softmax over channels
            0-1, sigmoid for channel 2
          - [[0, 1, 'sigmoid'], [1, 2, 'sigmoid']]  # Sigmoid for channels 0
            and 1 separately
        
        MEMORY OPTIMIZATION: Activations are applied in-place to avoid creating
        intermediate tensors that would double/triple GPU memory usage.
        For a volume with shape (1, 7, 1022, 545, 1082):
          - OLD: Creates 3 separate tensors (pred + affinity + edt) = ~47 GB
          - NEW: Modifies tensor in-place = ~16 GB
        """
        if not hasattr(self.cfg, "inference") or not hasattr(
            self.cfg.inference, "test_time_augmentation"
        ):
            return tensor

        channel_activations = getattr(
            self.cfg.inference.test_time_augmentation, "channel_activations", None
        )

        if channel_activations is not None:
            activation_types = []
            # MEMORY OPTIMIZATION: Apply activations in-place instead of creating
            # intermediate tensors and concatenating them
            for config_entry in channel_activations:
                if len(config_entry) != 3:
                    raise ValueError(
                        f"Invalid channel_activations entry: {config_entry}. "
                        f"Expected [start_ch, end_ch, activation]."
                    )

                start_ch, end_ch, act = config_entry
                start_ch, end_ch = self._resolve_channel_range(
                    start_ch,
                    end_ch,
                    tensor.shape[1],
                )
                for _ in range(start_ch, end_ch):
                    activation_types.append(act)

                if act == "sigmoid":
                    tensor[:, start_ch:end_ch, ...] = torch.sigmoid(
                        tensor[:, start_ch:end_ch, ...]
                    )
                elif act == "scale_sigmoid":
                    tensor[:, start_ch:end_ch, ...] = torch.sigmoid(
                        0.2 * tensor[:, start_ch:end_ch, ...]
                    )
                elif act == "tanh":
                    tensor[:, start_ch:end_ch, ...] = torch.tanh(
                        tensor[:, start_ch:end_ch, ...]
                    )
                elif act == "softmax":
                    if end_ch - start_ch > 1:
                        tensor[:, start_ch:end_ch, ...] = torch.softmax(
                            tensor[:, start_ch:end_ch, ...], dim=1
                        )
                    else:
                        warnings.warn(
                            f"Softmax activation for single channel "
                            f"({start_ch}:{end_ch}) is not meaningful. Skipping.",
                            UserWarning,
                        )
                elif act is None or (isinstance(act, str) and act.lower() == "none"):
                    pass
                else:
                    raise ValueError(
                        f"Unknown activation '{act}' for channels {start_ch}:{end_ch}. "
                        f"Supported: 'sigmoid', 'scale_sigmoid', 'softmax', 'tanh', None"
                    )
            self.channel_activation_types = activation_types if activation_types else None
            
            # No need for torch.cat - tensor was modified in-place
        else:
            self.channel_activation_types = None
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
                    f"Unknown TTA activation function '{tta_act}'. "
                    f"Supported: 'softmax', 'sigmoid', 'tanh', None",
                    UserWarning,
                )

        tta_channel = getattr(self.cfg.inference.test_time_augmentation, "select_channel", None)
        if tta_channel is None:
            tta_channel = getattr(self.cfg.inference, "output_channel", None)

        if tta_channel is not None:
            # Handle "all" string (skip channel selection)
            if isinstance(tta_channel, str) and tta_channel.lower() == "all":
                pass  # Keep all channels
            elif isinstance(tta_channel, int):
                if tta_channel != -1:
                    tensor = tensor[:, tta_channel:tta_channel + 1, ...]
            elif isinstance(tta_channel, (list, tuple, Sequence)):
                # Convert to list of integers (handle both int and string numbers
                # from OmegaConf)
                channel_list = [int(ch) for ch in tta_channel]
                tensor = tensor[:, channel_list, ...]

        return tensor

    def _run_network(self, images: torch.Tensor) -> torch.Tensor:
        """Run network with optional sliding window."""
        with torch.no_grad():
            if self.sliding_inferer is not None:
                return self.sliding_inferer(inputs=images, network=self._sliding_window_predict)
            keep_input_on_cpu = bool(
                getattr(
                    getattr(getattr(self.cfg, "inference", None), "sliding_window", None),
                    "keep_input_on_cpu",
                    False,
                )
            )
            if keep_input_on_cpu and images.device.type == "cpu":
                raise RuntimeError(
                    "inference.sliding_window.keep_input_on_cpu=True requires sliding-window "
                    "inference to be enabled (set inference.sliding_window.window_size or model output size)."
                )
            return self._sliding_window_predict(images)

    def _sliding_window_predict(self, inputs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.forward_fn(inputs)
            if isinstance(outputs, dict):
                if "output" not in outputs:
                    raise KeyError("Expected key 'output' in model outputs for deep supervision.")
                return outputs["output"]
            return outputs

    def _validate_and_prepare_mask(
        self,
        mask: torch.Tensor,
        prediction: torch.Tensor,
        align_to_image: bool = False,
    ) -> torch.Tensor:
        """Validate mask shape against prediction and return a binarized mask tensor."""
        if mask is None:
            raise ValueError("Mask is None while mask application is enabled.")

        if mask.device != prediction.device:
            mask = mask.to(device=prediction.device, non_blocking=True)

        # Normalize dimensions to match prediction tensor rank.
        if mask.ndim == prediction.ndim - 1:
            mask = mask.unsqueeze(1)
        elif mask.ndim == prediction.ndim - 2:
            mask = mask.unsqueeze(0).unsqueeze(0)

        # Handle 2D inference case where prediction is 4D and mask may still be 5D with singleton depth.
        if prediction.ndim == 4 and mask.ndim == 5 and mask.shape[2] == 1:
            mask = mask.squeeze(2)

        if mask.ndim != prediction.ndim:
            raise ValueError(
                f"Mask rank {mask.ndim} does not match prediction rank {prediction.ndim}. "
                f"mask.shape={tuple(mask.shape)}, prediction.shape={tuple(prediction.shape)}"
            )

        # Batch handling: allow singleton mask batch to broadcast.
        if mask.shape[0] != prediction.shape[0]:
            if mask.shape[0] == 1:
                expand_shape = (prediction.shape[0],) + tuple(mask.shape[1:])
                mask = mask.expand(expand_shape)
            else:
                raise ValueError(
                    f"Mask batch {mask.shape[0]} does not match prediction batch "
                    f"{prediction.shape[0]}."
                )

        # Channel handling: allow single-channel mask or per-channel mask.
        if mask.shape[1] not in (1, prediction.shape[1]):
            raise ValueError(
                f"Mask channels {mask.shape[1]} incompatible with prediction channels "
                f"{prediction.shape[1]}. Expected C=1 or C={prediction.shape[1]}."
            )

        # Spatial handling is strict: mismatched shapes indicate a data/config bug.
        if mask.shape[2:] != prediction.shape[2:]:
            if align_to_image:
                deltas = [t - s for s, t in zip(mask.shape[2:], prediction.shape[2:])]
                # Center align by pad/crop mask to match prediction.
                for spatial_idx, delta in enumerate(deltas):
                    dim = 2 + spatial_idx
                    if delta == 0:
                        continue
                    if delta < 0:
                        # Crop mask if it's larger than prediction on this axis.
                        target = prediction.shape[dim]
                        start = (-delta) // 2
                        end = start + target
                        slices = [slice(None)] * mask.ndim
                        slices[dim] = slice(start, end)
                        mask = mask[tuple(slices)]
                    else:
                        # Pad mask if it's smaller than prediction on this axis.
                        pad_before = delta // 2
                        pad_after = delta - pad_before
                        spatial_dims = mask.ndim - 2
                        pad_pairs = [(0, 0)] * spatial_dims
                        pad_pairs[spatial_idx] = (pad_before, pad_after)
                        pad = []
                        for before, after in reversed(pad_pairs):
                            pad.extend([before, after])
                        mask = F.pad(mask, tuple(pad), mode="constant", value=0)
            else:
                raise ValueError(
                    "Mask spatial shape must exactly match prediction spatial shape. "
                    f"Got mask.shape={tuple(mask.shape)} and prediction.shape={tuple(prediction.shape)}. "
                    "Fix test/tune mask preprocessing so they produce identical spatial dimensions."
                )

        return (mask > 0).to(dtype=prediction.dtype)

    def predict(
        self,
        images: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        mask_align_to_image: bool = False,
    ) -> torch.Tensor:
        """
        Perform test-time augmentation using flips, rotations, and ensemble predictions.

        Args:
            images: Input volume (B, C, D, H, W) or (B, D, H, W) or (D, H, W)
            mask: Optional mask to multiply with predictions after ensemble
                (B, C, D, H, W) or (B, 1, D, H, W)
            mask_align_to_image: If True, allow minor center pad/crop of mask to
                match prediction shape.
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
                "Input shape (B, D, H, W) automatically expanded to (B, 1, D, H, W)",
                UserWarning,
            )
        elif images.ndim != 5:
            raise ValueError(
                f"TTA requires 3D, 4D, or 5D input tensor. Got {images.ndim}D "
                f"tensor with shape {images.shape}. Expected shapes: (D, H, W), "
                f"(B, D, H, W), or (B, C, D, H, W)"
            )

        do_2d = bool(
            getattr(getattr(self.cfg.data, "train", None), "do_2d", False)
            or getattr(getattr(self.cfg.data, "val", None), "do_2d", False)
        )
        if do_2d and images.size(2) == 1:
            images = images.squeeze(2)

        self._last_distributed_sharding_active = False
        self._last_skip_postprocess_on_rank = False

        # Get TTA configuration (respect enabled flag for augmentations)
        tta_cfg = self._get_tta_cfg()
        if tta_cfg is not None:
            tta_enabled = getattr(tta_cfg, "enabled", True)
            if tta_enabled:
                tta_flip_axes_config = getattr(tta_cfg, "flip_axes", None)
                tta_rotation90_axes_config = getattr(tta_cfg, "rotation90_axes", None)
            else:
                tta_flip_axes_config = None
                tta_rotation90_axes_config = None
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
                                f"Invalid rotation plane: {axes}. Each plane must be "
                                f"a list/tuple of 2 axes."
                            )
                        # Convert spatial axes to full tensor axes
                        full_axes = tuple(a + spatial_offset for a in axes)
                        tta_rotation90_axes.append(full_axes)
                else:
                    tta_rotation90_axes = []
            elif (
                isinstance(tta_rotation90_axes_config, (list, tuple))
                and len(tta_rotation90_axes_config) > 0
            ):
                # User-specified rotation planes: e.g., [[1, 2], [2, 3]]
                # Validate that each entry is a list/tuple of length 2
                tta_rotation90_axes = []
                for axes in tta_rotation90_axes_config:
                    if not isinstance(axes, (list, tuple)) or len(axes) != 2:
                        raise ValueError(
                            f"Invalid rotation plane: {axes}. Each plane must be "
                            f"a list/tuple of 2 axes."
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
            empty_cache_interval = int(
                getattr(self.cfg.inference.test_time_augmentation, "empty_cache_interval", 4)
            )
            distributed_sharding = self.is_distributed_sharding_enabled()
            self._last_distributed_sharding_active = distributed_sharding
            is_dist, rank, world_size = self._distributed_context()
            reduction_device = (
                images.device
                if images.is_cuda
                else torch.device(f"cuda:{torch.cuda.current_device()}")
                if torch.cuda.is_available()
                else torch.device("cpu")
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

            local_augmentation_combinations = augmentation_combinations
            if distributed_sharding:
                local_augmentation_combinations = augmentation_combinations[rank::world_size]
                if not local_augmentation_combinations:
                    raise RuntimeError(
                        "Distributed TTA sharding produced an empty augmentation shard for "
                        f"rank {rank}. Reduce the GPU count or increase TTA variants."
                    )
                if rank == 0:
                    print(
                        "  Distributed TTA sharding active: "
                        f"{len(augmentation_combinations)} total pass(es), world_size={world_size}, "
                        f"passes/rank≈{len(local_augmentation_combinations)}"
                    )

            # Apply each augmentation combination
            for flip_axes, rotation_plane, k_rotations in local_augmentation_combinations:
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
                    ensemble_result = pred_processed.clone().to(dtype=torch.float32)
                else:
                    if ensemble_mode == "mean":
                        if distributed_sharding:
                            ensemble_result += pred_processed.to(dtype=torch.float32)
                        else:
                            ensemble_result = ensemble_result + (
                                pred_processed.to(dtype=torch.float32) - ensemble_result
                            ) / (num_predictions + 1)
                    elif ensemble_mode == "min":
                        ensemble_result = torch.minimum(
                            ensemble_result, pred_processed.to(dtype=torch.float32)
                        )
                    elif ensemble_mode == "max":
                        ensemble_result = torch.maximum(
                            ensemble_result, pred_processed.to(dtype=torch.float32)
                        )
                    else:
                        raise ValueError(
                            f"Unknown TTA ensemble mode: {ensemble_mode}. "
                            f"Use 'mean', 'min', or 'max'."
                        )

                num_predictions += 1

                if (
                    torch.cuda.is_available()
                    and empty_cache_interval > 0
                    and num_predictions % empty_cache_interval == 0
                ):
                    torch.cuda.empty_cache()

            if distributed_sharding:
                if ensemble_mode == "mean":
                    reduced_sum = self._reduce_cpu_tensor_to_rank_zero(
                        ensemble_result,
                        op=torch.distributed.ReduceOp.SUM,
                        reduction_device=reduction_device,
                    )
                    total_predictions = self._reduce_prediction_count_to_rank_zero(
                        num_predictions,
                        reduction_device=reduction_device,
                    )
                    if rank == 0:
                        if total_predictions <= 0:
                            raise RuntimeError(
                                "Distributed TTA sharding reduced zero predictions on rank 0."
                            )
                        ensemble_result = reduced_sum / float(total_predictions)
                elif ensemble_mode == "min":
                    ensemble_result = self._reduce_cpu_tensor_to_rank_zero(
                        ensemble_result,
                        op=torch.distributed.ReduceOp.MIN,
                        reduction_device=reduction_device,
                    )
                elif ensemble_mode == "max":
                    ensemble_result = self._reduce_cpu_tensor_to_rank_zero(
                        ensemble_result,
                        op=torch.distributed.ReduceOp.MAX,
                        reduction_device=reduction_device,
                    )

                if rank != 0:
                    self._last_skip_postprocess_on_rank = True
                    return torch.empty(0, device=images.device if images.is_cuda else "cpu")

        apply_mask = (
            getattr(self.cfg.inference.test_time_augmentation, "apply_mask", True)
            if hasattr(self.cfg, "inference")
            and hasattr(self.cfg.inference, "test_time_augmentation")
            else True
        )
        if apply_mask and mask is not None:
            mask = self._validate_and_prepare_mask(
                mask,
                ensemble_result,
                align_to_image=mask_align_to_image,
            )

            # Apply activation-aware masking: use minimum value for each activation type
            # - sigmoid/softmax: min=0 (masked regions should be 0)
            # - tanh: min=-1 (masked regions should be -1)
            if self.channel_activation_types is not None and len(self.channel_activation_types) == ensemble_result.shape[1]:
                # Per-channel masking with activation-aware values
                for c, act_type in enumerate(self.channel_activation_types):
                    # Use per-channel mask if provided, otherwise broadcast channel-0 mask.
                    mask_channel = (
                        mask[:, c : c + 1]
                        if mask.shape[1] == ensemble_result.shape[1]
                        else mask[:, 0:1]
                    )
                    if act_type == "tanh":
                        # For tanh: mask * value + (1 - mask) * (-1)
                        # Where mask=1 keeps original value, mask=0 sets to -1
                        ensemble_result[:, c:c+1] = (
                            mask_channel * ensemble_result[:, c:c+1] +
                            (1 - mask_channel) * (-1.0)
                        )
                    else:
                        # For sigmoid/softmax/others: mask * value + (1 - mask) * 0
                        # This is equivalent to: ensemble_result * mask
                        ensemble_result[:, c:c+1] = mask_channel * ensemble_result[:, c:c+1]
            else:
                # Fallback: simple multiplication (assumes all channels want 0 for masked regions)
                ensemble_result = ensemble_result * mask

        return ensemble_result

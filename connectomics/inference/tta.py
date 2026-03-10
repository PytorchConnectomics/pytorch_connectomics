"""
Test-time augmentation utilities for PyTorch Connectomics.

This module isolates activation/channel preprocessing and flip-based ensembling
so it can be reused independent of the LightningModule implementation.
"""

from __future__ import annotations

import logging
from itertools import combinations
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from monai.transforms import Flip

from ..utils.channel_slices import resolve_channel_indices
from .sliding import is_2d_inference_mode

try:
    from omegaconf import ListConfig, OmegaConf

    HAS_OMEGACONF = True
except ImportError:
    HAS_OMEGACONF = False
    ListConfig = list  # Fallback

logger = logging.getLogger(__name__)


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

    def _resolve_channel_activation_specs(
        self,
        num_channels: int,
    ) -> list[tuple[list[int], Any]]:
        """Resolve configured channel activation specs to explicit channel indices."""
        tta_cfg = self._get_tta_cfg()
        channel_activations = getattr(tta_cfg, "channel_activations", None) if tta_cfg else None
        if not channel_activations:
            return []

        resolved_specs: list[tuple[list[int], Any]] = []
        used_channels: set[int] = set()
        for idx, entry in enumerate(channel_activations):
            if not isinstance(entry, dict):
                raise ValueError(
                    "channel_activations entries must be mappings with keys "
                    f"'channels' and 'activation', got {type(entry).__name__}."
                )
            if "channels" not in entry or "activation" not in entry:
                raise ValueError(
                    f"channel_activations[{idx}] must define both 'channels' and 'activation'."
                )

            channels = resolve_channel_indices(
                entry["channels"],
                num_channels=num_channels,
                context=f"channel_activations[{idx}].channels",
            )
            overlap = used_channels.intersection(channels)
            if overlap:
                overlap_str = ", ".join(str(ch) for ch in sorted(overlap))
                raise ValueError(
                    f"channel_activations[{idx}] overlaps already assigned channels: {overlap_str}."
                )
            used_channels.update(channels)
            resolved_specs.append((channels, entry["activation"]))

        return resolved_specs

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

            activation_types: list[Optional[str]] = [None] * channel_count_hint
            for channels, act in self._resolve_channel_activation_specs(channel_count_hint):
                for channel_idx in channels:
                    activation_types[channel_idx] = act
            self.channel_activation_types = (
                activation_types if any(act is not None for act in activation_types) else None
            )

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
        Format: [{channels: ':', activation: 'sigmoid'}, ...]

        MEMORY OPTIMIZATION: Activations are applied in-place to avoid creating
        intermediate tensors that would double/triple GPU memory usage.
        """
        if not hasattr(self.cfg, "inference") or not hasattr(
            self.cfg.inference, "test_time_augmentation"
        ):
            return tensor

        channel_activations = getattr(
            self.cfg.inference.test_time_augmentation, "channel_activations", None
        )

        if channel_activations is not None:
            activation_types: list[Optional[str]] = [None] * int(tensor.shape[1])
            for channels, act in self._resolve_channel_activation_specs(int(tensor.shape[1])):
                for channel_idx in channels:
                    activation_types[channel_idx] = act

                channel_list = list(channels)
                if act == "sigmoid":
                    tensor[:, channel_list, ...] = torch.sigmoid(tensor[:, channel_list, ...])
                elif act == "scale_sigmoid":
                    tensor[:, channel_list, ...] = torch.sigmoid(0.2 * tensor[:, channel_list, ...])
                elif act == "tanh":
                    tensor[:, channel_list, ...] = torch.tanh(tensor[:, channel_list, ...])
                elif act == "softmax":
                    if len(channel_list) > 1:
                        tensor[:, channel_list, ...] = torch.softmax(
                            tensor[:, channel_list, ...], dim=1
                        )
                    else:
                        logger.warning(
                            f"Softmax activation for single channel "
                            f"({channel_list[0]}) is not meaningful. Skipping."
                        )
                elif act is None or (isinstance(act, str) and act.lower() == "none"):
                    pass
                else:
                    raise ValueError(
                        f"Unknown activation '{act}' for channels {channel_list}. "
                        f"Supported: 'sigmoid', 'scale_sigmoid', 'softmax', 'tanh', None"
                    )
            self.channel_activation_types = (
                activation_types if any(act is not None for act in activation_types) else None
            )
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
                logger.warning(
                    f"Unknown TTA activation function '{tta_act}'. "
                    f"Supported: 'softmax', 'sigmoid', 'tanh', None"
                )

        tta_channel = getattr(self.cfg.inference.test_time_augmentation, "select_channel", None)
        if tta_channel is None:
            tta_channel = getattr(self.cfg.inference, "output_channel", None)

        if tta_channel is not None:
            channel_list = resolve_channel_indices(
                tta_channel,
                num_channels=int(tensor.shape[1]),
                context="select_channel",
            )
            if channel_list != list(range(int(tensor.shape[1]))):
                tensor = tensor[:, channel_list, ...]
            if self.channel_activation_types is not None:
                self.channel_activation_types = [
                    self.channel_activation_types[idx] for idx in channel_list
                ]

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
                    "inference to be enabled (set inference.sliding_window.window_size "
                    "or model output size)."
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

        mask = self._coerce_mask_to_tensor(mask)

        if mask.device != prediction.device:
            mask = mask.to(device=prediction.device, non_blocking=True)

        # Normalize dimensions to match prediction tensor rank.
        if mask.ndim == prediction.ndim - 1:
            mask = mask.unsqueeze(1)
        elif mask.ndim == prediction.ndim - 2:
            mask = mask.unsqueeze(0).unsqueeze(0)

        # Handle 2D inference case where prediction is 4D and mask may still be 5D.
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
                for spatial_idx, delta in enumerate(deltas):
                    dim = 2 + spatial_idx
                    if delta == 0:
                        continue
                    if delta < 0:
                        target = prediction.shape[dim]
                        start = (-delta) // 2
                        end = start + target
                        slices = [slice(None)] * mask.ndim
                        slices[dim] = slice(start, end)
                        mask = mask[tuple(slices)]
                    else:
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
                    f"Got mask.shape={tuple(mask.shape)} and "
                    f"prediction.shape={tuple(prediction.shape)}. "
                    "Fix test/tune mask preprocessing so they produce identical "
                    "spatial dimensions."
                )

        return (mask > 0).to(dtype=prediction.dtype)

    def _coerce_mask_to_tensor(self, mask: Any) -> torch.Tensor:
        """Convert nested mask containers from dataloader collation into a tensor."""
        while isinstance(mask, (list, tuple)) and len(mask) == 1:
            mask = mask[0]

        if isinstance(mask, np.ndarray):
            return torch.from_numpy(mask)
        if torch.is_tensor(mask):
            return mask

        if isinstance(mask, (list, tuple)):
            converted = [self._coerce_mask_to_tensor(item) for item in mask]
            if not converted:
                raise ValueError("Mask list is empty after collation.")
            try:
                return torch.stack(converted)
            except RuntimeError:
                if len(converted) == 1:
                    return converted[0]
                shapes = [tuple(t.shape) for t in converted]
                raise ValueError(
                    "Mask list contains tensors with incompatible shapes for stacking: " f"{shapes}"
                )

        raise TypeError(f"Unsupported mask type: {type(mask).__name__}")

    # ------------------------------------------------------------------
    # predict() decomposed into focused helpers
    # ------------------------------------------------------------------

    def _normalize_input(self, images: torch.Tensor) -> torch.Tensor:
        """Validate and expand input to 5D (B, C, D, H, W) or squeeze for 2D."""
        if images.ndim == 3:
            images = images.unsqueeze(0).unsqueeze(0)
            logger.warning(
                f"Input shape {images.shape} (D, H, W) automatically expanded "
                f"to (1, 1, D, H, W)"
            )
        elif images.ndim == 4:
            images = images.unsqueeze(1)
            logger.warning("Input shape (B, D, H, W) automatically expanded to (B, 1, D, H, W)")
        elif images.ndim != 5:
            raise ValueError(
                f"TTA requires 3D, 4D, or 5D input tensor. Got {images.ndim}D "
                f"tensor with shape {images.shape}. Expected shapes: (D, H, W), "
                f"(B, D, H, W), or (B, C, D, H, W)"
            )

        if is_2d_inference_mode(self.cfg) and images.size(2) == 1:
            images = images.squeeze(2)

        return images

    def _build_augmentation_combinations(
        self, tta_cfg, ndim: int
    ) -> list[tuple[list, Optional[tuple], int]]:
        """Parse TTA config and build list of (flip_axes, rotation_plane, k_rotations)."""
        tta_flip_axes_config = getattr(tta_cfg, "flip_axes", None)
        tta_rotation90_axes_config = getattr(tta_cfg, "rotation90_axes", None)

        # Resolve flip axes
        if tta_flip_axes_config == "all" or tta_flip_axes_config == []:
            if ndim == 5:
                spatial_axes = [1, 2, 3]
            elif ndim == 4:
                spatial_axes = [1, 2]
            else:
                raise ValueError(f"Unsupported data dimensions: {ndim}")

            tta_flip_axes = [[]]
            for r in range(1, len(spatial_axes) + 1):
                for combo in combinations(spatial_axes, r):
                    tta_flip_axes.append(list(combo))
        elif tta_flip_axes_config is None:
            tta_flip_axes = [[]]
        else:
            config_list = self._to_plain_list(tta_flip_axes_config)
            tta_flip_axes = [[]] + config_list

        # Resolve rotation axes
        spatial_offset = 2  # Offset for batch and channel dimensions

        if tta_rotation90_axes_config == "all":
            if ndim == 5:
                tta_rotation90_axes = [(2, 3), (2, 4), (3, 4)]
            elif ndim == 4:
                tta_rotation90_axes = [(2, 3)]
            else:
                raise ValueError(f"Unsupported data dimensions: {ndim}")
        elif tta_rotation90_axes_config is None:
            tta_rotation90_axes = []
        else:
            raw_axes = self._to_plain_list(tta_rotation90_axes_config)
            tta_rotation90_axes = []
            for axes in raw_axes:
                if not isinstance(axes, (list, tuple)) or len(axes) != 2:
                    raise ValueError(
                        f"Invalid rotation plane: {axes}. "
                        f"Each plane must be a list/tuple of 2 axes."
                    )
                full_axes = tuple(a + spatial_offset for a in axes)
                tta_rotation90_axes.append(full_axes)

        # Build all combinations
        augmentation_combinations = []
        for flip_axes in tta_flip_axes:
            if not tta_rotation90_axes:
                augmentation_combinations.append((flip_axes, None, 0))
            else:
                for rotation_plane in tta_rotation90_axes:
                    for k in range(4):
                        augmentation_combinations.append((flip_axes, rotation_plane, k))

        return augmentation_combinations

    @staticmethod
    def _to_plain_list(config_value) -> list:
        """Convert OmegaConf ListConfig (or plain list) to nested plain Python lists."""
        if HAS_OMEGACONF and isinstance(config_value, ListConfig):
            return OmegaConf.to_container(config_value, resolve=True)
        if isinstance(config_value, (list, tuple)):
            return list(config_value)
        return [config_value]

    def _run_ensemble(
        self,
        images: torch.Tensor,
        augmentation_combinations: list,
        ensemble_mode: str,
        empty_cache_interval: int,
        distributed_sharding: bool,
    ) -> tuple[torch.Tensor, int]:
        """Run TTA ensemble loop over augmentation combinations."""
        is_dist, rank, world_size = self._distributed_context()

        local_combinations = augmentation_combinations
        if distributed_sharding:
            local_combinations = augmentation_combinations[rank::world_size]
            if not local_combinations:
                raise RuntimeError(
                    "Distributed TTA sharding produced an empty augmentation shard for "
                    f"rank {rank}. Reduce the GPU count or increase TTA variants."
                )
            if rank == 0:
                logger.info(
                    "Distributed TTA sharding active: "
                    f"{len(augmentation_combinations)} total pass(es), "
                    f"world_size={world_size}, "
                    f"passes/rank~={len(local_combinations)}"
                )

        ensemble_result = None
        num_predictions = 0

        for flip_axes, rotation_plane, k_rotations in local_combinations:
            x_aug = images

            if flip_axes:
                x_aug = Flip(spatial_axis=flip_axes)(x_aug)

            if rotation_plane is not None and k_rotations > 0:
                x_aug = torch.rot90(x_aug, k=k_rotations, dims=rotation_plane)

            pred = self._run_network(x_aug)

            if rotation_plane is not None and k_rotations > 0:
                pred = torch.rot90(pred, k=-k_rotations, dims=rotation_plane)

            if flip_axes:
                pred = Flip(spatial_axis=flip_axes)(pred)

            pred_processed = self.apply_preprocessing(pred)

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

        return ensemble_result, num_predictions

    def _apply_distributed_reduction(
        self,
        ensemble_result: torch.Tensor,
        num_predictions: int,
        ensemble_mode: str,
        reduction_device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Reduce ensemble results across DDP ranks. Returns None on non-zero ranks."""
        _is_dist, rank, _world_size = self._distributed_context()

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
                return reduced_sum / float(total_predictions)
        elif ensemble_mode == "min":
            result = self._reduce_cpu_tensor_to_rank_zero(
                ensemble_result,
                op=torch.distributed.ReduceOp.MIN,
                reduction_device=reduction_device,
            )
            if rank == 0:
                return result
        elif ensemble_mode == "max":
            result = self._reduce_cpu_tensor_to_rank_zero(
                ensemble_result,
                op=torch.distributed.ReduceOp.MAX,
                reduction_device=reduction_device,
            )
            if rank == 0:
                return result

        return None  # non-zero ranks

    def _apply_mask_to_result(
        self,
        ensemble_result: torch.Tensor,
        mask: Optional[torch.Tensor],
        mask_align_to_image: bool,
    ) -> torch.Tensor:
        """Apply activation-aware masking to ensemble result."""
        apply_mask = (
            getattr(self.cfg.inference.test_time_augmentation, "apply_mask", True)
            if hasattr(self.cfg, "inference")
            and hasattr(self.cfg.inference, "test_time_augmentation")
            else True
        )
        if not apply_mask or mask is None:
            return ensemble_result

        try:
            mask = self._validate_and_prepare_mask(
                mask,
                ensemble_result,
                align_to_image=mask_align_to_image,
            )
        except TypeError as exc:
            logger.warning(
                "Skipping mask application because the provided mask payload "
                "is not a tensor-like volume: %s",
                exc,
            )
            return ensemble_result

        if (
            self.channel_activation_types is not None
            and len(self.channel_activation_types) == ensemble_result.shape[1]
        ):
            for c, act_type in enumerate(self.channel_activation_types):
                mask_channel = (
                    mask[:, c : c + 1]
                    if mask.shape[1] == ensemble_result.shape[1]
                    else mask[:, 0:1]
                )
                if act_type == "tanh":
                    ensemble_result[:, c : c + 1] = mask_channel * ensemble_result[:, c : c + 1] + (
                        1 - mask_channel
                    ) * (-1.0)
                else:
                    ensemble_result[:, c : c + 1] = mask_channel * ensemble_result[:, c : c + 1]
        else:
            ensemble_result = ensemble_result * mask

        return ensemble_result

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
            mask_align_to_image: If True, allow minor center pad/crop of mask
                to match prediction shape.
        """
        images = self._normalize_input(images)

        self._last_distributed_sharding_active = False
        self._last_skip_postprocess_on_rank = False

        tta_cfg = self._get_tta_cfg()
        tta_enabled = tta_cfg is not None and getattr(tta_cfg, "enabled", True)

        if not tta_enabled:
            pred = self._run_network(images)
            ensemble_result = self.apply_preprocessing(pred)
            return self._apply_mask_to_result(ensemble_result, mask, mask_align_to_image)

        augmentation_combinations = self._build_augmentation_combinations(tta_cfg, images.dim())

        # If only the identity augmentation, run network once
        if len(augmentation_combinations) == 1 and augmentation_combinations[0] == ([], None, 0):
            pred = self._run_network(images)
            ensemble_result = self.apply_preprocessing(pred)
            return self._apply_mask_to_result(ensemble_result, mask, mask_align_to_image)

        ensemble_mode = getattr(tta_cfg, "ensemble_mode", "mean")
        empty_cache_interval = int(getattr(tta_cfg, "empty_cache_interval", 4))
        distributed_sharding = self.is_distributed_sharding_enabled()
        self._last_distributed_sharding_active = distributed_sharding

        ensemble_result, num_predictions = self._run_ensemble(
            images,
            augmentation_combinations,
            ensemble_mode,
            empty_cache_interval,
            distributed_sharding,
        )

        if distributed_sharding:
            reduction_device = (
                images.device
                if images.is_cuda
                else (
                    torch.device(f"cuda:{torch.cuda.current_device()}")
                    if torch.cuda.is_available()
                    else torch.device("cpu")
                )
            )

            reduced = self._apply_distributed_reduction(
                ensemble_result,
                num_predictions,
                ensemble_mode,
                reduction_device,
            )

            _is_dist, rank, _world_size = self._distributed_context()
            if rank != 0:
                self._last_skip_postprocess_on_rank = True
                return torch.empty(0, device=images.device if images.is_cuda else "cpu")
            ensemble_result = reduced

        return self._apply_mask_to_result(ensemble_result, mask, mask_align_to_image)

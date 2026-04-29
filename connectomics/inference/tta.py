"""
Test-time augmentation utilities for PyTorch Connectomics.

This module isolates activation/channel preprocessing and flip-based ensembling
so it can be reused independent of the LightningModule implementation.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from monai.data.utils import dense_patch_slices
from monai.inferers.utils import _get_scan_interval, compute_importance_map

from ..utils.channel_slices import resolve_channel_indices
from ..utils.model_outputs import (
    resolve_output_channels,
    resolve_output_head,
    select_output_tensor,
)
from .sliding import (
    _extract_padded_patch_batch,
    _resolve_sliding_window_runtime,
    is_2d_inference_mode,
    resolve_inferer_roi_size,
)
from .tta_combinations import (
    _resolve_ensemble_mode_map,
    _resolve_spatial_dims,
    _to_plain_list,
    resolve_tta_augmentation_combinations,
)

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

logger = logging.getLogger(__name__)


class TTAPredictor:
    """Encapsulates TTA preprocessing and flip ensemble logic."""

    def __init__(self, cfg, sliding_inferer, forward_fn):
        self.cfg = cfg
        self.sliding_inferer = sliding_inferer
        self.forward_fn = forward_fn
        # Track activation types per channel for proper masking
        self.channel_activation_types = None
        self._requested_output_head_override: Optional[str] = None
        self._last_distributed_sharding_active = False
        self._last_skip_postprocess_on_rank = False
        self._parse_channel_activations()

    def _resolve_requested_output_head(
        self,
        *,
        purpose: str,
        allow_none: bool = True,
    ) -> Optional[str]:
        return resolve_output_head(
            self.cfg,
            requested_head=self._requested_output_head_override,
            purpose=purpose,
            allow_none=allow_none,
        )

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
            channel_count_hint = resolve_output_channels(
                self.cfg,
                requested_head=self._requested_output_head_override,
                purpose="TTA channel activation parsing",
                allow_ambiguous=True,
            )
            if channel_count_hint is None:
                channel_count_hint = int(
                    getattr(getattr(self.cfg, "model", None), "out_channels", 0)
                )
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
                channel_indexer: slice | list[int]
                if channel_list == list(range(channel_list[0], channel_list[-1] + 1)):
                    channel_indexer = slice(channel_list[0], channel_list[-1] + 1)
                else:
                    channel_indexer = channel_list
                channel_view = tensor[:, channel_indexer, ...]
                if act == "sigmoid":
                    if isinstance(channel_indexer, slice):
                        channel_view.sigmoid_()
                    else:
                        tensor[:, channel_list, ...] = torch.sigmoid(channel_view)
                elif isinstance(act, str) and (
                    act == "scale_sigmoid" or act.startswith("scale_sigmoid:")
                ):
                    # `scale_sigmoid` defaults to BANIS' temperature 0.2.
                    # Override via `scale_sigmoid:<float>` (e.g. `scale_sigmoid:0.5`).
                    scale = 0.2
                    if ":" in act:
                        try:
                            scale = float(act.split(":", 1)[1])
                        except ValueError as exc:
                            raise ValueError(
                                f"Invalid scale_sigmoid scale in '{act}'. "
                                "Expected 'scale_sigmoid:<float>'."
                            ) from exc
                    if isinstance(channel_indexer, slice):
                        channel_view.mul_(scale).sigmoid_()
                    else:
                        tensor[:, channel_list, ...] = torch.sigmoid(scale * channel_view)
                elif act == "tanh":
                    if isinstance(channel_indexer, slice):
                        channel_view.tanh_()
                    else:
                        tensor[:, channel_list, ...] = torch.tanh(channel_view)
                elif act == "softmax":
                    if len(channel_list) > 1:
                        tensor[:, channel_list, ...] = torch.softmax(channel_view, dim=1)
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
                        f"Supported: 'sigmoid', 'scale_sigmoid' (or 'scale_sigmoid:<float>'), "
                        f"'softmax', 'tanh', None"
                    )
            self.channel_activation_types = (
                activation_types if any(act is not None for act in activation_types) else None
            )
        else:
            self.channel_activation_types = None

        select_channel = getattr(self.cfg.inference, "select_channel", None)
        if select_channel is not None:
            channel_list = resolve_channel_indices(
                select_channel,
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

    def _run_direct_network(self, images: torch.Tensor) -> torch.Tensor:
        """Run the network directly on a tensor batch without whole-volume sliding."""
        with torch.no_grad():
            return self._sliding_window_predict(images)

    def _is_patch_first_local_tta_enabled(self) -> bool:
        tta_cfg = self._get_tta_cfg()
        return bool(
            tta_cfg is not None
            and getattr(tta_cfg, "enabled", False)
            and getattr(tta_cfg, "patch_first_local", False)
            and self.sliding_inferer is not None
        )

    def _sliding_window_predict(self, inputs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.forward_fn(inputs)
            requested_head = self._resolve_requested_output_head(
                purpose="inference output selection",
                allow_none=True,
            )
            primary_head = getattr(getattr(self.cfg, "model", None), "primary_head", None)
            selected_output, _ = select_output_tensor(
                outputs,
                requested_head=requested_head,
                primary_head=primary_head,
                purpose="inference output selection",
            )
            return selected_output

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
        """Parse TTA config and build list of (flip_axes, rotation_plane, k_rotations).

        All axes use 0-indexed spatial convention matching training augmentation:
        0=z (depth), 1=y (height), 2=x (width).  Internally converted to tensor
        dimensions by adding ``spatial_offset`` (2 for B,C prefix).
        """
        spatial_dims = _resolve_spatial_dims(ndim)
        spatial_offset = 2  # Offset for batch and channel dimensions
        augmentation_combinations = []
        for flip_axes, rotation_plane, k_rotations in resolve_tta_augmentation_combinations(
            tta_cfg,
            spatial_dims=spatial_dims,
        ):
            full_axes = None
            if rotation_plane is not None:
                full_axes = tuple(axis + spatial_offset for axis in rotation_plane)
            augmentation_combinations.append((flip_axes, full_axes, k_rotations))

        return augmentation_combinations

    @staticmethod
    def _to_plain_list(config_value) -> list:
        """Convert OmegaConf ListConfig (or plain list) to nested plain Python lists."""
        return _to_plain_list(config_value)

    def _run_ensemble(
        self,
        images: torch.Tensor,
        augmentation_combinations: list,
        ensemble_mode: Any,
        empty_cache_interval: int,
        distributed_sharding: bool,
        network_fn,
    ) -> tuple[torch.Tensor, int]:
        """Run TTA ensemble loop over augmentation combinations."""
        local_combinations = self._resolve_local_augmentation_combinations(
            augmentation_combinations,
            distributed_sharding=distributed_sharding,
        )

        ensemble_result = None
        num_predictions = 0

        spatial_offset = 2  # batch + channel dims

        for flip_axes, rotation_plane, k_rotations in local_combinations:
            x_aug = images

            if flip_axes:
                flip_dims = [
                    a + spatial_offset
                    for a in (flip_axes if isinstance(flip_axes, list) else [flip_axes])
                ]
                x_aug = torch.flip(x_aug, dims=flip_dims)

            if rotation_plane is not None and k_rotations > 0:
                x_aug = torch.rot90(x_aug, k=k_rotations, dims=rotation_plane)

            pred = network_fn(x_aug)

            if rotation_plane is not None and k_rotations > 0:
                pred = torch.rot90(pred, k=-k_rotations, dims=rotation_plane)

            if flip_axes:
                flip_dims = [
                    a + spatial_offset
                    for a in (flip_axes if isinstance(flip_axes, list) else [flip_axes])
                ]
                pred = torch.flip(pred, dims=flip_dims)

            pred_processed = self.apply_preprocessing(pred)

            ensemble_result, num_predictions = self._accumulate_ensemble_prediction(
                ensemble_result,
                pred_processed,
                ensemble_mode=ensemble_mode,
                num_predictions=num_predictions,
                distributed_sharding=distributed_sharding,
            )

            if (
                torch.cuda.is_available()
                and empty_cache_interval > 0
                and num_predictions % empty_cache_interval == 0
            ):
                torch.cuda.empty_cache()

        return ensemble_result, num_predictions

    def _resolve_local_augmentation_combinations(
        self,
        augmentation_combinations: list,
        *,
        distributed_sharding: bool,
    ) -> list:
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
        return local_combinations

    @staticmethod
    def _accumulate_single_mode(
        ensemble_result: torch.Tensor,
        pred_float: torch.Tensor,
        mode: str,
        num_predictions: int,
        distributed_sharding: bool,
        ch_slice: slice = slice(None),
    ) -> None:
        """Apply a single ensemble mode to a channel slice (in-place)."""
        if mode == "mean":
            if distributed_sharding:
                ensemble_result[:, ch_slice] += pred_float[:, ch_slice]
            else:
                delta = pred_float[:, ch_slice] - ensemble_result[:, ch_slice]
                ensemble_result[:, ch_slice] += delta / (num_predictions + 1)
        elif mode == "min":
            ensemble_result[:, ch_slice] = torch.minimum(
                ensemble_result[:, ch_slice], pred_float[:, ch_slice]
            )
        elif mode == "max":
            ensemble_result[:, ch_slice] = torch.maximum(
                ensemble_result[:, ch_slice], pred_float[:, ch_slice]
            )
        else:
            raise ValueError(f"Unknown TTA ensemble mode: {mode!r}. Use 'mean', 'min', or 'max'.")

    @staticmethod
    def _accumulate_ensemble_prediction(
        ensemble_result: Optional[torch.Tensor],
        pred_processed: torch.Tensor,
        *,
        ensemble_mode: Any,
        num_predictions: int,
        distributed_sharding: bool,
    ) -> tuple[torch.Tensor, int]:
        pred_float = pred_processed.to(dtype=torch.float32)
        if ensemble_result is None:
            return pred_float.clone(), 1

        num_channels = pred_float.shape[1]
        mode_map = _resolve_ensemble_mode_map(ensemble_mode, num_channels)

        # Group consecutive channels with the same mode for efficiency.
        i = 0
        while i < num_channels:
            mode = mode_map[i]
            j = i + 1
            while j < num_channels and mode_map[j] == mode:
                j += 1
            TTAPredictor._accumulate_single_mode(
                ensemble_result,
                pred_float,
                mode,
                num_predictions,
                distributed_sharding,
                ch_slice=slice(i, j),
            )
            i = j

        return ensemble_result, num_predictions + 1

    def _predict_prepared_tensor(
        self,
        images: torch.Tensor,
        mask: Optional[torch.Tensor],
        mask_align_to_image: bool,
        *,
        use_sliding: bool,
    ) -> torch.Tensor:
        tta_cfg = self._get_tta_cfg()
        tta_enabled = tta_cfg is not None and getattr(tta_cfg, "enabled", True)
        network_fn = self._run_network if use_sliding else self._run_direct_network

        if not tta_enabled:
            pred = network_fn(images)
            ensemble_result = self.apply_preprocessing(pred)
            return self._apply_mask_to_result(ensemble_result, mask, mask_align_to_image)

        augmentation_combinations = self._build_augmentation_combinations(tta_cfg, images.dim())

        if len(augmentation_combinations) == 1 and augmentation_combinations[0] == ([], None, 0):
            pred = network_fn(images)
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
            network_fn,
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

    def _predict_patch_first_local(
        self,
        images: torch.Tensor,
        mask: Optional[torch.Tensor],
        mask_align_to_image: bool,
    ) -> torch.Tensor:
        tta_cfg = self._get_tta_cfg()
        if tta_cfg is None:
            raise ValueError("Patch-first local TTA requires test-time augmentation config.")

        augmentation_combinations = self._build_augmentation_combinations(tta_cfg, images.dim())
        if len(augmentation_combinations) == 1 and augmentation_combinations[0] == ([], None, 0):
            return self._predict_prepared_tensor(
                images,
                mask,
                mask_align_to_image,
                use_sliding=True,
            )

        roi_size = resolve_inferer_roi_size(self.cfg)
        if roi_size is None:
            raise ValueError(
                "Patch-first local TTA requires inference.sliding_window.window_size "
                "or model.output_size to be configured."
            )

        runtime = _resolve_sliding_window_runtime(self.cfg, roi_size)
        infer_device = (
            torch.device(runtime["sw_device"])
            if runtime["sw_device"] is not None
            else images.device
        )
        accumulation_device = (
            torch.device(runtime["output_device"])
            if runtime["output_device"] is not None
            else torch.device("cpu")
        )
        ensemble_mode = getattr(tta_cfg, "ensemble_mode", "mean")
        empty_cache_interval = int(getattr(tta_cfg, "empty_cache_interval", 4))
        distributed_sharding = self.is_distributed_sharding_enabled()
        local_combinations = self._resolve_local_augmentation_combinations(
            augmentation_combinations,
            distributed_sharding=distributed_sharding,
        )
        self._last_distributed_sharding_active = distributed_sharding

        logger.info(
            "Patch-first local TTA enabled: each rank slides once over its shard and "
            "evaluates its local TTA variants inside each ROI batch."
        )
        if runtime["output_device"] is None and images.device.type == "cuda":
            logger.info(
                "Patch-first local TTA is accumulating full-volume outputs on CPU by default "
                "to avoid GPU OOM from per-augmentation buffers. Set "
                "`inference.sliding_window.output_device` to override this."
            )

        outputs = []
        spatial_dims = len(roi_size)
        _is_dist, rank, _world_size = self._distributed_context()
        for batch_idx in range(images.shape[0]):
            sample = images[batch_idx : batch_idx + 1]
            original_size = tuple(int(v) for v in sample.shape[-spatial_dims:])
            self._validate_patch_first_local_supported(
                local_combinations,
                image_size=original_size,
                roi_size=tuple(int(v) for v in roi_size),
            )

            padded_size = tuple(
                max(original_size[axis], int(roi_size[axis])) for axis in range(spatial_dims)
            )
            scan_overlap = runtime["overlap"]
            if not isinstance(scan_overlap, tuple):
                scan_overlap = tuple(float(scan_overlap) for _ in range(spatial_dims))
            scan_interval = _get_scan_interval(
                padded_size,
                roi_size,
                num_spatial_dims=spatial_dims,
                overlap=scan_overlap,
            )
            patch_slices = dense_patch_slices(
                padded_size,
                roi_size,
                scan_interval,
                return_slice=True,
            )
            num_patch_batches = max(
                1,
                (len(patch_slices) + runtime["sw_batch_size"] - 1) // runtime["sw_batch_size"],
            )
            importance_map = (
                compute_importance_map(
                    tuple(int(v) for v in roi_size),
                    mode=runtime["mode"],
                    sigma_scale=(
                        runtime["overlap"]
                        if runtime["mode"] == "constant"
                        else runtime["sigma_scale"]
                    ),
                    device=accumulation_device,
                    dtype=torch.float32,
                )
                .unsqueeze(0)
                .unsqueeze(0)
            )

            raw_accumulators: list[Optional[torch.Tensor]] = [None] * len(local_combinations)
            weight_accumulator = torch.zeros(
                (1, 1, *padded_size),
                device=accumulation_device,
                dtype=torch.float32,
            )
            tta_forward_calls = 0
            progress_bar = None
            if tqdm is not None:
                desc = f"Patch-first TTA x{len(local_combinations)}"
                if distributed_sharding:
                    desc += f" rank {rank}"
                progress_bar = tqdm(total=num_patch_batches, desc=desc, leave=True)

            try:
                for batch_start in range(0, len(patch_slices), runtime["sw_batch_size"]):
                    current_slices = patch_slices[
                        batch_start : batch_start + runtime["sw_batch_size"]
                    ]
                    image_batch, locations = _extract_padded_patch_batch(
                        sample,
                        current_slices,
                        roi_size=tuple(int(v) for v in roi_size),
                        padding_mode=runtime["padding_mode"],
                        cval=runtime["cval"],
                    )
                    image_batch = image_batch.to(device=infer_device, dtype=torch.float32)

                    for aug_idx, (flip_axes, rotation_plane, k_rotations) in enumerate(
                        local_combinations
                    ):
                        x_aug = image_batch
                        flip_dims = None
                        if flip_axes:
                            flip_dims = [
                                axis + 2
                                for axis in (
                                    flip_axes if isinstance(flip_axes, list) else [flip_axes]
                                )
                            ]
                            x_aug = torch.flip(x_aug, dims=flip_dims)

                        if rotation_plane is not None and k_rotations > 0:
                            x_aug = torch.rot90(x_aug, k=k_rotations, dims=rotation_plane)

                        pred = self._run_direct_network(x_aug)

                        if rotation_plane is not None and k_rotations > 0:
                            pred = torch.rot90(pred, k=-k_rotations, dims=rotation_plane)
                        if flip_dims:
                            pred = torch.flip(pred, dims=flip_dims)

                        if tuple(int(v) for v in pred.shape[2:]) != tuple(int(v) for v in roi_size):
                            raise RuntimeError(
                                "Patch-first local TTA requires patch predictions to preserve "
                                f"the ROI spatial shape. Got prediction.shape={tuple(pred.shape)} "
                                f"and roi_size={roi_size}."
                            )

                        pred = pred.detach().to(device=accumulation_device, dtype=torch.float32)
                        if raw_accumulators[aug_idx] is None:
                            raw_accumulators[aug_idx] = torch.zeros(
                                (1, int(pred.shape[1]), *padded_size),
                                device=accumulation_device,
                                dtype=torch.float32,
                            )

                        for patch_idx, location in enumerate(locations):
                            slices = tuple(
                                slice(
                                    int(location[axis]),
                                    int(location[axis]) + int(roi_size[axis]),
                                )
                                for axis in range(spatial_dims)
                            )
                            raw_accumulators[aug_idx][(slice(None), slice(None), *slices)] += (
                                pred[patch_idx : patch_idx + 1] * importance_map
                            )
                            if aug_idx == 0:
                                weight_accumulator[
                                    (slice(None), slice(None), *slices)
                                ] += importance_map

                        tta_forward_calls += 1
                        if (
                            torch.cuda.is_available()
                            and empty_cache_interval > 0
                            and tta_forward_calls % empty_cache_interval == 0
                        ):
                            torch.cuda.empty_cache()

                    if progress_bar is not None:
                        progress_bar.update(1)
            finally:
                if progress_bar is not None:
                    progress_bar.close()

            crop_slices = tuple(slice(0, int(size)) for size in original_size)
            ensemble_result = None
            num_predictions = 0
            for raw_accumulator in raw_accumulators:
                if raw_accumulator is None:
                    continue
                raw_accumulator /= torch.clamp_min(weight_accumulator, 1.0e-6)
                pred_processed = self.apply_preprocessing(
                    raw_accumulator[(slice(None), slice(None), *crop_slices)]
                )
                ensemble_result, num_predictions = self._accumulate_ensemble_prediction(
                    ensemble_result,
                    pred_processed,
                    ensemble_mode=ensemble_mode,
                    num_predictions=num_predictions,
                    distributed_sharding=distributed_sharding,
                )

            if ensemble_result is None:
                raise RuntimeError("Patch-first local TTA generated no predictions.")

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
                    outputs.append(torch.empty(0, device=accumulation_device))
                    continue
                ensemble_result = reduced

            outputs.append(ensemble_result)

        if any(output.numel() == 0 for output in outputs):
            return torch.empty(0, device=accumulation_device)

        return self._apply_mask_to_result(
            torch.cat(outputs, dim=0),
            mask,
            mask_align_to_image,
        )

    def _validate_patch_first_local_supported(
        self,
        augmentation_combinations: list,
        *,
        image_size: tuple[int, ...],
        roi_size: tuple[int, ...],
    ) -> None:
        spatial_offset = 2
        for _flip_axes, rotation_plane, k_rotations in augmentation_combinations:
            if rotation_plane is None or k_rotations % 2 == 0:
                continue

            plane_axes = tuple(int(axis) - spatial_offset for axis in rotation_plane)
            image_dims = tuple(int(image_size[axis]) for axis in plane_axes)
            roi_dims = tuple(int(roi_size[axis]) for axis in plane_axes)
            if len(set(image_dims)) != 1 or len(set(roi_dims)) != 1:
                raise ValueError(
                    "Patch-first local TTA only supports odd 90-degree rotations when the "
                    "rotated axes have equal image and ROI sizes. "
                    f"Got rotation_plane={rotation_plane}, image_size={image_size}, "
                    f"roi_size={roi_size}. Use flip-only TTA, constrain rotations to equal-sized "
                    "axes such as square XY inputs, or disable "
                    "`inference.test_time_augmentation.patch_first_local`."
                )

    def _apply_distributed_reduction(
        self,
        ensemble_result: torch.Tensor,
        num_predictions: int,
        ensemble_mode: Any,
        reduction_device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Reduce ensemble results across DDP ranks. Returns None on non-zero ranks."""
        _is_dist, rank, _world_size = self._distributed_context()
        self._validate_distributed_reduction_shape(
            ensemble_result,
            reduction_device=reduction_device,
        )

        num_channels = ensemble_result.shape[1]
        mode_map = _resolve_ensemble_mode_map(ensemble_mode, num_channels)
        unique_modes = set(mode_map)

        # Fast path: all channels share the same mode.
        if len(unique_modes) == 1:
            return self._apply_distributed_reduction_single_mode(
                ensemble_result,
                num_predictions,
                mode_map[0],
                reduction_device,
            )

        # Per-channel-group reduction: reduce once per unique mode via the
        # appropriate op, then stitch the channel slices back together.
        # We need SUM for mean channels, MIN for min channels, MAX for max.
        op_map = {
            "mean": torch.distributed.ReduceOp.SUM,
            "min": torch.distributed.ReduceOp.MIN,
            "max": torch.distributed.ReduceOp.MAX,
        }
        reduced_by_op: dict[str, torch.Tensor | None] = {}
        for mode in unique_modes:
            reduced_by_op[mode] = self._reduce_cpu_tensor_to_rank_zero(
                ensemble_result,
                op=op_map[mode],
                reduction_device=reduction_device,
            )

        total_predictions = None
        if "mean" in unique_modes:
            total_predictions = self._reduce_prediction_count_to_rank_zero(
                num_predictions,
                reduction_device=reduction_device,
            )

        if rank == 0:
            result = ensemble_result.clone()
            i = 0
            while i < num_channels:
                mode = mode_map[i]
                j = i + 1
                while j < num_channels and mode_map[j] == mode:
                    j += 1
                ch = slice(i, j)
                if mode == "mean":
                    if total_predictions <= 0:
                        raise RuntimeError(
                            "Distributed TTA sharding reduced zero predictions on rank 0."
                        )
                    result[:, ch] = reduced_by_op["mean"][:, ch] / float(total_predictions)
                else:
                    result[:, ch] = reduced_by_op[mode][:, ch]
                i = j
            return result

        return None  # non-zero ranks

    def _apply_distributed_reduction_single_mode(
        self,
        ensemble_result: torch.Tensor,
        num_predictions: int,
        ensemble_mode: str,
        reduction_device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Reduce with a single mode for all channels (original fast path)."""
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

    def _validate_distributed_reduction_shape(
        self,
        ensemble_result: torch.Tensor,
        *,
        reduction_device: torch.device,
    ) -> None:
        """Fail fast when DDP ranks try to reduce different TTA prediction shapes."""
        is_dist, _rank, world_size = self._distributed_context()
        if not is_dist or world_size <= 1:
            return

        max_dims = 6
        if ensemble_result.ndim > max_dims:
            raise RuntimeError(
                "Distributed TTA shape validation only supports tensors with up to "
                f"{max_dims} dimensions, got shape {tuple(ensemble_result.shape)}."
            )

        shape_info = torch.zeros(max_dims + 1, device=reduction_device, dtype=torch.int64)
        shape_info[0] = int(ensemble_result.ndim)
        if ensemble_result.ndim:
            shape_info[1 : 1 + ensemble_result.ndim] = torch.tensor(
                tuple(int(v) for v in ensemble_result.shape),
                device=reduction_device,
                dtype=torch.int64,
            )

        gathered = [torch.empty_like(shape_info) for _ in range(world_size)]
        torch.distributed.all_gather(gathered, shape_info)

        shapes: list[tuple[int, ...]] = []
        for gathered_shape in gathered:
            ndim = int(gathered_shape[0].item())
            shapes.append(tuple(int(v.item()) for v in gathered_shape[1 : 1 + ndim]))

        if any(shape != shapes[0] for shape in shapes[1:]):
            shape_summary = ", ".join(
                f"rank {rank_idx}: {shape}" for rank_idx, shape in enumerate(shapes)
            )
            raise RuntimeError(
                "Distributed TTA sharding requires every DDP rank to reduce predictions "
                f"with the same shape, got {shape_summary}. This usually means multiple "
                "test volumes were sharded across ranks; disable "
                "`inference.test_time_augmentation.distributed_sharding` for multi-volume "
                "tests."
            )

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
        requested_head: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Perform test-time augmentation using flips, rotations, and ensemble predictions.

        Args:
            images: Input volume (B, C, D, H, W) or (B, D, H, W) or (D, H, W)
            mask: Optional mask to multiply with predictions after ensemble
            mask_align_to_image: If True, allow minor center pad/crop of mask
                to match prediction shape.
            requested_head: Optional named output head override for this inference call.
        """
        previous_head_override = self._requested_output_head_override
        override_changed = requested_head != previous_head_override
        if requested_head is not None:
            resolve_output_head(
                self.cfg,
                requested_head=requested_head,
                purpose="inference output selection",
                allow_none=False,
            )

        self._requested_output_head_override = requested_head
        if override_changed:
            self._parse_channel_activations()

        try:
            images = self._normalize_input(images)

            self._last_distributed_sharding_active = False
            self._last_skip_postprocess_on_rank = False
            if self._is_patch_first_local_tta_enabled():
                return self._predict_patch_first_local(images, mask, mask_align_to_image)
            return self._predict_prepared_tensor(
                images,
                mask,
                mask_align_to_image,
                use_sliding=True,
            )
        finally:
            self._requested_output_head_override = previous_head_override
            if override_changed:
                self._parse_channel_activations()

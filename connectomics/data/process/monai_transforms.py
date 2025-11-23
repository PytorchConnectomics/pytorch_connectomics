"""
MONAI-native transforms for PyTorch Connectomics processing.

This module provides MONAI MapTransform implementations of all the processing
functions previously handled by the custom DataProcessor system.
"""

from __future__ import annotations
from copy import deepcopy
from typing import Callable, Dict, Any, List, Optional, Sequence, Union, Tuple
import numpy as np
import torch
from monai.config import KeysCollection
from monai.transforms import MapTransform
from monai.utils import ensure_tuple_rep

# Import processing functions with correct names
from .target import seg_to_binary, seg_to_affinity, seg_to_instance_bd
from .target import seg_to_instance_edt, seg_to_semantic_edt, seg_to_polarity, seg_to_small_seg
from .segment import seg_selection
from .quantize import energy_quantize, decode_quantize
from .weight import seg_to_weights
from .distance import skeleton_aware_distance_transform


class SegToBinaryMaskd(MapTransform):
    """Convert segmentation to binary mask using MONAI MapTransform.

    Args:
        keys: Keys to transform
        segment_id: List of segment IDs to include as foreground.
                   Empty list [] means all non-zero labels (default).
                   E.g., [1, 2, 3] to select specific segments.
        allow_missing_keys: Whether to allow missing keys
    """

    def __init__(
        self,
        keys: KeysCollection,
        segment_id: List[int] = [],
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.segment_id = segment_id

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        for key in self.key_iterator(d):
            if key in d:
                d[key] = seg_to_binary(d[key], self.segment_id)
        return d


class SegToAffinityMapd(MapTransform):
    """Convert segmentation to affinity map using MONAI MapTransform."""

    def __init__(
        self,
        keys: KeysCollection,
        offsets: List[str] = ["1-1-0", "1-0-0", "0-1-0", "0-0-1"],
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.offsets = offsets

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        for key in self.key_iterator(d):
            if key in d:
                label = d[key]
                # Convert tensor to numpy if needed
                if isinstance(label, torch.Tensor):
                    label = label.detach().cpu().numpy()
                # Handle channel dimension: input may be [C, D, H, W] or [D, H, W]
                if label.ndim == 4 and label.shape[0] == 1:
                    label = label[0]  # Remove channel dim: [1, D, H, W] -> [D, H, W]
                elif label.ndim == 3 and label.shape[0] == 1:
                    # 2D case: [1, H, W] -> keep as is for 2D affinity
                    pass
                d[key] = seg_to_affinity(label, self.offsets)
        return d


class SegToInstanceBoundaryMaskd(MapTransform):
    """Convert segmentation to instance boundary mask using MONAI MapTransform.

    Args:
        keys: Keys to transform
        thickness: Thickness of the boundary (half-size of dilation struct) (default: 1)
        edge_mode: Edge detection mode - "all", "seg-all", or "seg-no-bg" (default: "seg-all")
        mode: '2d' for slice-by-slice or '3d' for full 3D boundary detection (default: '3d')
        allow_missing_keys: Whether to allow missing keys
    """

    def __init__(
        self,
        keys: KeysCollection,
        thickness: int = 1,
        edge_mode: str = "seg-all",
        mode: str = "3d",
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.thickness = thickness
        self.edge_mode = edge_mode
        self.mode = mode

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        for key in self.key_iterator(d):
            if key in d:
                d[key] = seg_to_instance_bd(d[key], self.thickness, self.edge_mode, self.mode)
        return d


class SegToInstanceEDTd(MapTransform):
    """Convert segmentation to instance EDT using MONAI MapTransform.

    Args:
        keys: Keys to transform
        mode: EDT computation mode: '2d' or '3d' (default: '2d')
        quantize: Whether to quantize the EDT values (default: False)
        allow_missing_keys: Whether to allow missing keys
    """

    def __init__(
        self,
        keys: KeysCollection,
        mode: str = "2d",
        quantize: bool = False,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.mode = mode
        self.quantize = quantize

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        for key in self.key_iterator(d):
            if key in d:
                d[key] = seg_to_instance_edt(d[key], mode=self.mode, quantize=self.quantize)
        return d


class SegToSkeletonAwareEDTd(MapTransform):
    """Convert segmentation to skeleton-aware distance transform using MONAI MapTransform.

    This transform implements the skeleton-based distance transform from:
    Lin, Zudi, et al. "Structure-Preserving Instance Segmentation via Skeleton-Aware
    Distance Transform." MICCAI 2023.

    The skeleton-aware EDT computes a normalized distance that combines:
    - Distance from skeleton (guides reconstruction toward medial axis)
    - Distance from boundary (standard EDT behavior)
    This preserves thin structures and improves topology in instance segmentation.

    Args:
        keys: Keys to transform
        bg_value: Background value to assign to non-object pixels (default: -1.0)
        relabel: Whether to relabel connected components (default: True)
        padding: Whether to pad the array before computing distance (default: False)
        resolution: Pixel/voxel resolution for anisotropic data (default: (1.0, 1.0))
        alpha: Exponent controlling skeleton influence strength (default: 0.8)
                Higher values increase skeleton influence
        smooth: Whether to apply Gaussian smoothing to object boundaries (default: True)
        smooth_skeleton_only: If True, only smooth the skeleton mask; if False, smooth
                              the entire object mask (default: True)
        allow_missing_keys: Whether to allow missing keys
    """

    def __init__(
        self,
        keys: KeysCollection,
        bg_value: float = -1.0,
        relabel: bool = True,
        padding: bool = False,
        resolution: Tuple[float, float] = (1.0, 1.0),
        alpha: float = 0.8,
        smooth: bool = True,
        smooth_skeleton_only: bool = True,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.bg_value = bg_value
        self.relabel = relabel
        self.padding = padding
        self.resolution = resolution
        self.alpha = alpha
        self.smooth = smooth
        self.smooth_skeleton_only = smooth_skeleton_only

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        for key in self.key_iterator(d):
            if key in d:
                label = d[key]
                # Handle different input dimensions
                # skeleton_aware_distance_transform expects 2D or 3D input
                # If input is 4D with channel dim, remove it
                if isinstance(label, np.ndarray):
                    label_np = label
                else:
                    label_np = np.array(label)

                # Remove channel dimension if present (e.g., [1, D, H, W] -> [D, H, W])
                if label_np.ndim == 4 and label_np.shape[0] == 1:
                    label_np = label_np[0]

                # Ensure resolution matches input dimensionality
                resolution = self.resolution
                if label_np.ndim == 3 and len(resolution) == 2:
                    # Extend 2D resolution to 3D by prepending anisotropic z-resolution
                    resolution = (1.0,) + tuple(resolution)
                elif label_np.ndim == 2 and len(resolution) == 3:
                    # Use only last 2 dimensions for 2D case
                    resolution = tuple(resolution[-2:])

                result = skeleton_aware_distance_transform(
                    label_np,
                    bg_value=self.bg_value,
                    relabel=self.relabel,
                    padding=self.padding,
                    resolution=resolution,
                    alpha=self.alpha,
                    smooth=self.smooth,
                    smooth_skeleton_only=self.smooth_skeleton_only,
                )
                d[key] = result
        return d


class SegToSemanticEDTd(MapTransform):
    """Convert segmentation to semantic EDT using MONAI MapTransform.

    Args:
        keys: Keys to transform
        mode: EDT computation mode: '2d' or '3d' (default: '2d')
        alpha_fore: Foreground distance weight (default: 8.0)
        alpha_back: Background distance weight (default: 50.0)
        allow_missing_keys: Whether to allow missing keys
    """

    def __init__(
        self,
        keys: KeysCollection,
        mode: str = "2d",
        alpha_fore: float = 8.0,
        alpha_back: float = 50.0,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.mode = mode
        self.alpha_fore = alpha_fore
        self.alpha_back = alpha_back

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        for key in self.key_iterator(d):
            if key in d:
                d[key] = seg_to_semantic_edt(
                    d[key], mode=self.mode, alpha_fore=self.alpha_fore, alpha_back=self.alpha_back
                )
        return d


class SegToFlowFieldd(MapTransform):
    """Convert segmentation to flow field using MONAI MapTransform."""

    def __init__(
        self,
        keys: KeysCollection,
        target_opt: List[str] = ["1"],
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.target_opt = target_opt

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        for key in self.key_iterator(d):
            if key in d:
                # Flow field not implemented - return identity
                d[key] = d[key]
        return d


class SegToSynapticPolarityd(MapTransform):
    """Convert segmentation to synaptic polarity using MONAI MapTransform.

    Args:
        keys: Keys to transform
        exclusive: If False, returns 3-channel non-exclusive masks (for BCE loss).
                  If True, returns single-channel exclusive classes (for CE loss).
        allow_missing_keys: Whether to allow missing keys
    """

    def __init__(
        self,
        keys: KeysCollection,
        exclusive: bool = False,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.exclusive = exclusive

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        for key in self.key_iterator(d):
            if key in d:
                d[key] = seg_to_polarity(d[key], exclusive=self.exclusive)
        return d


class SegToSmallObjectd(MapTransform):
    """Convert segmentation to small object mask using MONAI MapTransform.

    Args:
        keys: Keys to transform
        threshold: Maximum voxel count for objects to be considered small (default: 100)
        allow_missing_keys: Whether to allow missing keys
    """

    def __init__(
        self,
        keys: KeysCollection,
        threshold: int = 100,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.threshold = threshold

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        for key in self.key_iterator(d):
            if key in d:
                d[key] = seg_to_small_seg(d[key], threshold=self.threshold)
        return d


class ComputeBinaryRatioWeightd(MapTransform):
    """Compute binary ratio weights using MONAI MapTransform."""

    def __init__(
        self,
        keys: KeysCollection,
        target_opt: List[str] = ["1"],
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.target_opt = target_opt

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        for key in self.key_iterator(d):
            if key in d:
                d[key] = seg_to_weights([d[key]], [self.target_opt])[0]
        return d


class ComputeUNet3DWeightd(MapTransform):
    """Compute UNet3D weights using MONAI MapTransform."""

    def __init__(
        self,
        keys: KeysCollection,
        target_opt: List[str] = ["1", "1", "5.0", "0.3"],
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.target_opt = target_opt

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        for key in self.key_iterator(d):
            if key in d:
                d[key] = seg_to_weights([d[key]], [self.target_opt])[0]
        return d


class SegErosiond(MapTransform):
    """Apply morphological erosion to segmentation using MONAI MapTransform."""

    def __init__(
        self,
        keys: KeysCollection,
        kernel_size: int = 1,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.kernel_size = kernel_size

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        from skimage.morphology import erosion, disk
        import numpy as np

        d = dict(data)
        for key in self.key_iterator(d):
            if key in d:
                seg = d[key]
                # Create structuring element
                struct_elem = disk(self.kernel_size, dtype=bool)
                if seg.ndim == 3:
                    struct_elem = struct_elem[np.newaxis, :, :]

                result = seg.copy()
                for z in range(seg.shape[0]):
                    result[z] = erosion(seg[z], struct_elem[0] if seg.ndim == 3 else struct_elem)
                d[key] = result
        return d


class SegDilationd(MapTransform):
    """Apply morphological dilation to segmentation using MONAI MapTransform."""

    def __init__(
        self,
        keys: KeysCollection,
        kernel_size: int = 1,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.kernel_size = kernel_size

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        from skimage.morphology import dilation, disk
        import numpy as np

        d = dict(data)
        for key in self.key_iterator(d):
            if key in d:
                seg = d[key]
                # Create structuring element
                struct_elem = disk(self.kernel_size, dtype=bool)
                if seg.ndim == 3:
                    struct_elem = struct_elem[np.newaxis, :, :]

                result = seg.copy()
                for z in range(seg.shape[0]):
                    result[z] = dilation(seg[z], struct_elem[0] if seg.ndim == 3 else struct_elem)
                d[key] = result
        return d


class SegErosionInstanced(MapTransform):
    """Erode instance segmentation borders (Kisuk Lee's preprocessing for SNEMI3D).

    Marks voxels at boundaries between different segments as background.
    This is different from standard morphological erosion.
    """

    def __init__(
        self,
        keys: KeysCollection,
        tsz_h: int = 1,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.tsz_h = tsz_h

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        from .segment import seg_erosion_instance

        d = dict(data)
        for key in self.key_iterator(d):
            if key in d:
                d[key] = seg_erosion_instance(d[key].squeeze(), self.tsz_h)
        return d


class EnergyQuantized(MapTransform):
    """Quantize continuous energy maps using MONAI MapTransform.

    This transform converts continuous energy values to discrete quantized levels,
    useful for training neural networks on energy-based targets.
    """

    def __init__(
        self,
        keys: KeysCollection,
        levels: int = 10,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: Keys to be processed from the input dictionary.
            levels: Number of quantization levels. Default is 10.
            allow_missing_keys: Whether to ignore missing keys.
        """
        super().__init__(keys, allow_missing_keys)
        self.levels = levels

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        for key in self.key_iterator(d):
            if key in d:
                d[key] = energy_quantize(d[key], levels=self.levels)
        return d


class DecodeQuantized(MapTransform):
    """Decode quantized energy maps back to continuous values using MONAI MapTransform.

    This transform converts quantized discrete levels back to continuous energy values,
    typically used for inference or evaluation.
    """

    def __init__(
        self,
        keys: KeysCollection,
        mode: str = "max",
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: Keys to be processed from the input dictionary.
            mode: Decoding mode, either 'max' or 'mean'. Default is 'max'.
            allow_missing_keys: Whether to ignore missing keys.
        """
        super().__init__(keys, allow_missing_keys)
        if mode not in ["max", "mean"]:
            raise ValueError(f"Mode must be 'max' or 'mean', got {mode}")
        self.mode = mode

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        for key in self.key_iterator(d):
            if key in d:
                d[key] = decode_quantize(d[key], mode=self.mode)
        return d


class SegSelectiond(MapTransform):
    """Select specific segmentation indices using MONAI MapTransform.

    This transform selects only the specified label indices from a segmentation,
    renumbering them consecutively starting from 1.
    """

    def __init__(
        self,
        keys: KeysCollection,
        indices: Union[List[int], int],
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: Keys to be processed from the input dictionary.
            indices: List of label indices to select, or single index.
            allow_missing_keys: Whether to ignore missing keys.
        """
        super().__init__(keys, allow_missing_keys)
        self.indices = (
            ensure_tuple_rep(indices, 1) if not isinstance(indices, (list, tuple)) else indices
        )

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        for key in self.key_iterator(d):
            if key in d:
                d[key] = seg_selection(d[key], self.indices)
        return d


class MultiTaskLabelTransformd(MapTransform):
    """Generate multiple supervision targets from a single instance segmentation.

    The transform evaluates a sequence of configured target generators without
    making intermediate copies of the label. Each target can either contribute a
    channel to a stacked tensor (default) or write to its own key.

    Output Format:
        - Input: label with shape [D, H, W] or [1, D, H, W]
        - Output (stacked): [C, D, H, W] where C = sum of channels from all tasks
        - Output (non-stacked): Multiple keys, each with shape [C_i, D, H, W]
        - Single task: [1, D, H, W] (always has channel dimension)

    Example:
        Single task (binary):
            Input: [32, 256, 256]
            Output: [1, 32, 256, 256]

        Multi-task (binary + boundary + edt):
            Input: [32, 256, 256]
            Output: [3, 32, 256, 256]

        Multi-task (binary + affinity[3] + edt):
            Input: [32, 256, 256]
            Output: [5, 32, 256, 256]  # 1 + 3 + 1 channels
    """

    _TASK_REGISTRY: Dict[str, Callable[..., np.ndarray]] = {
        "binary": seg_to_binary,
        "affinity": seg_to_affinity,
        "instance_boundary": seg_to_instance_bd,
        "instance_edt": seg_to_instance_edt,
        "skeleton_aware_edt": skeleton_aware_distance_transform,
        "semantic_edt": seg_to_semantic_edt,
        "polarity": seg_to_polarity,
        "small_object": seg_to_small_seg,
        "energy_quantize": energy_quantize,
        "decode_quantize": decode_quantize,
    }
    _TASK_DEFAULTS: Dict[str, Dict[str, Any]] = {
        "binary": {},
        "affinity": {"offsets": ["1-1-0", "1-0-0", "0-1-0", "0-0-1"]},
        "instance_boundary": {"thickness": 1, "edge_mode": "seg-all", "mode": "3d"},
        "instance_edt": {"mode": "2d", "quantize": False},
        "skeleton_aware_edt": {
            "bg_value": -1.0,
            "relabel": True,
            "padding": False,
            "resolution": (1.0, 1.0, 1.0),
            "alpha": 0.8,
            "smooth": True,
            "smooth_skeleton_only": True,
        },
        "semantic_edt": {"mode": "2d", "alpha_fore": 8.0, "alpha_back": 50.0},
        "polarity": {"exclusive": False},
        "small_object": {"threshold": 100},
        "energy_quantize": {"levels": 10},
        "decode_quantize": {"mode": "max"},
    }

    def __init__(
        self,
        keys: KeysCollection,
        tasks: Sequence[Union[str, Dict[str, Any]]],
        *,
        stack_outputs: bool = True,
        output_dtype: Optional[torch.dtype] = torch.float32,
        retain_original: bool = False,
        output_key_format: str = "{key}_{task}",
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: Keys to be processed from the input dictionary.
            tasks: Sequence describing the targets to generate. Each entry can be
                either a string referencing :attr:`_TASK_REGISTRY` or a dict with
                ``{"name": ..., "kwargs": {...}, "output_key": ...}``.
            stack_outputs: If True, concatenate targets along a new channel axis
                and write them back to the input ``key``.
            output_dtype: Optional dtype applied to generated targets.
            retain_original: If True, keep the original label tensor alongside the
                generated outputs.
            output_key_format: Format string used when ``stack_outputs`` is False
                and a task does not supply ``output_key``. Receives ``key`` and
                ``task`` keyword arguments.
            allow_missing_keys: Whether missing keys are allowed.
        """
        super().__init__(keys, allow_missing_keys)
        self.stack_outputs = stack_outputs
        self.output_dtype = output_dtype
        self.retain_original = retain_original
        self.output_key_format = output_key_format

        self.task_specs = self._init_tasks(tasks)

    def _init_tasks(
        self,
        tasks: Sequence[Union[str, Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """Normalize task configuration into callable specs."""
        specs: List[Dict[str, Any]] = []
        for task in tasks:
            if isinstance(task, str):
                name = task
                kwargs = {}
                output_key = None
            elif isinstance(task, dict):
                name = task.get("name") or task.get("task") or task.get("type")
                if name is None:
                    raise ValueError(f"Task entry {task} missing 'name'/'task'/'type'.")
                kwargs = dict(task.get("kwargs", {}))
                output_key = task.get("output_key")
            else:
                raise TypeError(f"Unsupported task specification: {task!r}")

            if name not in self._TASK_REGISTRY:
                available = ", ".join(sorted(self._TASK_REGISTRY))
                raise KeyError(f"Unknown task '{name}'. Available: {available}")

            fn = self._TASK_REGISTRY[name]
            defaults = deepcopy(self._TASK_DEFAULTS.get(name, {}))
            kwargs = {**defaults, **kwargs}
            specs.append(
                {
                    "name": name,
                    "fn": fn,
                    "kwargs": kwargs,
                    "output_key": output_key,
                }
            )
        if not specs:
            raise ValueError("At least one task must be specified for MultiTaskLabelTransformd.")
        return specs

    def _prepare_label(self, label: Any) -> Tuple[np.ndarray, bool]:
        """Convert label to numpy without duplicating data where possible.

        MONAI transforms expect channel-first format [C, D, H, W], not batch-first [B, C, D, H, W].
        We should not remove the channel dimension.
        """
        if isinstance(label, torch.Tensor):
            label_cpu = label.detach().cpu()
            label_np = label_cpu.numpy()
            return label_np, False
        if isinstance(label, np.ndarray):
            return label, False
        return np.asarray(label), False

    def _to_tensor(self, array: np.ndarray, *, add_batch_dim: bool) -> torch.Tensor:
        # Ensure array is a proper numpy array (not a numpy scalar type like numpy.uint8)
        # torch.as_tensor cannot infer dtype from numpy scalar types
        if not isinstance(array, np.ndarray):
            array = np.asarray(array)
        # Convert to a supported dtype if needed (torch doesn't support all numpy dtypes)
        if array.dtype == np.uint8:
            array = array.astype(np.float32)
        elif array.dtype == np.uint16:
            array = array.astype(np.float32)
        elif array.dtype == np.int8:
            array = array.astype(np.int32)
        tensor = torch.as_tensor(array)
        if self.output_dtype is not None:
            tensor = tensor.to(self.output_dtype)
        if add_batch_dim:
            tensor = tensor.unsqueeze(0)
        return tensor

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        for key in self.key_iterator(d):
            if key not in d:
                continue

            label = d[key]
            label_np, had_batch_dim = self._prepare_label(label)

            # Determine if input is 2D or 3D based on original dimensions
            # After EnsureChannelFirstd: 2D images are [1, H, W], 3D volumes are [1, D, H, W]
            is_2d_input = label_np.ndim == 3 and label_np.shape[0] == 1
            is_3d_input = label_np.ndim == 4 and label_np.shape[0] == 1

            # Remove channel dimension (target functions don't expect it)
            if is_3d_input:
                label_np = label_np[0]  # [1, D, H, W] -> [D, H, W]
            elif is_2d_input:
                # For 2D, keep as [1, H, W] since some functions (boundary, edt) expect 3D input
                # even in 2D mode (they treat first dim as Z=1)
                pass  # Keep [1, H, W]

            outputs: List[np.ndarray] = []
            for spec in self.task_specs:
                try:
                    result = spec["fn"](label_np, **spec["kwargs"])
                except Exception as e:
                    raise RuntimeError(
                        f"Task '{spec['name']}' failed with error: {e}\n"
                        f"Label shape: {label_np.shape}, dtype: {label_np.dtype}\n"
                        f"Task kwargs: {spec['kwargs']}"
                    ) from e
                if result is None:
                    raise RuntimeError(
                        f"Task '{spec['name']}' returned None.\n"
                        f"Label shape: {label_np.shape}, dtype: {label_np.dtype}\n"
                        f"Task kwargs: {spec['kwargs']}"
                    )
                result_arr = np.asarray(
                    result, dtype=np.float32
                )  # Convert to float32 (handles bool->float)

                # Normalize output dimensions:
                # For 2D images (input [1, H, W]): functions return [H, W] or [1, H, W]
                # For 3D volumes (input [D, H, W]): functions return [D, H, W]
                # Goal: Add channel dimension to get [1, H, W] for 2D or [1, D, H, W] for 3D

                if is_2d_input:
                    # 2D case: some functions return [H, W], others return [1, H, W]
                    if result_arr.ndim == 3 and result_arr.shape[0] == 1:
                        # Function returned [1, H, W], squeeze Z dimension
                        result_arr = result_arr[0]  # [1, H, W] -> [H, W]
                    # Now result_arr is [H, W], add channel dimension
                    if result_arr.ndim == 2:
                        result_arr = result_arr[np.newaxis, ...]  # [H, W] -> [1, H, W]
                elif is_3d_input:
                    # 3D case: functions return [D, H, W], add channel dimension
                    if result_arr.ndim == 3:
                        result_arr = result_arr[np.newaxis, ...]  # [D, H, W] -> [1, D, H, W]

                outputs.append(result_arr)

            if self.stack_outputs:
                # Concatenate outputs along channel dimension (axis=0 for [C, D, H, W] format)
                stacked = np.concatenate(outputs, axis=0)
                if self.retain_original:
                    original_key = self.output_key_format.format(key=key, task="original")
                    d[original_key] = label
                d[key] = self._to_tensor(stacked, add_batch_dim=False)
            else:
                if not self.retain_original:
                    d.pop(key, None)
                else:
                    d[key] = label
                for spec, result in zip(self.task_specs, outputs):
                    out_key = spec["output_key"] or self.output_key_format.format(
                        key=key, task=spec["name"]
                    )
                    d[out_key] = self._to_tensor(result, add_batch_dim=False)
        return d


__all__ = [
    "SegToBinaryMaskd",
    "SegToAffinityMapd",
    "SegToInstanceBoundaryMaskd",
    "SegToInstanceEDTd",
    "SegToSkeletonAwareEDTd",
    "SegToSemanticEDTd",
    "SegToFlowFieldd",
    "SegToSynapticPolarityd",
    "SegToSmallObjectd",
    "ComputeBinaryRatioWeightd",
    "ComputeUNet3DWeightd",
    "SegErosiond",
    "SegDilationd",
    "SegErosionInstanced",
    "EnergyQuantized",
    "DecodeQuantized",
    "SegSelectiond",
    "MultiTaskLabelTransformd",
]

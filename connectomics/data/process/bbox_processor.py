"""
Bounding box processor for efficient per-instance operations.

This module provides a unified framework for processing instance segmentation
with the bbox-first optimization pattern:
1. Compute all bounding boxes once
2. Process each instance within its local bbox
3. Aggregate results back to full volume

This pattern provides 5-10x speedup by avoiding full-volume operations.
"""

from __future__ import annotations
from typing import Callable, Optional, Tuple, Dict
from dataclasses import dataclass
import numpy as np
from skimage.measure import label as label_cc

from .bbox import compute_bbox_all, bbox_relax
from .misc import get_padsize, array_unpad


@dataclass
class BBoxProcessorConfig:
    """Configuration for bbox-based instance processing."""

    bg_value: float = -1.0
    relabel: bool = True
    padding: bool = False
    pad_size: int = 2
    bbox_relax: int = 1

    # Output configuration
    output_dtype: type = np.float32
    combine_mode: str = "max"  # "max", "sum", "replace"


class BBoxInstanceProcessor:
    """
    Efficient per-instance processor using bounding box optimization.

    This class encapsulates the bbox-first pattern:
    1. Preprocess: relabel, padding
    2. Compute bboxes once
    3. For each instance:
       - Extract bbox crop
       - Process instance in callback
       - Aggregate result to output
    4. Postprocess: background value, unpadding

    Example:
        >>> def compute_edt(label_crop, instance_id, bbox, **kwargs):
        ...     mask = (label_crop == instance_id)
        ...     edt = distance_transform_edt(mask, kwargs['resolution'])
        ...     return edt / edt.max()
        ...
        >>> processor = BBoxInstanceProcessor(config)
        >>> result = processor.process(label, compute_edt, resolution=(1, 1, 1))
    """

    def __init__(self, config: Optional[BBoxProcessorConfig] = None):
        """
        Args:
            config: Configuration for bbox processing. If None, uses defaults.
        """
        self.config = config or BBoxProcessorConfig()

    def process(
        self,
        label: np.ndarray,
        instance_fn: Callable[[np.ndarray, int, Tuple[slice, ...], Dict], Optional[np.ndarray]],
        **kwargs,
    ) -> np.ndarray:
        """
        Process all instances using bounding box optimization.

        Args:
            label: Instance segmentation (H, W) or (D, H, W)
            instance_fn: Callback function with signature:
                instance_fn(label_crop, instance_id, bbox, **kwargs) -> result_crop

                Where:
                - label_crop: Cropped label array for this instance's bbox
                - instance_id: Integer instance ID
                - bbox: Tuple of slices defining the bbox in full volume
                - **kwargs: Additional arguments passed through

                Returns:
                - result_crop: Same shape as label_crop, or None to skip

            **kwargs: Additional arguments passed to instance_fn

        Returns:
            Processed distance/energy map with same shape as label
        """
        # 1. Preprocessing
        label, label_shape, was_padded = self._preprocess(label)

        # 2. Initialize output
        distance = np.zeros(label_shape, dtype=self.config.output_dtype)

        # 3. Early exit if empty
        if label.max() == 0:
            distance = self._apply_bg_value(distance)
            return self._postprocess(distance, was_padded)

        # 4. Compute all bounding boxes at once (MAJOR SPEEDUP)
        bbox_array = compute_bbox_all(label, do_count=False)

        if bbox_array is None:
            distance = self._apply_bg_value(distance)
            return self._postprocess(distance, was_padded)

        # 5. Process each instance within its bounding box
        for i in range(bbox_array.shape[0]):
            instance_id = int(bbox_array[i, 0])
            bbox = self._extract_bbox(bbox_array[i], label_shape, label.ndim)

            # Extract instance crop
            label_crop = label[bbox]

            # Call user-provided instance processing function
            try:
                result_crop = instance_fn(label_crop, instance_id, bbox, kwargs)
            except Exception as e:
                # Skip instance on error
                print(f"Warning: Failed to process instance {instance_id}: {e}")
                continue

            # Skip if function returned None or empty result
            if result_crop is None or not np.any(result_crop):
                continue

            # Aggregate result back to full volume
            self._aggregate_result(distance, bbox, result_crop)

        # 6. Postprocessing
        distance = self._apply_bg_value(distance)
        return self._postprocess(distance, was_padded)

    def _preprocess(self, label: np.ndarray) -> Tuple[np.ndarray, Tuple[int, ...], bool]:
        """Relabel and pad the input label."""
        was_padded = False

        if self.config.relabel:
            label = label_cc(label)

        if self.config.padding:
            label = np.pad(label, self.config.pad_size, mode="constant", constant_values=0)
            was_padded = True

        return label, label.shape, was_padded

    def _extract_bbox(
        self, bbox_row: np.ndarray, label_shape: Tuple[int, ...], ndim: int
    ) -> Tuple[slice, ...]:
        """Extract bounding box as tuple of slices."""
        if ndim == 2:
            # 2D: [id, y_min, y_max, x_min, x_max]
            bbox_coords = [
                bbox_row[1],
                bbox_row[2] + 1,
                bbox_row[3],
                bbox_row[4] + 1,
            ]
            relaxed = bbox_relax(bbox_coords, label_shape, relax=self.config.bbox_relax)
            return (
                slice(relaxed[0], relaxed[1]),
                slice(relaxed[2], relaxed[3]),
            )
        else:  # 3D
            # 3D: [id, z_min, z_max, y_min, y_max, x_min, x_max]
            bbox_coords = [
                bbox_row[1],
                bbox_row[2] + 1,
                bbox_row[3],
                bbox_row[4] + 1,
                bbox_row[5],
                bbox_row[6] + 1,
            ]
            relaxed = bbox_relax(bbox_coords, label_shape, relax=self.config.bbox_relax)
            return (
                slice(relaxed[0], relaxed[1]),
                slice(relaxed[2], relaxed[3]),
                slice(relaxed[4], relaxed[5]),
            )

    def _aggregate_result(
        self, distance: np.ndarray, bbox: Tuple[slice, ...], result_crop: np.ndarray
    ):
        """Aggregate crop result back to full volume."""
        if self.config.combine_mode == "max":
            distance[bbox] = np.maximum(distance[bbox], result_crop)
        elif self.config.combine_mode == "sum":
            distance[bbox] += result_crop
        elif self.config.combine_mode == "replace":
            distance[bbox] = result_crop
        else:
            raise ValueError(f"Unknown combine_mode: {self.config.combine_mode}")

    def _apply_bg_value(self, distance: np.ndarray) -> np.ndarray:
        """Apply background value to zero regions."""
        if self.config.bg_value != 0:
            distance[distance == 0] = self.config.bg_value
        return distance

    def _postprocess(self, distance: np.ndarray, was_padded: bool) -> np.ndarray:
        """Unpad if needed."""
        if was_padded:
            pad_tuple = get_padsize(self.config.pad_size, ndim=distance.ndim)
            distance = array_unpad(distance, pad_tuple)
        return distance


# ============================================================================
# Convenience wrappers for common patterns
# ============================================================================


def process_instances_with_bbox(
    label: np.ndarray, instance_fn: Callable, config: Optional[BBoxProcessorConfig] = None, **kwargs
) -> np.ndarray:
    """
    Functional wrapper for bbox-based instance processing.

    Args:
        label: Instance segmentation
        instance_fn: Per-instance processing function
        config: Optional configuration
        **kwargs: Additional arguments for instance_fn

    Returns:
        Processed result array

    Example:
        >>> def my_edt(label_crop, instance_id, bbox, context):
        ...     mask = (label_crop == instance_id)
        ...     return distance_transform_edt(mask, context['resolution'])
        >>>
        >>> result = process_instances_with_bbox(
        ...     label,
        ...     my_edt,
        ...     resolution=(40, 16, 16)
        ... )
    """
    processor = BBoxInstanceProcessor(config)
    return processor.process(label, instance_fn, **kwargs)


def make_instance_processor(operation: str, **default_kwargs) -> Callable[[np.ndarray], np.ndarray]:
    """
    Factory function to create reusable instance processors.

    Args:
        operation: Name of operation ("edt", "skeleton_edt", etc.)
        **default_kwargs: Default parameters for the operation

    Returns:
        Function that processes labels with the specified operation

    Example:
        >>> # Create a reusable EDT processor
        >>> edt_processor = make_instance_processor(
        ...     "edt",
        ...     resolution=(40, 16, 16),
        ...     bg_value=-1.0
        ... )
        >>>
        >>> # Use it multiple times
        >>> result1 = edt_processor(label1)
        >>> result2 = edt_processor(label2)
    """
    from . import distance  # Avoid circular import

    operation_map = {
        "edt": distance.distance_transform,
        "skeleton_edt": distance.skeleton_aware_distance_transform,
    }

    if operation not in operation_map:
        raise ValueError(f"Unknown operation: {operation}")

    fn = operation_map[operation]

    def processor(label: np.ndarray, **override_kwargs):
        """Process label with configured operation."""
        merged_kwargs = {**default_kwargs, **override_kwargs}
        return fn(label, **merged_kwargs)

    return processor


__all__ = [
    "BBoxProcessorConfig",
    "BBoxInstanceProcessor",
    "process_instances_with_bbox",
    "make_instance_processor",
]

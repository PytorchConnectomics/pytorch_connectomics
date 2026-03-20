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

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from skimage.measure import label as label_cc

from .bbox import bbox_relax, compute_bbox_all
from .misc import array_unpad, get_padsize


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
        num_workers: int = 0,
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

            num_workers: Number of threads for parallel instance processing.
                0 = sequential (default). Scipy EDT releases the GIL, so
                threads give real parallelism for the numeric heavy lifting.

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

        # 5. Prepare per-instance work items
        n = bbox_array.shape[0]
        work_items = []
        for i in range(n):
            instance_id = int(bbox_array[i, 0])
            bbox = self._extract_bbox(bbox_array[i], label_shape, label.ndim)
            label_crop = label[bbox]
            work_items.append((label_crop, instance_id, bbox))

        # 6. Process instances (parallel or sequential)
        if num_workers > 0:
            from concurrent.futures import ThreadPoolExecutor

            def _run(item):
                label_crop, instance_id, bbox = item
                try:
                    return bbox, instance_fn(label_crop, instance_id, bbox, kwargs)
                except Exception:
                    return bbox, None

            with ThreadPoolExecutor(max_workers=num_workers) as pool:
                for bbox, result_crop in pool.map(_run, work_items):
                    if result_crop is not None and np.any(result_crop):
                        self._aggregate_result(distance, bbox, result_crop)
        else:
            for label_crop, instance_id, bbox in work_items:
                try:
                    result_crop = instance_fn(label_crop, instance_id, bbox, kwargs)
                except Exception:
                    continue
                if result_crop is not None and np.any(result_crop):
                    self._aggregate_result(distance, bbox, result_crop)

        # 7. Postprocessing
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


__all__ = [
    "BBoxProcessorConfig",
    "BBoxInstanceProcessor",
]

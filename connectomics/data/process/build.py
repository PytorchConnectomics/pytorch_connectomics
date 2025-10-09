"""
Modern MONAI Compose factory functions for PyTorch Connectomics.

This module provides clean, modern factory functions to create MONAI Compose pipelines
for connectomics workflows using configuration objects.
"""

from __future__ import annotations
from typing import List, Dict, Any
from monai.transforms import Compose

from .monai_transforms import (
    SegToBinaryMaskd,
    SegToAffinityMapd,
    SegToInstanceBoundaryMaskd,
    SegToInstanceEDTd,
    SegToSemanticEDTd,
    SegToFlowFieldd,
    SegToSynapticPolarityd,
    SegToSmallObjectd,
    ComputeBinaryRatioWeightd,
    ComputeUNet3DWeightd,
    SegErosiond,
    SegDilationd,
    SegSelectiond,
)


def create_binary_segmentation_pipeline(cfg: Any) -> Compose:
    """Create a pipeline for binary segmentation tasks.

    Args:
        cfg: Configuration object with fields:
            - input_key: Key for input segmentation data (default: "label")
            - output_key: Key for output binary mask (default: "target")
            - foreground_classes: List of class IDs to treat as foreground (None = all non-zero)

    Returns:
        MONAI Compose pipeline for binary segmentation
    """
    input_key = getattr(cfg, 'input_key', 'label')
    output_key = getattr(cfg, 'output_key', 'target')
    foreground_classes = getattr(cfg, 'foreground_classes', None)

    transforms = [
        SegToBinaryMaskd(keys=[output_key], foreground_classes=foreground_classes)
    ]
    return Compose(transforms)


def create_affinity_segmentation_pipeline(cfg: Any) -> Compose:
    """Create a pipeline for affinity-based segmentation tasks.

    Args:
        cfg: Configuration object with fields:
            - input_key: Key for input segmentation data (default: "label")
            - output_key: Key for output affinity maps (default: "target")
            - offsets: List of affinity offsets in "z-y-x" format (e.g., ["1-0-0", "0-1-0", "0-0-1"])
            - include_long_range: Whether to include long-range affinities (default: True)
            - long_range_distance: Distance for long-range affinities (default: 10)

    Returns:
        MONAI Compose pipeline for affinity computation
    """
    input_key = getattr(cfg, 'input_key', 'label')
    output_key = getattr(cfg, 'output_key', 'target')
    offsets = getattr(cfg, 'offsets', None)
    include_long_range = getattr(cfg, 'include_long_range', True)
    long_range_distance = getattr(cfg, 'long_range_distance', 10)

    if offsets is None:
        # Standard short-range 3D affinities
        offsets = ['1-0-0', '0-1-0', '0-0-1']

        if include_long_range:
            # Add long-range affinities
            offsets.extend([
                f'{long_range_distance}-0-0',
                f'0-{long_range_distance}-0',
                f'0-0-{long_range_distance}'
            ])

    transforms = [
        SegToAffinityMapd(keys=[output_key], offsets=offsets)
    ]
    return Compose(transforms)


def create_instance_segmentation_pipeline(cfg: Any) -> Compose:
    """Create a pipeline for instance segmentation tasks.

    Args:
        cfg: Configuration object with fields:
            - input_key: Key for input segmentation data (default: "label")
            - boundary_output_key: Key for boundary mask output (default: "boundary")
            - edt_output_key: Key for EDT output (default: "edt")
            - use_boundary: Whether to compute instance boundaries (default: True)
            - use_edt: Whether to compute instance EDT (default: True)
            - boundary_thickness: Thickness of boundary masks (default: 1)
            - edt_distance_param: Distance parameter for EDT normalization (default: 200.0)

    Returns:
        MONAI Compose pipeline for instance segmentation
    """
    input_key = getattr(cfg, 'input_key', 'label')
    boundary_output_key = getattr(cfg, 'boundary_output_key', 'boundary')
    edt_output_key = getattr(cfg, 'edt_output_key', 'edt')
    use_binary = getattr(cfg, 'use_binary', True)
    binary_output_key = getattr(cfg, 'binary_output_key', 'binary')
    use_boundary = getattr(cfg, 'use_boundary', True)
    use_edt = getattr(cfg, 'use_edt', True)
    segment_id = getattr(cfg, 'segment_id', [])
    boundary_thickness = getattr(cfg, 'boundary_thickness', 1)
    edt_distance_param = getattr(cfg, 'edt_distance_param', 200.0)

    transforms = []

    if use_binary:
        transforms.append(
            SegToBinaryMaskd(
                keys=[binary_output_key],
                segment_id=segment_id
            )
        )
    if use_boundary:
        transforms.append(
            SegToInstanceBoundaryMaskd(
                keys=[boundary_output_key],
                thickness=boundary_thickness
            )
        )

    if use_edt:
        transforms.append(
            SegToInstanceEDTd(
                keys=[edt_output_key],
                distance_param=edt_distance_param
            )
        )

    return Compose(transforms)


def create_semantic_edt_pipeline(cfg: Any) -> Compose:
    """Create a pipeline for semantic EDT computation.

    Args:
        cfg: Configuration object with fields:
            - input_key: Key for input segmentation data (default: "label")
            - output_key: Key for semantic EDT output (default: "semantic_edt")
            - alpha_foreground: Alpha parameter for foreground EDT (default: 8.0)
            - alpha_background: Alpha parameter for background EDT (default: 50.0)

    Returns:
        MONAI Compose pipeline for semantic EDT
    """
    input_key = getattr(cfg, 'input_key', 'label')
    output_key = getattr(cfg, 'output_key', 'semantic_edt')
    alpha_foreground = getattr(cfg, 'alpha_foreground', 8.0)
    alpha_background = getattr(cfg, 'alpha_background', 50.0)

    transforms = [
        SegToSemanticEDTd(
            keys=[output_key],
            alpha_foreground=alpha_foreground,
            alpha_background=alpha_background
        )
    ]
    return Compose(transforms)


def create_synaptic_polarity_pipeline(cfg: Any) -> Compose:
    """Create a pipeline for synaptic polarity tasks.

    Args:
        cfg: Configuration object with fields:
            - input_key: Key for input segmentation data (default: "label")
            - output_key: Key for polarity output (default: "polarity")
            - use_exclusive_classes: Whether to use exclusive semantic classes (default: False)

    Returns:
        MONAI Compose pipeline for synaptic polarity
    """
    input_key = getattr(cfg, 'input_key', 'label')
    output_key = getattr(cfg, 'output_key', 'polarity')
    use_exclusive_classes = getattr(cfg, 'use_exclusive_classes', False)

    transforms = [
        SegToSynapticPolarityd(
            keys=[output_key],
            exclusive=use_exclusive_classes
        )
    ]
    return Compose(transforms)


def create_small_object_detection_pipeline(cfg: Any) -> Compose:
    """Create a pipeline for small object detection.

    Args:
        cfg: Configuration object with fields:
            - input_key: Key for input segmentation data (default: "label")
            - output_key: Key for small object mask output (default: "small_objects")
            - size_threshold: Maximum size for objects to be considered small (default: 100)

    Returns:
        MONAI Compose pipeline for small object detection
    """
    input_key = getattr(cfg, 'input_key', 'label')
    output_key = getattr(cfg, 'output_key', 'small_objects')
    size_threshold = getattr(cfg, 'size_threshold', 100)

    transforms = [
        SegToSmallObjectd(
            keys=[output_key],
            threshold=size_threshold
        )
    ]
    return Compose(transforms)


def create_erosion_pipeline(cfg: Any) -> Compose:
    """Create a pipeline for morphological erosion.

    Args:
        cfg: Configuration object with fields:
            - input_key: Key for input segmentation data (default: "label")
            - output_key: Key for eroded output (default: "eroded")
            - kernel_size: Size of erosion kernel (default: 1)

    Returns:
        MONAI Compose pipeline for erosion
    """
    input_key = getattr(cfg, 'input_key', 'label')
    output_key = getattr(cfg, 'output_key', 'eroded')
    kernel_size = getattr(cfg, 'kernel_size', 1)

    transforms = [
        SegErosiond(
            keys=[output_key],
            kernel_size=kernel_size
        )
    ]
    return Compose(transforms)


def create_dilation_pipeline(cfg: Any) -> Compose:
    """Create a pipeline for morphological dilation.

    Args:
        cfg: Configuration object with fields:
            - input_key: Key for input segmentation data (default: "label")
            - output_key: Key for dilated output (default: "dilated")
            - kernel_size: Size of dilation kernel (default: 1)

    Returns:
        MONAI Compose pipeline for dilation
    """
    input_key = getattr(cfg, 'input_key', 'label')
    output_key = getattr(cfg, 'output_key', 'dilated')
    kernel_size = getattr(cfg, 'kernel_size', 1)

    transforms = [
        SegDilationd(
            keys=[output_key],
            kernel_size=kernel_size
        )
    ]
    return Compose(transforms)


def create_segmentation_selection_pipeline(cfg: Any) -> Compose:
    """Create a pipeline for selecting specific segmentation indices.

    Args:
        cfg: Configuration object with fields:
            - input_key: Key for input segmentation data (default: "label")
            - output_key: Key for selected segmentation output (default: "selected")
            - selected_indices: List of indices to select and renumber (required)

    Returns:
        MONAI Compose pipeline for segmentation selection
    """
    input_key = getattr(cfg, 'input_key', 'label')
    output_key = getattr(cfg, 'output_key', 'selected')
    selected_indices = cfg.selected_indices  # Required parameter

    transforms = [
        SegSelectiond(keys=[output_key], indices=selected_indices)
    ]
    return Compose(transforms)


def create_weight_computation_pipeline(cfg: Any) -> Compose:
    """Create a pipeline for computing loss weights.

    Args:
        cfg: Configuration object with fields:
            - input_key: Key for input segmentation data (default: "label")
            - output_key: Key for weight output (default: "weight")
            - method: Weight computation method (default: "binary_ratio")
            - sigma: Sigma parameter for unet3d method (default: 5.0)
            - w0: w0 parameter for unet3d method (default: 0.3)

    Returns:
        MONAI Compose pipeline for weight computation
    """
    input_key = getattr(cfg, 'input_key', 'label')
    output_key = getattr(cfg, 'output_key', 'weight')
    method = getattr(cfg, 'method', 'binary_ratio')

    transforms = []

    if method == "binary_ratio":
        transforms.append(
            ComputeBinaryRatioWeightd(keys=[output_key])
        )
    elif method == "unet3d":
        sigma = getattr(cfg, 'sigma', 5.0)
        w0 = getattr(cfg, 'w0', 0.3)
        transforms.append(
            ComputeUNet3DWeightd(
                keys=[output_key],
                sigma=sigma,
                w0=w0
            )
        )

    return Compose(transforms)


def create_multi_task_pipeline(cfg: Any) -> Compose:
    """Create a multi-task processing pipeline from label_transform config.

    Args:
        cfg: Configuration object (label_transform section) with optional fields:
            - input_key: Key for input segmentation data (default: "label")
            - output_key: Key for output (default: "label")
            - affinity: Affinity config with 'offsets' list
            - skeleton_distance: EDT config with 'enabled', 'alpha', 'bg_value'
            - binary: Binary segmentation config
            - instance: Instance segmentation config
            - And other task-specific configs

    Returns:
        MONAI Compose pipeline for multi-task processing
    """
    input_key = getattr(cfg, 'input_key', 'label')
    output_key = getattr(cfg, 'output_key', 'label')

    transforms = []

    # Check for affinity transformation
    if hasattr(cfg, 'affinity') and hasattr(cfg.affinity, 'offsets') and cfg.affinity.offsets:
        # Create a config object for affinity pipeline
        class AffinityConfig:
            def __init__(self, parent_cfg):
                self.input_key = getattr(parent_cfg, 'input_key', 'label')
                self.output_key = getattr(parent_cfg, 'output_key', 'label')
                self.offsets = parent_cfg.affinity.offsets

        affinity_cfg = AffinityConfig(cfg)
        pipeline = create_affinity_segmentation_pipeline(affinity_cfg)
        transforms.extend(pipeline.transforms)

    # Check for skeleton distance (semantic EDT) transformation
    if hasattr(cfg, 'skeleton_distance') and hasattr(cfg.skeleton_distance, 'enabled') and cfg.skeleton_distance.enabled:
        # Create a config object for semantic EDT pipeline
        class SemanticEDTConfig:
            def __init__(self, parent_cfg):
                self.input_key = getattr(parent_cfg, 'input_key', 'label')
                self.output_key = getattr(parent_cfg, 'output_key', 'label')
                self.alpha_foreground = parent_cfg.skeleton_distance.alpha
                self.alpha_background = parent_cfg.skeleton_distance.bg_value

        edt_cfg = SemanticEDTConfig(cfg)
        pipeline = create_semantic_edt_pipeline(edt_cfg)
        transforms.extend(pipeline.transforms)

    # Check for binary transformation
    if hasattr(cfg, 'binary') and hasattr(cfg.binary, 'foreground_classes'):
        class BinaryConfig:
            def __init__(self, parent_cfg):
                self.input_key = getattr(parent_cfg, 'input_key', 'label')
                self.output_key = getattr(parent_cfg, 'output_key', 'label')
                self.foreground_classes = parent_cfg.binary.foreground_classes

        binary_cfg = BinaryConfig(cfg)
        pipeline = create_binary_segmentation_pipeline(binary_cfg)
        transforms.extend(pipeline.transforms)

    # Check for instance transformation
    if hasattr(cfg, 'instance'):
        class InstanceConfig:
            def __init__(self, parent_cfg):
                self.input_key = getattr(parent_cfg, 'input_key', 'label')
                self.boundary_output_key = getattr(parent_cfg.instance, 'boundary_output_key', 'boundary')
                self.edt_output_key = getattr(parent_cfg.instance, 'edt_output_key', 'edt')
                self.use_boundary = getattr(parent_cfg.instance, 'use_boundary', True)
                self.use_edt = getattr(parent_cfg.instance, 'use_edt', True)
                self.boundary_thickness = getattr(parent_cfg.instance, 'boundary_thickness', 1)
                self.edt_distance_param = getattr(parent_cfg.instance, 'edt_distance_param', 200.0)

        instance_cfg = InstanceConfig(cfg)
        pipeline = create_instance_segmentation_pipeline(instance_cfg)
        transforms.extend(pipeline.transforms)

    return Compose(transforms)


__all__ = [
    'create_binary_segmentation_pipeline',
    'create_affinity_segmentation_pipeline',
    'create_instance_segmentation_pipeline',
    'create_semantic_edt_pipeline',
    'create_synaptic_polarity_pipeline',
    'create_small_object_detection_pipeline',
    'create_erosion_pipeline',
    'create_dilation_pipeline',
    'create_segmentation_selection_pipeline',
    'create_weight_computation_pipeline',
    'create_multi_task_pipeline',
]

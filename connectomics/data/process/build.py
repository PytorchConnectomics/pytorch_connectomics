"""
Modern MONAI Compose factory functions for PyTorch Connectomics.

This module provides clean, modern factory functions to create MONAI Compose pipelines
for connectomics workflows with descriptive parameters instead of cryptic options.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Union, Literal
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
    SegErosionDilationd,
    SegSelectiond,
)

# Type aliases for better code clarity
AffinityOffset = Union[str, List[int]]
WeightMethod = Literal["binary_ratio", "unet3d"]
MorphologyOperation = Literal["erosion", "dilation", "both"]


def create_binary_segmentation_pipeline(
    input_key: str = "label",
    output_key: str = "target",
    foreground_classes: Optional[List[int]] = None,
) -> Compose:
    """Create a pipeline for binary segmentation tasks.

    Args:
        input_key: Key for input segmentation data
        output_key: Key for output binary mask
        foreground_classes: List of class IDs to treat as foreground (None = all non-zero)

    Returns:
        MONAI Compose pipeline for binary segmentation
    """
    if foreground_classes is None:
        target_opt = ['0']
    else:
        target_opt = ['0'] + [str(cls) for cls in foreground_classes]

    transforms = [
        SegToBinaryMaskd(keys=[output_key], target_opt=target_opt)
    ]
    return Compose(transforms)


def create_affinity_segmentation_pipeline(
    input_key: str = "label",
    output_key: str = "target",
    offsets: Optional[List[AffinityOffset]] = None,
    include_long_range: bool = True,
    long_range_distance: int = 10,
) -> Compose:
    """Create a pipeline for affinity-based segmentation tasks.

    Args:
        input_key: Key for input segmentation data
        output_key: Key for output affinity maps
        offsets: Custom affinity offsets (defaults to standard 3D short-range)
        include_long_range: Whether to include long-range affinities
        long_range_distance: Distance for long-range affinities

    Returns:
        MONAI Compose pipeline for affinity computation
    """
    if offsets is None:
        # Standard short-range 3D affinities
        offset_strings = ['1-0-0', '0-1-0', '0-0-1']

        if include_long_range:
            # Add long-range affinities
            offset_strings.extend([
                f'{long_range_distance}-0-0',
                f'0-{long_range_distance}-0',
                f'0-0-{long_range_distance}'
            ])
    else:
        offset_strings = []
        for offset in offsets:
            if isinstance(offset, str):
                offset_strings.append(offset)
            else:
                offset_strings.append('-'.join(map(str, offset)))

    target_opt = ['1'] + offset_strings
    transforms = [
        SegToAffinityMapd(keys=[output_key], target_opt=target_opt)
    ]
    return Compose(transforms)


def create_instance_segmentation_pipeline(
    input_key: str = "label",
    boundary_output_key: str = "boundary",
    edt_output_key: str = "edt",
    use_boundary: bool = True,
    use_edt: bool = True,
    boundary_thickness: int = 1,
    edt_distance_param: float = 200.0,
) -> Compose:
    """Create a pipeline for instance segmentation tasks.

    Args:
        input_key: Key for input segmentation data
        boundary_output_key: Key for boundary mask output
        edt_output_key: Key for EDT output
        use_boundary: Whether to compute instance boundaries
        use_edt: Whether to compute instance EDT
        boundary_thickness: Thickness of boundary masks
        edt_distance_param: Distance parameter for EDT normalization

    Returns:
        MONAI Compose pipeline for instance segmentation
    """
    transforms = []

    if use_boundary:
        transforms.append(
            SegToInstanceBoundaryMaskd(
                keys=[boundary_output_key],
                target_opt=['2', str(boundary_thickness)]
            )
        )

    if use_edt:
        transforms.append(
            SegToInstanceEDTd(
                keys=[edt_output_key],
                target_opt=['3', str(edt_distance_param)]
            )
        )

    return Compose(transforms)


def create_semantic_edt_pipeline(
    input_key: str = "label",
    output_key: str = "semantic_edt",
    alpha_foreground: float = 8.0,
    alpha_background: float = 50.0,
) -> Compose:
    """Create a pipeline for semantic EDT computation.

    Args:
        input_key: Key for input segmentation data
        output_key: Key for semantic EDT output
        alpha_foreground: Alpha parameter for foreground EDT
        alpha_background: Alpha parameter for background EDT

    Returns:
        MONAI Compose pipeline for semantic EDT
    """
    transforms = [
        SegToSemanticEDTd(
            keys=[output_key],
            target_opt=['4', str(alpha_foreground), str(alpha_background)]
        )
    ]
    return Compose(transforms)


def create_synaptic_polarity_pipeline(
    input_key: str = "label",
    output_key: str = "polarity",
    use_exclusive_classes: bool = False,
) -> Compose:
    """Create a pipeline for synaptic polarity tasks.

    Args:
        input_key: Key for input segmentation data
        output_key: Key for polarity output
        use_exclusive_classes: Whether to use exclusive semantic classes

    Returns:
        MONAI Compose pipeline for synaptic polarity
    """
    target_opt = ['6']
    if use_exclusive_classes:
        target_opt.append('1')

    transforms = [
        SegToSynapticPolarityd(keys=[output_key], target_opt=target_opt)
    ]
    return Compose(transforms)


def create_small_object_detection_pipeline(
    input_key: str = "label",
    output_key: str = "small_objects",
    size_threshold: int = 100,
) -> Compose:
    """Create a pipeline for small object detection.

    Args:
        input_key: Key for input segmentation data
        output_key: Key for small object mask output
        size_threshold: Maximum size for objects to be considered small

    Returns:
        MONAI Compose pipeline for small object detection
    """
    transforms = [
        SegToSmallObjectd(
            keys=[output_key],
            target_opt=['7', str(size_threshold)]
        )
    ]
    return Compose(transforms)


def create_morphology_pipeline(
    input_key: str = "label",
    output_key: str = "morphology",
    operation: MorphologyOperation = "erosion",
    kernel_size: int = 1,
) -> Compose:
    """Create a pipeline for morphological operations.

    Args:
        input_key: Key for input segmentation data
        output_key: Key for morphology output
        operation: Type of morphological operation
        kernel_size: Size of morphological kernel

    Returns:
        MONAI Compose pipeline for morphological operations
    """
    operation_map = {
        "erosion": "1",
        "dilation": "2",
        "both": "3"
    }

    transforms = [
        SegErosionDilationd(
            keys=[output_key],
            target_opt=[operation_map[operation], str(kernel_size)]
        )
    ]
    return Compose(transforms)


def create_segmentation_selection_pipeline(
    selected_indices: List[int],
    input_key: str = "label",
    output_key: str = "selected",
) -> Compose:
    """Create a pipeline for selecting specific segmentation indices.

    Args:
        input_key: Key for input segmentation data
        output_key: Key for selected segmentation output
        selected_indices: List of indices to select and renumber

    Returns:
        MONAI Compose pipeline for segmentation selection
    """
    transforms = [
        SegSelectiond(keys=[output_key], indices=selected_indices)
    ]
    return Compose(transforms)


def create_weight_computation_pipeline(
    input_key: str = "label",
    output_key: str = "weight",
    method: WeightMethod = "binary_ratio",
    **kwargs,
) -> Compose:
    """Create a pipeline for computing loss weights.

    Args:
        input_key: Key for input segmentation data
        output_key: Key for weight output
        method: Weight computation method
        **kwargs: Additional parameters for weight computation

    Returns:
        MONAI Compose pipeline for weight computation
    """
    transforms = []

    if method == "binary_ratio":
        transforms.append(
            ComputeBinaryRatioWeightd(keys=[output_key], target_opt=['1'])
        )
    elif method == "unet3d":
        sigma = kwargs.get('sigma', 5.0)
        w0 = kwargs.get('w0', 0.3)
        transforms.append(
            ComputeUNet3DWeightd(
                keys=[output_key],
                target_opt=['2', '1', str(sigma), str(w0)]
            )
        )

    return Compose(transforms)


def create_multi_task_pipeline(
    tasks: List[Dict[str, Any]],
    input_key: str = "label",
) -> Compose:
    """Create a multi-task processing pipeline.

    Args:
        input_key: Key for input segmentation data
        tasks: List of task configurations, each containing:
            - 'type': Task type ('binary', 'affinity', 'boundary', 'edt', etc.)
            - 'output_key': Output key for the task
            - Additional task-specific parameters

    Returns:
        MONAI Compose pipeline for multi-task processing
    """
    transforms = []

    for task in tasks:
        task_type = task['type']
        output_key = task['output_key']

        if task_type == 'binary':
            pipeline = create_binary_segmentation_pipeline(
                input_key=input_key,
                output_key=output_key,
                **{k: v for k, v in task.items() if k not in ['type', 'output_key']}
            )
        elif task_type == 'affinity':
            pipeline = create_affinity_segmentation_pipeline(
                input_key=input_key,
                output_key=output_key,
                **{k: v for k, v in task.items() if k not in ['type', 'output_key']}
            )
        elif task_type == 'instance':
            pipeline = create_instance_segmentation_pipeline(
                input_key=input_key,
                boundary_output_key=f"{output_key}_boundary",
                edt_output_key=f"{output_key}_edt",
                **{k: v for k, v in task.items() if k not in ['type', 'output_key']}
            )
        elif task_type == 'semantic_edt':
            pipeline = create_semantic_edt_pipeline(
                input_key=input_key,
                output_key=output_key,
                **{k: v for k, v in task.items() if k not in ['type', 'output_key']}
            )
        elif task_type == 'polarity':
            pipeline = create_synaptic_polarity_pipeline(
                input_key=input_key,
                output_key=output_key,
                **{k: v for k, v in task.items() if k not in ['type', 'output_key']}
            )
        elif task_type == 'small_objects':
            pipeline = create_small_object_detection_pipeline(
                input_key=input_key,
                output_key=output_key,
                **{k: v for k, v in task.items() if k not in ['type', 'output_key']}
            )
        elif task_type == 'morphology':
            pipeline = create_morphology_pipeline(
                input_key=input_key,
                output_key=output_key,
                **{k: v for k, v in task.items() if k not in ['type', 'output_key']}
            )
        elif task_type == 'selection':
            pipeline = create_segmentation_selection_pipeline(
                input_key=input_key,
                output_key=output_key,
                **{k: v for k, v in task.items() if k not in ['type', 'output_key']}
            )
        elif task_type == 'weight':
            pipeline = create_weight_computation_pipeline(
                input_key=input_key,
                output_key=output_key,
                **{k: v for k, v in task.items() if k not in ['type', 'output_key']}
            )
        else:
            continue

        transforms.extend(pipeline.transforms)

    return Compose(transforms)


__all__ = [
    'create_binary_segmentation_pipeline',
    'create_affinity_segmentation_pipeline',
    'create_instance_segmentation_pipeline',
    'create_semantic_edt_pipeline',
    'create_synaptic_polarity_pipeline',
    'create_small_object_detection_pipeline',
    'create_morphology_pipeline',
    'create_segmentation_selection_pipeline',
    'create_weight_computation_pipeline',
    'create_multi_task_pipeline',
]
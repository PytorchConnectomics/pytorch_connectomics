"""
MONAI Compose factory functions for PyTorch Connectomics.

This module provides factory functions to create MONAI Compose pipelines
for common connectomics workflows using the MONAI-native transforms.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Union
from monai.transforms import Compose
from yacs.config import CfgNode

from .monai_transforms import (
    SegToBinaryMaskd,
    SegToAffinityMapd,
    SegToInstanceBoundaryMaskd,
    SegToInstanceEDTd,
    SegToSemanticEDTd,
    SegToFlowFieldd,
    SegToSynapticPolarityd,
    SegToSmallObjectd,
    SegToGenericSemanticed,
    ComputeBinaryRatioWeightd,
    ComputeUNet3DWeightd,
    SegErosionDilationd,
)


def create_target_transforms(target_opts: List[List[str]], keys: List[str] = ['label']) -> List[Any]:
    """Create target processing transforms based on configuration."""
    transforms = []

    for i, opt in enumerate(target_opts):
        if len(opt) == 0:
            continue

        task_type = opt[0]
        target_key = f'target_{i}' if i > 0 else 'target'

        if task_type == '0':  # Binary segmentation
            transforms.append(SegToBinaryMaskd(keys=[target_key], target_opt=opt))
        elif task_type == '1':  # Affinity map
            transforms.append(SegToAffinityMapd(keys=[target_key], target_opt=opt))
        elif task_type == '2':  # Instance boundary
            transforms.append(SegToInstanceBoundaryMaskd(keys=[target_key], target_opt=opt))
        elif task_type == '3':  # Instance EDT
            transforms.append(SegToInstanceEDTd(keys=[target_key], target_opt=opt))
        elif task_type == '4':  # Semantic EDT
            transforms.append(SegToSemanticEDTd(keys=[target_key], target_opt=opt))
        elif task_type == '5':  # Flow field
            transforms.append(SegToFlowFieldd(keys=[target_key], target_opt=opt))
        elif task_type == '6':  # Synaptic polarity
            transforms.append(SegToSynapticPolarityd(keys=[target_key], target_opt=opt))
        elif task_type == '7':  # Small object
            transforms.append(SegToSmallObjectd(keys=[target_key], target_opt=opt))
        elif task_type == '9':  # Generic semantic
            transforms.append(SegToGenericSemanticed(keys=[target_key], target_opt=opt))

    return transforms


def create_weight_transforms(weight_opts: List[List[str]], keys: List[str] = ['weight']) -> List[Any]:
    """Create weight computation transforms based on configuration."""
    transforms = []

    for i, opt in enumerate(weight_opts):
        if len(opt) == 0:
            continue

        weight_type = opt[0]
        weight_key = f'weight_{i}' if i > 0 else 'weight'

        if weight_type == '1':  # Binary ratio weight
            transforms.append(ComputeBinaryRatioWeightd(keys=[weight_key], target_opt=opt))
        elif weight_type == '2':  # UNet3D weight
            transforms.append(ComputeUNet3DWeightd(keys=[weight_key], target_opt=opt))

    return transforms


def create_full_processing_pipeline(
    target_opts: List[List[str]],
    weight_opts: Optional[List[List[str]]] = None,
    erosion_opts: Optional[List[str]] = None,
    keys: List[str] = ['label']
) -> Compose:
    """Create a complete processing pipeline with targets and weights."""
    transforms = []

    # Add erosion/dilation if specified
    if erosion_opts:
        transforms.append(SegErosionDilationd(keys=keys, target_opt=erosion_opts))

    # Add target transforms
    transforms.extend(create_target_transforms(target_opts, keys))

    # Add weight transforms
    if weight_opts:
        transforms.extend(create_weight_transforms(weight_opts))

    return Compose(transforms)


def create_processor_from_config(cfg: CfgNode) -> Compose:
    """Create MONAI processor pipeline from configuration."""
    target_opts = cfg.MODEL.TARGET_OPT if hasattr(cfg.MODEL, 'TARGET_OPT') else []
    weight_opts = cfg.MODEL.WEIGHT_OPT if hasattr(cfg.MODEL, 'WEIGHT_OPT') else None

    return create_full_processing_pipeline(target_opts, weight_opts)


def create_binary_segmentation_pipeline(keys: List[str] = ['label']) -> Compose:
    """Create a pipeline for binary segmentation tasks."""
    transforms = [
        SegToBinaryMaskd(keys=['target'], target_opt=['0'])
    ]
    return Compose(transforms)


def create_affinity_segmentation_pipeline(
    offsets: Optional[List[str]] = None,
    keys: List[str] = ['label']
) -> Compose:
    """Create a pipeline for affinity-based segmentation tasks."""
    if offsets is None:
        offsets = ['1-1-0', '1-0-0', '0-1-0', '0-0-1']

    transforms = [
        SegToAffinityMapd(keys=['target'], target_opt=['1'] + offsets)
    ]
    return Compose(transforms)


def create_instance_segmentation_pipeline(
    use_boundary: bool = True,
    use_edt: bool = True,
    keys: List[str] = ['label']
) -> Compose:
    """Create a pipeline for instance segmentation tasks."""
    transforms = []

    if use_boundary:
        transforms.append(SegToInstanceBoundaryMaskd(keys=['target_0'], target_opt=['2']))

    if use_edt:
        transforms.append(SegToInstanceEDTd(keys=['target_1'], target_opt=['3', '200']))

    return Compose(transforms)


def create_multi_task_pipeline(
    task_configs: List[Dict[str, Any]],
    keys: List[str] = ['label']
) -> Compose:
    """Create a pipeline for multi-task learning."""
    transforms = []

    for i, task_config in enumerate(task_configs):
        task_type = task_config['type']
        task_opts = task_config.get('options', [])
        target_key = f'target_{i}'

        if task_type == 'binary':
            transforms.append(SegToBinaryMaskd(keys=[target_key], target_opt=['0'] + task_opts))
        elif task_type == 'affinity':
            transforms.append(SegToAffinityMapd(keys=[target_key], target_opt=['1'] + task_opts))
        elif task_type == 'boundary':
            transforms.append(SegToInstanceBoundaryMaskd(keys=[target_key], target_opt=['2'] + task_opts))
        elif task_type == 'edt':
            transforms.append(SegToInstanceEDTd(keys=[target_key], target_opt=['3'] + task_opts))

    return Compose(transforms)


__all__ = [
    'create_target_transforms',
    'create_weight_transforms',
    'create_full_processing_pipeline',
    'create_processor_from_config',
    'create_binary_segmentation_pipeline',
    'create_affinity_segmentation_pipeline',
    'create_instance_segmentation_pipeline',
    'create_multi_task_pipeline',
]
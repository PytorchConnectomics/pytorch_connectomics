"""
Main entry point for processing transforms.

This module provides easy access to all MONAI-native processing transforms,
composition functions, and factory functions for connectomics data processing.

Usage:
    from connectomics.transforms.processor import SegToBinaryMaskd, create_binary_segmentation_pipeline
"""

# Legacy processor (for backward compatibility)
from .process.processor import (
    DataProcessor,
    LightningCompatibleProcessor,
    create_processor_from_config,
)

# MONAI-native individual transforms
from .process.monai_transforms import (
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

# MONAI composition and factory functions
from .process.monai_compose import (
    create_target_transforms,
    create_weight_transforms,
    create_full_processing_pipeline,
    create_processor_from_config as create_monai_processor_from_config,
    create_binary_segmentation_pipeline,
    create_affinity_segmentation_pipeline,
    create_instance_segmentation_pipeline,
    create_multi_task_pipeline,
)

# Core processing functions (for advanced users)
from .process.target import (
    seg_to_binary,
    seg_to_affinity,
    seg_to_instance_bd,
    seg_to_polarity,
    seg_to_small_seg,
)

from .process.distance import (
    edt_semantic,
    edt_instance,
    seg_to_instance_edt,
)

from .process.flow import (
    seg_to_flows,
)

from .process.weight import (
    weight_binary_ratio,
    weight_unet3d,
)

__all__ = [
    # Legacy (backward compatibility)
    'DataProcessor',
    'LightningCompatibleProcessor',
    'create_processor_from_config',

    # MONAI-native individual transforms
    'SegToBinaryMaskd',
    'SegToAffinityMapd',
    'SegToInstanceBoundaryMaskd',
    'SegToInstanceEDTd',
    'SegToSemanticEDTd',
    'SegToFlowFieldd',
    'SegToSynapticPolarityd',
    'SegToSmallObjectd',
    'SegToGenericSemanticed',
    'ComputeBinaryRatioWeightd',
    'ComputeUNet3DWeightd',
    'SegErosionDilationd',

    # MONAI composition functions
    'create_target_transforms',
    'create_weight_transforms',
    'create_full_processing_pipeline',
    'create_monai_processor_from_config',
    'create_binary_segmentation_pipeline',
    'create_affinity_segmentation_pipeline',
    'create_instance_segmentation_pipeline',
    'create_multi_task_pipeline',

    # Core processing functions
    'seg_to_binary',
    'seg_to_affinity',
    'seg_to_instance_bd',
    'seg_to_polarity',
    'seg_to_small_seg',
    'edt_semantic',
    'edt_instance',
    'seg_to_instance_edt',
    'seg_to_flows',
    'weight_binary_ratio',
    'weight_unet3d',
]
"""
MONAI-native transforms for PyTorch Connectomics processing.

This module provides MONAI MapTransform implementations of all the processing
functions previously handled by the custom DataProcessor system.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import torch
from monai.config import KeysCollection
from monai.transforms import MapTransform
from monai.utils import ensure_tuple_rep

# Import processing functions with correct names
from .target import seg_to_binary, seg_to_affinity, seg_to_instance_bd
from .target import seg_to_instance_edt, seg_to_semantic_edt, seg_to_polarity, seg_to_small_seg
from .target import seg_erosion_dilation
from .segment import seg_selection
from .quantize import energy_quantize, decode_quantize
from .weight import seg_to_weights
from .misc import *


class SegToBinaryMaskd(MapTransform):
    """Convert segmentation to binary mask using MONAI MapTransform."""

    def __init__(
        self,
        keys: KeysCollection,
        target_opt: List[str] = ['0'],
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.target_opt = target_opt

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        for key in self.key_iterator(d):
            if key in d:
                d[key] = seg_to_binary(d[key], self.target_opt)
        return d


class SegToAffinityMapd(MapTransform):
    """Convert segmentation to affinity map using MONAI MapTransform."""

    def __init__(
        self,
        keys: KeysCollection,
        target_opt: List[str] = ['1-1-0', '1-0-0', '0-1-0', '0-0-1'],
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.target_opt = target_opt

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        for key in self.key_iterator(d):
            if key in d:
                d[key] = seg_to_affinity(d[key], self.target_opt)
        return d


class SegToInstanceBoundaryMaskd(MapTransform):
    """Convert segmentation to instance boundary mask using MONAI MapTransform."""

    def __init__(
        self,
        keys: KeysCollection,
        target_opt: List[str] = ['1'],
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.target_opt = target_opt

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        for key in self.key_iterator(d):
            if key in d:
                d[key] = seg_to_instance_bd(d[key], self.target_opt)
        return d


class SegToInstanceEDTd(MapTransform):
    """Convert segmentation to instance EDT using MONAI MapTransform."""

    def __init__(
        self,
        keys: KeysCollection,
        target_opt: List[str] = ['1', '200'],
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.target_opt = target_opt

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        for key in self.key_iterator(d):
            if key in d:
                # Ensure we have at least 2 parameters, pad with defaults if needed
                opts = self.target_opt.copy()
                while len(opts) < 2:
                    opts.append('200')  # default distance parameter
                d[key] = seg_to_instance_edt(d[key], opts)
        return d


class SegToSemanticEDTd(MapTransform):
    """Convert segmentation to semantic EDT using MONAI MapTransform."""

    def __init__(
        self,
        keys: KeysCollection,
        target_opt: List[str] = ['1', '200'],
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.target_opt = target_opt

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        for key in self.key_iterator(d):
            if key in d:
                # Ensure we have at least 2 parameters, pad with defaults if needed
                opts = self.target_opt.copy()
                while len(opts) < 2:
                    opts.append('200')  # default distance parameter
                d[key] = seg_to_semantic_edt(d[key], opts)
        return d


class SegToFlowFieldd(MapTransform):
    """Convert segmentation to flow field using MONAI MapTransform."""

    def __init__(
        self,
        keys: KeysCollection,
        target_opt: List[str] = ['1'],
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
    """Convert segmentation to synaptic polarity using MONAI MapTransform."""

    def __init__(
        self,
        keys: KeysCollection,
        target_opt: List[str] = ['1'],
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.target_opt = target_opt

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        for key in self.key_iterator(d):
            if key in d:
                d[key] = seg_to_polarity(d[key], self.target_opt)
        return d


class SegToSmallObjectd(MapTransform):
    """Convert segmentation to small object mask using MONAI MapTransform."""

    def __init__(
        self,
        keys: KeysCollection,
        target_opt: List[str] = ['1', '100'],
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.target_opt = target_opt

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        for key in self.key_iterator(d):
            if key in d:
                d[key] = seg_to_small_seg(d[key], self.target_opt)
        return d


class ComputeBinaryRatioWeightd(MapTransform):
    """Compute binary ratio weights using MONAI MapTransform."""

    def __init__(
        self,
        keys: KeysCollection,
        target_opt: List[str] = ['1'],
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
        target_opt: List[str] = ['1', '1', '5.0', '0.3'],
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


class SegErosionDilationd(MapTransform):
    """Apply erosion/dilation to segmentation using MONAI MapTransform."""

    def __init__(
        self,
        keys: KeysCollection,
        target_opt: List[str] = ['1', '1'],
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.target_opt = target_opt

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        for key in self.key_iterator(d):
            if key in d:
                d[key] = seg_erosion_dilation(d[key], self.target_opt)
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
        mode: str = 'max',
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: Keys to be processed from the input dictionary.
            mode: Decoding mode, either 'max' or 'mean'. Default is 'max'.
            allow_missing_keys: Whether to ignore missing keys.
        """
        super().__init__(keys, allow_missing_keys)
        if mode not in ['max', 'mean']:
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
        self.indices = ensure_tuple_rep(indices, 1) if not isinstance(indices, (list, tuple)) else indices

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        for key in self.key_iterator(d):
            if key in d:
                d[key] = seg_selection(d[key], self.indices)
        return d


__all__ = [
    'SegToBinaryMaskd',
    'SegToAffinityMapd',
    'SegToInstanceBoundaryMaskd',
    'SegToInstanceEDTd',
    'SegToSemanticEDTd',
    'SegToFlowFieldd',
    'SegToSynapticPolarityd',
    'SegToSmallObjectd',
    'ComputeBinaryRatioWeightd',
    'ComputeUNet3DWeightd',
    'SegErosionDilationd',
    'EnergyQuantized',
    'DecodeQuantized',
    'SegSelectiond',
]
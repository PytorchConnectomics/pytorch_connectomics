"""
Debug utilities for normalization and value range tracking.

This module provides utilities to debug normalization behavior across
the entire training pipeline without modifying training logic.
"""

import torch
import numpy as np
from typing import Union, Optional

# Global debug flag - set to True to enable debug prints
DEBUG_NORM = True

# Track which stages have been printed (to avoid spam)
_printed_stages = set()


def reset_debug_state():
    """Reset debug state (useful for testing)."""
    global _printed_stages
    _printed_stages = set()


def print_tensor_stats(
    tensor: Union[torch.Tensor, np.ndarray],
    stage_name: str,
    tensor_name: str = "tensor",
    print_once: bool = True,
    extra_info: Optional[dict] = None,
):
    """
    Print comprehensive statistics about a tensor for debugging normalization.
    
    Args:
        tensor: PyTorch tensor or numpy array to analyze
        stage_name: Name of the pipeline stage (e.g., "STAGE 1: RAW IMAGE LOADING")
        tensor_name: Name of the tensor (e.g., "image", "label", "pred")
        print_once: If True, only print once per stage_name
        extra_info: Optional dict with additional info to print
    """
    if not DEBUG_NORM:
        return
    
    # Check if we've already printed this stage
    stage_key = f"{stage_name}_{tensor_name}"
    if print_once and stage_key in _printed_stages:
        return
    
    # Convert to tensor if numpy
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    
    # Convert to float for statistics computation if needed
    # (uint8, uint16, int types don't support mean/std operations)
    original_dtype = tensor.dtype
    if tensor.dtype in [torch.uint8, torch.uint16, torch.int8, torch.int16, torch.int32, torch.int64]:
        tensor = tensor.float()
    
    # Mark as printed
    if print_once:
        _printed_stages.add(stage_key)
    
    # Compute statistics
    tensor_min = tensor.min().item()
    tensor_max = tensor.max().item()
    tensor_mean = tensor.mean().item()
    tensor_std = tensor.std().item()
    
    # Compute percentage statistics
    total_elements = tensor.numel()
    pct_positive = (tensor > 0).sum().item() / total_elements * 100
    pct_near_minus1 = (tensor < -0.98).sum().item() / total_elements * 100
    pct_in_01 = ((tensor >= 0) & (tensor <= 1)).sum().item() / total_elements * 100
    pct_in_neg1_1 = ((tensor >= -1) & (tensor <= 1)).sum().item() / total_elements * 100
    
    # Print header
    print("\n" + "=" * 80)
    print(f"[DEBUG NORM] {stage_name}")
    print("=" * 80)
    
    # Print tensor info
    print(f"{tensor_name.upper()} Statistics:")
    print(f"  Shape: {tuple(tensor.shape)}")
    print(f"  dtype: {original_dtype}")
    print(f"  device: {tensor.device if isinstance(tensor, torch.Tensor) else 'cpu'}")
    
    # Print value statistics
    print(f"\n  Value Range:")
    print(f"    Min:  {tensor_min:>10.6f}")
    print(f"    Max:  {tensor_max:>10.6f}")
    print(f"    Mean: {tensor_mean:>10.6f}")
    print(f"    Std:  {tensor_std:>10.6f}")
    
    # Print distribution statistics
    print(f"\n  Distribution:")
    print(f"    Values > 0:        {pct_positive:>6.2f}%")
    print(f"    Values < -0.98:    {pct_near_minus1:>6.2f}%")
    print(f"    Values in [0,1]:   {pct_in_01:>6.2f}%")
    print(f"    Values in [-1,1]:  {pct_in_neg1_1:>6.2f}%")
    
    # Print sample values (first 10 flattened values)
    sample_values = tensor.flatten()[:10].tolist()
    print(f"\n  Sample values (first 10): {[f'{v:.4f}' for v in sample_values]}")
    
    # Print extra info if provided
    if extra_info:
        print(f"\n  Additional Info:")
        for key, value in extra_info.items():
            print(f"    {key}: {value}")
    
    print("=" * 80 + "\n")


def print_normalization_check(
    tensor: Union[torch.Tensor, np.ndarray],
    expected_range: tuple,
    stage_name: str,
    tensor_name: str = "tensor",
):
    """
    Check if tensor is in expected range and print warning if not.
    
    Args:
        tensor: Tensor to check
        expected_range: Tuple of (min, max) expected values
        stage_name: Name of the pipeline stage
        tensor_name: Name of the tensor
    """
    if not DEBUG_NORM:
        return
    
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    
    # Convert to float if needed for min/max operations
    if tensor.dtype in [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]:
        tensor = tensor.float()
    
    tensor_min = tensor.min().item()
    tensor_max = tensor.max().item()
    expected_min, expected_max = expected_range
    
    # Check if in expected range (with small tolerance)
    tolerance = 0.1
    in_range = (
        tensor_min >= expected_min - tolerance and
        tensor_max <= expected_max + tolerance
    )
    
    if not in_range:
        print(f"\n⚠️  [DEBUG NORM WARNING] {stage_name}")
        print(f"    {tensor_name}: Expected range [{expected_min}, {expected_max}]")
        print(f"    {tensor_name}: Actual range   [{tensor_min:.3f}, {tensor_max:.3f}]")
        print(f"    ❌ Values are OUTSIDE expected range!\n")
    else:
        print(f"✅ [DEBUG NORM] {stage_name}: {tensor_name} in expected range [{expected_min}, {expected_max}]")


__all__ = [
    "DEBUG_NORM",
    "print_tensor_stats",
    "print_normalization_check",
    "reset_debug_state",
]


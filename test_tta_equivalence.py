#!/usr/bin/env python
"""
Test script to verify TTA with null flip_axes gives identical results to non-TTA.

This script tests that:
1. test_time_augmentation=false
2. test_time_augmentation=true, tta_flip_axes=null

Give identical predictions (bit-exact).
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from connectomics.config import load_config
from connectomics.lightning import ConnectomicsModule

def test_tta_equivalence():
    """Test that TTA with null flip axes is equivalent to no TTA."""

    # Load config
    config_path = "tutorials/monai_lucchi++.yaml"
    cfg = load_config(config_path)

    # Create a dummy input
    batch_size = 1
    channels = 1
    depth, height, width = 32, 32, 32
    dummy_input = torch.randn(batch_size, channels, depth, height, width)

    print("=" * 60)
    print("Testing TTA Equivalence")
    print("=" * 60)

    # Test 1: No TTA
    print("\n1. Testing without TTA...")
    cfg.inference.test_time_augmentation = False
    model1 = ConnectomicsModule(cfg)
    model1.eval()

    with torch.no_grad():
        # Directly call sliding window predict to simulate test_step path
        output1 = model1._sliding_window_predict(dummy_input)

    print(f"   Output shape: {output1.shape}")
    print(f"   Output mean: {output1.mean().item():.6f}")
    print(f"   Output std: {output1.std().item():.6f}")

    # Test 2: TTA with null flip_axes
    print("\n2. Testing with TTA but null flip_axes...")
    cfg.inference.test_time_augmentation = True
    cfg.inference.tta_flip_axes = None
    model2 = ConnectomicsModule(cfg)
    model2.eval()

    # Copy weights from model1 to ensure same model
    model2.load_state_dict(model1.state_dict())

    with torch.no_grad():
        output2 = model2._sliding_window_predict(dummy_input)

    print(f"   Output shape: {output2.shape}")
    print(f"   Output mean: {output2.mean().item():.6f}")
    print(f"   Output std: {output2.std().item():.6f}")

    # Compare outputs
    print("\n" + "=" * 60)
    print("Comparison Results:")
    print("=" * 60)

    # Check if outputs are identical
    abs_diff = torch.abs(output1 - output2)
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()

    print(f"Max absolute difference: {max_diff:.10f}")
    print(f"Mean absolute difference: {mean_diff:.10f}")

    # Check if bit-exact (or within numerical precision)
    tolerance = 1e-6
    is_identical = torch.allclose(output1, output2, rtol=tolerance, atol=tolerance)

    if is_identical:
        print(f"✅ PASS: Outputs are identical (within tolerance {tolerance})")
    else:
        print(f"❌ FAIL: Outputs differ by more than tolerance {tolerance}")
        print(f"   Max diff: {max_diff}")
        print(f"   This could indicate different code paths or model state")

    return is_identical

if __name__ == "__main__":
    success = test_tta_equivalence()
    sys.exit(0 if success else 1)

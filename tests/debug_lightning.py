#!/usr/bin/env python3
"""
Debug Lightning training to see what's happening with tensor dimensions.
"""

import os
import sys
import torch
import numpy as np

# Add debugging to the Lightning module
def debug_training_step():
    """Add debugging to understand the tensor dimension issue."""
    from connectomics.config import load_cfg
    from connectomics.engine.lightning_trainer import LightningTrainer
    from connectomics.engine.lightning_module import ConnectomicsDataModule

    # Create mock args for the real config
    class MockArgs:
        def __init__(self):
            self.config_file = "../configs/Lucchi-Mitochondria.yaml"
            self.config_base = None
            self.opts = [
                "SOLVER.ITERATION_TOTAL", "1",
                "SOLVER.SAMPLES_PER_BATCH", "1",
                "SYSTEM.NUM_GPUS", "0",
                "SYSTEM.PARALLEL", "NONE",
                "DATASET.OUTPUT_PATH", "/tmp/debug_output"
            ]
            self.inference = False
            self.distributed = False
            self.checkpoint = None
            self.manual_seed = None
            self.local_world_size = 1
            self.local_rank = None
            self.debug = False

    args = MockArgs()
    cfg = load_cfg(args, freeze=False)

    print("🔍 Debug: Lightning with Real Lucchi Data")
    print(f"Expected input size: {cfg.MODEL.INPUT_SIZE}")
    print(f"Expected output size: {cfg.MODEL.OUTPUT_SIZE}")

    # Create data module
    data_module = ConnectomicsDataModule(cfg, use_monai=False)  # Disable MONAI for debugging
    data_module.setup(stage='fit')

    # Get a single batch
    train_loader = data_module.train_dataloader()
    batch = next(iter(train_loader))

    print(f"\n📦 Batch Analysis:")
    print(f"Batch type: {type(batch)}")

    if hasattr(batch, 'out_input'):
        volume = batch.out_input
        target = batch.out_target_l
        weight = batch.out_weight_l
        print(f"VolumeDataset format detected")
    elif isinstance(batch, (list, tuple)):
        volume = batch[0]
        target = batch[1] if len(batch) > 1 else None
        weight = batch[2] if len(batch) > 2 else None
        print(f"List/tuple format detected")
    else:
        print(f"Unknown batch format")
        return

    print(f"Volume shape: {volume.shape if torch.is_tensor(volume) else 'Not a tensor'}")
    print(f"Volume type: {type(volume)}")
    if torch.is_tensor(volume):
        print(f"Volume dtype: {volume.dtype}")
        print(f"Volume device: {volume.device}")

    if target is not None:
        if isinstance(target, list):
            print(f"Target list length: {len(target)}")
            for i, t in enumerate(target):
                print(f"Target[{i}] shape: {t.shape if torch.is_tensor(t) else 'Not a tensor'}")
        else:
            print(f"Target shape: {target.shape if torch.is_tensor(target) else 'Not a tensor'}")

    # Test the model forward pass step by step
    print(f"\n🧠 Model Forward Pass Analysis:")

    from connectomics.engine.lightning_module import ConnectomicsModule
    lightning_module = ConnectomicsModule(cfg)

    # Handle batch format the same way the Lightning module does
    if hasattr(batch, 'out_input'):
        volume = batch.out_input
        target = batch.out_target_l
        weight = batch.out_weight_l
    elif isinstance(batch, (list, tuple)) and len(batch) >= 3:
        volume = batch[0]
        target = [batch[1]] if not isinstance(batch[1], list) else batch[1]
        weight = [batch[2]] if not isinstance(batch[2], list) else batch[2]

        if isinstance(volume, list):
            volume = torch.stack(volume) if len(volume) > 1 else volume[0]

    print(f"After batch processing - Volume shape: {volume.shape}")

    # Apply dimension fixes
    original_shape = volume.shape
    if volume.dim() == 2:
        volume = volume.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    elif volume.dim() == 3:
        volume = volume.unsqueeze(0).unsqueeze(0)
    elif volume.dim() == 4:
        if volume.shape[0] == 1:
            volume = volume.unsqueeze(1)
        else:
            volume = volume.unsqueeze(0)

    print(f"Original volume shape: {original_shape}")
    print(f"After dimension fix: {volume.shape}")

    # Convert to float
    if volume.dtype != torch.float32:
        volume = volume.float()

    print(f"After type conversion: {volume.dtype}")

    # Try forward pass
    try:
        with torch.no_grad():
            pred = lightning_module.forward(volume)
            print(f"Model output shape: {pred.shape}")
            print(f"Expected target shape: {target[0].shape if isinstance(target, list) else target.shape}")

            # Check if shapes match
            target_tensor = target[0] if isinstance(target, list) else target
            if pred.shape == target_tensor.shape:
                print("✅ Shapes match!")
            else:
                print("❌ Shape mismatch!")
                print(f"Difference: {pred.shape} vs {target_tensor.shape}")

    except Exception as e:
        print(f"❌ Model forward failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_training_step()
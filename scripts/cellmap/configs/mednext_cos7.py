"""
CellMap training config: MedNeXt on COS7 cells

Multi-organelle segmentation (nuc, mito, er, golgi, ves)

Usage:
    python scripts/cellmap/train_cellmap.py scripts/cellmap/configs/mednext_cos7.py
"""

from cellmap_segmentation_challenge.utils import get_tested_classes

# Model configuration
model_name = 'mednext'
mednext_size = 'M'              # Medium (17.6M params) - good balance
mednext_kernel_size = 5         # 5x5x5 kernels
deep_supervision = True         # Multi-scale loss (recommended)

# Classes to segment (all major organelles in COS7)
classes = ['nuc', 'mito', 'er', 'golgi', 'ves']

# Data configuration
input_array_info = {
    'shape': (128, 128, 128),   # Patch size in voxels
    'scale': (8, 8, 8),         # 8nm isotropic resolution
}
target_array_info = input_array_info
force_all_classes = 'both'        # keep every organelle present in both splits

# Output paths
output_dir = 'outputs/cellmap_cos7'
datasplit_path = f'{output_dir}/datasplit.csv'

# Spatial augmentation (CellMap format)
spatial_transforms = {
    'mirror': {
        'axes': {'x': 0.5, 'y': 0.5, 'z': 0.5}  # 50% chance to flip each axis
    },
    'transpose': {
        'axes': ['x', 'y', 'z']                  # Random axis permutation
    },
    'rotate': {
        'axes': {
            'x': [-180, 180],                    # Random rotation range
            'y': [-180, 180],
            'z': [-180, 180]
        }
    },
}

# Training hyperparameters
learning_rate = 1e-3            # MedNeXt recommended: constant 1e-3
batch_size = 2                  # Per GPU (effective batch = 2 * 4 = 8 with grad accum)
epochs = 500                    # Maximum epochs
num_gpus = 1                    # Number of GPUs
precision = '16-mixed'          # Mixed precision training
iterations_per_epoch = None     # Keep dataloader on the cheap shuffle path
train_batches_per_epoch = 2000  # Lightning caps epoch length at 2k steps
val_batches_per_epoch = 200     # Limit validation passes per epoch

# Learning rate scheduler (constant for MedNeXt)
scheduler_config = {
    'name': 'constant',         # No scheduler (MedNeXt recommendation)
}

"""
CellMap training config: MedNeXt for Mitochondria segmentation

Instance segmentation of mitochondria across multiple datasets

Usage:
    python scripts/cellmap/train_cellmap.py scripts/cellmap/configs/mednext_mito.py
"""

# Model configuration
model_name = 'mednext'
mednext_size = 'L'              # Large (61.8M params) - best for single class
mednext_kernel_size = 7         # 7x7x7 kernels for larger receptive field
deep_supervision = True         # Multi-scale loss (recommended)

# Classes to segment (mitochondria only)
classes = ['mito']

# Data configuration (higher resolution for mitochondria)
input_array_info = {
    'shape': (96, 96, 96),      # Smaller patches for higher resolution
    'scale': (4, 4, 4),         # 4nm isotropic (higher resolution)
}
target_array_info = input_array_info

# Output paths
output_dir = 'outputs/cellmap_mito'
datasplit_path = f'{output_dir}/datasplit.csv'

# Spatial augmentation
spatial_transforms = {
    'mirror': {
        'axes': {'x': 0.5, 'y': 0.5, 'z': 0.5}
    },
    'transpose': {
        'axes': ['x', 'y', 'z']
    },
    'rotate': {
        'axes': {
            'x': [-180, 180],
            'y': [-180, 180],
            'z': [-180, 180]
        }
    },
}

# Training hyperparameters
learning_rate = 5e-4            # Slightly lower for larger model
batch_size = 1                  # Smaller due to larger model + higher resolution
epochs = 1000                   # More epochs for single class
num_gpus = 1                    # Number of GPUs
precision = '16-mixed'          # Mixed precision training

# Learning rate scheduler
scheduler_config = {
    'name': 'constant',
}

"""
CellMap training config: MONAI Basic U-Net (Quick test)

Fast training for testing the pipeline with a lightweight model

Usage:
    python scripts/cellmap/train_cellmap.py scripts/cellmap/configs/monai_unet_quick.py
"""

# Model configuration (basic U-Net - lightweight)
model_name = 'monai_basic_unet3d'
deep_supervision = False        # Disabled for speed

# Classes to segment (just 2 classes for quick test)
classes = ['nuc', 'mito']

# Data configuration
input_array_info = {
    'shape': (64, 64, 64),      # Small patches for speed
    'scale': (8, 8, 8),         # 8nm isotropic
}
target_array_info = input_array_info

# Output paths
output_dir = 'outputs/cellmap_quick_test'
datasplit_path = f'{output_dir}/datasplit.csv'

# Minimal augmentation for speed
spatial_transforms = {
    'mirror': {
        'axes': {'x': 0.5, 'y': 0.5}  # Only XY flips
    },
}

# Training hyperparameters (fast settings)
learning_rate = 1e-3
batch_size = 4                  # Larger batch for small patches
epochs = 10                     # Just 10 epochs for quick test
num_gpus = 1
precision = '16-mixed'

# Cosine annealing scheduler
scheduler_config = {
    'name': 'cosine',
    'T_max': 10,
    'min_lr': 1e-6,
}

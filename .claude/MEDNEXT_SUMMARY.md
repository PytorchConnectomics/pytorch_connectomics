# MedNeXt Architecture Reference

This document provides guidance for understanding and integrating MedNeXt architectures from the `/projects/weilab/weidf/lib/MedNeXt` repository.

## Overview

**MedNeXt** is a fully ConvNeXt-based architecture for 3D medical image segmentation, developed by the German Cancer Research Center (DKFZ). It leverages ConvNeXt blocks customized for sparsely annotated medical imaging datasets, particularly for connectomics and volumetric segmentation tasks.

**Key Publication:** Roy et al., "MedNeXt: Transformer-driven Scaling of ConvNets for Medical Image Segmentation", MICCAI 2023

**Repository Location:** `/projects/weilab/weidf/lib/MedNeXt`

## Core Architecture Components

### 1. MedNeXt Block (`blocks.py`)

The fundamental building block consists of:
- **Depthwise Convolution**: Groups-based convolution with configurable kernel sizes (3x3x3, 5x5x5, 7x7x7)
- **Normalization**: GroupNorm or LayerNorm
- **Expansion Layer**: 1x1x1 convolution with expansion ratio (typically 2-8x)
- **GELU Activation**: Non-linear activation function
- **GRN (Global Response Normalization)**: Optional feature normalization
- **Compression Layer**: 1x1x1 convolution back to original channels
- **Residual Connection**: Optional skip connection

**Block Variants:**
```python
from nnunet_mednext import MedNeXtBlock, MedNeXtDownBlock, MedNeXtUpBlock

# Standard block (same resolution)
block = MedNeXtBlock(
    in_channels=32,
    out_channels=32,
    exp_r=4,              # Expansion ratio
    kernel_size=7,        # 3, 5, or 7
    do_res=True,          # Enable residual connection
    norm_type='group',    # 'group' or 'layer'
    dim='3d'              # '2d' or '3d'
)

# Downsampling block (2x stride)
down_block = MedNeXtDownBlock(
    in_channels=32,
    out_channels=64,
    exp_r=4,
    kernel_size=7,
    do_res=True
)

# Upsampling block (2x transposed conv)
up_block = MedNeXtUpBlock(
    in_channels=64,
    out_channels=32,
    exp_r=4,
    kernel_size=7,
    do_res=True
)
```

### 2. MedNeXt Architecture (`MedNextV1.py`)

**U-Net Style Encoder-Decoder:**
- 5 encoder levels (with 4 downsampling operations)
- Bottleneck layer
- 4 decoder levels (with 4 upsampling operations)
- Skip connections between encoder and decoder
- Optional deep supervision at multiple scales

**Architecture Parameters:**
```python
from nnunet_mednext.mednextv1 import MedNeXt

model = MedNeXt(
    in_channels=1,                          # Input channels
    n_channels=32,                          # Base number of channels
    n_classes=3,                            # Number of output classes
    exp_r=4,                                # Expansion ratio (int or list)
    kernel_size=7,                          # Kernel size for all layers
    enc_kernel_size=None,                   # Separate encoder kernel size
    dec_kernel_size=None,                   # Separate decoder kernel size
    deep_supervision=True,                  # Multi-scale outputs
    do_res=True,                            # Residual in MedNeXt blocks
    do_res_up_down=True,                    # Residual in up/down blocks
    checkpoint_style='outside_block',       # Gradient checkpointing
    block_counts=[2,2,2,2,2,2,2,2,2],      # Blocks per level
    norm_type='group',                      # Normalization type
    dim='3d',                               # 2D or 3D
    grn=False                               # Global Response Normalization
)
```

### 3. Predefined Model Sizes

MedNeXt provides 4 architecture sizes tested in the MICCAI 2023 paper:

| Model ID | Kernel | n_channels | exp_r | block_counts | Parameters | GFlops |
|----------|--------|------------|-------|--------------|------------|--------|
| S | 3x3x3 | 32 | 2 | [2,2,2,2,2,2,2,2,2] | 5.6M | 130 |
| S | 5x5x5 | 32 | 2 | [2,2,2,2,2,2,2,2,2] | 5.9M | 169 |
| B | 3x3x3 | 32 | [2,3,4,4,4,4,4,3,2] | [2,2,2,2,2,2,2,2,2] | 10.5M | 170 |
| B | 5x5x5 | 32 | [2,3,4,4,4,4,4,3,2] | [2,2,2,2,2,2,2,2,2] | 11.0M | 208 |
| M | 3x3x3 | 32 | [2,3,4,4,4,4,4,3,2] | [3,4,4,4,4,4,4,4,3] | 17.6M | 248 |
| M | 5x5x5 | 32 | [2,3,4,4,4,4,4,3,2] | [3,4,4,4,4,4,4,4,3] | 18.3M | 308 |
| L | 3x3x3 | 32 | [3,4,8,8,8,8,8,4,3] | [3,4,8,8,8,8,8,4,3] | 61.8M | 500 |
| L | 5x5x5 | 32 | [3,4,8,8,8,8,8,4,3] | [3,4,8,8,8,8,8,4,3] | 63.0M | 564 |

**Factory Function:**
```python
from nnunet_mednext import create_mednext_v1

model = create_mednext_v1(
    num_channels=1,
    num_classes=3,
    model_id='B',              # 'S', 'B', 'M', or 'L'
    kernel_size=3,             # 3 or 5
    deep_supervision=True
)
```

## Integration with PyTorch Connectomics

### Approach 1: Use MedNeXt as a Standalone Module

```python
# In connectomics/models/arch/
from nnunet_mednext import create_mednext_v1

class MedNeXtWrapper(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = create_mednext_v1(
            num_channels=cfg.MODEL.IN_PLANES,
            num_classes=cfg.MODEL.OUT_PLANES,
            model_id=cfg.MODEL.ARCHITECTURE,  # 'S', 'B', 'M', 'L'
            kernel_size=cfg.MODEL.KERNEL_SIZE,
            deep_supervision=cfg.MODEL.DEEP_SUPERVISION
        )

    def forward(self, x):
        return self.model(x)
```

### Approach 2: Integrate MedNeXt Blocks into Existing Architectures

```python
# Use MedNeXt blocks in custom architectures
from nnunet_mednext import MedNeXtBlock, MedNeXtDownBlock, MedNeXtUpBlock

class CustomMedNeXtArch(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            MedNeXtBlock(in_channels, 32, exp_r=4, kernel_size=7, do_res=True),
            MedNeXtDownBlock(32, 64, exp_r=4, kernel_size=7, do_res=True),
            MedNeXtBlock(64, 64, exp_r=4, kernel_size=7, do_res=True),
        )
        # ... decoder, etc.
```

### Approach 3: Adapt MedNeXt Training Pipeline

The MedNeXt repository includes nnUNet-based trainers that can be adapted:

**Key Trainer Features:**
- AdamW optimizer with initial LR = 1e-3
- No LR scheduler (constant learning rate)
- Weight decay for regularization
- Deep supervision loss
- Gradient checkpointing for large models

**Reference Trainer:** `/projects/weilab/weidf/lib/MedNeXt/nnunet_mednext/training/network_training/MedNeXt/nnUNetTrainerV2_MedNeXt.py`

## Key Design Principles

### 1. Isotropic Spacing
MedNeXt v1 was designed for **1.0mm isotropic spacing** data, unlike nnUNet's median spacing approach. This is important for preprocessing:

```python
# Preprocessing should target 1x1x1mm spacing
target_spacing = (1.0, 1.0, 1.0)
patch_size = (128, 128, 128)
```

### 2. Deep Supervision
Deep supervision is crucial for MedNeXt performance. The model outputs predictions at 5 different scales:
- Full resolution
- 1/2 resolution
- 1/4 resolution
- 1/8 resolution
- 1/16 resolution (bottleneck)

Loss is computed as weighted sum across all scales.

### 3. Residual Connections
Two types of residual connections:
- **Block-level**: Inside MedNeXt blocks (`do_res=True`)
- **Resampling-level**: In up/down blocks (`do_res_up_down=True`)

Both should typically be enabled for best performance.

### 4. UpKern Weight Initialization
UpKern allows initializing large kernel models (5x5x5) from small kernel models (3x3x3) via trilinear interpolation:

```python
from nnunet_mednext.run.load_weights import upkern_load_weights

# Train small kernel model first
model_k3 = create_mednext_v1(1, 3, 'B', kernel_size=3)
# ... train model_k3 ...

# Initialize large kernel model
model_k5 = create_mednext_v1(1, 3, 'B', kernel_size=5)
model_k5 = upkern_load_weights(model_k5, model_k3)
# ... continue training model_k5 ...
```

### 5. Gradient Checkpointing
For large models (M, L), use `checkpoint_style='outside_block'` to reduce memory usage at the cost of computation:

```python
model = MedNeXt(..., checkpoint_style='outside_block')
```

## Loss Functions

MedNeXt typically uses the same loss functions as nnUNet, available in:
`/projects/weilab/weidf/lib/MedNeXt/nnunet_mednext/training/loss_functions/`

- **dice_loss.py**: Soft Dice loss
- **crossentropy.py**: Cross-entropy loss
- **deep_supervision.py**: Multi-scale loss wrapper
- **focal_loss.py**: Focal loss for class imbalance
- **TopK_loss.py**: TopK loss for hard example mining

Default: Dice + Cross-Entropy with deep supervision

## Data Format and Preprocessing

### Expected Input Format
- **Shape**: (batch, channels, depth, height, width) for 3D
- **Spacing**: 1.0mm isotropic (recommended)
- **Patch Size**: 128x128x128 (standard)
- **Normalization**: Z-score normalization per sample

### Preprocessing Pipeline
The MedNeXt repository uses nnUNet v1 preprocessing:

```bash
# From MedNeXt repo
mednextv1_plan_and_preprocess \
    -t TaskXXX_YourTask \
    -pl3d ExperimentPlanner3D_v21_customTargetSpacing_1x1x1
```

## Training Commands (Reference)

```bash
# Basic training (from MedNeXt repo)
mednextv1_train 3d_fullres \
    nnUNetTrainerV2_MedNeXt_B_kernel3 \
    Task040_KiTS2019 \
    0 \
    -p nnUNetPlansv2.1_trgSp_1x1x1

# With UpKern initialization
mednextv1_train 3d_fullres \
    nnUNetTrainerV2_MedNeXt_B_kernel5 \
    Task040_KiTS2019 \
    0 \
    -p nnUNetPlansv2.1_trgSp_1x1x1 \
    -pretrained_weights /path/to/kernel3/model_final_checkpoint.model \
    -resample_weights
```

## Integration Checklist for PyTorch Connectomics

- [ ] Import MedNeXt modules: `from nnunet_mednext import MedNeXt, MedNeXtBlock`
- [ ] Add MedNeXt to model factory in `connectomics/models/build.py`
- [ ] Configure architecture in YACS config system
- [ ] Ensure input preprocessing matches 1mm isotropic spacing
- [ ] Implement deep supervision loss if not already available
- [ ] Test with 5D input tensors (batch, channel, D, H, W)
- [ ] Consider gradient checkpointing for large models
- [ ] Adapt optimizer settings (AdamW, lr=1e-3, no scheduler)

## File Structure in MedNeXt Repository

```
/projects/weilab/weidf/lib/MedNeXt/
├── nnunet_mednext/
│   ├── network_architecture/
│   │   └── mednextv1/
│   │       ├── MedNextV1.py          # Main architecture
│   │       ├── blocks.py              # MedNeXt blocks
│   │       ├── create_mednext_v1.py   # Factory functions
│   │       └── __init__.py
│   ├── training/
│   │   ├── loss_functions/            # Loss implementations
│   │   ├── network_training/
│   │   │   └── MedNeXt/
│   │   │       └── nnUNetTrainerV2_MedNeXt.py  # Trainers
│   │   └── data_augmentation/         # Augmentation pipeline
│   ├── preprocessing/                 # Data preprocessing
│   ├── inference/                     # Inference utilities
│   └── run/
│       └── load_weights.py            # UpKern implementation
├── documentation/                     # Usage guides
├── tests/                             # Architecture tests
├── README.md                          # Main documentation
└── setup.py                           # Installation script
```

## Important Notes

1. **Dimensionality**: While MedNeXt supports 2D (`dim='2d'`), it has only been tested in 3D mode
2. **Memory**: Large models (M, L) require gradient checkpointing on consumer GPUs
3. **Batch Size**: Typically 1-2 for 128³ patches on 24GB GPU
4. **Training Time**: Slower than standard UNet due to larger kernels and blocks
5. **Compatibility**: Built on PyTorch, easily integrates with PyTorch Connectomics

## Additional Resources

- **Paper**: https://arxiv.org/abs/2303.09975
- **Original Repo**: https://github.com/MIC-DKFZ/MedNeXt
- **nnUNet v1**: https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1

## Example Integration Pseudocode

```python
# In connectomics/models/build.py
def build_model(cfg, device):
    if cfg.MODEL.ARCHITECTURE == 'mednext':
        from nnunet_mednext import create_mednext_v1
        model = create_mednext_v1(
            num_channels=cfg.MODEL.IN_PLANES,
            num_classes=cfg.MODEL.OUT_PLANES,
            model_id=cfg.MODEL.MEDNEXT_SIZE,  # 'S', 'B', 'M', 'L'
            kernel_size=cfg.MODEL.KERNEL_SIZE,
            deep_supervision=cfg.MODEL.DEEP_SUPERVISION
        )
    # ... other architectures ...
    return model.to(device)
```
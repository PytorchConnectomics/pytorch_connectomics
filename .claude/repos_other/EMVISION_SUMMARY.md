# PyTorch EMVision Summary

**Location:** `/projects/weilab/weidf/lib/seg/pytorch-emvision`
**Author:** Kisuk Lee (MIT)
**Package:** `emvision` v0.0.1
**Purpose:** 3D Electron Microscopy (EM) models for PyTorch deep learning

---

## Overview

PyTorch EMVision is a collection of specialized 3D neural network architectures designed for electron microscopy image segmentation. The library focuses on **Residual Symmetric U-Net (RSUNet)** variants optimized for volumetric EM data with anisotropic resolution (typically high XY resolution, lower Z resolution).

### Key Philosophy

- **Residual connections** for better gradient flow and performance
- **Anisotropic convolutions** respecting EM data characteristics (e.g., (1,3,3) kernels)
- **Flexible normalization** (Batch Norm, Group Norm, Instance Norm)
- **Modular design** with interchangeable components

---

## Core Architectures

### 1. RSUNet (Residual Symmetric U-Net)

**File:** `emvision/models/rsunet.py`

The base architecture combining residual blocks with U-Net structure:

```python
from emvision.models import RSUNet

# Default configuration
model = RSUNet(width=[16, 32, 64, 128, 256, 512])
```

**Architecture:**
- **Encoder:** MaxPool3d downsampling with (1,2,2) factor (preserve Z resolution)
- **ResBlocks:** Pre-activation design (BN→ReLU→Conv)
- **Decoder:** BilinearUp upsampling with skip connections (element-wise addition)
- **Channels:** Configurable width at each level

**Key Components:**
- `ConvBlock`: Pre-ResBlock-Post pattern (3 BNReLUConv layers with residual)
- `ResBlock`: Two conv layers with residual connection
- `UpBlock`: Bilinear upsampling + 1x1 conv + skip connection
- `BilinearUp`: Caffe-style bilinear upsampling (learnable weights)

**Default Downsampling Factor:** (1,2,2) - anisotropic for EM data

---

### 2. isoRSUNet (Isotropic RSUNet)

**File:** `emvision/models/iso_rsunet.py`

RSUNet variant for **isotropic voxel resolution** data:

```python
from emvision.models import isoRSUNet

model = isoRSUNet(width=[16, 32, 64, 128, 256, 512])
```

**Difference from RSUNet:**
- Downsampling: (2,2,2) instead of (1,2,2)
- Upsampling: (2,2,2) instead of (1,2,2)
- Suitable for data with equal resolution in all dimensions

---

### 3. RSUNet with Group Normalization

**File:** `emvision/models/rsunet_gn.py`

RSUNet using **Group Normalization** instead of Batch Normalization:

```python
from emvision.models import rsunet_gn

model = rsunet_gn(width=[16, 32, 64, 128, 256, 512], group=16)
```

**Advantages:**
- Better for small batch sizes
- More stable training
- Independent of batch size

---

### 4. RSUNet with Flexible Activations

**Files:**
- `rsunet_act.py` - With Batch Norm
- `rsunet_act_gn.py` - With Group Norm
- `rsunet_act_in.py` - With Instance Norm
- `rsunet_act_nn.py` - Without normalization
- `rsunet_act_nn_gn.py` - Group Norm, no activation norm

Support for **multiple activation functions**:

```python
from emvision.models import rsunet_act

# PReLU activation
model = rsunet_act(width=[16,32,64,128], act='PReLU', init=0.1)

# LeakyReLU activation
model = rsunet_act(width=[16,32,64,128], act='LeakyReLU', negative_slope=0.1)

# ELU activation
model = rsunet_act(width=[16,32,64,128], act='ELU')

# ReLU (default)
model = rsunet_act(width=[16,32,64,128], act='ReLU')
```

**Supported Activations:**
- `ReLU` (default)
- `LeakyReLU` (configurable slope)
- `PReLU` (learnable parameter)
- `ELU` (exponential linear unit)

**Z-Factor Support:**
```python
# Anisotropic (default)
model = rsunet_act(width=[3,4,5,6], zfactor=[1,2,2], act='ELU')

# Isotropic
model = rsunet_act(width=[3,4,5,6], zfactor=[2,2,2], act='ReLU')
```

---

### 5. RSUNet 2D/3D Hybrid

**Files:**
- `rsunet_2d3d.py` - With Batch Norm
- `rsunet_2d3d_gn.py` - With Group Norm

Hybrid architecture using **2D convolutions at shallow layers**, **3D at deeper layers**:

```python
from emvision.models import rsunet_2d3d

# First 2 layers use 2D convolutions (1,3,3), rest use 3D
model = rsunet_2d3d(width=[16,32,64,128], depth2d=2, kernel2d=(1,3,3))
```

**Rationale:**
- Shallow layers: 2D (1,3,3) - learn in-plane features
- Deep layers: 3D (3,3,3) - learn volumetric features
- Reduces parameters while maintaining performance

---

### 6. Dynamic RSUNet

**File:** `emvision/models/dynamic_rsunet.py`

**Recurrent training** with multiple batch norm layers:

```python
from emvision.models import dynamic_rsunet

model = dynamic_rsunet(width=[16,32,64,128], unroll=3, act='ReLU')

# Inference with different unroll depths
y1 = model(x, unroll=1)
y2 = model(x, unroll=2)
y3 = model(x, unroll=3)
```

**Key Feature:**
- Each `BNAct` layer has multiple BatchNorm modules (`ModuleList`)
- Different unroll values activate different BN layers
- Enables **recurrent-style training** for iterative refinement

---

### 7. DTU2 (Down-Then-Up v2)

**File:** `emvision/models/dtu2.py`

**Valid convolutions** with explicit cropping:

```python
from emvision.models import DTU2

model = DTU2(
    width=[12, 72, 432, 2592],
    kszs=[(1,3,3), (1,3,3), (3,3,3), (3,3,3)],
    factors=[(1,3,3), (1,3,3), (3,3,3)]
)
```

**Key Differences from RSUNet:**
- **Valid convolutions** (no padding) instead of same
- **Explicit crop computation** for skip connections
- **Multi-scale kernel sizes** configurable per level
- **Flexible downsampling factors** per level

**Architecture:**
- `DownModule`: MaxPool → ConvModule
- `UpModule`: ConvTranspose → Residual sum with crop → ConvModule
- Automatic crop margin calculation based on kernel sizes

---

### 8. VRUNet (Valid Residual U-Net)

**File:** `emvision/models/vrunet.py`

U-Net with **valid convolutions** and **center cropping**:

```python
from emvision.models import vrunet

# Default: trilinear interpolation
model = vrunet(width=[16,32,64,128])

# Nearest neighbor interpolation
model = vrunet(width=[16,32,64,128], mode='nearest')
```

**Features:**
- Valid convolutions (reduces output size)
- Center cropping for skip connections
- Choice of upsampling: `trilinear` or `nearest`
- Input size: (48,148,148) → Output: (20,60,60)

---

### 9. RUNet (Residual U-Net)

**File:** `emvision/models/runet.py`

Simplified residual U-Net without the ConvBlock wrapper.

---

## Key Components & Utilities

### BilinearUp Layer

**File:** `emvision/models/layers.py`

Caffe-style learnable bilinear upsampling:

```python
from emvision.models.layers import BilinearUp

upsampler = BilinearUp(
    in_channels=64,
    out_channels=64,
    factor=(1,2,2)
)
```

**Features:**
- Fixed bilinear interpolation weights (non-learnable)
- Group convolution (each channel independent)
- Factor-dependent kernel size: `(2*f) - (f % 2)`

### Utility Functions

**File:** `emvision/models/utils.py`

Essential 3D operations:

```python
from emvision.models import utils

# Tuple operations
utils.sum3((1,2,3), (4,5,6))  # (5,7,9)
utils.mul3((1,2,3), (2,2,2))  # (2,4,6)
utils.div3((4,6,8), (2,2,2))  # (2,3,4)

# Cropping
utils.crop3d(x, margin=(1,2,2))  # Crop margins
utils.crop3d_center(x, ref)       # Center crop to match ref

# Padding
utils.pad3d_center(x, ref)        # Center pad to match ref

# Padding calculation
utils.pad_size(kernel_size=3, mode='same')   # (1,1,1)
utils.pad_size(kernel_size=3, mode='valid')  # (0,0,0)
```

---

## Model Naming Convention

| Suffix | Meaning |
|--------|---------|
| (none) | Batch Normalization + ReLU |
| `_gn` | Group Normalization |
| `_in` | Instance Normalization |
| `_act` | Configurable activation function |
| `_nn` | No normalization |
| `_2d3d` | Hybrid 2D/3D convolutions |
| `iso_` | Isotropic voxel resolution |

**Examples:**
- `rsunet_act_gn` = RSUNet + Configurable activation + Group Norm
- `rsunet_2d3d_gn` = RSUNet + 2D/3D hybrid + Group Norm
- `isoRSUNet` = RSUNet for isotropic data

---

## Common Usage Patterns

### Basic RSUNet

```python
from emvision.models import RSUNet

model = RSUNet(width=[16, 32, 64, 128, 256])
x = torch.randn(1, 16, 20, 256, 256)  # (B, C, D, H, W)
y = model(x)  # Same shape as input
```

### RSUNet with Group Norm + PReLU

```python
from emvision.models import rsunet_act_gn

model = rsunet_act_gn(
    width=[16, 32, 64, 128],
    group=8,
    act='PReLU',
    init=0.1
)
```

### 2D/3D Hybrid with Custom Depth

```python
from emvision.models import rsunet_2d3d

model = rsunet_2d3d(
    width=[16, 32, 64, 128],
    depth2d=2,          # First 2 layers use 2D
    kernel2d=(1,3,3)    # 2D kernel size
)
```

### Isotropic Data

```python
from emvision.models import rsunet_act

model = rsunet_act(
    width=[16, 32, 64, 128],
    zfactor=[2,2,2],    # Isotropic downsampling
    act='ELU'
)
```

---

## Architecture Comparison

| Model | Norm | Activation | Anisotropic | Valid Conv | Special Feature |
|-------|------|------------|-------------|------------|-----------------|
| RSUNet | BN | ReLU | ✓ | ✗ | Base model |
| isoRSUNet | BN | ReLU | ✗ | ✗ | Isotropic |
| rsunet_gn | GN | ReLU | ✓ | ✗ | Group Norm |
| rsunet_act | BN | Custom | ✓ | ✗ | Flexible activation |
| rsunet_act_gn | GN | Custom | ✓ | ✗ | GN + Custom act |
| rsunet_act_in | IN | Custom | ✓ | ✗ | Instance Norm |
| rsunet_2d3d | BN | ReLU | ✓ | ✗ | Hybrid 2D/3D |
| dynamic_rsunet | BN | Custom | ✓ | ✗ | Recurrent training |
| DTU2 | None | ReLU | ✓ | ✓ | Multi-scale kernels |
| VRUNet | BN | ReLU | ✓ | ✓ | Center crop |
| RUNet | BN | ReLU | ✓ | ✗ | Simplified |

---

## Key Design Principles

### 1. Anisotropic Convolutions

EM data typically has:
- **High XY resolution** (e.g., 4nm/pixel)
- **Lower Z resolution** (e.g., 40nm/section)

**Solution:** Use (1,2,2) downsampling and (1,3,3) kernels to respect this anisotropy.

### 2. Residual Connections

All models use **pre-activation residual blocks**:
```
x → BN → ReLU → Conv → BN → ReLU → Conv → (+) → out
↓_____________________________________________↑
```

### 3. Skip Connections

**Element-wise addition** (not concatenation) for efficiency:
```python
def forward(self, x, skip):
    return self.up(x) + skip  # Addition, not concat
```

### 4. Flexible Width

All models accept configurable channel widths:
```python
# Shallow network
model = RSUNet(width=[8, 16, 32, 64])

# Deep network
model = RSUNet(width=[16, 32, 64, 128, 256, 512])
```

### 5. Initialization

Kaiming (He) initialization for ReLU networks:
```python
def init_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
```

---

## Integration with PyTorch Connectomics

### Potential Integration Points

1. **Add to Model Registry**
   ```python
   from emvision.models import RSUNet, rsunet_act_gn, isoRSUNet

   # Register in connectomics/models/architectures/registry.py
   register_architecture('emvision_rsunet', RSUNet)
   register_architecture('emvision_rsunet_gn', rsunet_act_gn)
   register_architecture('emvision_iso_rsunet', isoRSUNet)
   ```

2. **Wrapper for Compatibility**
   ```python
   class EMVisionWrapper(ConnectomicsModel):
       def __init__(self, cfg):
           self.model = RSUNet(width=cfg.model.filters)

       def forward(self, x):
           # EMVision expects (B, C, D, H, W)
           return self.model(x)
   ```

3. **Configuration Example**
   ```yaml
   model:
     architecture: emvision_rsunet_gn
     filters: [16, 32, 64, 128, 256]
     group: 16  # For group norm
     zfactor: [1, 2, 2]  # Anisotropic
   ```

### Key Differences from PyTC Models

| Aspect | PyTorch Connectomics | EMVision |
|--------|---------------------|----------|
| **Skip Connections** | Concatenation | Addition |
| **Normalization** | MONAI default (BN) | BN/GN/IN variants |
| **Upsampling** | MONAI Upsample | BilinearUp (Caffe-style) |
| **Anisotropy** | Configurable | Built-in (1,2,2) |
| **Architecture** | MONAI-based | Custom from scratch |

---

## Testing

**File:** `test/test_models.py`

Comprehensive unit tests for all model variants:

```bash
python -m unittest test.test_models
```

**Test Coverage:**
- All RSUNet variants (BN, GN, IN)
- All activation functions (ReLU, LeakyReLU, PReLU, ELU)
- 2D/3D hybrid models
- Isotropic and anisotropic models
- Dynamic RSUNet with unrolling
- Valid convolution models (DTU2, VRUNet)

---

## Dependencies

**Minimal requirements:**
- `torch` (PyTorch)
- `numpy`
- `nose` (for testing)

**No heavy dependencies** - pure PyTorch implementation.

---

## Strengths

1. ✅ **Specialized for EM**: Anisotropic convolutions respect data characteristics
2. ✅ **Modular design**: Easy to swap components (norm, activation)
3. ✅ **Lightweight**: No heavy framework dependencies
4. ✅ **Well-tested**: Comprehensive test suite
5. ✅ **Flexible**: Multiple variants for different use cases
6. ✅ **Efficient**: Addition-based skip connections (vs concatenation)

---

## Limitations

1. ⚠️ **No multi-task support**: Single output only
2. ⚠️ **No deep supervision**: No intermediate outputs
3. ⚠️ **Limited documentation**: Minimal README
4. ⚠️ **No pre-trained models**: Train from scratch only
5. ⚠️ **Fixed architecture**: U-Net structure not customizable
6. ⚠️ **No attention mechanisms**: No self-attention or transformers

---

## Recommendations for PyTC Integration

### High Priority
1. **Integrate RSUNet variants** as alternative backbones
2. **Reuse BilinearUp** layer (proven effective)
3. **Adopt anisotropic defaults** for EM data
4. **Add Group Norm variants** (better for small batches)

### Medium Priority
5. **Integrate 2D/3D hybrid** approach for efficiency
6. **Support valid convolutions** (DTU2 style) as option
7. **Add dynamic training** (unroll mechanism)

### Low Priority
8. Keep as separate external dependency
9. Reference for architecture design patterns
10. Use for baseline comparisons

---

## Summary

**EMVision** is a focused, well-designed library of RSUNet variants optimized for 3D EM segmentation. It excels in:
- Handling anisotropic EM data
- Providing flexible normalization and activation options
- Efficient architecture with addition-based skip connections

**Best used for:** Electron microscopy segmentation with anisotropic resolution (typical EM datasets like CREMI, SNEMI3D, etc.)

**Integration value:** Moderate - provides proven architecture patterns and efficient components that could enhance PyTorch Connectomics, especially for EM-specific tasks.

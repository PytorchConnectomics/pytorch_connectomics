# Default Augmentation Analysis: Shape Misalignment Issues

## Overview

The PyTorch Connectomics codebase has aggressive default augmentations that **cause tensor shape misalignment**. This document analyzes which augmentations are problematic and provides solutions.

## Default Augmentation Configuration

Located in `connectomics/config/hydra_config.py` (lines 615-733):

### **Shape-Changing Augmentations (ENABLED BY DEFAULT)** ⚠️

#### 1. **ElasticConfig** (Line 634-640) - MOST PROBLEMATIC
```python
enabled: bool = True              # ← ENABLED!
prob: float = 0.3
sigma_range: Tuple[float, float] = (5.0, 8.0)
magnitude_range: Tuple[float, float] = (50.0, 150.0)
```
- **Issue**: Elastic deformation warps the grid, changing effective output dimensions
- **Impact**: Can shrink patches to `[56, 128, 128]` from `[64, 128, 128]`
- **Root Cause**: MONAI's `Rand3DElasticd` doesn't crop/pad to original shape

#### 2. **RotateConfig** (Line 625-630) - PROBLEMATIC
```python
enabled: bool = True              # ← ENABLED!
prob: float = 0.5
max_angle: float = 90.0
```
- **Issue**: Rotations (especially RandRotate90d at 90°) can crop edges
- **Impact**: 90° rotations may produce smaller output due to padding/cropping
- **Root Cause**: Rotation edges get clipped when not using appropriate padding

#### 3. **CutBlurConfig** (Line 696-703) - PROBLEMATIC
```python
enabled: bool = True              # ← ENABLED!
prob: float = 0.3
length_ratio: float = 0.25
down_ratio_range: Tuple[float, float] = (2.0, 8.0)
downsample_z: bool = False
```
- **Issue**: Creates blur regions with downsampling, affecting spatial dimensions
- **Impact**: Non-uniform spatial dimensions in batch

#### 4. **MisalignmentConfig** (Line 657-663) - MODERATE ISSUE
```python
enabled: bool = True              # ← ENABLED!
prob: float = 0.5
displacement: int = 16
rotate_ratio: float = 0.0
```
- **Issue**: Displacement in slices can affect padding/cropping logic
- **Impact**: May introduce small dimension mismatches

### **Safe Augmentations (Don't Change Shape)**

✅ **FlipConfig** (Line 616-621)
- Flips preserve tensor shape
- Safe to use

✅ **IntensityConfig** (Line 644-653)
- Gaussian noise, intensity shifts, contrast adjustments
- Only modify values, not shape
- Safe to use

## Why Shape Misalignment Occurs

### The Problem:
1. Each sample goes through augmentation pipeline
2. Elastic deformation warps coordinates
3. MONAI doesn't automatically pad/crop back to original size
4. Batch collation expects consistent shapes
5. **RuntimeError: stack expects each tensor to be equal size**

### Data Flow:
```
Input [64, 128, 128]
    ↓
Elastic Deformation (warping)
    ↓
Output [56, 128, 128] ← SHAPE CHANGED!
    ↓
Batch Collation (expects [64, 128, 128])
    ↓
RuntimeError! Cannot stack different sizes
```

## Solutions

### Solution 1: Disable Problematic Augmentations (IMMEDIATE FIX)

**In `tutorials/monai_fiber.yaml`**, add/modify:

```yaml
augmentation:
  enabled: true
  # Disable shape-changing augmentations
  elastic:
    enabled: false      # ← Disable elastic deformation
  rotate:
    enabled: false      # ← Disable rotation
  cut_blur:
    enabled: false      # ← Disable cut blur
  misalignment:
    enabled: false      # ← Disable misalignment
  
  # Keep safe augmentations
  flip:
    enabled: true       # ✅ Safe - preserves shape
  intensity:
    enabled: true       # ✅ Safe - only changes values
```

### Solution 2: Update Default Config (RECOMMENDED LONG-TERM)

**In `connectomics/config/hydra_config.py`**, change defaults:

```python
@dataclass
class ElasticConfig:
    """Elastic deformation augmentation."""
    
    enabled: bool = False           # ← Changed from True
    prob: float = 0.3
    sigma_range: Tuple[float, float] = (5.0, 8.0)
    magnitude_range: Tuple[float, float] = (50.0, 150.0)


@dataclass
class RotateConfig:
    """Random rotation augmentation."""
    
    enabled: bool = False           # ← Changed from True
    prob: float = 0.5
    max_angle: float = 90.0


@dataclass
class CutBlurConfig:
    """CutBlur augmentation."""
    
    enabled: bool = False           # ← Changed from True
    prob: float = 0.3
    length_ratio: float = 0.25
    down_ratio_range: Tuple[float, float] = (2.0, 8.0)
    downsample_z: bool = False


@dataclass
class MisalignmentConfig:
    """Misalignment augmentation for EM data."""
    
    enabled: bool = False           # ← Changed from True
    prob: float = 0.5
    displacement: int = 16
    rotate_ratio: float = 0.0
```

### Solution 3: Custom Collate Function (ADVANCED FIX)

Create a collate function that handles variable-sized tensors:

```python
# In connectomics/data/dataset/dataset_base.py

import torch
from torch.nn.utils.rnn import pad_sequence

def safe_collate_fn(batch):
    """Collate function that pads tensors to max size in batch."""
    if not batch:
        return batch
    
    keys = batch[0].keys() if isinstance(batch[0], dict) else range(len(batch[0]))
    result = {}
    
    for key in keys:
        tensors = [item[key] if isinstance(item, dict) else item[i] for i, item in enumerate(batch)]
        
        if not torch.is_tensor(tensors[0]):
            result[key] = tensors
            continue
        
        # Find max shape for each dimension
        max_shape = list(tensors[0].shape)
        for t in tensors[1:]:
            for i in range(len(max_shape)):
                max_shape[i] = max(max_shape[i], t.shape[i])
        
        # Pad all tensors to max shape
        padded = []
        for t in tensors:
            if t.shape == tuple(max_shape):
                padded.append(t)
            else:
                # Calculate padding needed
                pad_amounts = []
                for i in range(len(t.shape) - 1, -1, -1):
                    pad_amounts.extend([0, max_shape[i] - t.shape[i]])
                padded.append(torch.nn.functional.pad(t, pad_amounts))
        
        result[key] = torch.stack(padded) if padded else torch.tensor([])
    
    return result
```

Then use in DataLoader:
```python
DataLoader(dataset, collate_fn=safe_collate_fn)
```

## Recommendations

### Priority 1: IMMEDIATE (for monai_fiber.yaml)
- Disable elastic, rotate, cut_blur, misalignment augmentations
- Keep flip and intensity augmentations

### Priority 2: SHORT-TERM (within 1 week)
- Update default augmentation configuration in `hydra_config.py`
- Change shape-changing augmentations to disabled by default
- Users can opt-in rather than opt-out

### Priority 3: MEDIUM-TERM (within 1 month)
- Implement safe collate function for robustness
- Add tests for variable-sized tensors
- Document augmentation safety guidelines

## Testing

```bash
# Test with shape-changing augmentations disabled
python scripts/main.py --config tutorials/monai_fiber.yaml --demo

# Verify tensor shapes during training
# Add this to check shapes:
# print(f"Batch shapes: {[item['image'].shape for item in batch]}")
```

## Summary Table

| Augmentation | Enabled | Preserves Shape | Impact | Recommendation |
|---|---|---|---|---|
| Flip | ✅ True | ✅ Yes | Low | Keep enabled |
| Rotate | ⚠️ True | ❌ No | HIGH | **Disable** |
| Elastic | ⚠️ True | ❌ No | **CRITICAL** | **Disable** |
| Intensity | ✅ True | ✅ Yes | Low | Keep enabled |
| Misalignment | ⚠️ True | ❌ No | MEDIUM | **Disable** |
| CutBlur | ⚠️ True | ❌ No | HIGH | **Disable** |
| MissingParts | ⚠️ True | ⚠️ Maybe | MEDIUM | Disable for safety |
| Mixup | ⚠️ True | ⚠️ Maybe | LOW | Disable for safety |
| CopyPaste | ⚠️ True | ⚠️ Maybe | MEDIUM | Disable for safety |

# ✅ Unified RandElasticd: 2D/3D Elastic Deformation Support

## Overview

Implemented a unified `RandElasticd` wrapper that automatically switches between 2D and 3D elastic deformation based on the `do_2d` flag. This provides a consistent API for both 2D and 3D data.

## Changes Made

### 1. **New Wrapper Class: `RandElasticd` in `monai_transforms.py`**

**Location:** `connectomics/data/augment/monai_transforms.py` (lines ~1382-1487)

**Features:**
- Unified interface for both 2D and 3D elastic deformations
- Automatically selects `Rand2DElasticd` or `Rand3DElasticd` based on `do_2d` flag
- Consistent parameter names across both modes
- Proper randomization using MONAI's RandomizableTransform

**Parameters:**
```python
RandElasticd(
    keys=['image', 'label'],          # Transform keys
    do_2d=False,                      # 2D or 3D mode
    prob=0.5,                         # Probability of applying
    sigma_range=(5.0, 8.0),          # Smoothing range (3D) or spacing (2D)
    magnitude_range=(50.0, 150.0),   # Deformation magnitude
    allow_missing_keys=False,         # Handle missing keys
    mode='bilinear',                  # Interpolation mode
    padding_mode='reflection'         # Boundary handling
)
```

### 2. **Updated `build.py`**

**Changes:**
- Removed `Rand3DElasticd` from imports (line 18)
- Added `RandElasticd` to imports from custom transforms (line 47)
- Updated elastic augmentation logic (lines 580-590):
  - Removed conditional check `if not do_2d:`
  - Now always applies elastic with automatic 2D/3D selection
  - Passes `do_2d` flag to `RandElasticd`

**Before:**
```python
if should_augment("elastic", aug_cfg.elastic.enabled):
    # Skip elastic deformation for 2D data
    if not do_2d:
        transforms.append(Rand3DElasticd(...))
```

**After:**
```python
if should_augment("elastic", aug_cfg.elastic.enabled):
    # Unified elastic deformation
    transforms.append(RandElasticd(
        keys=keys,
        do_2d=do_2d,
        prob=aug_cfg.elastic.prob,
        sigma_range=aug_cfg.elastic.sigma_range,
        magnitude_range=aug_cfg.elastic.magnitude_range,
    ))
```

## How It Works

### Implementation Strategy

1. **Initialization**
   - Stores `do_2d` flag and parameters
   - Inherits from `MapTransform` and `RandomizableTransform` for MONAI compatibility

2. **Call (Transform Application)**
   - Randomizes probability independently
   - Creates appropriate transform based on `do_2d`:
     - **2D Mode**: Uses `Rand2DElasticd` with `spacing` parameter
     - **3D Mode**: Uses `Rand3DElasticd` with `sigma_range` parameter
   - Sets `prob=1.0` on selected transform (already randomized)
   - Applies and returns transformed data

### Data Flow

```
RandElasticd(data, do_2d=True/False)
  ├─ Randomize probability
  ├─ If apply:
  │   ├─ If do_2d=True:
  │   │   └─ Rand2DElasticd(spacing=sigma_range) → Apply to (C, H, W)
  │   └─ Else (do_2d=False):
  │       └─ Rand3DElasticd(sigma_range=sigma_range) → Apply to (C, D, H, W)
  └─ Return transformed data
```

## Usage Examples

### 3D Elastic Deformation (Default)
```python
from connectomics.data.augment.monai_transforms import RandElasticd

elastic_3d = RandElasticd(
    keys=['image', 'label'],
    do_2d=False,  # 3D mode
    prob=0.5,
    sigma_range=(5.0, 8.0),      # Gaussian smoothing sigma
    magnitude_range=(50.0, 150.0) # Deformation magnitude
)

# Input: (C, D, H, W) = (1, 64, 128, 128)
data_3d = {
    'image': torch.randn(1, 64, 128, 128),
    'label': torch.randint(0, 5, (1, 64, 128, 128))
}
result = elastic_3d(data_3d)
# Output: Same shape with elastic deformation applied
```

### 2D Elastic Deformation (Slice-by-Slice)
```python
elastic_2d = RandElasticd(
    keys=['image', 'label'],
    do_2d=True,   # 2D mode
    prob=0.5,
    sigma_range=(1.0, 2.0),      # Pixel spacing for 2D
    magnitude_range=(10.0, 30.0)  # Smaller for 2D
)

# Input: (C, H, W) = (1, 128, 128)
data_2d = {
    'image': torch.randn(1, 128, 128),
    'label': torch.randint(0, 5, (1, 128, 128))
}
result = elastic_2d(data_2d)
# Output: Same shape with 2D elastic deformation applied
```

### In Configuration (YAML)
```yaml
data:
  do_2d: false  # or true for 2D

  augmentation:
    preset: "some"
    elastic:
      enabled: true
      prob: 0.3
      sigma_range: [5.0, 8.0]      # 3D: smoothing; 2D: spacing
      magnitude_range: [50.0, 150.0]
```

## Key Benefits

### 1. **Unified API**
- Single class supports both 2D and 3D
- Parameters have consistent meaning across modes
- No need to write separate pipelines

### 2. **Automatic Selection**
- `do_2d` flag determines mode automatically
- No manual conditional logic needed in pipelines
- Reduces code duplication

### 3. **MONAI Compatibility**
- Inherits from `MapTransform` and `RandomizableTransform`
- Works seamlessly with `Compose` pipeline
- Follows MONAI conventions

### 4. **2D Support**
- Can now use elastic deformation for 2D data
- Previously was skipped with `if not do_2d` check
- Enables better 2D augmentation pipelines

## Technical Details

### Parameter Mapping

| Parameter | 2D Mode | 3D Mode | Notes |
|-----------|---------|---------|-------|
| `sigma_range` | `spacing` | `sigma_range` | MONAI parameter naming differs |
| `magnitude_range` | Same | Same | Consistent across modes |
| `prob` | Already applied | Already applied | Randomized upfront |
| Input shape | (C, H, W) | (C, D, H, W) | Adapted per mode |

### Probability Handling

- Randomization done once in `__call__`
- Selected transform receives `prob=1.0`
- Ensures consistent randomization behavior

## Verification

### Tests Passed
✅ 2D elastic deformation with torch.Size([1, 128, 128])
✅ 3D elastic deformation with torch.Size([1, 64, 128, 128])
✅ Probability control (0.0 = no apply, other = apply)
✅ Integration with `_build_augmentations()` for 2D
✅ Integration with `_build_augmentations()` for 3D
✅ Proper disabling when `elastic.enabled=False`
✅ No linting errors

## Backward Compatibility

✅ **Fully backward compatible** with existing 3D code:
- `do_2d=False` (default) maintains 3D behavior
- Same parameters as `Rand3DElasticd`
- 2D data now works automatically instead of being skipped

## Future Enhancements

### Possible Improvements
1. **Per-slice randomization** for 2D: Each slice can have different parameters
2. **Caching**: Cache transform configs to reduce creation overhead
3. **Batched 2D**: Apply same deformation to all slices in a batch
4. **Anisotropic support**: Different parameters for different axes

## Related Files

- `connectomics/data/augment/monai_transforms.py` - RandElasticd class
- `connectomics/data/augment/build.py` - Pipeline integration
- `connectomics/config/hydra_config.py` - ElasticConfig parameters
- `tutorials/monai_fiber.yaml` - Example configuration

## References

- [MONAI Rand2DElasticd](https://docs.monai.io/en/stable/transforms.html#rand2delasticd)
- [MONAI Rand3DElasticd](https://docs.monai.io/en/stable/transforms.html#rand3delasticd)
- [MONAI MapTransform](https://docs.monai.io/en/stable/transforms.html#maptransform)
- [MONAI RandomizableTransform](https://docs.monai.io/en/stable/transforms.html#randomizabletransform)


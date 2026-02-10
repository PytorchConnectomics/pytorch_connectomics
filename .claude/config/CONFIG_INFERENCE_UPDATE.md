# Config Files Updated for MONAI SlidingWindowInferer

## Summary

All tutorial configuration files have been updated to use MONAI's `SlidingWindowInferer` parameters instead of legacy stride-based inference configuration.

## Updated Files

1. ✅ `tutorials/monai_lucchi.yaml`
2. ✅ `tutorials/mednext_lucchi.yaml`
3. ✅ `tutorials/monai_lucchi_fast.yaml`
4. ✅ `tutorials/mednext_lucchi_auto.yaml`
5. ✅ `tutorials/mednext_custom.yaml`
6. ✅ `tutorials/default.yaml`
7. ⚠️ `tutorials/lucchi_split.yaml` - Needs actual file to update

## Changes Made

### Old Format (Legacy)
```yaml
inference:
  output_path: outputs/results/
  stride: [64, 64, 64]                 # Manual stride specification
  overlap: 0.5
  test_time_augmentation: false
```

### New Format (MONAI SlidingWindowInferer)
```yaml
inference:
  output_path: outputs/results/

  # MONAI SlidingWindowInferer parameters
  window_size: [128, 128, 128]         # Patch size for inference
  sw_batch_size: 4                     # Number of patches per forward pass
  overlap: 0.5                         # 50% overlap between patches
  blending: gaussian                   # Gaussian weighting for smooth blending
  padding_mode: constant               # Zero-padding at volume boundaries

  # Output configuration
  output_scale: 255.0                  # Scale predictions to [0, 255] for saving
  output_dtype: uint8                  # Save as uint8

  # Evaluation
  do_eval: true                        # Use eval mode for BatchNorm
  metrics: [dice, jaccard]             # Metrics to compute
  test_time_augmentation: false        # Disable TTA for faster inference
```

## Key Parameters

### MONAI SlidingWindowInferer Parameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `window_size` | Patch size for inference | Same as `data.patch_size` from training |
| `sw_batch_size` | Patches processed per forward pass | 2-8 (depends on GPU memory) |
| `overlap` | Overlap ratio between patches | 0.25-0.75 (0.5 is common) |
| `blending` | Weighting mode for patch stitching | `gaussian` or `constant` |
| `padding_mode` | How to handle volume boundaries | `constant`, `reflect`, `replicate`, `circular` |

### Output Configuration

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `output_scale` | Scale factor for predictions | 255.0 for uint8, 1.0 for float |
| `output_dtype` | Output data type | `uint8`, `uint16`, `float32` |

### Evaluation Configuration

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `do_eval` | Use eval mode (disables dropout/batchnorm) | `true` for inference |
| `metrics` | Metrics to compute | `[dice, jaccard, accuracy]` |
| `test_time_augmentation` | Enable TTA for robustness | `false` (slow), `true` (better) |
| `tta_num` | Number of TTA variants | 4, 8 (if TTA enabled) |

## Migration Guide

### For Users

If you have custom config files, update the `inference` section:

**Before:**
```yaml
inference:
  stride: [64, 64, 64]
  overlap: 0.5
```

**After:**
```yaml
inference:
  window_size: [128, 128, 128]  # Use same as data.patch_size
  sw_batch_size: 4
  overlap: 0.5
  blending: gaussian
  padding_mode: constant
```

### Config-Specific Notes

#### monai_lucchi.yaml
- **window_size**: `[112, 112, 112]` - matches training patch size
- **Architecture**: MONAI UNet with residual units
- **Use case**: Baseline configuration

#### mednext_lucchi.yaml
- **window_size**: `[128, 128, 128]` - matches training patch size
- **Architecture**: MedNeXt-S with deep supervision
- **Use case**: Production configuration

#### monai_lucchi_fast.yaml
- **window_size**: `[128, 128, 128]` - larger patches for efficiency
- **sw_batch_size**: 4 - optimized for speed
- **Use case**: Fast iteration/debugging

#### mednext_lucchi_auto.yaml
- **window_size**: `null` - auto-configured from `data.patch_size`
- **Use case**: Automatic planning mode

#### mednext_custom.yaml
- **window_size**: `[128, 128, 128]` - fully custom configuration
- **Use case**: Advanced users with specific requirements

#### default.yaml
- **window_size**: `[128, 128, 128]` - template values
- **Use case**: Starting point for new configs

## Implementation Status

### ✅ Completed
- All tutorial configs updated
- Consistent parameter naming
- Clear documentation in comments
- Default values match best practices

### 📝 Code Changes Required (Future)

The Lightning module (`connectomics/lightning/lit_model.py`) needs to be updated to:

1. **Initialize inferer** from config:
   ```python
   from monai.inferers import SlidingWindowInferer

   self.inferer = SlidingWindowInferer(
       roi_size=cfg.inference.window_size,
       sw_batch_size=cfg.inference.sw_batch_size,
       overlap=cfg.inference.overlap,
       mode=cfg.inference.blending,
       padding_mode=cfg.inference.padding_mode,
   )
   ```

2. **Update test_step** to use inferer:
   ```python
   def test_step(self, batch, batch_idx):
       inputs = batch["image"]
       logits = self.inferer(inputs=inputs, network=self)
       # ... rest of test logic
   ```

3. **Handle output scaling**:
   ```python
   def _write_outputs(self, logits, batch):
       output = logits * self.cfg.inference.output_scale
       output = output.to(dtype=getattr(torch, self.cfg.inference.output_dtype))
       # ... save to file
   ```

## Testing

To verify configs work correctly:

```bash
# Test config loading
python -c "from connectomics.config import load_config; cfg = load_config('tutorials/monai_lucchi.yaml'); print(cfg.inference)"

# Expected output:
# window_size: [112, 112, 112]
# sw_batch_size: 4
# overlap: 0.5
# blending: gaussian
# padding_mode: constant
# ...
```

## Benefits of New Format

1. **Clear Intent**: Parameters directly map to MONAI's API
2. **Type Safety**: Explicit parameters prevent configuration errors
3. **Flexibility**: Easy to adjust blending, padding, batch size independently
4. **Documentation**: Each parameter has inline comments
5. **Consistency**: All configs follow same structure
6. **Maintainability**: Changes to MONAI API are localized

## Backward Compatibility

The old `stride` parameter is **not** automatically converted. If you have old configs:

1. **Manual conversion**:
   ```yaml
   # Old
   stride: [64, 64, 64]

   # New (overlap = 1 - stride/window_size)
   window_size: [128, 128, 128]
   overlap: 0.5  # 1 - 64/128 = 0.5
   ```

2. **Or use auto-configuration**:
   ```yaml
   window_size: null  # Auto-set from data.patch_size
   ```

## Related Documentation

- **INFERENCE_DESIGN.md** - Comprehensive design comparison
- **INFERENCE_MONAI.md** - MONAI implementation guide
- **DATASET_CLEANUP_SUMMARY.md** - Dataset architecture changes
- **CLAUDE.md** - Project overview

## Next Steps

1. ✅ Update all tutorial configs (DONE)
2. ⏳ Update `lit_model.py` to use SlidingWindowInferer
3. ⏳ Add inference config validation
4. ⏳ Test on Lucchi dataset
5. ⏳ Update user documentation
6. ⏳ Add inference examples/tutorials

---

**Last Updated**: 2025-10-07
**Status**: Configs updated, implementation pending

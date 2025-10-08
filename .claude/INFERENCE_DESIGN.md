# Inference Design Comparison & Final Recommendation

## Executive Summary

Three distinct approaches exist for sliding-window inference on large EM volumes:

1. **MONAI SlidingWindowInferer** (current v2.0) - Leverages MONAI's built-in inferer
2. **PyTC v1 Custom Sampler** (legacy) - Custom grid-based sampling with explicit blending
3. **DeepEM ForwardScanner** (external reference) - dataprovider3-based streaming inference

**Recommendation:** Adopt **MONAI SlidingWindowInferer** as the primary approach with optional fallback to custom sampler for advanced use cases.

---

## Design 1: MONAI SlidingWindowInferer (Current v2.0)

### Architecture

```python
# Dataset: Returns full volumes with MONAI metadata
transforms = Compose([
    LoadImaged(keys=("image",), image_only=False),
    AddChanneld(keys=("image",)),
    EnsureTyped(keys=("image",)),
])

# LightningModule: Uses MONAI's inferer
self.inferer = SlidingWindowInferer(
    roi_size=cfg.inference.window_size,      # e.g., (128, 128, 128)
    sw_batch_size=cfg.inference.sw_batch_size,  # Patches per forward pass
    overlap=cfg.inference.overlap,           # e.g., 0.5 = 50% overlap
    mode=cfg.inference.blending,             # 'gaussian' or 'constant'
    padding_mode=cfg.inference.padding_mode,
)

# Inference
def test_step(self, batch, batch_idx):
    inputs = batch["image"].to(self.device)
    logits = self.inferer(inputs=inputs, network=self)
    self._write_outputs(logits, batch)
```

### Pros

1. **Zero Custom Code**: Delegates all grid computation, blending, and stitching to MONAI
2. **Battle-Tested**: Used by thousands of users in medical imaging community
3. **Well-Integrated**: Works seamlessly with MONAI transforms, metadata, and PyTorch Lightning
4. **Memory Efficient**: Automatic batching of patches (`sw_batch_size`) prevents OOM
5. **Flexible Blending**: Supports gaussian, constant, and custom importance maps
6. **Anisotropic Aware**: Handles non-cubic patches (e.g., `[32, 256, 256]`)
7. **Clean Separation**: Dataset loads volumes, Lightning module handles inference logic
8. **Easy Configuration**: Maps directly to Hydra config parameters
9. **Automatic Padding**: Handles edge cases at volume boundaries
10. **Maintains Metadata**: Preserves MONAI's meta dictionary for output saving

### Cons

1. **Less Control**: Cannot easily customize grid generation or blending weights
2. **Fixed API**: Limited to MONAI's interface (though highly flexible)
3. **Hidden Complexity**: Grid computation logic is abstracted (debugging harder)
4. **No Position Tracking**: Cannot access individual patch coordinates during inference
5. **Memory Overhead**: Requires loading full volume into memory (though patches are processed incrementally)
6. **Limited Custom Weighting**: Cannot implement custom per-patch weight functions beyond MONAI's built-ins

### Implementation Status (v2.0)

**Implemented:**
- `InferenceConfig` with `window_size`, `sw_batch_size`, `overlap`, `blending`, `padding_mode`
- Test dataset returns full volumes (no cropping in test mode)
- `LoadVolumed` attaches MONAI metadata for file saving
- `ConnectomicsModule.test_step()` uses `SlidingWindowInferer`
- Output saving via `_write_outputs()` using metadata

**Configuration:**
```yaml
inference:
  window_size: [128, 128, 128]   # Patch size
  sw_batch_size: 4               # Patches per GPU
  overlap: 0.5                   # 50% overlap
  blending: gaussian             # Gaussian weighting
  padding_mode: constant         # Zero-padding at borders
  output_scale: 255.0            # Scale predictions to [0, 255]
```

---

## Design 2: PyTC v1 Custom Sampler (Legacy)

### Architecture

```python
class VolumeDataset:
    def __init__(self, sample_stride=(64, 64, 64), ...):
        # Precompute grid
        self.sample_size = [
            count_volume(vol_size, sample_volume_size, sample_stride)
            for vol_size in self.volume_sizes
        ]
        self.sample_num = np.prod(self.sample_size)

    def __getitem__(self, index):
        if self.mode == 'test':
            pos = self._get_pos_test(index)  # [dataset_id, z, y, x]
            volume = crop_volume(self.volume[pos[0]], vol_size, pos[1:])
            return pos, volume

    def _get_pos_test(self, index):
        # Convert linear index to (dataset_id, z, y, x)
        did = self._index_to_dataset(index)
        index2 = index - self.sample_num_c[did]
        pos = self._index_to_location(index2, self.sample_size_test[did])
        # Handle boundary: tuck-in last patches
        ...
        return pos

# Inference loop (external to dataset)
predictions = np.zeros(volume_shape)
weights = np.zeros(volume_shape)

for pos, patch in dataloader:
    output = model(patch)
    # Manual blending
    z, y, x = pos[1:]
    predictions[z:z+patch_size[0], ...] += output * gaussian_weight
    weights[z:z+patch_size[0], ...] += gaussian_weight

predictions /= weights
```

### Pros

1. **Explicit Control**: Full visibility into grid generation and patch coordinates
2. **Custom Blending**: Can implement arbitrary weighting schemes (bump functions, learned weights)
3. **Position Tracking**: Dataset returns `(pos, patch)` for downstream analysis
4. **Flexible Grid**: Can customize stride, overlap, boundary handling independently
5. **Proven Legacy**: Battle-tested in PyTC v1 for years of production use
6. **Debugging Friendly**: Can inspect positions, validate coverage, visualize patches
7. **Multi-Volume Support**: Built-in handling of multiple volumes with different sizes
8. **Tuck-In Boundary**: Smart handling of edge patches to avoid padding artifacts
9. **Valid Mask Support**: Can restrict sampling to specific regions
10. **Rejection Sampling**: Can filter patches based on foreground content (though not typically used in test mode)

### Cons

1. **Boilerplate Code**: Requires manual implementation of grid, blending, normalization
2. **Error-Prone**: Easy to introduce bugs in position calculation or weight normalization
3. **Not MONAI-Native**: Doesn't integrate with MONAI transforms/metadata
4. **Memory Management**: Must manually allocate output arrays (OOM risk)
5. **Less Tested**: Custom code has fewer users than MONAI's implementation
6. **Duplicate Logic**: Reimplements functionality already in MONAI
7. **Lightning Integration**: Requires custom collation and output handling
8. **No Automatic Batching**: Must manually group patches for GPU efficiency
9. **Harder to Maintain**: More code surface area for bugs and regressions

### Implementation Status (v1)

**Key Functions:**
- `_get_pos_test(index)`: Converts linear index to 4D position `[did, z, y, x]`
- `_index_to_dataset(index)`: Maps global index to dataset ID
- `_index_to_location(index, sz)`: Converts 1D index to 3D coordinates
- `count_volume(vol_size, sample_size, stride)`: Computes grid dimensions

**Grid Calculation:**
```python
# Number of patches per dimension
def count_volume(volume_size, sample_size, stride):
    return np.ceil((volume_size - sample_size) / stride).astype(int) + 1

# Example: volume=[1024, 1024, 100], patch=[128, 128, 32], stride=[64, 64, 16]
# Grid size: [(1024-128)/64 + 1, ...] = [15, 15, 5] = 1125 patches
```

---

## Design 3: DeepEM ForwardScanner (External Reference)

### Architecture

```python
from dataprovider3 import Dataset, ForwardScanner

# Setup
dataset = Dataset(spec={'input': (1, fov_z, fov_y, fov_x)})
dataset.add_data('input', img_volume)

scanner = ForwardScanner(
    dataset,
    scan_spec={'output': (channels, out_z, out_y, out_x)},
    stride=(stride_z, stride_y, stride_x),
    blend='bump'  # or 'precomputed'
)

# Inference loop
while inputs := scanner.pull():
    outputs = model(inputs)
    scanner.push(outputs)

# Final output: automatically blended
final_output = scanner.outputs.get_data('output')
```

### Pros

1. **Streaming Interface**: Pull-push API for incremental processing
2. **Automatic Blending**: Built-in bump/gaussian blending via dataprovider3
3. **Memory Efficient**: Processes patches on-the-fly without storing all outputs
4. **Multi-Output Support**: Handles multiple output types (affinity, boundary, etc.)
5. **Test-Time Augmentation**: Built-in support for flipping/rotating patches
6. **Precomputed Weights**: Option to use externally computed blend weights
7. **Cloud-Volume Integration**: Direct support for reading/writing to cloud storage
8. **Benchmark Mode**: Dummy data generation for profiling
9. **Variance Computation**: Can compute prediction variance across augmentations
10. **Production-Proven**: Used in large-scale connectomics projects (FlyEM, etc.)

### Cons

1. **External Dependency**: Requires dataprovider3 (not maintained actively)
2. **Not MONAI-Compatible**: Different data format and API from MONAI
3. **Complex Setup**: Requires understanding of spec dictionaries and scanning parameters
4. **No PyTorch Lightning**: Would need custom integration layer
5. **Limited Documentation**: dataprovider3 is less documented than MONAI
6. **Legacy Codebase**: Uses older Python patterns (imp.load_source, SimpleNamespace)
7. **Harder Migration**: Significant refactoring needed to integrate with v2.0
8. **Non-Standard**: Deviates from MONAI/Lightning architecture philosophy
9. **Maintenance Burden**: Would need to maintain compatibility with dataprovider3
10. **Community Support**: Smaller user base compared to MONAI

### Implementation Status

**Not implemented in PyTC v2.0.** External reference only.

---

## Feature Comparison Matrix

| Feature | MONAI Inferer | PyTC v1 Sampler | DeepEM Scanner |
|---------|--------------|-----------------|----------------|
| **Ease of Use** | ★★★★★ | ★★★☆☆ | ★★☆☆☆ |
| **Code Simplicity** | ★★★★★ | ★★☆☆☆ | ★★★☆☆ |
| **Flexibility** | ★★★☆☆ | ★★★★★ | ★★★★☆ |
| **Memory Efficiency** | ★★★★☆ | ★★★☆☆ | ★★★★★ |
| **MONAI Integration** | ★★★★★ | ★☆☆☆☆ | ☆☆☆☆☆ |
| **Position Tracking** | ★☆☆☆☆ | ★★★★★ | ★★★☆☆ |
| **Custom Blending** | ★★★☆☆ | ★★★★★ | ★★★★☆ |
| **Debugging** | ★★★☆☆ | ★★★★★ | ★★★☆☆ |
| **Community Support** | ★★★★★ | ★★★☆☆ | ★★☆☆☆ |
| **Maintenance** | ★★★★★ | ★★★☆☆ | ★★☆☆☆ |
| **Production Ready** | ★★★★★ | ★★★★☆ | ★★★★★ |

---

## Use Case Recommendations

### Use MONAI SlidingWindowInferer When:

- Standard inference workflow (90% of cases)
- Using MONAI transforms and metadata
- Want minimal code and maximum reliability
- Gaussian or constant blending is sufficient
- Working with PyTorch Lightning
- Need quick prototyping
- Standard overlapping patches are acceptable

### Use PyTC v1 Custom Sampler When:

- Need explicit patch coordinates for analysis
- Implementing custom blending functions
- Debugging coverage or overlap issues
- Working with very large volumes (>100GB) where grid precomputation helps
- Need fine-grained control over boundary handling
- Migrating from PyTC v1 code
- Research requiring non-standard sampling patterns

### Use DeepEM ForwardScanner When:

- Working with cloud-stored volumes
- Need test-time augmentation with variance
- Streaming inference on volumes too large for memory
- Integrating with existing DeepEM pipelines
- NOT recommended for new PyTC v2.0 development

---

## Final Recommendation: Hybrid Approach

### Primary: MONAI SlidingWindowInferer (Default)

**Rationale:**
1. Aligns with PyTC v2.0 architecture (MONAI + Lightning)
2. Minimal code, maximum reliability
3. Sufficient for 90% of use cases
4. Easy to configure and maintain
5. Excellent community support

**Implementation:**
- Keep current v2.0 implementation as default
- Document configuration options clearly
- Provide examples for common scenarios

### Secondary: Custom Sampler Option (Advanced)

**Rationale:**
1. Some users need explicit position tracking
2. Research may require custom blending
3. Legacy code compatibility

**Implementation:**
- Create optional `InferenceDataset` class with explicit position tracking
- Use when `cfg.inference.mode = 'custom'`
- Provide utility functions for grid computation and blending
- Document clearly when to use vs. MONAI inferer

### Code Structure

```python
# connectomics/inference/
├── __init__.py
├── sliding_window.py      # MONAI SlidingWindowInferer wrapper (default)
├── custom_sampler.py      # Optional custom grid sampler
├── blending.py            # Shared blending utilities
└── utils.py               # Position calculation, grid generation

# Usage in LightningModule
if self.cfg.inference.mode == 'monai':
    # Use MONAI SlidingWindowInferer (default)
    self.inferer = create_monai_inferer(self.cfg)
elif self.cfg.inference.mode == 'custom':
    # Use custom sampler with explicit positions
    self.inferer = create_custom_inferer(self.cfg)
```

---

## Migration Path

### Short-Term (v2.0.1)

1. **Keep MONAI inferer as default** - Already implemented
2. **Improve documentation** - Add inference tutorial
3. **Add examples** - Show common configurations
4. **Test edge cases** - Boundary handling, anisotropic patches

### Medium-Term (v2.1)

1. **Add custom sampler option** - For advanced users
2. **Create blending utilities** - Shared between MONAI and custom
3. **Add position tracking mode** - Optional coordinate logging
4. **Benchmark performance** - Compare MONAI vs custom on real data

### Long-Term (v2.2+)

1. **Evaluate streaming inference** - For very large volumes
2. **Consider learned blending** - Neural blending weights
3. **Add cloud-volume support** - Optional integration
4. **Multi-scale inference** - Hierarchical sliding window

---

## Configuration Design

### Unified Inference Config

```yaml
inference:
  # Mode selection
  mode: monai  # 'monai' (default), 'custom', 'streaming'

  # Common parameters
  window_size: [128, 128, 128]
  overlap: 0.5  # Fraction or absolute pixels
  blending: gaussian  # 'gaussian', 'constant', 'bump', 'custom'

  # MONAI-specific
  sw_batch_size: 4
  padding_mode: constant

  # Custom sampler specific
  stride: null  # Auto-computed from overlap if null
  return_positions: false
  boundary_mode: tuck_in  # 'tuck_in', 'crop', 'pad'

  # Output
  output_scale: 255.0
  output_dtype: uint8
  save_intermediate: false
```

---

## Testing Strategy

### Unit Tests

1. **Grid Coverage** - Verify all voxels covered exactly once (no overlap) or with correct weights
2. **Boundary Handling** - Edge cases at volume boundaries
3. **Anisotropic Patches** - Non-cubic window sizes
4. **Weight Normalization** - Blending weights sum to 1.0
5. **Multi-Volume** - Multiple volumes with different sizes

### Integration Tests

1. **Lucchi Dataset** - Compare MONAI vs custom on real data
2. **Memory Profiling** - Track GPU/CPU memory usage
3. **Performance Benchmark** - Throughput (voxels/second)
4. **Numerical Parity** - MONAI vs PyTC v1 should match (within epsilon)

### Regression Tests

1. **Output Consistency** - Same input → same output across versions
2. **Metadata Preservation** - File paths, spacing, orientation
3. **Edge Case Handling** - Small volumes, single patches, etc.

---

## Key Decisions

### ✅ Adopted

1. **MONAI SlidingWindowInferer as default** - Best balance of simplicity and capability
2. **Full volume loading in test mode** - Required for MONAI inferer
3. **Metadata-based output saving** - Leverages MONAI's meta dictionary
4. **Hydra config for inference** - Type-safe, composable configuration

### ⚠️ Deferred

1. **Custom sampler option** - Wait for user demand
2. **Streaming inference** - Evaluate if memory becomes bottleneck
3. **DeepEM integration** - Not aligned with MONAI/Lightning architecture

### ❌ Rejected

1. **Reimplementing MONAI's sliding window** - Unnecessary duplication
2. **dataprovider3 dependency** - External, unmaintained library
3. **Multiple blending backends** - MONAI's is sufficient

---

## Conclusion

The **MONAI SlidingWindowInferer** approach (Design 1) provides the best foundation for PyTC v2.0 inference:

- **80/20 Rule**: Covers 90% of use cases with 10% of the code
- **Architecture Alignment**: Fits naturally with MONAI + Lightning design
- **Maintainability**: Delegates complexity to well-tested external library
- **Extensibility**: Easy to add custom sampler later if needed

The current v2.0 implementation is **production-ready** and should be the recommended approach. Custom sampler can be added in v2.1 if specific use cases emerge that require explicit position tracking or custom blending.

### Next Steps

1. **Document inference workflow** - Tutorial showing test mode usage
2. **Add configuration examples** - Common overlap/blending settings
3. **Test on large volumes** - Validate memory efficiency
4. **Benchmark performance** - Compare to PyTC v1 baseline
5. **Monitor user feedback** - Identify if custom sampler is needed

---

## References

- **MONAI Docs**: [SlidingWindowInferer](https://docs.monai.io/en/stable/inferers.html#slidingwindowinferer)
- **PyTC v1**: `pytorch_connectomics_v1/connectomics/data/dataset/dataset_volume.py`
- **DeepEM**: `/projects/weilab/weidf/lib/seg/DeepEM/deepem/test/`
- **Current Implementation**: `.claude/INFERENCE_MONAI.md`

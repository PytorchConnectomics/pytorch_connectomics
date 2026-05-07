# Inference Design

## Primary Approach: MONAI SlidingWindowInferer

PyTC v2.0 uses MONAI's `SlidingWindowInferer` as the default inference method. This delegates all grid computation, blending, and stitching to MONAI.

### Configuration

```yaml
inference:
  window_size: [128, 128, 128]   # Patch size
  sw_batch_size: 4               # Patches per GPU forward pass
  overlap: 0.5                   # 50% overlap between patches
  blending: gaussian             # 'gaussian' or 'constant'
  padding_mode: constant         # Padding at borders
  output_scale: 255.0            # Scale predictions to [0, 255]
```

### How It Works

1. **Dataset**: Returns full volumes with MONAI metadata (no cropping in test mode)
2. **Lightning Module**: Uses `SlidingWindowInferer` in `test_step()`
3. **Output**: Saves via MONAI meta dictionary for file naming

```python
# In ConnectomicsModule.test_step()
inputs = batch["image"].to(self.device)
logits = self.inferer(inputs=inputs, network=self)
self._write_outputs(logits, batch)
```

### Pros
- Zero custom code -- delegates to battle-tested MONAI
- Automatic batching (`sw_batch_size`) prevents OOM
- Supports gaussian, constant blending and anisotropic patches
- Clean integration with MONAI transforms and Lightning

### When to Consider Custom Sampling
- Need explicit patch coordinates for analysis
- Implementing custom blending functions
- Very large volumes (>100GB) where grid precomputation helps
- Non-standard sampling patterns

## Test-Time Augmentation (TTA)

Configured in `inference.test_time_augmentation`:
- `flip_axes`: axes to flip (e.g., `[[2], [3]]` for XY-only flips)
- `channel_activations`: per-channel activation functions

For anisotropic EM data, use XY-only flips (Z is different).

## Key Files
- `connectomics/inference/sliding.py` -- SlidingWindowInferer wrapper
- `connectomics/inference/tta.py` -- Test-time augmentation
- `connectomics/inference/output.py` -- Output saving + postprocessing
- `connectomics/inference/manager.py` -- Inference orchestration

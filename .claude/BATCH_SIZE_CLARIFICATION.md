# Batch Size Clarification: `batch_size` vs `sw_batch_size`

## TL;DR

**They are NOT the same and NOT redundant:**

- **`batch_size`**: Number of **volumes** loaded from dataloader
- **`sw_batch_size`**: Number of **patches** processed per GPU forward pass

## Detailed Explanation

### Training Mode

```yaml
data:
  batch_size: 8  # Load 8 random patches at once
```

During training, each batch contains **random patches** from the dataset:
- DataLoader yields 8 patches of size [112, 112, 112]
- Model processes all 8 patches in a single forward pass
- Gradient computed over all 8 patches

```
Batch = [patch1, patch2, patch3, patch4, patch5, patch6, patch7, patch8]
        ↓
      Model
        ↓
    Loss & Backprop
```

### Inference Mode (Sliding Window)

```yaml
inference:
  batch_size: 1       # Load 1 full volume at a time
  sw_batch_size: 4    # Process 4 patches simultaneously
```

During inference, the workflow is different:

1. **DataLoader** loads `batch_size` full volumes (typically 1 for large test volumes)
2. **SlidingWindowInferer** extracts many overlapping patches from that volume
3. **GPU** processes `sw_batch_size` patches at a time

```
Volume (e.g., [1, 165, 1024, 768])
        ↓
SlidingWindowInferer extracts patches:
[patch1, patch2, patch3, ..., patch_N]  (N = hundreds/thousands)
        ↓
Process in mini-batches of sw_batch_size:
[patch1, patch2, patch3, patch4] → Model → [output1, output2, output3, output4]
[patch5, patch6, patch7, patch8] → Model → [output5, output6, output7, output8]
...
        ↓
Blend outputs back into full volume prediction
```

## Configuration Examples

### Example 1: Single Large Volume

```yaml
data:
  batch_size: 32      # Training: 32 random patches

inference:
  batch_size: 1       # Load 1 volume at a time (volume too large for multiple)
  sw_batch_size: 8    # Process 8 patches simultaneously on GPU
```

**Inference workflow:**
- Load 1 volume: `[1, 165, 1024, 768]`
- Extract ~1000 patches of size `[112, 112, 112]`
- Process in batches of 8: `1000 / 8 = 125 forward passes`
- Stitch results back together

### Example 2: Multiple Small Volumes

```yaml
data:
  batch_size: 16      # Training: 16 random patches

inference:
  batch_size: 2       # Load 2 volumes at once (small enough to fit in memory)
  sw_batch_size: 4    # Process 4 patches per volume simultaneously
```

**Inference workflow:**
- Load 2 volumes: `[vol1, vol2]`
- For vol1: Extract ~200 patches → process in batches of 4
- For vol2: Extract ~200 patches → process in batches of 4
- Stitch both volumes separately

## Memory Considerations

### GPU Memory Usage

```
Training:
  GPU_memory = batch_size × patch_size × model_params

Inference:
  GPU_memory = sw_batch_size × patch_size × model_params
```

### Why `sw_batch_size < training batch_size`?

During inference:
- Full volume must stay in CPU/GPU memory
- Intermediate activations for blending
- Multiple forward passes accumulate

Typical values:
- **Training**: `batch_size = 8-32`
- **Inference**: `sw_batch_size = 2-8`

## When to Adjust

### Increase `sw_batch_size` if:
- ✅ You have available GPU memory
- ✅ Want faster inference (fewer forward passes)
- ✅ Small patch size (e.g., 64³)

### Decrease `sw_batch_size` if:
- ⚠️ Getting OOM (Out of Memory) errors
- ⚠️ Large patch size (e.g., 256³)
- ⚠️ High model complexity

### Adjust `batch_size` (inference) if:
- `batch_size = 1`: For large test volumes (typical)
- `batch_size > 1`: For many small test volumes

## Config Template

```yaml
# Training
data:
  batch_size: 16                       # Random patches per training step

# Inference
inference:
  # MONAI SlidingWindowInferer
  window_size: [112, 112, 112]         # Patch size
  sw_batch_size: 4                     # Patches per forward pass (DIFFERENT from batch_size!)

  # DataLoader settings
  batch_size: 1                        # Volumes per dataloader iteration

  # IMPORTANT: These are independent!
  #   batch_size: How many VOLUMES to load
  #   sw_batch_size: How many PATCHES to process at once
```

## Common Mistakes

### ❌ Wrong: Setting them equal

```yaml
inference:
  batch_size: 8
  sw_batch_size: 8  # Assumes batch_size controls patches - WRONG!
```

This would try to load 8 full volumes AND process 8 patches at once - likely OOM!

### ✅ Correct: Independent values

```yaml
inference:
  batch_size: 1        # Load 1 volume at a time
  sw_batch_size: 8     # Process 8 patches from that volume
```

## Advanced: Test-Time Augmentation (TTA)

With TTA, `sw_batch_size` affects each augmentation variant:

```yaml
inference:
  batch_size: 1
  sw_batch_size: 4
  test_time_augmentation: true
  tta_num: 8  # 8 augmentation variants
```

**Workflow:**
- Load 1 volume
- Generate 8 augmented versions
- For each version: Extract patches, process in batches of 4
- Average predictions across all 8 versions
- Stitch final result

Total forward passes = `(num_patches / sw_batch_size) × tta_num`

## Summary Table

| Parameter | Level | Controls | Typical Training | Typical Inference |
|-----------|-------|----------|-----------------|-------------------|
| `batch_size` | Data | Volumes from dataloader | 16-32 (patches) | 1 (volumes) |
| `sw_batch_size` | Inference | Patches per forward pass | N/A | 2-8 |

## Related Files

- **Config**: `connectomics/config/hydra_config.py`
- **Implementation**: `connectomics/lightning/lit_model.py`
- **Docs**: `.claude/INFERENCE_DESIGN.md`

---

**Key Takeaway**: `sw_batch_size` is a sliding-window-specific parameter for controlling GPU memory during patch extraction. It is completely independent from `batch_size`.

# Training Speed Optimization Guide

Comprehensive guide to speed up training for PyTorch Connectomics.

## Current Config Analysis

Your current setup:
- ✅ `use_preloaded_cache: true` - Already optimal (loads volumes into RAM)
- ✅ `benchmark: true` - cuDNN auto-tuning enabled
- ❌ `precision: "32"` - Using FP32 (slow, but needed for stability)
- ❌ `num_workers: 1` - Only 1 data loader worker
- ❌ `batch_size: 4` - Could be larger
- ❌ No gradient checkpointing
- ❌ No mixed precision (disabled for stability)

## Speed Optimization Strategies (Ranked by Impact)

### 🚀 **Level 1: Quick Wins (No Training Changes)**

#### 1. **Enable Mixed Precision** ⭐⭐⭐ (2-3x speedup)
```yaml
training:
  precision: "bf16-mixed"  # BF16 is more stable than FP16
```

**Why**: FP16/BF16 operations are 2-3x faster on modern GPUs (A100, V100, RTX 30xx+)
**Trade-off**: BF16 has same range as FP32, less likely to overflow than FP16
**When stable**: Switch from `"32"` → `"bf16-mixed"` or `"16-mixed"`

#### 2. **Increase Data Workers** ⭐⭐⭐ (1.5-2x speedup)
```yaml
data:
  num_workers: 4  # Increase from 1 (use 2-4 per GPU)
  pin_memory: true  # Already enabled
  persistent_workers: true  # Already enabled
```

**Why**: Parallel data loading prevents GPU starvation
**Optimal**: 2-4 workers per GPU (test to find best)
**Note**: You already use preloaded cache, so workers help with augmentation/batching

#### 3. **Increase Batch Size** ⭐⭐ (1.3-1.5x speedup)
```yaml
data:
  batch_size: 8  # Double from 4 (if GPU memory allows)
  # Or use gradient accumulation if OOM:
training:
  accumulate_grad_batches: 2  # Effective batch = 4 * 2 = 8
```

**Why**: Larger batches improve GPU utilization (amortize overhead)
**Check GPU memory**: `nvidia-smi` - aim for 80-90% usage
**Alternative**: Gradient accumulation if memory limited

#### 4. **Reduce Patch Size** ⭐⭐ (1.5-2x speedup)
```yaml
data:
  patch_size: [96, 96, 96]  # Reduce from [112, 112, 112]
  # Or even [80, 80, 80] for 3D
```

**Why**: Smaller patches = less computation, more patches per batch
**Trade-off**: Less spatial context (may hurt accuracy slightly)
**Sweet spot**: 64-96 for 3D, 128-256 for 2D

#### 5. **Optimize Logging** ⭐ (minor speedup)
```yaml
training:
  log_every_n_steps: 50  # Increase from 10 (less frequent logging)

visualization:
  enabled: false  # Disable during fast training runs
```

**Why**: Logging has overhead (especially visualization)
**When to enable**: Only for monitoring/debugging runs

---

### 🔥 **Level 2: Model Optimizations (Moderate Changes)**

#### 6. **Reduce Model Complexity** ⭐⭐⭐ (2-4x speedup)
```yaml
model:
  # Option A: Fewer filters
  filters: [16, 32, 64, 128, 256]  # Half channels (was [32, 64, 128, 256, 512])

  # Option B: Shallower network
  filters: [32, 64, 128, 256]  # Remove deepest layer (4 levels instead of 5)

  # Option C: Smaller model
  architecture: monai_basic_unet3d  # Simpler than monai_unet (no residual units)
```

**Why**: Fewer parameters = faster forward/backward pass
**Trade-off**: May reduce accuracy (test on validation set)
**Recommendation**: Start with half channels, check if accuracy acceptable

#### 7. **Enable Gradient Checkpointing** ⭐⭐ (saves memory → allows larger batch)
```yaml
model:
  use_checkpoint: true  # Trade compute for memory (if supported by model)
```

**Why**: Recomputes activations during backward instead of storing
**Trade-off**: 20-30% slower, but saves 30-50% memory → allows larger batches
**Net effect**: Often faster overall due to larger batch size

#### 8. **Use Efficient Architecture** ⭐⭐⭐ (2-3x speedup)
```yaml
model:
  # Option A: MedNeXt-S (faster than UNet)
  architecture: mednext
  mednext_size: S  # Small (5.6M params)
  deep_supervision: false  # Disable for speed

  # Option B: BasicUNet (simplest)
  architecture: monai_basic_unet3d
```

**Why**: MedNeXt and BasicUNet are more efficient than ResUNet
**When**: If accuracy is acceptable with simpler model

---

### ⚡ **Level 3: Advanced Optimizations (Significant Changes)**

#### 9. **Compile Model (PyTorch 2.0+)** ⭐⭐⭐ (1.5-2x speedup)
```python
# In lit_model.py or main.py
import torch
if hasattr(torch, 'compile'):
    model = torch.compile(model, mode='reduce-overhead')
```

**Why**: PyTorch 2.0 JIT compilation optimizes compute graphs
**Requirements**: PyTorch >= 2.0, CUDA >= 11.7
**Modes**: `'reduce-overhead'` (fastest), `'default'`, `'max-autotune'`

#### 10. **Reduce Iterations per Epoch** ⭐⭐ (proportional speedup)
```yaml
data:
  iter_num: 250  # Half from 500 → 2x fewer batches per epoch
  max_epochs: 400  # Double epochs to maintain total iterations
```

**Why**: Faster epoch completion, more frequent validation/checkpointing
**Trade-off**: More frequent callbacks (visualization, checkpointing)
**Note**: Total training iterations stays the same

#### 11. **Disable Deep Supervision** ⭐ (10-20% speedup)
```yaml
model:
  deep_supervision: false  # If using MedNeXt
```

**Why**: Deep supervision computes multi-scale losses (expensive)
**Trade-off**: May reduce accuracy slightly
**When**: If not using deep supervision models

#### 12. **Optimize Data Augmentation** ⭐⭐ (if enabled)
```yaml
augmentation:
  enabled: false  # Currently disabled ✓

  # If enabling, use only fast augmentations:
  flip: {enabled: true, prob: 0.5}  # Fast
  rotate: {enabled: false}  # Slow (3D rotation)
  elastic: {enabled: false}  # Very slow
```

**Why**: Elastic deformation and rotation are expensive in 3D
**Recommendation**: Keep augmentation disabled for speed, enable only flip/noise

---

### 💡 **Level 4: System-Level Optimizations**

#### 13. **Use Multiple GPUs** ⭐⭐⭐ (Nx speedup with N GPUs)
```yaml
system:
  num_gpus: 4  # Distributed training (if available)

data:
  batch_size: 4  # Per GPU (total = 4 * 4 = 16)
```

**Why**: Linear speedup with more GPUs (DDP)
**Note**: PyTorch Lightning handles DDP automatically
**Scaling**: 2 GPUs ≈ 1.8x, 4 GPUs ≈ 3.5x (not perfect due to communication)

#### 14. **Use Faster Storage** ⭐⭐ (1.2-1.5x speedup)
- **RAM disk**: Copy data to `/dev/shm` (if small dataset)
- **NVMe SSD**: Much faster than HDD
- **Network**: Use local storage, not network mount

**Why**: I/O can be bottleneck even with caching
**Your setup**: `use_preloaded_cache: true` already loads to RAM ✓

#### 15. **CPU Optimization** ⭐ (minor speedup)
```yaml
system:
  num_cpus: 8  # Increase from 1 (for data workers)
```

**Why**: More CPUs help with parallel data loading
**Note**: Only helps if `num_workers > 1`

---

## 🎯 **Recommended Quick Optimizations for Your Config**

Apply these changes for **immediate 2-3x speedup** once training is stable:

```yaml
# 1. Mixed precision (2-3x faster)
training:
  precision: "bf16-mixed"  # or "16-mixed" if no BF16 support

# 2. More data workers (1.5-2x faster)
data:
  num_workers: 4
  batch_size: 8  # If GPU memory allows

# 3. Smaller model (2x faster, test accuracy)
model:
  filters: [16, 32, 64, 128, 256]  # Half channels

# 4. Less frequent logging
training:
  log_every_n_steps: 50

visualization:
  enabled: false  # Enable only when needed
```

**Expected speedup**: 4-6x faster training overall

---

## 📊 **Measuring Speedup**

### Before Optimization:
```bash
# Measure baseline
time python scripts/main.py --config tutorials/monai_lucchi.yaml --fast-dev-run
```

### After Each Change:
```bash
# Test speed improvement
time python scripts/main.py --config tutorials/monai_lucchi.yaml --fast-dev-run
```

### Profile Data Loading:
```python
# Check if GPU is waiting for data
nvidia-smi dmon -s u -d 1  # GPU utilization (should be >80%)
```

If GPU utilization < 80%, increase `num_workers` or `batch_size`.

---

## ⚠️ **Important Trade-offs**

| Optimization | Speedup | Accuracy Impact | Memory Impact |
|--------------|---------|-----------------|---------------|
| Mixed precision (BF16) | 2-3x | Minimal | -30% memory |
| Smaller model | 2-4x | May reduce | -50% memory |
| Larger batch | 1.3-1.5x | May improve | +memory |
| Fewer workers | Slower | None | -memory |
| Less augmentation | Faster | May reduce | None |
| Gradient checkpointing | -20-30% | None | -40% memory |

---

## 🔍 **Debugging Slow Training**

### 1. Check GPU Utilization
```bash
nvidia-smi dmon -s u -d 1
# Target: >80% utilization
# <50% = CPU bottleneck (increase workers/batch)
# 100% = GPU bottleneck (good!)
```

### 2. Profile Training
```bash
# Use PyTorch profiler
python scripts/main.py --config tutorials/monai_lucchi.yaml --fast-dev-run
# Check for slowest operations in TensorBoard profiler
```

### 3. Check Data Loading
```bash
# Profile data loader
python scripts/profile_dataloader.py tutorials/monai_lucchi.yaml
```

---

## 📈 **Progressive Optimization Strategy**

**Phase 1: Stability (Current)**
- ✓ FP32 precision (stable but slow)
- ✓ Conservative settings
- Goal: Get NaN-free training

**Phase 2: Speed (After Stable)**
1. Enable BF16 mixed precision
2. Increase batch size to 8
3. Add 4 data workers
4. Expected: 3-4x speedup

**Phase 3: Optimization (Fine-tune)**
1. Try smaller model (half channels)
2. Reduce patch size to 96³
3. Enable gradient checkpointing if OOM
4. Expected: Additional 2x speedup

**Phase 4: Production (Best Performance)**
1. Multi-GPU training (if available)
2. PyTorch 2.0 compile
3. Optimize hyperparameters
4. Expected: 8-10x total speedup from baseline

---

## 🚀 **Quick Start: Fastest Config (After Stability)**

Once training is stable (no NaN), use this config for **maximum speed**:

```yaml
system:
  num_gpus: 1
  num_cpus: 8

model:
  architecture: monai_basic_unet3d  # Simpler model
  filters: [16, 32, 64, 128, 256]   # Half channels

data:
  patch_size: [96, 96, 96]  # Smaller patches
  batch_size: 8             # Larger batch
  num_workers: 4            # Parallel loading

training:
  precision: "bf16-mixed"   # Mixed precision
  log_every_n_steps: 50

visualization:
  enabled: false            # Disable for speed

augmentation:
  enabled: false            # Already disabled
```

**Expected**: 5-8x faster than current config while maintaining reasonable accuracy.

---

## 📚 **References**

- [PyTorch Performance Tuning](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [Lightning Performance Tips](https://lightning.ai/docs/pytorch/stable/advanced/training_tricks.html)
- [Mixed Precision Training](https://pytorch.org/docs/stable/amp.html)
- [PyTorch 2.0 Compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)

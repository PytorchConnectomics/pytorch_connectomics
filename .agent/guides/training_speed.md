# Training Speed Optimization Guide

## Quick Wins (No Training Changes)

### 1. Mixed Precision (2-3x speedup)
```yaml
training:
  precision: "bf16-mixed"  # BF16 is more stable than FP16
```

### 2. Increase Data Workers (1.5-2x speedup)
```yaml
data:
  num_workers: 4  # 2-4 per GPU
  pin_memory: true
  persistent_workers: true
```

### 3. Increase Batch Size (1.3-1.5x speedup)
```yaml
data:
  batch_size: 8  # Or use gradient accumulation if OOM
```

### 4. Reduce Patch Size (1.5-2x speedup)
```yaml
data:
  patch_size: [96, 96, 96]  # Sweet spot: 64-96 for 3D
```

## Model Optimizations

### 5. Fewer Filters (2-4x speedup)
```yaml
model:
  filters: [16, 32, 64, 128, 256]  # Half channels
```

### 6. Efficient Architecture
```yaml
model:
  architecture: mednext
  mednext_size: S  # 5.6M params, fast
```

### 7. Gradient Checkpointing (saves memory -> allows larger batch)
Trades 20-30% compute for 30-50% memory savings.

## Advanced

### 8. PyTorch 2.0 Compile (1.5-2x speedup)
```python
model = torch.compile(model, mode='reduce-overhead')
```

### 9. Multi-GPU (Nx speedup)
```yaml
system:
  num_gpus: 4  # Lightning handles DDP automatically
```

## Debugging Slow Training

```bash
nvidia-smi dmon -s u -d 1  # GPU utilization (target >80%)
# <50% = CPU bottleneck (increase workers/batch)
# 100% = GPU bound (good)
```

## Trade-offs

| Optimization | Speedup | Accuracy Impact | Memory |
|-------------|---------|----------------|--------|
| Mixed precision (BF16) | 2-3x | Minimal | -30% |
| Smaller model | 2-4x | May reduce | -50% |
| Larger batch | 1.3-1.5x | May improve | +memory |
| Gradient checkpointing | -20-30% | None | -40% |

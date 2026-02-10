# Preventing NaN/Inf in Training

This guide provides solutions for when a Conv layer (or other layer) produces `inf` values during training.

## Problem Diagnosis

Use the NaN detection hooks to identify the problematic layer:

```python
# In pdb:
pl_module.enable_nan_hooks()
outputs = pl_module(batch['image'])

# Output will show:
# Layer: model.down_layers.3.blocks.2.conv.conv
# Max: inf
# Input range: [-5.2341, 8.9123]  # Input is fine
# Output has Inf: True              # Output exploded
```

## Solutions (Ranked by Effectiveness)

### 1. **Reduce Learning Rate** ⭐ (MOST EFFECTIVE)

Large learning rates are the #1 cause of exploding activations.

```yaml
# In config.yaml
optimizer:
  lr: 0.0001  # Try 10x smaller (was 0.001)
  # Or even: 0.00001

# Or use gradient clipping:
training:
  gradient_clip_val: 0.5  # Clip gradients to prevent explosion
```

**Why it works**: Prevents weights from updating too aggressively, which causes activations to explode.

### 2. **Enable Gradient Clipping** ⭐

Already in your config, but ensure it's set correctly:

```yaml
training:
  gradient_clip_val: 1.0  # Or try 0.5 for more aggressive clipping
```

**Why it works**: Limits gradient magnitude, preventing weight explosions.

### 3. **Use Mixed Precision Correctly**

Your config has `precision: "16-mixed"` which can cause numerical instability:

```yaml
training:
  precision: "32"  # Try full precision first
  # OR
  precision: "bf16-mixed"  # Better numerical stability than fp16
```

**Why it works**: FP16 has limited range (~65k), easily overflows to inf. FP32 or BF16 have much larger ranges.

### 4. **Add Activation Clamping** (Temporary Fix)

Clamp outputs to prevent inf:

```yaml
model:
  clamp_activations: true
  clamp_min: -10.0
  clamp_max: 10.0
```

**Why it works**: Forces activations to stay in a safe range. This is a band-aid - better to fix root cause.

### 5. **Check Model Initialization**

If using MedNeXt or custom models, ensure proper initialization:

```python
# Check weight initialization in pdb:
for name, param in pl_module.named_parameters():
    if 'weight' in name:
        print(f"{name}: std={param.std():.4f}, max={param.max():.4f}")

# Large std (>1.0) or max (>10.0) indicates bad initialization
```

**Fix**: Add weight initialization in model builder:

```python
def init_weights(m):
    if isinstance(m, nn.Conv3d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

model.apply(init_weights)
```

### 6. **Check Batch Normalization**

Ensure BatchNorm is present and working:

```yaml
model:
  norm: batch  # or 'instance', 'group'
```

**Why it works**: Normalization keeps activations stable across layers.

### 7. **Reduce Model Depth/Width**

If using very deep or wide networks:

```yaml
model:
  filters: [16, 32, 64, 128, 256]  # Try smaller (was [32, 64, 128, 256, 512])
```

**Why it works**: Fewer parameters = less capacity for instability.

### 8. **Use Warmup Scheduler**

Your config already has warmup, ensure it's long enough:

```yaml
scheduler:
  warmup_epochs: 10  # Try longer warmup (was 5)
  warmup_start_lr: 0.00001  # Start lower (was 0.0001)
```

**Why it works**: Gradual learning rate increase prevents early training instability.

### 9. **Check Input Normalization**

Ensure inputs are properly normalized (you already fixed this):

```yaml
data:
  normalize: true
  mean: 127.5
  std: 127.5
```

**Verify in pdb**:
```python
batch['image'].min(), batch['image'].max()
# Should be approximately [0, 1] or [-1, 1]
```

### 10. **Use Gradient Accumulation**

Simulate larger batch size for stability:

```yaml
training:
  accumulate_grad_batches: 4  # Effective batch = 4 * batch_size
data:
  batch_size: 2  # Smaller per-GPU batch
```

**Why it works**: Larger effective batch size = more stable gradients.

## Recommended Quick Fixes

**Try these in order:**

### Step 1: Reduce Learning Rate + Full Precision
```yaml
optimizer:
  lr: 0.0001  # 10x smaller

training:
  precision: "32"  # Full precision
  gradient_clip_val: 0.5  # Aggressive clipping
```

### Step 2: If still failing, add clamping temporarily
```yaml
model:
  clamp_activations: true
  clamp_min: -10.0
  clamp_max: 10.0
```

### Step 3: Check initialization
```python
# In pdb:
for name, param in pl_module.named_parameters():
    if torch.isinf(param).any() or param.std() > 2.0:
        print(f"Bad init: {name}")
```

## Long-term Solutions

Once training is stable, gradually:

1. Increase learning rate back up
2. Switch back to mixed precision (`bf16-mixed` recommended over `fp16`)
3. Remove activation clamping
4. Tune gradient clipping threshold

## Debugging Commands

```python
# In pdb when inf detected:

# 1. Check which layer
pl_module.enable_nan_hooks()
outputs = pl_module(batch['image'])

# 2. Inspect that layer's weights
# If layer is: model.down_layers.3.blocks.2.conv.conv
conv = pl_module.model.down_layers[3].blocks[2].conv.conv
conv.weight.min(), conv.weight.max(), conv.weight.std()
conv.bias.min(), conv.bias.max() if conv.bias is not None else None

# 3. Check optimizer state
optimizer = pl_module.optimizers()
lr = optimizer.param_groups[0]['lr']
print(f"Learning rate: {lr}")

# 4. Re-run with clamping
pl_module.clamp_activations = True
outputs = pl_module(batch['image'])
```

## Common Patterns

### Pattern 1: Inf in deep layers
- **Cause**: Activation explosion accumulates through layers
- **Fix**: Gradient clipping + lower LR

### Pattern 2: Inf in first few batches
- **Cause**: Bad initialization or too high initial LR
- **Fix**: Warmup scheduler + weight initialization

### Pattern 3: Inf after many epochs
- **Cause**: Learning rate too high for fine-tuning stage
- **Fix**: Learning rate decay (cosine annealing)

### Pattern 4: Inf only in mixed precision
- **Cause**: FP16 overflow (~65k limit)
- **Fix**: Use BF16 or FP32

## References

- PyTorch Mixed Precision: https://pytorch.org/docs/stable/amp.html
- Gradient Clipping: https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html
- Weight Initialization: https://pytorch.org/docs/stable/nn.init.html

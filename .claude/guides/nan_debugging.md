# Preventing NaN/Inf and Checkerboard Artifacts

## NaN/Inf Debugging

### Diagnosis
```python
# Enable NaN detection hooks
pl_module.enable_nan_hooks()
outputs = pl_module(batch['image'])
# Shows which layer produces inf/nan and input/output ranges
```

### Solutions (Ranked by Effectiveness)

1. **Reduce learning rate** -- #1 cause of exploding activations
   ```yaml
   optimizer:
     lr: 0.0001  # Try 10x smaller
   training:
     gradient_clip_val: 0.5
   ```

2. **Use BF16 instead of FP16** -- FP16 overflows at ~65k
   ```yaml
   training:
     precision: "bf16-mixed"  # or "32" for full precision
   ```

3. **Longer warmup** -- prevents early training instability
   ```yaml
   scheduler:
     warmup_epochs: 10
     warmup_start_lr: 0.00001
   ```

4. **Check weight initialization** -- large std (>1.0) or max (>10.0) indicates bad init
   ```python
   for name, param in pl_module.named_parameters():
       if 'weight' in name and param.std() > 2.0:
           print(f"Bad init: {name}")
   ```

### Common Patterns
- **Inf in deep layers**: activation explosion -- fix with gradient clipping + lower LR
- **Inf in first few batches**: bad initialization -- fix with warmup
- **Inf after many epochs**: LR too high for fine-tuning -- fix with cosine annealing
- **Inf only in mixed precision**: FP16 overflow -- use BF16 or FP32

## Checkerboard Artifacts

### Cause
Transposed convolutions in MONAI UNet upsampling path produce checkerboard patterns.

### Solution: Use RSUNet
```yaml
model:
  architecture: rsunet  # Uses bilinear upsampling + conv (no transposed conv)
  filters: [32, 64, 128, 256]
```

### Additional Fixes
- Use anisotropic patch size for EM: `[18, 160, 160]` instead of `[112, 112, 112]`
- Reduce inference overlap: `0.25` instead of `0.5`
- Use single-channel output with `BCEWithLogitsLoss` for binary tasks
- Use XY-only TTA flips for anisotropic data

### Verification
Check for artifacts in frequency domain -- checkerboard appears as cross pattern in FFT.

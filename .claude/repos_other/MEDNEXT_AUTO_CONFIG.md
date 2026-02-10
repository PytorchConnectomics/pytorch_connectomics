# Automatic Hyperparameter Configuration

**Inspired by nnUNet's experiment planning system**

PyTorch Connectomics now supports automatic hyperparameter configuration based on your GPU capabilities and dataset properties. The system intelligently determines optimal values for batch size, patch size, learning rate, and more.

---

## Features

✅ **GPU-Aware Planning**
- Queries available GPU memory
- Estimates model memory requirements
- Suggests maximum batch size that fits in memory
- Enables mixed precision when beneficial

✅ **Dataset-Aware Planning**
- Adjusts patch size based on voxel spacing (isotropic preference)
- Considers median dataset shape
- Handles anisotropic data (e.g., thick-slice CT)

✅ **Architecture-Specific Defaults**
- MedNeXt: lr=1e-3, constant scheduler
- MONAI models: lr=1e-4, cosine annealing

✅ **Manual Override Support**
- Any value can be manually overridden
- Manual values always take precedence over auto-planned

---

## Quick Start

###  Enable Auto-Planning

```yaml
# tutorials/mednext_auto.yaml
system:
  auto_plan: true  # Enable auto-planning
  print_auto_plan: true  # Print results

model:
  architecture: mednext
  deep_supervision: true

data:
  # Optional: provide dataset statistics for better planning
  target_spacing: [1.0, 1.0, 1.0]  # Voxel spacing in mm
  median_shape: [128, 256, 256]  # Median volume shape

  # These will be AUTO-DETERMINED:
  # batch_size: [auto]
  # patch_size: [auto]
  # num_workers: [auto]

optimizer:
  # lr: [auto]  # Architecture-specific default

training:
  # precision: [auto]  # Mixed precision if GPU supports
  # accumulate_grad_batches: [auto]  # If batch_size=1
```

### Run Training

```bash
python scripts/main.py --config tutorials/mednext_auto.yaml
```

Output:
```
🤖 Running automatic configuration planning...
======================================================================
🤖 Automatic Configuration Planning Results
======================================================================

📊 Data Configuration:
  Patch Size: [128, 128, 128]
  Batch Size: 4
  Num Workers: 8

🧠 Model Configuration:
  Base Features: 32
  Max Features: 320

⚙️  Training Configuration:
  Precision: 16-mixed
  Learning Rate: 0.001

💾 GPU Memory:
  Available: 23.45 GB
  Estimated Usage: 18.23 GB (77.7%)
  Per Sample: 4.56 GB

📝 Planning Notes:
  • Architecture: mednext
  • Patch size: [128, 128, 128]
  • GPU: NVIDIA RTX 3090 (23.5 GB available)
  • Estimated memory: 18.23 GB (77.7% of GPU)
  • Batch size: 4
  • Num workers: 8
  • Learning rate: 0.001

======================================================================
💡 Tip: You can manually override any of these values in your config!
======================================================================
```

---

## How It Works

### 1. GPU Memory Estimation

The system estimates memory requirements using a simplified version of nnUNet's approach:

```python
memory = (
    activation_memory +   # Feature maps during forward pass
    gradient_memory +     # Gradients during backward pass
    parameter_memory +    # Model weights (~100MB for typical 3D U-Net)
    optimizer_memory +    # AdamW state (2x parameters)
    workspace_memory      # CUDNN overhead (~20%)
)
```

**Mixed precision (FP16) reduces activation/gradient memory by 50%.**

### 2. Patch Size Planning

Strategy:
1. Start with median dataset shape
2. Adjust for anisotropic spacing (prefer isotropic receptive field)
3. Ensure divisible by 16 (for 4 pooling stages: 2^4 = 16)
4. Consider GPU memory limits

Example:
- Anisotropic CT: [5.0, 0.5, 0.5] mm spacing → reduce Z-axis patch size
- Isotropic MRI: [1.0, 1.0, 1.0] mm spacing → balanced patch size

### 3. Batch Size Determination

Algorithm:
1. Estimate memory for batch_size=1
2. Binary search for maximum batch_size that fits in 85% of GPU memory
3. If batch_size=1 and still tight, enable gradient accumulation

Safety margin: Uses 85% of available GPU memory by default

### 4. Architecture-Specific Defaults

**MedNeXt**:
- Learning rate: 1e-3 (paper recommendation)
- Scheduler: Constant LR (no scheduler)
- Reason: MedNeXt paper found constant LR works best

**MONAI Models**:
- Learning rate: 1e-4 (typical for medical imaging)
- Scheduler: Cosine annealing with warmup

### 5. Worker Count

Rule of thumb: 4-8 workers per GPU, capped at CPU count
```python
num_workers = min(4 * num_gpus, cpu_count)
```

---

## Manual Overrides

Any auto-planned value can be manually overridden:

```yaml
system:
  auto_plan: true

data:
  batch_size: 2  # OVERRIDE: Force batch_size=2
  patch_size: [64, 64, 64]  # OVERRIDE: Smaller patches

training:
  precision: "32"  # OVERRIDE: No mixed precision

optimizer:
  lr: 0.0005  # OVERRIDE: Custom learning rate
```

**Manual values always take precedence over auto-planned values.**

---

## API Usage

### Programmatic Planning

```python
from connectomics.config import Config, auto_plan_config
from omegaconf import OmegaConf

# Load config
cfg = OmegaConf.structured(Config())
cfg.system.auto_plan = True
cfg.model.architecture = 'mednext'
cfg.model.deep_supervision = True
cfg.data.target_spacing = [1.0, 1.0, 1.0]
cfg.data.median_shape = [128, 256, 256]

# Auto-plan
cfg = auto_plan_config(cfg, print_results=True)

# Access planned values
print(f"Planned batch_size: {cfg.data.batch_size}")
print(f"Planned patch_size: {cfg.data.patch_size}")
print(f"Planned precision: {cfg.training.precision}")
```

### GPU Utilities

```python
from connectomics.config import (
    get_gpu_info,
    print_gpu_info,
    suggest_batch_size,
    estimate_gpu_memory_required,
)

# Query GPU
gpu_info = get_gpu_info()
print(f"GPUs: {gpu_info['num_gpus']}")
print(f"Memory: {gpu_info['total_memory_gb']} GB")

# Estimate memory
memory_gb = estimate_gpu_memory_required(
    patch_size=(128, 128, 128),
    batch_size=4,
    in_channels=1,
    out_channels=2,
    deep_supervision=True,
    mixed_precision=True,
)
print(f"Estimated: {memory_gb:.2f} GB")

# Suggest batch size
batch_size = suggest_batch_size(
    patch_size=(128, 128, 128),
    in_channels=1,
    out_channels=2,
    available_gpu_memory_gb=24.0,
    deep_supervision=True,
)
print(f"Suggested batch_size: {batch_size}")
```

### Direct Planner Usage

```python
from connectomics.config import AutoConfigPlanner

planner = AutoConfigPlanner(
    architecture='mednext',
    target_spacing=[1.0, 1.0, 1.0],
    median_shape=[128, 256, 256],
    manual_overrides={'batch_size': 2},  # Force batch_size=2
)

result = planner.plan(
    in_channels=1,
    out_channels=2,
    deep_supervision=True,
    use_mixed_precision=True,
)

planner.print_plan(result)

# Access results
print(f"Patch size: {result.patch_size}")
print(f"Batch size: {result.batch_size}")
print(f"Learning rate: {result.lr}")
print(f"Warnings: {result.warnings}")
```

---

## Example Configs

### 1. Full Auto-Planning

```yaml
# tutorials/mednext_lucchi_auto.yaml
system:
  auto_plan: true

model:
  architecture: mednext
  mednext_size: S
  deep_supervision: true

data:
  target_spacing: [1.0, 1.0, 1.0]
  median_shape: [165, 768, 1024]
  # All other parameters auto-determined
```

### 2. Partial Auto-Planning

```yaml
system:
  auto_plan: true

data:
  batch_size: 4  # Manual: force batch_size=4
  # patch_size: [auto]
  # num_workers: [auto]

training:
  precision: "16-mixed"  # Manual: force mixed precision
  # accumulate_grad_batches: [auto]

optimizer:
  # lr: [auto]  # Let auto-planner decide
```

### 3. Manual Configuration (No Auto-Planning)

```yaml
system:
  auto_plan: false  # Disable auto-planning

data:
  patch_size: [128, 128, 128]
  batch_size: 2
  num_workers: 4

training:
  precision: "32"

optimizer:
  lr: 0.0001
```

---

## Comparison: nnUNet vs PyTC Auto-Planning

| Feature | nnUNet | PyTC |
|---------|--------|------|
| **Planning Scope** | Full preprocessing + training | Training hyperparameters only |
| **Dataset Analysis** | Automatic from raw data | Optional: provide spacing/shape |
| **GPU Awareness** | ✓ Memory-based batch size | ✓ Memory-based batch size + precision |
| **Architecture Support** | U-Net variants | MedNeXt, MONAI models |
| **Manual Override** | Limited | Full (any parameter) |
| **Integration** | Standalone pipeline | Integrated into training script |
| **Lightning Support** | ✗ | ✓ (native integration) |

---

## Best Practices

### When to Use Auto-Planning

✅ **Use auto-planning when:**
- Training on new hardware
- Experimenting with different models
- Working with anisotropic data
- Unsure about optimal hyperparameters
- Want to maximize GPU utilization

❌ **Don't use auto-planning when:**
- You have carefully tuned hyperparameters
- Reproducing published results
- Need exact control over all parameters
- Working with non-standard architectures

### Tips for Best Results

1. **Provide Dataset Statistics**
   ```yaml
   data:
     target_spacing: [1.0, 1.0, 1.0]
     median_shape: [128, 256, 256]
   ```
   Better estimates → better planning

2. **Start with Auto-Planning**
   - Let it suggest initial values
   - Train for a few epochs
   - Fine-tune manually if needed

3. **Monitor GPU Usage**
   ```bash
   watch -n 1 nvidia-smi
   ```
   - If using <70% GPU memory → increase batch_size
   - If OOM errors → decrease batch_size or patch_size

4. **Architecture Matters**
   - MedNeXt: Use auto-planning (respects paper recommendations)
   - MONAI models: Auto-planning provides good defaults
   - Custom models: Test auto-planning, may need manual tuning

5. **Anisotropic Data**
   - Auto-planner adjusts patch size automatically
   - Review planned patch_size, may need manual adjustment
   - Consider data spacing in acquisition planning

---

## Troubleshooting

### Issue: Auto-planned batch_size=1

**Cause:** Limited GPU memory or large patch size

**Solutions:**
1. Reduce patch_size manually:
   ```yaml
   data:
     patch_size: [64, 64, 64]  # Smaller
   ```
2. Use gradient accumulation (auto-enabled if batch_size=1)
3. Enable mixed precision (auto-enabled if GPU supports)

### Issue: Planning takes a long time

**Cause:** No GPU info cached, querying hardware

**Solutions:**
- Normal on first run
- Subsequent runs are fast
- Set `print_auto_plan: false` to skip detailed output

### Issue: Planned values seem wrong

**Cause:** Incorrect dataset statistics or architecture defaults

**Solutions:**
1. Verify `target_spacing` and `median_shape`
2. Check GPU memory availability (may be in use)
3. Override with manual values
4. Report issue with:
   ```python
   from connectomics.config import print_gpu_info
   print_gpu_info()
   ```

### Issue: Manual overrides not working

**Cause:** Config structure issue

**Solutions:**
1. Ensure values are explicitly set in YAML
2. Check config after loading:
   ```python
   from connectomics.config import load_config, print_config
   cfg = load_config('config.yaml')
   print_config(cfg)
   ```

---

## Advanced: Custom Planning

Extend the auto-planner for custom needs:

```python
from connectomics.config import AutoConfigPlanner

class CustomPlanner(AutoConfigPlanner):
    def _plan_patch_size(self):
        """Override patch size planning logic."""
        # Custom logic here
        patch_size = super()._plan_patch_size()
        # Modify patch_size
        return patch_size

    def _get_architecture_defaults(self):
        """Add custom architecture defaults."""
        defaults = super()._get_architecture_defaults()
        defaults['my_custom_arch'] = {
            'base_features': 64,
            'lr': 5e-4,
        }
        return defaults
```

---

## Implementation Details

### Files
- `connectomics/config/gpu_utils.py` - GPU query and memory estimation
- `connectomics/config/auto_config.py` - Auto-planning logic
- `connectomics/config/hydra_config.py` - Config with auto-plan fields
- `scripts/main.py` - Integration into training script

### Key Functions
- `get_gpu_info()` - Query CUDA devices
- `estimate_gpu_memory_required()` - Estimate model memory
- `suggest_batch_size()` - Binary search for max batch size
- `auto_plan_config()` - Main auto-planning entry point
- `AutoConfigPlanner.plan()` - Core planning logic

### Memory Estimation Formula

```python
# Activations (feature maps)
activation_memory = total_voxels * bytes_per_element  # 4 (FP32) or 2 (FP16)

# Gradients (same size as activations)
gradient_memory = activation_memory

# Parameters (~100MB for typical 3D U-Net)
parameter_memory = 0.1 GB

# Optimizer state (AdamW: 2x parameters)
optimizer_memory = parameter_memory * 2

# Workspace (CUDNN, etc.: 20% overhead)
workspace_memory = (activation_memory + gradient_memory) * 0.2

# Total
total_memory = (activation_memory + gradient_memory +
                parameter_memory + optimizer_memory +
                workspace_memory)
```

---

## Future Improvements

Planned enhancements:
- [ ] Multi-GPU memory estimation
- [ ] Automatic data preprocessing planning (spacing normalization)
- [ ] Model architecture search based on dataset size
- [ ] Automatic augmentation strategy selection
- [ ] Integration with dataset analyzers (nnUNet-style)
- [ ] Caching of planning results
- [ ] Benchmark database for reference

---

## References

- **nnUNet**: Isensee et al., "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation", Nature Methods 2021
- **MedNeXt**: Roy et al., "MedNeXt: Transformer-driven Scaling of ConvNets for Medical Image Segmentation", MICCAI 2023
- **PyTorch Lightning**: Automated DDP, mixed precision, etc.

---

## Conclusion

Auto-planning simplifies the configuration process by intelligently determining hyperparameters based on your hardware and data. It's inspired by nnUNet's success but adapted for Lightning + MONAI workflows.

**Key Benefits:**
✅ Faster experimentation
✅ Better GPU utilization
✅ Architecture-aware defaults
✅ Still supports full manual control

**Try it out:**
```bash
python scripts/main.py --config tutorials/mednext_lucchi_auto.yaml
```

---

**Documentation Version:** 1.0.0
**Last Updated:** 2025-09-30

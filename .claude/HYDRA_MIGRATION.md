# Hydra Configuration System Migration

## Summary

Successfully replaced the legacy YACS configuration system with a modern Hydra/OmegaConf-based configuration system for PyTorch Connectomics.

## What's New

### ✅ Modern Configuration System
- **Type-safe dataclasses** with full type hints
- **Structured configs** using OmegaConf
- **Composable and mergeable** configurations
- **CLI-friendly** with dot-notation overrides
- **IDE-friendly** with autocomplete support
- **Built-in validation** logic

### ✅ New Files Created

#### Configuration System
- `connectomics/config/hydra_config.py` - Core dataclass configs
- `connectomics/config/hydra_utils.py` - Utility functions
- `connectomics/transforms/augment/hydra_compose.py` - Transform builder
- `configs/hydra/default.yaml` - Default configuration
- `configs/hydra/lucchi.yaml` - Lucchi dataset configuration
- `configs/hydra/README.md` - Complete documentation

#### Testing
- `tests/test_hydra_config.py` - Comprehensive test suite

### ✅ Configuration Structure

```
Config
├── system (SystemConfig)
│   ├── num_gpus, num_cpus, seed
├── model (ModelConfig)
│   ├── architecture, filters, dropout
│   ├── in_channels, out_channels
│   ├── loss_functions, loss_weights
├── data (DataConfig)
│   ├── train_image, train_label
│   ├── batch_size, num_workers
│   ├── patch_size, pad_size
│   ├── use_cache, normalize
├── optimizer (OptimizerConfig)
│   ├── name, lr, weight_decay
├── scheduler (SchedulerConfig)
│   ├── name, warmup_epochs, min_lr
├── training (TrainingConfig)
│   ├── max_epochs, precision
│   ├── gradient_clip_val
│   ├── val_check_interval
├── checkpoint (CheckpointConfig)
│   ├── save_top_k, monitor, mode
├── early_stopping (EarlyStoppingConfig)
│   ├── enabled, patience
├── augmentation (AugmentationConfig)
│   ├── flip, rotate, elastic
│   ├── intensity, misalignment
│   ├── missing_section, motion_blur
│   ├── cut_noise, cut_blur
│   ├── mixup, copy_paste
└── inference (InferenceConfig)
    ├── output_path, stride, overlap
```

## Key Features

### 1. Clean API

```python
from connectomics.config import Config, load_config, save_config

# Create default
cfg = Config()

# Load from YAML
cfg = load_config("configs/hydra/lucchi.yaml")

# Save
save_config(cfg, "outputs/config.yaml")
```

### 2. CLI Overrides

```python
from connectomics.config import update_from_cli

cfg = load_config("configs/hydra/default.yaml")
cfg = update_from_cli(cfg, [
    'data.batch_size=8',
    'model.architecture=unetr',
    'optimizer.lr=0.001'
])
```

### 3. Config Merging

```python
from connectomics.config import merge_configs

base = load_config("configs/hydra/default.yaml")
custom = {"data": {"batch_size": 16}}
cfg = merge_configs(base, custom)
```

### 4. Validation

```python
from connectomics.config import validate_config

validate_config(cfg)  # Raises ValueError if invalid
```

### 5. Transform Building

```python
from connectomics.transforms.augment.hydra_compose import (
    build_train_transforms,
    build_val_transforms,
    build_transform_dict
)

transforms = build_transform_dict(cfg)
```

## Augmentation Support

All augmentations are now configurable through the Hydra config:

### Standard Augmentations
- Flip, Rotate, Elastic Deformation
- Intensity (noise, shift, contrast)

### EM-Specific Augmentations
- Misalignment
- Missing Sections
- Motion Blur
- Cut Noise
- Cut Blur
- Missing Parts

### Advanced Augmentations
- **Mixup** - Batch-level mixing
- **Copy-Paste** - Instance augmentation

## Testing

All tests pass successfully:

```bash
python tests/test_hydra_config.py
```

Test coverage:
- ✅ Default config creation
- ✅ Config validation
- ✅ Dict conversion
- ✅ CLI updates
- ✅ Config merging
- ✅ Save/load
- ✅ Hashing
- ✅ Experiment naming
- ✅ Augmentation configs
- ✅ Example config loading

## Dependencies

New dependencies added (automatically installed):
- `omegaconf>=2.3.0`
- `hydra-core>=1.3.0`

## Example Configs

### default.yaml
Clean baseline with standard augmentations.

### lucchi.yaml
Optimized for EM data with:
- Instance normalization
- EM-specific augmentations enabled
- Smaller patch sizes for 2.5D
- Early stopping on val/dice

## Benefits Over YACS

1. **Type Safety** - Catch errors at IDE time, not runtime
2. **Composability** - Mix and match config files
3. **Clarity** - Clear hierarchy and structure
4. **Flexibility** - Easy CLI overrides without code changes
5. **Modern** - Active development, better tooling
6. **Standards** - Uses Python dataclasses standard

## Backward Compatibility

The old YACS system remains available through:
```python
from connectomics.config import get_cfg_defaults  # Legacy
```

But new code should use:
```python
from connectomics.config import Config, load_config  # Modern
```

## Next Steps

Recommended usage:
1. Use Hydra configs for all new experiments
2. Create experiment-specific YAML files in `configs/hydra/`
3. Use CLI overrides for hyperparameter sweeps
4. Save final configs with experiment outputs

## Files Modified

- `connectomics/config/__init__.py` - Updated exports
- `connectomics/transforms/augment/__init__.py` - No changes needed

## Documentation

Full documentation available in:
- `configs/hydra/README.md` - Complete user guide
- `connectomics/config/hydra_config.py` - Docstrings
- `connectomics/config/hydra_utils.py` - API reference

---

**Status**: ✅ Complete and tested
**Python**: 3.9+
**Dependencies**: OmegaConf 2.3+, Hydra 1.3+

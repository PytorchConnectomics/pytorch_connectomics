# MedNeXt Quick Start Guide

**TL;DR:** Use MedNeXt-S with deep supervision for best performance.

---

## Installation

```bash
# Install MedNeXt library
pip install -e /projects/weilab/weidf/lib/MedNeXt

# Verify installation
python -c "from nnunet_mednext import create_mednext_v1; print('✓ MedNeXt installed')"
```

---

## Quick Start (5 minutes)

### Option 1: Use Example Config (Recommended)

```bash
# Train MedNeXt-S on Lucchi dataset
python scripts/main.py --config tutorials/mednext_lucchi.yaml

# Override from CLI
python scripts/main.py --config tutorials/mednext_lucchi.yaml \
    data.batch_size=8 \
    training.max_epochs=200 \
    model.mednext_size=B
```

### Option 2: Create Custom Config

**Minimal config** (`my_mednext.yaml`):
```yaml
model:
  architecture: mednext
  in_channels: 1
  out_channels: 2
  mednext_size: S              # S (5.6M), B (10.5M), M (17.6M), or L (61.8M)
  mednext_kernel_size: 3       # Start with 3
  deep_supervision: true       # RECOMMENDED

data:
  train_image: "path/to/train_image.h5"
  train_label: "path/to/train_label.h5"
  val_image: "path/to/val_image.h5"
  val_label: "path/to/val_label.h5"

optimizer:
  name: AdamW
  lr: 0.001                    # MedNeXt default: 1e-3

training:
  max_epochs: 100
  precision: "16-mixed"
```

**Run:**
```bash
python scripts/main.py --config my_mednext.yaml
```

---

## Model Sizes

| Size | Params  | Use Case              |
|------|---------|----------------------|
| S    | 5.6M    | Fast training, limited GPU memory |
| B    | 10.5M   | Balanced (RECOMMENDED for most) |
| M    | 17.6M   | Higher capacity |
| L    | 61.8M   | Maximum performance (requires 24GB+ GPU) |

---

## Common Configurations

### Binary Segmentation (e.g., Mitochondria)

```yaml
model:
  architecture: mednext
  in_channels: 1
  out_channels: 1              # Binary
  mednext_size: S
  deep_supervision: true
  loss_functions: [DiceLoss, BCEWithLogitsLoss]
  loss_weights: [1.0, 1.0]
```

### Multi-Class Segmentation

```yaml
model:
  architecture: mednext
  in_channels: 1
  out_channels: 5              # 5 classes
  mednext_size: B
  deep_supervision: true
  loss_functions: [DiceLoss, CrossEntropyLoss]
  loss_weights: [1.0, 1.0]
```

### Multi-Channel Input (e.g., RGB or Multi-Modal)

```yaml
model:
  architecture: mednext
  in_channels: 3               # RGB
  out_channels: 2
  mednext_size: B
  deep_supervision: true
```

---

## Hyperparameters

### Recommended (from MedNeXt paper)

```yaml
model:
  mednext_size: S
  mednext_kernel_size: 3       # Start with 3x3x3
  deep_supervision: true       # Critical for performance

optimizer:
  name: AdamW
  lr: 0.001                    # 1e-3 (higher than typical)
  weight_decay: 0.0001

scheduler:
  name: none                   # Constant LR (paper recommendation)
  # OR use mild cosine annealing:
  # name: CosineAnnealingLR
  # min_lr: 0.0001

training:
  precision: "16-mixed"        # Mixed precision
  gradient_clip_val: 1.0
  max_epochs: 100

data:
  batch_size: 2                # Adjust based on GPU memory
  # MedNeXt prefers 1mm isotropic spacing
```

### Advanced: UpKern (Larger Kernels)

**Step 1:** Train with kernel_size=3
```yaml
model:
  mednext_kernel_size: 3
```

**Step 2:** Initialize larger kernel from trained model
```python
from connectomics.models.architectures.mednext_models import upkern_load_weights

# Load k=3 model
model_k3 = load_checkpoint('checkpoint_k3.ckpt')

# Create k=5 model
cfg.model.mednext_kernel_size = 5
model_k5 = build_model(cfg)

# Transfer weights with UpKern
upkern_load_weights(model_k5, model_k3)

# Fine-tune
# ...
```

---

## Troubleshooting

### Issue: Out of Memory

**Solutions:**
1. Reduce batch size: `data.batch_size=1`
2. Use smaller model: `model.mednext_size=S`
3. Enable gradient checkpointing (MedNeXt-M/L):
   ```yaml
   model:
     mednext_checkpoint_style: outside_block  # Trades speed for memory
   ```
4. Reduce patch size:
   ```yaml
   data:
     patch_size: [64, 64, 64]  # Smaller than [128, 128, 128]
   ```

### Issue: Slow Training

**Solutions:**
1. Enable mixed precision (if not already):
   ```yaml
   training:
     precision: "16-mixed"
   ```
2. Increase batch size (if memory allows):
   ```yaml
   data:
     batch_size: 4
   ```
3. Use persistent workers:
   ```yaml
   data:
     persistent_workers: true
     num_workers: 4
   ```
4. Enable data caching:
   ```yaml
   data:
     use_cache: true
     cache_rate: 1.0
   ```

### Issue: Poor Performance

**Solutions:**
1. **Enable deep supervision** (most important):
   ```yaml
   model:
     deep_supervision: true
   ```
2. Check learning rate (MedNeXt uses 1e-3, not 1e-4)
3. Ensure 1mm isotropic spacing (if possible)
4. Try larger model size: S → B → M
5. Check data augmentation is enabled
6. Increase training epochs (MedNeXt may need 300-500 epochs)

### Issue: MedNeXt Not Found

**Error:** `ModuleNotFoundError: No module named 'nnunet_mednext'`

**Solution:**
```bash
# Install MedNeXt
pip install -e /projects/weilab/weidf/lib/MedNeXt

# OR add to PYTHONPATH
export PYTHONPATH=/projects/weilab/weidf/lib/MedNeXt:$PYTHONPATH
```

---

## Comparison: MedNeXt vs MONAI UNet

| Feature | MedNeXt-S | MONAI BasicUNet |
|---------|-----------|----------------|
| Params | 5.6M | ~53M (depends on filters) |
| Speed | Fast | Medium |
| Memory | Low | Medium |
| Performance | Excellent with deep supervision | Good |
| Best for | 3D medical segmentation | General purpose |

**When to use MedNeXt:**
- 3D medical image segmentation
- Limited GPU memory
- State-of-the-art performance needed
- Willing to enable deep supervision

**When to use MONAI UNet:**
- Simple baseline
- Don't want deep supervision
- Need very flexible architecture

---

## CLI Reference

```bash
# Basic training
python scripts/main.py --config path/to/config.yaml

# Override config
python scripts/main.py --config config.yaml key=value

# Fast dev run (1 batch)
python scripts/main.py --config config.yaml --fast-dev-run

# Resume from checkpoint
python scripts/main.py --config config.yaml --checkpoint path/to/ckpt.ckpt

# Test mode
python scripts/main.py --config config.yaml --mode test --checkpoint path/to/ckpt.ckpt

# Multi-GPU training (automatic with Lightning)
python scripts/main.py --config config.yaml system.num_gpus=4
```

---

## Python API

```python
from omegaconf import OmegaConf
from connectomics.models import build_model
from connectomics.lightning import ConnectomicsModule
import pytorch_lightning as pl

# Load config
cfg = OmegaConf.load('tutorials/mednext_lucchi.yaml')

# Build model
model = build_model(cfg)

# Get info
info = model.get_model_info()
print(f"Parameters: {info['parameters']:,}")
print(f"Deep supervision: {info['deep_supervision']}")

# Create Lightning module
lit_model = ConnectomicsModule(cfg, model=model)

# Train
trainer = pl.Trainer(
    max_epochs=cfg.training.max_epochs,
    accelerator='gpu',
    devices=cfg.system.num_gpus,
    precision=cfg.training.precision,
)
trainer.fit(lit_model, datamodule)
```

---

## Example Workflow

### Complete Training Pipeline

```bash
# 1. Prepare data (your scripts)
python prepare_data.py --input raw/ --output datasets/

# 2. Train MedNeXt-S with deep supervision
python scripts/main.py --config tutorials/mednext_lucchi.yaml \
    data.train_image=datasets/train_image.h5 \
    data.train_label=datasets/train_label.h5 \
    data.val_image=datasets/val_image.h5 \
    data.val_label=datasets/val_label.h5 \
    experiment_name=my_experiment

# 3. Monitor training
tensorboard --logdir outputs/my_experiment/logs/

# 4. Test best model
python scripts/main.py --config tutorials/mednext_lucchi.yaml \
    --mode test \
    --checkpoint outputs/my_experiment/checkpoints/best.ckpt

# 5. Inference on new data
python scripts/main.py --config tutorials/mednext_lucchi.yaml \
    --mode test \
    --checkpoint outputs/my_experiment/checkpoints/best.ckpt \
    data.test_image=datasets/test_image.h5
```

---

## Further Reading

- **MEDNEXT.md**: Complete MedNeXt integration guide
- **IMPLEMENTATION_SUMMARY.md**: What was implemented
- **DESIGN.md**: Architecture philosophy
- **MedNeXt Paper**: https://arxiv.org/abs/2303.09975

---

## Need Help?

1. Check existing configs: `tutorials/mednext_lucchi.yaml`, `tutorials/mednext_custom.yaml`
2. List available architectures:
   ```python
   from connectomics.models.architectures import print_available_architectures
   print_available_architectures()
   ```
3. Validate config:
   ```python
   from connectomics.config import load_config, validate_config
   cfg = load_config('config.yaml')
   validate_config(cfg)
   ```

---

**Good luck with your MedNeXt training! 🚀**
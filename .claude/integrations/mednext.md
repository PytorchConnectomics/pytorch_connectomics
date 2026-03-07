# MedNeXt Integration

MedNeXt (MICCAI 2023) is a ConvNeXt-based 3D medical image segmentation architecture from DKFZ.

## Installation

```bash
pip install -e /projects/weilab/weidf/lib/MedNeXt
python -c "from nnunet_mednext import create_mednext_v1; print('OK')"
```

Optional external dependency -- graceful fallback if not installed.

## Usage

### Predefined Sizes (`mednext` architecture)

```yaml
model:
  architecture: mednext
  mednext_size: S              # S (5.6M), B (10.5M), M (17.6M), L (61.8M)
  mednext_kernel_size: 3       # 3, 5, or 7
  deep_supervision: true       # RECOMMENDED for best performance
```

### Custom Configuration (`mednext_custom` architecture)

```yaml
model:
  architecture: mednext_custom
  mednext_base_channels: 32
  mednext_exp_r: [2, 3, 4, 4, 4, 4, 4, 3, 2]
  mednext_block_counts: [3, 4, 8, 8, 8, 8, 8, 4, 3]
  mednext_kernel_size: 7
  deep_supervision: true
  mednext_grn: true
```

## Model Sizes

| Size | Params | GFlops | Use Case |
|------|--------|--------|----------|
| S | 5.6M | 130 | Fast training, limited GPU memory |
| B | 10.5M | 170 | Balanced (RECOMMENDED) |
| M | 17.6M | 248 | Higher capacity |
| L | 61.8M | 500 | Maximum performance (24GB+ GPU) |

## Key Features

- **Deep Supervision**: 5-scale outputs for improved training (critical for performance)
- **UpKern**: Initialize larger kernels (5/7) from trained smaller kernel (3) models
- **Gradient Checkpointing**: `mednext_checkpoint_style: outside_block` for large models
- **Isotropic Spacing**: Prefers 1mm isotropic spacing

## Recommended Hyperparameters

```yaml
optimizer:
  name: AdamW
  lr: 0.001           # Higher than typical (MedNeXt paper default)
scheduler:
  name: none           # Constant LR (paper recommendation)
training:
  precision: "16-mixed"
  gradient_clip_val: 1.0
```

## Architecture Details

- **Block**: Depthwise conv + GroupNorm + expansion (1x1x1) + GELU + compression + residual
- **5 encoder levels** with 4 downsampling, bottleneck, 4 decoder levels with skip connections
- **Source**: `connectomics/models/arch/mednext_models.py`
- **External lib**: `/projects/weilab/weidf/lib/MedNeXt/nnunet_mednext/`

## Troubleshooting

- **OOM**: Reduce batch_size, use smaller model (S), enable gradient checkpointing
- **Poor performance**: Enable deep_supervision, check lr=1e-3, try larger model
- **Not found**: `pip install -e /projects/weilab/weidf/lib/MedNeXt`

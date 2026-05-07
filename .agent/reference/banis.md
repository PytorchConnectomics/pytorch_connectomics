# BANIS Reference

**BANIS** (Baseline for Affinity-based Neuron Instance Segmentation) is a NISB benchmark baseline combining affinity prediction with MedNeXt and connected components.

**Location**: `/projects/weilab/weidf/lib/seg/banis`

## Architecture

- **Model**: MedNeXt (S/B/M/L) predicting 6 affinity channels (3 short-range + 3 long-range)
- **Post-processing**: Numba JIT-compiled connected components (6-connectivity)
- **Training**: PyTorch Lightning, AdamW lr=1e-3, mixed precision
- **Evaluation**: VOI, NERL (skeleton-based), threshold sweep

## What PyTC Adopted from BANIS

1. MedNeXt architecture integration
2. Numba-accelerated connected components
3. Skeleton-based metrics (NERL)
4. Weighted concat dataset for multi-dataset mixing
5. Optuna-based threshold optimization
6. Slice-level EM augmentations (drop/shift)

## Key Differences from PyTC

| Feature | PyTC | BANIS |
|---------|------|-------|
| Config | Hydra/OmegaConf | Argparse |
| Scope | General framework | NISB-specific |
| Post-processing | Multiple (watershed, mutex, CC) | CC only |
| Models | Many (UNet, UNETR, Swin, MedNeXt, RSUNet) | MedNeXt only |
| Training | Epoch-based (Lightning) | Step-based (Lightning) |

## NISB Data Settings

base, liconn, multichannel, neg_guidance, pos_guidance, no_touch_thick, touching_thin, slice_perturbed, train_100

# nnUNet Integration

Pre-trained nnUNet v2 models can be used for inference in PyTC via wrapper architectures.

## Architectures

- `nnunet_2d_pretrained`: 2D semantic segmentation (e.g., mitochondria)
- `nnunet_3d_pretrained`: 3D semantic segmentation

## Configuration

```yaml
model:
  architecture: nnunet_2d_pretrained
  in_channels: 1
  out_channels: 2
  nnunet:
    model_path: "/path/to/nnunet/model/"
    fold: 0
    checkpoint: "checkpoint_best.pth"
```

## Key Features

- Loads pre-trained nnUNet v2 plans and weights
- Handles nnUNet-specific preprocessing (resampling, normalization, crop-to-nonzero)
- Inverse transforms restore predictions to original input space
- Supports sliding window inference via MONAI inferer

## Preprocessing

The `NNUNetPreprocessd` transform handles:
- Resampling to nnUNet planned spacing
- Z-score normalization per nnUNet plan
- Crop-to-nonzero (optional)
- Located in `connectomics/data/process/nnunet_preprocess.py`

## Implementation

- Wrapper: `connectomics/models/arch/nnunet_models.py`
- Config schema: `connectomics/config/schema/model_nnunet.py`
- Example config: `tutorials/misc/mito_2dsem_seg.yaml`

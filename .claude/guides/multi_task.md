# Multi-Task Learning Guide

Multi-task learning allows a single model to predict multiple targets simultaneously (e.g., binary mask + boundary + distance transform).

## Configuration

### 1. Model Output Channels

```yaml
model:
  out_channels: 3  # Total across all tasks (1 binary + 1 boundary + 1 EDT)
```

### 2. Loss Functions

```yaml
model:
  loss_functions: [DiceLoss, BCEWithLogitsLoss, WeightedMSE]
  loss_weights: [1.0, 0.5, 5.0]
  loss_kwargs:
    - {sigmoid: true, smooth_nr: 1e-5, smooth_dr: 1e-5}
    - {}
    - {tanh: true}
```

### 3. Channel-to-Task Mapping

```yaml
model:
  multi_task_config:
    - [0, 1, "label", [0, 1]]       # Channel 0: binary -> DiceLoss + BCE
    - [1, 2, "boundary", [0, 1]]    # Channel 1: boundary -> DiceLoss + BCE
    - [2, 3, "edt", [2]]            # Channel 2: EDT -> WeightedMSE
```

Format: `[start_channel, end_channel, "task_name", [loss_indices]]`

### 4. Label Targets

```yaml
data:
  label_transform:
    targets:
      - name: binary
      - name: instance_boundary
        kwargs: {tsz_h: 1, do_bg: false, do_convolve: false}
      - name: instance_edt
        kwargs: {mode: "2d", quantize: false}
```

## Available Target Types

| Name | Description | Output Shape |
|------|-------------|-------------|
| `binary` | Foreground mask | [1, D, H, W] |
| `affinity` | Connectivity maps | [N_offsets, D, H, W] |
| `instance_boundary` | Instance boundaries (2D/3D) | [1, D, H, W] |
| `instance_edt` | Euclidean Distance Transform | [1, D, H, W] |
| `semantic_edt` | EDT with fg/bg channels | [2, D, H, W] |
| `polarity` | Synaptic polarity | [3, D, H, W] |
| `small_object` | Small object detection | [1, D, H, W] |

## Loss Selection Guide

| Task | Recommended Losses |
|------|-------------------|
| Binary segmentation | DiceLoss + BCE |
| Boundary detection | DiceLoss + BCE |
| Distance transform | WeightedMSE or MSE |
| Affinity maps | BCE or WeightedBCE |
| Multi-class | CrossEntropy + Dice |

## Single-Task Mode

If `multi_task_config` is null, all losses apply to all output channels.

## Troubleshooting

- **NaN in loss**: Check `loss_kwargs` (add `smooth_nr`, `smooth_dr` for DiceLoss)
- **Imbalanced tasks**: Adjust `loss_weights`, monitor per-task losses
- **Channel mismatch**: Ensure `out_channels` = sum of task channels, `multi_task_config` covers all channels

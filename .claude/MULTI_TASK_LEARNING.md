# Multi-Task Learning in PyTorch Connectomics

This guide explains how to configure multi-task learning where the model predicts multiple targets simultaneously, each with specific loss functions.

## Overview

Multi-task learning allows a single model to predict multiple related outputs. For example:
- **Binary mask** (foreground/background)
- **Boundary mask** (instance boundaries)
- **Distance transform** (EDT for instance separation)

Each task can have its own dedicated loss function(s).

## Configuration

### 1. Model Output Channels

Set `out_channels` to the total number of output channels across all tasks:

```yaml
model:
  out_channels: 3  # 1 (binary) + 1 (boundary) + 1 (EDT)
```

### 2. Define Loss Functions

List all loss functions you want to use:

```yaml
model:
  loss_functions: [DiceLoss, BCEWithLogitsLoss, WeightedMSE]
  loss_weights: [1.0, 0.5, 5.0]
  loss_kwargs:
    - {sigmoid: true, smooth_nr: 1e-5, smooth_dr: 1e-5}  # DiceLoss with sigmoid activation
    - {}                                                  # BCEWithLogitsLoss
    - {tanh: true}                                        # WeightedMSE with tanh activation (for distance transforms)
```

**Note:** Loss functions can apply activations during loss computation:
- `DiceLoss`: Use `sigmoid: true` to apply sigmoid activation to predictions
- `WeightedMSE/WeightedMAE`: Use `tanh: true` to apply tanh activation (useful for distance transforms in range [-1, 1])

### 3. Multi-Task Configuration

Specify which output channels belong to which tasks and which losses to apply:

```yaml
model:
  multi_task_config:
    - [0, 1, "label", [0, 1]]       # Channel 0 → binary: DiceLoss + BCE
    - [1, 2, "boundary", [0, 1]]    # Channel 1 → boundary: DiceLoss + BCE
    - [2, 3, "edt", [2]]            # Channel 2 → EDT: WeightedMSE
```

**Format:** `[start_channel, end_channel, "task_name", [loss_indices]]`

- `start_channel`: Starting channel index (inclusive)
- `end_channel`: Ending channel index (exclusive)
- `task_name`: Name for logging (e.g., "label", "boundary", "edt")
  - Use `"label"` for primary segmentation target (matches ground truth key)
  - Use descriptive names for auxiliary tasks
- `loss_indices`: List of loss function indices to apply (from `loss_functions`)

### 4. Label Transformation

Configure label transforms to generate all required targets using the `targets` format:

```yaml
data:
  label_transform:
    targets:
      - name: binary                # Channel 0: foreground mask
      - name: instance_boundary     # Channel 1: contour map
        kwargs:
          tsz_h: 1                  # Boundary thickness
          do_bg: false              # Don't include background boundaries
          do_convolve: false        # No convolution filtering
      - name: instance_edt          # Channel 2: distance transform
        kwargs:
          mode: "2d"                # 2D EDT computation (per-slice)
          quantize: false           # Continuous distance values
```

## How It Works

### During Training

1. **Model Forward Pass:**
   ```
   Input (B, 1, H, W) → Model → Output (B, 3, H, W)
   ```

2. **Channel Extraction:**
   - Task "binary": extracts channels `[0:1]` from output and labels
   - Task "boundary": extracts channels `[1:2]` from output and labels
   - Task "edt": extracts channels `[2:3]` from output and labels

3. **Loss Computation:**
   ```python
   # Binary task
   binary_output = output[:, 0:1, ...]
   binary_label = label[:, 0:1, ...]
   loss_binary = DiceLoss(binary_output, binary_label) * 1.0 + \
                 BCELoss(binary_output, binary_label) * 0.5
   
   # Boundary task
   boundary_output = output[:, 1:2, ...]
   boundary_label = label[:, 1:2, ...]
   loss_boundary = BCELoss(boundary_output, boundary_label) * 0.5
   
   # EDT task
   edt_output = output[:, 2:3, ...]
   edt_label = label[:, 2:3, ...]
   loss_edt = MSELoss(edt_output, edt_label) * 1.0
   
   # Total
   total_loss = loss_binary + loss_boundary + loss_edt
   ```

4. **Logging:**
   ```
   train_loss_binary_loss0: DiceLoss value
   train_loss_binary_loss1: BCELoss value
   train_loss_binary_total: Sum for binary task
   train_loss_boundary_loss1: BCELoss value
   train_loss_boundary_total: Sum for boundary task
   train_loss_edt_loss2: MSELoss value
   train_loss_edt_total: Sum for EDT task
   train_loss_total: Total loss across all tasks
   ```

## Example Configurations

### Example 1: NucMM Mouse Nucleus Segmentation (Binary + Boundary + EDT)

Complete example from `tutorials/monai_nucmm-z.yaml`:

```yaml
model:
  architecture: monai_unet
  out_channels: 3                      # 3 channels: binary, contour, distance

  # Multi-task loss configuration
  loss_functions: [DiceLoss, BCEWithLogitsLoss, WeightedMSE]
  loss_weights: [1.0, 0.5, 2.0]
  loss_kwargs:
    - {sigmoid: true, smooth_nr: 1e-5, smooth_dr: 1e-5}  # DiceLoss with sigmoid
    - {}                                                  # BCEWithLogitsLoss
    - {tanh: true}                                        # WeightedMSE with tanh for distance

  # Channel → Task → Loss mapping
  multi_task_config:
    - [0, 1, "label", [0, 1]]          # Binary: Dice + BCE
    - [1, 2, "boundary", [0, 1]]       # Boundary: Dice + BCE
    - [2, 3, "edt", [2]]               # Distance: WeightedMSE (with tanh activation)

data:
  label_transform:
    targets:
      - name: binary                    # Channel 0: foreground mask
      - name: instance_boundary         # Channel 1: contour map
        kwargs:
          tsz_h: 1
          do_bg: false
          do_convolve: false
      - name: instance_edt              # Channel 2: distance transform
        kwargs:
          mode: "2d"
          quantize: false
```

### Example 2: SNEMI Instance Segmentation (Binary + Affinity)

```yaml
model:
  out_channels: 4                       # 1 binary + 3 affinity channels
  loss_functions: [DiceLoss, BCEWithLogitsLoss]
  loss_weights: [1.0, 1.0]
  loss_kwargs:
    - {sigmoid: true}
    - {}
  multi_task_config:
    - [0, 1, "label", [0, 1]]           # Binary: Dice + BCE
    - [1, 4, "affinity", [1]]           # Affinity: BCE only

data:
  label_transform:
    targets:
      - name: binary
      - name: affinity
        kwargs:
          offsets: ["1-0-0", "0-1-0", "0-0-1"]
```

### Example 3: Synaptic Polarity Detection

```yaml
model:
  out_channels: 3                       # pre-synaptic, post-synaptic, synapse
  loss_functions: [DiceLoss, BCEWithLogitsLoss]
  loss_weights: [1.0, 1.0]
  multi_task_config:
    - [0, 3, "polarity", [0, 1]]        # All 3 channels: Dice + BCE

data:
  label_transform:
    targets:
      - name: polarity
        kwargs:
          exclusive: false              # Non-exclusive multi-label (for BCE)
```

### Example 4: Single Task (No Multi-Task)

If `multi_task_config` is not specified or is `null`, the system applies all losses to all output channels (standard single-task mode):

```yaml
model:
  out_channels: 2
  loss_functions: [DiceLoss, CrossEntropyLoss]
  loss_weights: [1.0, 0.5]
  # No multi_task_config → applies both losses to all 2 channels

data:
  label_transform:
    targets:
      - name: binary
```

### Example 5: No Label Transformation (Use Raw Labels)

For pre-processed labels or when you want to use raw labels directly:

```yaml
model:
  out_channels: 2                      # Must match label channels
  loss_functions: [DiceLoss, CrossEntropyLoss]
  loss_weights: [1.0, 0.5]

data:
  label_transform:
    targets: null                      # No transformation - identity transform
    # Or: targets: []                 # Empty list also works
```

**Use cases:**
- Labels are already in the correct format (e.g., pre-computed during data preparation)
- Multi-class segmentation with one-hot encoded labels already prepared
- When you want to handle label processing in custom dataset code
- Debugging: verify raw label loading without transformation pipeline

## Available Task Types

The `targets:` list supports the following task names (see [monai_transforms.py:474-484](connectomics/data/process/monai_transforms.py#L474-L484) for the registry):

| Task Name | Description | Default Parameters | Output Shape |
|-----------|-------------|-------------------|--------------|
| `binary` | Binary foreground mask | `segment_id: []` (all non-zero) | [1, D, H, W] |
| `affinity` | Affinity maps for connectivity | `offsets: ['1-1-0', '1-0-0', '0-1-0', '0-0-1']` | [N_offsets, D, H, W] |
| `instance_boundary` | Instance boundary detection | `thickness: 1, do_bg_edges: False, mode: "3d"` | [1, D, H, W] |
| `instance_edt` | Instance Euclidean Distance Transform | `mode: "2d", quantize: False` | [1, D, H, W] |
| `semantic_edt` | Semantic EDT with foreground/background | `mode: "2d", alpha_fore: 8.0, alpha_back: 50.0` | [2, D, H, W] |
| `polarity` | Synaptic polarity (pre/post/synapse) | `exclusive: False` | [3, D, H, W] or [1, D, H, W] |
| `small_object` | Small object detection mask | `threshold: 100` | [1, D, H, W] |
| `energy_quantize` | Quantized energy levels | `levels: 10` | [levels, D, H, W] |
| `decode_quantize` | Decode quantized values | `mode: "max"` | [1, D, H, W] |

### Customizing Task Parameters

All tasks accept a `kwargs` dict to override defaults:

```yaml
targets:
  - name: binary
    kwargs:
      segment_id: [1, 2, 3]        # Only these segment IDs as foreground

  - name: instance_boundary
    kwargs:
      thickness: 2                  # Thicker boundaries
      do_bg_edges: true            # Include background boundaries
      mode: "3d"                   # Use 3D boundary detection (default)

  - name: affinity
    kwargs:
      offsets: ["1-0-0", "0-1-0", "0-0-1", "9-0-0", "0-9-0", "0-0-9"]  # Short + long range
```

### Instance Boundary: 2D vs 3D Mode

The `instance_boundary` task supports two modes, with different algorithms depending on the `do_bg_edges` setting:

#### Mode Selection

**3D Mode (Recommended for Isotropic Data):**
```yaml
- name: instance_boundary
  kwargs:
    thickness: 1
    do_bg_edges: false           # Instance-only boundaries
    mode: "3d"                   # Uses 3D max/min filters (default)
```

**Algorithm:**
- `do_bg_edges=True`: 3D Separable Sobel (gradient-based edge detection)
- `do_bg_edges=False`: Grey dilation/erosion (optimized single-pass, ~30-40% faster)

**Benefits:**
- ✅ Detects boundaries in all 3 dimensions (X, Y, Z)
- ✅ Better for isotropic voxel spacing (e.g., 5×5×5 nm)
- ✅ Captures full 3D topology of objects
- ✅ Efficient algorithms (~2-3x slower than 2D)

**2D Mode (For Anisotropic Data):**
```yaml
- name: instance_boundary
  kwargs:
    thickness: 1
    do_bg_edges: false
    mode: "2d"                    # Slice-by-slice processing
```

**Algorithm:**
- `do_bg_edges=True`: 2D Sobel per slice (gradient-based edge detection)
- `do_bg_edges=False`: Grey dilation/erosion per slice (optimized, consistent with 3D)

**Use cases:**
- Anisotropic voxel spacing (e.g., 30×5×5 nm)
- Legacy compatibility
- Memory constraints

**Performance Comparison:**
| Mode | do_bg_edges=True | do_bg_edges=False | Best For |
|------|------------------|-------------------|----------|
| 3D | Separable Sobel (~1.0s) | Grey dilation/erosion (~0.27s) | Isotropic data (Lucchi: 5×5×5 nm) |
| 2D | 2D Sobel per slice (~0.43s) | Grey dilation/erosion (~0.15s) | Anisotropic data (SNEMI: 30×5×5 nm) |

*Timings shown for 128×256×256 volume. Instance-only mode is ~4x faster than all-edges mode.*

## Benefits

1. **Shared Representations:** The model learns shared features useful for all tasks
2. **Improved Generalization:** Multi-task learning often improves performance on each individual task
3. **Efficiency:** One model instead of multiple separate models
4. **Flexible Loss Combinations:** Each task can use different losses suited to its characteristics
5. **Simplified Pipeline:** Single unified `targets:` format replaces multiple specialized pipelines

## Loss Function Selection Guide

| Task Type | Recommended Losses | Notes |
|-----------|-------------------|-------|
| Binary Segmentation | DiceLoss + BCE | Dice for overlap, BCE for pixel accuracy |
| Boundary Detection | DiceLoss + BCE | Same as binary, handles class imbalance |
| Distance Transform | WeightedMSE or MSE | Regression task, weighted version helps with outliers |
| Affinity Maps | BCE or WeightedBCE | Often needs rebalancing due to connectivity sparsity |
| Multi-Class Polarity | CrossEntropy + Dice | CE for classification, Dice for overlap |

## Migrating from Legacy Formats

### Old Format (Deprecated)

```yaml
# OLD - Do not use
data:
  label_transform:
    binary:
      use_binary: true
    boundary:
      use_boundary: true
    edt:
      distance_transform: true
```

### New Format (Current)

```yaml
# NEW - Use this format
data:
  label_transform:
    targets:
      - name: binary
      - name: instance_boundary
        kwargs:
          tsz_h: 1
          do_bg: false
          do_convolve: false
      - name: instance_edt
        kwargs:
          mode: "2d"
          quantize: false
```

### Migration Benefits

1. ✅ **Explicit**: Each target is clearly listed with its parameters
2. ✅ **Flexible**: Easy to add/remove/reorder tasks
3. ✅ **Composable**: Natural MONAI transform composition
4. ✅ **Extensible**: Simple to add new task types
5. ✅ **Type-safe**: Better validation and error messages

## Troubleshooting

### NaN/Inf in Loss

If you encounter NaN or Inf values:
1. Check that output channels match label channels for each task
2. Verify loss functions are appropriate for the target type
3. Use `monitor.detect_anomaly: true` for debugging
4. Check label value ranges (EDT should be normalized if using MSE)
5. Ensure `loss_kwargs` are appropriate (e.g., `sigmoid: true` for DiceLoss)

**Example fix for NaN in DiceLoss:**
```yaml
loss_kwargs:
  - {sigmoid: true, smooth_nr: 1e-5, smooth_dr: 1e-5}  # Add smoothing
```

### Imbalanced Tasks

If one task dominates the training:
1. Adjust `loss_weights` to balance task contributions
2. Monitor individual task losses: `train_loss_<task>_total`
3. Consider different loss functions (e.g., FocalLoss for imbalanced binary)
4. Check if EDT values need scaling (use WeightedMSE with higher weight)

**Example from NucMM config:**
```yaml
loss_weights: [1.0, 0.5, 5.0]  # Binary: 1.0, Boundary: 0.5, EDT: 5.0
```

### Channel Mismatch Errors

Ensure:
- `model.out_channels` = sum of all task output channels
- Label transforms generate correct number of channels
- `multi_task_config` covers all channels exactly once (no gaps or overlaps)

**Example for 3-channel output:**
```yaml
model:
  out_channels: 3

  multi_task_config:
    - [0, 1, "label", [0, 1]]       # Channel 0
    - [1, 2, "boundary", [0, 1]]    # Channel 1
    - [2, 3, "edt", [2]]            # Channel 2
    # ✅ Covers [0, 3) completely

data:
  label_transform:
    targets:
      - name: binary                 # Produces 1 channel
      - name: instance_boundary      # Produces 1 channel
      - name: instance_edt           # Produces 1 channel
    # ✅ Total: 3 channels
```

### Task Name Convention

For multi-task configs, use consistent naming:
- **Primary task**: Use `"label"` as the task name (matches ground truth key)
- **Auxiliary tasks**: Use descriptive names (`"boundary"`, `"edt"`, `"affinity"`)

This ensures proper logging and metric computation:
```
train_loss_label_total: 1.234      # Primary segmentation loss
train_loss_boundary_total: 0.456   # Boundary auxiliary loss
train_loss_edt_total: 0.789        # EDT auxiliary loss
```

## Advanced Usage

### Per-Task Output Keys

By default, tasks are stacked into a single tensor. To write each task to a separate key:

```yaml
data:
  label_transform:
    stack_outputs: false              # Don't stack, use separate keys
    output_key_format: "{key}_{task}" # Format: label_binary, label_boundary, etc.
    targets:
      - name: binary
        output_key: "binary_mask"     # Override: write to this specific key
      - name: instance_boundary
        # Uses format: label_instance_boundary
      - name: instance_edt
        # Uses format: label_instance_edt
```

### Retaining Original Labels

Keep the original segmentation alongside generated targets:

```yaml
data:
  label_transform:
    retain_original: true             # Keep original label
    targets:
      - name: binary
      - name: instance_boundary
      - name: instance_edt
```

Result: `data["label"]` contains [3, D, H, W] stacked targets, and `data["label_original"]` contains the original instance segmentation.

### Custom Segment Selection

Generate binary masks for specific segment IDs:

```yaml
targets:
  - name: binary
    kwargs:
      segment_id: [1, 2, 3]  # Only segments 1, 2, 3 as foreground
                             # Empty list [] means all non-zero
```

## Summary

Multi-task learning in PyTorch Connectomics uses:

1. **Unified config format**: `label_transform.targets` with task list
2. **Flexible loss assignment**: Map channel ranges to loss functions
3. **Automatic target generation**: `MultiTaskLabelTransformd` creates all targets
4. **Shared model backbone**: One model learns all tasks simultaneously
5. **Detailed logging**: Per-task and per-loss metrics

**Key principle**: Define what targets to generate (`targets:`) and how to train on them (`multi_task_config`), everything else is handled automatically.


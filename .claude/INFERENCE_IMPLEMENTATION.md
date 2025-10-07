# Inference with Blending - Implementation Guide

## Understanding Legacy Position Calculation

From `dataset_volume.py` in v1:

```python
def _get_pos_test(self, index):
    """
    Convert linear dataset index to 4D position (dataset_id, z, y, x).

    Key concepts:
    - sample_size: Grid dimensions for each volume [num_z, num_y, num_x]
    - sample_stride: Spacing between patch centers
    - sample_num_c: Cumulative sample counts [0, vol1_samples, vol1+vol2_samples, ...]
    """
    pos = [0, 0, 0, 0]

    # 1. Find which dataset/volume this index belongs to
    did = self._index_to_dataset(index)  # Binary search in sample_num_c
    pos[0] = did

    # 2. Convert to local index within that dataset
    index2 = index - self.sample_num_c[did]

    # 3. Convert linear index to 3D grid position
    pos[1:] = self._index_to_location(index2, self.sample_size_test[did])
    # sample_size_test[did] = [y*x, x] for fast indexing
    # Returns [z_idx, y_idx, x_idx]

    # 4. Convert grid indices to actual coordinates
    for i in range(1, 4):
        if pos[i] != self.sample_size[pos[0]][i-1] - 1:
            # Normal case: multiply by stride
            pos[i] = int(pos[i] * self.sample_stride[i-1])
        else:
            # Boundary case: tuck in to ensure patch fits
            pos[i] = int(self.volume_size[pos[0]][i-1] -
                        self.sample_volume_size[i-1])

    return pos  # [dataset_id, z_start, y_start, x_start]
```

### Grid Calculation

```python
def count_volume(data_sz, vol_sz, stride):
    """
    Calculate number of patches needed to cover volume with stride.

    Formula: 1 + ceil((data_size - patch_size) / stride)

    Examples:
    - data_sz=256, vol_sz=128, stride=64 -> 1 + ceil((256-128)/64) = 1 + 2 = 3 patches
    - data_sz=256, vol_sz=128, stride=128 -> 1 + ceil((256-128)/128) = 1 + 1 = 2 patches
    """
    return 1 + np.ceil((data_sz - vol_sz) / stride.astype(float)).astype(int)
```

## Implementation for PyTorch Lightning

### Step 1: Add Helper Functions to Dataset

```python
# In connectomics/data/utils/sampling.py or similar

def calculate_inference_grid(
    volume_shape: Tuple[int, int, int],
    patch_size: Tuple[int, int, int],
    stride: Tuple[int, int, int]
) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    """
    Calculate grid of patch positions for inference.

    Args:
        volume_shape: (D, H, W) of input volume
        patch_size: (D, H, W) of each patch
        stride: (D, H, W) stride between patches

    Returns:
        positions: Array of (z, y, x) positions for each patch
        grid_shape: (num_z, num_y, num_x) grid dimensions
    """
    grid_shape = tuple(
        1 + int(np.ceil((vol_sz - patch_sz) / stride_val))
        for vol_sz, patch_sz, stride_val in zip(volume_shape, patch_size, stride)
    )

    positions = []
    for z_idx in range(grid_shape[0]):
        for y_idx in range(grid_shape[1]):
            for x_idx in range(grid_shape[2]):
                # Calculate position with boundary handling
                z = z_idx * stride[0] if z_idx < grid_shape[0] - 1 else volume_shape[0] - patch_size[0]
                y = y_idx * stride[1] if y_idx < grid_shape[1] - 1 else volume_shape[1] - patch_size[1]
                x = x_idx * stride[2] if x_idx < grid_shape[2] - 1 else volume_shape[2] - patch_size[2]
                positions.append([z, y, x])

    return np.array(positions), grid_shape
```

### Step 2: Modify Dataset for Inference Mode

```python
# In connectomics/data/dataset/dataset_volume.py

class MonaiVolumeDataset:
    def __init__(self, ..., mode='train', stride=None):
        ...
        if mode == 'test' and stride is not None:
            self.inference_positions = self._calculate_inference_grid()

    def _calculate_inference_grid(self):
        """Calculate all patch positions for inference."""
        all_positions = []
        for vol_id, vol_shape in enumerate(self.volume_shapes):
            positions, _ = calculate_inference_grid(
                vol_shape,
                self.patch_size,
                self.stride
            )
            # Add volume_id to each position
            positions_with_id = np.concatenate([
                np.full((len(positions), 1), vol_id),
                positions
            ], axis=1)
            all_positions.append(positions_with_id)

        return np.concatenate(all_positions, axis=0)

    def __len__(self):
        if self.mode == 'test':
            return len(self.inference_positions)
        return super().__len__()

    def __getitem__(self, idx):
        if self.mode == 'test':
            pos = self.inference_positions[idx]
            vol_id = int(pos[0])
            z, y, x = int(pos[1]), int(pos[2]), int(pos[3])

            # Load patch
            patch = self._load_patch(vol_id, z, y, x)

            # Return with position metadata
            return {
                'image': patch,
                'pos': pos,  # [vol_id, z, y, x]
            }
        else:
            return super().__getitem__(idx)
```

### Step 3: Add Blending Logic to LightningModule

```python
# In connectomics/lightning/lit_model.py

class ConnectomicsModule(pl.LightningModule):

    def on_test_start(self):
        """Initialize buffers for blending."""
        super().on_test_start()

        # Get config
        if not hasattr(self.cfg, 'inference'):
            return

        # Initialize blending matrix
        from connectomics.data.process.blend import build_blending_matrix

        output_size = tuple(self.cfg.model.output_size)
        self.blending_matrix = build_blending_matrix(
            output_size,
            mode=self.cfg.inference.blending
        )

        # Initialize result buffers (will be created dynamically per volume)
        self.inference_results = {}  # vol_id -> result array
        self.inference_weights = {}  # vol_id -> weight array

    def test_step(self, batch, batch_idx):
        """Test step with optional blending accumulation."""
        # Standard metric computation
        images = batch['image']
        outputs = self(images)

        # If we have position metadata, accumulate for blending
        if 'pos' in batch and hasattr(self, 'blending_matrix'):
            self._accumulate_predictions(outputs, batch['pos'])

        # Compute metrics if labels available
        if 'label' in batch:
            labels = batch['label']
            # ... existing metric computation ...

        return outputs

    def _accumulate_predictions(self, outputs, positions):
        """Accumulate predictions with blending weights."""
        batch_size = outputs.shape[0]

        for i in range(batch_size):
            pos = positions[i]  # [vol_id, z, y, x]
            vol_id = int(pos[0])
            z, y, x = int(pos[1]), int(pos[2]), int(pos[3])

            # Initialize buffers for this volume if needed
            if vol_id not in self.inference_results:
                vol_shape = self._get_volume_shape(vol_id)
                num_channels = outputs.shape[1]
                self.inference_results[vol_id] = np.zeros(
                    [num_channels] + list(vol_shape),
                    dtype=np.float32
                )
                self.inference_weights[vol_id] = np.zeros(
                    vol_shape,
                    dtype=np.float32
                )

            # Get prediction and convert to numpy
            pred = outputs[i].detach().cpu().numpy()

            # Get patch shape
            _, d, h, w = pred.shape

            # Accumulate with blending
            self.inference_results[vol_id][
                :, z:z+d, y:y+h, x:x+w
            ] += pred * self.blending_matrix[np.newaxis, :]

            self.inference_weights[vol_id][
                z:z+d, y:y+h, x:x+w
            ] += self.blending_matrix

    def on_test_epoch_end(self):
        """Normalize and save results."""
        if not hasattr(self, 'inference_results'):
            return

        # Normalize by weights
        for vol_id in self.inference_results.keys():
            # Expand weights for broadcasting
            weights = self.inference_weights[vol_id][np.newaxis, :]
            self.inference_results[vol_id] /= (weights + 1e-8)

            # Convert to uint8 if needed
            result = self.inference_results[vol_id]
            result = np.clip(result * 255, 0, 255).astype(np.uint8)

            # Save result
            self._save_volume(vol_id, result)

        # Cleanup
        del self.inference_results
        del self.inference_weights
```

### Step 4: Update Config

Already done - config has:
- `inference.blending` - 'gaussian' or 'bump'
- `inference.stride` - stride for patch sampling
- `inference.output_scale` - output scaling factor

## Testing

```python
# In scripts/main.py for test mode

# Create inference dataset with stride
test_dataset = create_volume_dataset(
    cfg,
    mode='test',
    stride=cfg.inference.stride,  # Enable grid-based sampling
)

# Run test with blending
trainer.test(model, test_dataloader)

# Results will be saved automatically in on_test_epoch_end
```

## Summary

**What this implementation does:**

1. ✅ **Position tracking**: Dataset returns `pos` = [vol_id, z, y, x] for each patch
2. ✅ **Grid-based sampling**: Patches cover full volume with stride/overlap
3. ✅ **Blending accumulation**: `test_step` accumulates weighted predictions
4. ✅ **Normalization**: `on_test_epoch_end` divides by weight sum
5. ✅ **Volume saving**: Save reconstructed volumes to disk

**Blending formula:**
```
result[z:z+d, y:y+h, x:x+w] += prediction * blending_weights
weight[z:z+d, y:y+h, x:x+w] += blending_weights
final_result = result / weight
```

This matches the legacy v1 implementation exactly.

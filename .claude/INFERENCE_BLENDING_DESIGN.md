# Inference Blending Design

## Current Status

The current PyTorch Lightning implementation (`test_step` in `lit_model.py`) performs batch-by-batch inference without:
- Spatial position tracking
- Patch blending/stitching
- Full volume reconstruction

This means patches are evaluated independently, which is fine for metrics but doesn't produce a full reconstructed volume output.

## Legacy Implementation (v1)

The legacy `trainer.py` `test()` method implements proper blending:

```python
def test(self):
    # Build blending matrix (Gaussian or bump)
    ww = build_blending_matrix(spatial_size, self.cfg.INFERENCE.BLENDING)

    # Initialize result and weight arrays
    result = [np.zeros(output_size) for each volume]
    weight = [np.zeros(output_size) for each volume]

    # Process each batch
    for sample in dataloader:
        pos, volume = sample.pos, sample.out_input  # Position metadata
        output = model(volume)

        # Accumulate with blending weights
        for idx in range(output.shape[0]):
            st = pos[idx]  # Start position (vol_id, z, y, x)
            result[st[0]][:, st[1]:st[1]+sz[1], ...] += out_block * ww
            weight[st[0]][st[1]:st[1]+sz[1], ...] += ww

    # Normalize by accumulated weights
    for vol_id in range(len(result)):
        result[vol_id] /= weight[vol_id]
```

**Key Components:**
1. **Blending matrix** (`ww`): Pre-computed Gaussian/bump weights for patch size
2. **Position tracking** (`pos`): Each patch knows its (volume_id, z, y, x) location
3. **Weight accumulation**: Track total weight at each voxel for averaging
4. **Result accumulation**: Sum weighted predictions
5. **Normalization**: Divide by total weights to get final result

## Required Changes for Lightning Implementation

### 1. ✅ Blending Utilities (DONE)
- Located at: `connectomics/data/process/blend.py`
- Functions: `build_blending_matrix()`, `blend_gaussian()`, `blend_bump()`

### 2. ✅ Config Updates (DONE)
- Added to `InferenceConfig`:
  - `blending: str = "gaussian"` - Blending mode
  - `do_eval: bool = True` - Eval vs train mode
  - `output_scale: List[float]` - Output scaling factor

### 3. ⚠️ Dataset Changes (TODO)
**Option A: Modify existing dataset to return position metadata**
- Add `return_pos=True` flag to dataset
- Return dict with `{'image': ..., 'label': ..., 'pos': (vol_id, z, y, x)}`

**Option B: Create specialized inference dataset**
- `InferenceVolumeDataset` that:
  - Loads full volume(s)
  - Generates patches with stride/overlap
  - Returns position metadata
  - No random sampling

### 4. ⚠️ Lightning Module Changes (TODO)
**Option A: Override `test_step` + `on_test_epoch_end`**
```python
def on_test_start(self):
    # Initialize result and weight buffers
    self.test_results = []
    self.test_weights = []
    self.blending_matrix = build_blending_matrix(...)

def test_step(self, batch, batch_idx):
    images, labels, pos = batch['image'], batch['label'], batch['pos']
    outputs = self(images)

    # Accumulate results with blending
    for i in range(len(pos)):
        self._accumulate_prediction(outputs[i], pos[i])

    # Still compute metrics on patches
    return metrics

def on_test_epoch_end(self):
    # Normalize accumulated results
    for vol_id in range(len(self.test_results)):
        self.test_results[vol_id] /= self.test_weights[vol_id]

    # Save results to disk
    self._save_results()
```

**Option B: Separate inference function**
- Keep `test_step` for metrics only
- Add `inference()` method that handles full volume reconstruction
- Called separately from `trainer.test()`

### 5. ⚠️ Main Script Changes (TODO)
Update `scripts/main.py` to:
- Use inference dataset for test mode
- Call volume reconstruction after metrics
- Save reconstructed volumes

## Recommended Approach

**Phase 1: Minimal Implementation (Quick Win)**
- Modify `test_step` to accumulate predictions with blending
- Store results in module state
- Save in `on_test_epoch_end`
- Requires position metadata from dataset

**Phase 2: Full Implementation**
- Create dedicated `InferenceDataset` with stride-based sampling
- Add MONAI-style sliding window inference support
- Support test-time augmentation
- Add tensorstore/zarr support for large volumes

## Implementation Priority

1. ✅ Config support (DONE)
2. ✅ Blending utilities (DONE)
3. 🔄 Add position metadata to dataset
4. 🔄 Implement accumulation in `test_step`
5. 🔄 Save reconstructed volumes
6. ⏸️ Test-time augmentation (future)
7. ⏸️ Tensorstore integration (future)

## Notes

- Current `test_step` computes metrics correctly on patches
- Blending is only needed for full volume reconstruction
- Can keep metrics separate from volume reconstruction
- Consider memory constraints for large volumes (incremental saving)

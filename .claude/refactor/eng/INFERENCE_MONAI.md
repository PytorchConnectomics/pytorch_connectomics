# Inference with Blending - Implementation Guide

## Legacy Reference (v1)

The v1 dataset emitted explicit patch coordinates and blended the resulting
predictions back into a full-volume array. The helper below illustrates how the
position grid used to be generated:

```python
def _get_pos_test(self, index):
    # Returns [dataset_id, z_start, y_start, x_start]
    ...  # Converts flattened index into a 3D grid position
```

The historical approach is still a useful reference when validating coverage
and patch ordering, but we no longer need to re-implement that logic.

## Design Goals

- Reuse MONAI's battle-tested sliding window inferer instead of maintaining a
  custom grid/blending stack.
- Preserve the ability to control overlap, weighting mode, and stride through
  `cfg.inference`.
- Keep dataset code focused on loading volumes and meta-data; inference assembly
  happens inside the Lightning module.

## Leveraging MONAI Sliding Window

### 1. Dataset

Ensure the MONAI dataset returns the entire volume tensor together with the
usual meta data produced by `LoadImaged`. No positional bookkeeping is needed.

```python
from monai.transforms import Compose, LoadImaged, EnsureTyped, AddChanneld

transforms = Compose([
    LoadImaged(keys=("image", "label"), image_only=False),
    AddChanneld(keys=("image", "label")),
    EnsureTyped(keys=("image", "label")),
])
```

### 2. Lightning Module Setup

Instantiate a `SlidingWindowInferer` (or use the functional
`sliding_window_inference`) during module construction so configuration stays in
one place:

```python
from monai.inferers import SlidingWindowInferer

class ConnectomicsModule(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.inferer = SlidingWindowInferer(
            roi_size=cfg.inference.window_size,
            sw_batch_size=cfg.inference.sw_batch_size,
            overlap=cfg.inference.overlap,
            mode=cfg.inference.blending,       # e.g. "gaussian", "constant"
            padding_mode=cfg.inference.padding_mode,
        )
```

### 3. Running Inference

During `validation_step`/`test_step`, call the inferer with the network and the
incoming volume. MONAI performs the patch extraction, network calls, weighting,
and stitching automatically.

```python
def test_step(self, batch, batch_idx):
    inputs = batch["image"].to(self.device)
    logits = self.inferer(inputs=inputs, network=self)

    if "label" in batch:
        self.log_metrics(logits, batch["label"].to(self.device))

    self._write_outputs(logits, batch)
    return logits
```

`SlidingWindowInferer` already applies gaussian blending when `mode="gaussian"`
and supports anisotropic patch sizes as long as `roi_size` matches the expected
input shape of the network.

### 4. Saving Results

Use MONAI's meta dictionary to look up the original volume shape and file name.
The inferer preserves spatial size, so the output can be saved directly without
manual weight normalization.

```python
def _write_outputs(self, logits, batch):
    meta = batch["image_meta_dict"]
    for vol_idx in range(logits.shape[0]):
        filename = meta["filename_or_obj"][vol_idx]
        result = logits[vol_idx].detach().cpu()
        self._save_volume(filename, result)
```

If additional post-processing is required (thresholding, dtype conversion),
apply it before writing the volume to disk.

## Configuration Hooks

Keep the configuration keys that previously drove the custom sampler and map
them to the inferer parameters:

- `cfg.inference.window_size` -> `roi_size`
- `cfg.inference.stride` -> `overlap = 1 - stride / window_size`
- `cfg.inference.blending` -> `mode`
- `cfg.inference.sw_batch_size` -> patch mini-batch size (controls memory usage)
- `cfg.inference.padding_mode` (optional) -> pad behavior at borders

For convenience, add a helper to translate stride into overlap. Example:

```python
def compute_overlap(window_size, stride):
    return tuple(1.0 - (s / w) for w, s in zip(window_size, stride))
```

## Testing

1. Enable test mode with the target configuration:
   ```python
   test_dataset = create_volume_dataset(cfg, mode="test")
   test_loader = DataLoader(test_dataset, batch_size=1)
   trainer.test(model, test_loader)
   ```
2. Compare reconstructed volumes against the legacy pipeline on a small sample
   to verify numerical parity.
3. Stress-test with large volumes to confirm memory usage and runtime match
   expectations (adjust `sw_batch_size` if necessary).

This approach offloads the sliding window mechanics to MONAI while keeping the
rest of the pipeline unchanged.




Implementation Summary

Added window_size, sw_batch_size, and padding_mode to the inference config so MONAI’s inferer can be configured directly from Hydra (connectomics/config/hydra_config.py:403).
Let test datasets hand back full volumes and attach MONAI-style metadata during loading to support sliding-window stitching and file naming (connectomics/data/dataset/dataset_volume.py:165, connectomics/data/io/monai_transforms.py:52).
Replaced the custom blending pipeline with MONAI’s SlidingWindowInferer, including ROI/overlap resolution, per-volume saving via metadata, and a streamlined test step that still reuses the existing loss/metric logic (connectomics/lightning/lit_model.py:121, connectomics/lightning/lit_model.py:236, connectomics/lightning/lit_model.py:444).
Tests

python -m compileall connectomics/lightning/lit_model.py connectomics/data/io/monai_transforms.py
Next Steps

Set inference.window_size (if it differs from the model output size) and run trainer.test(...) to confirm stitched predictions look correct.
Inspect the saved HDF5 volumes to verify scaling/format; tweak inference.output_scale if downstream tooling expects a different range.
# Visualizer Migration Guide

## Overview

The visualizer has been updated from legacy YACS config to work seamlessly with PyTorch Lightning and Hydra configs.

## What Changed

### Before (Legacy)
```python
from connectomics.utils.visualizer import Visualizer

# Required YACS config with specific fields
viz = Visualizer(cfg, vis_opt=0, N=16)

# Manual visualization calls
viz.visualize(volume, label, output, weight, iter_total, writer)
```

### After (Lightning-Compatible)
```python
from connectomics.utils.visualizer import Visualizer
from connectomics.lightning import VisualizationCallback

# Simple standalone visualizer
viz = Visualizer(cfg, max_images=16)
viz.visualize(volume, label, output, iteration, writer, prefix='train')

# Or use as Lightning callback (recommended)
callback = VisualizationCallback(cfg, max_images=8, log_every_n_steps=100)
trainer = Trainer(callbacks=[callback])
```

## New Features

### 1. Automatic 3D → 2D Conversion
```python
# Handles both 2D and 3D data automatically
volume = torch.randn(2, 1, 64, 128, 128)  # 3D: (B, C, D, H, W)
# Automatically takes middle slice for visualization

volume = torch.randn(2, 1, 128, 128)  # 2D: (B, C, H, W)
# Works as-is
```

### 2. Simplified API
```python
# No more weight maps or split activations required
viz.visualize(
    volume=input_image,      # Input
    label=ground_truth,       # Target
    output=prediction,        # Model output
    iteration=step,          # Current step
    writer=tensorboard,      # TensorBoard writer
    prefix='train'           # Log prefix
)
```

### 3. Lightning Callback Integration
```python
from connectomics.lightning import create_callbacks

# Automatically creates visualization callback
callbacks = create_callbacks(cfg)

trainer = Trainer(callbacks=callbacks)
trainer.fit(model, datamodule)
```

## Usage Examples

### Standalone Usage
```python
from torch.utils.tensorboard import SummaryWriter
from connectomics.utils.visualizer import Visualizer

# Create visualizer
viz = Visualizer(max_images=8)

# During training
writer = SummaryWriter('logs')
for epoch in range(epochs):
    for batch_idx, batch in enumerate(dataloader):
        # ... training step ...

        if batch_idx % 100 == 0:
            viz.visualize(
                volume=batch['image'],
                label=batch['label'],
                output=model_output,
                iteration=global_step,
                writer=writer,
                prefix='train'
            )
```

### Lightning Callback Usage
```python
from connectomics.lightning import ConnectomicsModule, VisualizationCallback
from pytorch_lightning import Trainer

# Create callback
vis_callback = VisualizationCallback(
    cfg,
    max_images=8,
    log_every_n_steps=100
)

# Create trainer with callback
trainer = Trainer(
    max_epochs=100,
    callbacks=[vis_callback],
    logger=TensorBoardLogger('logs')
)

# Train (visualization happens automatically)
trainer.fit(model, datamodule)
```

### Consecutive Slices Visualization
```python
# Visualize multiple consecutive slices from 3D volume
viz.visualize_consecutive_slices(
    volume=volume_3d,         # (B, C, D, H, W)
    label=label_3d,
    output=prediction_3d,
    writer=writer,
    iteration=step,
    prefix='train',
    num_slices=8              # Show 8 consecutive slices
)
```

## Configuration

### Config File (Hydra)
```yaml
# tutorials/your_config.yaml

visualization:
  enabled: true
  max_images: 8
  log_every_n_steps: 100

# Visualization config is optional
# Default values used if not specified
```

### Programmatic Configuration
```python
from connectomics.config import from_dict

cfg = from_dict({
    'visualization': {
        'enabled': True,
        'max_images': 8,
        'log_every_n_steps': 100
    }
})
```

## Migration Checklist

### For Existing Code

1. ✅ **Remove legacy dependencies**
   ```python
   # Remove
   from ..transforms.process import decode_quantize, dx_to_circ
   from connectomics.models.utils import SplitActivation

   # Not needed anymore
   ```

2. ✅ **Update visualizer instantiation**
   ```python
   # Before
   viz = Visualizer(cfg, vis_opt=0, N=16)

   # After
   viz = Visualizer(cfg, max_images=16)
   # or
   viz = Visualizer(max_images=16)  # cfg optional
   ```

3. ✅ **Update visualize() calls**
   ```python
   # Before
   viz.visualize(volume, label, output, weight, iter_total, writer, suffix='train')

   # After
   viz.visualize(volume, label, output, iter_total, writer, prefix='train')
   # Note: weight parameter removed
   ```

4. ✅ **Add Lightning callbacks** (if using Lightning)
   ```python
   from connectomics.lightning import VisualizationCallback

   trainer = Trainer(
       callbacks=[VisualizationCallback(cfg)]
   )
   ```

## Advanced Features

### Custom Color Maps
```python
viz = Visualizer(max_images=8)

# Add custom semantic segmentation colors
viz.semantic_colors['my_task'] = torch.tensor([
    [0.0, 0.0, 0.0],  # Class 0: black
    [1.0, 0.0, 0.0],  # Class 1: red
    [0.0, 1.0, 0.0],  # Class 2: green
    [0.0, 0.0, 1.0],  # Class 3: blue
])
```

### Conditional Visualization
```python
callback = VisualizationCallback(cfg)

# Only visualize on specific conditions
def should_visualize(trainer, batch_idx):
    return (trainer.global_step % 100 == 0 and
            batch_idx == 0 and
            trainer.current_epoch > 5)
```

## Backward Compatibility

The legacy `create_visualizer()` function is still available:

```python
from connectomics.utils.visualizer import create_visualizer

viz = create_visualizer(cfg, max_images=16)
# Returns Visualizer instance (backward compatible)
```

## TensorBoard Logging

Visualizations are logged to TensorBoard under:
- `train/visualization` - Training visualizations
- `val/visualization` - Validation visualizations
- `train_slices/visualization` - Consecutive slices

View with:
```bash
tensorboard --logdir logs/
```

## Troubleshooting

### Visualization not showing up
1. Check TensorBoard is running: `tensorboard --logdir logs/`
2. Verify logger is TensorBoard: `trainer.logger = TensorBoardLogger(...)`
3. Check visualization frequency: `log_every_n_steps`

### Memory issues with many images
```python
# Reduce max_images
viz = Visualizer(max_images=4)  # Show fewer images
```

### 3D volumes not visualizing correctly
```python
# Use consecutive_slices for better 3D visualization
viz.visualize_consecutive_slices(...)
```

## Summary

✅ **Updated visualizer:**
- Works with both Hydra and YACS configs
- Simplified API (no weight maps required)
- Automatic 3D/2D handling
- Lightning callback support
- TensorBoard integration
- Backward compatible

✅ **New components:**
- `Visualizer` - Core visualization class
- `LightningVisualizer` - Lightning-aware wrapper
- `VisualizationCallback` - Lightning callback
- `create_callbacks()` - Factory for all callbacks

✅ **Migration complete** - Legacy code still works, new code is cleaner!

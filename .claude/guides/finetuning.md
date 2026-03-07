# Fine-tuning Guide

## Two Approaches

### Option 1: `external_weights_path` (Recommended)

Loads only model weights (not optimizer/scheduler). Use when fine-tuning on different data or with new hyperparameters.

```yaml
model:
  architecture: rsunet  # Must match pretrained model
  external_weights_path: "outputs/my_run/checkpoints/last.ckpt"
  external_weights_key_prefix: "model."  # Default for Lightning checkpoints

optimization:
  optimizer:
    lr: 0.0001  # Lower than pretraining
  max_epochs: 500  # Fewer than pretraining

data:
  train_image: "datasets/new-data/train.h5"
  train_label: "datasets/new-data/label.h5"
```

### Option 2: `--checkpoint` with Resets

Loads full checkpoint then resets specific components:

```bash
python scripts/main.py --config tutorials/my_config.yaml \
    --checkpoint outputs/my_run/checkpoints/last.ckpt \
    --reset-optimizer \
    --reset-scheduler \
    --reset-epoch
```

## Key Differences for Fine-tuning Configs

1. **Lower learning rate**: 5-10x smaller than pretraining
2. **Fewer epochs**: Pretrained model needs less training
3. **Shorter warmup/patience**: Already converged features
4. **New data paths**: Different dataset

## Recommendation

Create a separate YAML file for fine-tuning (e.g., `my_config_finetune.yaml`) rather than modifying the original. This keeps pretraining and fine-tuning configs cleanly separated.

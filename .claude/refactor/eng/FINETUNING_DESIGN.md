# Fine-tuning Design Recommendation for hydra-lv.yaml

## Question
Given the pretrained model from `hydra-lv.yaml`, what's the best way to finetune upon it? Should we create a new YAML or add a new section in `hydra-lv.yaml`?

## Recommendation: **Create a New YAML File**

### Why a Separate File?

1. **Separation of Concerns**
   - Original config (`hydra-lv.yaml`) remains focused on pretraining
   - Fine-tuning config (`hydra-lv-finetune.yaml`) is clearly for fine-tuning
   - Easier to maintain and understand

2. **Different Requirements**
   - Fine-tuning typically needs:
     - Different data paths (new dataset)
     - Lower learning rate (10x-100x smaller)
     - Fewer epochs (pretrained model needs less training)
     - Possibly different augmentation strategy
   - These differences are cleaner in a separate file

3. **Follows Existing Patterns**
   - The codebase already has examples like `mito2dsem_nnunet.yaml` that load pretrained models
   - Uses the same `external_weights_path` pattern

4. **Flexibility**
   - Can create multiple fine-tuning configs for different datasets
   - Easy to experiment with different hyperparameters
   - Original config stays untouched

## Implementation Approach

### Option 1: `external_weights_path` (Recommended for Fine-tuning)

**Use when:**
- You want to load only model weights (not optimizer/scheduler)
- You want to use new hyperparameters (lower LR, different scheduler)
- You're fine-tuning on different data

**Configuration:**
```yaml
model:
  architecture: rsunet  # Must match pretrained model
  external_weights_path: "outputs/hydra-lv_rsunet/checkpoints/last.ckpt"
  external_weights_key_prefix: "model."  # Default for Lightning checkpoints
  # ... rest of model config (must match pretrained model)
```

**Usage:**
```bash
python scripts/main.py --config tutorials/hydra-lv-finetune.yaml
```

**How it works:**
- `build_model()` loads weights during model construction
- Only model weights are loaded (not optimizer/scheduler state)
- You can use completely new optimizer/scheduler settings

### Option 2: `--checkpoint` CLI Argument (For Resuming)

**Use when:**
- You want to resume training with the same optimizer/scheduler
- You're continuing training on the same dataset
- You want to preserve training state (epoch, optimizer, scheduler)

**Usage:**
```bash
python scripts/main.py --config tutorials/hydra-lv.yaml \
    --checkpoint outputs/hydra-lv_rsunet/checkpoints/last.ckpt
```

**With resets (for fine-tuning):**
```bash
python scripts/main.py --config tutorials/hydra-lv.yaml \
    --checkpoint outputs/hydra-lv_rsunet/checkpoints/last.ckpt \
    --reset-optimizer \
    --reset-scheduler \
    --reset-epoch
```

**How it works:**
- PyTorch Lightning loads the full checkpoint
- Includes optimizer, scheduler, epoch counter, etc.
- Use `--reset-*` flags to reset specific components

## Comparison

| Aspect | New YAML File | Add Section to Existing |
|--------|---------------|-------------------------|
| **Clarity** | ✅ Clear separation | ❌ Mixed concerns |
| **Maintainability** | ✅ Easy to maintain | ❌ Config becomes complex |
| **Flexibility** | ✅ Multiple fine-tuning configs | ❌ Single config for all |
| **Reusability** | ✅ Can reference original | ❌ Duplicated content |
| **Best Practice** | ✅ Follows codebase patterns | ❌ Not used elsewhere |

## Key Differences in Fine-tuning Config

1. **Model Loading**
   ```yaml
   model:
     external_weights_path: "path/to/pretrained.ckpt"
   ```

2. **Lower Learning Rate**
   ```yaml
   optimization:
     optimizer:
       lr: 0.0001  # 5x lower than pretraining (0.0005)
   ```

3. **Fewer Epochs**
   ```yaml
   optimization:
     max_epochs: 500  # vs 10000 in pretraining
   ```

4. **New Data Paths**
   ```yaml
   data:
     train_image: "datasets/new-data/vol*_im.h5"
     train_label: "datasets/new-data/vol*_vesicle_ins.h5"
   ```

5. **Shorter Warmup/Patience**
   ```yaml
   scheduler:
     warmup_epochs: 50  # vs 200 in pretraining
   early_stopping:
     patience: 50  # vs 100 in pretraining
   ```

## Example: Fine-tuning Workflow

1. **Train pretrained model:**
   ```bash
   python scripts/main.py --config tutorials/hydra-lv.yaml
   ```
   → Saves checkpoint to `outputs/hydra-lv_rsunet/checkpoints/`

2. **Fine-tune on new data:**
   ```bash
   # Update hydra-lv-finetune.yaml with:
   # - external_weights_path pointing to pretrained checkpoint
   # - New data paths
   # - Fine-tuning hyperparameters
   
   python scripts/main.py --config tutorials/hydra-lv-finetune.yaml
   ```

3. **Test fine-tuned model:**
   ```bash
   python scripts/main.py --config tutorials/hydra-lv-finetune.yaml \
       --mode test \
       --checkpoint outputs/hydra-lv_rsunet_finetune/checkpoints/best.ckpt
   ```

## Alternative: Using `--checkpoint` with Resets

If you prefer to use the original config with checkpoint loading:

```bash
# Fine-tune by loading checkpoint but resetting optimizer/scheduler
python scripts/main.py --config tutorials/hydra-lv.yaml \
    --checkpoint outputs/hydra-lv_rsunet/checkpoints/last.ckpt \
    --reset-optimizer \
    --reset-scheduler \
    --reset-epoch \
    optimization.optimizer.lr=0.0001 \
    optimization.max_epochs=500 \
    data.train_image="datasets/new-data/vol*_im.h5" \
    data.train_label="datasets/new-data/vol*_vesicle_ins.h5"
```

**Pros:**
- No new file needed
- Can override config values via CLI

**Cons:**
- Long command line
- Harder to track fine-tuning experiments
- Mixes pretraining and fine-tuning configs

## Conclusion

**Recommended approach:** Create `hydra-lv-finetune.yaml` with `external_weights_path`.

This provides:
- ✅ Clear separation of pretraining vs fine-tuning
- ✅ Easy to maintain and understand
- ✅ Follows existing codebase patterns
- ✅ Flexible for multiple fine-tuning experiments
- ✅ Clean configuration without command-line overrides

The template file `hydra-lv-finetune.yaml` is provided as a starting point.


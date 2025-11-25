# Mode Integration Summary: Train/Test/Tune Unified System

## Quick Reference

### Current Modes (Existing)
```bash
just train <dataset>              # Train model
just resume <dataset> <ckpt>      # Resume training
just test <dataset> <ckpt>        # Test with fixed params
```

### New Modes (Optuna Integration)
```bash
just tune <dataset> <ckpt>        # Tune decoding params
just tune-test <dataset> <ckpt>   # Tune + test (recommended)
just infer <dataset> <ckpt>       # Inference (alias for test)
```

## Mode Comparison Table

| Mode | Purpose | Needs Ground Truth? | Output | Use Case |
|------|---------|-------------------|--------|----------|
| **train** | Train model from scratch | Yes (train+val) | Model checkpoint | Initial model training |
| **resume** | Continue training | Yes (train+val) | Model checkpoint | Resume interrupted training |
| **test** | Inference + evaluation | Yes (test labels) | Segmentation + metrics | Evaluate with known params |
| **predict** | Inference only | No | Segmentation | Production deployment |
| **tune** | Optimize params | Yes (val labels) | Best params + plots | Find optimal parameters |
| **tune+test** | Optimize then evaluate | Yes (val + test labels) | Params + segmentation + metrics | Complete workflow |
| **infer** | Alias for test | Depends | Segmentation ± metrics | Clearer naming |

## Workflow Diagrams

### Traditional Workflow (Before Optuna)
```
┌─────────────┐
│   Train     │  just train lucchi
└──────┬──────┘
       │ (produces checkpoint)
       ↓
┌─────────────┐
│   Test      │  just test lucchi ckpt.pt
│  (fixed     │  (manually set params in config)
│   params)   │
└──────┬──────┘
       │
       ↓
   Results ❌ (params may not be optimal)
```

### New Workflow (With Optuna)
```
┌─────────────┐
│   Train     │  just train lucchi
└──────┬──────┘
       │ (produces checkpoint)
       ↓
┌─────────────────────────────────────────┐
│           Tune + Test                   │
│  just tune-test lucchi ckpt.pt         │
│                                         │
│  1. Tune on validation                 │
│     → Find best params                  │
│                                         │
│  2. Apply to test                       │
│     → Evaluate with best params         │
└──────┬──────────────────────────────────┘
       │
       ↓
   Results ✅ (optimized params, better quality)
```

### Separate Tuning Workflow (For Flexibility)
```
┌─────────────┐
│   Train     │  just train lucchi
└──────┬──────┘
       │
       ↓
┌─────────────┐
│   Tune      │  just tune lucchi ckpt.pt
│  (save best │  → outputs/tuning/best_params.yaml
│   params)   │
└──────┬──────┘
       │
       ↓
┌─────────────┐
│   Test      │  just test lucchi ckpt.pt
│  (load      │  (uses best_params.yaml)
│   params)   │
└──────┬──────┘
       │
       ↓
   Results ✅
```

## Parameter Sources

### Three Ways to Get Parameters

```
┌──────────────────────────────────────────────────────────────┐
│                     Parameter Sources                        │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  1. FIXED (Traditional)                                      │
│     • Manually specified in config                          │
│     • Good for: known optimal values                        │
│     • Example:                                               │
│       inference:                                             │
│         decoding:                                            │
│           parameter_source: fixed                            │
│           fixed_params:                                      │
│             binary_threshold: 0.85                           │
│                                                              │
│  2. TUNED (From File)                                        │
│     • Load from previous tuning                             │
│     • Good for: applying tuned params to new data           │
│     • Example:                                               │
│       inference:                                             │
│         decoding:                                            │
│           parameter_source: tuned                            │
│           tuned_params:                                      │
│             path: "outputs/tuning/best_params.yaml"         │
│                                                              │
│  3. OPTUNA (Automatic)                                       │
│     • Run optimization automatically                         │
│     • Good for: finding optimal params                      │
│     • Example:                                               │
│       inference:                                             │
│         decoding:                                            │
│           parameter_source: optuna                           │
│           optuna:                                            │
│             enabled: true                                    │
│             n_trials: 50                                     │
└──────────────────────────────────────────────────────────────┘
```

## CLI Usage Examples

### Example 1: Train → Tune → Test (Complete Pipeline)
```bash
# Step 1: Train model
just train hydra-lv
# Output: outputs/hydra-lv_rsunet/checkpoints/best.ckpt

# Step 2: Tune parameters on validation
just tune hydra-lv outputs/hydra-lv_rsunet/checkpoints/best.ckpt
# Output: outputs/hydra-lv_rsunet/tuning/best_params.yaml

# Step 3: Test with optimized parameters
just test hydra-lv outputs/hydra-lv_rsunet/checkpoints/best.ckpt
# (Automatically uses best_params.yaml if parameter_source: tuned)
```

### Example 2: One-Shot Tune + Test (Recommended)
```bash
# All-in-one command
just tune-test hydra-lv outputs/hydra-lv_rsunet/checkpoints/best.ckpt

# Equivalent to running:
# 1. just tune hydra-lv ckpt.pt
# 2. just test hydra-lv ckpt.pt (with tuned params)
```

### Example 3: Quick Tuning Test (20 trials)
```bash
# Quick tuning for testing
just tune-quick hydra-lv outputs/checkpoints/best.ckpt

# Or manually specify trials
python scripts/main.py \
  --config tutorials/hydra-lv.yaml \
  --mode tune \
  --checkpoint outputs/checkpoints/best.ckpt \
  --tune-trials 20
```

### Example 4: Use Specific Parameter File
```bash
# Test with specific parameters
just test-with-params hydra-lv checkpoints/best.ckpt params_v2.yaml

# Or override from CLI
python scripts/main.py \
  --config tutorials/hydra-lv.yaml \
  --mode test \
  --checkpoint checkpoints/best.ckpt \
  --params outputs/tuning/best_params.yaml
```

## Config Examples

### Minimal Config (Fixed Parameters)
```yaml
# tutorials/hydra-lv-simple.yaml
model:
  architecture: rsunet
  checkpoint: "checkpoints/best.ckpt"

data:
  test_image: "datasets/hydra/test_image.h5"
  test_label: "datasets/hydra/test_label.h5"

inference:
  decoding:
    parameter_source: fixed
    fixed_params:
      binary_threshold: 0.85
      contour_threshold: 0.95
      distance_threshold: 0.40
      min_instance_size: 32
      min_seed_size: 8
```

Usage:
```bash
just test hydra-lv-simple checkpoints/best.ckpt
```

---

### Full Config (With Tuning Support)
```yaml
# tutorials/hydra-lv-full.yaml
model:
  architecture: rsunet
  checkpoint: "checkpoints/best.ckpt"

data:
  # Validation (for tuning)
  val_image: "datasets/hydra/val_image.h5"
  val_label: "datasets/hydra/val_label.h5"

  # Test (for final evaluation)
  test_image: "datasets/hydra/test_image.h5"
  test_label: "datasets/hydra/test_label.h5"

inference:
  decoding:
    # Switch this based on mode
    parameter_source: optuna  # fixed | tuned | optuna

    # Fixed parameters (for --mode test with parameter_source: fixed)
    fixed_params:
      binary_threshold: 0.85
      contour_threshold: 0.95
      distance_threshold: 0.40
      min_instance_size: 32
      min_seed_size: 8

    # Tuned parameters (for --mode test with parameter_source: tuned)
    tuned_params:
      path: "outputs/tuning/best_params.yaml"

    # Optuna tuning (for --mode tune or --mode tune+test)
    optuna:
      enabled: true
      tune_on_data: validation
      n_trials: 50

      optimization:
        mode: single
        single_objective:
          metric: adapted_rand
          direction: maximize

      parameter_space:
        binary_threshold:
          type: float
          range: [0.5, 0.95]
          step: 0.05

        contour_threshold:
          type: float
          range: [0.6, 1.2]
          step: 0.05

        distance_threshold:
          type: float
          range: [0.0, 0.8]
          step: 0.05

        min_instance_size:
          type: int
          range: [8, 128]
          step: 8

        min_seed_size:
          type: int
          range: [4, 64]
          step: 4
```

Usage:
```bash
# Tune only
python scripts/main.py --config tutorials/hydra-lv-full.yaml --mode tune

# Tune + test
python scripts/main.py --config tutorials/hydra-lv-full.yaml --mode tune+test

# Test with tuned params (set parameter_source: tuned first)
python scripts/main.py --config tutorials/hydra-lv-full.yaml --mode test
```

## Mode Selection Guide

### Use `train` when:
- ✓ Starting from scratch
- ✓ Training a new model
- ✓ Have training data with labels

### Use `resume` when:
- ✓ Training was interrupted
- ✓ Want to continue from checkpoint
- ✓ Changing hyperparameters (lr, epochs)

### Use `test` when:
- ✓ Have optimal parameters already
- ✓ Want to evaluate on test set
- ✓ Have ground truth labels

### Use `predict` when:
- ✓ No ground truth available
- ✓ Production deployment
- ✓ Just need segmentation output

### Use `tune` when:
- ✓ First time with new dataset
- ✓ Want to find optimal parameters
- ✓ Have validation set with labels
- ✓ Want parameter analysis

### Use `tune+test` when:
- ✓ Complete evaluation workflow
- ✓ Have both val and test sets
- ✓ Want best results
- ✓ **RECOMMENDED for production**

## justfile Command Reference

```bash
# Training
just train <dataset> [arch] [ARGS...]
just resume <dataset> <arch_or_ckpt> [ckpt_or_args] [ARGS...]

# Testing (traditional)
just test <dataset> <ckpt> [ARGS...]

# Inference (new)
just infer <dataset> <ckpt> [ARGS...]              # Alias for test

# Parameter Tuning (new)
just tune <dataset> <ckpt> [ARGS...]               # Tune only
just tune-test <dataset> <ckpt> [ARGS...]          # Tune + test
just tune-quick <dataset> <ckpt> [ARGS...]         # Quick tune (20 trials)
just test-with-params <dataset> <ckpt> <params> [ARGS...]  # Use specific params

# Monitoring
just tensorboard <experiment> [port]
just tensorboard-all [port]

# Visualization
just visualize <config> <mode> [ARGS...]
just visualize-files [image] [label] [port] [ARGS...]

# SLURM
just slurm <partition> <cpus> <gpus> <command>
```

## Tips & Best Practices

### 1. **Always Tune First**
```bash
# ❌ Don't do this (guess parameters)
just test hydra-lv ckpt.pt

# ✅ Do this (optimize first)
just tune-test hydra-lv ckpt.pt
```

### 2. **Save Tuning Results**
```bash
# Tuning results saved to:
outputs/hydra-lv_rsunet/tuning/
├── best_params.yaml         # ← Use this in your config
├── optimization_history.png
└── param_importance.png
```

### 3. **Use Config Profiles**
```yaml
# Create separate configs for different stages
tutorials/
├── hydra-lv-train.yaml    # For training
├── hydra-lv-tune.yaml     # For tuning
└── hydra-lv-test.yaml     # For testing
```

### 4. **Override from CLI**
```bash
# Quick parameter override without editing config
python scripts/main.py \
  --config tutorials/hydra-lv.yaml \
  --mode test \
  --checkpoint ckpt.pt \
  inference.decoding.fixed_params.binary_threshold=0.90
```

### 5. **Monitor Tuning Progress**
```bash
# Run tuning in background
just tune hydra-lv ckpt.pt &

# Monitor in another terminal
tail -f outputs/hydra-lv_rsunet/tuning/tuning.log
```

## Migration Guide: Old → New

### If you're using manual parameters:
```yaml
# Before
inference:
  decoding:
    - name: decode_binary_contour_distance_watershed
      kwargs:
        binary_threshold: [0.9, 0.85]  # Guessed values
        contour_threshold: [0.8, 1.1]

# After (Option 1: Keep as fixed)
inference:
  decoding:
    parameter_source: fixed
    fixed_params:
      binary_threshold: 0.85
      contour_threshold: 0.95

# After (Option 2: Use tuning)
inference:
  decoding:
    parameter_source: optuna
    optuna:
      enabled: true
      n_trials: 50
```

### If you're using sweep configs:
```yaml
# Before (sweep_example.yaml - grid search)
params:
  binary_threshold: [0.7, 0.8, 0.9]
  contour_threshold: [0.6, 0.8, 1.0]
# → 3 × 3 = 9 combinations (exhaustive)

# After (optuna - smart search)
inference:
  decoding:
    optuna:
      parameter_space:
        binary_threshold:
          range: [0.5, 0.95]
        contour_threshold:
          range: [0.6, 1.2]
      n_trials: 50
# → 50 trials (finds better result faster)
```

## Summary

✅ **Unified System:**
- All modes integrated into `main.py`
- Clean CLI interface
- Flexible config system
- Backward compatible

✅ **Recommended Workflow:**
```bash
just train <dataset>           # Train model
just tune-test <dataset> <ckpt>  # Optimize + evaluate
```

✅ **Next Steps:**
- See `.claude/MODE_INTEGRATION_DESIGN.md` for full design
- See `tutorials/optuna_decoding_tuning.yaml` for config examples
- See `tutorials/OPTUNA_QUICKSTART.md` for quick start guide

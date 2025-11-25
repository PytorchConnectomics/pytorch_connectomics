# Unified Inference and Parameter Tuning Guide

## Problem: Multiple Datasets, Multiple Purposes

In real-world scenarios, you often have:
- **Validation data**: For tuning parameters (has ground truth)
- **Test data**: For final evaluation (may or may not have ground truth)
- **Production data**: Real unlabeled data to segment

You want to:
1. Tune parameters on validation set
2. Apply optimized parameters to test/production data
3. All in one config file!

## Solution: Unified Configuration

The `unified_inference_tuning.yaml` config supports multiple workflows in a single file.

## Data Organization

```
datasets/
├── hydra/
│   ├── train_image.h5          # Training (for model training)
│   ├── train_label.h5
│   │
│   ├── val_image.h5            # Validation (for parameter tuning)
│   ├── val_label.h5            # ✓ Has ground truth
│   │
│   ├── test_image.h5           # Test (for final evaluation)
│   ├── test_label.h5           # ✓ Has ground truth
│   │
│   ├── test_volume2.h5         # Another test volume
│   └── test_volume3.h5         # Production data (no labels)
```

## Configuration Structure

### 1. Define Multiple Datasets

```yaml
inference:
  data:
    # Validation set (for tuning)
    validation:
      test_image: "datasets/hydra/val_image.h5"
      test_label: "datasets/hydra/val_label.h5"
      test_resolution: [30, 6, 6]

    # Test set (for final evaluation)
    test:
      test_image: "datasets/hydra/test_image.h5"
      test_label: "datasets/hydra/test_label.h5"
      test_resolution: [30, 6, 6]

    # Multiple test sets
    test_sets:
      - name: "test_main"
        test_image: "datasets/hydra/test_image.h5"
        test_label: "datasets/hydra/test_label.h5"

      - name: "test_volume2"
        test_image: "datasets/hydra/test_volume2.h5"
        test_label: null  # No ground truth

      - name: "production_data"
        test_image: "datasets/hydra/production_volume.h5"
        test_label: null
```

### 2. Configure Parameter Mode

```yaml
inference:
  decoding:
    # Choose mode: 'fixed', 'tuned', or 'optuna'
    parameter_mode: optuna

    # Fixed parameters (manual tuning)
    fixed_params:
      binary_threshold: 0.85
      contour_threshold: 0.95
      # ...

    # Use previously tuned parameters
    tuned_params:
      params_file: "outputs/tuning/best_params.yaml"

    # Optuna automatic tuning
    optuna_tuning:
      enabled: true
      tune_on_data: "validation"  # Which dataset to tune on
      n_trials: 50
      # ... (full Optuna config)
```

### 3. Define Workflow

```yaml
workflows:
  # Option 1: Just run inference
  inference_only:
    steps:
      - load_model
      - run_inference: [test]
      - decode: fixed_params

  # Option 2: Just tune parameters
  tuning_only:
    steps:
      - load_model
      - run_inference: [validation]
      - optuna_tuning

  # Option 3: Tune then apply to test (RECOMMENDED)
  tune_then_test:
    steps:
      - load_model
      - run_inference: [validation, test]
      - optuna_tuning: validation
      - apply_best_params: test
```

## Usage Examples

### Example 1: Quick Inference (Fixed Parameters)

**Use case:** You already know good parameters, just want to run inference

```yaml
# Set in config
inference:
  decoding:
    parameter_mode: fixed
    fixed_params:
      binary_threshold: 0.85
      contour_threshold: 0.95
      # ...
```

```bash
# Run inference
python scripts/main.py --config unified_inference_tuning.yaml --mode test

# Or override from CLI
python scripts/main.py \
  --config unified_inference_tuning.yaml \
  --mode test \
  inference.decoding.parameter_mode=fixed
```

**Result:**
```
outputs/hydra_lv_unified/test/
├── test_image_prediction.h5
├── test_image_segmentation.h5
└── metrics.json
```

### Example 2: Parameter Tuning Only

**Use case:** Find optimal parameters on validation set

```yaml
# Set in config
inference:
  decoding:
    parameter_mode: optuna
    optuna_tuning:
      enabled: true
      tune_on_data: "validation"
      n_trials: 50
```

```bash
python scripts/main.py --config unified_inference_tuning.yaml --mode tune
```

**Result:**
```
outputs/hydra_lv_unified/tuning/
├── best_params.yaml              # 🎯 Best parameters found
├── optuna_study.db                # Study database
├── all_trials.csv                 # All trial results
├── optimization_history.png       # Optimization progress
└── param_importance.png           # Which params matter most
```

**best_params.yaml:**
```yaml
binary_threshold: 0.87
contour_threshold: 0.94
distance_threshold: 0.42
min_instance_size: 28
min_seed_size: 12
```

### Example 3: Tune Then Test (Recommended Workflow)

**Use case:** Optimize on validation, then apply to test set

```yaml
# Set in config
inference:
  decoding:
    parameter_mode: optuna
    optuna_tuning:
      enabled: true
      tune_on_data: "validation"
      n_trials: 50
      apply_to_test:
        enabled: true
        test_datasets: ["test_main"]
```

```bash
python scripts/main.py --config unified_inference_tuning.yaml --mode tune+test
```

**Workflow:**
1. Load model
2. Run inference on validation set
3. Optimize parameters using Optuna
4. Save best parameters
5. Run inference on test set with best parameters
6. Evaluate and save results

**Result:**
```
outputs/hydra_lv_unified/
├── validation/
│   ├── val_image_prediction.h5
│   └── val_image_segmentation.h5
├── tuning/
│   ├── best_params.yaml          # 🎯 Optimized parameters
│   ├── optimization_history.png
│   └── param_importance.png
└── test/
    ├── test_image_prediction.h5
    ├── test_image_segmentation.h5  # 🎯 Segmented with optimal params
    └── metrics.json                 # 🎯 Final performance
```

### Example 4: Use Previously Tuned Parameters

**Use case:** You already tuned parameters, now apply to new data

```yaml
# Set in config
inference:
  decoding:
    parameter_mode: tuned
    tuned_params:
      params_file: "outputs/tuning/best_params.yaml"
```

```bash
python scripts/main.py --config unified_inference_tuning.yaml --mode test
```

**Workflow:**
1. Load best_params.yaml
2. Run inference with those parameters
3. Save results

### Example 5: Multiple Test Sets

**Use case:** Tune once, apply to many test volumes

```yaml
inference:
  data:
    validation:
      test_image: "datasets/hydra/val_image.h5"
      test_label: "datasets/hydra/val_label.h5"

    test_sets:
      - name: "test_volume1"
        test_image: "datasets/hydra/test_volume1.h5"
        test_label: "datasets/hydra/test_label1.h5"

      - name: "test_volume2"
        test_image: "datasets/hydra/test_volume2.h5"
        test_label: null  # No ground truth

      - name: "production_data"
        test_image: "datasets/hydra/production.h5"
        test_label: null

  decoding:
    parameter_mode: optuna
    optuna_tuning:
      tune_on_data: "validation"
      apply_to_test:
        enabled: true
        test_datasets: ["test_volume1", "test_volume2", "production_data"]
```

```bash
python scripts/main.py --config unified_inference_tuning.yaml --mode tune+test
```

**Result:**
```
outputs/hydra_lv_unified/
├── tuning/
│   └── best_params.yaml          # Tuned once on validation
└── test/
    ├── test_volume1_segmentation.h5   # Applied to all test sets
    ├── test_volume2_segmentation.h5
    └── production_data_segmentation.h5
```

## CLI Interface Design

### Option 1: Mode-based CLI

```bash
# Inference only (use fixed parameters)
python scripts/main.py --config config.yaml --mode test

# Tuning only (optimize on validation)
python scripts/main.py --config config.yaml --mode tune

# Combined workflow (tune + test)
python scripts/main.py --config config.yaml --mode tune+test

# Use previously tuned parameters
python scripts/main.py --config config.yaml --mode test --use-tuned
```

### Option 2: Explicit CLI

```bash
# Standard inference
python scripts/main.py \
  --config config.yaml \
  --mode test \
  --inference-params fixed

# Tune parameters
python scripts/tune_and_infer.py \
  --config config.yaml \
  --tune-on validation \
  --n-trials 50

# Tune and apply to test
python scripts/tune_and_infer.py \
  --config config.yaml \
  --tune-on validation \
  --apply-to test \
  --n-trials 50
```

### Option 3: Two-stage workflow (Most flexible)

```bash
# Stage 1: Optimize parameters
python scripts/tune_decoding.py \
  --config config.yaml \
  --tune-on validation \
  --output outputs/tuning

# Stage 2: Apply to test set
python scripts/main.py \
  --config config.yaml \
  --mode test \
  --params outputs/tuning/best_params.yaml
```

## Parameter Modes Comparison

| Mode | When to Use | Pros | Cons |
|------|------------|------|------|
| **fixed** | Quick inference, known params | Fast, simple | May not be optimal |
| **tuned** | Apply previous tuning | Fast, optimized | Requires prior tuning |
| **optuna** | First time, or re-optimize | Best results | Slower (tuning time) |

## Advanced: Conditional Tuning

**Use case:** Only tune if not already tuned

```yaml
inference:
  decoding:
    parameter_mode: optuna
    optuna_tuning:
      enabled: true

      # Only tune if no existing study found
      skip_if_exists: true

      # Or: only tune if existing results are poor
      retune_if:
        metric: adapted_rand
        threshold: 0.85  # Retune if previous best < 0.85
```

## Best Practices

### 1. Organize Data Clearly

```yaml
inference:
  data:
    validation:      # Always use for tuning
      test_image: "datasets/val_image.h5"
      test_label: "datasets/val_label.h5"  # Required for tuning

    test:            # Use for final evaluation
      test_image: "datasets/test_image.h5"
      test_label: "datasets/test_label.h5"  # Optional

    production:      # Real data (no labels)
      test_image: "datasets/production_data.h5"
      test_label: null
```

### 2. Use Descriptive Names

```yaml
test_sets:
  - name: "mouse_brain_vol1"
    test_image: "datasets/mouse/brain_vol1.h5"

  - name: "mouse_brain_vol2"
    test_image: "datasets/mouse/brain_vol2.h5"

  - name: "drosophila_vnc"
    test_image: "datasets/drosophila/vnc.h5"
```

### 3. Save Intermediate Results

```yaml
inference:
  test_time_augmentation:
    save_predictions: true  # Save before decoding

  decoding:
    optuna_tuning:
      output:
        save_all_trials: true
        save_best_segmentation: true
```

### 4. Use Study Persistence

```yaml
optuna_tuning:
  study_name: "hydra_lv_tuning"
  storage: "sqlite:///outputs/studies/hydra_lv.db"
  load_if_exists: true  # Resume if interrupted
```

## Troubleshooting

### Issue: Tuning on wrong dataset

**Symptom:** Parameters optimized but don't work on test set

**Solution:** Make sure validation and test have similar characteristics
```yaml
# Check data stats
validation: mean=128, std=45, shape=(100, 512, 512)
test:       mean=125, std=48, shape=(120, 512, 512)  # Similar ✓

# If very different, might need separate tuning
```

### Issue: Tuning takes too long

**Solution 1: Use cached predictions**
```yaml
inference:
  cache_predictions: true
  predictions_cache_dir: "outputs/predictions_cache"
```

**Solution 2: Reduce tuning data size**
```yaml
inference:
  data:
    validation:
      test_image: "datasets/val_image.h5"
      # Use subset (e.g., first 50 slices)
      subset_slices: [0, 50]
```

**Solution 3: Fewer trials**
```yaml
optuna_tuning:
  n_trials: 30  # Instead of 100
```

### Issue: Parameters don't transfer well

**Symptom:** Good on validation, poor on test

**Possible causes:**
1. **Overfitting to validation set**: Use more trials, wider search space
2. **Different data characteristics**: Check intensity distributions
3. **Need per-dataset tuning**: Tune separately for each test set

## Summary

The unified config approach provides:

✅ **Single source of truth**: One config for all inference workflows
✅ **Flexible workflows**: Choose fixed, tuned, or Optuna parameters
✅ **Multiple datasets**: Handle validation, test, production data
✅ **Reproducibility**: All settings in one file
✅ **Efficiency**: Tune once, apply to many test sets

**Recommended Workflow:**
```bash
# 1. Tune on validation set
python scripts/main.py --config unified_config.yaml --mode tune

# 2. Apply to test sets
python scripts/main.py --config unified_config.yaml --mode test --use-tuned

# Or do both in one command
python scripts/main.py --config unified_config.yaml --mode tune+test
```

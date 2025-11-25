# Optuna Parameter Tuning - Quick Start Guide

## What is Optuna Parameter Tuning?

Instead of manually guessing good parameter values for post-processing/decoding, Optuna automatically finds optimal parameters using smart Bayesian optimization.

## Current Manual Approach ❌

```yaml
# hydra-lv.yaml
inference:
  decoding:
    - name: decode_binary_contour_distance_watershed
      kwargs:
        binary_threshold: [0.9, 0.85]     # Guess 1
        contour_threshold: [0.8, 1.1]     # Guess 2
        distance_threshold: [0.5, 0]      # Guess 3
        min_instance_size: 16             # Guess 4
        min_seed_size: 8                  # Guess 5
```

**Problems:**
- Need to manually try different values
- Don't know which parameters matter most
- Time-consuming trial-and-error
- May miss optimal combinations

## New Optuna Approach ✅

```yaml
# optuna_decoding_tuning.yaml
parameter_space:
  decoder_name: decode_binary_contour_distance_watershed

  parameters:
    binary_threshold:
      type: float
      range: [0.5, 0.95]    # Let Optuna search this range

    contour_threshold:
      type: float
      range: [0.6, 1.2]     # Optuna will find best value

    min_instance_size:
      type: int
      range: [8, 128]       # Try all reasonable values

optuna:
  n_trials: 50              # Run 50 experiments automatically
  optimization:
    mode: single
    single_objective:
      metric: adapted_rand  # Maximize this metric
      direction: maximize
```

```bash
# One command to optimize everything
python scripts/tune_decoding.py --config tutorials/optuna_decoding_tuning.yaml
```

**Benefits:**
- ✅ Automatically finds best parameters
- ✅ Shows which parameters matter most
- ✅ 10-50x fewer trials than grid search
- ✅ Gets better results faster

## Example Output

```
[I 2025-01-25 10:00:00] Study started: hydra_lv_decoding_optimization
[I 2025-01-25 10:00:05] Trial 0: adapted_rand=0.823 | params={'binary_threshold': 0.75, 'contour_threshold': 0.85}
[I 2025-01-25 10:00:10] Trial 1: adapted_rand=0.846 | params={'binary_threshold': 0.80, 'contour_threshold': 0.90}
[I 2025-01-25 10:00:15] Trial 2: adapted_rand=0.867 | params={'binary_threshold': 0.82, 'contour_threshold': 0.88}
...
[I 2025-01-25 10:15:00] Trial 50: adapted_rand=0.891 | params={'binary_threshold': 0.85, 'contour_threshold': 0.92}

Best trial:
  Value: 0.912 (adapted_rand)
  Params:
    binary_threshold: 0.85
    contour_threshold: 0.95
    distance_threshold: 0.40
    min_instance_size: 32
    min_seed_size: 8

Results saved to: outputs/optuna_decoding_tuning/
  ✓ best_params.yaml         # Copy these to your config
  ✓ optimization_history.png  # See how optimization progressed
  ✓ param_importance.png      # See which params matter most
  ✓ all_trials.csv            # All trial results
```

## Usage Workflow

### Step 1: Create Tuning Config

Copy `tutorials/optuna_decoding_tuning.yaml` and customize:

```yaml
# 1. Point to your data
data:
  val_image: "path/to/your/val_image.h5"
  val_label: "path/to/your/val_label.h5"

# 2. Point to your model
model:
  checkpoint: "path/to/your/best_model.ckpt"
  architecture: rsunet  # Match your trained model

# 3. Define parameter search space
parameter_space:
  parameters:
    binary_threshold:
      type: float
      range: [0.5, 0.95]
      step: 0.05

# 4. Set optimization goal
optuna:
  n_trials: 50  # More trials = better results (but slower)
  optimization:
    single_objective:
      metric: adapted_rand
      direction: maximize
```

### Step 2: Run Optimization

```bash
python scripts/tune_decoding.py --config your_tuning_config.yaml
```

This will:
1. Load your model and validation data
2. Try 50 different parameter combinations
3. Evaluate each with your chosen metric (e.g., adapted_rand)
4. Find the best parameters automatically

### Step 3: Use Best Parameters

Copy the best parameters from `outputs/optuna_decoding_tuning/best_params.yaml` into your main config:

```yaml
# hydra-lv.yaml (after tuning)
inference:
  decoding:
    - name: decode_binary_contour_distance_watershed
      kwargs:
        binary_threshold: 0.85      # Optimized by Optuna!
        contour_threshold: 0.95     # Optimized by Optuna!
        distance_threshold: 0.40    # Optimized by Optuna!
        min_instance_size: 32       # Optimized by Optuna!
        min_seed_size: 8            # Optimized by Optuna!
```

### Step 4: Run Inference with Optimized Params

```bash
python scripts/main.py --config hydra-lv.yaml --mode test
```

## Advanced: Multi-Objective Optimization

Want to optimize multiple metrics at once? Use multi-objective mode:

```yaml
optuna:
  optimization:
    mode: multi
    multi_objective:
      objectives:
        - metric: adapted_rand
          direction: maximize

        - metric: voi_sum
          direction: minimize

      sampler: NSGAIIISampler  # Pareto optimization
```

This finds a **Pareto front** of optimal trade-offs between metrics.

## Comparison: Manual vs Optuna

| Aspect | Manual Tuning | Optuna Tuning |
|--------|--------------|---------------|
| **Time to find good params** | Hours/Days | Minutes/Hours |
| **Number of experiments** | 100-1000+ | 20-100 |
| **Find optimal params?** | Maybe | Very likely |
| **Understand param importance?** | No | Yes (automatic) |
| **Reproducible?** | Hard | Easy (saved study) |
| **Effort required** | High | Low |

## Key Parameters to Optimize

### For Watershed-Based Decoding

**High Priority:**
- `binary_threshold`: Foreground/background separation (0.5-0.95)
- `contour_threshold`: Instance boundary detection (0.6-1.2)
- `distance_threshold`: Seed placement (0.0-0.8)

**Medium Priority:**
- `min_instance_size`: Remove small objects (8-128 voxels)
- `min_seed_size`: Minimum seed size (4-64 voxels)

**Low Priority (usually fixed):**
- `use_numba`: Performance optimization (always true)
- `scale_factors`: Resolution scaling (usually [1,1,1])

## Tips for Best Results

1. **Start with reasonable ranges**
   - Too wide: wastes trials on bad values
   - Too narrow: might miss optimal value
   - Use log scale for wide ranges (e.g., min_instance_size)

2. **Use enough trials**
   - Simple problems: 20-50 trials
   - Complex problems: 50-200 trials
   - More parameters = need more trials

3. **Choose the right metric**
   - `adapted_rand`: Good general metric (0-1, higher better)
   - `voi_sum`: Alternative to adapted_rand (lower better)
   - `nerl`: Skeleton-based metric (requires skeleton file)

4. **Resume interrupted runs**
   ```yaml
   optuna:
     study_name: "my_study"
     storage: "sqlite:///optuna_study.db"
     load_if_exists: true  # Continue from last run
   ```

5. **Run parallel optimization**
   ```bash
   # Terminal 1
   python scripts/tune_decoding.py --config config.yaml &

   # Terminal 2
   python scripts/tune_decoding.py --config config.yaml &

   # Terminal 3
   python scripts/tune_decoding.py --config config.yaml &
   ```
   All workers share results via database!

## Troubleshooting

### "No improvement after many trials"
- ✅ Check if your search ranges are reasonable
- ✅ Try more trials (50 → 100+)
- ✅ Check if your metric is computed correctly

### "Optimization is very slow"
- ✅ Cache predictions (set `run_inference: false` and provide `prediction_path`)
- ✅ Use smaller validation set
- ✅ Run parallel workers with shared database

### "Results are not reproducible"
- ✅ Set `system.seed` in config
- ✅ Use deterministic algorithms
- ✅ Save study to database with `storage: "sqlite:///study.db"`

## Next Steps

1. ✅ Start with the provided example config: `tutorials/optuna_decoding_tuning.yaml`
2. ✅ Customize for your data and model
3. ✅ Run optimization: `python scripts/tune_decoding.py --config config.yaml`
4. ✅ Analyze results (plots and best parameters)
5. ✅ Use best parameters in your main config
6. ✅ Enjoy better segmentation results! 🎉

## Learn More

- **Optuna Documentation**: https://optuna.readthedocs.io/
- **Design Document**: `.claude/OPTUNA_DECODING_DESIGN.md`
- **Example Config**: `tutorials/optuna_decoding_tuning.yaml`
- **Integration Tests**: `tests/integration/test_optuna_tuning.py`

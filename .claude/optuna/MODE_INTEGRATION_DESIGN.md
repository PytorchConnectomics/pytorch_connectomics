# Integrating Optuna Tuning with Existing main.py Modes

## Problem Statement

**Current main.py modes:**
- `train` - Train a model
- `test` - Test/inference on test set
- `predict` - Prediction without ground truth

**New Optuna workflows needed:**
- `tune` - Optimize decoding parameters on validation set
- `tune-test` - Optimize on validation, then apply to test
- Use previously tuned parameters

**Question:** How to cleanly integrate these workflows without breaking existing code?

## Current Architecture

### main.py Mode Structure
```python
# main.py --mode <mode>
def main():
    args = parse_args()

    if args.mode == "train":
        # Setup data, model, trainer
        trainer.fit(model, datamodule)

    elif args.mode == "test":
        # Setup test data
        trainer.test(model, datamodule)

    elif args.mode == "predict":
        # Setup prediction data
        trainer.predict(model, datamodule)
```

### justfile Commands
```bash
just train <dataset>              # Train
just test <dataset> <ckpt>        # Test
just resume <dataset> <ckpt>      # Resume training
```

## Design Options

### Option 1: Extend --mode Flag (RECOMMENDED)

**Pros:**
- Consistent with existing interface
- Clear separation of concerns
- Easy to understand

**Cons:**
- Many mode values
- May need sub-modes

**Implementation:**
```python
# CLI
parser.add_argument(
    "--mode",
    choices=["train", "test", "predict", "tune", "tune-test"],
    default="train",
    help="Mode: train, test, predict, tune, or tune-test"
)

# main.py
def main():
    if args.mode == "train":
        run_training(cfg, args)

    elif args.mode == "test":
        run_testing(cfg, args)

    elif args.mode == "predict":
        run_prediction(cfg, args)

    elif args.mode == "tune":
        run_parameter_tuning(cfg, args)

    elif args.mode == "tune-test":
        # Two-stage workflow
        best_params = run_parameter_tuning(cfg, args)
        run_testing(cfg, args, params=best_params)
```

**justfile additions:**
```bash
# Tune decoding parameters on validation set
tune dataset ckpt *ARGS='':
    python scripts/main.py --config tutorials/{{dataset}}.yaml --mode tune --checkpoint {{ckpt}} {{ARGS}}

# Tune then test
tune-test dataset ckpt *ARGS='':
    python scripts/main.py --config tutorials/{{dataset}}.yaml --mode tune+test --checkpoint {{ckpt}} {{ARGS}}
```

---

### Option 2: Separate Script (tune_decoding.py)

**Pros:**
- Complete separation from training/inference
- No changes to main.py
- Focused tool for tuning

**Cons:**
- Separate script to maintain
- Duplicate config loading logic
- Less integrated workflow

**Implementation:**
```bash
# Standalone tuning
python scripts/tune_decoding.py --config optuna_decoding_tuning.yaml

# Then use optimized params in main.py
python scripts/main.py --config hydra-lv.yaml --mode test --params outputs/best_params.yaml
```

---

### Option 3: Config-Driven Workflows

**Pros:**
- Maximum flexibility
- No CLI changes needed
- Declarative approach

**Cons:**
- Less discoverable
- Config complexity
- Harder to understand workflow

**Implementation:**
```yaml
# In config file
workflow:
  type: tune+test  # or 'train', 'test', 'tune'

  tune_stage:
    enabled: true
    tune_on: validation
    n_trials: 50

  test_stage:
    enabled: true
    apply_tuned_params: true
```

```python
# main.py reads workflow from config
def main():
    workflow_type = cfg.workflow.type

    if workflow_type == "train":
        run_training(cfg)
    elif workflow_type == "tune+test":
        if cfg.workflow.tune_stage.enabled:
            best_params = run_tuning(cfg)
        if cfg.workflow.test_stage.enabled:
            run_testing(cfg, params=best_params)
```

---

## Recommended Approach: Hybrid (Option 1 + Config Flexibility)

Combine the best of both worlds:
- Use `--mode` flag for main workflows (discoverable, easy)
- Use config for detailed control (flexible, powerful)

### Architecture

```
┌────────────────────────────────────────────────────────────┐
│                     CLI Interface                          │
│                                                            │
│  --mode train          → Training                         │
│  --mode test           → Testing with fixed params        │
│  --mode predict        → Prediction                        │
│  --mode tune           → Parameter tuning only            │
│  --mode tune-test      → Tune then test                   │
│  --mode infer          → Alias for test (clarity)         │
│                                                            │
│  --params <file>       → Use specific params (optional)   │
│  --tune-config <file>  → Optuna config (optional)         │
└────────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────┐
│                  Mode Dispatcher                           │
│                                                            │
│  dispatch_mode(cfg, args) → execute appropriate workflow  │
└────────────────────────────────────────────────────────────┘
                            ↓
        ┌───────────────────┴───────────────────┐
        ↓                                       ↓
┌─────────────────┐                   ┌──────────────────────┐
│ Training Path   │                   │ Inference Path       │
│                 │                   │                      │
│ • train         │                   │ • test (infer)       │
│ • resume        │                   │ • predict            │
└─────────────────┘                   │ • tune               │
                                      │ • tune+test          │
                                      └──────────────────────┘
                                                ↓
                        ┌───────────────────────┴─────────────┐
                        ↓                                     ↓
              ┌──────────────────┐                 ┌──────────────────┐
              │ Parameter Source │                 │ Decoding         │
              │                  │                 │                  │
              │ • fixed          │───────────────→ │ • Binary+Cont.   │
              │ • tuned file     │                 │ • Watershed      │
              │ • optuna study   │                 │ • Affinity       │
              │ • optuna tuning  │                 │ • Custom         │
              └──────────────────┘                 └──────────────────┘
```

### Implementation Details

#### 1. Config Structure
```yaml
# Main config (e.g., hydra-lv.yaml)
inference:
  # Parameter strategy
  decoding:
    # How to get parameters
    parameter_source: fixed  # fixed | tuned | optuna

    # Option 1: Fixed parameters (traditional)
    fixed_params:
      binary_threshold: 0.85
      contour_threshold: 0.95
      # ...

    # Option 2: Load from tuned file
    tuned_params:
      path: "outputs/tuning/best_params.yaml"

    # Option 3: Optuna tuning
    optuna:
      enabled: false  # Set to true for --mode tune
      tune_on_data: validation
      n_trials: 50
      # ... (full optuna config)
```

#### 2. CLI Arguments
```python
def parse_args():
    parser.add_argument(
        "--mode",
        choices=["train", "test", "predict", "tune", "tune-test", "infer"],
        default="train",
        help="Execution mode"
    )

    parser.add_argument(
        "--params",
        type=str,
        default=None,
        help="Path to parameter file (overrides config)"
    )

    parser.add_argument(
        "--param-source",
        choices=["fixed", "tuned", "optuna"],
        default=None,
        help="Parameter source (overrides config)"
    )

    parser.add_argument(
        "--tune-trials",
        type=int,
        default=None,
        help="Number of Optuna trials (overrides config)"
    )

    # Existing args
    parser.add_argument("--checkpoint", ...)
    parser.add_argument("overrides", ...)
```

#### 3. Mode Dispatcher
```python
def main():
    args = parse_args()
    cfg = setup_config(args)

    # Dispatch to appropriate workflow
    if args.mode == "train":
        run_training_workflow(cfg, args)

    elif args.mode in ["test", "infer"]:
        run_inference_workflow(cfg, args, tune=False)

    elif args.mode == "predict":
        run_prediction_workflow(cfg, args)

    elif args.mode == "tune":
        run_tuning_workflow(cfg, args)

    elif args.mode == "tune-test":
        # Two-stage workflow
        run_tuning_and_inference_workflow(cfg, args)


def run_inference_workflow(cfg, args, tune=False):
    """
    Run inference (test/predict) workflow.

    Steps:
        1. Load model
        2. Determine parameter source (fixed/tuned/optuna)
        3. Run inference
        4. Apply decoding with parameters
        5. Evaluate (if ground truth available)
    """
    # Load model
    model = load_model(cfg, args.checkpoint)

    # Determine parameter source
    if args.params:
        # CLI override: load from file
        params = load_params_from_file(args.params)
        param_source = "file"

    elif args.param_source:
        # CLI override: use specified source
        param_source = args.param_source

    else:
        # Use config
        param_source = cfg.inference.decoding.parameter_source

    # Get parameters based on source
    if param_source == "fixed":
        params = cfg.inference.decoding.fixed_params

    elif param_source == "tuned":
        params_path = cfg.inference.decoding.tuned_params.path
        params = load_params_from_file(params_path)

    elif param_source == "optuna" and tune:
        # Run tuning to get params
        params = run_optuna_tuning(cfg, model)

    elif param_source == "optuna" and not tune:
        # Load best params from previous tuning
        study_db = cfg.inference.decoding.optuna.study_db
        params = load_best_params_from_study(study_db)

    # Run inference
    predictions = run_model_inference(model, cfg)

    # Apply decoding
    segmentation = apply_decoding(predictions, params, cfg)

    # Evaluate if ground truth available
    if has_ground_truth(cfg):
        metrics = evaluate_segmentation(segmentation, cfg)
        print_metrics(metrics)

    # Save results
    save_results(segmentation, params, cfg)


def run_tuning_workflow(cfg, args):
    """
    Run parameter tuning workflow.

    Steps:
        1. Load model
        2. Run inference on validation set
        3. Run Optuna optimization
        4. Save best parameters
        5. Generate visualizations
    """
    print("=" * 80)
    print("PARAMETER TUNING WORKFLOW")
    print("=" * 80)

    # Load model
    model = load_model(cfg, args.checkpoint)

    # Override n_trials from CLI if provided
    if args.tune_trials:
        cfg.inference.decoding.optuna.n_trials = args.tune_trials

    # Run inference on validation set
    print("\n1. Running inference on validation set...")
    val_predictions = run_validation_inference(model, cfg)

    # Run Optuna tuning
    print("\n2. Running Optuna parameter optimization...")
    tuner = OptunaDecodingTuner(cfg, val_predictions)
    study = tuner.optimize()

    # Save results
    print("\n3. Saving results...")
    best_params = study.best_params
    save_tuning_results(study, cfg)

    # Generate visualizations
    print("\n4. Generating visualizations...")
    generate_tuning_plots(study, cfg)

    print("\n" + "=" * 80)
    print("TUNING COMPLETE!")
    print(f"Best parameters saved to: {cfg.output_dir}/best_params.yaml")
    print(f"Best {cfg.optuna.metric}: {study.best_value:.4f}")
    print("=" * 80)

    return best_params


def run_tuning_and_inference_workflow(cfg, args):
    """
    Combined workflow: tune on validation, then test.

    Steps:
        1. Load model
        2. Run tuning on validation set
        3. Apply best params to test set
        4. Evaluate and save results
    """
    print("=" * 80)
    print("TUNE + TEST WORKFLOW")
    print("=" * 80)

    # Stage 1: Tuning
    print("\n" + "=" * 80)
    print("STAGE 1: PARAMETER TUNING")
    print("=" * 80)
    best_params = run_tuning_workflow(cfg, args)

    # Stage 2: Testing with best params
    print("\n" + "=" * 80)
    print("STAGE 2: TESTING WITH OPTIMIZED PARAMETERS")
    print("=" * 80)

    # Override parameter source to use tuned params
    cfg.inference.decoding.parameter_source = "fixed"
    cfg.inference.decoding.fixed_params = best_params

    # Run inference on test set
    run_inference_workflow(cfg, args, tune=False)

    print("\n" + "=" * 80)
    print("TUNE + TEST COMPLETE!")
    print("=" * 80)
```

#### 4. justfile Integration
```bash
# ============================================================================
# Inference Commands (NEW)
# ============================================================================

# Run inference with fixed parameters (alias for test)
infer dataset ckpt *ARGS='':
    python scripts/main.py --config tutorials/{{dataset}}.yaml --mode infer --checkpoint {{ckpt}} {{ARGS}}

# Tune decoding parameters on validation set
tune dataset ckpt *ARGS='':
    python scripts/main.py --config tutorials/{{dataset}}.yaml --mode tune --checkpoint {{ckpt}} {{ARGS}}

# Tune parameters then test (all-in-one)
tune-test dataset ckpt *ARGS='':
    python scripts/main.py --config tutorials/{{dataset}}.yaml --mode tune+test --checkpoint {{ckpt}} {{ARGS}}

# Test with specific parameter file
test-with-params dataset ckpt params *ARGS='':
    python scripts/main.py --config tutorials/{{dataset}}.yaml --mode test --checkpoint {{ckpt}} --params {{params}} {{ARGS}}

# Quick tuning (20 trials for testing)
tune-quick dataset ckpt *ARGS='':
    python scripts/main.py --config tutorials/{{dataset}}.yaml --mode tune --checkpoint {{ckpt}} --tune-trials 20 {{ARGS}}
```

### Usage Examples

#### Example 1: Traditional Test (No Tuning)
```bash
# Using fixed parameters from config
just test hydra-lv checkpoints/best.ckpt

# Equivalent to:
python scripts/main.py \
  --config tutorials/hydra-lv.yaml \
  --mode test \
  --checkpoint checkpoints/best.ckpt
```

Config uses `parameter_source: fixed`:
```yaml
inference:
  decoding:
    parameter_source: fixed
    fixed_params:
      binary_threshold: 0.85
      contour_threshold: 0.95
```

---

#### Example 2: Tune Parameters Only
```bash
# Tune on validation set
just tune hydra-lv checkpoints/best.ckpt

# With more trials
just tune hydra-lv checkpoints/best.ckpt --tune-trials 100
```

Config enables tuning:
```yaml
inference:
  decoding:
    parameter_source: optuna
    optuna:
      enabled: true
      tune_on_data: validation
      n_trials: 50
```

**Output:**
```
outputs/hydra-lv_tuning/
├── best_params.yaml
├── optuna_study.db
├── optimization_history.png
└── param_importance.png
```

---

#### Example 3: Tune Then Test (Recommended)
```bash
# One command: tune + test
just tune-test hydra-lv checkpoints/best.ckpt
```

**Workflow:**
1. Tune on validation → get best params
2. Apply best params to test set
3. Evaluate and save results

**Output:**
```
outputs/hydra-lv_rsunet/
├── tuning/
│   ├── best_params.yaml
│   ├── optuna_study.db
│   └── *.png (plots)
└── test/
    ├── test_segmentation.h5
    └── metrics.json
```

---

#### Example 4: Use Previously Tuned Parameters
```bash
# Option 1: Set in config
# inference.decoding.parameter_source: tuned
# inference.decoding.tuned_params.path: "outputs/tuning/best_params.yaml"
just test hydra-lv checkpoints/best.ckpt

# Option 2: CLI override
just test-with-params hydra-lv checkpoints/best.ckpt outputs/tuning/best_params.yaml

# Option 3: CLI override with param-source
python scripts/main.py \
  --config tutorials/hydra-lv.yaml \
  --mode test \
  --checkpoint checkpoints/best.ckpt \
  --param-source tuned \
  inference.decoding.tuned_params.path=outputs/tuning/best_params.yaml
```

---

#### Example 5: Load Best Params from Optuna Study
```yaml
# Config
inference:
  decoding:
    parameter_source: optuna
    optuna:
      enabled: false  # Don't tune
      study_db: "sqlite:///outputs/studies/hydra-lv.db"
      study_name: "hydra_lv_optimization"
      use_best_trial: true
```

```bash
just test hydra-lv checkpoints/best.ckpt
# Loads best params from study database
```

---

### Decision Matrix: When to Use Each Mode

| Scenario | Mode | Parameter Source | Command |
|----------|------|------------------|---------|
| **Quick test with known params** | `test` | `fixed` | `just test dataset ckpt` |
| **First time tuning** | `tune` | `optuna` | `just tune dataset ckpt` |
| **Production workflow** | `tune-test` | `optuna` | `just tune-test dataset ckpt` |
| **Use previous tuning** | `test` | `tuned` | `just test-with-params dataset ckpt params.yaml` |
| **Resume tuning** | `tune` | `optuna` (load_if_exists) | `just tune dataset ckpt` |
| **Prediction (no labels)** | `predict` | `fixed` or `tuned` | `python main.py --mode predict` |

---

### Config Organization

#### Option A: Single Config (Simpler)
```yaml
# hydra-lv.yaml - Everything in one file

model:
  architecture: rsunet
  checkpoint: "checkpoints/best.ckpt"

data:
  train_image: ...
  val_image: ...    # For tuning
  test_image: ...   # For final eval

inference:
  decoding:
    parameter_source: optuna  # Change based on needs

    fixed_params:
      binary_threshold: 0.85

    tuned_params:
      path: "outputs/tuning/best_params.yaml"

    optuna:
      enabled: true
      n_trials: 50
```

#### Option B: Separate Configs (Cleaner)
```yaml
# hydra-lv-train.yaml - Training
model: ...
data:
  train_image: ...
  train_label: ...

# hydra-lv-tune.yaml - Tuning
model:
  checkpoint: "outputs/hydra-lv/best.ckpt"
data:
  val_image: ...
  val_label: ...
inference:
  decoding:
    parameter_source: optuna
    optuna:
      n_trials: 50

# hydra-lv-test.yaml - Testing
model:
  checkpoint: "outputs/hydra-lv/best.ckpt"
data:
  test_image: ...
  test_label: ...
inference:
  decoding:
    parameter_source: tuned
    tuned_params:
      path: "outputs/tuning/best_params.yaml"
```

**Recommendation:** Use Option A (single config) with parameter_source switching. More convenient for users.

---

## Summary

### Recommended Implementation

✅ **Mode System:**
- Extend `--mode` with `tune` and `tune-test`
- Keep existing modes (`train`, `test`, `predict`)
- Add `infer` as alias for `test` (clarity)

✅ **Parameter System:**
- Three sources: `fixed`, `tuned`, `optuna`
- Config-driven defaults
- CLI overrides available

✅ **justfile Commands:**
```bash
just test <dataset> <ckpt>              # Test with fixed params
just tune <dataset> <ckpt>              # Tune only
just tune-test <dataset> <ckpt>         # Tune + test (recommended)
just test-with-params <dataset> <ckpt> <params>  # Use specific params
```

✅ **Benefits:**
- Backward compatible (existing commands work)
- Clear, discoverable interface
- Flexible config system
- No code duplication

### Next Steps

1. ✅ Design complete
2. ⬜ Implement mode dispatcher
3. ⬜ Implement OptunaDecodingTuner
4. ⬜ Update justfile
5. ⬜ Add CLI arguments
6. ⬜ Write tests
7. ⬜ Update documentation

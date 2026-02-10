# Optuna-Based Decoding Parameter Optimization Design

## Overview

This document describes the design for automated hyperparameter optimization of post-processing/decoding parameters using Optuna, replacing manual parameter sweeping with intelligent Bayesian optimization.

## Motivation

**Current Problem:**
- Decoding parameters (thresholds, min_instance_size, etc.) are manually specified in configs
- Finding optimal parameters requires tedious manual grid search or random search
- No automated way to optimize for specific metrics (adapted_rand, VOI, etc.)
- Parameter interactions are not explored efficiently

**Solution:**
- Integrate Optuna for automated hyperparameter optimization
- Define flexible parameter search spaces in YAML configs
- Support both single-objective and multi-objective optimization
- Provide visualization and analysis tools for understanding parameter importance

## Architecture

### 1. Component Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Interface                          │
│  - YAML config (optuna_decoding_tuning.yaml)                   │
│  - CLI script (scripts/tune_decoding.py)                       │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Optimization Controller                      │
│  - Load config and validate                                    │
│  - Create Optuna study                                        │
│  - Configure sampler (TPE, Random, CmaEs, etc.)              │
│  - Configure pruner (optional early stopping)                │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Objective Function                           │
│  1. Sample parameters from Optuna                              │
│  2. Run inference (or load cached predictions)                │
│  3. Apply decoding with sampled parameters                    │
│  4. Compute evaluation metrics                                │
│  5. Return objective value(s) to Optuna                       │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Decoding Pipeline                          │
│  - decode_binary_contour_distance_watershed                    │
│  - decode_binary_watershed                                     │
│  - Other custom decoders                                       │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Evaluation Metrics                           │
│  - Volume metrics: adapted_rand, VOI, precision, recall, F1   │
│  - Skeleton metrics: NERL, ERL (requires funlib)              │
└─────────────────────────────────────────────────────────────────┘
```

### 2. Key Components

#### A. Configuration System (`optuna_decoding_tuning.yaml`)

**Parameter Search Space Definition:**
```yaml
parameter_space:
  decoder_name: decode_binary_contour_distance_watershed

  parameters:
    binary_threshold:
      type: float          # Parameter type
      range: [0.5, 0.95]   # Search range
      step: 0.05           # Optional step size
      log: false           # Use log scale?

    contour_threshold:
      type: float
      range: [0.6, 1.2]
      step: 0.05

    min_instance_size:
      type: int
      range: [8, 128]
      log: true  # Log scale for wider range exploration
```

**Supported Parameter Types:**
- `float`: Continuous float values
- `int`: Integer values (with optional step)
- `categorical`: Discrete choices (e.g., ["mode_a", "mode_b"])
- `log_float`: Float with log-scale sampling
- `log_int`: Integer with log-scale sampling

#### B. Optimization Controller (`connectomics/decoding/optuna_tuner.py`)

**Core Class:**
```python
class OptunaDecodingTuner:
    """
    Automated hyperparameter optimization for decoding parameters.

    Features:
    - Single-objective and multi-objective optimization
    - Flexible parameter space definition
    - Early stopping with pruning
    - Visualization and reporting
    - Study persistence (SQLite, PostgreSQL)
    """

    def __init__(self, config):
        self.cfg = config
        self.study = self._create_study()
        self.predictions = None  # Cache predictions
        self.ground_truth = None

    def _create_study(self):
        """Create Optuna study with sampler and pruner."""
        sampler = self._get_sampler()
        pruner = self._get_pruner()

        return optuna.create_study(
            study_name=self.cfg.optuna.study_name,
            storage=self.cfg.optuna.storage,
            sampler=sampler,
            pruner=pruner,
            direction=self._get_direction(),
            load_if_exists=self.cfg.optuna.load_if_exists
        )

    def _objective(self, trial: optuna.Trial):
        """
        Objective function for optimization.

        Args:
            trial: Optuna trial object

        Returns:
            Objective value (single float or tuple for multi-objective)
        """
        # 1. Sample parameters
        params = self._sample_parameters(trial)

        # 2. Run decoding with sampled parameters
        segmentation = self._decode(params)

        # 3. Evaluate metrics
        metrics = self._evaluate(segmentation)

        # 4. Log results
        self._log_trial(trial, params, metrics)

        # 5. Return objective value
        return self._get_objective_value(metrics)

    def optimize(self):
        """Run optimization."""
        self.study.optimize(
            self._objective,
            n_trials=self.cfg.optuna.n_trials,
            timeout=self.cfg.optuna.timeout,
            show_progress_bar=self.cfg.logging.show_progress_bar
        )

        return self._get_results()
```

#### C. Parameter Sampling

**Dynamic Parameter Sampling:**
```python
def _sample_parameters(self, trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sample parameters from search space using trial.suggest_*.

    Maps YAML parameter definitions to Optuna suggest methods.
    """
    params = {}

    for param_name, param_config in self.cfg.parameter_space.parameters.items():
        param_type = param_config.type

        if param_type == "float":
            params[param_name] = trial.suggest_float(
                param_name,
                param_config.range[0],
                param_config.range[1],
                step=param_config.get("step"),
                log=param_config.get("log", False)
            )

        elif param_type == "int":
            params[param_name] = trial.suggest_int(
                param_name,
                param_config.range[0],
                param_config.range[1],
                step=param_config.get("step", 1),
                log=param_config.get("log", False)
            )

        elif param_type == "categorical":
            params[param_name] = trial.suggest_categorical(
                param_name,
                param_config.choices
            )

    # Add fixed parameters
    params.update(self.cfg.parameter_space.fixed_parameters)

    return params
```

#### D. Decoding Integration

**Decoder Registry:**
```python
DECODER_REGISTRY = {
    "decode_binary_contour_distance_watershed": decode_binary_contour_distance_watershed,
    "decode_binary_watershed": decode_binary_watershed,
    # Add more decoders...
}

def _decode(self, params: Dict[str, Any]) -> np.ndarray:
    """
    Apply decoding with given parameters.

    Args:
        params: Decoding parameters sampled by Optuna

    Returns:
        Instance segmentation (D, H, W)
    """
    decoder_name = self.cfg.parameter_space.decoder_name
    decoder_fn = DECODER_REGISTRY[decoder_name]

    # Load or use cached predictions
    if self.predictions is None:
        self.predictions = self._load_predictions()

    # Apply decoder
    segmentation = decoder_fn(self.predictions, **params)

    return segmentation
```

#### E. Evaluation Metrics

**Metric Computation:**
```python
def _evaluate(self, segmentation: np.ndarray) -> Dict[str, float]:
    """
    Compute evaluation metrics.

    Returns:
        Dictionary of metric names to values
    """
    metrics = {}

    # Volume-based metrics
    if "adapted_rand" in self.cfg.evaluation.metrics:
        metrics["adapted_rand"] = compute_adapted_rand(
            segmentation, self.ground_truth
        )

    if "voi_sum" in self.cfg.evaluation.metrics:
        voi_split, voi_merge = compute_voi(
            segmentation, self.ground_truth
        )
        metrics["voi_sum"] = voi_split + voi_merge
        metrics["voi_split"] = voi_split
        metrics["voi_merge"] = voi_merge

    # Skeleton-based metrics (if enabled)
    if self.cfg.evaluation.skeleton_metrics.enabled:
        skeleton_metrics = self._compute_skeleton_metrics(segmentation)
        metrics.update(skeleton_metrics)

    return metrics
```

### 3. Optimization Modes

#### A. Single-Objective Optimization

**Goal:** Maximize/minimize a single metric (e.g., maximize adapted_rand)

```yaml
optuna:
  optimization:
    mode: single
    single_objective:
      metric: adapted_rand
      direction: maximize
```

**Optuna Study:**
```python
study = optuna.create_study(
    direction="maximize",  # or "minimize"
    sampler=optuna.samplers.TPESampler()
)
```

#### B. Multi-Objective Optimization (Pareto Front)

**Goal:** Optimize multiple conflicting objectives simultaneously

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

      sampler: NSGAIIISampler  # Multi-objective sampler
```

**Optuna Study:**
```python
study = optuna.create_study(
    directions=["maximize", "minimize"],  # One per objective
    sampler=optuna.samplers.NSGAIIISampler()
)

def objective(trial):
    params = sample_parameters(trial)
    seg = decode(params)
    metrics = evaluate(seg)

    # Return tuple of objective values
    return metrics["adapted_rand"], metrics["voi_sum"]
```

**Result:** Pareto front of non-dominated solutions

### 4. Advanced Features

#### A. Early Stopping (Pruning)

**Problem:** Some parameter combinations are clearly bad early in evaluation

**Solution:** Optuna pruners can terminate unpromising trials

```yaml
optuna:
  pruner:
    enabled: true
    name: MedianPruner
    kwargs:
      n_startup_trials: 5
      n_warmup_steps: 0
```

**Note:** Pruning requires intermediate values (e.g., metrics on subsets). For post-processing, pruning is less applicable but can be used if evaluating on multiple volumes.

#### B. Distributed Optimization

**Problem:** Optimization can be slow for large datasets

**Solution:** Use shared storage backend for parallel trials

```yaml
optuna:
  storage: "postgresql://user:pass@host:5432/optuna_db"
  # or
  storage: "sqlite:///shared_study.db"
```

**Run multiple workers:**
```bash
# Terminal 1
python scripts/tune_decoding.py --config config.yaml --worker-id 0

# Terminal 2
python scripts/tune_decoding.py --config config.yaml --worker-id 1

# Terminal 3
python scripts/tune_decoding.py --config config.yaml --worker-id 2
```

#### C. Study Resumption

**Problem:** Long optimization runs may be interrupted

**Solution:** Persist study to database and resume

```yaml
optuna:
  study_name: "my_optimization"
  storage: "sqlite:///optuna_study.db"
  load_if_exists: true  # Resume existing study
```

#### D. Parameter Importance Analysis

**Goal:** Understand which parameters matter most

**Method:** Optuna's built-in parameter importance calculation

```python
# After optimization
importances = optuna.importance.get_param_importances(study)

# Visualize
fig = optuna.visualization.plot_param_importances(study)
fig.write_image("param_importance.png")
```

### 5. Visualization and Reporting

#### A. Optimization History

**Plot objective value vs trial number:**
```python
fig = optuna.visualization.plot_optimization_history(study)
```

#### B. Parallel Coordinate Plot

**Visualize parameter interactions:**
```python
fig = optuna.visualization.plot_parallel_coordinate(study)
```

Shows all trials as lines, with color indicating objective value.

#### C. Slice Plot

**2D parameter space visualization:**
```python
fig = optuna.visualization.plot_slice(study, params=["binary_threshold", "contour_threshold"])
```

#### D. Contour Plot

**Heatmap of 2-parameter interactions:**
```python
fig = optuna.visualization.plot_contour(study, params=["binary_threshold", "min_instance_size"])
```

#### E. Pareto Front (Multi-Objective)

**Plot non-dominated solutions:**
```python
fig = optuna.visualization.plot_pareto_front(study)
```

### 6. Implementation Plan

#### Phase 1: Core Infrastructure
1. Create `connectomics/decoding/optuna_tuner.py`
   - `OptunaDecodingTuner` class
   - Parameter sampling
   - Objective function
   - Study creation and management

2. Create `scripts/tune_decoding.py`
   - CLI entry point
   - Config loading
   - Result reporting

3. Update config system
   - Add Optuna config dataclasses to `hydra_config.py`
   - Parameter space definition schema

#### Phase 2: Integration
1. Integrate with existing decoders
   - Registry of available decoders
   - Dynamic parameter passing

2. Integrate with metrics
   - Volume-based metrics (adapted_rand, VOI)
   - Skeleton-based metrics (NERL, ERL)

3. Inference caching
   - Save/load predictions to avoid redundant inference
   - Memory-efficient handling for large volumes

#### Phase 3: Advanced Features
1. Multi-objective optimization
   - NSGA-II/III sampler integration
   - Pareto front visualization

2. Distributed optimization
   - Database backend support
   - Worker coordination

3. Visualization and reporting
   - All Optuna visualization types
   - Markdown/HTML report generation
   - Best parameter export

#### Phase 4: Testing and Documentation
1. Unit tests for core components
2. Integration tests with example data
3. Tutorial notebook
4. Documentation updates

### 7. Example Usage

#### A. Quick Start (Single-Objective)

```yaml
# config: tutorials/optuna_decoding_tuning.yaml
optuna:
  n_trials: 50
  optimization:
    mode: single
    single_objective:
      metric: adapted_rand
      direction: maximize

parameter_space:
  decoder_name: decode_binary_contour_distance_watershed
  parameters:
    binary_threshold:
      type: float
      range: [0.5, 0.95]

    contour_threshold:
      type: float
      range: [0.6, 1.2]
```

```bash
python scripts/tune_decoding.py --config tutorials/optuna_decoding_tuning.yaml
```

**Output:**
```
[I 2025-01-25 10:00:00,123] A new study created in RDB with name: hydra_lv_decoding_optimization
[I 2025-01-25 10:00:05,456] Trial 0 finished with value: 0.8234 and parameters: {'binary_threshold': 0.75, 'contour_threshold': 0.85}
[I 2025-01-25 10:00:10,789] Trial 1 finished with value: 0.8456 and parameters: {'binary_threshold': 0.80, 'contour_threshold': 0.90}
...
[I 2025-01-25 10:30:00,000] Best trial:
  Value: 0.9123
  Params:
    binary_threshold: 0.85
    contour_threshold: 0.95
    distance_threshold: 0.40
    min_instance_size: 32
    min_seed_size: 8
```

#### B. Multi-Objective Optimization

```yaml
optuna:
  n_trials: 100
  optimization:
    mode: multi
    multi_objective:
      objectives:
        - metric: adapted_rand
          direction: maximize

        - metric: voi_sum
          direction: minimize

      sampler: NSGAIIISampler
```

```bash
python scripts/tune_decoding.py --config config.yaml
```

**Output:** Pareto front with multiple optimal solutions trading off adapted_rand vs VOI.

#### C. Resume Interrupted Study

```yaml
optuna:
  study_name: "my_study"
  storage: "sqlite:///optuna_study.db"
  load_if_exists: true
  n_trials: 100  # Run 100 MORE trials
```

```bash
# First run (interrupted after 30 trials)
python scripts/tune_decoding.py --config config.yaml

# Resume (will run 100 more trials)
python scripts/tune_decoding.py --config config.yaml
```

### 8. Integration with Existing Workflow

**Current Manual Workflow:**
```yaml
# hydra-lv.yaml
inference:
  decoding:
    - name: decode_binary_contour_distance_watershed
      kwargs:
        binary_threshold: [0.9, 0.85]  # Manually specified
        contour_threshold: [0.8, 1.1]
        distance_threshold: [0.5, 0]
        min_instance_size: 16
        min_seed_size: 8
```

**New Optuna-Optimized Workflow:**

1. **Optimize parameters:**
```bash
python scripts/tune_decoding.py --config tutorials/optuna_decoding_tuning.yaml
```

2. **Best parameters saved to:**
```
outputs/optuna_decoding_tuning/best_params.yaml
```

3. **Use optimized parameters:**
```yaml
# hydra-lv.yaml
inference:
  decoding:
    - name: decode_binary_contour_distance_watershed
      kwargs: !include outputs/optuna_decoding_tuning/best_params.yaml
```

Or copy-paste best parameters manually.

### 9. Benefits

#### A. Efficiency
- **TPE sampler** explores parameter space intelligently
- **10-50x fewer trials** than grid search for similar results
- **Parallel optimization** with shared database

#### B. Insights
- **Parameter importance** shows which params matter
- **Interaction plots** reveal parameter relationships
- **Multi-objective** finds trade-offs automatically

#### C. Reproducibility
- **Persistent studies** in database
- **Trial history** fully logged
- **Best parameters** automatically exported

#### D. Flexibility
- **Easy to add new parameters** (just update YAML)
- **Support for any decoder** (via registry)
- **Custom metrics** easily integrated

### 10. Comparison to Manual Sweeping

| Aspect | Manual Grid Search | Optuna TPE |
|--------|-------------------|------------|
| **Trials needed** | 100-1000+ | 20-100 |
| **Search strategy** | Exhaustive | Intelligent (Bayesian) |
| **Parameter interactions** | Not explored | Automatically discovered |
| **Early stopping** | No | Yes (pruning) |
| **Multi-objective** | Separate runs | Single Pareto run |
| **Resumability** | Manual tracking | Built-in |
| **Visualization** | Manual plotting | Automatic |
| **Best use case** | Small search space | Large/complex space |

### 11. Future Enhancements

1. **Neural Architecture Search (NAS)**
   - Optimize model architecture params (filters, layers, etc.)
   - Combine with decoding optimization

2. **AutoML Pipeline**
   - Joint optimization of training + post-processing
   - End-to-end hyperparameter tuning

3. **Transfer Learning**
   - Use optimal params from one dataset on another
   - Domain adaptation for parameter transfer

4. **Real-time Visualization**
   - Web dashboard for monitoring optimization
   - Live updates during optimization

5. **Ensemble Decoding**
   - Combine multiple decoders with learned weights
   - Stacking for improved performance

## Conclusion

This design provides a comprehensive, efficient, and user-friendly system for automated hyperparameter optimization of decoding parameters. By integrating Optuna with the existing PyTorch Connectomics infrastructure, users can achieve better segmentation quality with less manual effort.

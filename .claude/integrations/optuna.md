# Optuna Parameter Tuning

Automated hyperparameter optimization for decoding/post-processing parameters using Bayesian optimization.

## Installation

```bash
pip install -e .[optim]  # Installs optuna>=2.10.0
```

## Usage

### CLI

```bash
python scripts/main.py --config tutorials/my_config.yaml --mode tune
```

### Configuration

```yaml
tune:
  inference:
    decoding:
      method: connected_components
      tuning:
        enabled: true
        n_trials: 100
        metric: nerl          # Optimize for normalized ERL
        direction: maximize
        params:
          threshold:
            type: float
            low: 0.3
            high: 0.9
          min_size:
            type: int
            low: 50
            high: 500
```

## How It Works

1. Loads trained model checkpoint
2. Runs inference on validation data
3. Optuna suggests decoding parameters (threshold, min_size, etc.)
4. Evaluates segmentation quality (VOI, NERL, etc.)
5. Repeats with Bayesian optimization to find optimal parameters
6. Saves best parameters to config

## Key Components

- `connectomics/decoding/tuning/optuna_tuner.py` -- OptunaTuner class
- `connectomics/decoding/tuning/auto_tuning.py` -- Auto-tuning pipeline
- `connectomics/config/schema/stages.py` -- TuneConfig dataclass

## Parameter Types

- `float`: Continuous (e.g., threshold)
- `int`: Integer (e.g., min_size)
- `categorical`: Discrete choices (e.g., method name)

## Supported Metrics

- VOI (split/merge/sum)
- NERL (normalized expected run length)
- Adapted Rand Error
- Custom metrics via registry

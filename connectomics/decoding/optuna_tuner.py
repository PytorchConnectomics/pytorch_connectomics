"""
Optuna-based hyperparameter tuning for decoding/post-processing parameters.

This module provides automated parameter optimization for instance segmentation
post-processing, particularly for watershed-based decoding with binary, contour,
and distance predictions.

Usage:
    from connectomics.decoding.optuna_tuner import OptunaDecodingTuner

    tuner = OptunaDecodingTuner(cfg, predictions, ground_truth)
    study = tuner.optimize()
    best_params = study.best_params
"""

from __future__ import annotations
from typing import Dict, Any, Optional
from pathlib import Path
import warnings
from collections import defaultdict

import numpy as np
import h5py
from omegaconf import DictConfig, OmegaConf

try:
    import optuna
    from optuna.samplers import TPESampler, CmaEsSampler, RandomSampler
    from optuna.pruners import MedianPruner, HyperbandPruner

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    warnings.warn(
        "Optuna not available. Install with: pip install optuna\n"
        "Parameter tuning will not work without Optuna."
    )

# Import decoding functions
from .segmentation import decode_instance_binary_contour_distance
from .utils import remove_small_instances

# Import metrics
from connectomics.metrics.metrics_seg import adapted_rand


__all__ = ["OptunaDecodingTuner", "run_tuning", "load_and_apply_best_params"]


class OptunaDecodingTuner:
    """
    Optuna-based parameter tuner for decoding/post-processing.

    This class handles automated hyperparameter optimization for instance
    segmentation post-processing, supporting:
    - Binary + Contour + Distance watershed decoding
    - Post-processing (small instance removal)
    - Single and multi-objective optimization
    - Flexible parameter search spaces with tuple support

    Args:
        cfg: Hydra configuration with tune and tune.parameter_space sections
        predictions: Model predictions (C, D, H, W) or path to .h5 file
        ground_truth: Ground truth labels (D, H, W) or path to .h5 file
        mask: Optional foreground mask (D, H, W) or path to .h5 file

    Example:
        >>> tuner = OptunaDecodingTuner(cfg, predictions, ground_truth)
        >>> study = tuner.optimize()
        >>> print(f"Best adapted_rand: {study.best_value:.4f}")
        >>> print(f"Best params: {study.best_params}")
    """

    def __init__(
        self,
        cfg: DictConfig,
        predictions: np.ndarray | str | Path | list,
        ground_truth: np.ndarray | str | Path | list,
        mask: Optional[np.ndarray | str | Path | list] = None,
    ):
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna not available. Install with: pip install optuna")

        self.cfg = cfg

        # Load data (can be lists of volumes or single volume)
        self.predictions = self._load_data_list(predictions, "predictions")
        self.ground_truth = self._load_data_list(ground_truth, "ground_truth")
        self.mask = self._load_data_list(mask, "mask") if mask is not None else None

        # Validate data shapes
        self._validate_data()

        # Extract config sections
        self.tune_cfg = cfg.tune
        self.param_space_cfg = cfg.tune.parameter_space

        # Initialize trial counter
        self.trial_count = 0

    def _load_data_list(self, data: np.ndarray | str | Path | list, name: str) -> list:
        """Load data from list of arrays/files or single array/file."""
        # Handle list of volumes
        if isinstance(data, list):
            return [self._load_data(item, name) for item in data]
        # Single volume - wrap in list for uniform handling
        return [self._load_data(data, name)]

    def _load_data(self, data: np.ndarray | str | Path, name: str) -> np.ndarray:
        """Load data from array or HDF5 file."""
        if isinstance(data, np.ndarray):
            return data

        # Load from file
        path = Path(data)
        if not path.exists():
            raise FileNotFoundError(f"{name} file not found: {path}")

        with h5py.File(path, "r") as f:
            # Try common HDF5 dataset names
            for key in ["main", "data", "volume", "image", "label"]:
                if key in f:
                    return f[key][:]

            # Use first dataset
            first_key = list(f.keys())[0]
            warnings.warn(
                f"Using first dataset '{first_key}' from {path}. Available keys: {list(f.keys())}"
            )
            return f[first_key][:]

    def _validate_data(self):
        """Validate data shapes and types."""
        # Validate number of volumes match
        n_pred = len(self.predictions)
        n_gt = len(self.ground_truth)
        if n_pred != n_gt:
            raise ValueError(
                f"Number of prediction volumes ({n_pred}) must match "
                f"number of ground truth volumes ({n_gt})"
            )

        # Validate each volume pair
        for i, (pred, gt) in enumerate(zip(self.predictions, self.ground_truth)):
            # Predictions should be (C, D, H, W)
            if pred.ndim != 4:
                raise ValueError(
                    f"Prediction volume {i} should be 4D (C, D, H, W), got shape {pred.shape}"
                )

            # Ground truth should be (D, H, W)
            if gt.ndim != 3:
                raise ValueError(
                    f"Ground truth volume {i} should be 3D (D, H, W), got shape {gt.shape}"
                )

            # Check spatial dimensions match
            if pred.shape[1:] != gt.shape:
                raise ValueError(
                    f"Spatial dimensions mismatch in volume {i}: "
                    f"predictions {pred.shape[1:]} vs ground_truth {gt.shape}"
                )

        # Validate masks if provided
        if self.mask is not None:
            if len(self.mask) != n_pred:
                raise ValueError(
                    f"Number of mask volumes ({len(self.mask)}) must match "
                    f"number of prediction volumes ({n_pred})"
                )
            for i, (mask, gt) in enumerate(zip(self.mask, self.ground_truth)):
                if mask.ndim != 3:
                    raise ValueError(
                        f"Mask volume {i} should be 3D (D, H, W), got shape {mask.shape}"
                    )
                if mask.shape != gt.shape:
                    raise ValueError(
                        f"Mask shape {mask.shape} doesn't match ground truth shape {gt.shape} in volume {i}"
                    )

    def optimize(self) -> optuna.Study:
        """
        Run Optuna optimization.

        Returns:
            Optuna study object with optimization results
        """
        # Create sampler
        sampler = self._create_sampler()

        # Create pruner
        pruner = self._create_pruner()

        # Get optimization direction
        direction = self._get_optimization_direction()

        # Create storage directory if using SQLite
        storage = self.tune_cfg.storage
        if storage and storage.startswith("sqlite:///"):
            # Extract database file path from SQLite URL
            db_path = storage.replace("sqlite:///", "")
            db_path_obj = Path(db_path)
            db_dir = db_path_obj.parent
            db_dir.mkdir(parents=True, exist_ok=True)
            print(f"📁 Created optuna storage directory: {db_dir}")

            # Ensure database file has write permissions if it exists
            if db_path_obj.exists():
                db_path_obj.chmod(0o664)  # rw-rw-r--
                print(f"✓ Set write permissions on database file: {db_path_obj}")

        # Create or load study
        study = optuna.create_study(
            study_name=self.tune_cfg.study_name,
            storage=storage,
            load_if_exists=self.tune_cfg.load_if_exists,
            sampler=sampler,
            pruner=pruner,
            direction=direction,
        )

        # Run optimization
        n_trials = self.tune_cfg.n_trials
        timeout = self.tune_cfg.timeout

        print(f"\n{'=' * 80}")
        print(f"Starting Optuna optimization: {self.tune_cfg.study_name}")
        print(f"{'=' * 80}")
        print(f"Trials: {n_trials}")
        print(f"Metric: {self.tune_cfg.optimization['single_objective']['metric']}")
        print(f"Direction: {direction}")
        print(f"{'=' * 80}\n")

        study.optimize(
            self._objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=self.tune_cfg.logging.get("show_progress_bar", True),
        )

        # Print results
        self._print_results(study)

        # Save results
        self._save_results(study)

        return study

    def _create_sampler(self) -> optuna.samplers.BaseSampler:
        """Create Optuna sampler from config."""
        sampler_cfg = self.tune_cfg.sampler
        sampler_name = sampler_cfg["name"]
        sampler_kwargs = sampler_cfg.get("kwargs", {})

        # Convert OmegaConf to dict
        if isinstance(sampler_kwargs, DictConfig):
            sampler_kwargs = OmegaConf.to_container(sampler_kwargs, resolve=True)

        if sampler_name == "TPE":
            return TPESampler(**sampler_kwargs)
        elif sampler_name == "CmaEs":
            return CmaEsSampler(**sampler_kwargs)
        elif sampler_name == "Random":
            return RandomSampler(**sampler_kwargs)
        else:
            raise ValueError(f"Unknown sampler: {sampler_name}")

    def _create_pruner(self) -> Optional[optuna.pruners.BasePruner]:
        """Create Optuna pruner from config."""
        pruner_cfg = self.tune_cfg.pruner

        if pruner_cfg is None or not pruner_cfg.get("enabled", False):
            return None

        pruner_name = pruner_cfg.get("name", "Median")
        pruner_kwargs = pruner_cfg.get("kwargs", {})

        # Convert OmegaConf to dict
        if isinstance(pruner_kwargs, DictConfig):
            pruner_kwargs = OmegaConf.to_container(pruner_kwargs, resolve=True)

        if pruner_name == "Median":
            return MedianPruner(**pruner_kwargs)
        elif pruner_name == "Hyperband":
            return HyperbandPruner(**pruner_kwargs)
        else:
            warnings.warn(f"Unknown pruner: {pruner_name}, using None")
            return None

    def _get_optimization_direction(self) -> str:
        """Get optimization direction from config."""
        opt_cfg = self.tune_cfg.optimization

        if opt_cfg["mode"] == "single":
            return opt_cfg["single_objective"]["direction"]
        else:
            raise NotImplementedError("Multi-objective optimization not yet implemented")

    def _objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna optimization.

        Args:
            trial: Optuna trial object

        Returns:
            Metric value to optimize
        """
        self.trial_count += 1

        # Sample parameters
        params = self._sample_parameters(trial)

        # Reconstruct decoding parameters from sampled values
        decoding_params = self._reconstruct_decoding_params(params)

        # Reconstruct postprocessing parameters if enabled
        postproc_params = None
        if (
            hasattr(self.param_space_cfg, "postprocessing")
            and self.param_space_cfg.postprocessing.enabled
        ):
            postproc_params = self._reconstruct_postproc_params(params)

        # Process each volume and collect metrics
        metric_name = self.tune_cfg.optimization["single_objective"]["metric"]
        volume_metrics = []

        for vol_idx, pred_vol in enumerate(self.predictions):
            # Decode predictions for this volume
            try:
                segmentation = decode_instance_binary_contour_distance(pred_vol, **decoding_params)
            except Exception as e:
                import traceback

                print(f"\n❌ Trial {self.trial_count} failed during decoding (volume {vol_idx}):")
                print(f"   Parameters: {decoding_params}")
                print(f"   Error: {e}")
                print(f"   Traceback:\n{traceback.format_exc()}")
                return (
                    float("-inf")
                    if self._get_optimization_direction() == "maximize"
                    else float("inf")
                )

            # Apply post-processing if enabled
            if postproc_params is not None:
                try:
                    segmentation = remove_small_instances(segmentation, **postproc_params)
                except Exception as e:
                    import traceback

                    print(
                        f"\n❌ Trial {self.trial_count} failed during post-processing (volume {vol_idx}):"
                    )
                    print(f"   Parameters: {postproc_params}")
                    print(f"   Error: {e}")
                    print(f"   Traceback:\n{traceback.format_exc()}")
                    return (
                        float("-inf")
                        if self._get_optimization_direction() == "maximize"
                        else float("inf")
                    )

            # Compute metric for this volume
            try:
                gt_vol = self.ground_truth[vol_idx]
                mask_vol = self.mask[vol_idx] if self.mask is not None else None
                vol_metric = self._compute_metric_single(
                    segmentation, gt_vol, mask_vol, metric_name
                )
                volume_metrics.append(vol_metric)
            except Exception as e:
                import traceback

                print(
                    f"\n❌ Trial {self.trial_count} failed during metric computation (volume {vol_idx}):"
                )
                print(f"   Metric: {metric_name}")
                print(f"   Segmentation shape: {segmentation.shape}, dtype: {segmentation.dtype}")
                print(f"   Unique labels in segmentation: {len(np.unique(segmentation))}")
                print(f"   Error: {e}")
                print(f"   Traceback:\n{traceback.format_exc()}")
                return (
                    float("-inf")
                    if self._get_optimization_direction() == "maximize"
                    else float("inf")
                )

        # Average metrics across volumes
        metric_value = np.mean(volume_metrics)

        # Print progress
        if self.tune_cfg.logging.get("verbose", True):
            direction = self._get_optimization_direction()
            # Show per-volume and average
            vol_str = ", ".join([f"{m:.4f}" for m in volume_metrics])
            print(
                f"Trial {self.trial_count:3d}: {metric_name}=[{vol_str}] avg={metric_value:.6f} ({direction})"
            )

        return metric_value

    def _sample_parameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Sample parameters from search space.

        Args:
            trial: Optuna trial object

        Returns:
            Dictionary of sampled parameter values
        """
        params = {}

        # Sample decoding parameters
        if hasattr(self.param_space_cfg, "decoding") and self.param_space_cfg.decoding.parameters:
            dec_params = self.param_space_cfg.decoding.parameters

            for param_name, param_cfg in dec_params.items():
                param_type = param_cfg["type"]
                param_range = param_cfg["range"]

                if param_type == "float":
                    step = param_cfg.get("step", None)
                    log = param_cfg.get("log", False)

                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_range[0],
                        param_range[1],
                        step=step,
                        log=log,
                    )

                elif param_type == "int":
                    step = param_cfg.get("step", 1)
                    log = param_cfg.get("log", False)

                    params[param_name] = trial.suggest_int(
                        param_name,
                        param_range[0],
                        param_range[1],
                        step=step,
                        log=log,
                    )

                elif param_type == "categorical":
                    params[param_name] = trial.suggest_categorical(param_name, param_cfg.choices)

        # Sample post-processing parameters
        if (
            hasattr(self.param_space_cfg, "postprocessing")
            and self.param_space_cfg.postprocessing.enabled
        ):
            postproc_params = self.param_space_cfg.postprocessing.parameters

            for param_name, param_cfg in postproc_params.items():
                param_type = param_cfg["type"]
                param_range = param_cfg["range"]

                if param_type == "float":
                    step = param_cfg.get("step", None)
                    log = param_cfg.get("log", False)

                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_range[0],
                        param_range[1],
                        step=step,
                        log=log,
                    )

                elif param_type == "int":
                    step = param_cfg.get("step", 1)
                    log = param_cfg.get("log", False)

                    params[param_name] = trial.suggest_int(
                        param_name,
                        param_range[0],
                        param_range[1],
                        step=step,
                        log=log,
                    )

        return params

    def _reconstruct_decoding_params(self, sampled_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reconstruct decoding function parameters from sampled values.

        Handles tuple parameters by grouping _seed and _foreground suffixes.

        Args:
            sampled_params: Dictionary of sampled parameter values

        Returns:
            Dictionary of parameters ready for decoding function
        """
        decoding_defaults = self.param_space_cfg.decoding.defaults
        decoding_params = dict(decoding_defaults)  # Start with defaults

        # Group tuple parameters
        tuple_params = defaultdict(dict)
        scalar_params = {}

        for param_name, value in sampled_params.items():
            # Skip post-processing parameters (check if param is in postprocessing section)
            if (
                hasattr(self.param_space_cfg, "postprocessing")
                and self.param_space_cfg.postprocessing.enabled
                and param_name in self.param_space_cfg.postprocessing.parameters
            ):
                continue

            # Check if this is part of a tuple parameter
            param_cfg = self.param_space_cfg.decoding.parameters.get(param_name, {})
            if "param_group" in param_cfg:
                # This is part of a tuple
                group_name = param_cfg["param_group"]
                tuple_index = param_cfg["tuple_index"]
                tuple_params[group_name][tuple_index] = value
            else:
                # Scalar parameter
                scalar_params[param_name] = value

        # Reconstruct tuples
        for group_name, indexed_values in tuple_params.items():
            # Sort by index and extract values
            sorted_items = sorted(indexed_values.items())
            tuple_values = tuple(val for idx, val in sorted_items)
            decoding_params[group_name] = tuple_values

        # Add scalar parameters
        decoding_params.update(scalar_params)

        return decoding_params

    def _reconstruct_postproc_params(self, sampled_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reconstruct post-processing parameters from sampled values.

        Args:
            sampled_params: Dictionary of sampled parameter values

        Returns:
            Dictionary of parameters for post-processing
        """
        if (
            not hasattr(self.param_space_cfg, "postprocessing")
            or not self.param_space_cfg.postprocessing.enabled
        ):
            return {}

        postproc_defaults = self.param_space_cfg.postprocessing.defaults
        postproc_params = dict(postproc_defaults)  # Start with defaults

        # Update with sampled parameters
        for param_name in self.param_space_cfg.postprocessing.parameters.keys():
            if param_name in sampled_params:
                postproc_params[param_name] = sampled_params[param_name]

        return postproc_params

    def _compute_metric_single(
        self,
        segmentation: np.ndarray,
        ground_truth: np.ndarray,
        mask: Optional[np.ndarray],
        metric_name: str,
    ) -> float:
        """
        Compute evaluation metric for a single volume.

        Args:
            segmentation: Predicted segmentation (D, H, W)
            ground_truth: Ground truth labels (D, H, W)
            mask: Optional mask (D, H, W)
            metric_name: Name of metric to compute

        Returns:
            Metric value
        """
        # Apply mask if provided
        if mask is not None:
            gt_masked = ground_truth * mask
            seg_masked = segmentation * mask
        else:
            gt_masked = ground_truth
            seg_masked = segmentation

        if metric_name == "adapted_rand":
            # Compute adapted Rand error (lower is better)
            are = adapted_rand(seg_masked, gt_masked)
            return are  # Return error directly for minimization

        else:
            raise ValueError(f"Unknown metric: {metric_name}")

    def _print_results(self, study: optuna.Study):
        """Print optimization results."""
        print(f"\n{'=' * 80}")
        print("OPTIMIZATION COMPLETE")
        print(f"{'=' * 80}")
        print(f"Number of finished trials: {len(study.trials)}")
        print(f"\nBest trial: #{study.best_trial.number}")
        print(f"  Value: {study.best_value:.4f}")
        print("\n  Params:")

        # Reconstruct and print parameters
        best_decoding_params = self._reconstruct_decoding_params(study.best_params)
        for key, value in best_decoding_params.items():
            print(f"    {key}: {value}")

        if (
            hasattr(self.param_space_cfg, "postprocessing")
            and self.param_space_cfg.postprocessing.enabled
        ):
            best_postproc_params = self._reconstruct_postproc_params(study.best_params)
            if best_postproc_params:
                print("\n  Post-processing params:")
                for key, value in best_postproc_params.items():
                    print(f"    {key}: {value}")

        print(f"{'=' * 80}\n")

    def _save_results(self, study: optuna.Study):
        """Save optimization results to disk."""
        output_dir = Path(self.tune_cfg.output.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save best parameters
        best_params_file = output_dir / "best_params.yaml"
        best_decoding_params = self._reconstruct_decoding_params(study.best_params)
        best_postproc_params = self._reconstruct_postproc_params(study.best_params)

        # Create YAML content
        params_dict = {
            "best_trial": study.best_trial.number,
            "best_value": float(study.best_value),
            "metric": self.tune_cfg.optimization["single_objective"]["metric"],
            "decoding_params": best_decoding_params,
        }

        if best_postproc_params:
            params_dict["postprocessing_params"] = best_postproc_params

        # Save as YAML
        with open(best_params_file, "w") as f:
            OmegaConf.save(params_dict, f)

        print(f"✓ Best parameters saved to: {best_params_file}")

        # Save study if requested
        if self.tune_cfg.output.save_study:
            if self.tune_cfg.storage:
                print(f"✓ Study saved to database: {self.tune_cfg.storage}")
            else:
                warnings.warn("No storage configured, study not persisted to database")


# ============================================================================
# High-level API Functions
# ============================================================================


def run_tuning(model, trainer, cfg, checkpoint_path=None):
    """
    Run Optuna-based parameter tuning for instance segmentation decoding.

    This function performs automated hyperparameter optimization for post-processing
    parameters (thresholds, sizes, etc.) using Optuna on a validation/tuning dataset.

    Args:
        model: Lightning module (ConnectomicsModule)
        trainer: PyTorch Lightning Trainer
        cfg: Configuration object (Config dataclass)
        checkpoint_path: Optional path to model checkpoint

    Returns:
        None (results are saved to disk)

    Workflow:
        1. Check if best_params.yaml already exists (skip if it does)
        2. Run inference on tune dataset to get predictions
        3. Load ground truth labels for tuning
        4. Create OptunaDecodingTuner instance
        5. Run optimization study
        6. Save best parameters to YAML file

    Example:
        >>> from connectomics.training.lit import ConnectomicsModule, create_trainer
        >>> from connectomics.decoding import run_tuning
        >>> model = ConnectomicsModule(cfg)
        >>> trainer = create_trainer(cfg)
        >>> run_tuning(model, trainer, cfg, checkpoint_path='best.ckpt')
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError(
            "Optuna is required for parameter tuning. Install with: pip install optuna"
        )

    # Get output directory from tune config
    if cfg.tune is None or not hasattr(cfg.tune, "output") or not cfg.tune.output.output_dir:
        raise ValueError("Missing tune.output.output_dir in configuration")

    output_dir = Path(cfg.tune.output.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_params_file = output_dir / "best_params.yaml"

    # Check if best parameters already exist
    if best_params_file.exists():
        print(f"\n{'=' * 80}")
        print("SKIPPING PARAMETER TUNING")
        print(f"{'=' * 80}")
        print(f"✓ Best parameters already exist: {best_params_file}")
        print("  To re-run tuning, delete this file and run again.")
        return

    print(f"\n{'=' * 80}")
    print("STARTING PARAMETER TUNING")
    print(f"{'=' * 80}")
    print(f"Output directory: {output_dir}")

    # Step 1: Run inference on tune dataset
    from connectomics.training.lit import create_datamodule
    from connectomics.data.io import read_volume
    import glob

    print("\n[1/4] Running inference on tuning dataset...")

    # Get tune config sections (used later for loading predictions, ground truth, masks)
    tune_data = cfg.tune.data
    tune_output = cfg.tune.output

    # Create datamodule with tune mode (reads from cfg.tune.data)
    # Uses inference settings from cfg.inference (sliding window, TTA, save_predictions, etc.)
    datamodule = create_datamodule(cfg, mode="tune")

    # Run test (will check for cached files and skip inference if they exist)
    # test_step will read output path and cache suffix from cfg.tune.output
    results = trainer.test(model, datamodule=datamodule, ckpt_path=checkpoint_path)

    print(f"✓ Test completed. Results: {results}")

    # Step 2: Load predictions from saved files
    print("\n[2/4] Loading predictions from saved files...")
    output_pred_dir = (
        tune_output.output_pred
        if tune_output.output_pred is not None
        else str(output_dir.parent / "results")
    )
    cache_suffix = tune_output.cache_suffix
    predictions_dir = Path(output_pred_dir)

    # Find all prediction files using cache_suffix from config
    pred_pattern = f"*{cache_suffix}"
    pred_files = sorted(glob.glob(str(predictions_dir / pred_pattern)))

    if not pred_files:
        raise FileNotFoundError(
            f"No prediction files found in: {predictions_dir}\n"
            f"Expected files matching pattern: {pred_pattern}"
        )

    print(f"  Found {len(pred_files)} prediction file(s)")

    # Load all prediction files (keep as list for per-volume processing)
    all_predictions = []
    for pred_file in pred_files:
        pred = read_volume(pred_file)
        print(f"  ✓ Loaded {Path(pred_file).name}: shape {pred.shape}")
        all_predictions.append(pred)

    # Keep as list for per-volume metric computation
    predictions = all_predictions

    print(f"✓ Loaded {len(predictions)} prediction volume(s)")
    for i, pred in enumerate(predictions):
        print(
            f"  Volume {i}: shape {pred.shape}, dtype {pred.dtype}, range [{pred.min():.3f}, {pred.max():.3f}]"
        )

    # Step 3: Load ground truth
    print("\n[3/4] Loading ground truth labels...")
    tune_label_pattern = tune_data.tune_label

    if tune_label_pattern is None:
        raise ValueError("Missing tune.data.tune_label in configuration")

    # Handle both string (glob pattern) and list of strings (explicit file paths)
    if isinstance(tune_label_pattern, list):
        # If it's a list, use the files directly (no glob needed)
        label_files = sorted(tune_label_pattern)
    else:
        # If it's a string, treat it as a glob pattern
        label_files = sorted(glob.glob(tune_label_pattern))

    if not label_files:
        raise FileNotFoundError(f"No label files found matching pattern: {tune_label_pattern}")

    print(f"  Found {len(label_files)} label file(s)")

    # Load all label files (keep as list for per-volume processing)
    all_labels = []
    for label_file in label_files:
        label = read_volume(label_file)
        print(f"  ✓ Loaded {Path(label_file).name}: shape {label.shape}")
        all_labels.append(label)

    # Keep as list for per-volume metric computation
    ground_truth = all_labels

    print(f"✓ Loaded {len(ground_truth)} ground truth volume(s)")
    for i, gt in enumerate(ground_truth):
        unique_labels = np.unique(gt)
        n_nonzero_labels = len(unique_labels) - (1 if 0 in unique_labels else 0)
        print(
            f"  Volume {i}: shape {gt.shape}, dtype {gt.dtype}, range [{gt.min()}, {gt.max()}], unique labels: {n_nonzero_labels}"
        )

        # Validate ground truth for this volume
        if n_nonzero_labels == 0:
            raise ValueError(f"⚠️  Ground truth volume {i} contains no non-zero labels!")
        if np.all(gt == 0):
            raise ValueError(f"⚠️  Ground truth volume {i} is all zeros!")
        if n_nonzero_labels == 1:
            warnings.warn(f"⚠️  Ground truth volume {i} contains only one non-zero label.")

    # Load mask if available
    mask = None
    tune_mask_pattern = tune_data.tune_mask
    if tune_mask_pattern:
        # Handle both string (glob pattern) and list of strings (explicit file paths)
        if isinstance(tune_mask_pattern, list):
            # If it's a list, use the files directly (no glob needed)
            mask_files = sorted(tune_mask_pattern)
        else:
            # If it's a string, treat it as a glob pattern
            mask_files = sorted(glob.glob(tune_mask_pattern))
        if not mask_files:
            print(f"  ⚠️  No mask files found matching pattern: {tune_mask_pattern}")
        else:
            print(f"  Found {len(mask_files)} mask file(s)")
            all_masks = []
            for mask_file in mask_files:
                m = read_volume(mask_file)
                print(f"  ✓ Loaded {Path(mask_file).name}: shape {m.shape}")
                all_masks.append(m)

            # Keep as list for per-volume processing
            mask = all_masks
            print(f"✓ Loaded {len(mask)} mask volume(s)")

    # Step 4: Validate predictions and ground truth alignment
    print("\n[4/5] Validating data alignment...")
    # Check number of volumes match
    if len(predictions) != len(ground_truth):
        raise ValueError(
            f"⚠️  Number of prediction volumes ({len(predictions)}) doesn't match "
            f"ground truth volumes ({len(ground_truth)})"
        )

    # Check spatial dimensions match for each volume
    for i, (pred, gt) in enumerate(zip(predictions, ground_truth)):
        if pred.shape[1:] != gt.shape:
            raise ValueError(
                f"⚠️  Spatial dimension mismatch in volume {i}!\n"
                f"   Predictions spatial shape: {pred.shape[1:]}\n"
                f"   Ground truth shape: {gt.shape}"
            )
    print(f"✓ All {len(predictions)} volumes have matching spatial dimensions")

    # Step 5: Create tuner and run optimization
    print("\n[5/6] Creating Optuna tuner...")
    tuner = OptunaDecodingTuner(
        cfg=cfg, predictions=predictions, ground_truth=ground_truth, mask=mask
    )

    print("\n[6/6] Running optimization study...")
    study = tuner.optimize()

    print(f"\n{'=' * 80}")
    print("TUNING COMPLETED")
    print(f"{'=' * 80}")
    print(f"✓ Best parameters saved to: {best_params_file}")
    print("\nBest trial:")
    print(f"  Value: {study.best_value:.4f}")
    print(f"  Parameters: {study.best_params}")


def load_and_apply_best_params(cfg):
    """
    Load best parameters from Optuna tuning and apply them to test config.

    This function loads the best_params.yaml file generated by run_tuning()
    and updates the test.decoding section of the config with optimized parameters.

    Args:
        cfg: Configuration object (Config dataclass)

    Returns:
        cfg: Updated configuration object with best parameters applied

    Example:
        >>> cfg = load_config('tutorials/hydra-lv.yaml')
        >>> cfg = load_and_apply_best_params(cfg)
        >>> # cfg.test now has optimized decoding parameters
    """
    # Get output directory from tune config
    if cfg.tune is None or not hasattr(cfg.tune, "output") or not cfg.tune.output.output_dir:
        raise ValueError("Missing tune.output.output_dir in configuration")

    output_dir = Path(cfg.tune.output.output_dir)
    best_params_file = output_dir / "best_params.yaml"

    if not best_params_file.exists():
        raise FileNotFoundError(
            f"Best parameters file not found: {best_params_file}\n"
            f"Run parameter tuning first with --mode tune"
        )

    print(f"Loading best parameters from: {best_params_file}")

    # Load best parameters
    best_params = OmegaConf.load(best_params_file)

    print("✓ Loaded best parameters:")
    print(OmegaConf.to_yaml(best_params))

    # Apply to test.decoding config
    # Note: test is Dict[str, Any], so we need to handle it carefully
    if cfg.test is None:
        cfg.test = {}

    if "decoding" not in cfg.test:
        cfg.test["decoding"] = []

    # Find the decoding function in test.decoding that matches the tuned function
    decoding_function = best_params.get("decoding_function", None)

    if decoding_function is None:
        warnings.warn("No decoding_function found in best_params, applying to first decoder")
        decoder_idx = 0
    else:
        # Find decoder with matching function name
        decoder_idx = None
        for idx, decoder in enumerate(cfg.test["decoding"]):
            if decoder.get("name") == decoding_function:
                decoder_idx = idx
                break

        if decoder_idx is None:
            # Create new decoder entry
            decoder_idx = len(cfg.test["decoding"])
            cfg.test["decoding"].append({"name": decoding_function, "kwargs": {}})

    # Update parameters
    if decoder_idx < len(cfg.test["decoding"]):
        decoder = cfg.test["decoding"][decoder_idx]
        if "kwargs" not in decoder:
            decoder["kwargs"] = {}

        # Apply best parameters
        decoder["kwargs"].update(OmegaConf.to_container(best_params["parameters"]))

        print(f"✓ Applied best parameters to test.decoding[{decoder_idx}]")

    return cfg

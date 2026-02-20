"""
Automatic Hyperparameter Configuration System.

Inspired by nnUNet's experiment planning, this module automatically determines
optimal hyperparameters based on:
- Available GPU memory
- Dataset characteristics (spacing, size)
- Model architecture
- Training strategy

Users can manually override any auto-determined parameters.
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from omegaconf import OmegaConf, DictConfig
import warnings
import os

from .gpu_utils import (
    get_gpu_info,
    suggest_batch_size,
    get_optimal_num_workers,
    estimate_gpu_memory_required,
)


def _available_cpus_for_current_run() -> int:
    """
    Detect CPU slots available to the current process (SLURM/cgroup aware).

    Priority:
    1) CPU affinity mask (best under cgroups/SLURM)
    2) SLURM_CPUS_PER_TASK
    3) os.cpu_count()
    """
    try:
        affinity = os.sched_getaffinity(0)
        if affinity:
            return len(affinity)
    except Exception:
        pass

    slurm_cpus_per_task = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus_per_task and slurm_cpus_per_task.isdigit():
        return max(int(slurm_cpus_per_task), 1)

    return max(os.cpu_count() or 1, 1)


def resolve_runtime_resource_sentinels(
    config: DictConfig,
    print_results: bool = True,
) -> DictConfig:
    """
    Resolve runtime resource sentinels in system.{training,inference}.

    Sentinel convention:
      - num_gpus = -1 -> use all GPUs visible to this run
      - num_workers = -1 -> use all CPU slots available to this run

    This is runtime-oriented (SLURM/cgroup aware) and complements auto-planning.
    """
    if not hasattr(config, "system"):
        return config

    gpu_info = get_gpu_info()
    available_gpus = gpu_info["num_gpus"] if gpu_info["cuda_available"] else 0
    available_cpus = _available_cpus_for_current_run()

    for section_name in ("training", "inference"):
        section = getattr(config.system, section_name, None)
        if section is None:
            continue

        if getattr(section, "num_gpus", None) == -1:
            section.num_gpus = available_gpus
            if print_results:
                print(
                    f"🔧 Auto-detected system.{section_name}.num_gpus: "
                    f"-1 → {section.num_gpus}"
                )

        if getattr(section, "num_workers", None) == -1:
            section.num_workers = available_cpus
            if print_results:
                print(
                    f"🔧 Auto-detected system.{section_name}.num_workers: "
                    f"-1 → {section.num_workers}"
                )

        if getattr(section, "num_gpus", 0) < -1:
            raise ValueError(f"system.{section_name}.num_gpus must be >= -1")
        if getattr(section, "num_workers", 0) < -1:
            raise ValueError(f"system.{section_name}.num_workers must be >= -1")

    return config


@dataclass
class AutoPlanResult:
    """Results from automatic planning."""

    # Data parameters
    patch_size: List[int] = field(default_factory=list)
    batch_size: int = 2
    num_workers: int = 4

    # Model parameters
    base_features: int = 32
    max_features: int = 320

    # Training parameters
    precision: str = "16-mixed"
    accumulate_grad_batches: int = 1

    # Learning rate
    lr: float = 1e-3

    # GPU info
    gpu_memory_per_sample_gb: float = 0.0
    estimated_gpu_memory_gb: float = 0.0
    available_gpu_memory_gb: float = 0.0

    # Metadata
    auto_planned: bool = True
    planning_notes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class AutoConfigPlanner:
    """
    Automatic configuration planner based on GPU capabilities and dataset properties.

    Similar to nnUNet's experiment planning but adapted for PyTorch Lightning + MONAI.
    """

    def __init__(
        self,
        architecture: str = "mednext",
        target_spacing: Optional[List[float]] = None,
        median_shape: Optional[List[int]] = None,
        manual_overrides: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize auto planner.

        Args:
            architecture: Model architecture name
            target_spacing: Target voxel spacing [z, y, x] in mm
            median_shape: Median dataset shape [D, H, W]
            manual_overrides: Dict of manual parameter overrides
        """
        self.architecture = architecture
        self.target_spacing = target_spacing or [1.0, 1.0, 1.0]
        self.median_shape = median_shape or [128, 128, 128]
        self.manual_overrides = manual_overrides or {}

        # Get GPU info
        self.gpu_info = get_gpu_info()

        # Architecture-specific defaults
        self.arch_defaults = self._get_architecture_defaults()

    def _get_architecture_defaults(self) -> Dict[str, Any]:
        """Get architecture-specific default parameters."""
        defaults = {
            "mednext": {
                "base_features": 32,
                "max_features": 320,
                "lr": 1e-3,  # MedNeXt paper recommends 1e-3
                "use_scheduler": False,  # MedNeXt uses constant LR
            },
            "mednext_custom": {
                "base_features": 32,
                "max_features": 320,
                "lr": 1e-3,
                "use_scheduler": False,
            },
            "monai_basic_unet3d": {
                "base_features": 32,
                "max_features": 512,
                "lr": 1e-4,
                "use_scheduler": True,
            },
            "monai_unet": {
                "base_features": 32,
                "max_features": 512,
                "lr": 1e-4,
                "use_scheduler": True,
            },
        }

        return defaults.get(self.architecture, defaults["monai_basic_unet3d"])

    def plan(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
        deep_supervision: bool = False,
        use_mixed_precision: bool = True,
    ) -> AutoPlanResult:
        """
        Plan optimal hyperparameters.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output classes
            deep_supervision: Whether to use deep supervision
            use_mixed_precision: Whether to use mixed precision training

        Returns:
            AutoPlanResult with planned hyperparameters
        """
        result = AutoPlanResult()
        result.planning_notes.append(f"Architecture: {self.architecture}")

        # Step 1: Determine patch size
        patch_size = self._plan_patch_size()
        result.patch_size = patch_size
        result.planning_notes.append(f"Patch size: {patch_size}")

        # Step 2: Get model parameters
        result.base_features = self.arch_defaults["base_features"]
        result.max_features = self.arch_defaults["max_features"]

        # Step 3: Determine precision
        result.precision = "16-mixed" if use_mixed_precision else "32"

        # Step 4: Estimate memory and determine batch size
        if not self.gpu_info["cuda_available"]:
            result.batch_size = 1
            result.precision = "32"  # CPU doesn't support mixed precision well
            result.warnings.append("CUDA not available, using CPU with batch_size=1")
            result.planning_notes.append("Training on CPU (slow!)")
        else:
            gpu_memory_gb = self.gpu_info["available_memory_gb"][0]  # Use first GPU
            result.available_gpu_memory_gb = gpu_memory_gb

            # Calculate number of pooling stages (log2 of patch size / 4)
            num_pool_stages = int(np.log2(min(patch_size) / 4))

            # Suggest batch size
            batch_size = suggest_batch_size(
                patch_size=tuple(patch_size),
                in_channels=in_channels,
                out_channels=out_channels,
                available_gpu_memory_gb=gpu_memory_gb,
                base_features=result.base_features,
                num_pool_stages=num_pool_stages,
                deep_supervision=deep_supervision,
                mixed_precision=use_mixed_precision,
            )
            result.batch_size = batch_size

            # Estimate actual memory usage
            result.estimated_gpu_memory_gb = estimate_gpu_memory_required(
                patch_size=tuple(patch_size),
                batch_size=batch_size,
                in_channels=in_channels,
                out_channels=out_channels,
                base_features=result.base_features,
                num_pool_stages=num_pool_stages,
                deep_supervision=deep_supervision,
                mixed_precision=use_mixed_precision,
            )
            result.gpu_memory_per_sample_gb = result.estimated_gpu_memory_gb / batch_size

            result.planning_notes.append(
                f"GPU: {self.gpu_info['gpu_names'][0]} ({gpu_memory_gb:.1f} GB available)"
            )
            result.planning_notes.append(
                f"Estimated memory: {result.estimated_gpu_memory_gb:.2f} GB "
                f"({result.estimated_gpu_memory_gb / gpu_memory_gb * 100:.1f}% of GPU)"
            )
            result.planning_notes.append(f"Batch size: {batch_size}")

            # Gradient accumulation if batch size is very small
            if batch_size == 1:
                result.accumulate_grad_batches = 4
                result.planning_notes.append(
                    "Using gradient accumulation (4 batches) for effective batch_size=4"
                )

        # Step 5: Determine num_workers
        num_gpus = self.gpu_info["num_gpus"] if self.gpu_info["cuda_available"] else 0
        result.num_workers = get_optimal_num_workers(num_gpus)
        result.planning_notes.append(f"Num workers: {result.num_workers}")

        # Step 6: Learning rate
        result.lr = self.arch_defaults["lr"]
        result.planning_notes.append(f"Learning rate: {result.lr}")

        # Step 7: Apply manual overrides
        if self.manual_overrides:
            result.planning_notes.append("Manual overrides applied:")
            for key, value in self.manual_overrides.items():
                if hasattr(result, key):
                    old_value = getattr(result, key)
                    setattr(result, key, value)
                    result.planning_notes.append(f"  {key}: {old_value} → {value}")

        return result

    def _plan_patch_size(self) -> List[int]:
        """
        Determine optimal patch size based on spacing and median shape.

        Strategy:
        1. Start with median shape
        2. Adjust based on target spacing (prefer isotropic)
        3. Ensure divisible by 2^n for pooling
        4. Consider GPU memory constraints
        """
        # Start with median shape
        patch_size = np.array(self.median_shape, dtype=np.int32)

        # If anisotropic spacing, adjust patch size to be more isotropic
        spacing_ratio = np.max(self.target_spacing) / np.min(self.target_spacing)
        if spacing_ratio > 3:
            # Anisotropic data (e.g., medical CT with thick slices)
            # Reduce patch size in high-resolution dimensions
            warnings.warn(
                f"Anisotropic spacing detected (ratio={spacing_ratio:.1f}). "
                f"Adjusting patch size for balanced receptive field."
            )

            # Normalize spacing
            norm_spacing = np.array(self.target_spacing) / np.min(self.target_spacing)
            # Adjust patch size inversely proportional to spacing
            patch_size = (patch_size / np.sqrt(norm_spacing)).astype(np.int32)

        # Ensure patch size is reasonable (not too small, not too large)
        patch_size = np.clip(patch_size, 32, 256)

        # Make patch size divisible by 16 (for 4 pooling stages: 2^4 = 16)
        patch_size = ((patch_size + 15) // 16) * 16

        # If GPU memory is limited, may need to reduce patch size
        # (This is a simplified heuristic)
        if self.gpu_info["cuda_available"]:
            gpu_memory_gb = self.gpu_info["available_memory_gb"][0]
            if gpu_memory_gb < 8:
                # Very limited GPU, use smaller patches
                patch_size = np.minimum(patch_size, [64, 64, 64])
            elif gpu_memory_gb < 12:
                # Limited GPU, use medium patches
                patch_size = np.minimum(patch_size, [128, 128, 128])

        return patch_size.tolist()

    def print_plan(self, result: AutoPlanResult):
        """Print formatted planning results."""
        print("=" * 70)
        print("🤖 Automatic Configuration Planning Results")
        print("=" * 70)
        print()

        print("📊 Data Configuration:")
        print(f"  Patch Size: {result.patch_size}")
        print(f"  Batch Size: {result.batch_size}")
        if result.accumulate_grad_batches > 1:
            effective_bs = result.batch_size * result.accumulate_grad_batches
            print(
                f"  Gradient Accumulation: {result.accumulate_grad_batches} "
                f"(effective batch_size={effective_bs})"
            )
        print(f"  Num Workers: {result.num_workers}")
        print()

        print("🧠 Model Configuration:")
        print(f"  Base Features: {result.base_features}")
        print(f"  Max Features: {result.max_features}")
        print()

        print("⚙️  Training Configuration:")
        print(f"  Precision: {result.precision}")
        print(f"  Learning Rate: {result.lr}")
        print()

        if result.available_gpu_memory_gb > 0:
            print("💾 GPU Memory:")
            print(f"  Available: {result.available_gpu_memory_gb:.2f} GB")
            print(
                f"  Estimated Usage: {result.estimated_gpu_memory_gb:.2f} GB "
                f"({result.estimated_gpu_memory_gb / result.available_gpu_memory_gb * 100:.1f}%)"
            )
            print(f"  Per Sample: {result.gpu_memory_per_sample_gb:.2f} GB")
            print()

        if result.warnings:
            print("⚠️  Warnings:")
            for warning in result.warnings:
                print(f"  - {warning}")
            print()

        print("📝 Planning Notes:")
        for note in result.planning_notes:
            print(f"  • {note}")
        print()

        print("=" * 70)
        print("💡 Tip: You can manually override any of these values in your config!")
        print("=" * 70)


def auto_plan_config(
    config: DictConfig,
    print_results: bool = True,
) -> DictConfig:
    """
    Automatically plan hyperparameters and update config.

    This function will:
    1. Read dataset properties from config
    2. Query GPU capabilities
    3. Plan optimal hyperparameters
    4. Update config with planned values (respecting manual overrides)

    Args:
        config: OmegaConf config object
        print_results: Whether to print planning results

    Returns:
        Updated config with auto-planned parameters
    """
    # Check if auto-planning is disabled
    if hasattr(config, "system") and hasattr(config.system, "auto_plan"):
        if not config.system.auto_plan:
            print("ℹ️  Auto-planning disabled in config")
            return config

    # Extract relevant config values
    architecture = config.model.architecture if hasattr(config.model, "architecture") else "mednext"
    in_channels = config.model.in_channels if hasattr(config.model, "in_channels") else 1
    out_channels = config.model.out_channels if hasattr(config.model, "out_channels") else 2
    deep_supervision = (
        config.model.deep_supervision if hasattr(config.model, "deep_supervision") else False
    )

    # Get target spacing and median shape if provided
    target_spacing = None
    if hasattr(config, "data") and hasattr(config.data, "target_spacing"):
        target_spacing = config.data.target_spacing

    median_shape = None
    if hasattr(config, "data") and hasattr(config.data, "median_shape"):
        median_shape = config.data.median_shape

    # Collect manual overrides (values explicitly set in config)
    manual_overrides = {}
    training_cfg = getattr(config.system, "training", None) if hasattr(config, "system") else None
    if hasattr(config, "data"):
        if training_cfg and getattr(training_cfg, "batch_size", None) is not None:
            manual_overrides["batch_size"] = training_cfg.batch_size
        if training_cfg and getattr(training_cfg, "num_workers", None) is not None:
            manual_overrides["num_workers"] = training_cfg.num_workers
        if hasattr(config.data, "patch_size") and config.data.patch_size is not None:
            manual_overrides["patch_size"] = config.data.patch_size

    if hasattr(config, "optimization"):
        if getattr(config.optimization, "precision", None) is not None:
            manual_overrides["precision"] = config.optimization.precision
        if getattr(config.optimization, "accumulate_grad_batches", None) is not None:
            manual_overrides["accumulate_grad_batches"] = (
                config.optimization.accumulate_grad_batches
            )

        opt_cfg = getattr(config.optimization, "optimizer", None)
        if opt_cfg and getattr(opt_cfg, "lr", None) is not None:
            manual_overrides["lr"] = opt_cfg.lr

    # Create planner
    planner = AutoConfigPlanner(
        architecture=architecture,
        target_spacing=target_spacing,
        median_shape=median_shape,
        manual_overrides=manual_overrides,
    )

    # Plan
    use_mixed_precision = not (
        hasattr(config, "optimization") and getattr(config.optimization, "precision", None) == "32"
    )

    result = planner.plan(
        in_channels=in_channels,
        out_channels=out_channels,
        deep_supervision=deep_supervision,
        use_mixed_precision=use_mixed_precision,
    )

    # Update config with planned values (if not manually overridden)
    OmegaConf.set_struct(config, False)  # Allow adding new fields

    if "batch_size" not in manual_overrides and training_cfg is not None:
        training_cfg.batch_size = result.batch_size
    if "num_workers" not in manual_overrides and training_cfg is not None:
        training_cfg.num_workers = result.num_workers
    if "patch_size" not in manual_overrides:
        config.data.patch_size = result.patch_size

    if "precision" not in manual_overrides:
        config.optimization.precision = result.precision
    if "accumulate_grad_batches" not in manual_overrides:
        config.optimization.accumulate_grad_batches = result.accumulate_grad_batches

    if "lr" not in manual_overrides and hasattr(config, "optimization"):
        config.optimization.optimizer.lr = result.lr

    OmegaConf.set_struct(config, True)  # Re-enable struct mode

    # Print results
    if print_results:
        planner.print_plan(result)

    return config


if __name__ == "__main__":
    # Test auto planning
    from connectomics.config import Config

    # Create test config
    cfg = OmegaConf.structured(Config())
    cfg.model.architecture = "mednext"
    cfg.model.deep_supervision = True

    # Auto plan
    cfg = auto_plan_config(cfg, print_results=True)

    print("\nFinal Config Values:")
    print(f"  batch_size: {cfg.system.training.batch_size}")
    print(f"  patch_size: {cfg.data.patch_size}")
    print(f"  precision: {cfg.optimization.precision}")
    print(f"  lr: {cfg.optimization.optimizer.lr}")

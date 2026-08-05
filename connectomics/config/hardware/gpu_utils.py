"""
GPU and System Information Utilities.

Provides functions to query GPU memory, count available GPUs,
and estimate memory requirements for training.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Tuple

import psutil
import torch


SUPPORTED_ACCELERATORS = ("auto", "cpu", "cuda", "mps")


def is_mps_available() -> bool:
    """Return whether PyTorch can use Apple's Metal Performance Shaders backend."""
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is None:
        return False
    try:
        return bool(mps_backend.is_available())
    except Exception:
        return False


def resolve_accelerator_type(requested: str = "auto") -> str:
    """Resolve an accelerator preference to ``cpu``, ``cuda``, or ``mps``.

    CUDA takes precedence over MPS for ``auto`` so existing NVIDIA hosts retain
    their current behavior. Explicit unavailable accelerators fail instead of
    silently running a job on a different device.
    """
    normalized = str(requested or "auto").strip().lower()
    if normalized not in SUPPORTED_ACCELERATORS:
        choices = ", ".join(SUPPORTED_ACCELERATORS)
        raise ValueError(f"system.accelerator must be one of: {choices} (got {requested!r})")

    if normalized == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if is_mps_available():
            return "mps"
        return "cpu"
    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("system.accelerator='cuda' was requested but CUDA is not available")
    if normalized == "mps" and not is_mps_available():
        raise RuntimeError("system.accelerator='mps' was requested but MPS is not available")
    return normalized


def get_accelerator_device_count(accelerator: str = "auto") -> int:
    """Return the number of devices available for the resolved accelerator."""
    resolved = resolve_accelerator_type(accelerator)
    if resolved == "cuda":
        try:
            return int(torch.cuda.device_count())
        except Exception as exc:
            warnings.warn(f"Failed to query CUDA device count: {exc}")
            return 0
    if resolved == "mps":
        # PyTorch exposes Apple silicon's unified GPU as one MPS device.
        return 1
    return 0


def empty_accelerator_cache() -> str | None:
    """Release cached CUDA or MPS allocator memory and return the backend used."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        return "cuda"
    if is_mps_available():
        mps_module = getattr(torch, "mps", None)
        if mps_module is not None:
            mps_module.empty_cache()
            return "mps"
    return None


def get_gpu_info() -> Dict[str, Any]:
    """
    Get comprehensive GPU information.

    Returns:
        dict: Dictionary containing GPU information:
            - num_gpus: Number of available GPUs
            - gpu_names: List of GPU names
            - total_memory_gb: List of total memory per GPU in GB
            - available_memory_gb: List of available memory per GPU in GB
            - accelerator: Selected accelerator (``cuda``, ``mps``, or ``cpu``)
            - cuda_available: Whether CUDA is available
            - mps_available: Whether Apple MPS is available
    """
    cuda_available = torch.cuda.is_available()
    mps_available = is_mps_available()
    accelerator = "cuda" if cuda_available else "mps" if mps_available else "cpu"
    info = {
        "accelerator": accelerator,
        "cuda_available": cuda_available,
        "mps_available": mps_available,
        "num_gpus": 0,
        "gpu_names": [],
        "total_memory_gb": [],
        "available_memory_gb": [],
    }

    if not cuda_available:
        if mps_available:
            # Apple GPUs use unified memory. System RAM is the meaningful upper
            # bound, while currently available RAM is the safest planning budget.
            info["num_gpus"] = 1
            info["gpu_names"] = ["Apple silicon GPU (MPS)"]
            info["total_memory_gb"] = [get_system_memory_gb()]
            info["available_memory_gb"] = [get_available_system_memory_gb()]
        return info

    try:
        info["num_gpus"] = torch.cuda.device_count()
    except Exception as e:
        warnings.warn(f"Failed to query CUDA device count: {e}")
        return info

    for i in range(info["num_gpus"]):
        try:
            # Get GPU name
            info["gpu_names"].append(torch.cuda.get_device_name(i))

            # Get memory info
            props = torch.cuda.get_device_properties(i)
            total_memory = props.total_memory / (1024**3)  # Convert to GB
            info["total_memory_gb"].append(total_memory)

            # Try to get available memory (may require GPU to be initialized)
            try:
                torch.cuda.set_device(i)
                torch.cuda.empty_cache()
                available_memory = (props.total_memory - torch.cuda.memory_allocated(i)) / (1024**3)
                info["available_memory_gb"].append(available_memory)
            except Exception:
                # Fallback: assume 90% is available
                info["available_memory_gb"].append(total_memory * 0.9)
        except Exception as e:
            # Keep the process alive when a specific GPU index is invalid/broken.
            # This is especially useful on clusters with dead devices or mismatched
            # CUDA_VISIBLE_DEVICES / SLURM GPU cgroup mappings.
            warnings.warn(f"Skipping GPU index {i} during CUDA probe: {e}")
            info["gpu_names"].append(f"<unavailable:{i}>")
            info["total_memory_gb"].append(0.0)
            info["available_memory_gb"].append(0.0)

    return info


def get_system_memory_gb() -> float:
    """Get total system RAM in GB."""
    return psutil.virtual_memory().total / (1024**3)


def get_available_system_memory_gb() -> float:
    """Get available system RAM in GB."""
    return psutil.virtual_memory().available / (1024**3)


def estimate_gpu_memory_required(
    patch_size: Tuple[int, int, int],
    batch_size: int,
    in_channels: int,
    out_channels: int,
    base_features: int = 32,
    num_pool_stages: int = 4,
    deep_supervision: bool = False,
    mixed_precision: bool = True,
) -> float:
    """
    Estimate GPU memory requirement in GB for training.

    Based on nnUNet's VRAM estimation approach but simplified.
    This is a rough estimate and should be used with a safety margin.

    Args:
        patch_size: Input patch size (D, H, W)
        batch_size: Batch size
        in_channels: Number of input channels
        out_channels: Number of output classes
        base_features: Base number of features (e.g., 32 for MedNeXt)
        num_pool_stages: Number of pooling stages
        deep_supervision: Whether deep supervision is used
        mixed_precision: Whether mixed precision (FP16) is used

    Returns:
        float: Estimated GPU memory in GB
    """
    import numpy as np

    # Calculate feature maps for each stage
    current_size = np.array(patch_size, dtype=np.float64)
    total_voxels = 0
    num_features = base_features

    # Input
    total_voxels += np.prod(current_size) * in_channels * batch_size

    # Encoder
    for stage in range(num_pool_stages + 1):
        # Each stage has ~3 conv layers, each producing feature maps
        total_voxels += np.prod(current_size) * num_features * 3 * batch_size

        # Deep supervision outputs
        if deep_supervision and stage > 0 and stage < num_pool_stages:
            total_voxels += np.prod(current_size) * out_channels * batch_size

        # Pooling (divide by 2 in each dimension)
        current_size = current_size / 2
        num_features = min(num_features * 2, 320)  # Cap at 320 like nnUNet

    # Decoder (mirror of encoder)
    for stage in range(num_pool_stages):
        current_size = current_size * 2
        num_features = num_features // 2
        total_voxels += np.prod(current_size) * num_features * 3 * batch_size

        if deep_supervision and stage < num_pool_stages - 1:
            total_voxels += np.prod(current_size) * out_channels * batch_size

    # Output
    current_size = np.array(patch_size, dtype=np.float64)
    total_voxels += np.prod(current_size) * out_channels * batch_size

    # Bytes per element (4 for FP32, 2 for FP16)
    bytes_per_element = 2 if mixed_precision else 4

    # Estimate memory:
    # - Feature maps (activations): total_voxels * bytes_per_element
    # - Gradients (same size as activations): total_voxels * bytes_per_element
    # - Parameters: rough estimate ~100MB for typical 3D U-Net
    # - Optimizer state (AdamW): 2x parameters
    # - Workspace (CUDNN, etc.): 20% overhead

    activation_memory_gb = (total_voxels * bytes_per_element) / (1024**3)
    gradient_memory_gb = activation_memory_gb  # Same size
    parameter_memory_gb = 0.1  # Rough estimate
    optimizer_memory_gb = parameter_memory_gb * 2  # AdamW uses 2x param memory
    workspace_memory_gb = (activation_memory_gb + gradient_memory_gb) * 0.2  # 20% overhead

    total_memory_gb = (
        activation_memory_gb
        + gradient_memory_gb
        + parameter_memory_gb
        + optimizer_memory_gb
        + workspace_memory_gb
    )

    return total_memory_gb


def suggest_batch_size(
    patch_size: Tuple[int, int, int],
    in_channels: int,
    out_channels: int,
    available_gpu_memory_gb: float,
    base_features: int = 32,
    num_pool_stages: int = 4,
    deep_supervision: bool = False,
    mixed_precision: bool = True,
    safety_margin: float = 0.85,  # Use 85% of available memory
) -> int:
    """
    Suggest optimal batch size based on available GPU memory.

    Args:
        patch_size: Input patch size (D, H, W)
        in_channels: Number of input channels
        out_channels: Number of output classes
        available_gpu_memory_gb: Available GPU memory in GB
        base_features: Base number of features
        num_pool_stages: Number of pooling stages
        deep_supervision: Whether deep supervision is used
        mixed_precision: Whether mixed precision is used
        safety_margin: Fraction of GPU memory to use (default: 0.85)

    Returns:
        int: Suggested batch size (minimum 1)
    """
    target_memory = available_gpu_memory_gb * safety_margin

    # Linear scan for maximum batch size that fits in memory
    min_bs = 1
    max_bs = 32  # Reasonable upper limit
    best_bs = 1

    for bs in range(min_bs, max_bs + 1):
        estimated_memory = estimate_gpu_memory_required(
            patch_size=patch_size,
            batch_size=bs,
            in_channels=in_channels,
            out_channels=out_channels,
            base_features=base_features,
            num_pool_stages=num_pool_stages,
            deep_supervision=deep_supervision,
            mixed_precision=mixed_precision,
        )

        if estimated_memory <= target_memory:
            best_bs = bs
        else:
            break

    return max(1, best_bs)


def print_gpu_info():
    """Print formatted GPU information."""
    info = get_gpu_info()

    print("=" * 60)
    print("GPU Information")
    print("=" * 60)

    if info["accelerator"] == "cpu":
        print("No supported accelerator is available. Training will use CPU.")
        print(
            f"System RAM: {get_system_memory_gb():.1f} GB total, "
            f"{get_available_system_memory_gb():.1f} GB available"
        )
        return

    print(f"Accelerator: {info['accelerator'].upper()}")

    print(f"Number of GPUs: {info['num_gpus']}")
    print()

    for i in range(info["num_gpus"]):
        print(f"GPU {i}:")
        print(f"  Name: {info['gpu_names'][i]}")
        print(f"  Total Memory: {info['total_memory_gb'][i]:.2f} GB")
        print(f"  Available Memory: {info['available_memory_gb'][i]:.2f} GB")
        print()

    print(
        f"System RAM: {get_system_memory_gb():.1f} GB total, "
        f"{get_available_system_memory_gb():.1f} GB available"
    )
    print("=" * 60)


def get_optimal_num_workers(num_gpus: int = 1) -> int:
    """
    Suggest optimal number of data loader workers.

    Rule of thumb: 4-8 workers per GPU, but not more than CPU count.

    Args:
        num_gpus: Number of GPUs being used

    Returns:
        int: Suggested number of workers
    """
    import multiprocessing

    cpu_count = multiprocessing.cpu_count()
    workers_per_gpu = 4
    suggested = min(workers_per_gpu * max(1, num_gpus), cpu_count)

    return max(2, suggested)  # Minimum 2 workers

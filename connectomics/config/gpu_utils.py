"""
GPU and System Information Utilities.

Provides functions to query GPU memory, count available GPUs,
and estimate memory requirements for training.
"""

import torch
import psutil
from typing import Dict, Tuple


def get_gpu_info() -> Dict[str, any]:
    """
    Get comprehensive GPU information.

    Returns:
        dict: Dictionary containing GPU information:
            - num_gpus: Number of available GPUs
            - gpu_names: List of GPU names
            - total_memory_gb: List of total memory per GPU in GB
            - available_memory_gb: List of available memory per GPU in GB
            - cuda_available: Whether CUDA is available
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "num_gpus": 0,
        "gpu_names": [],
        "total_memory_gb": [],
        "available_memory_gb": [],
    }

    if not torch.cuda.is_available():
        return info

    info["num_gpus"] = torch.cuda.device_count()

    for i in range(info["num_gpus"]):
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

    # Binary search for maximum batch size
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

    if not info["cuda_available"]:
        print("CUDA is not available. Training will use CPU.")
        print(
            f"System RAM: {get_system_memory_gb():.1f} GB total, "
            f"{get_available_system_memory_gb():.1f} GB available"
        )
        return

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


if __name__ == "__main__":
    # Test GPU info
    print_gpu_info()

    # Test batch size suggestion
    if torch.cuda.is_available():
        info = get_gpu_info()
        if info["num_gpus"] > 0:
            print("\nBatch Size Suggestions:")
            print("=" * 60)

            patch_sizes = [(64, 64, 64), (128, 128, 128), (192, 192, 192)]
            for patch_size in patch_sizes:
                bs = suggest_batch_size(
                    patch_size=patch_size,
                    in_channels=1,
                    out_channels=2,
                    available_gpu_memory_gb=info["available_memory_gb"][0],
                    deep_supervision=True,
                    mixed_precision=True,
                )
                print(f"Patch {patch_size}: batch_size={bs}")

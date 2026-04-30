"""Distributed helpers for lazy sliding-window inference."""

from __future__ import annotations

from typing import Optional

import torch


def distributed_context() -> tuple[bool, int, int]:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return False, 0, 1
    return True, torch.distributed.get_rank(), torch.distributed.get_world_size()


def is_distributed_window_sharding_enabled(cfg) -> bool:
    sliding_cfg = getattr(getattr(cfg, "inference", None), "sliding_window", None)
    if sliding_cfg is None:
        return False
    is_dist, _rank, world_size = distributed_context()
    return bool(
        getattr(sliding_cfg, "lazy_load", False)
        and getattr(sliding_cfg, "distributed_sharding", False)
        and is_dist
        and world_size > 1
    )


def distributed_reduction_device(infer_device: torch.device) -> torch.device:
    if infer_device.type == "cuda":
        return infer_device
    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return torch.device("cpu")


def validate_distributed_tensor_shape(
    tensor: torch.Tensor,
    *,
    name: str,
    reduction_device: torch.device,
) -> None:
    is_dist, _rank, world_size = distributed_context()
    if not is_dist:
        return

    max_ndim = 8
    if tensor.ndim > max_ndim:
        raise RuntimeError(f"{name} has rank {tensor.ndim}, exceeding supported rank {max_ndim}.")

    shape_info = torch.full((max_ndim + 1,), -1, dtype=torch.int64, device=reduction_device)
    shape_info[0] = int(tensor.ndim)
    for idx, dim in enumerate(tensor.shape):
        shape_info[idx + 1] = int(dim)

    gathered = [torch.empty_like(shape_info) for _ in range(world_size)]
    torch.distributed.all_gather(gathered, shape_info)
    shapes = []
    for item in gathered:
        ndim = int(item[0].item())
        shapes.append(tuple(int(v.item()) for v in item[1 : 1 + ndim]))

    if any(shape != shapes[0] for shape in shapes[1:]):
        shape_summary = ", ".join(
            f"rank {rank_idx}: {shape}" for rank_idx, shape in enumerate(shapes)
        )
        raise RuntimeError(
            f"Distributed lazy sliding-window sharding requires every rank to reduce {name} "
            f"with the same shape, got {shape_summary}."
        )


def reduce_cpu_tensor_to_rank_zero(
    tensor: torch.Tensor,
    *,
    op,
    reduction_device: torch.device,
    chunk_mb: int,
    name: str,
) -> Optional[torch.Tensor]:
    """Reduce a large CPU accumulator to rank 0 through manageable device chunks."""
    is_dist, rank, _world_size = distributed_context()
    if not is_dist:
        return tensor

    validate_distributed_tensor_shape(tensor, name=name, reduction_device=reduction_device)

    chunk_bytes = max(1, int(chunk_mb or 128)) * 1024 * 1024
    flat_tensor = tensor.contiguous().view(-1)
    elems_per_chunk = max(1, chunk_bytes // max(1, flat_tensor.element_size()))
    reduced_flat = torch.empty_like(flat_tensor) if rank == 0 else None

    for start in range(0, flat_tensor.numel(), elems_per_chunk):
        end = min(start + elems_per_chunk, flat_tensor.numel())
        reduced_chunk = flat_tensor[start:end].to(device=reduction_device, non_blocking=False)
        torch.distributed.reduce(reduced_chunk, dst=0, op=op)
        if rank == 0:
            reduced_flat[start:end].copy_(reduced_chunk.cpu())

    if rank == 0:
        return reduced_flat.view_as(tensor)
    return None


def validate_distributed_patch_shard(
    *,
    local_count: int,
    total_count: int,
    reduction_device: torch.device,
) -> None:
    is_dist, _rank, world_size = distributed_context()
    if not is_dist:
        return

    count = torch.tensor([int(local_count)], dtype=torch.int64, device=reduction_device)
    gathered = [torch.empty_like(count) for _ in range(world_size)]
    torch.distributed.all_gather(gathered, count)
    counts = [int(item.item()) for item in gathered]
    if any(v <= 0 for v in counts):
        raise RuntimeError(
            "Distributed lazy sliding-window sharding assigned an empty window shard "
            f"(total_windows={total_count}, per_rank={counts}). Use fewer GPUs or a "
            "smaller inference.sliding_window.window_size."
        )


__all__ = [
    "distributed_context",
    "distributed_reduction_device",
    "is_distributed_window_sharding_enabled",
    "reduce_cpu_tensor_to_rank_zero",
    "validate_distributed_patch_shard",
    "validate_distributed_tensor_shape",
]

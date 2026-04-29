"""Runtime helpers for test-stage resource and volume sharding."""

from __future__ import annotations

import os
from typing import Any

import torch

from ..config import Config, resolve_data_paths, resolve_default_profiles
from ..config.hardware import resolve_runtime_resource_sentinels
from .output_naming import compute_tta_passes


def resolve_test_stage_runtime(cfg: Config) -> Config:
    """Switch runtime config to test stage and re-resolve resource sentinels."""
    cfg = resolve_default_profiles(cfg, mode="test")
    cfg = resolve_data_paths(cfg)
    cfg = resolve_runtime_resource_sentinels(cfg, print_results=True)

    if not torch.cuda.is_available():
        if cfg.system.num_gpus > 0:
            print("CUDA not available, setting num_gpus=0")
            cfg.system.num_gpus = 0
        if cfg.system.num_workers > 0:
            print("CUDA not available, setting num_workers=0 to avoid dataloader crashes")
            cfg.system.num_workers = 0

    return cfg


def estimate_tta_total_passes(cfg: Config) -> int:
    """Estimate total TTA passes from config for device-cap decisions."""
    do_2d = bool(
        getattr(getattr(cfg.data, "train", None), "do_2d", False)
        or getattr(getattr(cfg.data, "val", None), "do_2d", False)
    )
    spatial_dims = 2 if do_2d else 3
    return max(1, compute_tta_passes(cfg, spatial_dims=spatial_dims))


def maybe_limit_test_devices(cfg: Config, datamodule: Any) -> bool:
    """Reduce multi-GPU test runs when the dataset has fewer volumes than devices."""
    requested_devices = int(getattr(cfg.system, "num_gpus", 0) or 0)
    if requested_devices <= 1:
        return False

    test_data_dicts = getattr(datamodule, "test_data_dicts", None)
    if not test_data_dicts:
        return False

    test_volume_count = len(test_data_dicts)

    tta_cfg = getattr(getattr(cfg, "inference", None), "test_time_augmentation", None)
    distributed_tta_sharding = bool(
        getattr(tta_cfg, "enabled", False) and getattr(tta_cfg, "distributed_sharding", False)
    )
    sliding_cfg = getattr(getattr(cfg, "inference", None), "sliding_window", None)
    distributed_window_sharding = bool(
        getattr(sliding_cfg, "lazy_load", False)
        and getattr(sliding_cfg, "distributed_sharding", False)
    )
    if distributed_tta_sharding and test_volume_count != 1:
        print(
            "  WARNING: Disabling distributed TTA sharding for multi-volume test datasets. "
            "DDP ranks would otherwise reduce predictions from different volumes, which can "
            "mix samples or hang when shapes differ."
        )
        tta_cfg.distributed_sharding = False
        distributed_tta_sharding = False

    if distributed_window_sharding and test_volume_count != 1:
        print(
            "  WARNING: Disabling distributed sliding-window sharding for multi-volume "
            "test datasets. Use independent volume sharding instead."
        )
        sliding_cfg.distributed_sharding = False
        distributed_window_sharding = False

    if distributed_tta_sharding and test_volume_count == 1:
        safe_devices = max(1, min(requested_devices, estimate_tta_total_passes(cfg)))
        if safe_devices < requested_devices:
            print(
                "  WARNING: Reducing devices to match available TTA passes: "
                f"{requested_devices} -> {safe_devices}."
            )
            cfg.system.num_gpus = safe_devices
            return True

        print(
            "  INFO: Keeping multi-GPU test enabled for single-volume TTA sharding "
            f"({safe_devices} device(s), {estimate_tta_total_passes(cfg)} total TTA pass(es))."
        )
        return False

    if distributed_window_sharding and test_volume_count == 1:
        print(
            "  INFO: Keeping multi-GPU test enabled for single-volume lazy "
            "sliding-window sharding "
            f"({requested_devices} device(s))."
        )
        return False

    safe_devices = max(1, min(requested_devices, test_volume_count))
    if safe_devices >= requested_devices:
        return False

    print(
        "  WARNING: Test dataset has fewer volumes than requested GPUs; "
        f"reducing devices from {requested_devices} to {safe_devices} "
        "to avoid duplicated DDP test work and output-file write collisions."
    )
    cfg.system.num_gpus = safe_devices
    return True


def resolve_test_rank_shard_from_env() -> tuple[int | None, int | None]:
    """Return rank/world_size for externally launched multi-process test jobs."""
    for rank_key, world_key in (("RANK", "WORLD_SIZE"), ("SLURM_PROCID", "SLURM_NTASKS")):
        rank_raw = os.environ.get(rank_key)
        world_raw = os.environ.get(world_key)
        if rank_raw is None or world_raw is None:
            continue
        try:
            rank = int(rank_raw)
            world_size = int(world_raw)
        except ValueError:
            continue
        if world_size > 1:
            return rank, world_size

    return None, None


def resolve_test_image_paths(cfg: Config) -> list[str]:
    """Resolve test image paths from config for shard planning."""
    data_cfg = getattr(cfg, "data", None)
    test_image = getattr(getattr(data_cfg, "test", None), "image", None)
    if not test_image:
        return []

    from connectomics.training.lightning.path_utils import expand_file_paths

    try:
        return expand_file_paths(test_image)
    except Exception as exc:
        print(f"  WARNING: Failed to resolve test_image paths for sharding: {exc}")
        return []


def maybe_enable_independent_test_sharding(args: Any, cfg: Config) -> bool:
    """Run test as independent single-GPU shards instead of DDP when rank info is available."""
    requested_devices = int(getattr(cfg.system, "num_gpus", 0) or 0)
    if requested_devices <= 1:
        return False

    shard_id = getattr(args, "shard_id", None)
    num_shards = getattr(args, "num_shards", None)
    source = None

    if shard_id is not None and num_shards is not None and int(num_shards) > 1:
        source = "explicit shard arguments"
    else:
        test_image_paths = resolve_test_image_paths(cfg)
        if len(test_image_paths) <= 1:
            return False

        shard_id, num_shards = resolve_test_rank_shard_from_env()
        if shard_id is None or num_shards is None:
            return False

        args.shard_id = shard_id
        args.num_shards = num_shards
        source = "distributed launcher environment"

    tta_cfg = getattr(getattr(cfg, "inference", None), "test_time_augmentation", None)
    if tta_cfg is not None and bool(
        getattr(tta_cfg, "enabled", False) and getattr(tta_cfg, "distributed_sharding", False)
    ):
        print(
            "  WARNING: Disabling distributed TTA sharding for independent per-rank test sharding."
        )
        tta_cfg.distributed_sharding = False

    cfg.system.num_gpus = 1 if torch.cuda.is_available() else 0
    print(
        "  INFO: Independent multi-GPU test sharding enabled "
        f"({source}); each process will handle its own shard with no DDP communication."
    )
    return True


def has_assigned_test_shard(cfg: Config, args: Any) -> bool:
    """Return True if the current shard has at least one test volume to process."""
    shard_id = getattr(args, "shard_id", None)
    num_shards = getattr(args, "num_shards", None)
    if shard_id is None or num_shards is None:
        return True

    test_image_paths = resolve_test_image_paths(cfg)
    if not test_image_paths:
        return True

    if test_image_paths[shard_id::num_shards]:
        return True

    print(f"  Shard {shard_id}/{num_shards} is empty, nothing to do.")
    print("[OK]Test completed successfully (empty shard).")
    return False


def shard_test_datamodule(datamodule: Any, shard_id: int, num_shards: int):
    """Shard test volumes across machines."""
    data_dicts = getattr(datamodule, "test_data_dicts", None)
    if not data_dicts:
        raise ValueError("No test_data_dicts to shard")
    n = len(data_dicts)
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError(f"shard_id={shard_id} out of range for num_shards={num_shards}")
    if num_shards > n:
        print(
            f"  WARNING: num_shards={num_shards} > test volumes={n}; "
            f"shard {shard_id} may be empty"
        )

    shard = data_dicts[shard_id::num_shards]
    if not shard:
        raise ValueError(f"Shard {shard_id}/{num_shards} is empty (only {n} test volumes)")

    print(
        f"  Test sharding: shard {shard_id}/{num_shards}, " f"processing {len(shard)}/{n} volumes"
    )
    datamodule.test_data_dicts = shard
    return datamodule


__all__ = [
    "estimate_tta_total_passes",
    "has_assigned_test_shard",
    "maybe_enable_independent_test_sharding",
    "maybe_limit_test_devices",
    "resolve_test_image_paths",
    "resolve_test_rank_shard_from_env",
    "resolve_test_stage_runtime",
    "shard_test_datamodule",
]

#!/usr/bin/env python3
"""
PyTorch Connectomics training script with Hydra configuration and Lightning framework.

This script provides modern deep learning training with:
- Hydra-based configuration management
- Automatic distributed training and mixed precision
- MONAI-based data augmentation
- PyTorch Lightning callbacks and logging

Usage:
    # Basic training
    python scripts/main.py --config tutorials/mito_lucchi++.yaml

    # Testing mode
    python scripts/main.py --config tutorials/mito_lucchi++.yaml --mode test \
        --checkpoint path/to/checkpoint.ckpt

    # Fast dev run (1 batch for debugging, auto-sets num_workers=0 and uses
    # GPU only if CUDA is available)
    python scripts/main.py --config tutorials/mito_lucchi++.yaml --fast-dev-run
    python scripts/main.py --config tutorials/mito_lucchi++.yaml --fast-dev-run 2  # Run 2 batches

    # Demo mode (uses tutorials/minimal.yaml + fast-dev-run)
    python scripts/main.py --demo

    # Override config parameters
    python scripts/main.py --config tutorials/mito_lucchi++.yaml \
        data.dataloader.batch_size=8 optimization.max_epochs=200

    # Print fully resolved runtime config
    python scripts/main.py --config tutorials/mito_lucchi++.yaml --debug-config

    # Resume training with different max_epochs
    python scripts/main.py --config tutorials/mito_lucchi++.yaml \
        --checkpoint path/to/ckpt.ckpt --reset-max-epochs 500
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch  # noqa: E402

import connectomics.config.schema as config_schema  # noqa: E402

# Import Hydra config system
from connectomics.config import (  # noqa: E402
    Config,
    resolve_data_paths,
    resolve_default_profiles,
    resolve_runtime_resource_sentinels,
)

# Register safe globals for PyTorch 2.6+ checkpoint loading
# Allowlist all Config dataclasses used inside Lightning checkpoints
try:
    _config_classes = [
        obj
        for obj in config_schema.__dict__.values()
        if isinstance(obj, type) and obj.__name__.endswith("Config")
    ]
    torch.serialization.add_safe_globals(_config_classes)
except AttributeError:
    # PyTorch < 2.6 doesn't have add_safe_globals
    pass

# Import Lightning components and utilities
from connectomics.training.lightning import (  # noqa: E402
    ConnectomicsModule,
    cleanup_run_directory,
    create_datamodule,
    create_trainer,
    modify_checkpoint_state,
    parse_args,
    setup_config,
    setup_run_directory,
    setup_seed_everything,
)

# Setup seed_everything helper
seed_everything = setup_seed_everything()

_RANK_STDOUT_REDIRECT = None


def suppress_nonzero_rank_stdout() -> None:
    """Reduce duplicate stdout spam from DDP subprocesses.

    In local multi-GPU spawn, each subprocess executes this script and prints
    the same setup logs. Keep rank 0 stdout visible and silence stdout on
    non-zero ranks. stderr is untouched for error visibility.
    """
    global _RANK_STDOUT_REDIRECT
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is None or local_rank == "0":
        return
    _RANK_STDOUT_REDIRECT = open(os.devnull, "w")
    sys.stdout = _RANK_STDOUT_REDIRECT


def configure_matmul_precision(cfg: Config) -> None:
    """Enable Tensor Core matmul precision when supported by available CUDA devices."""
    requested_gpus = cfg.system.num_gpus
    if requested_gpus <= 0 or not torch.cuda.is_available():
        return

    try:
        visible_gpus = torch.cuda.device_count()
        check_gpus = min(requested_gpus, visible_gpus)

        has_tensor_cores = False
        for idx in range(check_gpus):
            major, _minor = torch.cuda.get_device_capability(idx)
            # Tensor Cores are available on NVIDIA Volta (SM 7.0) and newer.
            if major >= 7:
                has_tensor_cores = True
                break

        if has_tensor_cores:
            torch.set_float32_matmul_precision("medium")
            print("Enabled float32 matmul precision='medium' (Tensor Cores detected)")
    except Exception as exc:
        print(f"WARNING: Could not configure float32 matmul precision automatically: {exc}")


def get_output_base_from_checkpoint(checkpoint_path: str) -> Path:
    """
    Determine the output base directory from checkpoint path.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Path to use as base for results/ and tuning/ folders

    Logic:
        1. If checkpoint contains timestamp folder (YYYYMMDD_HHMMSS), use that folder
        2. Otherwise, use checkpoint parent folder / checkpoint_stem

    Examples:
        "outputs/exp/20241124_203930/checkpoints/last.ckpt"
            → "outputs/exp/20241124_203930/"
        "pretrained_models/model.ckpt"
            → "pretrained_models/model/"
    """
    import re

    ckpt_path = Path(checkpoint_path)

    # Look for timestamp pattern (YYYYMMDD_HHMMSS) in path parts
    timestamp_pattern = re.compile(r"^\d{8}_\d{6}$")

    for parent in ckpt_path.parents:
        if timestamp_pattern.match(parent.name):
            # Found timestamp folder, use it as base
            return parent

    # No timestamp found, create folder based on checkpoint filename
    # Use checkpoint's grandparent / checkpoint_stem
    ckpt_stem = ckpt_path.stem  # e.g., "last" or "model"
    return ckpt_path.parent.parent / ckpt_stem


def extract_step_from_checkpoint(checkpoint_path: str) -> str:
    """
    Extract step number from checkpoint filename.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Step string (e.g., "step=195525") or empty string if not found

    Examples:
        "epoch=494-step=195525.ckpt" → "step=195525"
        "last.ckpt" → ""
        "model-epoch=10-step=5000.ckpt" → "step=5000"
    """
    import re

    ckpt_path = Path(checkpoint_path)
    filename = ckpt_path.stem  # Remove .ckpt extension

    # Look for step=XXXXX pattern
    step_pattern = re.compile(r"step=(\d+)")
    match = step_pattern.search(filename)

    if match:
        return f"step={match.group(1)}"

    return ""


def _resolve_cached_prediction_files(
    output_dir: Path,
    filenames: list[str],
    cache_suffix: str,
) -> tuple[bool, str | None, list[Path]]:
    """Resolve cached prediction files with optional TTA-suffix fallback."""
    if not filenames:
        return False, None, []

    resolved_files: list[Path] = []
    loaded_suffix = cache_suffix

    for filename in filenames:
        pred_file = output_dir / f"{filename}{cache_suffix}"
        current_suffix = cache_suffix

        if not pred_file.exists() and cache_suffix != "_tta_prediction.h5":
            tta_pred_file = output_dir / f"{filename}_tta_prediction.h5"
            if tta_pred_file.exists():
                pred_file = tta_pred_file
                current_suffix = "_tta_prediction.h5"

        if not pred_file.exists():
            return False, None, []

        if not _is_valid_hdf5_prediction_file(pred_file):
            return False, None, []

        if current_suffix == "_tta_prediction.h5":
            loaded_suffix = "_tta_prediction.h5"
        resolved_files.append(pred_file)

    return True, loaded_suffix, resolved_files


def _is_valid_hdf5_prediction_file(path: Path, dataset: str = "main") -> bool:
    """Return True when a cached prediction file is readable and contains the dataset."""
    import h5py

    try:
        with h5py.File(path, "r") as handle:
            if dataset not in handle:
                print(
                    f"  WARNING: Cached prediction file missing dataset '{dataset}': {path}. "
                    "Ignoring cache entry."
                )
                return False
            _ = handle[dataset].shape
        return True
    except Exception as exc:
        print(
            "  WARNING: Cached prediction file is unreadable: "
            f"{path} ({exc}). Ignoring cache entry."
        )
        return False


def _resolve_tta_result_path_override(cfg: Config) -> str:
    """Return explicit intermediate prediction file from inference.tta_result_path."""
    value = getattr(cfg.inference, "tta_result_path", "")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return ""


def preflight_test_cache_hit(cfg: Config, datamodule) -> tuple[bool, str | None, int]:
    """Check if test outputs already exist so inference (and ckpt restore) can be skipped."""
    save_pred_cfg = getattr(cfg.inference, "save_prediction", None)
    if save_pred_cfg is None:
        return False, None, 0

    explicit_prediction = _resolve_tta_result_path_override(cfg)
    if isinstance(explicit_prediction, str) and explicit_prediction.strip():
        pred_file = Path(explicit_prediction).expanduser()
        if not pred_file.is_absolute():
            pred_file = Path.cwd() / pred_file

        # If explicit intermediate prediction exists, skip TTA inference and ckpt restore.
        if pred_file.exists() and _is_valid_hdf5_prediction_file(pred_file):
            return True, "_tta_prediction.h5", 1

        print(
            "  WARNING: inference.tta_result_path file missing or unreadable "
            f"during preflight: {pred_file}. "
            "Falling back to normal cache/inference flow."
        )

    output_dir_value = getattr(save_pred_cfg, "output_path", None)
    if not output_dir_value:
        return False, None, 0

    test_data_dicts = getattr(datamodule, "test_data_dicts", None)
    if not test_data_dicts:
        return False, None, 0

    filenames = []
    for data_dict in test_data_dicts:
        if not isinstance(data_dict, dict):
            return False, None, 0
        image_path = data_dict.get("image")
        if not image_path:
            return False, None, 0
        filenames.append(Path(str(image_path)).stem)

    cache_hit, loaded_suffix, _resolved_files = _resolve_cached_prediction_files(
        Path(output_dir_value),
        filenames,
        getattr(save_pred_cfg, "cache_suffix", "_prediction.h5"),
    )
    if not cache_hit:
        return False, None, len(filenames)
    return True, loaded_suffix, len(filenames)


def _is_test_evaluation_enabled(cfg: Config) -> bool:
    """Return whether test-time evaluation is enabled."""
    evaluation_cfg = getattr(cfg.inference, "evaluation", None)
    if evaluation_cfg is None:
        return False
    if isinstance(evaluation_cfg, dict):
        return bool(evaluation_cfg.get("enabled", False))
    return bool(getattr(evaluation_cfg, "enabled", False))


def resolve_test_stage_runtime(cfg: Config) -> Config:
    """Switch runtime config to test stage and re-resolve resource sentinels."""
    cfg = resolve_default_profiles(cfg, mode="test")
    cfg = resolve_data_paths(cfg)
    cfg = resolve_runtime_resource_sentinels(cfg, print_results=True)

    # Keep runtime behavior consistent with setup_config() for CPU-only environments.
    if not torch.cuda.is_available():
        if cfg.system.num_gpus > 0:
            print("CUDA not available, setting num_gpus=0")
            cfg.system.num_gpus = 0
        if cfg.system.num_workers > 0:
            print("CUDA not available, setting num_workers=0 to avoid dataloader crashes")
            cfg.system.num_workers = 0

    return cfg


def _estimate_tta_total_passes(cfg: Config) -> int:
    """Estimate total TTA passes from config for device-cap decisions."""
    tta_cfg = getattr(getattr(cfg, "inference", None), "test_time_augmentation", None)
    if tta_cfg is None or not bool(getattr(tta_cfg, "enabled", False)):
        return 1

    do_2d = bool(
        getattr(getattr(cfg.data, "train", None), "do_2d", False)
        or getattr(getattr(cfg.data, "val", None), "do_2d", False)
    )
    spatial_dims = 2 if do_2d else 3

    def _cfg_len(value) -> int:
        if value is None or isinstance(value, str):
            return 0
        try:
            return len(value)
        except TypeError:
            return 0

    flip_axes_cfg = getattr(tta_cfg, "flip_axes", None)
    if flip_axes_cfg == "all" or flip_axes_cfg == []:
        flip_variants = 2**spatial_dims
    elif flip_axes_cfg is None:
        flip_variants = 1
    else:
        flip_variants = 1 + _cfg_len(flip_axes_cfg)

    rotation90_axes_cfg = getattr(tta_cfg, "rotation90_axes", None)
    if rotation90_axes_cfg == "all":
        rotation_planes = 3 if spatial_dims == 3 else 1
    elif rotation90_axes_cfg is None:
        rotation_planes = 0
    else:
        rotation_planes = _cfg_len(rotation90_axes_cfg)

    passes_per_flip = 1 if rotation_planes == 0 else rotation_planes * 4
    return max(1, flip_variants * passes_per_flip)


def maybe_limit_test_devices(cfg: Config, datamodule) -> bool:
    """Reduce multi-GPU test runs when the dataset has fewer volumes than devices.

    Lightning's default DistributedSampler can replicate test samples across ranks
    to keep per-rank batch sizes balanced. For whole-volume inference with output
    saving, that causes duplicate work and multiple ranks writing the same file.

    Returns:
        True if cfg.system.num_gpus was changed and the test trainer should be rebuilt.
    """
    requested_devices = int(getattr(cfg.system, "num_gpus", 0) or 0)
    if requested_devices <= 1:
        return False

    test_data_dicts = getattr(datamodule, "test_data_dicts", None)
    if not test_data_dicts:
        return False

    test_volume_count = len(test_data_dicts)

    tta_cfg = getattr(getattr(cfg, "inference", None), "test_time_augmentation", None)
    distributed_tta_sharding = bool(getattr(tta_cfg, "distributed_sharding", False))
    if distributed_tta_sharding and test_volume_count != 1:
        print(
            "  WARNING: Disabling distributed TTA sharding for multi-volume test datasets. "
            "DDP ranks would otherwise reduce predictions from different volumes, which can "
            "mix samples or hang when shapes differ."
        )
        tta_cfg.distributed_sharding = False
        distributed_tta_sharding = False

    if distributed_tta_sharding and test_volume_count == 1:
        safe_devices = max(1, min(requested_devices, _estimate_tta_total_passes(cfg)))
        if safe_devices < requested_devices:
            print(
                "  WARNING: Reducing devices to match available TTA passes: "
                f"{requested_devices} → {safe_devices}."
            )
            cfg.system.num_gpus = safe_devices
            return True

        print(
            "  INFO: Keeping multi-GPU test enabled for single-volume TTA sharding "
            f"({safe_devices} device(s), {_estimate_tta_total_passes(cfg)} total TTA pass(es))."
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


def maybe_enable_independent_test_sharding(args, cfg: Config) -> bool:
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
    if tta_cfg is not None and bool(getattr(tta_cfg, "distributed_sharding", False)):
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


def has_assigned_test_shard(cfg: Config, args) -> bool:
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


def shard_test_datamodule(datamodule, shard_id: int, num_shards: int):
    """Shard test volumes across machines.

    Splits test_data_dicts into num_shards chunks and keeps only the
    chunk at index shard_id. This allows running test mode in parallel
    across N machines, each processing a disjoint subset of volumes.

    Usage:
        python scripts/main.py --config ... --mode test --shard-id 0 --num-shards 4
        python scripts/main.py --config ... --mode test --shard-id 1 --num-shards 4
        ...
    """
    data_dicts = getattr(datamodule, "test_data_dicts", None)
    if not data_dicts:
        raise ValueError("No test_data_dicts to shard")
    n = len(data_dicts)
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError(
            f"shard_id={shard_id} out of range for num_shards={num_shards}"
        )
    if num_shards > n:
        print(
            f"  WARNING: num_shards={num_shards} > test volumes={n}; "
            f"shard {shard_id} may be empty"
        )

    shard = data_dicts[shard_id::num_shards]
    if not shard:
        raise ValueError(
            f"Shard {shard_id}/{num_shards} is empty (only {n} test volumes)"
        )

    print(
        f"  Test sharding: shard {shard_id}/{num_shards}, "
        f"processing {len(shard)}/{n} volumes"
    )
    datamodule.test_data_dicts = shard
    return datamodule


def _invert_save_prediction_transform(cfg: Config, data):
    """Invert save_prediction intensity scaling (matches ConnectomicsModule behavior)."""
    import numpy as np

    inference_cfg = getattr(cfg, "inference", None)
    if inference_cfg is None or getattr(inference_cfg, "save_prediction", None) is None:
        return data.astype(np.float32)

    save_pred_cfg = inference_cfg.save_prediction
    intensity_scale = getattr(save_pred_cfg, "intensity_scale", None)

    data = data.astype(np.float32)
    if intensity_scale is not None and intensity_scale > 0 and intensity_scale != 1.0:
        data = data / float(intensity_scale)
        print(f"  Inverted intensity scaling by {intensity_scale}")
    elif intensity_scale is not None and intensity_scale < 0:
        print(
            f"  INFO: Intensity scaling was disabled (scale={intensity_scale}), no inversion needed"
        )

    return data


def try_cache_only_test_execution(
    cfg: Config, mode: str, shard_id: int = None, num_shards: int = None
) -> bool:
    """Run cache-only test path before model/trainer/datamodule creation when possible.

    Returns True if test processing completed and caller should exit early.
    """
    if mode != "test":
        return False
    data_cfg = getattr(cfg, "data", None)
    if data_cfg is None:
        return False
    save_pred_cfg = getattr(cfg.inference, "save_prediction", None)
    if save_pred_cfg is None:
        return False
    output_dir_value = getattr(save_pred_cfg, "output_path", None)
    test_image = getattr(getattr(data_cfg, "test", None), "image", None)
    if not output_dir_value or not test_image:
        return False

    from connectomics.data.io import read_volume
    from connectomics.decoding import apply_decode_mode, resolve_decode_modes_from_cfg
    from connectomics.inference.output import apply_postprocessing, write_outputs
    from connectomics.training.lightning.path_utils import expand_file_paths

    try:
        test_image_paths = expand_file_paths(test_image)
    except Exception as exc:
        print(f"  WARNING: Cache-only preflight skipped: failed to resolve test_image paths: {exc}")
        return False

    if not test_image_paths:
        return False

    # Apply sharding to cache-only path
    if shard_id is not None and num_shards is not None:
        test_image_paths = test_image_paths[shard_id::num_shards]
        if not test_image_paths:
            print(f"  Shard {shard_id}/{num_shards} is empty, nothing to do.")
            return True

    output_dir = Path(output_dir_value)
    cache_suffix = getattr(save_pred_cfg, "cache_suffix", "_prediction.h5")
    filenames = [Path(str(p)).stem for p in test_image_paths]

    cache_hit, loaded_suffix, resolved_files = _resolve_cached_prediction_files(
        output_dir,
        filenames,
        cache_suffix,
    )
    if not cache_hit:
        return False

    cached_arrays = []
    for pred_file in resolved_files:
        try:
            cached_arrays.append(read_volume(str(pred_file), dataset="main"))
        except Exception as exc:
            print(
                f"  WARNING: Cache-only preflight skipped: failed to read {pred_file.name}: {exc}"
            )
            return False

    if loaded_suffix != "_tta_prediction.h5":
        if _is_test_evaluation_enabled(cfg):
            print(
                "  [OK]Loaded final predictions from disk, skipping "
                "inference/decoding/postprocessing"
            )
            print("  INFO:Test evaluation is enabled; using trainer.test() " "for eval pipeline.")
            return False
        print(
            "  [OK]Loaded final predictions from disk, skipping inference/decoding/postprocessing"
        )
        print(
            f"  INFO:Cache preflight hit for {len(filenames)} volume(s); skipping trainer.test()."
        )
        print("[OK]Test completed successfully (cache-only preflight).")
        return True

    print("  [OK]Loaded intermediate predictions from disk, skipping inference")
    print(f"  INFO:Cache preflight hit for {len(filenames)} volume(s).")

    if _is_test_evaluation_enabled(cfg):
        print("  INFO:Test evaluation is enabled; using trainer.test() for decode/eval pipeline.")
        return False

    import numpy as np

    if len(cached_arrays) == 1:
        predictions_np = cached_arrays[0]
        if predictions_np.ndim < 4:
            predictions_np = predictions_np[np.newaxis, ...]
    else:
        predictions_np = np.stack(
            [arr[np.newaxis, ...] if arr.ndim < 4 else arr for arr in cached_arrays], axis=0
        )

    print("Cache-only decode/postprocess/save path (evaluation disabled)")
    predictions_np = _invert_save_prediction_transform(cfg, predictions_np)
    has_decoding_cfg = bool(resolve_decode_modes_from_cfg(cfg))
    decoded_predictions = apply_decode_mode(cfg, predictions_np)
    if has_decoding_cfg:
        postprocessed_predictions = apply_postprocessing(cfg, decoded_predictions)
        write_outputs(
            cfg,
            postprocessed_predictions,
            filenames,
            suffix="prediction",
            mode="test",
            batch_meta=None,
        )
    else:
        print("Skipping postprocessing (no decoding configuration)")
        print("Skipping final prediction save (no decoding configuration)")
    print("[OK]Test completed successfully (cache-only decode/postprocess).")
    return True


def _configure_checkpoint_output_paths(args, cfg: Config) -> tuple[Path | None, str | None]:
    """Resolve mode-specific output directories derived from a checkpoint path."""
    if args.mode not in ["test", "tune", "tune-test"] or not args.checkpoint:
        return None, None

    output_base = get_output_base_from_checkpoint(args.checkpoint)
    output_base.mkdir(parents=True, exist_ok=True)
    step_suffix = extract_step_from_checkpoint(args.checkpoint)

    if args.mode in ["tune", "tune-test"]:
        if step_suffix:
            results_folder_name = f"results_{step_suffix}"
            tuning_folder_name = f"tuning_{step_suffix}"
        else:
            results_folder_name = "results"
            tuning_folder_name = "tuning"

        save_pred_cfg = cfg.inference.save_prediction
        save_pred_cfg.output_path = str(output_base / results_folder_name)
        save_pred_cfg.cache_suffix = "_tta_prediction.h5"

        if args.mode == "tune-test":
            print(f"Test output: {save_pred_cfg.output_path}")
            print(f"Test cache suffix: {save_pred_cfg.cache_suffix}")

        return output_base, str(output_base / tuning_folder_name)

    results_folder_name = "results"
    if step_suffix:
        results_folder_name = f"results_{step_suffix}"
        print(f"Using checkpoint {step_suffix} - output will be saved to: {results_folder_name}")

    cfg.inference.save_prediction.output_path = str(output_base / results_folder_name)
    return output_base, cfg.inference.save_prediction.output_path


def _setup_runtime_directories(args, cfg: Config) -> tuple[Path, Path]:
    """Create the run directory and return `(run_dir, output_base)`."""
    output_base, dirpath = _configure_checkpoint_output_paths(args, cfg)
    if output_base is not None and dirpath is not None:
        run_dir = setup_run_directory(args.mode, cfg, dirpath)
        print(f"Output base: {output_base}")
        return run_dir, output_base

    resume_checkpoint_path = None
    if args.mode == "train" and args.checkpoint and args.external_prefix is None:
        resume_checkpoint_path = args.checkpoint

    run_dir = setup_run_directory(
        args.mode,
        cfg,
        cfg.monitor.checkpoint.dirpath,
        resume_checkpoint_path=resume_checkpoint_path,
    )
    return run_dir, run_dir.parent


def _handle_test_cache_hit(
    args,
    cfg: Config,
    cached_suffix: str | None,
    cache_count: int,
    ckpt_path: str | None,
) -> tuple[bool, None]:
    """Print cache-hit status and return whether the test loop can be skipped."""
    if cached_suffix == "_tta_prediction.h5":
        print("  [OK]Loaded intermediate predictions from disk, skipping inference")
    else:
        print(
            "  [OK]Loaded final predictions from disk, skipping "
            "inference/decoding/postprocessing"
        )

    if ckpt_path:
        print(
            f"  INFO:Cache preflight hit for {cache_count} volume(s); "
            "skipping checkpoint weight restore for test."
        )

    should_skip_test_loop = (
        args.mode == "test"
        and cached_suffix != "_tta_prediction.h5"
        and not _is_test_evaluation_enabled(cfg)
    )
    if should_skip_test_loop:
        print("Skipping trainer.test() entirely (cache preflight hit).")
        print("[OK]Test completed successfully (cache-only preflight).")

    return should_skip_test_loop, None


def main():
    """Main training function."""
    suppress_nonzero_rank_stdout()

    # Parse arguments
    args = parse_args()

    # Handle demo mode: route to canonical minimal config workflow.
    if args.demo:
        minimal_config = Path(__file__).parent.parent / "tutorials" / "minimal.yaml"
        if not minimal_config.exists():
            print(f"Error: Demo config not found: {minimal_config}")
            sys.exit(1)
        if not args.config:
            args.config = str(minimal_config)
        if args.fast_dev_run == 0:
            args.fast_dev_run = 1
        if args.mode != "train":
            args.mode = "train"
        print(f"Demo mode: using minimal config {args.config}")

    # Validate that config is provided for non-demo modes
    if not args.config:
        print("Error: --config is required (or use --demo for a quick test)")
        print("\nUsage:")
        print("  python scripts/main.py --config tutorials/mito_lucchi++.yaml")
        print("  python scripts/main.py --demo")
        sys.exit(1)

    # Setup config
    print("\n" + "=" * 60)
    print("PyTorch Connectomics Hydra Training")
    print("=" * 60)
    cfg = setup_config(args)
    configure_matmul_precision(cfg)

    # Tuning expects cached intermediate predictions by default.
    if args.mode in ["tune", "tune-test"]:
        cfg.inference.save_prediction.cache_suffix = "_tta_prediction.h5"

    # Run preflight checks for training mode
    if args.mode == "train":
        from connectomics.utils.errors import preflight_check, print_preflight_issues

        issues = preflight_check(cfg)
        if issues:
            print_preflight_issues(issues)

    # Setup run directory (handles DDP coordination and config saving)
    run_dir, output_base = _setup_runtime_directories(args, cfg)

    # Set random seed
    if cfg.system.seed is not None:
        print(f"Random seed set to: {cfg.system.seed}")
        seed_everything(cfg.system.seed, workers=True)

    if args.mode == "test":
        maybe_enable_independent_test_sharding(args, cfg)
        if not has_assigned_test_shard(cfg, args):
            return

    # Cache-only preflight path for test mode (can skip model/trainer/dataloader entirely).
    if try_cache_only_test_execution(cfg, args.mode, args.shard_id, args.num_shards):
        return

    # Create model
    print(f"Creating model: {cfg.model.arch.type}")
    model = ConnectomicsModule(cfg)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {num_params:,}")

    # Don't use checkpoint path if external weights were loaded (already in model state)
    # External weights are loaded during config setup via model.external_weights_path
    if args.external_prefix:
        print(
            "   WARNING: External weights loaded - checkpoint path will not "
            "be used for training/testing"
        )
        ckpt_path = None
    else:
        ckpt_path = modify_checkpoint_state(
            args.checkpoint,
            run_dir,
            reset_optimizer=args.reset_optimizer,
            reset_scheduler=args.reset_scheduler,
            reset_epoch=args.reset_epoch,
            reset_early_stopping=args.reset_early_stopping,
        )

    # Create trainer (pass run_dir for checkpoints and logs, and checkpoint path for resume)
    trainer = create_trainer(
        cfg,
        run_dir=run_dir,
        fast_dev_run=args.fast_dev_run,
        ckpt_path=ckpt_path,
        mode=args.mode,
    )

    # Main training/testing/tuning workflow
    try:
        if args.mode == "train":
            # Create datamodule
            datamodule = create_datamodule(
                cfg, mode=args.mode, fast_dev_run=bool(args.fast_dev_run)
            )
            print("\n" + "=" * 60)
            print("STARTING TRAINING")
            print("=" * 60)

            trainer.fit(
                model,
                datamodule=datamodule,
                ckpt_path=ckpt_path,
            )
            print("\n[OK]Training completed successfully!")

        # Handle tune modes
        if args.mode in ["tune", "tune-test"]:
            from connectomics.decoding import run_tuning

            # Run parameter tuning (automatically skips if best_params.yaml exists)
            run_tuning(model, trainer, cfg, checkpoint_path=ckpt_path)

        # Handle test modes
        if args.mode in ["tune-test", "test"]:
            print("\n" + "=" * 60)
            print("RUNNING TEST")
            print("=" * 60)

            # Re-resolve test-stage runtime overrides after tuning, including sentinels.
            cfg = resolve_test_stage_runtime(cfg)

            if maybe_enable_independent_test_sharding(args, cfg):
                trainer = create_trainer(
                    cfg,
                    run_dir=run_dir,
                    fast_dev_run=args.fast_dev_run,
                    ckpt_path=ckpt_path,
                    mode="test",
                )
            if not has_assigned_test_shard(cfg, args):
                return

            # Create datamodule
            datamodule = create_datamodule(cfg, mode="test")

            # Apply test volume sharding across machines
            if args.shard_id is not None and args.num_shards is not None:
                datamodule = shard_test_datamodule(
                    datamodule, args.shard_id, args.num_shards
                )

            if maybe_limit_test_devices(cfg, datamodule):
                trainer = create_trainer(
                    cfg,
                    run_dir=run_dir,
                    fast_dev_run=args.fast_dev_run,
                    ckpt_path=ckpt_path,
                    mode="test",
                )

            if args.mode == "tune-test":
                from connectomics.decoding import load_and_apply_best_params

                print("\n" + "=" * 80)
                print("LOADING BEST PARAMETERS")
                print("=" * 80)

                # Load and apply best parameters
                cfg = load_and_apply_best_params(cfg)

            test_ckpt_path = ckpt_path
            cache_hit, cached_suffix, cache_count = preflight_test_cache_hit(cfg, datamodule)
            if cache_hit:
                skip_test_loop, test_ckpt_path = _handle_test_cache_hit(
                    args,
                    cfg,
                    cached_suffix,
                    cache_count,
                    ckpt_path,
                )
                if skip_test_loop:
                    return

            trainer.test(
                model,
                datamodule,
                ckpt_path=test_ckpt_path,
            )

    except Exception as e:
        mode_name = args.mode.capitalize() if args.mode else "Operation"
        print(f"\n{mode_name} failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        # Cleanup: Remove timestamp file after training
        if args.mode == "train" and "output_base" in locals():
            cleanup_run_directory(output_base)


if __name__ == "__main__":
    main()

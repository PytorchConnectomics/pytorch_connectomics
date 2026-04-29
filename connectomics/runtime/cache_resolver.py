"""Runtime cache discovery and cache-only test execution helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from ..config import Config
from .output_naming import (
    is_tta_cache_suffix,
    resolve_prediction_cache_suffix,
    tta_cache_suffix,
    tta_cache_suffix_candidates,
)
from .sharding import resolve_test_image_paths


def resolve_cached_prediction_files(
    output_dir: Path,
    filenames: list[str],
    cache_suffix: str,
    fallback_tta_suffixes: list[str] | None = None,
) -> tuple[bool, str | None, list[Path]]:
    """Resolve cached prediction files with optional TTA-suffix fallback."""
    if not filenames:
        return False, None, []

    suffixes_to_try = [cache_suffix]
    if not is_tta_cache_suffix(cache_suffix) and fallback_tta_suffixes:
        for try_suffix in fallback_tta_suffixes:
            if try_suffix not in suffixes_to_try:
                suffixes_to_try.append(try_suffix)

    for try_suffix in suffixes_to_try:
        resolved_files: list[Path] = []
        all_exist = True
        for filename in filenames:
            pred_file = output_dir / f"{filename}{try_suffix}"
            if not os.path.exists(pred_file):
                all_exist = False
                break
            if not is_valid_hdf5_prediction_file(pred_file):
                all_exist = False
                break
            resolved_files.append(pred_file)
        if all_exist and len(resolved_files) == len(filenames):
            return True, try_suffix, resolved_files

    return False, None, []


def is_valid_hdf5_prediction_file(path: Path, dataset: str = "main") -> bool:
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


def resolve_tta_result_path_override(cfg: Config) -> str:
    """Return explicit intermediate prediction file from inference.tta_result_path."""
    value = getattr(cfg.inference, "tta_result_path", "")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return ""


def has_tta_prediction_file(cfg: Config) -> bool:
    """Return True if an explicit tta_result_path exists and is a valid HDF5 file."""
    tta_path = resolve_tta_result_path_override(cfg)
    if not tta_path:
        return False
    pred_file = Path(tta_path).expanduser()
    if not pred_file.is_absolute():
        pred_file = Path.cwd() / pred_file
    return os.path.exists(pred_file) and is_valid_hdf5_prediction_file(pred_file)


def create_decode_only_datamodule(cfg: Config, input_prediction_path: str):
    """Create a minimal datamodule for decode-only mode."""
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader, Dataset

    pred_stem = Path(input_prediction_path).stem

    class _DummyDataset(Dataset):
        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return {"image": torch.zeros(1, 1, 1, 1), "filename": pred_stem}

    class _DummyDataModule(pl.LightningDataModule):
        def test_dataloader(self):
            return DataLoader(_DummyDataset(), batch_size=1)

    return _DummyDataModule()


def has_cached_predictions_in_output_dir(
    cfg: Config, mode: str, checkpoint_path: str | None = None
) -> bool:
    """Return True if all expected TTA prediction files exist in the output directory."""
    save_pred_cfg = getattr(cfg.inference, "save_prediction", None)
    if save_pred_cfg is None:
        return False
    output_dir = getattr(save_pred_cfg, "output_path", None)
    if not output_dir:
        return False

    test_image_paths = resolve_test_image_paths(cfg)
    if not test_image_paths:
        return False

    suffix = tta_cache_suffix(cfg, checkpoint_path=checkpoint_path)
    output_path = Path(output_dir)
    for image_path in test_image_paths:
        pred_file = output_path / f"{Path(image_path).stem}{suffix}"
        if not os.path.exists(pred_file):
            return False
        if not is_valid_hdf5_prediction_file(pred_file):
            return False
    return True


def preflight_test_cache_hit(
    cfg: Config, datamodule: Any, checkpoint_path: str | None = None
) -> tuple[bool, str | None, int]:
    """Check if test outputs already exist so inference and ckpt restore can be skipped."""
    save_pred_cfg = getattr(cfg.inference, "save_prediction", None)
    if save_pred_cfg is None:
        return False, None, 0

    explicit_prediction = resolve_tta_result_path_override(cfg)
    if isinstance(explicit_prediction, str) and explicit_prediction.strip():
        pred_file = Path(explicit_prediction).expanduser()
        if not pred_file.is_absolute():
            pred_file = Path.cwd() / pred_file

        if os.path.exists(pred_file) and is_valid_hdf5_prediction_file(pred_file):
            return True, tta_cache_suffix(cfg, checkpoint_path=checkpoint_path), 1

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

    cache_hit, loaded_suffix, _resolved_files = resolve_cached_prediction_files(
        Path(output_dir_value),
        filenames,
        resolve_prediction_cache_suffix(cfg, mode="test", checkpoint_path=checkpoint_path),
        fallback_tta_suffixes=tta_cache_suffix_candidates(cfg, checkpoint_path=checkpoint_path),
    )
    if not cache_hit:
        return False, None, len(filenames)
    return True, loaded_suffix, len(filenames)


def is_test_evaluation_enabled(cfg: Config) -> bool:
    """Return whether test-time evaluation is enabled."""
    evaluation_cfg = getattr(cfg, "evaluation", None)
    if evaluation_cfg is None:
        return False
    if isinstance(evaluation_cfg, dict):
        return bool(evaluation_cfg.get("enabled", False))
    return bool(getattr(evaluation_cfg, "enabled", False))


def try_cache_only_test_execution(
    cfg: Config,
    mode: str,
    shard_id: int = None,
    num_shards: int = None,
    checkpoint_path: str | None = None,
) -> bool:
    """Run cache-only test path before model/trainer/datamodule creation when possible."""
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
    from connectomics.decoding import run_decoding_stage
    from connectomics.inference.output import write_outputs
    from connectomics.training.lightning.path_utils import expand_file_paths

    try:
        test_image_paths = expand_file_paths(test_image)
    except Exception as exc:
        print(f"  WARNING: Cache-only preflight skipped: failed to resolve test_image paths: {exc}")
        return False

    if not test_image_paths:
        return False

    if shard_id is not None and num_shards is not None:
        test_image_paths = test_image_paths[shard_id::num_shards]
        if not test_image_paths:
            print(f"  Shard {shard_id}/{num_shards} is empty, nothing to do.")
            return True

    output_dir = Path(output_dir_value)
    cache_suffix = resolve_prediction_cache_suffix(cfg, mode, checkpoint_path=checkpoint_path)
    filenames = [Path(str(p)).stem for p in test_image_paths]

    cache_hit, loaded_suffix, resolved_files = resolve_cached_prediction_files(
        output_dir,
        filenames,
        cache_suffix,
        fallback_tta_suffixes=tta_cache_suffix_candidates(cfg, checkpoint_path=checkpoint_path),
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

    if not is_tta_cache_suffix(loaded_suffix):
        if is_test_evaluation_enabled(cfg):
            print(
                "  [OK]Loaded final predictions from disk, skipping "
                "inference/decoding/postprocessing"
            )
            print("  INFO:Test evaluation is enabled; using trainer.test() for eval pipeline.")
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

    if is_test_evaluation_enabled(cfg):
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
    decoding_result = run_decoding_stage(cfg, predictions_np)
    if decoding_result.has_decoding_config:
        write_outputs(
            cfg,
            decoding_result.postprocessed,
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


def handle_test_cache_hit(
    args: Any,
    cfg: Config,
    cached_suffix: str | None,
    cache_count: int,
    ckpt_path: str | None,
) -> tuple[bool, None]:
    """Print cache-hit status and return whether the test loop can be skipped."""
    if is_tta_cache_suffix(cached_suffix):
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
        and not is_tta_cache_suffix(cached_suffix)
        and not is_test_evaluation_enabled(cfg)
    )
    if should_skip_test_loop:
        print("Skipping trainer.test() entirely (cache preflight hit).")
        print("[OK]Test completed successfully (cache-only preflight).")

    return should_skip_test_loop, None


__all__ = [
    "create_decode_only_datamodule",
    "handle_test_cache_hit",
    "has_cached_predictions_in_output_dir",
    "has_tta_prediction_file",
    "is_test_evaluation_enabled",
    "is_valid_hdf5_prediction_file",
    "preflight_test_cache_hit",
    "resolve_cached_prediction_files",
    "resolve_tta_result_path_override",
    "try_cache_only_test_execution",
]

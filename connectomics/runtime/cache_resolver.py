"""Runtime cache discovery and cache-only test execution helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from ..config import Config
from .output_naming import (
    final_prediction_decoded_glob_suffix,
    final_prediction_output_tag,
    intermediate_prediction_cache_suffix,
    intermediate_prediction_cache_suffix_candidates,
    is_tta_cache_suffix,
    resolve_prediction_cache_suffix,
)
from .sharding import resolve_test_image_paths


def resolve_cached_prediction_files(
    output_dir: Path,
    filenames: list[str],
    cache_suffix: str,
    fallback_tta_suffixes: list[str] | None = None,
    preferred_decoded_suffix: str | None = None,
    decoded_glob_suffix: str | None = None,
) -> tuple[bool, str | None, list[Path]]:
    """Resolve cached prediction files, preferring any decoded final file.

    When ``decoded_glob_suffix`` is provided, the resolver first looks for a
    pre-existing decoded final file matching that glob (regardless of decoding
    kwargs). This avoids reloading multi-GB intermediate predictions when a
    usable final segmentation is already on disk.
    """
    if not filenames:
        return False, None, []

    if preferred_decoded_suffix:
        resolved_files: list[Path] = []
        all_exist = True
        for filename in filenames:
            pred_file = output_dir / f"{filename}{preferred_decoded_suffix}"
            if not os.path.exists(pred_file):
                all_exist = False
                break
            if not is_valid_hdf5_prediction_file(pred_file):
                all_exist = False
                break
            resolved_files.append(pred_file)
        if all_exist and len(resolved_files) == len(filenames):
            return True, preferred_decoded_suffix, resolved_files

    if decoded_glob_suffix:
        decoded_files: list[Path] = []
        for filename in filenames:
            matches = sorted(output_dir.glob(f"{filename}{decoded_glob_suffix}"))
            usable = [m for m in matches if is_valid_hdf5_prediction_file(m)]
            if not usable:
                decoded_files = []
                break
            decoded_files.append(usable[-1])
        if decoded_files and len(decoded_files) == len(filenames):
            chosen_suffix = decoded_files[0].name[len(filenames[0]) :]
            return True, chosen_suffix, decoded_files

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


def _resolve_dataset_image_paths(cfg: Config, mode: str) -> list[str]:
    """Resolve image paths for the dataset used by the given mode.

    ``test``/``tune-test`` reads ``cfg.data.test.image``. ``tune`` reads
    ``cfg.data.val.image`` (Optuna evaluates against the validation/tune set).
    """
    if mode == "tune":
        from connectomics.training.lightning.path_utils import expand_file_paths

        val_image = getattr(getattr(cfg.data, "val", None), "image", None)
        if not val_image:
            return []
        try:
            return expand_file_paths(val_image)
        except Exception as exc:
            print(f"  WARNING: Failed to resolve tune data.val.image paths: {exc}")
            return []
    return resolve_test_image_paths(cfg)


def has_cached_predictions_in_output_dir(
    cfg: Config, mode: str, checkpoint_path: str | None = None
) -> bool:
    """Return True if all expected TTA prediction files exist in the output directory."""
    save_pred_cfg = getattr(cfg.inference, "save_prediction", None)
    if save_pred_cfg is None:
        return False
    output_dir = getattr(save_pred_cfg, "output_path", None)
    output_dirs = [output_dir] if output_dir else []
    if mode == "tune":
        tune_output_dir = getattr(
            getattr(getattr(cfg, "tune", None), "output", None),
            "output_pred",
            None,
        )
        output_dirs = [path for path in (tune_output_dir, output_dir) if path]
    if not output_dirs:
        return False

    image_paths = _resolve_dataset_image_paths(cfg, mode)
    if not image_paths:
        return False

    suffix = intermediate_prediction_cache_suffix(cfg, checkpoint_path=checkpoint_path)
    seen_dirs: set[str] = set()
    for output_dir_value in output_dirs:
        output_dir_str = str(output_dir_value)
        if output_dir_str in seen_dirs:
            continue
        seen_dirs.add(output_dir_str)
        output_path = Path(output_dir_str)
        all_found = True
        for image_path in image_paths:
            pred_file = output_path / f"{Path(image_path).stem}{suffix}"
            if not os.path.exists(pred_file):
                all_found = False
                break
            if not is_valid_hdf5_prediction_file(pred_file):
                all_found = False
                break
        if all_found:
            return True
    return False


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
            return (
                True,
                intermediate_prediction_cache_suffix(cfg, checkpoint_path=checkpoint_path),
                1,
            )

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
        fallback_tta_suffixes=intermediate_prediction_cache_suffix_candidates(
            cfg, checkpoint_path=checkpoint_path
        ),
        preferred_decoded_suffix=(
            "_" + final_prediction_output_tag(cfg, checkpoint_path=checkpoint_path) + ".h5"
        ),
        decoded_glob_suffix=final_prediction_decoded_glob_suffix(
            cfg, checkpoint_path=checkpoint_path
        ),
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
        fallback_tta_suffixes=intermediate_prediction_cache_suffix_candidates(
            cfg, checkpoint_path=checkpoint_path
        ),
        preferred_decoded_suffix=(
            "_" + final_prediction_output_tag(cfg, checkpoint_path=checkpoint_path) + ".h5"
        ),
        decoded_glob_suffix=final_prediction_decoded_glob_suffix(
            cfg, checkpoint_path=checkpoint_path
        ),
    )
    if not cache_hit:
        return False

    # Defer the (potentially multi-GB) read until we know the data will be
    # consumed here. trainer.test() reloads the cache itself when evaluation
    # is enabled, so reading here would just waste time and memory.
    is_intermediate = is_tta_cache_suffix(loaded_suffix)
    evaluation_enabled = is_test_evaluation_enabled(cfg)
    if evaluation_enabled and is_intermediate:
        # Direct decode+eval path that bypasses Lightning when GT labels are
        # locally accessible. Falls back silently if the path can't be taken.
        if _try_cache_only_intermediate_eval(
            cfg, resolved_files, filenames, checkpoint_path=checkpoint_path
        ):
            return True
        return False
    if evaluation_enabled and not is_intermediate:
        # Final-prediction cache + eval still routes through trainer.test();
        # test_pipeline logs cache-hit status itself, so skip the duplicate.
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

    if not is_intermediate:
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


def _try_cache_only_intermediate_eval(
    cfg: Config,
    resolved_files: list[Path],
    filenames: list[str],
    *,
    checkpoint_path: str | None,
) -> bool:
    """Decode + evaluate cached intermediate predictions without spinning up Lightning.

    Returns ``True`` on success. Returns ``False`` (silently) when the path
    can't be taken — e.g. label files unavailable or counts mismatched —
    so the caller can fall back to ``trainer.test()``.
    """
    import numpy as np

    from connectomics.data.io import read_volume
    from connectomics.decoding import run_decoding_stage
    from connectomics.evaluation import (
        EvaluationContext,
        run_evaluation_stage,
    )
    from connectomics.inference.output import apply_prediction_transform, write_outputs
    from connectomics.training.lightning.path_utils import expand_file_paths

    test_cfg = getattr(getattr(cfg, "data", None), "test", None)
    label_value = getattr(test_cfg, "label", None)
    nerl_only = label_value is None and _evaluation_metric_requested(cfg, "nerl")
    if label_value is None and not nerl_only:
        return False

    label_paths: list[str] | None = None
    if label_value is not None:
        try:
            label_paths = expand_file_paths(label_value)
        except Exception:
            return False
        if len(label_paths) != len(resolved_files):
            return False

    print(
        f"  [OK]Found intermediate prediction cache for {len(filenames)} volume(s); "
        "skipping inference and running decode + eval directly."
    )

    final_suffix = final_prediction_output_tag(cfg, checkpoint_path=checkpoint_path)
    inference_cfg = getattr(cfg, "inference", None)
    evaluation_cfg = getattr(cfg, "evaluation", None)

    for idx, pred_file in enumerate(resolved_files):
        volume_name = filenames[idx]
        print(f"  Loading {pred_file.name} ...")
        try:
            predictions_np = read_volume(str(pred_file), dataset="main")
        except Exception as exc:
            print(f"  WARNING: failed to read {pred_file.name}: {exc}; falling back to trainer.test().")
            return False

        if predictions_np.ndim < 4:
            predictions_np = predictions_np[np.newaxis, ...]
        predictions_np = apply_prediction_transform(cfg, predictions_np)

        decoding_result = run_decoding_stage(cfg, predictions_np)
        if decoding_result.has_decoding_config:
            write_outputs(
                cfg,
                decoding_result.postprocessed,
                [volume_name],
                suffix=final_suffix,
                mode="test",
                batch_meta=None,
            )
        else:
            print("  Skipping postprocessing (no decoding configuration)")

        labels_tensor = None
        if label_paths is not None:
            try:
                label_np = read_volume(label_paths[idx], dataset="main")
            except Exception as exc:
                print(f"  WARNING: failed to read label {label_paths[idx]}: {exc}")
                return False
            labels_tensor = torch.from_numpy(label_np[np.newaxis, ...])

        context = EvaluationContext(
            cfg=cfg,
            evaluation_cfg=evaluation_cfg,
            inference_cfg=inference_cfg,
            checkpoint_path=checkpoint_path,
        )
        run_evaluation_stage(
            context,
            decoding_result.decoded,
            labels_tensor,
            filenames=[volume_name],
            batch_idx=idx,
        )

        del predictions_np
        if labels_tensor is not None:
            del labels_tensor

    print("[OK]Test completed successfully (cache-only decode + eval).")
    return True


def _evaluation_metric_requested(cfg: Config, metric_name: str) -> bool:
    evaluation_cfg = getattr(cfg, "evaluation", None)
    if evaluation_cfg is None:
        return False
    metrics = getattr(evaluation_cfg, "metrics", None)
    if metrics is None:
        return False
    if isinstance(metrics, str):
        return metrics.lower() == metric_name.lower()
    return any(str(m).lower() == metric_name.lower() for m in metrics)


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

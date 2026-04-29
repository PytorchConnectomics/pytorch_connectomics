"""DataModule factory functions for Lightning training."""

from __future__ import annotations

import logging
from glob import glob
from pathlib import Path
from typing import Any, List, Optional

from monai.transforms import Compose

from ...config import Config
from ...data.augmentation.build import (
    build_test_transforms,
    build_train_transforms,
    build_val_transforms,
)
from ...data.datasets import create_data_dicts_from_paths
from ...data.io import get_vol_shape, volume_exists
from .data import ConnectomicsDataModule, SimpleDataModule
from .path_utils import expand_file_paths

logger = logging.getLogger(__name__)


def _target_context(cfg: Config) -> tuple[int, ...]:
    context = getattr(cfg.data.dataloader, "target_context", None) or []
    if not context:
        return tuple(0 for _ in cfg.data.dataloader.patch_size)
    return tuple(int(v) for v in context)


def _effective_patch_size(cfg: Config) -> tuple[int, ...]:
    patch_size = tuple(int(v) for v in cfg.data.dataloader.patch_size)
    context = _target_context(cfg)
    if len(context) != len(patch_size):
        raise ValueError(
            "data.dataloader.target_context must have the same length as patch_size: "
            f"{context} vs {patch_size}"
        )
    return tuple(patch_size[i] + context[i] for i in range(len(patch_size)))


def _validation_dataset_mode(cfg: Config) -> str:
    return "train" if bool(getattr(cfg.data.dataloader, "val_random_sampling", False)) else "val"


def _maybe_precompute_label_aux(
    cfg: Config,
    split_cfg: Any,
    label_paths: Optional[List[str]],
    *,
    split_name: str,
) -> Optional[List[str]]:
    """Auto-precompute label_aux if label_transform includes ``skeleton_aware_edt``.

    Reads ``data.<split>.label_aux_type`` to decide mode:
    - ``"skeleton"`` (default): precompute skeleton volume; EDT computed per crop.
    - ``"sdt"``: precompute full SDT volume (slower precompute, zero training cost).
    - ``"none"``: no precompute, compute everything per crop.

    If ``skeleton_aware_edt`` requests ``relabel: true``, the skeleton shortcut is
    not semantics-preserving because the per-crop relabeling can no longer be
    reconstructed from a globally precomputed skeleton volume. In that case we
    transparently promote ``label_aux_type: skeleton`` to full ``sdt`` so the
    loaded auxiliary target matches the configured transform exactly.

    Returns list of label_aux paths, or None.
    """
    if label_paths is None:
        return None

    mode = getattr(split_cfg, "label_aux_type", "skeleton")
    if mode == "none":
        return None

    targets = getattr(cfg.data.label_transform, "targets", None)
    if not targets:
        return None

    sdt_target = None
    for t in targets:
        name = t.get("name") if isinstance(t, dict) else getattr(t, "name", None)
        if name == "skeleton_aware_edt":
            sdt_target = t
            break

    if sdt_target is None:
        return None

    kwargs = sdt_target.get("kwargs", {}) if isinstance(sdt_target, dict) else {}
    resolution = kwargs.get("resolution") or getattr(cfg.data.label_transform, "resolution", None)
    if resolution is None:
        resolution = (1.0, 1.0, 1.0)
    resolution = tuple(float(r) for r in resolution)
    cache_dir = getattr(cfg.data.label_transform, "cache_dir", "") or None
    alpha = float(kwargs.get("alpha", 0.8))
    bg_value = float(kwargs.get("bg_value", -1.0))
    relabel = bool(kwargs.get("relabel", False))

    effective_mode = mode
    if mode == "skeleton" and relabel:
        effective_mode = "sdt"
        logger.warning(
            "label_aux_type=skeleton requested for %s, but skeleton_aware_edt has "
            "relabel=true. Promoting auxiliary target precompute to full SDT so the "
            "cached target matches crop-time relabel semantics.",
            split_name,
        )

    from ...data.processing.distance import (
        precompute_sdt_volume,
        precompute_skeleton_volume,
        sdt_path_for_label,
    )

    print(
        f"label_aux_type={effective_mode} ({split_name}): "
        f"resolution={list(resolution)}, alpha={alpha}"
    )

    paths = []
    for lp in label_paths:
        sp = sdt_path_for_label(lp, mode=effective_mode, cache_dir=cache_dir)
        if not volume_exists(sp):
            if effective_mode == "sdt":
                precompute_sdt_volume(lp, sp, resolution=resolution, alpha=alpha, bg_value=bg_value)
            else:
                precompute_skeleton_volume(lp, sp, resolution=resolution)
        else:
            print(f"  Using cached {effective_mode}: {sp}")
        paths.append(sp)

    return paths


def _populate_split_label_aux_if_needed(cfg: Config, split_name: str) -> None:
    """Populate split label_aux via auto-precompute when applicable."""
    split_cfg = getattr(cfg.data, split_name, None)
    if split_cfg is None or split_cfg.label_aux or not split_cfg.label:
        return

    label_paths = expand_file_paths(split_cfg.label)
    auto_aux = _maybe_precompute_label_aux(
        cfg,
        split_cfg,
        label_paths,
        split_name=split_name,
    )
    if auto_aux:
        split_cfg.label_aux = auto_aux[0] if len(auto_aux) == 1 else auto_aux


def _maybe_prepare_random_data(cfg: Config, mode: str) -> None:
    """Generate random train/val H5 data when config requests random://."""
    if mode != "train":
        return

    train_image = cfg.data.train.image
    if not isinstance(train_image, str) or not train_image.startswith("random://"):
        return

    import h5py
    import numpy as np

    seed = int(getattr(cfg.system, "seed", 42))
    rng = np.random.default_rng(seed)
    patch_size = tuple(int(v) for v in cfg.data.dataloader.patch_size)
    vol_shape = (
        max(64, patch_size[0] * 2),
        max(128, patch_size[1] * 2),
        max(128, patch_size[2] * 2),
    )

    data_root = Path("outputs") / "minimal" / "random_data"
    data_root.mkdir(parents=True, exist_ok=True)

    train_image_path = data_root / "train_image.h5"
    train_label_path = data_root / "train_label.h5"
    val_image_path = data_root / "val_image.h5"
    val_label_path = data_root / "val_label.h5"

    if not all(
        p.exists() for p in [train_image_path, train_label_path, val_image_path, val_label_path]
    ):

        def _write_pair(image_path: Path, label_path: Path) -> None:
            image = rng.random(vol_shape, dtype=np.float32)
            label = (rng.random(vol_shape) > 0.85).astype(np.uint8)
            with h5py.File(image_path, "w") as f:
                f.create_dataset("main", data=image, compression="gzip")
            with h5py.File(label_path, "w") as f:
                f.create_dataset("main", data=label, compression="gzip")

        logger.info("Generating random demo data...")
        _write_pair(train_image_path, train_label_path)
        _write_pair(val_image_path, val_label_path)
        logger.info(f"Random demo data saved to: {data_root}")
    else:
        logger.info(f"Using existing random demo data: {data_root}")

    cfg.data.train.image = str(train_image_path)
    cfg.data.train.label = str(train_label_path)
    cfg.data.val.image = str(val_image_path)
    cfg.data.val.label = str(val_label_path)


def _calculate_validation_steps_per_epoch(
    val_data_dicts: List[dict],
    patch_size: tuple[int, int, int],
    min_steps: int = 50,
    max_steps: Optional[int] = 200,
    default_steps: int = 100,
    fallback_volume_shape: Optional[tuple[int, int, int]] = None,
    return_default_on_error: bool = True,
) -> int:
    """
    Calculate validation steps per epoch based on validation volume size and patch size.

    Args:
        val_data_dicts: Validation data dictionaries
        patch_size: Patch size (D, H, W)
        min_steps: Minimum steps per epoch
        max_steps: Maximum steps per epoch
        default_steps: Default steps per epoch when calculation fails
        fallback_volume_shape: Volume shape fallback for unknown file formats
        return_default_on_error: Return default_steps on errors instead of raising

    Returns:
        Calculated validation steps per epoch
    """
    try:
        # Get first validation volume size
        img_path = Path(val_data_dicts[0]["image"])

        # Load volume to get shape (get_vol_shape handles all supported formats)
        supported_suffixes = {".nii", ".gz", ".h5", ".hdf5", ".tif", ".tiff"}
        if img_path.suffix in supported_suffixes:
            vol_shape = get_vol_shape(str(img_path))
        elif fallback_volume_shape is not None:
            vol_shape = fallback_volume_shape
        else:
            # Unknown format, use default
            logger.warning(
                f"Unknown file format {img_path.suffix}, "
                f"using default validation steps={default_steps}"
            )
            return default_steps

        # Handle channel dimension if present
        if len(vol_shape) == 4:
            vol_shape = vol_shape[1:]  # Remove channel dim: (C, D, H, W) -> (D, H, W)

        # Calculate number of possible patches (with 50% overlap)
        stride = tuple(p // 2 for p in patch_size)  # 50% overlap
        num_patches_per_dim = [
            max(1, (vol_shape[i] - patch_size[i]) // stride[i] + 1) for i in range(3)
        ]
        total_possible_patches = (
            num_patches_per_dim[0] * num_patches_per_dim[1] * num_patches_per_dim[2]
        )

        # Calculate validation steps as a fraction of possible patches
        val_steps = int(total_possible_patches * 0.075)  # 7.5% of possible patches
        val_steps = max(min_steps, val_steps)
        if max_steps is not None:
            val_steps = min(max_steps, val_steps)

        logger.info(f"Validation volume shape: {vol_shape}")
        logger.info(f"Patch size: {patch_size}")
        logger.info(f"Stride (50% overlap): {stride}")
        logger.info(f"Possible patches per dim: {num_patches_per_dim}")
        logger.info(f"Total possible patches: {total_possible_patches}")
        logger.info(f"Using 7.5% of patches: {val_steps}")

        return val_steps

    except Exception as e:
        if not return_default_on_error:
            raise
        logger.warning(f"Error calculating validation steps: {e}")
        logger.info(f"Using default validation steps={default_steps}")
        return default_steps


def create_datamodule(
    cfg: Config, mode: str = "train", fast_dev_run: bool = False
) -> ConnectomicsDataModule:
    """
    Create Lightning DataModule from config.

    Args:
        cfg: Hydra Config object
        mode: 'train', 'test', or 'tune'
        fast_dev_run: If True, config overrides have already been applied in setup_config()

    Returns:
        ConnectomicsDataModule instance
    """
    logger.info("Creating datasets...")
    _maybe_prepare_random_data(cfg, mode)

    # Auto-download tutorial data if missing
    if mode == "train" and cfg.data.train.image:
        from pathlib import Path as PathLib

        # Check if data exists (support glob patterns and lists)
        data_exists = False

        # Handle list of files
        if isinstance(cfg.data.train.image, list):
            # Check if at least one file in the list exists
            data_exists = any(PathLib(img).exists() for img in cfg.data.train.image)
        # Handle glob pattern
        elif "*" in cfg.data.train.image or "?" in cfg.data.train.image:
            # Glob pattern - check if any files match
            matched_files = glob(cfg.data.train.image)
            data_exists = len(matched_files) > 0
        # Handle single file path
        else:
            data_exists = PathLib(cfg.data.train.image).exists()

        if not data_exists:
            logger.warning(f"Training data not found: {cfg.data.train.image}")

            # Try to infer dataset name from path
            from ...data.download import DATASETS, download_dataset

            path_str = str(cfg.data.train.image).lower()
            dataset_name = None
            for name in DATASETS.keys():
                if name in path_str and not name.endswith("++"):  # Skip aliases
                    dataset_name = name
                    break

            if dataset_name:
                logger.info(f"Attempting to auto-download '{dataset_name}' dataset...")
                logger.info("You can disable auto-download by manually downloading data")

                # Prompt user
                try:
                    size_mb = DATASETS[dataset_name]["size_mb"]
                    prompt = f"   Download {dataset_name} dataset (~{size_mb} MB)? [Y/n]: "
                    response = input(prompt).strip().lower()
                    if response in ["", "y", "yes"]:
                        if download_dataset(dataset_name, base_dir=PathLib.cwd()):
                            logger.info("Data downloaded successfully!")
                        else:
                            logger.warning("Download failed. Please download manually:")
                            logger.warning(f"wget {DATASETS[dataset_name]['url']}")
                            raise FileNotFoundError(
                                f"Training data not found: {cfg.data.train.image}"
                            )
                    else:
                        logger.warning("Download cancelled. Please download manually.")
                        raise FileNotFoundError(f"Training data not found: {cfg.data.train.image}")
                except KeyboardInterrupt:
                    logger.warning("Download cancelled by user")
                    raise FileNotFoundError(f"Training data not found: {cfg.data.train.image}")
            else:
                logger.info("Available datasets:")
                from ...data.download import list_datasets

                list_datasets()
                raise FileNotFoundError(f"Training data not found: {cfg.data.train.image}")

    # Check dataset type early
    dataset_type = getattr(cfg.data.train, "dataset_type", None) or getattr(
        cfg.data.val, "dataset_type", None
    )

    # Auto-precompute label_aux (skeleton/SDT) before building transforms so the
    # active splits include label_aux in their load/crop pipeline.
    if mode == "train":
        _populate_split_label_aux_if_needed(cfg, "train")
        _populate_split_label_aux_if_needed(cfg, "val")

    # Build transforms
    train_transforms = build_train_transforms(cfg)
    val_transforms = build_val_transforms(cfg)
    lazy_test_inference = bool(
        mode in ["test", "tune"]
        and getattr(getattr(cfg.inference, "sliding_window", None), "lazy_load", False)
    )
    if lazy_test_inference:
        test_transforms = Compose([])
        logger.info(
            "Lazy sliding-window inference enabled for %s mode: test volumes will stay on disk "
            "and be read ROI-by-ROI during inference.",
            mode,
        )
    else:
        test_transforms = (
            build_test_transforms(cfg, mode=mode) if mode in ["test", "tune"] else val_transforms
        )

    logger.info(f"Train transforms: {len(train_transforms.transforms)} steps")
    logger.info(f"Val transforms: {len(val_transforms.transforms)} steps")
    if mode in ["test", "tune"]:
        logger.info(
            f"Test transforms: {len(test_transforms.transforms)} steps (no sliding-window crop)"
        )

    # For test/tune modes, skip training data setup entirely
    if mode in ["test", "tune"]:
        train_data_dicts = []
        val_data_dicts = None
    # Check if automatic train/val split is enabled
    elif cfg.data.split_enabled and not cfg.data.val.image:
        logger.info("Using automatic train/val split (DeepEM-style)")

        from ...data.datasets.split import split_volume_train_val

        train_path = Path(cfg.data.train.image)
        volume_shape = get_vol_shape(str(train_path))

        logger.info(f"Volume shape: {volume_shape}")

        # Calculate split ranges
        train_ratio = cfg.data.split_train_range[1] - cfg.data.split_train_range[0]

        train_slices, val_slices = split_volume_train_val(
            volume_shape=volume_shape,
            train_ratio=train_ratio,
            axis=cfg.data.split_axis,
        )

        # Calculate train and val regions
        axis = cfg.data.split_axis
        train_start = int(volume_shape[axis] * cfg.data.split_train_range[0])
        train_end = int(volume_shape[axis] * cfg.data.split_train_range[1])
        val_start = int(volume_shape[axis] * cfg.data.split_val_range[0])
        val_end = int(volume_shape[axis] * cfg.data.split_val_range[1])

        logger.info(f"Split axis: {axis} ({'Z' if axis == 0 else 'Y' if axis == 1 else 'X'})")
        logger.info(f"Train region: [{train_start}:{train_end}] ({train_end - train_start} slices)")
        logger.info(f"Val region: [{val_start}:{val_end}] ({val_end - val_start} slices)")

        if cfg.data.split_pad_val:
            target_size = tuple(cfg.data.dataloader.patch_size)
            logger.info(f"Val padding enabled: target size = {target_size}")

        # Create data dictionaries with split info
        train_label_aux_paths = (
            expand_file_paths(cfg.data.train.label_aux) if cfg.data.train.label_aux else None
        )
        train_data_dicts = create_data_dicts_from_paths(
            image_paths=[cfg.data.train.image],
            label_paths=[cfg.data.train.label] if cfg.data.train.label else None,
            label_aux_paths=train_label_aux_paths,
        )

        # Add split metadata to train dict
        train_data_dicts[0]["split_slices"] = train_slices
        train_data_dicts[0]["split_mode"] = "train"

        # Create validation data dicts using same volume
        val_data_dicts = create_data_dicts_from_paths(
            image_paths=[cfg.data.train.image],
            label_paths=[cfg.data.train.label] if cfg.data.train.label else None,
            label_aux_paths=train_label_aux_paths,
        )

        # Add split metadata to val dict
        val_data_dicts[0]["split_slices"] = val_slices
        val_data_dicts[0]["split_mode"] = "val"
        val_data_dicts[0]["split_pad"] = cfg.data.split_pad_val
        val_data_dicts[0]["split_pad_mode"] = cfg.data.split_pad_mode
        if cfg.data.split_pad_val:
            val_data_dicts[0]["split_pad_size"] = tuple(cfg.data.dataloader.patch_size)

    else:
        # Check dataset type to determine how to load data
        if dataset_type == "filename":
            # Check if train_json is empty or doesn't exist
            train_json_empty = False
            if cfg.data.train.json is None or cfg.data.train.json == "":
                train_json_empty = True
            else:
                try:
                    import json

                    json_path = Path(cfg.data.train.json)
                    if not json_path.exists():
                        train_json_empty = True
                    else:
                        # Check if JSON file is empty or has no images
                        with open(json_path) as f:
                            json_data = json.load(f)
                        image_files = json_data.get(cfg.data.train.image_key, [])
                        if not image_files:
                            train_json_empty = True
                except (FileNotFoundError, json.JSONDecodeError, KeyError):
                    train_json_empty = True

            if train_json_empty:
                # Fallback to volume-based dataset when train_json is empty
                logger.warning(
                    "Train JSON is empty or invalid, falling back to volume-based dataset"
                )
                logger.warning(f"Train JSON: {cfg.data.train.json}")
                dataset_type = None  # Switch to volume-based
            else:
                # Filename-based dataset: uses JSON file lists
                logger.info("Using filename-based dataset")
                logger.info(f"Train JSON: {cfg.data.train.json}")
                logger.info(f"Image key: {cfg.data.train.image_key}")
                logger.info(f"Label key: {cfg.data.train.label_key}")

                # For filename dataset, we'll create data dicts later in the DataModule
                # Here we just need placeholder dicts
                train_data_dicts = [{"dataset_type": "filename"}]
                val_data_dicts = None  # Handled by train_val_split in DataModule

        if dataset_type != "filename":
            # Standard mode: separate train and val files (supports glob patterns)
            if cfg.data.train.image is None:
                raise ValueError(
                    "For volume-based datasets, data.train.image must be specified.\n"
                    "Either set data.train.image or use "
                    "data.dataset_type='filename' with data.train.json"
                )

            train_image_paths = expand_file_paths(cfg.data.train.image)
            train_label_paths = (
                expand_file_paths(cfg.data.train.label) if cfg.data.train.label else None
            )
            train_mask_paths = (
                expand_file_paths(cfg.data.train.mask) if cfg.data.train.mask else None
            )

            logger.info(f"Training volumes: {len(train_image_paths)} files")
            if len(train_image_paths) <= 5:
                for path in train_image_paths:
                    logger.info(f"  - {path}")
            else:
                logger.info(f"  - {train_image_paths[0]}")
                logger.info(f"  - ... ({len(train_image_paths) - 2} more files)")
                logger.info(f"  - {train_image_paths[-1]}")

            if train_mask_paths:
                logger.info(f"Training masks: {len(train_mask_paths)} files")

            # label_aux: already set by early precompute step or explicit config.
            train_label_aux_paths = (
                expand_file_paths(cfg.data.train.label_aux) if cfg.data.train.label_aux else None
            )

            train_data_dicts = create_data_dicts_from_paths(
                image_paths=train_image_paths,
                label_paths=train_label_paths,
                label_aux_paths=train_label_aux_paths,
                mask_paths=train_mask_paths,
            )

            val_data_dicts = None
            if cfg.data.val.image:
                val_image_paths = expand_file_paths(cfg.data.val.image)
                val_label_paths = (
                    expand_file_paths(cfg.data.val.label) if cfg.data.val.label else None
                )
                val_mask_paths = expand_file_paths(cfg.data.val.mask) if cfg.data.val.mask else None

                logger.info(f"Validation volumes: {len(val_image_paths)} files")
                if val_mask_paths:
                    logger.info(f"Validation masks: {len(val_mask_paths)} files")

                val_label_aux_paths = (
                    expand_file_paths(cfg.data.val.label_aux) if cfg.data.val.label_aux else None
                )

                val_data_dicts = create_data_dicts_from_paths(
                    image_paths=val_image_paths,
                    label_paths=val_label_paths,
                    label_aux_paths=val_label_aux_paths,
                    mask_paths=val_mask_paths,
                )

    # Create test data dicts if in test or tune mode
    test_data_dicts = None
    if mode == "test":
        split = cfg.data.test
        # Skip image validation when using decoding.input_prediction_path (decode-only)
        _saved = getattr(getattr(cfg, "decoding", None), "input_prediction_path", "")
        if not split.image and not _saved:
            raise ValueError(
                "Test mode requires data.test.image to be set.\n"
                f"Current resolved image = {split.image}"
            )
        if split.image:
            logger.info(f"Creating test dataset from: {split.image}")
            test_image_paths = expand_file_paths(split.image)
        else:
            # Decode-only: derive filename from decoding.input_prediction_path
            from pathlib import Path

            pred_stem = Path(_saved).stem if _saved else "decoded"
            test_image_paths = [pred_stem]
            logger.info(f"Decode-only mode: using filename from input_prediction_path: {pred_stem}")

        test_label_paths = expand_file_paths(split.label) if split.label else None
        test_label_aux_paths = expand_file_paths(split.label_aux) if split.label_aux else None
        test_mask_paths = expand_file_paths(split.mask) if split.mask else None
    elif mode == "tune":
        split = cfg.data.val
        if not split.image:
            raise ValueError(
                "Tune mode requires data.val.image to be set.\n"
                f"Current resolved val.image = {getattr(cfg.data.val, 'image', None)}"
            )

        logger.info(f"Creating tune dataset from: {split.image}")

        # Expand glob patterns for tune data
        test_image_paths = expand_file_paths(split.image)
        test_label_paths = expand_file_paths(split.label) if split.label else None
        test_label_aux_paths = expand_file_paths(split.label_aux) if split.label_aux else None
        test_mask_paths = expand_file_paths(split.mask) if split.mask else None

    # Common printing and data dict creation for test and tune modes
    if mode in ["test", "tune"]:
        mode_label = "Test" if mode == "test" else "Tune"
        logger.info(f"{mode_label} volumes: {len(test_image_paths)} files")
        if len(test_image_paths) <= 5:
            for path in test_image_paths:
                logger.info(f"  - {path}")
        else:
            logger.info(f"  - {test_image_paths[0]}")
            logger.info(f"  - ... ({len(test_image_paths) - 2} more files)")
            logger.info(f"  - {test_image_paths[-1]}")

        if test_mask_paths:
            logger.info(f"{mode_label} masks: {len(test_mask_paths)} files")

        test_data_dicts = create_data_dicts_from_paths(
            image_paths=test_image_paths,
            label_paths=test_label_paths,
            label_aux_paths=test_label_aux_paths,
            mask_paths=test_mask_paths,
        )
        logger.info(f"{mode_label} dataset size: {len(test_data_dicts)}")

    if mode == "train":
        logger.info(f"Train dataset size: {len(train_data_dicts)}")
        if val_data_dicts:
            logger.info(f"Val dataset size: {len(val_data_dicts)}")

    # Auto-compute iter_num from volume size if not specified (only for training).
    # IMPORTANT: cfg.optimization.n_steps_per_epoch is interpreted as optimizer steps/epoch.
    # Dataset iter_num is sample-count based, so we convert steps -> samples.
    iter_num = None
    if mode == "train":
        iter_num_cfg = cfg.optimization.n_steps_per_epoch
        if iter_num_cfg > 0:
            # Convert requested steps/epoch to per-epoch sample count expected by datasets.
            # Account for per-device batch size, number of training devices, and
            # gradient accumulation (each optimizer step consumes accumulate_grad_batches
            # dataloader batches).
            num_devices = cfg.system.num_gpus if cfg.system.num_gpus > 0 else 1
            accumulate = max(1, int(cfg.optimization.accumulate_grad_batches))
            iter_num = int(iter_num_cfg * cfg.data.dataloader.batch_size * num_devices * accumulate)
            logger.info(
                f"Requested n_steps_per_epoch={iter_num_cfg} steps -> "
                f"dataset samples={iter_num} "
                f"(batch_size={cfg.data.dataloader.batch_size}, devices={num_devices}, "
                f"accumulate_grad_batches={accumulate})"
            )
        elif iter_num_cfg == -1 and dataset_type != "filename":
            # For filename datasets, iter_num is determined by the number of files
            logger.info("Auto-computing iter_num from volume size...")

            from ...data.datasets.sampling import compute_total_samples
            from ...data.io import get_vol_shape

            # Get volume sizes
            volume_sizes = []
            for data_dict in train_data_dicts:
                vol_shape = get_vol_shape(str(data_dict["image"]))

                # Handle both (z, y, x) and (c, z, y, x)
                if len(vol_shape) == 4:
                    vol_shape = vol_shape[1:]  # Skip channel dim
                volume_sizes.append(vol_shape)

            # Compute total possible samples
            total_samples, samples_per_vol = compute_total_samples(
                volume_sizes=volume_sizes,
                patch_size=_effective_patch_size(cfg),
                stride=tuple(cfg.data.data_transform.stride),
            )

            iter_num = total_samples
            logger.info(f"Volume sizes: {volume_sizes}")
            logger.info(f"Patch size: {cfg.data.dataloader.patch_size}")
            logger.info(f"Stride: {cfg.data.data_transform.stride}")
            logger.info(f"Samples per volume: {samples_per_vol}")
            logger.info(f"Total possible samples (iter_num): {iter_num:,}")
            # Approximate optimizer steps/epoch for informational logging.
            num_devices = cfg.system.num_gpus if cfg.system.num_gpus > 0 else 1
            accumulate = max(1, int(cfg.optimization.accumulate_grad_batches))
            denom = max(1, cfg.data.dataloader.batch_size * num_devices * accumulate)
            logger.info(f"Approx optimizer steps per epoch: {iter_num // denom:,}")
        elif iter_num_cfg == -1 and dataset_type == "filename":
            # For filename datasets, iter_num will be determined by dataset length
            logger.info("Filename dataset: iter_num will be determined by number of files in JSON")

    # Create DataModule
    logger.info("Creating data loaders...")

    # For test/tune modes, disable iter_num (process full volumes once)
    if mode in ["test", "tune"]:
        iter_num_for_dataset: int | None = -1  # Process full volumes without random sampling
    else:
        iter_num_for_dataset = iter_num

    # system/data are already mode-resolved via section merging in setup_config.
    batch_size = cfg.data.dataloader.batch_size
    num_workers = cfg.system.num_workers
    persistent_workers_cfg = bool(cfg.data.dataloader.persistent_workers)
    logger.info(f"Using runtime settings: batch_size={batch_size}, num_workers={num_workers}")

    # Explicit preload settings (no legacy fallback).
    train_preload_cfg = cfg.data.dataloader.use_preloaded_cache_train
    val_preload_cfg = cfg.data.dataloader.use_preloaded_cache_train

    # Use optimized pre-loaded cache for train dataset when iter_num > 0.
    use_preloaded = (
        train_preload_cfg
        and iter_num is not None
        and iter_num > 0
        and mode == "train"
        and dataset_type != "filename"
    )
    use_lazy_zarr = (
        cfg.data.dataloader.use_lazy_zarr
        and iter_num is not None
        and iter_num > 0
        and mode == "train"
        and dataset_type != "filename"
    )
    use_lazy_h5 = (
        cfg.data.dataloader.use_lazy_h5
        and iter_num is not None
        and iter_num > 0
        and mode == "train"
        and dataset_type != "filename"
    )
    if use_lazy_zarr and use_lazy_h5:
        raise ValueError("data.dataloader.use_lazy_zarr and use_lazy_h5 are mutually exclusive.")

    if use_preloaded:
        logger.info("Using pre-loaded volume cache (loads once, crops in memory)")
        from torch.utils.data import DataLoader

        from ...data.datasets.dataset_volume_cached import CachedVolumeDataset
        from ...data.processing.nnunet_preprocess import NNUNetPreprocessd

        def _build_preloaded_nnunet_preprocess(split: str):
            nnunet_pre_cfg = getattr(cfg.data, "nnunet_preprocessing", None)
            if not bool(getattr(nnunet_pre_cfg, "enabled", False)):
                return None

            if split == "val":
                source_spacing = getattr(cfg.data.val, "resolution", None) or getattr(
                    cfg.data.train, "resolution", None
                )
            else:
                source_spacing = getattr(cfg.data.train, "resolution", None)
            source_spacing = getattr(nnunet_pre_cfg, "source_spacing", None) or source_spacing

            logger.info(f"Applying nnU-Net preprocessing before caching ({split})")
            return NNUNetPreprocessd(
                keys=["image", "label", "mask"],
                image_key="image",
                enabled=True,
                crop_to_nonzero=getattr(nnunet_pre_cfg, "crop_to_nonzero", True),
                source_spacing=source_spacing,
                target_spacing=getattr(nnunet_pre_cfg, "target_spacing", None),
                normalization=getattr(nnunet_pre_cfg, "normalization", "zscore"),
                normalization_use_nonzero_mask=getattr(
                    nnunet_pre_cfg, "normalization_use_nonzero_mask", True
                ),
                clip_percentile_low=getattr(nnunet_pre_cfg, "clip_percentile_low", 0.0),
                clip_percentile_high=getattr(nnunet_pre_cfg, "clip_percentile_high", 1.0),
                force_separate_z=getattr(nnunet_pre_cfg, "force_separate_z", None),
                anisotropy_threshold=getattr(nnunet_pre_cfg, "anisotropy_threshold", 3.0),
                image_order=getattr(nnunet_pre_cfg, "image_order", 3),
                label_order=getattr(nnunet_pre_cfg, "label_order", 0),
                order_z=getattr(nnunet_pre_cfg, "order_z", 0),
                allow_missing_keys=True,
            )

        # Build transforms without loading/cropping (handled by dataset)
        augment_only_transforms = build_train_transforms(cfg, skip_loading=True)
        train_pre_cache_transforms = _build_preloaded_nnunet_preprocess("train")

        # Get padding parameters from data_transform config
        pad_size = cfg.data.data_transform.pad_size
        pad_mode = cfg.data.data_transform.pad_mode

        # Create optimized cached datasets
        train_dataset = CachedVolumeDataset(
            image_paths=[d["image"] for d in train_data_dicts],
            label_paths=[d.get("label") for d in train_data_dicts],
            label_aux_paths=(
                [d.get("label_aux") for d in train_data_dicts]
                if any(d.get("label_aux") for d in train_data_dicts)
                else None
            ),
            mask_paths=[d.get("mask") for d in train_data_dicts],
            patch_size=_effective_patch_size(cfg),
            iter_num=iter_num,
            transforms=augment_only_transforms,
            pre_cache_transforms=train_pre_cache_transforms,
            mode="train",
            pad_size=tuple(pad_size) if pad_size else None,
            pad_mode=pad_mode,
            max_attempts=cfg.data.dataloader.cached_sampling_max_attempts,
            foreground_threshold=cfg.data.dataloader.cached_sampling_foreground_threshold,
            crop_to_nonzero_mask=cfg.data.dataloader.cached_sampling_crop_to_nonzero_mask,
            sample_nonzero_mask=cfg.data.dataloader.cached_sampling_sample_nonzero_mask,
        )

        preloaded_num_workers = num_workers
        logger.info(f"Using {preloaded_num_workers} workers")

        # Create simple dataloader
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,  # Already random
            num_workers=preloaded_num_workers,
            pin_memory=cfg.data.dataloader.pin_memory,
            persistent_workers=preloaded_num_workers > 0,
        )

        logger.info(f"Preload policy: train={train_preload_cfg}, val={val_preload_cfg}")

        # Create validation dataset and loader if validation data exists
        val_loader = None
        if val_data_dicts and len(val_data_dicts) > 0:
            if val_preload_cfg:
                logger.info("Creating validation dataset with pre-loaded cache...")
            else:
                logger.info("Creating validation dataset without pre-loaded cache...")

            # Build validation transforms (no augmentation, only normalization)
            val_only_transforms = build_val_transforms(cfg, skip_loading=True)
            val_pre_cache_transforms = _build_preloaded_nnunet_preprocess("val")

            # Get validation steps/epoch (auto-calculate if not specified)
            val_steps_per_epoch = cfg.optimization.val_steps_per_epoch

            if val_steps_per_epoch is None:
                # Auto-calculate validation steps from volume size
                logger.info("Auto-calculating validation steps from volume size...")
                val_steps_per_epoch = _calculate_validation_steps_per_epoch(
                    val_data_dicts=val_data_dicts,
                    patch_size=_effective_patch_size(cfg),
                    min_steps=1,
                    max_steps=None,
                    default_steps=100,
                    fallback_volume_shape=(100, 4096, 4096),
                    return_default_on_error=False,
                )
                logger.info(f"Validation steps: {val_steps_per_epoch} (auto-calculated)")

            # Create validation dataset
            if val_preload_cfg:
                val_dataset = CachedVolumeDataset(
                    image_paths=[d["image"] for d in val_data_dicts],
                    label_paths=[d.get("label") for d in val_data_dicts],
                    label_aux_paths=(
                        [d.get("label_aux") for d in val_data_dicts]
                        if any(d.get("label_aux") for d in val_data_dicts)
                        else None
                    ),
                    mask_paths=[d.get("mask") for d in val_data_dicts],
                    patch_size=_effective_patch_size(cfg),
                    iter_num=val_steps_per_epoch,
                    transforms=val_only_transforms,
                    pre_cache_transforms=val_pre_cache_transforms,
                    mode=_validation_dataset_mode(cfg),
                    pad_size=tuple(pad_size) if pad_size else None,
                    pad_mode=pad_mode,
                    max_attempts=cfg.data.dataloader.cached_sampling_max_attempts,
                    foreground_threshold=cfg.data.dataloader.cached_sampling_foreground_threshold,
                    crop_to_nonzero_mask=cfg.data.dataloader.cached_sampling_crop_to_nonzero_mask,
                    sample_nonzero_mask=cfg.data.dataloader.cached_sampling_sample_nonzero_mask,
                )
            else:
                from monai.data import CacheDataset, Dataset

                from .data import _IterNumDataset

                use_cache = cfg.data.dataloader.use_cache
                if use_cache:
                    val_dataset = CacheDataset(
                        data=val_data_dicts,
                        transform=val_transforms,
                        cache_rate=cfg.data.dataloader.cache_rate,
                    )
                else:
                    val_dataset = Dataset(
                        data=val_data_dicts,
                        transform=val_transforms,
                    )

                if val_steps_per_epoch and val_steps_per_epoch > 0:
                    val_dataset = _IterNumDataset(val_dataset, val_steps_per_epoch)

            # Create validation dataloader
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=preloaded_num_workers,
                pin_memory=cfg.data.dataloader.pin_memory,
                persistent_workers=preloaded_num_workers > 0,
            )
            logger.info(f"Validation dataloader created with {val_steps_per_epoch} steps")

        datamodule = SimpleDataModule(train_loader, val_loader)
    elif use_lazy_zarr or use_lazy_h5:
        # Lazy crop loading: keep volume handles open, read only sampled patches.
        if use_lazy_h5:
            from ...data.datasets.dataset_volume_h5_lazy import LazyH5VolumeDataset

            LazyDatasetCls = LazyH5VolumeDataset
            backend_name = "h5"
            flag_name = "use_lazy_h5"

            def _path_ok(p) -> bool:
                s = str(p)
                return ".h5" in s or ".hdf5" in s

        else:
            from ...data.datasets.dataset_volume_zarr_lazy import LazyZarrVolumeDataset

            LazyDatasetCls = LazyZarrVolumeDataset
            backend_name = "zarr"
            flag_name = "use_lazy_zarr"

            def _path_ok(p) -> bool:
                return ".zarr" in str(p)

        train_images = [d["image"] for d in train_data_dicts]
        train_labels = [d.get("label") for d in train_data_dicts]
        train_label_auxs = [d.get("label_aux") for d in train_data_dicts]
        train_masks = [d.get("mask") for d in train_data_dicts]
        if not all(_path_ok(p) for p in train_images):
            raise ValueError(
                f"data.{flag_name}=true requires {backend_name} image paths. "
                f"Got: {train_images[:3]}"
            )

        logger.info("Using lazy %s volume loading (crop-on-read, no full preload)", backend_name)
        from torch.utils.data import DataLoader

        train_transforms_lazy = build_train_transforms(cfg, skip_loading=True)

        train_dataset = LazyDatasetCls(
            image_paths=train_images,
            label_paths=None if all(p is None for p in train_labels) else train_labels,
            label_aux_paths=None if all(p is None for p in train_label_auxs) else train_label_auxs,
            mask_paths=None if all(p is None for p in train_masks) else train_masks,
            patch_size=_effective_patch_size(cfg),
            iter_num=iter_num,
            transforms=train_transforms_lazy,
            mode="train",
            max_attempts=cfg.data.dataloader.cached_sampling_max_attempts,
            foreground_threshold=cfg.data.dataloader.cached_sampling_foreground_threshold,
            transpose_axes=(
                cfg.data.data_transform.train_transpose
                if cfg.data.data_transform.train_transpose
                else None
            ),
        )

        lazy_num_workers = num_workers
        logger.info(f"Using {lazy_num_workers} workers")
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=lazy_num_workers,
            pin_memory=cfg.data.dataloader.pin_memory,
            persistent_workers=lazy_num_workers > 0,
        )

        val_loader = None
        if val_data_dicts and len(val_data_dicts) > 0:
            val_images = [d["image"] for d in val_data_dicts]
            val_labels = [d.get("label") for d in val_data_dicts]
            val_label_aux = [d.get("label_aux") for d in val_data_dicts]
            val_masks = [d.get("mask") for d in val_data_dicts]
            if not all(_path_ok(p) for p in val_images):
                raise ValueError(f"data.{flag_name}=true requires {backend_name} val image paths.")

            val_transforms_lazy = build_val_transforms(cfg, skip_loading=True)
            val_steps_per_epoch = cfg.optimization.val_steps_per_epoch
            if val_steps_per_epoch is None:
                logger.info("Auto-calculating validation steps from volume size...")
                val_steps_per_epoch = _calculate_validation_steps_per_epoch(
                    val_data_dicts=val_data_dicts,
                    patch_size=_effective_patch_size(cfg),
                    min_steps=50,
                    max_steps=200,
                )
                logger.info(f"Validation steps: {val_steps_per_epoch} (auto-calculated)")

            val_dataset = LazyDatasetCls(
                image_paths=val_images,
                label_paths=None if all(p is None for p in val_labels) else val_labels,
                label_aux_paths=None if all(p is None for p in val_label_aux) else val_label_aux,
                mask_paths=None if all(p is None for p in val_masks) else val_masks,
                patch_size=_effective_patch_size(cfg),
                iter_num=val_steps_per_epoch,
                transforms=val_transforms_lazy,
                mode=_validation_dataset_mode(cfg),
                max_attempts=cfg.data.dataloader.cached_sampling_max_attempts,
                foreground_threshold=cfg.data.dataloader.cached_sampling_foreground_threshold,
                transpose_axes=(
                    cfg.data.data_transform.val_transpose
                    if cfg.data.data_transform.val_transpose
                    else None
                ),
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=lazy_num_workers,
                pin_memory=cfg.data.dataloader.pin_memory,
                persistent_workers=lazy_num_workers > 0,
            )
            logger.info(f"Validation dataloader created with {val_steps_per_epoch} steps")

        datamodule = SimpleDataModule(train_loader, val_loader)
    elif dataset_type == "filename":
        # Filename-based dataset using JSON file lists
        logger.info("Creating filename-based datamodule...")
        from torch.utils.data import DataLoader

        from ...data.datasets.dataset_filename import create_filename_datasets

        # Create train and val datasets from JSON
        train_dataset, val_dataset = create_filename_datasets(
            json_path=cfg.data.train.json,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            train_val_split=cfg.data.train.split_ratio if cfg.data.train.split_ratio else 0.9,
            random_seed=cfg.system.seed,
            images_key=cfg.data.train.image_key,
            labels_key=cfg.data.train.label_key,
            use_labels=True,
        )

        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"Val dataset size: {len(val_dataset)}")

        # Build DataLoaders and wrap with SimpleDataModule
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=cfg.data.dataloader.pin_memory,
            persistent_workers=persistent_workers_cfg and num_workers > 0,
        )

        val_loader = None
        if val_dataset is not None and len(val_dataset) > 0:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=cfg.data.dataloader.pin_memory,
                persistent_workers=persistent_workers_cfg and num_workers > 0,
            )

        datamodule = SimpleDataModule(train_loader, val_loader)
    else:
        # Standard data module
        # Disable caching for test/tune modes to avoid issues with partial cache returning 0 length
        use_cache = cfg.data.dataloader.use_cache and mode == "train"
        tta_cfg = getattr(getattr(cfg, "inference", None), "test_time_augmentation", None)
        sliding_cfg = getattr(getattr(cfg, "inference", None), "sliding_window", None)
        distributed_tta_sharding = bool(
            mode in ["test", "tune"]
            and tta_cfg is not None
            and getattr(tta_cfg, "enabled", False)
            and getattr(tta_cfg, "distributed_sharding", False)
        )
        distributed_window_sharding = bool(
            mode in ["test", "tune"]
            and sliding_cfg is not None
            and getattr(sliding_cfg, "lazy_load", False)
            and getattr(sliding_cfg, "distributed_sharding", False)
        )

        if mode in ["test", "tune"] and cfg.data.dataloader.use_cache:
            logger.warning("Caching disabled for test/tune mode (incompatible with partial cache)")

        # Note: transpose_axes handled in transform builders (build_train/val/test_transforms)
        # They embed the transpose in LoadVolumed, so no need to pass it here

        # Get validation steps/epoch (separate from training n_steps_per_epoch)
        val_steps_per_epoch = cfg.optimization.val_steps_per_epoch
        if val_steps_per_epoch is None and val_data_dicts:
            # Auto-calculate validation steps based on volume size and patch size
            logger.info("Auto-calculating validation steps from volume size...")
            val_steps_per_epoch = _calculate_validation_steps_per_epoch(
                val_data_dicts=val_data_dicts,
                patch_size=_effective_patch_size(cfg),
                min_steps=50,
                max_steps=200,
            )
            logger.info(f"Validation steps: {val_steps_per_epoch} (auto-calculated)")

        datamodule = ConnectomicsDataModule(
            train_data_dicts=train_data_dicts,
            val_data_dicts=val_data_dicts,
            test_data_dicts=test_data_dicts,
            transforms={
                "train": train_transforms,
                "val": val_transforms,
                "test": test_transforms,
            },
            dataset_type="cached" if use_cache else "standard",
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=cfg.data.dataloader.pin_memory,
            persistent_workers=persistent_workers_cfg,
            cache_rate=cfg.data.dataloader.cache_rate if use_cache else 0.0,
            iter_num=iter_num_for_dataset,
            val_steps_per_epoch=val_steps_per_epoch,
            seed=cfg.system.seed,  # [FIX 1] Pass seed for validation reseeding
            distributed_tta_sharding=distributed_tta_sharding,
            distributed_window_sharding=distributed_window_sharding,
            sample_size=_effective_patch_size(cfg),
            do_2d=bool(
                getattr(cfg.data.train, "do_2d", False) or getattr(cfg.data.val, "do_2d", False)
            ),
        )
        # Setup datasets based on mode
        if mode == "train":
            datamodule.setup(stage="fit")
        elif mode in ["test", "tune"]:
            datamodule.setup(stage="test")

    # Print dataset info based on mode
    if mode == "train":
        logger.info(f"Train batches: {len(datamodule.train_dataloader())}")
        if val_data_dicts:
            logger.info(f"Val batches: {len(datamodule.val_dataloader())}")
    elif mode in ["test", "tune"]:
        logger.info(f"Test batches: {len(datamodule.test_dataloader())}")

    return datamodule


__all__ = [
    "create_datamodule",
    "_calculate_validation_steps_per_epoch",
]

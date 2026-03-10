#!/usr/bin/env python -i
"""
Neuroglancer visualization script for PyTorch Connectomics.

Visualize image and label volumes in a web browser using Neuroglancer.
Runs in interactive mode so you can examine loaded volumes.

Usage:
    python -i scripts/visualize_neuroglancer.py --config tutorials/monai_lucchi.yaml
    python -i scripts/visualize_neuroglancer.py --image path/to/image.tif \
        --label path/to/label.h5
    python -i scripts/visualize_neuroglancer.py --volumes image:path/img.tif \
        label:path/lbl.h5 seg:path/seg.h5

Interactive mode variables:
    viewer      - Neuroglancer viewer instance
    volumes     - Dictionary of loaded volumes {name: (data, type, resolution, offset)}
    cfg         - Config object (if loaded from --config)
    add_layer() - Helper function to add new layers from files or data
    ngLayer()   - Helper function to create Neuroglancer layers

Examples:
    # Examine volume data
    >>> volumes['train_image'][0].shape
    >>> volumes['train_image'][0].dtype

    # Access raw numpy arrays
    >>> img = volumes['train_image'][0]
    >>> lbl = volumes['train_label'][0]

    # Add new layer from file
    >>> add_layer('prediction', file_path='outputs/pred.h5', res=[5, 5, 5], tt='seg')

    # Add layer from existing data
    >>> add_layer('filtered', data=my_array, res=[30, 6, 6], tt='image')

    # Manual layer creation (advanced)
    >>> with viewer.txn() as s:
    ...     s.layers.append(name='custom', layer=ngLayer(data, [5, 5, 5], tt='image'))
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from connectomics.config import (  # noqa: E402
    load_config,
    resolve_data_paths,
    resolve_default_profiles,
)

# Import connectomics modules (needed for helper functions)
from connectomics.data.io import read_volume  # noqa: E402

# Lazy import for neuroglancer (checked at runtime)
if TYPE_CHECKING:
    import neuroglancer
else:
    neuroglancer = None  # Will be imported in main() after arg parsing

logger = logging.getLogger(__name__)


def normalize_resolution_zyx(
    resolution: Optional[Sequence[float]],
    *,
    context: str = "resolution",
) -> Optional[Tuple[float, float, float]]:
    """Normalize resolution to explicit zyx order.

    Accepts:
    - 3 values: interpreted as (z, y, x)
    - 2 values: interpreted as (y, x) and padded to (1.0, y, x)
    """
    if resolution is None:
        return None

    values = tuple(float(v) for v in resolution)
    if len(values) == 3:
        return values  # already zyx
    if len(values) == 2:
        return (1.0, values[0], values[1])
    raise ValueError(
        f"{context} must have 2 or 3 values in zyx convention; got {len(values)}: {values}"
    )


def apply_default_scaling(data: np.ndarray, scale: float, is_image: bool = True) -> np.ndarray:
    """
    Keep loaded volume intensities unchanged.

    Args:
        data: Input array
        scale: Retained for CLI backward compatibility (ignored).
        is_image: Retained for backward compatibility (ignored).

    Returns:
        Unchanged input array
    """
    return data


def apply_image_transform(data: np.ndarray, cfg) -> np.ndarray:
    """
    Apply image transformations from config (normalization and clipping).

    Args:
        data: Input image array
        cfg: Config object with image_transform settings

    Returns:
        Transformed image array (same dtype as input)
    """
    if not hasattr(cfg.data, "image_transform"):
        return data

    transform_cfg = cfg.data.image_transform
    original_dtype = data.dtype
    data = data.astype(np.float32)  # Work in float32
    normalize_mode = getattr(transform_cfg, "normalize", "none")

    print("  Applying image transformations from config:")
    print(f"    Original range: [{data.min():.2f}, {data.max():.2f}]")

    # Percentile clipping
    if hasattr(transform_cfg, "clip_percentile_low") and hasattr(
        transform_cfg, "clip_percentile_high"
    ):
        low_pct = transform_cfg.clip_percentile_low
        high_pct = transform_cfg.clip_percentile_high

        if low_pct > 0.0 or high_pct < 1.0:
            low_val = np.percentile(data, low_pct * 100)
            high_val = np.percentile(data, high_pct * 100)
            print(
                f"    Clipping: {low_pct*100:.1f}th percentile ({low_val:.2f}) "
                f"to {high_pct*100:.1f}th percentile ({high_val:.2f})"
            )
            data = np.clip(data, low_val, high_val)

    # Normalization
    if hasattr(transform_cfg, "normalize"):
        if normalize_mode == "0-1":
            # Min-max normalization to [0, 1]
            data_min = data.min()
            data_max = data.max()
            if data_max > data_min:
                data = (data - data_min) / (data_max - data_min)
                print("    Normalized to [0, 1] (min-max)")
            else:
                print("    Warning: data_min == data_max, skipping normalization")

        elif normalize_mode == "normal":
            # Z-score normalization
            data_mean = data.mean()
            data_std = data.std()
            if data_std > 0:
                data = (data - data_mean) / data_std
                print(f"    Normalized (z-score): mean={data_mean:.2f}, std={data_std:.2f}")
            else:
                print("    Warning: data_std == 0, skipping normalization")

        elif normalize_mode == "none":
            print("    No normalization applied")

    print(f"    Final range: [{data.min():.2f}, {data.max():.2f}]")

    # Convert back to original dtype for visualization
    if normalize_mode == "0-1" and original_dtype == np.uint8:
        data = (data * 255).astype(np.uint8)
    elif normalize_mode == "0-1" and original_dtype == np.uint16:
        data = (data * 65535).astype(np.uint16)
    else:
        # Keep as float32 for normalized data
        pass

    return data


def _ensure_spatial_3d(data: np.ndarray, *, context: str) -> np.ndarray:
    """Insert a singleton z-axis for 2D arrays so downstream logic always sees 3D."""
    if data.ndim != 2:
        return data

    converted = data[None, :, :]
    logger.info("Converted 2D %s to 3D: %s", context, converted.shape)
    return converted


def _select_file_paths(
    files_list: Sequence[str],
    selector: str,
    *,
    allow_all: bool,
) -> list[str]:
    """Select one or more files from a resolved file list."""
    files = list(files_list)
    if len(files) <= 1:
        return files

    if selector.lower() == "all":
        if allow_all:
            print(f"  Found {len(files)} files, loading all...")
            return files
        print(
            f"  Found {len(files)} files, 'all' not supported here, using first: "
            f"{Path(files[0]).name}"
        )
        return [files[0]]

    try:
        index = int(selector)
    except ValueError:
        matching = [f for f in files if Path(f).name == selector or Path(f).stem == selector]
        if not matching:
            matching = [f for f in files if selector in Path(f).name]
        if matching:
            print(
                f"  Found {len(files)} files, selected by name '{selector}': "
                f"{Path(matching[0]).name}"
            )
            return [matching[0]]

        print(
            f"  Warning: No file matches selector '{selector}', using first of "
            f"{len(files)} files"
        )
        return [files[0]]

    if index < -len(files) or index >= len(files):
        print(f"  Warning: Index {index} out of range for {len(files)} files, using first")
        return [files[0]]

    selected = files[index]
    print(f"  Found {len(files)} files, selected index [{index}]: {Path(selected).name}")
    return [selected]


def _load_config_volume_array(
    path: str,
    *,
    context: str,
    cfg=None,
    apply_transform: bool = False,
) -> np.ndarray:
    """Read a config-backed volume, normalize 2D inputs, and optionally transform images."""
    data = read_volume(path)
    data = _ensure_spatial_3d(data, context=context)
    if apply_transform:
        data = apply_image_transform(data, cfg)
    return data


def _resolve_prediction_matched_path(
    test_image_path: str,
    prediction_base_name: Optional[str],
) -> str:
    """Best-effort match of a test image path to a prediction basename."""
    if (
        not prediction_base_name
        or not isinstance(test_image_path, str)
        or ("*" not in test_image_path and "?" not in test_image_path)
    ):
        return test_image_path

    print(f"  Auto-matching specific test_image for prediction base name: {prediction_base_name}")
    test_path_obj = Path(test_image_path)
    test_dir = test_path_obj.parent
    print(f"  Search directory: {test_dir}")

    extensions_to_try = [".tif", ".tiff", ".h5", ".hdf5", ".png", ".jpg", ".jpeg"]
    for ext in extensions_to_try:
        potential_file = test_dir / f"{prediction_base_name}{ext}"
        if potential_file.exists():
            matched_file = str(potential_file)
            print(f"  Found matching test_image: {matched_file}")
            return matched_file

    matching_files = sorted(test_dir.glob(f"{prediction_base_name}.*"))
    if matching_files:
        matched_file = str(matching_files[0])
        print(f"  Found matching test_image: {matched_file}")
        return matched_file

    print(f"  No matching test_image found for base name: {prediction_base_name}")
    print("  Falling back to loading all files from glob pattern")
    return test_image_path


def _store_config_volume_entries(
    volumes: Dict[str, Tuple[np.ndarray, str, Optional[Tuple], None]],
    raw_paths,
    *,
    select: str,
    allow_all: bool,
    display_label: str,
    volume_name: str,
    volume_type: str,
    resolution: Optional[Tuple[float, float, float]],
    cfg=None,
    apply_transform: bool = False,
) -> None:
    """Load one config entry that may resolve to a single path or a list of paths."""
    if not raw_paths:
        return

    print(f"Loading {display_label}: {raw_paths}")
    if isinstance(raw_paths, list):
        paths_to_load = _select_file_paths(raw_paths, select, allow_all=allow_all)
    else:
        paths_to_load = [raw_paths]

    for idx, path in enumerate(paths_to_load):
        if len(paths_to_load) > 1:
            print(f"  [{idx + 1}/{len(paths_to_load)}] Loading: {path}")
        try:
            data = _load_config_volume_array(
                path,
                context=volume_name,
                cfg=cfg,
                apply_transform=apply_transform,
            )
            resolved_name = f"{volume_name}_{idx}" if len(paths_to_load) > 1 else volume_name
            volumes[resolved_name] = (data, volume_type, resolution, None)
            print(f"      Loaded: {data.shape}, dtype={data.dtype}")
        except Exception as exc:
            print(f"      Error loading {path}: {exc}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize volumes with Neuroglancer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From config file (interactive mode recommended with -i)
  python -i scripts/visualize_neuroglancer.py --config tutorials/monai_lucchi.yaml

  # Specify files directly
  python -i scripts/visualize_neuroglancer.py \\
    --image datasets/lucchi/img/train_im.tif \\
    --label datasets/lucchi/label/train_label.tif

  # Multiple volumes with custom names (type inferred from name)
  python -i scripts/visualize_neuroglancer.py \\
    --volumes image:datasets/img.tif label:datasets/label.h5 prediction:outputs/pred.h5

  # Volumes with explicit type (format: name:type:path)
  python -i scripts/visualize_neuroglancer.py \\
    --volumes output:image:outputs/pred.h5 ground_truth:seg:datasets/label.h5

  # Volumes with channel selection (format: name:type:path:channel)
  python -i scripts/visualize_neuroglancer.py \\
    --volumes prediction:image:outputs/pred.h5:0 multi_channel:seg:outputs/multi.h5:2

  # Volumes with channel and resolution (format: name:type:path:channel:resolution)
  python -i scripts/visualize_neuroglancer.py \\
    --volumes prediction:image:outputs/pred.h5:0:5-5-5 label:seg:datasets/label.h5:0:30-6-6

  # Volumes with channel, resolution, and offset (format: name:type:path:channel:resolution:offset)
  python -i scripts/visualize_neuroglancer.py \\
    --volumes prediction:image:outputs/pred.h5:0:5-5-5:100-200-300

  # Mix config with additional volumes
  python -i scripts/visualize_neuroglancer.py \\
    --config tutorials/monai_lucchi.yaml --mode test \\
    --volumes prediction:image:outputs/lucchi_monai_unet/results/test_im_prediction.h5:5-5-5

  # Select from glob pattern by index or filename
  python -i scripts/visualize_neuroglancer.py \\
    --volumes "image:datasets/*.tiff[0]" "label:datasets/*.tiff[train_label]"

  # Custom server settings
  python -i scripts/visualize_neuroglancer.py \\
    --config tutorials/monai_lucchi.yaml \\
    --ip 0.0.0.0 --port 8080 \\
    --resolution 30-6-6

  # 2D images with 2D resolution (automatically padded to 3D)
  python -i scripts/visualize_neuroglancer.py \\
    --image datasets/2d_image.tif \\
    --label datasets/2d_label.tif \\
    --resolution 0.365-0.365

  # Disable intensity scaling
  python -i scripts/visualize_neuroglancer.py \\
    --image datasets/image.tif \\
    --scale -1

  # Custom intensity scaling factor
  python -i scripts/visualize_neuroglancer.py \\
    --image datasets/image.tif \\
    --scale 255

Interactive mode (with -i flag):
  Access loaded variables:
    volumes['train_image'][0]  # numpy array of image data
    viewer                     # Neuroglancer viewer instance
    cfg                        # config object (if --config used)
        """,
    )

    # Input sources (at least one required, but not mutually exclusive)
    parser.add_argument(
        "--config", type=str, help="Path to config YAML file (reads train/test image/label paths)"
    )
    parser.add_argument(
        "--volumes",
        type=str,
        nargs="+",
        help='Volume paths in format "name:type:path[:channel[:resolution[:offset]]]" '
        'where type is "image" or "seg", '
        "channel is an optional channel index (serves as --select for this volume), "
        'resolution is "z-y-x" in nm, and offset is "z-y-x" in voxels. '
        "Type can be omitted for backward compatibility (inferred from name). "
        "Glob patterns supported with selectors: path/*.tiff[0] (index), "
        "path/*.tiff[name] (filename). "
        'Examples: "pred:image:path.h5", "label:seg:data/*.tiff[0]", "multi:image:path.h5:2", '
        '"pred:image:path.h5:0:5-5-5", "pred:image:path.h5:0:5-5-5:100-200-300"',
    )
    parser.add_argument("--image", type=str, help="Path to image volume")
    parser.add_argument("--label", type=str, help="Path to label volume")

    # Server settings
    parser.add_argument(
        "--ip",
        type=str,
        default="localhost",
        help="Server IP address (default: localhost, use 0.0.0.0 for remote access)",
    )
    parser.add_argument("--port", type=int, default=9999, help="Server port (default: 9999)")

    # Volume metadata
    parser.add_argument(
        "--resolution",
        type=str,
        default="30-6-6",
        help='Voxel resolution in nm in zyx order as "z-y-x" (or "y-x" for 2D). '
        "Default: 30-6-6 for EM data. "
        "2D resolution will be padded to 3D with z=1.0",
    )
    parser.add_argument(
        "--offset",
        type=str,
        default="0-0-0",
        help='Volume offset as "z-y-x" or "y-x" for 2D in voxels (default: 0-0-0). '
        "2D offset will be padded to 3D with z=0",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Deprecated (no-op). Loaded volume intensities are kept unchanged.",
    )

    # Display options
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "test", "both"],
        default="train",
        help="Which data to load from config (default: train)",
    )
    parser.add_argument(
        "--select",
        type=str,
        default="0",
        help="Select specific file from glob patterns by index (e.g., '0', '-1') "
        "or filename (e.g., 'volume_001'). "
        "Default: '0' (first file). Use 'all' to load all files.",
    )
    parser.add_argument(
        "--bbox",
        type=str,
        default=None,
        help=(
            "Crop all loaded volumes to a spatial bounding box before visualization. "
            "Format: 'zmin,ymin,xmin,zmax,ymax,xmax' (Python slicing, end-exclusive)."
        ),
    )

    return parser.parse_args()


def parse_bbox_arg(bbox_str: str) -> Tuple[int, int, int, int, int, int]:
    """Parse bbox string 'zmin,ymin,xmin,zmax,ymax,xmax' into integer coordinates."""
    parts = [p.strip() for p in bbox_str.split(",")]
    if len(parts) != 6:
        raise ValueError("bbox must have 6 comma-separated integers: zmin,ymin,xmin,zmax,ymax,xmax")

    try:
        zmin, ymin, xmin, zmax, ymax, xmax = (int(p) for p in parts)
    except ValueError as e:
        raise ValueError(
            "bbox values must be integers in format zmin,ymin,xmin,zmax,ymax,xmax"
        ) from e

    if min(zmin, ymin, xmin, zmax, ymax, xmax) < 0:
        raise ValueError("bbox values must be non-negative")
    if not (zmin < zmax and ymin < ymax and xmin < xmax):
        raise ValueError("bbox min values must be strictly less than max values")

    return zmin, ymin, xmin, zmax, ymax, xmax


def crop_volumes_to_bbox(
    volumes: Dict[str, Tuple],
    bbox: Tuple[int, int, int, int, int, int],
    default_offset: Tuple[int, int, int],
) -> Dict[str, Tuple]:
    """
    Crop all volumes to the same bbox and update voxel offsets accordingly.

    Args:
        volumes: Mapping of volume names to (data, type[, resolution, offset]) tuples.
        bbox: (zmin, ymin, xmin, zmax, ymax, xmax), end-exclusive.
        default_offset: Fallback offset used when a volume has no explicit offset.

    Returns:
        New volume mapping with cropped arrays and adjusted offsets.
    """
    zmin, ymin, xmin, zmax, ymax, xmax = bbox
    cropped_volumes: Dict[str, Tuple] = {}

    print(f"\nApplying bbox crop to all volumes: {bbox} (end-exclusive)")

    for name, vol_data in volumes.items():
        if len(vol_data) == 2:
            data, vol_type = vol_data
            vol_resolution = None
            vol_offset = None
        else:
            data, vol_type, vol_resolution, vol_offset = vol_data

        if data.ndim == 3:
            spatial_shape = data.shape
            crop_slices = (slice(zmin, zmax), slice(ymin, ymax), slice(xmin, xmax))
        elif data.ndim == 4:
            spatial_shape = data.shape[-3:]
            crop_slices = (
                slice(None),
                slice(zmin, zmax),
                slice(ymin, ymax),
                slice(xmin, xmax),
            )
        else:
            raise ValueError(
                f"Volume '{name}' has unsupported ndim={data.ndim} for bbox "
                "cropping (expected 3D or 4D)"
            )

        if not (
            0 <= zmin < zmax <= spatial_shape[0]
            and 0 <= ymin < ymax <= spatial_shape[1]
            and 0 <= xmin < xmax <= spatial_shape[2]
        ):
            raise ValueError(
                f"bbox {bbox} is out of bounds for volume '{name}' spatial shape {spatial_shape}"
            )

        cropped_data = data[crop_slices]
        base_offset = vol_offset if vol_offset is not None else default_offset
        new_offset = (base_offset[0] + zmin, base_offset[1] + ymin, base_offset[2] + xmin)

        print(
            f"  {name}: {data.shape} -> {cropped_data.shape}, offset {base_offset} -> {new_offset}"
        )
        cropped_volumes[name] = (cropped_data, vol_type, vol_resolution, new_offset)

    return cropped_volumes


def load_volumes_from_config(
    config_path: str,
    mode: str = "train",
    prediction_base_name: Optional[str] = None,
    select: str = "0",
) -> Dict[str, Tuple[np.ndarray, str, Optional[Tuple], None]]:
    """
    Load volumes from a config file.

    Args:
        config_path: Path to YAML config file
        mode: Which data to load ('train', 'test', 'both')
        prediction_base_name: Base name for auto-matching prediction files
        select: Select file by index ('0', '-1'), filename ('volume_001'), or 'all' for all files

    Returns:
        Dictionary mapping volume names to (data, type, resolution, offset) tuples
        where type is 'image' or 'segmentation', resolution is from config
        (or None), and offset is None
    """
    cfg = load_config(config_path)

    # Resolve stage defaults so stage-local paths (e.g., train.data.train.*)
    # are materialized into runtime sections (cfg.data.*) for visualization.
    resolve_mode = "train" if mode in ["train", "both"] else "test"
    cfg = resolve_default_profiles(cfg, mode=resolve_mode)
    cfg = resolve_data_paths(cfg)  # Resolve paths and expand globs
    volumes = {}

    # Get resolution from config (explicit zyx convention).
    train_resolution = None
    if mode in ["train", "both"] and hasattr(cfg.data, "train") and cfg.data.train.resolution:
        train_resolution = normalize_resolution_zyx(
            cfg.data.train.resolution,
            context="cfg.data.train.resolution",
        )
        print(f"Using train resolution from config (zyx): {train_resolution} nm")

    test_resolution = None
    if (
        mode in ["test", "both"]
        and hasattr(cfg, "data")
        and hasattr(cfg.data, "test")
        and cfg.data.test.resolution
    ):
        test_resolution = normalize_resolution_zyx(
            cfg.data.test.resolution,
            context="cfg.data.test.resolution",
        )
        print(f"Using test resolution from config (zyx): {test_resolution} nm")

    # Training data
    if mode in ["train", "both"]:
        if hasattr(cfg.data, "train"):
            _store_config_volume_entries(
                volumes,
                getattr(cfg.data.train, "image", None),
                select=select,
                allow_all=True,
                display_label="train images",
                volume_name="train_image",
                volume_type="image",
                resolution=train_resolution,
                cfg=cfg,
                apply_transform=True,
            )
            _store_config_volume_entries(
                volumes,
                getattr(cfg.data.train, "label", None),
                select=select,
                allow_all=True,
                display_label="train labels",
                volume_name="train_label",
                volume_type="segmentation",
                resolution=train_resolution,
            )
            _store_config_volume_entries(
                volumes,
                getattr(cfg.data.train, "mask", None),
                select=select,
                allow_all=True,
                display_label="train masks",
                volume_name="train_mask",
                volume_type="segmentation",
                resolution=train_resolution,
            )

    # Test data
    if mode in ["test", "both"]:
        if hasattr(cfg, "data") and hasattr(cfg.data, "test"):
            test_image_paths = getattr(cfg.data.test, "image", None)
            if isinstance(test_image_paths, list):
                selected_paths = _select_file_paths(test_image_paths, select, allow_all=False)
                test_image_paths = selected_paths[0] if selected_paths else None
            test_image_path = _resolve_prediction_matched_path(
                test_image_paths,
                prediction_base_name,
            )
            if test_image_path:
                _store_config_volume_entries(
                    volumes,
                    test_image_path,
                    select=select,
                    allow_all=False,
                    display_label="test image",
                    volume_name="test_image",
                    volume_type="image",
                    resolution=test_resolution,
                )

            _store_config_volume_entries(
                volumes,
                getattr(cfg.data.test, "label", None),
                select=select,
                allow_all=False,
                display_label="test label",
                volume_name="test_label",
                volume_type="segmentation",
                resolution=test_resolution,
            )
            _store_config_volume_entries(
                volumes,
                getattr(cfg.data.test, "mask", None),
                select=select,
                allow_all=False,
                display_label="test mask",
                volume_name="test_mask",
                volume_type="segmentation",
                resolution=test_resolution,
            )

    if not volumes:
        print(f"WARNING: No volumes found in config for mode='{mode}'")

    return volumes


def resolve_glob_with_selector(path: str, default_selector: str = "0") -> str:
    """
    Resolve a glob pattern with an optional selector.

    Supports:
        - path/*.tiff[0] - select by index (0-based)
        - path/*.tiff[-1] - select last file
        - path/*.tiff[filename.tiff] - select by filename
        - path/*.tiff with default_selector - use default selector

    Args:
        path: Path that may contain glob pattern and selector
        default_selector: Default selector to use if path doesn't have inline selector

    Returns:
        Resolved single file path
    """
    import glob as glob_module
    import re

    # Check for selector pattern: path[selector]
    match = re.match(r"^(.+)\[(.+)\]$", path)
    if match:
        glob_pattern = match.group(1)
        selector = match.group(2)
    elif "*" in path or "?" in path:
        # No inline selector, use default
        glob_pattern = path
        selector = default_selector
    else:
        # Not a glob pattern
        return path

    # Expand glob pattern
    files = sorted(glob_module.glob(glob_pattern))
    if not files:
        raise FileNotFoundError(f"No files match pattern: {glob_pattern}")

    selected = _select_file_paths(files, selector, allow_all=False)
    return selected[0]


def _parse_optional_channel(raw_channel: Optional[str]) -> Optional[int]:
    if raw_channel is None:
        return None
    try:
        return int(raw_channel)
    except ValueError:
        print(f"  Warning: Invalid channel '{raw_channel}', ignoring")
        return None


def _parse_optional_resolution(
    raw_resolution: Optional[str],
    *,
    global_is_2d: bool,
) -> Tuple[Optional[Tuple[float, float, float]], bool]:
    if raw_resolution is None:
        return None, global_is_2d

    try:
        resolution_values = tuple(float(x) for x in raw_resolution.split("-"))
    except ValueError:
        print(f"  Warning: Invalid resolution '{raw_resolution}', ignoring")
        return None, global_is_2d

    if len(resolution_values) == 2:
        padded = (1.0,) + resolution_values
        logger.info("2D resolution detected, padded to 3D: %s", padded)
        return padded, True
    if len(resolution_values) == 3:
        return resolution_values, False

    print(f"  Warning: Invalid resolution '{raw_resolution}' (expected 2 or 3 values), ignoring")
    return None, global_is_2d


def _parse_optional_offset(raw_offset: Optional[str]) -> Optional[Tuple[int, int, int]]:
    if raw_offset is None:
        return None

    try:
        offset_values = tuple(int(x) for x in raw_offset.split("-"))
    except ValueError:
        print(f"  Warning: Invalid offset '{raw_offset}', ignoring")
        return None

    if len(offset_values) == 2:
        padded = (0,) + offset_values
        logger.info("2D offset detected, padded to 3D: %s", padded)
        return padded
    if len(offset_values) == 3:
        return offset_values

    print(f"  Warning: Invalid offset '{raw_offset}' (expected 2 or 3 values), ignoring")
    return None


def _parse_volume_spec(
    spec: str,
    *,
    global_is_2d: bool,
) -> Tuple[
    str,
    str,
    Optional[str],
    Optional[int],
    Optional[Tuple[float, float, float]],
    Optional[Tuple[int, int, int]],
    bool,
]:
    """Parse one --volumes spec into normalized fields."""
    parts = spec.split(":")
    has_explicit_type = len(parts) >= 3 and parts[1] in ["image", "img", "seg", "segmentation"]

    optional_parts: List[str] = []
    if len(parts) == 1:
        name = Path(parts[0]).stem
        path = parts[0]
        vol_type = None
    elif len(parts) == 2 and not has_explicit_type:
        name, path = parts
        vol_type = None
    elif has_explicit_type:
        name = parts[0]
        vol_type = "segmentation" if parts[1] in ["seg", "segmentation"] else "image"
        path = parts[2]
        optional_parts = parts[3:]
    else:
        # Legacy format: name:path[:channel[:resolution[:offset]]]
        name = parts[0]
        path = parts[1]
        vol_type = None
        optional_parts = parts[2:]

    raw_channel = optional_parts[0] if len(optional_parts) >= 1 else None
    raw_resolution = optional_parts[1] if len(optional_parts) >= 2 else None
    raw_offset = optional_parts[2] if len(optional_parts) >= 3 else None

    channel = _parse_optional_channel(raw_channel)
    resolution, is_2d_resolution = _parse_optional_resolution(
        raw_resolution,
        global_is_2d=global_is_2d,
    )
    offset = _parse_optional_offset(raw_offset)

    return name, path, vol_type, channel, resolution, offset, is_2d_resolution


def load_volumes_from_paths(
    volume_specs: List[str],
    select: str = "0",
    scale: float = 1.0,
    global_resolution: Optional[Tuple[Tuple, bool]] = None,
) -> Dict[str, Tuple[np.ndarray, str, Optional[Tuple], Optional[Tuple]]]:
    """
    Load volumes from path specifications.

    Args:
        volume_specs: List of volume specifications in format:
            - "path" - just path (type inferred from name)
            - "name:path" - name and path (type inferred from name)
            - "name:type:path" - name, type (image/seg), and path
            - "name:type:path:channel" - with channel index (e.g., "0", "2")
            - "name:type:path:channel:resolution" - with channel and
              resolution (e.g., "0:5-5-5")
            - "name:type:path:channel:resolution:offset" - with channel,
              resolution, and offset (e.g., "0:5-5-5:100-200-300")

        Paths can include glob patterns with selectors:
            - "name:type:path/*.tiff[0]" - select first file
            - "name:type:path/*.tiff[-1]" - select last file
            - "name:type:path/*.tiff[filename]" - select by name
        select: Default selector for glob patterns without inline selector
        scale: Intensity scaling factor (< 0 disables scaling)
        global_resolution: Tuple of (resolution_tuple, is_2d_flag) from --resolution flag
                          Used for 2D detection when per-volume resolution not specified

    Returns:
        Dictionary mapping volume names to (data, type, resolution, offset) tuples
        where resolution and offset can be None (use defaults)
    """
    volumes = {}

    # Extract global resolution info
    global_is_2d = False
    if global_resolution is not None:
        _, global_is_2d = global_resolution

    for spec in volume_specs:
        (
            name,
            path,
            vol_type,
            channel,
            resolution,
            offset,
            is_2d_resolution,
        ) = _parse_volume_spec(spec, global_is_2d=global_is_2d)

        # Resolve glob patterns with selectors
        resolved_path = resolve_glob_with_selector(path, select)

        print(f"Loading {name}: {resolved_path}")
        if channel is not None:
            print(f"  Channel selection: {channel}")
        if resolution:
            print(f"  Custom resolution: {resolution}")
        if offset:
            print(f"  Custom offset: {offset}")

        data = read_volume(resolved_path)
        print(f"  Loaded data shape: {data.shape}, dtype: {data.dtype}")

        # Handle channel selection for multi-channel data
        if channel is not None:
            if data.ndim == 4:  # (C, Z, Y, X) - 3D multi-channel
                if channel >= data.shape[0]:
                    print(
                        f"  Warning: Channel {channel} out of range "
                        f"(max: {data.shape[0]-1}), using channel 0"
                    )
                    data = data[0]
                else:
                    print(f"  Selected channel {channel} from 4D shape {data.shape}")
                    data = data[channel]
            elif data.ndim == 3:
                # Could be (C, H, W) for 2D multi-channel OR (Z, H, W) for 3D single-channel
                if is_2d_resolution:
                    # Treat as (C, H, W) - 2D multi-channel
                    if channel >= data.shape[0]:
                        print(
                            f"  Warning: Channel {channel} out of range "
                            f"(max: {data.shape[0]-1}), using channel 0"
                        )
                        data = data[0]
                    else:
                        print(
                            f"  Selected channel {channel} from 2D multi-channel shape {data.shape}"
                        )
                        data = data[channel]
                else:
                    # Treat as (Z, H, W) - 3D single-channel, ignore channel selection
                    print(
                        f"  Warning: Channel {channel} specified but data is 3D "
                        f"{data.shape} (single-channel), ignoring"
                    )
            else:
                print(f"  Warning: Unexpected data shape {data.shape}, ignoring channel selection")
        else:
            # Squeeze to remove singleton dimensions if no channel specified
            data = data.squeeze()

        # Convert 2D to 3D if needed
        if data.ndim == 2:
            if is_2d_resolution:
                # For 2D data, add singleton z dimension: (H, W) -> (1, H, W)
                data = data[None, :, :]
                logger.info("Converted 2D to 3D (single-channel): %s", data.shape)
            else:
                # Shouldn't happen, but handle it
                data = data[None, :, :]
                logger.info("Converted 2D to 3D: %s", data.shape)
        elif data.ndim == 3 and is_2d_resolution and channel is None:
            # Multi-channel 2D data (C, H, W) that wasn't explicitly channel-selected
            # Insert singleton dimension in middle: (C, H, W) -> (C, 1, H, W)
            data = data[:, None, :, :]
            logger.info("Converted 2D multi-channel to 3D: %s", data.shape)

        # Infer type if not explicitly specified
        if vol_type is None:
            name_lower = name.lower()
            if any(keyword in name_lower for keyword in ["label", "seg", "gt", "pred", "mask"]):
                vol_type = "segmentation"
            else:
                vol_type = "image"

        # Apply default intensity scaling (only for images, not segmentations)
        is_image = vol_type == "image"
        data = apply_default_scaling(data, scale, is_image=is_image)

        print(f"  Volume type: {vol_type}, final shape: {data.shape}")
        volumes[name] = (data, vol_type, resolution, offset)

    return volumes


def create_neuroglancer_layer(
    data: np.ndarray,
    resolution: Tuple[float, float, float],
    offset: Tuple[int, int, int] = (0, 0, 0),
    volume_type: str = "image",
) -> "neuroglancer.LocalVolume":
    """
    Create a Neuroglancer layer from volume data.

    Args:
        data: Volume data (3D or 4D numpy array)
        resolution: Voxel resolution in nm as (z, y, x)
        offset: Volume offset as (z, y, x)
        volume_type: 'image' or 'segmentation'

    Returns:
        Neuroglancer LocalVolume layer
    """
    # Handle 4D data (C, Z, Y, X) -> use first channel
    if data.ndim == 4:
        print(f"  4D volume detected: {data.shape}, using first channel")
        data = data[0]

    # Ensure 3D
    if data.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {data.shape}")

    # Create coordinate space
    coord_space = neuroglancer.CoordinateSpace(
        names=["z", "y", "x"], units=["nm", "nm", "nm"], scales=resolution
    )

    print(f"  Shape: {data.shape}, Type: {volume_type}, Resolution: {resolution}")

    return neuroglancer.LocalVolume(
        data, dimensions=coord_space, volume_type=volume_type, voxel_offset=offset
    )


def visualize_volumes(
    volumes: Dict[str, Tuple],
    ip: str = "localhost",
    port: int = 9999,
    resolution: Tuple[float, float, float] = (30, 6, 6),
    offset: Tuple[int, int, int] = (0, 0, 0),
) -> "neuroglancer.Viewer":
    """
    Visualize volumes with Neuroglancer.

    Args:
        volumes: Dictionary mapping names to (data, type, resolution, offset) tuples
                 where resolution and offset can be None (use defaults)
        ip: Server IP address
        port: Server port
        resolution: Default voxel resolution in nm as (z, y, x)
        offset: Default volume offset as (z, y, x)

    Returns:
        Neuroglancer viewer instance
    """
    if not volumes:
        print("ERROR: No volumes to visualize")
        return None

    # Set up Neuroglancer server
    print(f"\nStarting Neuroglancer server on {ip}:{port}")
    neuroglancer.set_server_bind_address(bind_address=ip, bind_port=port)
    viewer = neuroglancer.Viewer()

    # Add all volumes as layers
    with viewer.txn() as state:
        for name, vol_data in volumes.items():
            print(f"\nAdding layer: {name}")

            # Handle both old format (data, type) and new format (data, type, resolution, offset)
            if len(vol_data) == 2:
                data, vol_type = vol_data
                vol_resolution = resolution  # Use default
                vol_offset = offset  # Use default
            else:
                data, vol_type, vol_resolution, vol_offset = vol_data
                # Use volume-specific values if provided, otherwise use defaults
                vol_resolution = vol_resolution if vol_resolution is not None else resolution
                vol_offset = vol_offset if vol_offset is not None else offset

            layer = create_neuroglancer_layer(data, vol_resolution, vol_offset, vol_type)
            state.layers.append(name=name, layer=layer)

    # Print viewer URL and instructions
    print("\n" + "=" * 70)
    print("Neuroglancer viewer ready!")
    print("=" * 70)
    print("\nOpen this URL in your browser:")
    print(f"  {viewer}")
    print(f"\nServer: {ip}:{port}")
    print(f"Volumes: {list(volumes.keys())}")
    print("\n" + "=" * 70)
    print("Interactive Python session - available variables and functions:")
    print("  viewer      - Neuroglancer viewer instance")
    print("  volumes     - Dictionary of loaded volumes")
    print("  cfg         - Config object (if --config used)")
    print(
        "  add_layer() - Add new layer: add_layer("
        "'name', file_path='path.h5', res=[5,5,5], tt='image')"
    )
    print("  ngLayer()   - Create layer: ngLayer(data, [5,5,5], tt='image')")
    print("\nExamples:")
    print("  volumes['name'][0]  # Access numpy array")
    print("  add_layer('pred', file_path='outputs/pred.h5', res=[5,5,5], tt='seg')")
    print("\nExit with: exit() or Ctrl+D")
    print("=" * 70 + "\n")

    return viewer


def main():
    """Main entry point."""
    args = parse_args()

    # Check for neuroglancer (after argparse so --help works without it)
    try:
        import neuroglancer as ng

        global neuroglancer
        neuroglancer = ng
    except ImportError:
        print("\nERROR: neuroglancer not installed.")
        print("Install with: pip install neuroglancer")
        print("Or: pip install neuroglancer-python\n")
        sys.exit(1)

    # Validate that at least one input source is provided
    # Empty strings count as no input
    has_config = bool(args.config)
    has_image = bool(args.image and args.image.strip())
    has_label = bool(args.label and args.label.strip())
    has_volumes = bool(args.volumes)

    if not any([has_config, has_image, has_label, has_volumes]):
        print("ERROR: At least one input source is required:")
        print("  --config CONFIG      Load from config file")
        print("  --image IMG          Load image volume")
        print("  --label LBL          Load label volume")
        print("  --volumes VOL...     Load multiple volumes")
        print(
            "\nExample: python scripts/visualize_neuroglancer.py --image img.tif --label label.h5"
        )
        sys.exit(1)

    # Load volumes based on input method (can combine multiple sources!)
    # Use global to make variables accessible in interactive mode
    global volumes, viewer, cfg
    volumes = {}
    cfg = None

    # Extract prediction base name from --volumes if provided (for auto-matching test_image)
    prediction_base_name = None
    if args.volumes:
        for spec in args.volumes:
            _, path, _, _, _, _, _ = _parse_volume_spec(spec, global_is_2d=False)
            # Check if this is a prediction file
            path_obj = Path(path)
            if "_prediction" in path_obj.stem:
                # Extract base name by removing "_prediction" and extension
                prediction_base_name = path_obj.stem.replace("_prediction", "")
                print(f"Detected prediction file: {path_obj.name}")
                print(f"   Extracted base name for auto-matching: {prediction_base_name}")
                break

    # Parse resolution early - needed for 2D detection in volume loading
    # Parse resolution and offset from string format "z-y-x" or "y-x" (for 2D)
    is_global_2d = False
    try:
        resolution = tuple(float(x) for x in args.resolution.split("-"))
        # Detect 2D resolution BEFORE padding
        if len(resolution) == 2:
            is_global_2d = True
            resolution = (1.0,) + resolution  # [y, x] -> [1, y, x]
            logger.info("2D resolution detected, padded to 3D: %s", resolution)
        elif len(resolution) != 3:
            raise ValueError(f"Resolution must have 2 or 3 components, got {len(resolution)}")
    except (ValueError, AttributeError):
        print(
            f"ERROR: Invalid resolution format '{args.resolution}'. "
            "Expected format: 'z-y-x' or 'y-x' for 2D "
            "(e.g., '30-6-6' or '0.365-0.365')"
        )
        sys.exit(1)

    # Load from config first (if provided)
    if args.config:
        cfg = load_config(args.config)  # Store config for interactive access
        volumes.update(
            load_volumes_from_config(
                args.config,
                args.mode,
                prediction_base_name=prediction_base_name,
                select=args.select,
            )
        )

    # Add image/label (if provided and not empty strings)
    if args.image and args.image.strip():
        print(f"Loading image: {args.image}")
        data = _load_config_volume_array(args.image, context="image")
        # Apply default intensity scaling
        data = apply_default_scaling(data, args.scale, is_image=True)
        volumes["image"] = (data, "image", None, None)
    if args.label and args.label.strip():
        print(f"Loading label: {args.label}")
        data = _load_config_volume_array(args.label, context="label")
        # No scaling for labels (they are segmentation)
        volumes["label"] = (data, "segmentation", None, None)

    # Add additional volumes (if provided) - these can override config volumes
    if args.volumes:
        volumes.update(
            load_volumes_from_paths(
                args.volumes,
                select=args.select,
                scale=args.scale,
                global_resolution=(resolution, is_global_2d),
            )
        )

    if not volumes:
        print("ERROR: No volumes loaded. Check your input paths.")
        sys.exit(1)

    try:
        offset = tuple(int(x) for x in args.offset.split("-"))
        # Handle 2D offset: pad with z=0 to make it 3D
        if len(offset) == 2:
            offset = (0,) + offset  # [y, x] -> [0, y, x]
            logger.info("2D offset detected, padded to 3D: %s", offset)
        elif len(offset) != 3:
            raise ValueError(f"Offset must have 2 or 3 components, got {len(offset)}")
    except (ValueError, AttributeError):
        print(
            f"ERROR: Invalid offset format '{args.offset}'. Expected format: "
            "'z-y-x' or 'y-x' for 2D (e.g., '0-0-0' or '0-0')"
        )
        sys.exit(1)

    if args.bbox:
        try:
            bbox = parse_bbox_arg(args.bbox)
            volumes = crop_volumes_to_bbox(volumes, bbox, default_offset=offset)
        except ValueError as e:
            print(f"ERROR: Invalid bbox '{args.bbox}': {e}")
            sys.exit(1)

    # Start visualization (returns viewer for interactive access)
    viewer = visualize_volumes(
        volumes=volumes,
        ip=args.ip,
        port=args.port,
        resolution=resolution,
        offset=offset,
    )

    # Helper function for interactive mode: create neuroglancer layer from data and resolution
    def ngLayer(data, res, oo=[0, 0, 0], tt="segmentation"):
        """
        Create a Neuroglancer layer from volume data.

        Args:
            data: Volume data (3D or 4D numpy array)
            res: Resolution as list/tuple [z, y, x] in nm, or CoordinateSpace object
            oo: Offset as list/tuple [z, y, x] (default: [0, 0, 0])
            tt: Volume type 'image' or 'segmentation' (default: 'segmentation')

        Returns:
            Neuroglancer LocalVolume layer
        """
        # Handle 4D data (C, Z, Y, X) -> use first channel
        if data.ndim == 4:
            data = data[0]

        # Ensure 3D
        if data.ndim != 3:
            raise ValueError(f"Expected 3D volume, got shape {data.shape}")

        # Create coordinate space from resolution if it's a list/tuple
        if isinstance(res, (list, tuple)):
            coord_space = neuroglancer.CoordinateSpace(
                names=["z", "y", "x"], units=["nm", "nm", "nm"], scales=res
            )
        else:
            # Assume it's already a CoordinateSpace object
            coord_space = res

        return neuroglancer.LocalVolume(
            data, dimensions=coord_space, volume_type=tt, voxel_offset=oo
        )

    def add_layer(name, file_path=None, data=None, res=None, oo=[0, 0, 0], tt="image"):
        """
        Add a layer to the viewer (convenience function for interactive mode).

        Args:
            name: Display name for the layer
            file_path: Path to volume file (alternative to data parameter)
            data: Volume data as numpy array (alternative to file_path parameter)
            res: Resolution as list/tuple [z, y, x] in nm (default: uses args.resolution)
            oo: Offset as list/tuple [z, y, x] (default: [0, 0, 0])
            tt: Volume type 'image' or 'segmentation' (default: 'image')

        Example usage:
            # Load from file with custom resolution
            add_layer('prediction', file_path='outputs/pred.h5', res=[5, 5, 5], tt='seg')

            # Load from existing numpy array
            add_layer('my_volume', data=volumes['test_image'][0], res=[30, 6, 6], tt='image')
        """
        # Load data from file if file_path is provided
        if file_path is not None:
            if data is not None:
                print("Warning: Both file_path and data provided, using file_path")
            print(f"Loading {name} from: {file_path}")
            data = read_volume(file_path)

            # Convert 2D to 3D if needed
            if data.ndim == 2:
                data = data[None, :, :]
                logger.info("Converted 2D to 3D: %s", data.shape)
        elif data is None:
            raise ValueError("Either file_path or data must be provided")

        # Use default resolution if not provided
        if res is None:
            res = list(resolution)  # Use parsed resolution, not args.resolution
            print(f"  Using default resolution: {res}")

        # Create and add layer
        print(f"  Adding layer '{name}': shape={data.shape}, type={tt}, resolution={res}")
        layer = ngLayer(data, res, oo, tt)

        with viewer.txn() as s:
            s.layers.append(name=name, layer=layer)

        print(f"  Layer '{name}' added successfully")

        # Store in volumes dict for future reference
        volumes[name] = (data, "segmentation" if tt == "seg" else tt, tuple(res), tuple(oo))

    # Make helper functions available in interactive mode
    globals()["ngLayer"] = ngLayer
    globals()["add_layer"] = add_layer

    # Return viewer for interactive mode
    return viewer


if __name__ == "__main__":
    main()

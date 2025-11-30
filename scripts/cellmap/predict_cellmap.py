#!/usr/bin/env python
"""
Inference on CellMap test crops using trained PyTC model.

Uses:
- CellMap's TEST_CROPS for official metadata
- MONAI's SlidingWindowInferer for efficient inference
- PyTC's trained models

Usage:
    python scripts/cellmap/predict_cellmap.py \
        --checkpoint outputs/cellmap/checkpoints/mednext-epoch=100-val_dice=0.850.ckpt \
        --config scripts/cellmap/configs/mednext_cos7.py \
        --output outputs/cellmap/predictions

    # Predict specific crops only
    python scripts/cellmap/predict_cellmap.py \
        --checkpoint best.ckpt \
        --config config.py \
        --crops 234,236,237

Requirements:
    pip install cellmap-data cellmap-segmentation-challenge
"""

import os
import sys
from pathlib import Path

PYTC_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PYTC_ROOT))

import torch
import zarr
import numpy as np
from tqdm import tqdm
from monai.inferers import SlidingWindowInferer

# CellMap utilities
from cellmap_segmentation_challenge.utils import TEST_CROPS, load_safe_config
from cellmap_segmentation_challenge.evaluate import match_crop_space

# PyTC models
from connectomics.models import build_model
from omegaconf import OmegaConf


def find_scale_level(zarr_path, target_resolution):
    """Find the scale level that best matches target resolution."""
    store = zarr.open(zarr_path, mode='r')

    # Read OME-NGFF multiscale metadata
    multiscale_meta = store.attrs.get('multiscales', [{}])[0]
    datasets_meta = multiscale_meta.get('datasets', [])

    if not datasets_meta:
        # Fallback: use s2 (typically 8nm)
        return 2

    # Find closest scale to target resolution
    best_scale = 0
    min_diff = float('inf')

    for i, ds_meta in enumerate(datasets_meta):
        transforms = ds_meta.get('coordinateTransformations', [{}])
        scale = transforms[0].get('scale', [1, 1, 1]) if transforms else [1, 1, 1]
        # scale is [z, y, x] in nm
        avg_resolution = np.mean(scale)
        diff = abs(avg_resolution - np.mean(target_resolution))

        if diff < min_diff:
            min_diff = diff
            best_scale = i

    return best_scale


def predict_cellmap(checkpoint_path, config_path, output_dir, crop_filter=None):
    """
    Run inference on all test crops.

    Args:
        checkpoint_path: Path to trained model checkpoint
        config_path: Path to training config file
        output_dir: Directory to save predictions
        crop_filter: List of crop IDs to predict (None = all crops)
    """

    # Load config
    print(f"Loading config from: {config_path}")
    config = load_safe_config(config_path)
    classes = getattr(config, 'classes', ['nuc', 'mito', 'er'])
    model_name = getattr(config, 'model_name', 'mednext')
    target_resolution = getattr(config, 'input_array_info', {}).get('scale', (8, 8, 8))

    print(f"Prediction configuration:")
    print(f"  Model: {model_name}")
    print(f"  Classes: {classes}")
    print(f"  Target resolution: {target_resolution} nm")

    # Build model
    print(f"Building model: {model_name}")
    model_config = OmegaConf.create({
        'model': {
            'architecture': model_name,
            'in_channels': 1,
            'out_channels': len(classes),
            'mednext_size': getattr(config, 'mednext_size', 'B'),
            'mednext_kernel_size': getattr(config, 'mednext_kernel_size', 5),
            'deep_supervision': getattr(config, 'deep_supervision', True),
        }
    })
    model = build_model(model_config)

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Handle Lightning checkpoint format
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        # Remove 'model.' prefix from keys if present
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Move to GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    print(f"Using device: {device}")

    # Setup sliding window inferer (MONAI)
    inferer = SlidingWindowInferer(
        roi_size=(128, 128, 128),
        sw_batch_size=4,
        overlap=0.5,
        mode='gaussian',
        device=torch.device(device),
    )

    # Filter test crops if specified
    if crop_filter:
        crop_ids = [int(c) for c in crop_filter]
        test_crops = [crop for crop in TEST_CROPS if crop.id in crop_ids]
        print(f"Predicting on {len(test_crops)} crops: {crop_ids}")
    else:
        test_crops = TEST_CROPS
        print(f"Predicting on all {len(test_crops)} test crops")

    # Group crops by dataset for efficiency
    crops_by_dataset = {}
    for crop in test_crops:
        if crop.dataset not in crops_by_dataset:
            crops_by_dataset[crop.dataset] = []
        crops_by_dataset[crop.dataset].append(crop)

    # Predict on all test crops
    os.makedirs(output_dir, exist_ok=True)

    for dataset, dataset_crops in crops_by_dataset.items():
        print(f"\nProcessing dataset: {dataset}")

        zarr_path = f"/projects/weilab/dataset/cellmap/{dataset}/{dataset}.zarr"

        # Find appropriate scale level for target resolution
        em_path = f"{zarr_path}/recon-1/em/fibsem-uint8"
        scale_level = find_scale_level(em_path, target_resolution)
        print(f"  Using scale level: s{scale_level} (target resolution: {target_resolution} nm)")

        # Load EM data once for all crops in this dataset
        try:
            raw_array = zarr.open(f"{em_path}/s{scale_level}", mode='r')
        except Exception as e:
            print(f"  Error loading EM data: {e}")
            print(f"  Skipping dataset {dataset}")
            continue

        for crop in tqdm(dataset_crops, desc=f"  Crops in {dataset}"):
            crop_id = crop.id
            class_label = crop.class_label

            # Skip if this class is not in our training classes
            if class_label not in classes:
                continue

            # Extract crop region from full volume
            # Note: This is simplified - in production, use crop.translation and crop.shape
            # to extract exact region
            crop_output_dir = f"{output_dir}/{dataset}/crop{crop_id}"
            os.makedirs(crop_output_dir, exist_ok=True)

            # Load a reasonable-sized region (simplified)
            # In production, use crop metadata to extract exact region
            try:
                # Get raw data shape
                raw_shape = raw_array.shape

                # Simple extraction (center crop for demo)
                # TODO: Use actual crop.translation and crop.shape for exact extraction
                d, h, w = min(256, raw_shape[0]), min(256, raw_shape[1]), min(256, raw_shape[2])
                raw_volume = raw_array[:d, :h, :w]

                # Normalize and convert to tensor
                raw_volume = np.array(raw_volume).astype(np.float32) / 255.0
                raw_tensor = torch.from_numpy(raw_volume[None, None, ...]).to(device)  # (1, 1, D, H, W)

                # Run inference
                with torch.no_grad():
                    predictions = inferer(raw_tensor, model)
                    predictions = torch.sigmoid(predictions).cpu().numpy()[0]  # (C, D, H, W)

                # Save predictions for each class
                for i, cls in enumerate(classes):
                    pred_array = (predictions[i] > 0.5).astype(np.uint8)

                    # Save as Zarr (CellMap format)
                    zarr_out_path = f"{crop_output_dir}/{cls}"
                    os.makedirs(zarr_out_path, exist_ok=True)

                    zarr_out = zarr.open(
                        f"{zarr_out_path}/s0",
                        mode='w',
                        shape=pred_array.shape,
                        dtype='uint8',
                        chunks=(64, 64, 64),
                        compressor=zarr.Blosc(cname='zstd', clevel=5),
                    )
                    zarr_out[:] = pred_array

                    # Add metadata
                    zarr_out.attrs['voxel_size'] = crop.voxel_size
                    zarr_out.attrs['translation'] = crop.translation
                    zarr_out.attrs['shape'] = crop.shape

            except Exception as e:
                print(f"    Error processing crop {crop_id}: {e}")
                continue

    print(f"\nPredictions saved to: {output_dir}")
    print(f"Next step: python scripts/cellmap/submit_cellmap.py --predictions {output_dir}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run inference on CellMap test crops')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--output', default='outputs/cellmap/predictions', help='Output directory')
    parser.add_argument('--crops', type=str, help='Comma-separated crop IDs to predict (default: all)')
    args = parser.parse_args()

    crop_filter = args.crops.split(',') if args.crops else None

    predict_cellmap(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        output_dir=args.output,
        crop_filter=crop_filter,
    )

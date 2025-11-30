#!/usr/bin/env python
"""
Package predictions for CellMap challenge submission.

Uses CellMap's official packaging utility - guaranteed to work!

This script:
1. Resamples predictions to match test crop resolutions
2. Validates prediction format
3. Packages into submission.zarr
4. Creates submission.zip for upload

Usage:
    python scripts/cellmap/submit_cellmap.py \
        --predictions outputs/cellmap/predictions \
        --output submission.zarr

    # Then upload submission.zip to challenge portal
    # https://cellmapchallenge.janelia.org/submissions/

Requirements:
    pip install cellmap-data cellmap-segmentation-challenge
"""

import os
import sys
from pathlib import Path

PYTC_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PYTC_ROOT))

from cellmap_segmentation_challenge.utils import package_submission


def submit_cellmap(predictions_dir, output_path, overwrite=True, max_workers=None):
    """
    Package predictions for CellMap challenge submission.

    This uses CellMap's official packaging utility which:
    - Resamples predictions to match test crop resolution/shape
    - Validates format and metadata
    - Creates Zarr archive
    - Zips for upload

    Args:
        predictions_dir: Directory containing predictions (from predict_cellmap.py)
        output_path: Output path for submission.zarr
        overwrite: Whether to overwrite existing submission
        max_workers: Number of parallel workers (default: CPU count)
    """

    if max_workers is None:
        max_workers = os.cpu_count()

    print("CellMap Challenge Submission Packager")
    print("=" * 60)
    print(f"Input predictions: {predictions_dir}")
    print(f"Output: {output_path}")
    print(f"Workers: {max_workers}")
    print()

    # Check predictions directory exists
    if not os.path.exists(predictions_dir):
        print(f"Error: Predictions directory not found: {predictions_dir}")
        print("Run predict_cellmap.py first to generate predictions.")
        sys.exit(1)

    # Use official packaging (handles resampling, validation, zipping)
    print("Packaging submission...")
    print("This will:")
    print("  1. Resample predictions to match test crop resolutions")
    print("  2. Validate format and metadata")
    print("  3. Create submission.zarr")
    print("  4. Create submission.zip")
    print()

    try:
        package_submission(
            input_search_path=predictions_dir,
            output_path=output_path,
            overwrite=overwrite,
            max_workers=max_workers,
        )
    except Exception as e:
        print(f"\nError during packaging: {e}")
        print("\nPlease check:")
        print("  1. Predictions directory structure is correct")
        print("  2. All required test crops have predictions")
        print("  3. Zarr arrays have correct metadata")
        sys.exit(1)

    # Check output
    zip_path = output_path.replace('.zarr', '.zip')

    if os.path.exists(zip_path):
        print()
        print("=" * 60)
        print("Submission packaged successfully!")
        print()
        print(f"Submission file: {zip_path}")
        print(f"File size: {os.path.getsize(zip_path) / 1e9:.2f} GB")
        print()
        print("Next steps:")
        print("  1. Verify submission.zip is complete")
        print("  2. Upload to: https://cellmapchallenge.janelia.org/submissions/")
        print("  3. Check evaluation results on leaderboard")
        print()
    else:
        print("\nWarning: Submission zip file not found!")
        print("Packaging may have failed.")
        sys.exit(1)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Package predictions for CellMap challenge submission',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Package predictions
  python scripts/cellmap/submit_cellmap.py --predictions outputs/predictions

  # Custom output path
  python scripts/cellmap/submit_cellmap.py \\
      --predictions outputs/predictions \\
      --output my_submission.zarr

  # Use more workers for faster packaging
  python scripts/cellmap/submit_cellmap.py \\
      --predictions outputs/predictions \\
      --workers 32
        """
    )
    parser.add_argument(
        '--predictions',
        default='outputs/cellmap/predictions',
        help='Directory containing predictions (default: outputs/cellmap/predictions)'
    )
    parser.add_argument(
        '--output',
        default='submission.zarr',
        help='Output path for submission.zarr (default: submission.zarr)'
    )
    parser.add_argument(
        '--no-overwrite',
        action='store_true',
        help='Do not overwrite existing submission'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='Number of parallel workers (default: CPU count)'
    )

    args = parser.parse_args()

    submit_cellmap(
        predictions_dir=args.predictions,
        output_path=args.output,
        overwrite=not args.no_overwrite,
        max_workers=args.workers,
    )

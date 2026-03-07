#!/usr/bin/env python3
"""
Download datasets for PyTorch Connectomics tutorials.

Usage:
    python scripts/download_data.py lucchi++
    python scripts/download_data.py --list
    python scripts/download_data.py all
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from connectomics.data.download import DATASETS, download_dataset, list_datasets


def main():
    parser = argparse.ArgumentParser(
        description="Download datasets for PyTorch Connectomics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/download_data.py lucchi++
  python scripts/download_data.py snemi mitoem
  python scripts/download_data.py all
  python scripts/download_data.py --list
        """
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        help="Dataset name(s) to download, or 'all' for all datasets"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available datasets"
    )
    parser.add_argument(
        "--output", "-o",
        default=".",
        help="Base directory (default: current dir, datasets saved to <output>/datasets/)"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force re-download even if dataset exists"
    )

    args = parser.parse_args()

    if args.list:
        list_datasets()
        return 0

    if not args.datasets:
        parser.print_help()
        return 1

    # Expand 'all' to all datasets
    datasets = args.datasets
    if "all" in datasets:
        datasets = list(DATASETS.keys())

    # Download each dataset
    success = True
    for name in datasets:
        print(f"\n{'='*60}")
        if not download_dataset(name, Path(args.output), args.force):
            success = False

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

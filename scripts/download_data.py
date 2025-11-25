#!/usr/bin/env python3
"""
Download datasets for PyTorch Connectomics tutorials.

Usage:
    python scripts/download_data.py lucchi++
    python scripts/download_data.py --list
    python scripts/download_data.py all
"""

import argparse
import os
import sys
import zipfile
import tarfile
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import URLError


# Dataset registry with download information
DATASETS = {
    "lucchi++": {
        "url": "https://huggingface.co/datasets/pytc/tutorial/resolve/main/lucchi%2B%2B.zip",
        "filename": "lucchi++.zip",
        "archive_dir": "lucchi++",  # Directory name inside the archive
        "extract_dir": "lucchi++",   # Target directory name (lowercase)
        "description": "Lucchi++ mitochondria segmentation (EM, 5nm isotropic)",
        "size": "~50 MB",
    },
    "snemi": {
        "url": "https://huggingface.co/datasets/pytc/tutorial/resolve/main/SNEMI3D.zip",
        "filename": "SNEMI3D.zip",
        "archive_dir": "SNEMI3D",
        "extract_dir": "snemi",
        "description": "SNEMI3D neuron segmentation challenge dataset",
        "size": "~200 MB",
    },
    "mitoem": {
        "url": "https://huggingface.co/datasets/pytc/tutorial/resolve/main/MitoEM.zip",
        "filename": "MitoEM.zip",
        "archive_dir": "MitoEM",
        "extract_dir": "mitoem",
        "description": "MitoEM large-scale mitochondria 3D segmentation",
        "size": "~2 GB",
    },
    "cremi": {
        "url": "https://huggingface.co/datasets/pytc/tutorial/resolve/main/CREMI.zip",
        "filename": "CREMI.zip",
        "archive_dir": "CREMI",
        "extract_dir": "cremi",
        "description": "CREMI synaptic cleft detection challenge",
        "size": "~500 MB",
    },
}


def progress_hook(count, block_size, total_size):
    """Display download progress."""
    percent = min(100, count * block_size * 100 // total_size)
    bar_length = 40
    filled = int(bar_length * percent // 100)
    bar = "=" * filled + "-" * (bar_length - filled)
    sys.stdout.write(f"\r[{bar}] {percent}%")
    sys.stdout.flush()


def download_dataset(name: str, output_dir: str = "datasets", force: bool = False) -> bool:
    """
    Download and extract a dataset.

    Args:
        name: Dataset name (must be in DATASETS registry)
        output_dir: Directory to save datasets
        force: If True, re-download even if exists

    Returns:
        True if successful, False otherwise
    """
    if name not in DATASETS:
        print(f"Error: Unknown dataset '{name}'")
        print(f"Available datasets: {', '.join(DATASETS.keys())}")
        return False

    dataset = DATASETS[name]
    output_path = Path(output_dir)
    extract_path = output_path / dataset["extract_dir"]

    # Check if already exists
    if extract_path.exists() and not force:
        print(f"Dataset '{name}' already exists at {extract_path}")
        print("Use --force to re-download")
        return True

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Download
    zip_path = output_path / dataset["filename"]
    print(f"Downloading {name} ({dataset['size']})...")
    print(f"  URL: {dataset['url']}")
    print(f"  Destination: {zip_path}")

    try:
        urlretrieve(dataset["url"], zip_path, reporthook=progress_hook)
        print()  # New line after progress bar
    except URLError as e:
        print(f"\nError downloading: {e}")
        return False
    except KeyboardInterrupt:
        print("\nDownload cancelled")
        if zip_path.exists():
            zip_path.unlink()
        return False

    # Extract
    print(f"Extracting to {output_path}...")
    try:
        if zip_path.suffix == ".zip":
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(output_path)
        elif zip_path.suffix in [".tar", ".gz", ".tgz"]:
            with tarfile.open(zip_path, 'r:*') as tf:
                tf.extractall(output_path)
        else:
            print(f"Unknown archive format: {zip_path.suffix}")
            return False
    except Exception as e:
        print(f"Error extracting: {e}")
        return False

    # Rename to lowercase directory name if needed
    archive_dir = dataset.get("archive_dir")
    if archive_dir and archive_dir != dataset["extract_dir"]:
        archive_path = output_path / archive_dir
        if archive_path.exists():
            # Remove existing target if it exists
            if extract_path.exists():
                import shutil
                shutil.rmtree(extract_path)
            archive_path.rename(extract_path)
            print(f"Renamed {archive_dir} -> {dataset['extract_dir']}")

    # Clean up
    print("Cleaning up...")
    zip_path.unlink()

    print(f"Successfully downloaded {name} to {extract_path}")
    return True


def list_datasets():
    """Print available datasets."""
    print("Available datasets:\n")
    for name, info in DATASETS.items():
        print(f"  {name}")
        print(f"    Description: {info['description']}")
        print(f"    Size: {info['size']}")
        print()


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
        default="datasets",
        help="Output directory (default: datasets)"
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
        if not download_dataset(name, args.output, args.force):
            success = False

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

"""
Data download utilities for PyTorch Connectomics.

Provides automatic download of tutorial datasets from HuggingFace.
"""

from __future__ import annotations

import shutil
import sys
import tarfile
import zipfile
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlretrieve

# Dataset registry with download information
DATASETS = {
    "lucchi": {
        "name": "Lucchi++ Mitochondria Segmentation",
        "url": "https://huggingface.co/datasets/pytc/tutorial/resolve/main/lucchi%2B%2B.zip",
        "filename": "lucchi++.zip",
        "archive_dir": "lucchi++",
        "extract_dir": "lucchi++",
        "description": "EM images with mitochondria annotations from Lucchi et al.",
        "size": "~50 MB",
        "files": [
            "datasets/lucchi++/train_image.h5",
            "datasets/lucchi++/train_label.h5",
            "datasets/lucchi++/test_image.h5",
            "datasets/lucchi++/test_label.h5",
        ],
    },
    "lucchi++": {  # Alias for lucchi
        "name": "Lucchi++ Mitochondria Segmentation",
        "url": "https://huggingface.co/datasets/pytc/tutorial/resolve/main/lucchi%2B%2B.zip",
        "filename": "lucchi++.zip",
        "archive_dir": "lucchi++",
        "extract_dir": "lucchi++",
        "description": "EM images with mitochondria annotations from Lucchi et al.",
        "size": "~50 MB",
        "files": [
            "datasets/lucchi++/train_image.h5",
            "datasets/lucchi++/train_label.h5",
            "datasets/lucchi++/test_image.h5",
            "datasets/lucchi++/test_label.h5",
        ],
    },
    "snemi": {
        "name": "SNEMI3D Neuron Segmentation",
        "url": "https://huggingface.co/datasets/pytc/tutorial/resolve/main/SNEMI3D.zip",
        "filename": "SNEMI3D.zip",
        "archive_dir": "SNEMI3D",
        "extract_dir": "snemi",
        "description": "SNEMI3D neuron segmentation challenge dataset",
        "size": "~200 MB",
    },
    "mitoem": {
        "name": "MitoEM Mitochondria Segmentation",
        "url": "https://huggingface.co/datasets/pytc/tutorial/resolve/main/MitoEM.zip",
        "filename": "MitoEM.zip",
        "archive_dir": "MitoEM",
        "extract_dir": "mitoem",
        "description": "MitoEM large-scale mitochondria 3D segmentation",
        "size": "~2 GB",
    },
    "cremi": {
        "name": "CREMI Synaptic Cleft Detection",
        "url": "https://huggingface.co/datasets/pytc/tutorial/resolve/main/CREMI.zip",
        "filename": "CREMI.zip",
        "archive_dir": "CREMI",
        "extract_dir": "cremi",
        "description": "CREMI synaptic cleft detection challenge",
        "size": "~500 MB",
    },
}


def _progress_hook(count, block_size, total_size):
    """Display download progress."""
    percent = min(100, count * block_size * 100 // total_size)
    bar_length = 40
    filled = int(bar_length * percent // 100)
    bar = "=" * filled + "-" * (bar_length - filled)
    sys.stdout.write(f"\r[{bar}] {percent}%")
    sys.stdout.flush()


def download_dataset(
    dataset_name: str,
    base_dir: Path = Path("."),
    force: bool = False,
) -> bool:
    """
    Download and extract a tutorial dataset.

    Args:
        dataset_name: Name of dataset (e.g., "lucchi", "snemi")
        base_dir: Base directory to download to (default: current directory)
        force: Force re-download even if exists

    Returns:
        True if successful, False otherwise
    """
    if dataset_name not in DATASETS:
        print(f"ERROR: Unknown dataset: {dataset_name}")
        print(f"   Available datasets: {', '.join(DATASETS.keys())}")
        return False

    dataset_info = DATASETS[dataset_name]
    output_dir = base_dir / "datasets"
    extract_path = output_dir / dataset_info["extract_dir"]

    # Check if already exists
    if extract_path.exists() and not force:
        print(f"Dataset '{dataset_name}' already exists at {extract_path}")
        print("   Use force=True to re-download")
        return True

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {dataset_info['name']}")
    print(f"   Size: {dataset_info['size']}")
    print(f"   URL: {dataset_info['url']}")

    zip_path = output_dir / dataset_info["filename"]

    try:
        urlretrieve(dataset_info["url"], zip_path, reporthook=_progress_hook)
        print()  # New line after progress bar
    except URLError as e:
        print(f"\nERROR: Download failed: {e}")
        return False
    except KeyboardInterrupt:
        print("\nDownload cancelled")
        if zip_path.exists():
            zip_path.unlink()
        return False

    # Extract
    print(f"Extracting to {output_dir}...")
    try:
        if zip_path.suffix == ".zip":
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(output_dir)
        elif zip_path.suffix in [".tar", ".gz", ".tgz"]:
            with tarfile.open(zip_path, "r:*") as tf:
                tf.extractall(output_dir)
        else:
            print(f"ERROR: Unknown archive format: {zip_path.suffix}")
            return False
    except Exception as e:
        print(f"ERROR: Extraction failed: {e}")
        return False

    # Rename archive directory to canonical name if needed
    archive_dir = dataset_info.get("archive_dir")
    if archive_dir and archive_dir != dataset_info["extract_dir"]:
        archive_path = output_dir / archive_dir
        if archive_path.exists():
            if extract_path.exists():
                shutil.rmtree(extract_path)
            archive_path.rename(extract_path)

    # Clean up archive
    zip_path.unlink()

    print(f"Dataset '{dataset_name}' ready at {extract_path}")
    return True


def list_datasets():
    """Print available datasets."""
    print("Available Tutorial Datasets:\n")
    for name, info in DATASETS.items():
        if name.endswith("++"):  # Skip aliases
            continue
        print(f"  {name}")
        print(f"    {info['description']}")
        print(f"    Size: {info['size']}")
        print()

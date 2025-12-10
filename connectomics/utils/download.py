"""
Data download utilities for PyTorch Connectomics.

Provides automatic download of tutorial datasets from HuggingFace.
"""

import urllib.request
import zipfile
from pathlib import Path
from typing import Optional
from tqdm import tqdm


# Dataset registry
DATASETS = {
    "lucchi": {
        "name": "Lucchi++ Mitochondria Segmentation",
        "url": "https://huggingface.co/datasets/pytc/tutorial/resolve/main/lucchi%2B%2B.zip",
        "size_mb": 100,
        "description": "EM images with mitochondria annotations from Lucchi et al.",
        "files": [
            "datasets/lucchi++/train_image.h5",
            "datasets/lucchi++/train_label.h5",
            "datasets/lucchi++/test_image.h5",
            "datasets/lucchi++/test_label.h5",
        ],
    },
    "lucchi++": {  # Alias
        "name": "Lucchi++ Mitochondria Segmentation",
        "url": "https://huggingface.co/datasets/pytc/tutorial/resolve/main/lucchi%2B%2B.zip",
        "size_mb": 100,
        "description": "EM images with mitochondria annotations from Lucchi et al.",
        "files": [
            "datasets/lucchi++/train_image.h5",
            "datasets/lucchi++/train_label.h5",
            "datasets/lucchi++/test_image.h5",
            "datasets/lucchi++/test_label.h5",
        ],
    },
}


class DownloadProgressBar(tqdm):
    """Progress bar for urllib downloads."""

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url: str, output_path: Path, desc: Optional[str] = None):
    """
    Download file from URL with progress bar.

    Args:
        url: URL to download from
        output_path: Path to save file
        desc: Description for progress bar
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with DownloadProgressBar(
        unit="B", unit_scale=True, miniters=1, desc=desc or "Downloading"
    ) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def extract_zip(zip_path: Path, extract_dir: Path):
    """
    Extract zip file with progress bar.

    Args:
        zip_path: Path to zip file
        extract_dir: Directory to extract to
    """
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        members = zip_ref.namelist()
        print(f"📦 Extracting {len(members)} files...")
        for member in tqdm(members, desc="Extracting"):
            zip_ref.extract(member, extract_dir)


def check_dataset_exists(dataset_name: str, base_dir: Path = Path(".")) -> bool:
    """
    Check if dataset is already downloaded.

    Args:
        dataset_name: Name of dataset (e.g., "lucchi")
        base_dir: Base directory to check

    Returns:
        True if all dataset files exist
    """
    if dataset_name not in DATASETS:
        return False

    dataset_info = DATASETS[dataset_name]
    for file_path in dataset_info["files"]:
        full_path = base_dir / file_path
        if not full_path.exists():
            return False

    return True


def download_dataset(dataset_name: str, base_dir: Path = Path("."), force: bool = False) -> bool:
    """
    Download tutorial dataset.

    Args:
        dataset_name: Name of dataset (e.g., "lucchi")
        base_dir: Base directory to download to (default: current directory)
        force: Force re-download even if exists

    Returns:
        True if successful, False otherwise
    """
    if dataset_name not in DATASETS:
        print(f"❌ Unknown dataset: {dataset_name}")
        print(f"   Available datasets: {', '.join(DATASETS.keys())}")
        return False

    dataset_info = DATASETS[dataset_name]

    # Check if already exists
    if not force and check_dataset_exists(dataset_name, base_dir):
        print(f"✅ Dataset '{dataset_name}' already exists")
        print("   Use force=True to re-download")
        return True

    print(f"📥 Downloading {dataset_info['name']}")
    print(f"   Size: ~{dataset_info['size_mb']} MB")
    print(f"   Description: {dataset_info['description']}")

    # Create temp directory for download
    import tempfile

    temp_dir = Path(tempfile.mkdtemp(prefix="pytc_download_"))

    try:
        # Download zip file
        zip_path = temp_dir / f"{dataset_name}.zip"
        download_url(dataset_info["url"], zip_path, desc=f"Downloading {dataset_name}")

        print(f"✅ Download complete: {zip_path}")

        # Extract
        extract_zip(zip_path, base_dir)

        print("✅ Extraction complete")

        # Verify files
        missing_files = []
        for file_path in dataset_info["files"]:
            full_path = base_dir / file_path
            if not full_path.exists():
                missing_files.append(file_path)

        if missing_files:
            print("⚠️  Some files are missing:")
            for f in missing_files:
                print(f"   - {f}")
            return False

        print(f"✅ Dataset '{dataset_name}' ready!")
        print(f"   Location: {base_dir / 'datasets'}")

        return True

    except Exception as e:
        print(f"❌ Download failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        # Cleanup temp directory
        import shutil

        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass


def list_datasets():
    """Print available datasets."""
    print("📚 Available Tutorial Datasets:\n")
    for name, info in DATASETS.items():
        if name.endswith("++"):  # Skip aliases
            continue
        print(f"  {name}")
        print(f"    Name: {info['name']}")
        print(f"    Size: ~{info['size_mb']} MB")
        print(f"    Description: {info['description']}")
        print(f"    Files: {len(info['files'])} files")
        print()


def auto_download_if_missing(config_path: str, dataset_name: Optional[str] = None) -> bool:
    """
    Automatically download dataset if config references missing files.

    Args:
        config_path: Path to config file
        dataset_name: Dataset name to download (auto-detect if None)

    Returns:
        True if data is available (existed or downloaded), False otherwise
    """
    from connectomics.config import load_config
    from pathlib import Path

    # Load config
    cfg = load_config(config_path)

    # Check if train_image exists
    if cfg.data.train_image and not Path(cfg.data.train_image).exists():
        print(f"⚠️  Training data not found: {cfg.data.train_image}")

        # Try to infer dataset name from path
        if dataset_name is None:
            path_str = str(cfg.data.train_image).lower()
            for name in DATASETS.keys():
                if name in path_str:
                    dataset_name = name
                    break

        if dataset_name:
            print(f"💡 Attempting to download '{dataset_name}' dataset...")
            if download_dataset(dataset_name):
                print("✅ Data download successful!")
                return True
            else:
                print("❌ Data download failed")
                return False
        else:
            print("❌ Could not determine which dataset to download")
            print("💡 Available datasets:")
            list_datasets()
            return False

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download PyTorch Connectomics tutorial datasets")
    parser.add_argument("dataset", nargs="?", help="Dataset name (e.g., 'lucchi')")
    parser.add_argument("--list", action="store_true", help="List available datasets")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    parser.add_argument("--dir", type=str, default=".", help="Base directory (default: current)")

    args = parser.parse_args()

    if args.list:
        list_datasets()
    elif args.dataset:
        download_dataset(args.dataset, Path(args.dir), force=args.force)
    else:
        print("Usage: python -m connectomics.utils.download <dataset>")
        print("       python -m connectomics.utils.download --list")
        list_datasets()

#!/usr/bin/env python3
"""
Convert a directory of image files (TIFF, PNG, JPG, etc.) into a single 3D HDF5 volume.

Supports various image formats:
- TIFF (single or multi-page)
- PNG
- JPEG/JPG
- BMP
- And any other formats supported by the I/O backend

Usage:
    python scripts/images_to_h5.py <input_pattern> <output_file.h5> [dataset_key]

Examples:
    # TIFF files (default dataset key: 'main')
    python scripts/images_to_h5.py "datasets/bouton-lv/LV_EM/*.tiff" datasets/bouton-lv/LV_EM.h5

    # PNG files (default dataset key: 'main')
    python scripts/images_to_h5.py "/path/to/images/*.png" output.h5

    # Custom dataset key
    python scripts/images_to_h5.py "/path/to/labels/*.png" labels.h5 labels

Note: Use quotes around the input pattern to prevent shell expansion.
"""

import sys
import os
from pathlib import Path
from connectomics.data.io import read_volume, write_hdf5


def main():
    """Main conversion function."""
    if len(sys.argv) < 3:
        print("Usage: python scripts/images_to_h5.py <input_pattern> <output_file.h5> [dataset_key]")
        print("")
        print("Examples:")
        print('  python scripts/images_to_h5.py "datasets/images/*.tiff" output.h5')
        print('  python scripts/images_to_h5.py "/path/to/images/*.png" output.h5')
        print('  python scripts/images_to_h5.py "/path/to/labels/*.png" labels.h5 labels')
        print("")
        print("Note: Use quotes around the input pattern to prevent shell expansion.")
        sys.exit(1)

    input_pattern = sys.argv[1]
    output_file = sys.argv[2]
    dataset_key = sys.argv[3] if len(sys.argv) > 3 else "main"

    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

    # Detect file format from pattern
    pattern_lower = input_pattern.lower()
    if any(ext in pattern_lower for ext in ['.tif', '.tiff']):
        format_name = "TIFF"
    elif '.png' in pattern_lower:
        format_name = "PNG"
    elif any(ext in pattern_lower for ext in ['.jpg', '.jpeg']):
        format_name = "JPEG"
    else:
        format_name = "image"

    print(f"Reading {format_name} files matching: {input_pattern}")
    print("This may take a while for large volumes...")

    # Read all image files as a 3D volume
    try:
        volume = read_volume(input_pattern)
    except Exception as e:
        print(f"Error reading images: {e}")
        print("\nTips:")
        print("  - Check that the file pattern is correct")
        print("  - Ensure all images have the same dimensions")
        print("  - Verify the image files are readable")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Volume Information:")
    print(f"{'='*60}")
    print(f"  Shape:      {volume.shape}")
    print(f"  Data type:  {volume.dtype}")
    print(f"  Size:       {volume.nbytes / (1024**3):.2f} GB")
    print(f"  Min value:  {volume.min()}")
    print(f"  Max value:  {volume.max()}")
    print(f"{'='*60}")

    print(f"\nSaving to: {output_file}")
    print(f"Dataset key: '{dataset_key}'")

    # Save as HDF5
    try:
        write_hdf5(output_file, volume, dataset=dataset_key)
        print(f"\n✓ Successfully saved 3D volume to {output_file}")
        print(f"  You can access it with h5py using: f['{dataset_key}']")
    except Exception as e:
        print(f"\n✗ Error writing HDF5 file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

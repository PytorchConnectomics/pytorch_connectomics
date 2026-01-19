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

import os
import sys

from connectomics.data.io import read_images, write_hdf5


def main():
    """Main conversion function."""
    if len(sys.argv) < 3:
        print(
            "Usage: python scripts/images_to_h5.py <input_pattern> <output_file.h5> [dataset_key]"
        )
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
    image_type = sys.argv[3] if len(sys.argv) > 3 else "image"
    dataset_key = sys.argv[4] if len(sys.argv) > 4 else "main"

    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

    # Read all image files as a 3D volume
    volume = read_images(input_pattern, image_type=image_type)

    print(f"\n{'=' * 60}")
    print("Volume Information:")
    print(f"{'=' * 60}")
    print(f"  Shape:      {volume.shape}")
    print(f"  Data type:  {volume.dtype}")
    print(f"  Size:       {volume.nbytes / (1024**3):.2f} GB")
    print(f"  Min value:  {volume.min()}")
    print(f"  Max value:  {volume.max()}")
    print(f"{'=' * 60}")

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

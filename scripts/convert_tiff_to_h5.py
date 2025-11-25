#!/usr/bin/env python3
"""
Convert a directory of TIFF files into a single 3D HDF5 volume.

Usage:
    python scripts/convert_tiff_to_h5.py <input_dir> <output_file.h5>
    python scripts/convert_tiff_to_h5.py datasets/bouton-lv/LV_EM/*.tiff datasets/bouton-lv/LV_EM.h5
"""

import sys
import os
from pathlib import Path
from connectomics.data.io import read_volume, write_hdf5


def main():
    if len(sys.argv) < 3:
        print("Usage: python scripts/convert_tiff_to_h5.py <input_pattern> <output_file.h5>")
        print("Example: python scripts/convert_tiff_to_h5.py datasets/bouton-lv/LV_EM/*.tiff datasets/bouton-lv/LV_EM.h5")
        sys.exit(1)
    
    input_pattern = sys.argv[1]
    output_file = sys.argv[2]
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"Reading TIFF files matching: {input_pattern}")
    print("This may take a while for large volumes...")
    
    # Read all TIFF files as a 3D volume
    volume = read_volume(input_pattern)
    
    print(f"Volume shape: {volume.shape}")
    print(f"Volume dtype: {volume.dtype}")
    print(f"Volume size: {volume.nbytes / (1024**3):.2f} GB")
    
    print(f"\nSaving to: {output_file}")
    print("Using dataset key: 'main'")
    
    # Save as HDF5 with key "main"
    write_hdf5(output_file, volume, dataset="main")
    
    print(f"✓ Successfully saved 3D volume to {output_file}")


if __name__ == "__main__":
    main()


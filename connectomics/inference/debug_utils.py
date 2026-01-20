"""
Debug utilities for inference pipeline analysis.

Provides automatic analysis and visualization of prediction outputs during inference.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np


def analyze_array(arr: np.ndarray, name: str, verbose: bool = True) -> None:
    """
    Analyze and print statistics about a numpy array.
    
    Args:
        arr: Numpy array to analyze
        name: Descriptive name for the array
        verbose: Whether to print detailed information
    """
    if not verbose:
        return
    
    print(f"\n{'='*70}")
    print(f"ARRAY ANALYSIS: {name}")
    print(f"{'='*70}")
    
    # Basic properties
    print(f"Shape:        {arr.shape}")
    print(f"Dtype:        {arr.dtype}")
    print(f"Size:         {arr.size:,} elements")
    print(f"Memory:       {arr.nbytes / (1024**2):.2f} MB")
    
    # Statistical properties
    print(f"\nStatistics:")
    print(f"  Min:        {arr.min()}")
    print(f"  Max:        {arr.max()}")
    print(f"  Mean:       {arr.mean():.6f}")
    print(f"  Std:        {arr.std():.6f}")
    
    # Unique values analysis
    # For large arrays, sample if needed to avoid memory issues
    if arr.size > 10_000_000:
        # Sample for unique value estimation
        sample = arr.flat[::max(1, arr.size // 1_000_000)]
        unique_vals = np.unique(sample)
        print(f"\nUnique values (sampled):")
        print(f"  Count:      {len(unique_vals)} (estimated from sample)")
    else:
        unique_vals = np.unique(arr)
        print(f"\nUnique values:")
        print(f"  Count:      {len(unique_vals)}")
    
    # Show first 30 unique values
    display_vals = unique_vals[:30]
    if len(unique_vals) <= 30:
        print(f"  Values:     {display_vals.tolist()}")
    else:
        print(f"  First 30:   {display_vals.tolist()}")
        print(f"  ... and {len(unique_vals) - 30} more")
    
    # Non-zero analysis
    nonzero_count = np.count_nonzero(arr)
    nonzero_pct = 100.0 * nonzero_count / arr.size
    print(f"\nNon-zero pixels:")
    print(f"  Count:      {nonzero_count:,} / {arr.size:,}")
    print(f"  Percentage: {nonzero_pct:.2f}%")
    
    # Value distribution (for reasonable sized arrays)
    if len(unique_vals) <= 100 and arr.size <= 10_000_000:
        print(f"\nValue distribution (top 10):")
        values, counts = np.unique(arr, return_counts=True)
        # Sort by count descending
        sorted_indices = np.argsort(counts)[::-1][:10]
        for val, cnt in zip(values[sorted_indices], counts[sorted_indices]):
            pct = 100.0 * cnt / arr.size
            print(f"  {val:>8}: {cnt:>12,} pixels ({pct:>6.2f}%)")
    
    # Warning checks
    warnings = []
    if arr.max() == arr.min():
        warnings.append(f"⚠️  CONSTANT ARRAY - All values are {arr.min()}")
    if arr.max() == 0:
        warnings.append("⚠️  ALL ZEROS - Array is completely empty")
    if len(unique_vals) == 2 and 0 in unique_vals:
        warnings.append(f"⚠️  BINARY - Only 0 and {unique_vals[unique_vals != 0][0]} present")
    if nonzero_pct < 0.1:
        warnings.append(f"⚠️  SPARSE - Less than 0.1% non-zero pixels")
    
    if warnings:
        print(f"\n{'!'*70}")
        for warning in warnings:
            print(warning)
        print(f"{'!'*70}")
    
    print(f"{'='*70}\n")


def analyze_h5_file(path: Path, verbose: bool = True) -> None:
    """
    Analyze and print statistics about an HDF5 file.
    
    Args:
        path: Path to the HDF5 file
        verbose: Whether to print detailed information
    """
    if not verbose:
        return
    
    try:
        import h5py
    except ImportError:
        print(f"⚠️  h5py not available, skipping HDF5 analysis for {path}")
        return
    
    if not path.exists():
        print(f"⚠️  File does not exist: {path}")
        return
    
    print(f"\n{'='*70}")
    print(f"HDF5 FILE ANALYSIS: {path.name}")
    print(f"{'='*70}")
    print(f"Path: {path}")
    print(f"Size: {path.stat().st_size / (1024**2):.2f} MB")
    
    try:
        with h5py.File(path, 'r') as f:
            print(f"\nKeys: {list(f.keys())}")
            
            # Analyze main dataset
            if 'main' in f:
                data = f['main'][:]
                print(f"\nDataset 'main':")
                analyze_array(data, f"HDF5:{path.name}", verbose=False)
                
                # Print summary inline
                print(f"  Shape:      {data.shape}")
                print(f"  Dtype:      {data.dtype}")
                print(f"  Min:        {data.min()}")
                print(f"  Max:        {data.max()}")
                print(f"  Mean:       {data.mean():.6f}")
                
                unique_vals = np.unique(data)
                print(f"  Unique:     {len(unique_vals)} values")
                if len(unique_vals) <= 30:
                    print(f"  Values:     {unique_vals.tolist()}")
                else:
                    print(f"  First 30:   {unique_vals[:30].tolist()}")
                
                nonzero = np.count_nonzero(data)
                nonzero_pct = 100.0 * nonzero / data.size
                print(f"  Non-zero:   {nonzero:,} / {data.size:,} ({nonzero_pct:.2f}%)")
                
                # Warnings
                if data.max() == 0:
                    print(f"\n  ⚠️  WARNING: All values are 0 (completely empty)")
                elif data.max() == data.min():
                    print(f"\n  ⚠️  WARNING: Constant value {data.min()}")
            else:
                print(f"\n⚠️  'main' dataset not found in file")
                
    except Exception as e:
        print(f"❌ Error analyzing HDF5 file: {e}")
    
    print(f"{'='*70}\n")


def save_as_nifti(
    arr: np.ndarray,
    output_path: Path,
    affine: Optional[np.ndarray] = None,
    verbose: bool = True
) -> None:
    """
    Save array as NIfTI file (.nii.gz).
    
    Args:
        arr: Numpy array to save
        output_path: Output path (should end with .nii.gz)
        affine: Optional affine matrix (defaults to identity)
        verbose: Whether to print information
    """
    try:
        import nibabel as nib
    except ImportError:
        if verbose:
            print(f"⚠️  nibabel not available, skipping NIfTI export for {output_path}")
        return
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Cast to appropriate dtype
    if np.issubdtype(arr.dtype, np.floating):
        # Float data (e.g., probabilities, SDT)
        arr_to_save = arr.astype(np.float32)
        if verbose:
            print(f"  💾 Converting to float32 for NIfTI export")
    else:
        # Integer data (e.g., instance labels)
        arr_to_save = arr.astype(np.int32)
        if verbose:
            print(f"  💾 Converting to int32 for NIfTI export")
    
    # Create affine matrix (identity if not provided)
    if affine is None:
        affine = np.eye(4)
    
    # Create NIfTI image
    nifti_img = nib.Nifti1Image(arr_to_save, affine)
    
    # Save
    nib.save(nifti_img, str(output_path))
    
    if verbose:
        file_size = output_path.stat().st_size / (1024**2)
        print(f"  ✓ Saved NIfTI: {output_path.name} ({file_size:.2f} MB)")


__all__ = [
    'analyze_array',
    'analyze_h5_file',
    'save_as_nifti',
]

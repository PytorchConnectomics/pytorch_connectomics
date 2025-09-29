#!/usr/bin/env python3
"""
Test Lightning training with Lucchi-Mitochondria config.

This script creates dummy data and runs a short training session to validate
the Lightning integration works end-to-end with real configs.
"""

import os
import sys
import tempfile
import shutil
import subprocess
import numpy as np
from skimage import io
import h5py

def create_dummy_lucchi_data(base_path):
    """Create dummy data that matches the Lucchi dataset structure."""
    print("Creating dummy Lucchi dataset...")

    # Create directory structure
    img_dir = os.path.join(base_path, "img")
    label_dir = os.path.join(base_path, "label")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    # Create dummy 3D data (smaller than real dataset for quick testing)
    # Real Lucchi: 165x768x1024, we'll use 64x128x128 for testing
    shape = (64, 128, 128)

    # Generate synthetic EM-like data
    print(f"Generating synthetic data with shape: {shape}")

    # Create synthetic EM data (grayscale, 0-255)
    np.random.seed(42)  # For reproducible results

    # Base noise
    img_data = np.random.normal(128, 30, shape).astype(np.uint8)

    # Add some structure (simulate mitochondria-like patterns)
    for i in range(10):  # Add some blob-like structures
        center_z = np.random.randint(5, shape[0]-5)
        center_y = np.random.randint(15, shape[1]-15)
        center_x = np.random.randint(15, shape[2]-15)

        # Create ellipsoid-like structure
        zz, yy, xx = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
        ellipsoid = ((zz - center_z)/3)**2 + ((yy - center_y)/8)**2 + ((xx - center_x)/8)**2 < 1

        img_data[ellipsoid] = np.random.randint(180, 220)  # Brighter regions

    # Create corresponding binary labels (mitochondria segmentation)
    label_data = np.zeros(shape, dtype=np.uint8)

    # Create mitochondria-like segmentations
    for i in range(8):
        center_z = np.random.randint(5, shape[0]-5)
        center_y = np.random.randint(10, shape[1]-10)
        center_x = np.random.randint(10, shape[2]-10)

        # Create mitochondria-like shapes
        zz, yy, xx = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
        mito = ((zz - center_z)/2)**2 + ((yy - center_y)/6)**2 + ((xx - center_x)/6)**2 < 1

        label_data[mito] = 1  # Binary segmentation

    # Save as TIFF files (as expected by the config)
    print("Saving training data...")
    img_path = os.path.join(img_dir, "train_im.tif")
    label_path = os.path.join(label_dir, "train_label.tif")

    # Convert to uint8 for TIFF compatibility
    io.imsave(img_path, img_data.astype(np.uint8))
    io.imsave(label_path, (label_data * 255).astype(np.uint8))  # Scale labels to 0-255

    # Create test data (for inference testing if needed)
    print("Saving test data...")
    test_img_path = os.path.join(img_dir, "test_im.tif")
    io.imsave(test_img_path, img_data.astype(np.uint8))  # Use same data for simplicity

    print(f"✅ Created dummy Lucchi dataset at: {base_path}")
    print(f"   - Training image: {img_path} (shape: {shape})")
    print(f"   - Training labels: {label_path} (shape: {shape})")
    print(f"   - Test image: {test_img_path} (shape: {shape})")

    return base_path

def run_lightning_training(dataset_path, output_path, iterations=50):
    """Run Lightning training with the Lucchi config."""
    print(f"\n🚀 Starting Lightning training...")
    print(f"Dataset path: {dataset_path}")
    print(f"Output path: {output_path}")
    print(f"Training iterations: {iterations}")

    # Prepare command
    cmd = [
        sys.executable, "scripts/main_lightning.py",
        "--config-file", "../configs/Lucchi-Mitochondria.yaml",
        "--lightning",  # Use Lightning trainer
        "--use-monai",  # Enable MONAI transforms
        "--gpus", "0",  # Force CPU for compatibility
        # Override config via command line
        f"DATASET.INPUT_PATH", dataset_path,
        f"DATASET.OUTPUT_PATH", output_path,
        f"SOLVER.ITERATION_TOTAL", str(iterations),
        f"SOLVER.SAMPLES_PER_BATCH", "1",  # Small batch for testing
        f"SYSTEM.NUM_GPUS", "0",  # Force CPU
        f"SYSTEM.PARALLEL", "NONE",  # No parallelism for CPU
        "MONITOR", "None",  # Disable monitoring for clean testing
    ]

    print(f"Command: {' '.join(cmd)}")

    try:
        # Run training
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        print(f"\n📊 Training Results:")
        print(f"Return code: {result.returncode}")

        if result.stdout:
            print(f"\n📝 STDOUT:\n{result.stdout}")

        if result.stderr:
            print(f"\n⚠️ STDERR:\n{result.stderr}")

        if result.returncode == 0:
            print("✅ Lightning training completed successfully!")

            # Check if output files were created
            if os.path.exists(output_path):
                files = os.listdir(output_path)
                print(f"📁 Output files created: {files}")

                # Look for checkpoints
                for file in files:
                    if file.endswith('.ckpt'):
                        print(f"💾 Checkpoint created: {file}")

            return True
        else:
            print("❌ Lightning training failed!")
            return False

    except subprocess.TimeoutExpired:
        print("⏱️ Training timed out (5 minutes)")
        return False
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        return False

def run_comparison_training(dataset_path, output_path, iterations=50):
    """Run original trainer for comparison."""
    print(f"\n🔄 Running original trainer for comparison...")

    # Prepare command (without --lightning flag)
    cmd = [
        sys.executable, "scripts/main_lightning.py",
        "--config-file", "../configs/Lucchi-Mitochondria.yaml",
        # No --lightning flag = use original trainer
        "--no-monai",  # Disable MONAI for original trainer
        # Override config via command line
        f"DATASET.INPUT_PATH", dataset_path,
        f"DATASET.OUTPUT_PATH", output_path + "_original",
        f"SOLVER.ITERATION_TOTAL", str(iterations),
        f"SOLVER.SAMPLES_PER_BATCH", "1",  # Small batch for testing
        f"SYSTEM.NUM_GPUS", "0",  # Force CPU
        f"SYSTEM.PARALLEL", "NONE",  # No parallelism for CPU
        "MONITOR", "None",  # Disable monitoring for clean testing
    ]

    print(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        print(f"\n📊 Original Trainer Results:")
        print(f"Return code: {result.returncode}")

        if result.returncode == 0:
            print("✅ Original training completed successfully!")
            return True
        else:
            print("❌ Original training failed!")
            if result.stderr:
                print(f"Error: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print("⏱️ Original training timed out")
        return False
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        return False

def main():
    """Main test function."""
    print("🧪 Testing Lightning Integration with Lucchi-Mitochondria Config")
    print("=" * 70)

    # Create temporary directories
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Working in temporary directory: {temp_dir}")

        # Set up paths
        dataset_path = os.path.join(temp_dir, "datasets", "Lucchi")
        output_path = os.path.join(temp_dir, "outputs", "Lucchi_Lightning_Test")

        # Create dummy data
        create_dummy_lucchi_data(dataset_path)

        # Test Lightning training
        lightning_success = run_lightning_training(dataset_path, output_path, iterations=10)

        # Test original training for comparison
        original_success = run_comparison_training(dataset_path, output_path, iterations=10)

        # Summary
        print("\n" + "=" * 70)
        print("🎯 Test Summary:")
        print(f"Lightning Training: {'✅ SUCCESS' if lightning_success else '❌ FAILED'}")
        print(f"Original Training:  {'✅ SUCCESS' if original_success else '❌ FAILED'}")

        if lightning_success and original_success:
            print("\n🎉 Both trainers work! Lightning integration is successful!")
        elif lightning_success:
            print("\n⚡ Lightning trainer works! (Original trainer had issues)")
        elif original_success:
            print("\n⚠️ Original trainer works, but Lightning trainer has issues")
        else:
            print("\n💥 Both trainers failed - investigate further")

        # Keep results if successful
        if lightning_success:
            persistent_output = "/tmp/lucchi_lightning_test_results"
            if os.path.exists(persistent_output):
                shutil.rmtree(persistent_output)
            shutil.copytree(output_path, persistent_output)
            print(f"\n📁 Results saved to: {persistent_output}")

        return lightning_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
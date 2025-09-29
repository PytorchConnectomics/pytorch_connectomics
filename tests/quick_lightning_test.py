#!/usr/bin/env python3
"""
Quick Lightning test with minimal setup.
"""

import os
import sys
import tempfile
import subprocess
import numpy as np
from skimage import io

# Create a very simple test
with tempfile.TemporaryDirectory() as temp_dir:
    print(f"Working in: {temp_dir}")

    # Create dataset structure
    dataset_path = os.path.join(temp_dir, "datasets", "Lucchi")
    img_dir = os.path.join(dataset_path, "img")
    label_dir = os.path.join(dataset_path, "label")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    # Create dummy data (needs to be large enough for model + padding)
    # Model expects 112x112x112, plus padding [8, 28, 28], so let's use 150x200x200
    shape = (150, 200, 200)
    img_data = np.random.randint(0, 255, shape, dtype=np.uint8)
    label_data = np.random.randint(0, 2, shape, dtype=np.uint8) * 255

    # Save data
    io.imsave(os.path.join(img_dir, "train_im.tif"), img_data)
    io.imsave(os.path.join(label_dir, "train_label.tif"), label_data)
    io.imsave(os.path.join(img_dir, "test_im.tif"), img_data)

    print("✅ Created tiny dataset")

    # Test Lightning with absolute minimal config
    output_path = os.path.join(temp_dir, "outputs")

    cmd = [
        sys.executable, "scripts/main_lightning.py",
        "--config-file", "../configs/Lucchi-Mitochondria.yaml",
        "--lightning",
        "--no-monai",  # Disable MONAI to avoid transform issues
        "--gpus", "0",  # CPU only
        "--fast-dev-run",  # Lightning fast dev run (1 batch)
        # Config overrides
        "DATASET.INPUT_PATH", dataset_path,
        "DATASET.OUTPUT_PATH", output_path,
        "SYSTEM.NUM_GPUS", "0",
        "SYSTEM.PARALLEL", "NONE",
    ]

    print(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        print(f"Return code: {result.returncode}")

        if result.returncode == 0:
            print("🎉 SUCCESS! Lightning training worked!")
        else:
            print("❌ FAILED")
            print("STDERR:", result.stderr[-1000:])  # Last 1000 chars

    except subprocess.TimeoutExpired:
        print("⏱️ Timed out")
    except Exception as e:
        print(f"Error: {e}")
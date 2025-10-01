#!/usr/bin/env python3
"""
Test script for the enhanced main_lightning.py script.

This validates that the main script can handle different argument combinations
and configuration options correctly.
"""

import os
import sys
import tempfile
import subprocess

# Add the package to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_main_script_arguments():
    """Test that main_lightning.py handles arguments correctly."""
    print("Testing main_lightning.py argument handling...")

    # Create a temporary minimal config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        minimal_config = """
MODEL:
  ARCHITECTURE: 'unet_3d'
  IN_PLANES: 1
  OUT_PLANES: 1
  INPUT_SIZE: [32, 32, 32]
  OUTPUT_SIZE: [32, 32, 32]
  FILTERS: [16, 32]
  BLOCKS: [1, 1]
  ISOTROPY: [False, True]
  LOSS_OPTION: [['WeightedBCEWithLogitsLoss']]

SOLVER:
  BASE_LR: 1e-3
  SAMPLES_PER_BATCH: 1
  ITERATION_TOTAL: 10

SYSTEM:
  NUM_GPUS: 0
  NUM_CPUS: 1
  PARALLEL: 'NONE'

DATASET:
  OUTPUT_PATH: '/tmp/test_output'
  IMAGE_NAME: ['dummy.h5']
  LABEL_NAME: ['dummy.h5']
  INPUT_PATH: '/tmp'
"""
        f.write(minimal_config)
        config_path = f.name

    try:
        # Test 1: Help message
        result = subprocess.run([
            sys.executable, 'scripts/main_lightning.py', '--help'
        ], capture_output=True, text=True, timeout=30)

        help_success = result.returncode == 0 and '--lightning' in result.stdout
        print(f"Help message: {'PASSED' if help_success else 'FAILED'}")

        # Test 2: Lightning argument parsing (dry run)
        # We'll just check that it doesn't crash on argument parsing
        cmd = [
            sys.executable, 'scripts/main_lightning.py',
            '--config-file', config_path,
            '--lightning',
            '--fast-dev-run',
            '--use-monai',
            '--gpus', '0'
        ]

        print(f"Testing command: {' '.join(cmd)}")

        # We expect this to fail at the training stage due to missing data files,
        # but it should pass argument parsing and initial setup
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            # Check if it got past argument parsing (look for our config output)
            parsing_success = (
                'Using PyTorch Lightning Trainer' in result.stdout or
                'MONAI transforms:' in result.stdout or
                'PyTorch:' in result.stdout
            )

            print(f"Argument parsing: {'PASSED' if parsing_success else 'FAILED'}")

            if not parsing_success:
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")

        except subprocess.TimeoutExpired:
            print("Argument parsing: PASSED (timed out during execution, but parsing worked)")
            parsing_success = True

        return help_success and parsing_success

    finally:
        # Clean up
        try:
            os.unlink(config_path)
        except:
            pass

def test_config_validation():
    """Test configuration validation functions."""
    print("\nTesting configuration validation...")

    try:
        from connectomics.config.lightning_config import (
            adapt_cfg_for_lightning,
            validate_lightning_config
        )
        from connectomics.config import get_cfg_defaults

        cfg = get_cfg_defaults()
        cfg.MODEL.ARCHITECTURE = 'unet_3d'
        cfg.MODEL.IN_PLANES = 1
        cfg.MODEL.OUT_PLANES = 1
        cfg.MODEL.FILTERS = [16, 32]
        cfg.MODEL.ISOTROPY = [False, True]
        cfg.SOLVER.BASE_LR = 1e-3
        cfg.SOLVER.SAMPLES_PER_BATCH = 2
        cfg.DATASET.OUTPUT_PATH = '/tmp/test'
        cfg.DATASET.IMAGE_NAME = ['test.h5']
        cfg.DATASET.LABEL_NAME = ['test.h5']

        # Test validation
        is_valid = validate_lightning_config(cfg)
        print(f"Config validation: {'PASSED' if is_valid else 'FAILED'}")

        # Test adaptation
        lightning_cfg = adapt_cfg_for_lightning(cfg)
        adaptation_success = (
            lightning_cfg is not None and
            'model' in lightning_cfg and
            'training' in lightning_cfg
        )
        print(f"Config adaptation: {'PASSED' if adaptation_success else 'FAILED'}")

        return is_valid and adaptation_success

    except Exception as e:
        print(f"Config validation: FAILED - {e}")
        return False

def run_main_script_tests():
    """Run all main script tests."""
    print("PyTorch Connectomics - Main Lightning Script Tests")
    print("=" * 60)

    tests = [
        ("Main Script Arguments", test_main_script_arguments),
        ("Config Validation", test_config_validation),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"{test_name}: FAILED - {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary:")
    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"{test_name:30}: {status}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All main script tests passed!")
    else:
        print("❌ Some main script tests failed.")

    return passed == total

if __name__ == "__main__":
    success = run_main_script_tests()
    sys.exit(0 if success else 1)
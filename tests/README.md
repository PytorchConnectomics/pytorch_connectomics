# PyTorch Connectomics Test Suite

This directory contains the test suite for PyTorch Connectomics, organized by test type.

## Directory Structure

```
tests/
├── unit/              # Unit tests (fast, isolated)
├── integration/       # Integration tests (multiple components)
├── e2e/               # End-to-end tests (complete workflows)
└── test_banis_features.py  # BANIS Phase 12 integration tests
```

## Test Categories

### Unit Tests (`unit/`)

Fast, isolated tests for individual components:
- **test_hydra_config.py** - Configuration system (dataclasses, validation)
- **test_architecture_registry.py** - Model registry system
- **test_registry_basic.py** - Basic registry functionality
- **test_loss_functions.py** - Loss function implementations
- **test_augmentations.py** - Data augmentations
- **test_monai_transforms.py** - MONAI transform wrappers
- **test_em_augmentations.py** - EM-specific augmentations

Run unit tests only:
```bash
pytest tests/unit/
```

### Integration Tests (`integration/`)

Tests for multiple components working together:
- **test_config_integration.py** - Config system + training pipeline smoke
- **test_dataset_multi.py** - Multi-dataset loading and mixing
- **test_auto_config.py** - Auto-configuration system
- **test_auto_tuning.py** - Hyperparameter auto-tuning
- **test_affinity_cc3d.py** - Connected components and post-processing

Run integration tests only:
```bash
pytest tests/integration/
```

### End-to-End Tests (`e2e/`)

Complete workflow tests (slowest, most comprehensive):
- **test_lucchi_training.py** - Full training run on Lucchi dataset
- **test_main_lightning.py** - Complete main.py workflow

Run e2e tests only:
```bash
pytest tests/e2e/
```

### BANIS Features Tests (`test_banis_features.py`)

Tests for BANIS-inspired features (Phase 6-12):
- Slice augmentations (DropSliced, ShiftSliced)
- Numba connected components
- Weighted dataset concatenation
- Skeleton-based metrics
- SLURM utilities

## Running Tests

### Run All Tests
```bash
pytest tests/
```

### Run Specific Test Category
```bash
pytest tests/unit/          # Fast unit tests
pytest tests/integration/   # Integration tests
pytest tests/e2e/          # End-to-end tests
```

### Run Specific Test File
```bash
pytest tests/unit/test_hydra_config.py
pytest tests/integration/test_lightning_integration.py
```

### Run with Coverage
```bash
pytest tests/ --cov=connectomics --cov-report=html
```

### Run with Markers
```bash
pytest tests/ -m "not slow"  # Skip slow tests
pytest tests/ -v             # Verbose output
```

## Test Organization Principles

1. **Unit tests** should be:
   - Fast (<1 second per test)
   - Isolated (no external dependencies)
   - Focused on single components

2. **Integration tests** should:
   - Test component interactions
   - May require test data
   - Run in <10 seconds

3. **E2E tests** should:
   - Test complete workflows
   - Use real or realistic data
   - May be slower (>10 seconds)

## Adding New Tests

When adding new tests, place them in the appropriate category:

```python
# tests/unit/test_my_feature.py
"""Unit tests for my feature."""
import pytest
from connectomics.my_module import MyClass

def test_my_feature():
    """Test basic functionality."""
    obj = MyClass()
    assert obj.method() == expected_value
```

## CI/CD Integration

Tests are run automatically on:
- Pull requests
- Commits to main branches
- Nightly builds

Fast tests (unit + integration) run on every PR.
E2E tests run on nightly builds.

## Dependencies

Tests require:
```bash
pip install pytest pytest-cov
```

Optional dependencies for specific tests:
```bash
pip install funlib  # For skeleton metrics tests
```

## Troubleshooting

### ImportError
If you get import errors, ensure the package is installed in editable mode:
```bash
pip install -e .
```

### CUDA Tests
GPU tests are automatically skipped if CUDA is not available.

### Slow Tests
Use `-k` to run specific tests:
```bash
pytest tests/ -k "not training"  # Skip training tests
```

## Phase 12 Completion

✅ **Completed:**
- Reorganized tests into `unit/`, `integration/`, `e2e/` directories
- Created `__init__.py` documentation for each category
- Added `test_banis_features.py` for BANIS Phase 6-12 features
- Updated test structure documentation

**Test Coverage:**
- Unit: 7 test files
- Integration: 6 test files
- E2E: 3 test files
- BANIS: 1 test file

Total: **17 test files** organized by category

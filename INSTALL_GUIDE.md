# PyTorch Connectomics Installation Guide

## Quick Install (Recommended)

Just run the Python installation script - it handles everything automatically:

```bash
cd /projects/weilab/weidf/lib/pytorch_connectomics
python install.py
```

The script will:
1. **Auto-detect CUDA version** from your system
2. **Detect or create conda environment** (uses existing environment if you're already in one)
3. **Check and install scientific packages** via conda-forge (pre-built binaries)
   - Core packages (always installed): NumPy, h5py, Cython, connected-components-3d (cc3d)
   - Optional packages (prompted): SciPy, scikit-learn, scikit-image, OpenCV
     - **Default: Skip** (faster, pip will install if needed)
     - Conda installation can take 5-10 minutes due to dependency resolution
   - Checks which packages are already installed (shows versions)
   - Only installs missing packages
   - **CRITICAL**: cc3d installed with NumPy to avoid version conflicts
4. **Install PyTorch** with matching CUDA support
5. **Install PyTorch Connectomics** using `--no-build-isolation` (uses conda packages)
6. **Verify** the installation

**Key advantages:**
- Uses pre-built conda binaries, so no compilation needed even with old GCC (4.8.5+)
- cc3d installed via conda prevents NumPy version conflicts during pip install
- Intelligently skips already-installed packages (saves time)
- Optional packages skipped by default (faster installation, 2-3 min vs 10+ min)
- Uses your current conda environment if you're already in one
- `--no-build-isolation` ensures pip uses conda-installed dependencies

## What the Script Detects

### CUDA Detection Methods (in order)

1. **nvidia-smi** - If you have NVIDIA GPU drivers
2. **nvcc** - If CUDA toolkit is in PATH
3. **module system** - For SLURM/HPC systems (e.g., `module avail cuda`)
4. **/usr/local/** - Checks for CUDA installations

### CUDA to PyTorch Mapping

| Your CUDA Version | PyTorch Version |
|-------------------|-----------------|
| 11.x | cu118 (CUDA 11.8) |
| 12.0 | cu118 (CUDA 11.8) |
| 12.1 - 12.3 | cu121 (CUDA 12.1) |
| 12.4+ | cu124 (CUDA 12.4) |

### Manual Override

If auto-detection fails, the script offers:
1. **CPU-only** installation (no GPU)
2. **Manual CUDA specification** (enter version manually)
3. **Exit** to install CUDA first

## Command-Line Options

```bash
python install.py --help                    # Show all options
python install.py --env-name my_env         # Custom environment name
python install.py --python 3.10             # Use Python 3.10 (DEFAULT, required for cc3d)
python install.py --cuda 12.4               # Manually specify CUDA version
python install.py --cpu-only                # CPU-only installation (no GPU)
python install.py --yes                     # Skip all prompts (for CI/CD)
python install.py --no-color                # Disable colored output
```

**Note:** Python 3.10 is the default and strongly recommended version because `connected-components-3d` (cc3d) requires it.

## Example: SLURM System

On systems with module support (like systems with CUDA 12.4):

```bash
# Run the script
python install.py

# Output will show:
# ℹ CUDA found in module system: 12.4
# Will install PyTorch with CUDA 12.4 (cu124)
```

Then to use:
```bash
# Activate environment
conda activate pytc

# Load CUDA module (if needed)
module load cuda/12.4.1_gcc11.4.1-fq5rwhn

# Run training
python scripts/main.py --config tutorials/lucchi.yaml
```

## Common Issues

### Python 3.13 Error

```
ERROR: Package 'connectomics' requires Python: 3.13.5 not in '<3.13,>=3.8'
```

**Solution:** This is expected! Python 3.13 is not supported. The script will create a Python 3.10 environment automatically (required for cc3d).

### GCC Too Old

```
ERROR: NumPy requires GCC >= 9.3
```

**Solution:** The script installs pre-built wheels (no compilation needed). If you still see this, the system will use conda to install NumPy with pre-built binaries.

### No CUDA Detected

If the script can't detect CUDA, you have three options:
1. Install CPU-only (for testing)
2. Manually specify CUDA version
3. Exit and set up CUDA first

## Verification

After installation, verify:

```bash
conda activate pytc

# Check PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Check PyTorch Connectomics
python -c "from connectomics.models.arch import list_architectures; print(list_architectures())"
```

Expected output:
```
PyTorch: 2.x.x+cu124
CUDA: True
['mednext', 'mednext_custom', 'monai_basic_unet3d', ...]
```

## Manual Installation

If you prefer manual installation:

```bash
# Create environment
conda create -n pytc python=3.10 -y  # Python 3.10 required for cc3d
conda activate pytc

# Install core packages via conda (CRITICAL for compatibility)
conda install -c conda-forge numpy h5py cython connected-components-3d

# Install PyTorch (adjust CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install PyTorch Connectomics
cd /projects/weilab/weidf/lib/pytorch_connectomics
pip install -e .
```

## Troubleshooting

### Script fails at CUDA detection

```bash
# Option 1: Specify CUDA version manually
python install.py --cuda 12.4

# Option 2: Run in interactive mode and choose manual input
python install.py
# Select option 2 when prompted, enter CUDA version
```

### Package build errors (h5py, NumPy)

**Error:** `NumPy requires GCC >= 9.3`

**Solution:**
```bash
conda activate pytc

# Install core packages via conda FIRST (critical!)
conda install -c conda-forge numpy h5py cython

# Then install PyTorch Connectomics
pip install -e . --no-build-isolation
```

This uses pre-built conda binaries instead of building from source.

### Want different Python version

```bash
python install.py --python 3.10  # Use Python 3.10
python install.py --python 3.12  # Use Python 3.12
```

## Support

- **GitHub Issues**: https://github.com/zudi-lin/pytorch_connectomics/issues
- **Documentation**: https://connectomics.readthedocs.io
- **Slack**: https://join.slack.com/t/pytorchconnectomics/shared_invite/...

## Summary

✅ **Automatic CUDA detection**
✅ **Multi-method detection** (nvidia-smi, nvcc, modules, /usr/local)
✅ **Smart PyTorch version matching**
✅ **Python 3.13 protection**
✅ **Pre-built wheel usage** (no GCC issues)
✅ **Smart environment detection** (uses existing conda env)
✅ **Package version checking** (skips already-installed packages)
✅ **Installation verification**
✅ **SLURM/HPC support**

# Troubleshooting Guide

Common issues and solutions for PyTorch Connectomics.

---

## Installation Issues

### ❌ "No module named 'connectomics'"

**Cause:** Package not installed or wrong environment activated.

**Solution:**
```bash
# Make sure you're in the right environment
conda activate pytc

# Reinstall
cd pytorch_connectomics
pip install -e . --no-build-isolation

# Verify
python -c "import connectomics; print('Success!')"
```

---

### ❌ "NumPy requires GCC >= 9.3"

**Cause:** Old GCC version (common on HPC clusters).

**Solution 1 - Use conda (recommended):**
```bash
conda activate pytc
conda install -c conda-forge numpy h5py cython connected-components-3d -y
pip install -e . --no-build-isolation
```

**Solution 2 - Use Python 3.10:**
```bash
conda create -n pytc python=3.10 -y
conda activate pytc
pip install -e .
```

**Why?** Conda provides pre-built binaries that don't need compilation.

---

### ❌ "AttributeError: module 'numpy' has no attribute 'float'"

**Cause:** Mahotas version incompatibility with NumPy 2.0+. This occurs with older mahotas versions (< 1.4.18).

**Solution 1 - Upgrade packages (recommended):**
```bash
pip install --upgrade numpy mahotas
```

**Solution 2 - Pin compatible versions:**
```bash
pip install numpy>=1.23.0 mahotas>=1.4.18
```

**Why?** Mahotas 1.4.18+ is compatible with NumPy 2.x. The deprecated `np.float` alias was removed in NumPy 2.0.

---

### ❌ "Matplotlib requires numpy>=1.23"

**Cause:** Matplotlib requires NumPy 1.23 or higher for compatibility.

**Solution:**
```bash
conda activate pytc
conda install -c conda-forge matplotlib -y
pip install -e . --no-build-isolation
```

---

### ❌ "Could not find a version that satisfies connected-components-3d"

**Cause:** Python version incompatibility (cc3d requires Python 3.10).

**Solution:**
```bash
# Recreate environment with Python 3.10
conda remove -n pytc --all -y
conda create -n pytc python=3.10 -y
conda activate pytc
pip install -e .
```

---

### ❌ "CUDA not available" (but you have a GPU)

**Cause:** PyTorch CPU-only version installed or CUDA not loaded.

**Solution 1 - Reinstall PyTorch with CUDA:**
```bash
# For CUDA 12.1
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Verify
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

**Solution 2 - Load CUDA module (HPC):**
```bash
module avail cuda  # See available versions
module load cuda/12.1
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

---

### ❌ "ImportError: libcudnn.so.8: cannot open shared object file"

**Cause:** cuDNN not found (common on HPC).

**Solution:**
```bash
# Load cuDNN module
module load cudnn/8.9.0

# Or install via conda
conda install -c conda-forge cudnn
```

---

## Training Issues

### ❌ "CUDA out of memory"

**Cause:** Batch size or model too large for GPU memory.

**Solution 1 - Reduce batch size:**
```bash
python scripts/main.py --config tutorials/lucchi.yaml system.training.batch_size=1
```

**Solution 2 - Use gradient accumulation:**
```yaml
# In config file:
optimization:
  accumulate_grad_batches: 4  # Effective batch size = 4x
system:
  training:
    batch_size: 1
```

**Solution 3 - Use mixed precision:**
```yaml
optimization:
  precision: "16-mixed"  # Reduces memory by 50%
```

**Solution 4 - Reduce patch size:**
```yaml
data:
  patch_size: [64, 64, 64]  # Smaller patches
```

---

### ❌ "RuntimeError: DataLoader worker is killed by signal: Killed"

**Cause:** Insufficient system memory.

**Solution:**
```bash
# Reduce num_workers
python scripts/main.py --config tutorials/lucchi.yaml system.training.num_workers=2

# Or disable workers entirely
python scripts/main.py --config tutorials/lucchi.yaml system.training.num_workers=0
```

---

### ❌ "Loss is NaN" or "Loss exploding"

**Cause:** Learning rate too high, numerical instability, or bad data.

**Solution 1 - Reduce learning rate:**
```yaml
optimizer:
  lr: 1e-5  # Try lower LR (was 1e-4)
```

**Solution 2 - Enable gradient clipping:**
```yaml
optimization:
  gradient_clip_val: 1.0
```

**Solution 3 - Use FP32 instead of FP16:**
```yaml
optimization:
  precision: "32"  # More stable than "16-mixed"
```

**Solution 4 - Enable anomaly detection:**
```yaml
monitor:
  detect_anomaly: true  # Helps find exact operation causing NaN
```

**Solution 5 - Check your data:**
```python
# Check for NaN/inf in data
import h5py
with h5py.File('train_image.h5', 'r') as f:
    data = f['main'][:]
    print(f"Has NaN: {np.isnan(data).any()}")
    print(f"Has inf: {np.isinf(data).any()}")
    print(f"Range: [{data.min()}, {data.max()}]")
```

---

### ❌ "Training is very slow"

**Cause:** Multiple possible reasons.

**Solution 1 - Use mixed precision:**
```yaml
optimization:
  precision: "16-mixed"  # 2x faster
```

**Solution 2 - Increase num_workers:**
```yaml
system:
  training:
    num_workers: 8  # More parallel data loading
```

**Solution 3 - Use pre-loaded cache:**
```yaml
data:
  use_preloaded_cache: true  # Load volumes once, crop in memory
```

**Solution 4 - Disable progress bar:**
```yaml
# Add to trainer creation in main.py
enable_progress_bar: False
```

**Solution 5 - Check GPU utilization:**
```bash
nvidia-smi  # Should show high GPU utilization (>80%)
```

---

## Data Issues

### ❌ "FileNotFoundError: No such file or directory"

**Cause:** Incorrect path in config.

**Solution:**
```bash
# Check current directory
pwd

# Use absolute paths in config
data:
  train_image: "/full/path/to/train_image.h5"

# Or relative to working directory
data:
  train_image: "datasets/train_image.h5"
```

---

### ❌ "OSError: Unable to open file (file is truncated)"

**Cause:** Corrupted HDF5 file or incomplete download.

**Solution:**
```bash
# Re-download data
rm corrupted_file.h5
wget https://...

# Verify file integrity
h5ls train_image.h5
```

---

### ❌ "ValueError: patch_size must be smaller than volume size"

**Cause:** Patch size larger than input volume.

**Solution:**
```yaml
# Reduce patch size
data:
  patch_size: [64, 64, 64]  # Smaller than volume

# Or pad volume (advanced)
data:
  split_pad_val: true
  split_pad_size: [128, 128, 128]
```

---

### ❌ "Auto-download not working"

**Cause:** Network issues or missing credentials.

**Solution - Manual download:**
```bash
# Download from HuggingFace
wget https://huggingface.co/datasets/pytc/tutorial/resolve/main/lucchi%2B%2B.zip
unzip lucchi++.zip -d datasets/

# Or use git-lfs
git lfs install
git clone https://huggingface.co/datasets/pytc/tutorial
```

---

## Configuration Issues

### ❌ "KeyError: 'missing_key'"

**Cause:** Config file missing required fields.

**Solution:**
```bash
# Use example config as template
cp tutorials/lucchi.yaml my_config.yaml

# Check required fields
python -c "from connectomics.config import load_config; load_config('my_config.yaml')"
```

---

### ❌ "OmegaConf errors" (various)

**Cause:** YAML syntax error or type mismatch.

**Solution:**
```bash
# Check YAML syntax
python -c "import yaml; yaml.safe_load(open('config.yaml'))"

# Common issues:
# - Use spaces, not tabs
# - Quote strings with special characters
# - Check indentation
```

---

## Testing Issues

### ❌ "No test dataset found"

**Cause:** `inference.data.test_image` not set in config.

**Solution:**
```yaml
# Add to config file
inference:
  data:
    test_image: "path/to/test_image.h5"
    test_label: "path/to/test_label.h5"  # Optional
```

---

### ❌ "Checkpoint not found"

**Cause:** Incorrect checkpoint path.

**Solution:**
```bash
# Find checkpoints
find outputs/ -name "*.ckpt"

# Use full path
python scripts/main.py --config config.yaml --mode test \
    --checkpoint outputs/experiment/20241012_123456/checkpoints/epoch=099.ckpt
```

---

## HPC/SLURM Issues

### ❌ "sbatch: command not found"

**Cause:** Not on a SLURM cluster or SLURM not in PATH.

**Solution:**
```bash
# Check if on SLURM system
which sbatch

# If not found, use direct execution instead
python scripts/main.py --config config.yaml
```

---

### ❌ "Job killed without error message"

**Cause:** Exceeded memory or time limits.

**Solution:**
```bash
# Request more memory
#SBATCH --mem=64G

# Request more time
#SBATCH --time=48:00:00

# Check logs
cat slurm-123456.out
```

---

## Environment Issues

### ❌ "Conda command not found"

**Cause:** Conda not installed or not in PATH.

**Solution:**
```bash
# Initialize conda
source ~/miniconda3/bin/activate

# Or install miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

---

### ❌ "Wrong Python version"

**Cause:** Using system Python instead of conda environment.

**Solution:**
```bash
# Check Python version
which python  # Should show conda path
python --version  # Should be 3.10

# Activate correct environment
conda activate pytc
```

---

## Getting More Help

If your issue isn't listed here:

1. **Check logs:** Look for detailed error messages in terminal output
2. **Search issues:** [GitHub Issues](https://github.com/zudi-lin/pytorch_connectomics/issues)
3. **Ask community:** [Slack channel](https://join.slack.com/t/pytorchconnectomics/shared_invite/zt-obufj5d1-v5_NndNS5yog8vhxy4L12w)
4. **Report bug:** Create a new [GitHub Issue](https://github.com/zudi-lin/pytorch_connectomics/issues/new)

**When reporting issues, include:**
- Python version: `python --version`
- PyTorch version: `python -c "import torch; print(torch.__version__)"`
- CUDA version: `nvcc --version` or `nvidia-smi`
- Full error traceback
- Config file (if relevant)

---

## Common Warnings (Can Ignore)

### ⚠️ "UserWarning: The dataloader X does not have many workers"

**Safe to ignore.** Increase `num_workers` for faster data loading if desired.

---

### ⚠️ "UserWarning: TypedStorage is deprecated"

**Safe to ignore.** This is a PyTorch internal warning and doesn't affect functionality.

---

### ⚠️ "FutureWarning: `torch.cuda.amp.autocast`"

**Safe to ignore.** This is about API changes in future PyTorch versions.

---

<p align="center">
Still stuck? We're here to help! 💬<br>
<a href="https://join.slack.com/t/pytorchconnectomics/shared_invite/zt-obufj5d1-v5_NndNS5yog8vhxy4L12w">Join our Slack community</a>
</p>

# Quick Start Guide

Get PyTorch Connectomics running in **5 minutes**! 🚀

## What You'll Do

1. **Install** PyTorch Connectomics (2-3 minutes)
2. **Run a demo** to verify installation (30 seconds)
3. **Try a tutorial** with real data (optional)

---

## Step 1: Install (Choose ONE method)

### 🚀 Method A: One-Command Install (Recommended)

```bash
curl -fsSL https://raw.githubusercontent.com/zudi-lin/pytorch_connectomics/v2.0/quickstart.sh | bash
```

That's it! The script will:
- ✅ Install conda (if needed)
- ✅ Detect your CUDA version
- ✅ Install PyTorch + PyTorch Connectomics
- ✅ Verify installation

**Time:** 2-3 minutes

---

### 🐍 Method B: Python Script (More Control)

```bash
# Clone repository
git clone https://github.com/zudi-lin/pytorch_connectomics.git
cd pytorch_connectomics

# Run installer
python install.py

# Activate environment
conda activate pytc
```

**Time:** 2-3 minutes

---

### 🛠️ Method C: Manual Installation

```bash
# Create conda environment
conda create -n pytc python=3.10 -y
conda activate pytc

# Install pre-built packages (avoids compilation)
conda install -c conda-forge numpy=1.23 h5py cython connected-components-3d mahotas -y

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install PyTorch Connectomics
git clone https://github.com/zudi-lin/pytorch_connectomics.git
cd pytorch_connectomics
pip install -e . --no-build-isolation
```

**Time:** 3-5 minutes

---

## Step 2: Verify Installation

### Quick Demo (30 seconds)

```bash
conda activate pytc
python scripts/main.py --demo
```

This creates synthetic data and trains a small model for 5 epochs. If this works, your installation is successful! ✅

**Expected output:**
```
🎯 PyTorch Connectomics Demo Mode
...
✅ DEMO COMPLETED SUCCESSFULLY!
Your installation is working correctly! 🎉
```

---

## Step 3: Try a Real Tutorial (Optional)

### Download Tutorial Data

The Lucchi++ dataset contains mitochondria segmentation data from EM images.

```bash
# Download from HuggingFace (recommended)
mkdir -p datasets/
wget https://huggingface.co/datasets/pytc/tutorial/resolve/main/lucchi%2B%2B.zip
unzip lucchi++.zip -d datasets/
rm lucchi++.zip
```

**Size:** ~100 MB

### Run Training

```bash
# Quick test (1 batch, ~30 seconds)
python scripts/main.py --config tutorials/monai_lucchi++.yaml --fast-dev-run

# Full training (~2 hours on GPU)
python scripts/main.py --config tutorials/monai_lucchi++.yaml
```

### Monitor Progress

```bash
# Launch TensorBoard (in a separate terminal)
tensorboard --logdir outputs/lucchi++_monai_unet

# Open browser to http://localhost:6006
```

---

## Common Issues

### Issue: "No module named 'connectomics'"

**Solution:**
```bash
conda activate pytc
pip install -e . --no-build-isolation
```

### Issue: "CUDA out of memory"

**Solution:** Reduce batch size in config:
```bash
python scripts/main.py --config tutorials/lucchi.yaml data.batch_size=1
```

### Issue: "Could not find CUDA"

**Solution 1:** Install CPU-only version:
```bash
python install.py --cpu-only
```

**Solution 2:** Load CUDA module (HPC clusters):
```bash
module load cuda/12.1
python install.py --cuda 12.1
```

---

## Next Steps

### Learn More
- 📚 **Full Documentation:** [connectomics.readthedocs.io](https://connectomics.readthedocs.io)
- 📖 **Developer Guide:** [.claude/CLAUDE.md](.claude/CLAUDE.md)
- 🎯 **More Tutorials:** See `tutorials/` directory

### Get Help
- 💬 **Slack Community:** [Join here](https://join.slack.com/t/pytorchconnectomics/shared_invite/zt-obufj5d1-v5_NndNS5yog8vhxy4L12w)
- 🐛 **Report Issues:** [GitHub Issues](https://github.com/zudi-lin/pytorch_connectomics/issues)
- 📧 **Email:** See README for contact info

### Customize Your Workflow

**Train on your own data:**
```bash
# Create a config file (e.g., my_config.yaml)
# See tutorials/*.yaml for examples

python scripts/main.py --config my_config.yaml
```

**Use different models:**
```yaml
# In your config file:
model:
  architecture: mednext  # Try MedNeXt (state-of-the-art)
  mednext_size: S        # S, B, M, or L
  deep_supervision: true
```

**Distributed training:**
```yaml
system:
  training:
    num_gpus: 4  # Automatically uses DDP
```

---

## Tips for Success

1. **Start small:** Use `--fast-dev-run` to test configs quickly
2. **Monitor training:** Always use TensorBoard to watch loss curves
3. **GPU memory:** Start with small batch sizes, increase gradually
4. **Ask questions:** Join our Slack community - we're friendly! 😊

---

## What's Next?

Now that you're set up, explore:

1. **Different architectures:** MONAI models, MedNeXt
2. **Advanced features:** Mixed precision, deep supervision
3. **Custom data:** HDF5, TIFF, Zarr formats
4. **Deployment:** Docker/Singularity containers

Happy segmenting! 🔬🧠

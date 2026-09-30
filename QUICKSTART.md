# Quick Start Guide

Get PyTorch Connectomics running in **5 minutes**! 🚀

## What You'll Do

1. **Install** PyTorch Connectomics (2-3 minutes)
2. **Run a demo** to verify installation (30 seconds)
3. **Try a tutorial** with real data (optional)

---

## Step 1: Install

```bash
curl -fsSL https://raw.githubusercontent.com/PytorchConnectomics/pytorch_connectomics/master/quickstart.sh | bash
cd pytorch_connectomics
conda activate pytc
```

For CUDA-specific options, manual install, or installing via an AI
coding agent (Claude Code / Codex), see
[INSTALLATION.md](INSTALLATION.md).

---

## Step 2: Verify Installation

### Quick Demo (30 seconds)

```bash
conda activate pytc
python scripts/main.py --demo
```

`--demo` creates synthetic data and runs one training batch and one validation batch on CPU. Run `python scripts/check_install.py` for GPU checks.

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
python scripts/download_data.py lucchi++
```

**Size:** ~211 MiB

### Run Training

```bash
# Quick test (1 batch, ~30 seconds)
python scripts/main.py --config tutorials/mito_lucchi++/mito_lucchi++.yaml --fast-dev-run

# Full training (~2 hours on GPU)
python scripts/main.py --config tutorials/mito_lucchi++/mito_lucchi++.yaml
```

### Monitor Progress

```bash
# Launch TensorBoard (in a separate terminal)
tensorboard --logdir outputs/mito_lucchi++

# Open browser to http://localhost:6006
```

---

## Common Issues

For installation problems, see
[INSTALLATION.md](INSTALLATION.md#common-install-issues).

### Issue: "CUDA out of memory"

**Solution:** Reduce batch size in config:
```bash
python scripts/main.py --config tutorials/mito_lucchi++/mito_lucchi++.yaml train.data.dataloader.batch_size=1
```

---

## Next Steps

### Learn More
- 📚 **Full Documentation:** [connectomics.readthedocs.io](https://connectomics.readthedocs.io)
- 📖 **Developer Guide:** [CLAUDE.md](CLAUDE.md)
- 🎯 **More Tutorials:** See `tutorials/` directory

### Get Help
- 💬 **Slack Community:** [Join here](https://join.slack.com/t/pytorchconnectomics/shared_invite/zt-obufj5d1-v5_NndNS5yog8vhxy4L12w)
- 🐛 **Report Issues:** [GitHub Issues](https://github.com/PytorchConnectomics/pytorch_connectomics/issues)
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
train:
  model:
    arch:
      type: mednext
    mednext:
      size: S  # S, B, M, or L
    loss:
      deep_supervision: true
```

**Distributed training:**
```yaml
train:
  system:
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

<a href="https://github.com/zudi-lin/pytorch_connectomics">
<img src="./.github/logo_fullname.png" width="450"></a>

<p align="left">
    <a href="https://www.python.org/">
      <img src="https://img.shields.io/badge/Python-3.8+-ff69b4.svg" /></a>
    <a href= "https://pytorch.org/">
      <img src="https://img.shields.io/badge/PyTorch-1.8+-2BAF2B.svg" /></a>
    <a href= "https://lightning.ai/">
      <img src="https://img.shields.io/badge/Lightning-2.0+-792EE5.svg" /></a>
    <a href= "https://monai.io/">
      <img src="https://img.shields.io/badge/MONAI-0.9+-00A3E0.svg" /></a>
    <a href= "https://github.com/zudi-lin/pytorch_connectomics/blob/master/LICENSE">
      <img src="https://img.shields.io/badge/License-MIT-blue.svg" /></a>
    <a href= "https://zudi-lin.github.io/pytorch_connectomics/build/html/index.html">
      <img src="https://img.shields.io/badge/Doc-Latest-2BAF2B.svg" /></a>
    <a href= "https://join.slack.com/t/pytorchconnectomics/shared_invite/zt-obufj5d1-v5_NndNS5yog8vhxy4L12w">
      <img src="https://img.shields.io/badge/Slack-Join-CC8899.svg" /></a>
    <a href= "https://arxiv.org/abs/2112.05754">
      <img src="https://img.shields.io/badge/arXiv-2112.05754-FF7F50.svg" /></a>
</p>

---

## What is PyTorch Connectomics (PyTC)?

**Automatic segmentation of neural structures in electron microscopy images** 🔬🧠

PyTorch Connectomics (PyTC) helps neuroscientists:
- ✅ **Segment** mitochondria, synapses, and neurons in 3D EM volumes
- ✅ **Train models** without deep ML expertise
- ✅ **Process** large-scale connectomics datasets efficiently

**Built on:** [PyTorch Lightning](https://lightning.ai/) + [MONAI](https://monai.io/) + [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) for modern, scalable deep learning.

---

## Quick Start (5 Minutes)

### 1. Install

Choose your preferred method:

<details open>
<summary><b>🚀 One-Command Install (Recommended)</b></summary>

```bash
curl -fsSL https://raw.githubusercontent.com/zudi-lin/pytorch_connectomics/refs/heads/master/quickstart.sh | bash
conda activate pytc
```

Done! ✅
</details>

<details>
<summary><b>🐍 Python Script Install</b></summary>

```bash
git clone https://github.com/zudi-lin/pytorch_connectomics.git
cd pytorch_connectomics
python install.py
conda activate pytc
```
</details>

<details>
<summary><b>🛠️ Manual Install</b></summary>

```bash
conda create -n pytc python=3.10 -y
conda activate pytc
conda install -c conda-forge numpy h5py cython connected-components-3d -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
git clone https://github.com/zudi-lin/pytorch_connectomics.git
cd pytorch_connectomics
pip install -e . --no-build-isolation
```
</details>

**📖 Detailed instructions:** [INSTALLATION.md](INSTALLATION.md) | **🚀 Quick start:** [QUICKSTART.md](QUICKSTART.md)

---

### 2. Run Demo

Verify your installation with a 30-second demo:

```bash
python scripts/main.py --demo
```

**Expected output:**
```
🎯 PyTorch Connectomics Demo Mode
...
✅ DEMO COMPLETED SUCCESSFULLY!
```

---

### 3. Try a Tutorial

Train on real mitochondria segmentation data:

```bash
# Download tutorial data (~50 MB)
just download mito_lucchi++

# Quick test (1 batch)
just train mito_lucchi++ --fast-dev-run

# Full training on a single GPU
just train mito_lucchi++ system.training.num_gpus=1
```

**Monitor progress:**
```bash
just tensorboard lucchi++
```

**Resume training from checkpoint:**
```bash
just resume lucchi++ outputs/lucchi++_monai_unet/*/checkpoints/last.ckpt
```

**Run inference:**
```bash
just test lucchi++ outputs/lucchi++_monai_unet/*/checkpoints/best.ckpt
```

---

## Key Features

### 🚀 Modern Architecture (v2.0)
- **PyTorch Lightning:** Automatic distributed training, mixed precision, callbacks
- **MONAI:** Medical imaging models, transforms, losses optimized for 3D volumes
- **Hydra/OmegaConf:** Type-safe configurations with CLI overrides
- **Extensible:** Easy to add custom models, losses, and transforms

### 🏗️ State-of-the-Art Models
- **MONAI Models:** BasicUNet3D, UNet, UNETR, Swin UNETR
- **MedNeXt (MICCAI 2023):** ConvNeXt-based architecture for medical imaging
- **Custom Models:** Easily integrate your own architectures

### ⚡ Performance
- **Distributed Training:** Automatic multi-GPU with DDP
- **Mixed Precision:** FP16/BF16 training for 2x speedup
- **Efficient Data Loading:** Pre-loaded caching, MONAI transforms
- **Gradient Accumulation:** Train with large effective batch sizes

### 📊 Monitoring & Logging
- **TensorBoard:** Training curves, images, metrics
- **Weights & Biases:** Experiment tracking (optional)
- **Early Stopping:** Automatic stopping when training plateaus
- **Checkpointing:** Save best models automatically

---

## Documentation

- 🚀 **[Quick Start Guide](QUICKSTART.md)** - Get running in 5 minutes
- 📦 **[Installation Guide](INSTALLATION.md)** - Detailed setup instructions
- 📚 **[Full Documentation](https://connectomics.readthedocs.io)** - Complete reference
- 🎯 **[Tutorials](tutorials/)** - Example configurations
- 🔧 **[Troubleshooting](TROUBLESHOOTING.md)** - Common issues and solutions
- 👨‍💻 **[Developer Guide](.claude/CLAUDE.md)** - Contributing and architecture

---

## Example: Train a Model

Create a config file (`my_config.yaml`):

```yaml
system:
  training:
    num_gpus: 1
    num_cpus: 4
    batch_size: 2

model:
  architecture: monai_basic_unet3d
  in_channels: 1
  out_channels: 2
  loss_functions: [DiceLoss]

data:
  train_image: "path/to/train_image.h5"
  train_label: "path/to/train_label.h5"
  patch_size: [128, 128, 128]

optimization:
  max_epochs: 100
  precision: "16-mixed"  # Mixed precision for speed

optimizer:
  name: AdamW
  lr: 1e-4
```

Train:
```bash
python scripts/main.py --config my_config.yaml
```

Override from CLI:
```bash
python scripts/main.py --config my_config.yaml data.batch_size=4 optimization.max_epochs=200
```

---

## Supported Models

### MONAI Models
- **BasicUNet3D** - Fast, simple 3D U-Net (recommended for beginners)
- **UNet** - U-Net with residual units
- **UNETR** - Transformer-based architecture
- **Swin UNETR** - Swin Transformer U-Net

### MedNeXt Models (MICCAI 2023)
- **MedNeXt-S** - 5.6M parameters (fast)
- **MedNeXt-B** - 10.5M parameters (balanced)
- **MedNeXt-M** - 17.6M parameters (accurate)
- **MedNeXt-L** - 61.8M parameters (state-of-the-art)

**See:** [.claude/MEDNEXT.md](.claude/MEDNEXT.md) for MedNeXt integration guide

---

## Data Formats

- **HDF5** (.h5) - Primary format (recommended)
- **TIFF** (.tif, .tiff) - Multi-page TIFF stacks
- **Zarr** - For large-scale datasets
- **NumPy** - Direct array loading

**Input shape:** `(batch, channels, depth, height, width)`

---

## Community & Support

- 💬 **Slack:** [Join our community](https://join.slack.com/t/pytorchconnectomics/shared_invite/zt-obufj5d1-v5_NndNS5yog8vhxy4L12w) (friendly and helpful!)
- 🐛 **Issues:** [GitHub Issues](https://github.com/zudi-lin/pytorch_connectomics/issues)
- 📧 **Contact:** See lab website
- 📄 **Paper:** [arXiv:2112.05754](https://arxiv.org/abs/2112.05754)

---

## Citation

If PyTorch Connectomics helps your research, please cite:

```bibtex
@article{lin2021pytorch,
  title={PyTorch Connectomics: A Scalable and Flexible Segmentation Framework for EM Connectomics},
  author={Lin, Zudi and Wei, Donglai and Lichtman, Jeff and Pfister, Hanspeter},
  journal={arXiv preprint arXiv:2112.05754},
  year={2021}
}
```

---

## Acknowledgements

**Powered by:**
- [PyTorch Lightning](https://lightning.ai/) - Lightning AI Team
- [MONAI](https://monai.io/) - MONAI Consortium
- [MedNeXt](https://github.com/MIC-DKFZ/MedNeXt) - DKFZ Medical Image Computing

**Supported by:**
- NSF awards IIS-1835231, IIS-2124179, IIS-2239688

---

## License

**MIT License** - See [LICENSE](LICENSE) for details.

Copyright © PyTorch Connectomics Contributors

---

## Version History

- **v2.0** (2025) - Complete rewrite with PyTorch Lightning + MONAI
- **v1.0** (2021) - Initial release

See [RELEASE_NOTES.md](RELEASE_NOTES.md) for detailed release notes.

---

<a href="https://github.com/zudi-lin/pytorch_connectomics">
<img src="./.github/logo_fullname.png" width="450"></a>

<p align="left">
    <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.8+-ff69b4.svg" /></a>
    <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-1.8+-2BAF2B.svg" /></a>
    <a href="https://lightning.ai/"><img src="https://img.shields.io/badge/Lightning-2.0+-792EE5.svg" /></a>
    <a href="https://monai.io/"><img src="https://img.shields.io/badge/MONAI-0.9+-00A3E0.svg" /></a>
    <a href="https://github.com/zudi-lin/pytorch_connectomics/blob/master/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" /></a>
    <a href="https://join.slack.com/t/pytorchconnectomics/shared_invite/zt-obufj5d1-v5_NndNS5yog8vhxy4L12w"><img src="https://img.shields.io/badge/Slack-Join-CC8899.svg" /></a>
    <a href="https://arxiv.org/abs/2112.05754"><img src="https://img.shields.io/badge/arXiv-2112.05754-FF7F50.svg" /></a>
</p>

**Modern deep learning for connectomics.** Train, run inference, decode, and evaluate segmentation pipelines on large EM volumes — from a single GPU to multi-node clusters.

Built on **[Lightning](https://lightning.ai/)** (orchestration) + **[MONAI](https://monai.io/)** (medical imaging) + **[Hydra](https://hydra.cc/)** (configs).

---

## Quick Start

```bash
# Install (one command)
curl -fsSL https://raw.githubusercontent.com/zudi-lin/pytorch_connectomics/refs/heads/master/quickstart.sh | bash
conda activate pytc

# Verify
python scripts/main.py --demo
```

Need a manual install or specific CUDA version? See **[INSTALLATION.md](INSTALLATION.md)**.

### Run a tutorial

```bash
just download lucchi++              # ~50 MB sample data
just train mito_lucchi++            # train from scratch
just test  mito_lucchi++ <ckpt>     # inference + decoding + evaluation
just tensorboard mito_lucchi++      # monitor
```

---

## Pipeline

PyTC is organized as five composable stages, each with its own config section
and entry point:

```
train   →   infer   →   decode   →   evaluate
                          ↘
                          tune  (Optuna search over decode/postproc params)
```

A single CLI dispatches them all:

```bash
python scripts/main.py --config tutorials/mito_lucchi++.yaml                       # train
python scripts/main.py --config <cfg> --mode test --checkpoint <ckpt>              # infer + decode + evaluate
python scripts/main.py --config <cfg> --mode tune --checkpoint <ckpt>              # decode-param search
```

Override anything from the CLI:

```bash
python scripts/main.py --config <cfg> data.dataloader.batch_size=4 optimization.max_epochs=200
```

---

## Features

- **State-of-the-art models** — MONAI (BasicUNet3D, UNet, UNETR, Swin UNETR), MedNeXt (S/B/M/L), nnU-Net, RSUNet
- **Distributed by default** — DDP multi-GPU, mixed precision (FP16/BF16), gradient accumulation, persistent workers
- **Big volumes** — chunked sliding-window inference, lazy zarr/HDF5 loading, streamed decode + CC stitching
- **Decoders** — waterz (with `aff85_his256` scoring), distance watershed, connected components, ABISS, dust merge
- **Tuning** — Optuna search over decode params; auto-logged to `decode_experiments.tsv`
- **Composable configs** — Hydra/OmegaConf with profile registries, strict-key validation, CLI overrides

---

## Minimal Config

```yaml
model:
  arch: { type: monai_unet }
  in_channels: 1
  out_channels: 1
  loss:
    losses:
      - { function: DiceLoss, weight: 1.0 }

data:
  train: { image: train.h5, label: train_mask.h5 }
  val:   { image: val.h5,   label: val_mask.h5 }
  dataloader: { batch_size: 2, patch_size: [128, 128, 128] }

optimization:
  optimizer: { name: AdamW, lr: 1e-4 }
  max_epochs: 100
  precision: "16-mixed"

decoding:
  - template: decoding_waterz   # instance segmentation via waterz

evaluation:
  enabled: true
  metrics: [adapted_rand, voi]
```

See [`tutorials/`](tutorials/) for 16 dataset-specific configs (mitochondria, neurons,
synapses, vesicles, fibers, nuclei).

---

## Documentation

- 🚀 [Quick Start](QUICKSTART.md) — get running in 5 minutes
- 📦 [Installation](INSTALLATION.md) — detailed setup
- 📚 [Full docs](https://connectomics.readthedocs.io)
- 🔧 [Troubleshooting](TROUBLESHOOTING.md)
- 👨‍💻 [Developer guide](CLAUDE.md) — architecture, contributing, refactor history

---

## Community

- 💬 **[Slack](https://join.slack.com/t/pytorchconnectomics/shared_invite/zt-obufj5d1-v5_NndNS5yog8vhxy4L12w)** — friendly help
- 🐛 **[Issues](https://github.com/zudi-lin/pytorch_connectomics/issues)**
- 📄 **Paper:** [arXiv:2112.05754](https://arxiv.org/abs/2112.05754)

---

## Citation

```bibtex
@article{lin2021pytorch,
  title={PyTorch Connectomics: A Scalable and Flexible Segmentation Framework for EM Connectomics},
  author={Lin, Zudi and Wei, Donglai and Lichtman, Jeff and Pfister, Hanspeter},
  journal={arXiv preprint arXiv:2112.05754},
  year={2021}
}
```

Maintained by Harvard's [Visual Computing Group](https://vcg.seas.harvard.edu/).
Supported by NSF IIS-1835231, IIS-2124179, IIS-2239688.

**MIT License** — see [LICENSE](LICENSE).

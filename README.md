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

**Modern deep learning for 2D / 3D connectomics.** Train, run inference, decode, and evaluate segmentation pipelines on large EM volumes — from a single GPU to multi-node clusters.

---

## What you can segment

| Scale           | Structure       | Datasets                                    | Tutorial config                                                  |
|-----------------|-----------------|---------------------------------------------|------------------------------------------------------------------|
| Tissue-scale    | Blood vessel    | —                                           | (coming)                                                         |
| Tissue-scale    | Nuclei          | NucMM-Z                                     | `nuc_nucmm-z`                                                    |
| Cell-scale      | Neurons         | SNEMI3D, BANIS, LiConn-MIT                  | `neuron_snemi/*`, `neuron_nisb/*`, `neuron_liconn_mit`           |
| Cell-scale      | Synapses        | CREMI                                       | `syn_cremi`                                                      |
| Cell-scale      | Fibers          | Linghu26                                    | `fiber_linghu26`                                                 |
| Organelle-scale | Mitochondria    | Lucchi++, MitoEM, MitoLab, BetaSeg          | `mito_lucchi++`, `mitoEM/*`, `mito_mitolab`, `mito_betaseg`      |
| Organelle-scale | Vesicles        | XM                                          | `vesicle_xm`                                                     |

Sample data downloads in <1 min for `lucchi++`, `snemi`, `mitoem`, `cremi` (`just download <name>`).

---

## Benchmarks

Headline metric per public benchmark. Full tables, training curves, and pretrained
checkpoints live in **[`docs/benchmarks/`](docs/benchmarks/)**.

| Dataset   | Task                    | Architecture       | Metric          | Score |
|-----------|-------------------------|--------------------|-----------------|-------|
| Lucchi++  | Mito — semantic         | MedNeXt-S          | Jaccard ↑       | [0.935](https://huggingface.co/pytc/tutorial/tree/main/mito_lucchi++) |
| MitoEM-R  | Mito — instance         | MedNeXt-L + waterz | AP ↑            | —     |
| SNEMI3D   | Neurons — instance      | RSUNet + waterz    | adapted Rand ↓  | —     |
| BANIS     | Neurons — instance      | MedNeXt-M + abiss  | NERL ↑          | —     |
| CREMI     | Synapse — semantic      | MONAI U-Net        | F1 ↑            | —     |

<!-- Maintainer TODO: replace "—" with current scores from the latest released
checkpoints, prune rows you don't have data for, and link the docs/benchmarks/
folder once it exists. -->

---

## Quick Start [[more details]](QUICKSTART.md)

Pick one install path, then **[verify](#verify)**.

### a) Auto installation

```bash
curl -fsSL https://raw.githubusercontent.com/zudi-lin/pytorch_connectomics/refs/heads/master/quickstart.sh | bash
cd pytorch_connectomics
conda activate pytc
```

### b) LLM-assisted installation

```bash
git clone https://github.com/zudi-lin/pytorch_connectomics.git
cd pytorch_connectomics
just install-claude          # or: just install-codex
conda activate pytc
```

Reads [`INSTALL_PROMPT.md`](INSTALL_PROMPT.md) and lets the agent drive `install.py`.
Requires an authenticated `claude` or `codex` CLI.

### c) Manual installation

```bash
git clone https://github.com/zudi-lin/pytorch_connectomics.git
cd pytorch_connectomics
python install.py --install-type basic --python 3.11 --env-name pytc
conda activate pytc
```

See **[INSTALLATION.md](INSTALLATION.md)** for CUDA versions, extras, and troubleshooting.

### Verify

```bash
python scripts/main.py --demo
```

---

## Recipes

**1. Train + evaluate on a built-in benchmark** — sample data is tiny:

```bash
just download lucchi++              # ~50 MB
just train     mito_lucchi++        # train from scratch
just test      mito_lucchi++ <ckpt> # infer + decode + evaluate
just tensorboard mito_lucchi++      # monitor
```

**2. Train on your own EM volume** — copy the closest tutorial, point at your data:

```bash
cp tutorials/mito_lucchi++.yaml tutorials/my_mito.yaml
# edit data.{train,val}.{image,label} paths inside my_mito.yaml
just train my_mito
```

…or override on the CLI without copying:

```bash
python scripts/main.py --config tutorials/mito_lucchi++.yaml \
    data.train.image=/path/to/train.h5 \
    data.train.label=/path/to/label.h5
```

**3. Fine-tune from a published checkpoint:**

```bash
just resume my_mito <pretrained.ckpt>
```

**4. Predict on a new volume (no labels needed):**

```bash
just test my_mito <ckpt> evaluation.enabled=false
```

**5. Sweep decode params with Optuna:**

```bash
python scripts/main.py --config tutorials/mito_lucchi++.yaml \
    --mode tune --checkpoint <ckpt>
```

**6. Drive a workflow with a coding agent** — agent reads a prompt
and runs the steps for you (interactive):

```bash
just add-dataset-claude       # or: just add-dataset-codex
just add-arch-claude          # or: just add-arch-codex
just debug-tutorial-claude    # or: just debug-tutorial-codex
```

Requires an authenticated `claude` or `codex` CLI.

---

## Under the hood

Five composable stages, each with its own config section and entry point.
Data flows left-to-right; each stage owns one transformation:

```
EM volume ─[train]─▶ checkpoint ─[infer]─▶ representation ─[decode]─▶ instances ─[evaluate]─▶ metrics
                                            (aff / dist / flow)       (cc3d / waterz / abiss)   (Rand / VOI / NERL)
                                                                              ▲
                                                                        [tune]┘  Optuna over decode params
```

Single CLI dispatches all of them; any field is overridable from the command line:

```bash
python scripts/main.py --config <cfg> data.dataloader.batch_size=4 optimization.max_epochs=200
```

---

## Acknowledgments

PyTC stands on the work of many open-source projects. Organized by what they enable here:

- **High-performance pipeline** — [PyTorch Lightning](https://lightning.ai/), [MONAI](https://monai.io/), [Zarr](https://zarr.dev/), [HDF5](https://www.h5py.org/)
- **State-of-the-art models** — [Swin UNETR](https://docs.monai.io/en/stable/networks.html#swinunetr), [MedNeXt](https://github.com/PytorchConnectomics/MedNeXt), [nnU-Net](https://github.com/MIC-DKFZ/nnUNet), [Cellpose](https://github.com/MouseLand/cellpose)
- **Effective decoding** — [cc3d](https://github.com/seung-lab/connected-components-3d), [waterz](https://github.com/funkey/waterz), [abiss](https://github.com/seung-lab/abiss)
- **Reproducible experiments** — [Hydra](https://hydra.cc/), [Optuna](https://optuna.org/)
- **Agent-driven development** — [Claude Code](https://claude.com/claude-code), [Codex](https://github.com/openai/codex), orchestrated via [ccc-agent-flow](https://github.com/donglaiw/ccc-agent-flow)

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

Actively developed and maintained by [Donglai Wei's group](https://donglaiw.github.io/) at Boston College.
Supported by NSF IIS-1835231, IIS-2124179, IIS-2239688.

**MIT License** — see [LICENSE](LICENSE).

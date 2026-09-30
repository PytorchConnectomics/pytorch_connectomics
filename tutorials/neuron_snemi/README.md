# SNEMI3D neuron segmentation

These public-data workflows train affinity predictors and decode neuron instances
with waterz. The reference recipe uses the SNEMI3D challenge volume at 30 × 6 × 6 nm.

| Config | Purpose |
|---|---|
| `tutorials/neuron_snemi/neuron_snemi.yaml` | MedNeXt affinity training and waterz decoding |
| `tutorials/neuron_snemi/neuron_snemi_efficient.yaml` | Shorter training schedule |
| `tutorials/neuron_snemi/neuron_snemi_sdt.yaml` | Affinity and skeleton-distance supervision |
| `tutorials/neuron_snemi/neuron_snemi_sdt_multitask.yaml` | Separate affinity and distance heads |

Download the data into the layout used by the configs:

```bash
python scripts/download_data.py snemi
```

Files are placed under `datasets/SNEMI/`. The public challenge provides training
images, training labels, and test images; the PyTC mirror also includes labels
for offline test evaluation. Review the config data paths if using another source.

Train and then run inference, decoding, and evaluation:

```bash
python scripts/main.py --config tutorials/neuron_snemi/neuron_snemi.yaml
python scripts/main.py --config tutorials/neuron_snemi/neuron_snemi.yaml \
  --mode test --checkpoint /path/to/checkpoints/last.ckpt
```

Use `--mode tune` with the same config and checkpoint to select decoder settings
on the configured validation split. Inspect each YAML for its training schedule,
data split, TTA settings, and optional skeleton cache requirements.

Install the optional MedNeXt and waterz packages described in `INSTALLATION.md`.

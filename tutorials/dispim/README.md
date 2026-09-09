# Dual-view diSPIM MedNeXt experiments

These three configs isolate the encoder/fusion decision while keeping the
MedNeXt-S backbone, nine-affinity target, data split, augmentation, optimizer,
patch size, and training budget fixed.

| Config | Encoder/fusion | Role |
|---|---|---|
| `dispim_early_fusion_mednext_s.yaml` | Ordinary two-channel MedNeXt-S | Minimal baseline |
| `dispim_siamese_mean_mednext_s.yaml` | Fully shared encoder; mean fusion at five scales | Strict Siamese ablation |
| `dispim_partial_gated_mednext_s.yaml` | View-specific shallow path; shared deep path; angle FiLM; reliability gates | Recommended design |

The simulator stores the two views as separate ZYX arrays. Prepare paired CZYX
arrays and the fixed 80/20 Z split once:

```bash
conda run -n pytc python tutorials/dispim/prepare_paired_zarr.py
```

The `pytc` environment must provide the optional readers and MedNeXt package:

```bash
conda run -n pytc python -m pip install zarr
conda run -n pytc python -m pip install \
  git+https://github.com/PytorchConnectomics/MedNeXt.git
```

Use one short smoke run before committing to the full sequence:

```bash
conda run -n pytc python scripts/main.py \
  --config tutorials/dispim/dispim_partial_gated_mednext_s.yaml \
  --mode train --fast-dev-run 2
```

Then train serially on the single MPS device:

```bash
tutorials/dispim/train_three_sequential.sh
```

The M5 Max settings shared by all three configs are deliberately conservative:
`32×128×128` patches, batch size 1, four-batch gradient accumulation,
activation checkpointing, four loader workers, lazy Zarr reads, and FP32. PyTC
forces FP32 on MPS because Lightning mixed precision is not considered reliable
there. Do not run the three jobs concurrently: they would contend for the same
GPU and 36 GB unified-memory pool and invalidate throughput comparisons.

On this M5 Max, an actual `partial_gated` FP32 training step at batch 1 took
about 5.45 seconds and used about 10.5 GB of Metal driver allocation. Batch 2
was both slower per sample and reached about 34.9 GB, so it is not a safe choice.
The configured 50 optimizer steps per epoch correspond to 200 patches with
four-step gradient accumulation.

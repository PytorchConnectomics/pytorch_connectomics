# Lucchi++ mitochondria segmentation

This tutorial reproduces binary mitochondria segmentation on the Lucchi++ EM
benchmark. It treats the task as semantic segmentation: a 3D encoder-decoder
predicts the mitochondria foreground mask, and evaluation reports Jaccard/IoU.

Lucchi++ is isotropic at 5 nm along all three axes, so the recipe uses a fully
3D MedNeXt with isotropic 112³ patches.

## Goal

The configuration in [`mito_lucchi++.yaml`](mito_lucchi++.yaml) uses:

- 112 × 112 × 112 input patches at 5 × 5 × 5 nm.
- MedNeXt-S with kernel size 3 and no deep supervision.
- The `binary` pipeline profile with one foreground channel.
- Cached loading, batch size 4, and the `aug_strong` augmentation profile.
- AdamW at `lr=1e-3` and `weight_decay=0.01`, with linear warmup followed by
  cosine decay for 150 epochs × 1000 steps.
- Mixed-precision training and gradient clipping at 1.0.
- 112³ sliding-window inference with 50% overlap, bump blending, and all-axis
  flip test-time augmentation.
- Jaccard/IoU evaluation.

## 1. Get the data

Download and extract the archive from the
[PyTC Hugging Face tutorial repository](https://huggingface.co/pytc/tutorial/tree/main/mito_lucchi%2B%2B):

```bash
just download lucchi++
```

The 211 MiB download has SHA-256
`d6c29c25db29780f068b40edf27918ad4af8a1ce7d699f043ed91c6b012a0637`.
After extraction, the repository should contain:

```text
datasets/lucchi++/
├── train_im.h5
├── train_mito.h5
├── test_im.h5
└── test_mito.h5
```

To download the archive manually:

```bash
mkdir -p datasets/lucchi++
curl -L \
  'https://huggingface.co/pytc/tutorial/resolve/main/mito_lucchi%2B%2B/lucchi%2B%2B.zip?download=true' \
  -o datasets/lucchi++/lucchi++.zip
unzip datasets/lucchi++/lucchi++.zip -d datasets/lucchi++
```

Lucchi++ is the relabeled version of the original Lucchi 2012 dataset released
by Casser et al. See the
[EPFL CVLab dataset page](https://www.epfl.ch/labs/cvlab/data/data-em/) for the
upstream description.

## 2. Run training

From the repository root:

```bash
conda activate pytc
python scripts/main.py \
  --config tutorials/mito_lucchi++/mito_lucchi++.yaml
```

The config uses all visible GPUs by default. Override the GPU count or per-GPU
batch size as needed:

```bash
python scripts/main.py \
  --config tutorials/mito_lucchi++/mito_lucchi++.yaml \
  system.num_gpus=4 data.dataloader.batch_size=4
```

Training runs for 150 epochs of 1000 steps. Checkpoints and logs are written
under `outputs/mito_lucchi++/<timestamp>/`. Monitor them with:

```bash
just tensorboard mito_lucchi++
```

## 3. Run inference, decoding, and evaluation

Run the combined `test` mode with a trained checkpoint:

```bash
python scripts/main.py \
  --config tutorials/mito_lucchi++/mito_lucchi++.yaml \
  --mode test \
  --checkpoint outputs/mito_lucchi++/<timestamp>/checkpoints/last.ckpt
```

The pipeline first averages eight all-axis flip predictions, then applies the
binary decoding pipeline and evaluates the result against
`datasets/lucchi++/test_mito.h5`. To disable test-time augmentation:

```bash
python scripts/main.py \
  --config tutorials/mito_lucchi++/mito_lucchi++.yaml \
  --mode test \
  --checkpoint <ckpt> \
  inference.test_time_augmentation.enabled=false
```

## 4. Reference behavior

- Training loss should fall sharply during warmup and then decline more slowly
  through cosine decay.
- Inference on the 165 × 1024 × 768 test volume takes tens of seconds on an
  A100/H100 and a few minutes on an L40S when test-time augmentation is enabled.
- Jaccard/IoU should be in the same range as published Lucchi++ benchmarks; the
  most consequential inference-time option is test-time augmentation.

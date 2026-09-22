---
license: mit
tags:
  - connectomics
  - expansion-microscopy
  - light-microscopy
  - liconn
  - 3d-segmentation
  - affinity-prediction
  - pytorch-connectomics
  - mednext
---

# LICONN ExPID82_1 affinity model (18 nm, from scratch)

Voxel affinity prediction for the **LICONN** `ExPID82_1` volume — expansion *light*
microscopy of mouse brain tissue, not EM — for use with
[PyTorch Connectomics](https://github.com/PytorchConnectomics/pytorch_connectomics)
(`tutorials/neuron_liconn_ist`).

| | |
|---|---|
| Files | `affinity_expid82_18nm_128x128x128.ckpt` (247 MB) · `train_config.yaml` (the frozen run config) |
| Architecture | MedNeXt-L, kernel 3 (61,779,399 parameters) |
| Input / output | 1 channel image → 6 channel affinity (`banis`; channels 0–2 are the r1 nearest-neighbour affinity used downstream) |
| Training patch | `[128, 128, 128]` ZYX |
| Grid | `[24, 18, 18]` nm ZYX = 18×18×24 nm XYZ |
| Training data | `ExPID82_1` train split, `270 × 4290 × 3345`, FFN-proofread GT |
| Schedule | 200 k steps from scratch, AdamW lr 1e-3, batch 2 on 1 GPU, cosine to 0 |
| Result | **0.9129 VOI** on the full held-out validation volume (ABISS decode) |

## Inference window must be `[128, 128, 128]`

MedNeXt normalizes with `GroupNorm` — per-sample, per-channel, **no running
statistics** — so normalization is computed over the sliding window's spatial extent at
every block and **the forward pass is window-size dependent**. The published result was
produced with sliding-window ROI `[128, 128, 128]`, matching the training patch. Change
the window and the numbers below no longer apply.

## Usage

```bash
hf download pytc/liconn affinity_expid82_18nm_128x128x128.ckpt --local-dir ckpt/

python scripts/main.py --config tutorials/neuron_liconn_ist/1_affinity.yaml \
  --mode test --checkpoint ckpt/affinity_expid82_18nm_128x128x128.ckpt
```

Output is float16 CZYX. Arrays are **ZYX**, so channel `c` is the edge along array axis
`c`: ch0 = Z, ch1 = Y, ch2 = X.

## Affinities are `scale_sigmoid`, not probabilities

The pipeline stores `sigmoid(0.2 · logit)`, **not** a calibrated edge probability. Measured
on this checkpoint, the stored affinity spans about **[0.01, 0.80] and never reaches 0.88**,
so ABISS thresholds copied from other PyTC tutorials (e.g. Pinky's `ws_high_threshold: 0.88`)
sit *above the data maximum* and seed nothing.

`tutorials/neuron_liconn_ist/2_abiss.yaml` therefore gives `ws_high`/`ws_low` as
**percentiles** and keeps `ws_merge_function: max`, which is monotone-invariant so a sweep
in the compressed space covers the same family of segmentations. **`mean` is not
monotone-invariant** and must not be substituted without uncompressing first.

For ABISS also set `channels: [2, 1, 0]` (its `ws` reads XYZC with channel 0 = X edge) and
`edge_storage: source` (this repo writes edge (i, i+1) at voxel i; `ws` reads the value at i
as edge (i−1, i) — without this the boundary map is one voxel off on every axis).

## Results

Full held-out validation volume, `145 × 4290 × 3345` = 2.08 G voxels, ABISS max-affinity
agglomeration at `ws_high`/`ws_low` = 94th/20th percentile, `ws_merge_threshold` 0.47:

| | VOI ↓ | split | merge | Adapted-Rand err ↓ | pred segs | GT segs |
|---|---:|---:|---:|---:|---:|---:|
| ABISS max, mt 0.47 | **0.9129** | 0.6732 | 0.2397 | 0.3195 | 79,056 | 35,815 |

The decoder is **split-dominated** at this setting, as configured. The threshold was tuned
on 1024² slabs, which truncate GT in XY and so under-penalise cross-slab splits; the
whole-volume optimum is likely at or below 0.45. Treat 0.47 as a working value, not a
tuned one.

## Training data and its ceiling

Image and GT both come from the public `ExPID82_1` release: image `image_230130b`,
segmentation `231030_agg_240123` (FFN, proofread). Crop `[140,240,240] → [555,4530,3585]`
ZYX, split at z = 270 into a 270-slice train and a 145-slice validation block.

**18 nm is the finest scale at which the proofread segmentation exists.** The source image
has a finer 9×9×12 nm level, but there is no GT there, so this model is trained at the
finest paired image+label resolution available.

## ⚠️ The checkpoint holds raw training weights, not EMA weights

EMA was enabled during training (`decay 0.999`, `validate_with_ema: true`), which means the
logged `val_loss_total` values were computed **on EMA weights**, while the checkpoint's
`state_dict` holds the **raw training weights** — the framework undoes the EMA swap before
writing the checkpoint. EMA-state persistence was added to PyTC three weeks *after* this run,
so this model's EMA weights were never saved and are unrecoverable.

Practical consequences:

- Do not quote the training-time validation loss as this checkpoint's loss; they describe
  different weights.
- The **0.9129 above is unaffected and reproducible**: in test mode the EMA state is never
  initialised, so that inference ran on exactly the weights in this file.

## Limitations

- One acquisition. All numbers are within-volume development evidence on `ExPID82_1`, not
  an independent test score, and thresholds are not known to transfer to other specimens.
- Trained on proofread labels from the target volume — supervision on the target domain.
- The GT is 42.5% background (proofread FFN leaves much unlabelled); VOI here ignores the
  background label.

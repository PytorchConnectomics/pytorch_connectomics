# EMNeuron neuron segmentation (SegNeuron's split)

BANIS+ trained on the labeled part of EMNeuron, the multi-source EM database of
[SegNeuron](https://github.com/yanchaoz/SegNeuron) (Zhang et al., MICCAI 2024), and scored on
the paper's eight evaluation volumes with the paper's metrics.

| YAML | Deep learning | Decoding |
|---|---|---|
| `banis+.yaml` | BANIS+ (MedNeXt-L/k3, 6-channel ABISS-layout affinity, class-balanced BCE, EMA, erosion 2, 200k steps) with the MedNeXt speed knobs | ABISS ws |

## Results

Protocol, fixed before the run: the final-step (200k) EMA model; ABISS settings and the TTA
choice selected on the 10 held-out `tune/` chunks only (lowest mean VI); the 8 evaluation
volumes scored once with that setting. Selected: TTA16, ABISS high 0.9, low 0.05, size
threshold 1e9, dust 0, merge function max, merge threshold 0.6 (tune mean VI 0.6280).

VI / adapted Rand error against `label.tif`, lower is better. "Paper" is SegNeuron's Table 3
(3D); "reproduced" is the released checkpoint (`yanchaoz/SegNeuron`, `SegNeuronModel.ckpt`)
with SegNeuron's own inference and `FRMC_post.py` (beta 0.25); "released + ABISS" is the same
checkpoint's affinities through this tutorial's ABISS grid and tune selection. All are scored
with the same metric calls:

| volume | split | SegNeuron (paper) | SegNeuron (reproduced) | SegNeuron released + ABISS | this tutorial |
|---|---|---|---|---|---|
| FAFB-1 | in-distribution | (bar chart) | 0.9337 / 0.1771 | 1.0446 / 0.2009 | 0.9965 / 0.1793 |
| FAFB-2 | in-distribution | (bar chart) | 0.7232 / 0.2139 | 0.7349 / 0.2009 | 0.7403 / 0.1827 |
| FIB25-1 | in-distribution | (bar chart) | 0.6031 / 0.0643 | 0.4288 / 0.0425 | 0.5006 / 0.0095 |
| FIB25-2 | in-distribution | (bar chart) | 0.4082 / 0.0325 | 0.3148 / 0.0329 | 0.3734 / 0.0159 |
| Kstr | in-distribution | (bar chart) | 1.0971 / 0.0618 | 1.1746 / 0.0633 | 1.2464 / 0.0556 |
| Ionsem (vEM4) | out-of-distribution | 0.8655 / 0.1022 | 0.8789 / 0.1022 | 0.9816 / 0.1148 | 1.0083 / 0.1125 |
| MicronsB (Basil) | out-of-distribution | 0.7719 / 0.1531 | 1.0240 / 0.1531 | 1.1967 / 0.1768 | 0.9050 / 0.1724 |
| Harris | out-of-distribution | 1.0221 / 0.1170 | 1.2742 / 0.1205 | 1.2133 / 0.1145 | 1.1844 / 0.0987 |
| **mean, in-distribution** | | | 0.7531 / 0.1099 | 0.7395 / 0.1081 | 0.7714 / 0.0886 |
| **mean, out-of-distribution** | | 0.8865 / 0.1241 | 1.0590 / 0.1253 | 1.1305 / 0.1354 | 1.0326 / 0.1279 |
| **mean, all 8** | | | 0.8678 / 0.1157 | 0.8862 / 0.1183 | 0.8694 / 0.1033 |

One training run, scored once. Mean VI over the 8 volumes equals the reproduced SegNeuron
(0.8694 vs 0.8678) and adapted Rand error is lower (0.1033 vs 0.1157); VI is above the paper's
on each out-of-distribution volume. Without TTA (x1, its own tune-selected setting) the mean
over the 8 volumes is 1.0040 / 0.1684.

The reproduction matches the paper's adapted Rand error on vEM4 and Basil to four decimals but
not its VI; the cause is unknown (the paper's environment used an older python-elf than the
pinned 0.8.1). `label_0.tif` (membranes filled) is not the paper's ground truth: against it the
reproduced numbers are far from every published value.

## Data

Download the dataset (two RAR archives, `labeled.rar` and `valid.rar`; ~11 GB extracted),
extract them, and convert:

```bash
hf download yanchaoz/EMNeuron --repo-type dataset --revision 62df92b933550cc2a5d3849fe8543191c356f2d4 \
    --local-dir /path/to/EMNeuron
cd /path/to/EMNeuron && unrar x labeled.rar && unrar x valid.rar   # or: 7z x <archive>
python scripts/prepare_emneuron.py --src /path/to/EMNeuron --out /path/to/emneuron_pytc
```

The revision is the one the reference run used. `manifest.json` from that run has sha256
`058211d8718f49a3210e99de160dcc0ba5861cafa190f966a0f8d2abf4401d6c`; a different hash means
different input volumes.

`prepare_emneuron.py` reproduces SegNeuron's training-data semantics
(`Train_and_Inference/supervised_provider.py`):

- **Ignore regions** become label `-1`, which the affinity loss masks: label 10000 in
  J0126-sbem and Mira-scn, labels 121 and 122 in Kasthuri-atum volume 0.
- **Multi-z-ratio augmentation** becomes precomputed copies: the isotropic FIB volumes
  (Hemi-brain, FIB-25) are also stored at z-stride 2 and 3.
- **Dataset-balanced sampling**: SegNeuron draws a source, then a volume. PyTC draws a volume
  uniformly, so every source volume is tiled into the same number of chunks and every source
  contributes 24-36 chunks (7.5-11% of crops each; `manifest.json` lists the shares).
- **Held-out tuning split** (`tune/`): the last chunk of the last volume of each source
  (about 3% of the training voxels; its z-stride copies are dropped). It is the validation
  set during training and the only data used to choose decoder settings.

Deviations from SegNeuron: no self-supervised pretraining, no frequency/spatial domain mixing,
no z-striding of J0126, and the 3% tuning holdout.

## Train

```bash
python scripts/main.py --config tutorials/neuron_segneuron/banis+.yaml \
    train.data.train.path=/path/to/emneuron_pytc/train \
    train.data.val.path=/path/to/emneuron_pytc/tune
```

`data.dataloader.batch_size` is per GPU: the recipe's global batch of 16 is 4 per GPU on four
GPUs; on two GPUs add `data.dataloader.batch_size=8`.

Measured on 4x L40S, global batch 16 of 32x128x128, steady state over 100-step windows:

| setting | s/step | 200k steps |
|---|---|---|
| knobs off: fp16 GradScaler, eager, native depthwise conv, unfused AdamW, no `cudnn.benchmark`, MONAI resampler, DDP `find_unused_parameters=true` | 0.457 | ~25.4 h |
| this config | 0.295 | ~16.4 h |

The knobs cost a one-time ~4 min compile at start and ~4.5 min at the first validation. The
308 chunks are preloaded per DDP rank (about 156 GB host RAM for four ranks).

## Predict, decode, score

```bash
python scripts/main.py --config tutorials/neuron_segneuron/banis+.yaml --mode test \
    --checkpoint <run>/checkpoints/step=00200000.ckpt \
    test.data.test.path=/path/to/emneuron_pytc/val   # or .../tune for decoder selection
python scripts/evaluate_emneuron.py --abiss-home /path/to/abiss --volume Harris \
    --pred <run>/test_step=00200000/Harris_img/raw_x16_ch0-1-2_weights-ema.h5 \
    --gt /path/to/EMNeuron/valid/OutofDistribution/Harris/label.tif --out scores.tsv
```

Use the final-step checkpoint `step=00200000.ckpt`: `last.ckpt` is the best-validation
checkpoint (step 180000 in the reference run). Test outputs go to `<run>/test_<checkpoint
name>/`.

The results use TTA16, averaged over the 16 views of every axis-flip subset times xy
quarter-turns k in {0, 1}:

```bash
python scripts/main.py ... --mode test \
    default.inference.test_time_augmentation.enabled=true \
    default.inference.test_time_augmentation.flip_axes=all \
    'default.inference.test_time_augmentation.rotation90_axes=[[1,2]]' \
    'default.inference.test_time_augmentation.rotate90_k=[0,1]' \
    default.inference.test_time_augmentation.patch_first_local=false \
    default.inference.window.sw_batch_size=64 test.system.num_gpus=4
```

Whole-volume TTA (`patch_first_local=false`) is needed because patch-first TTA refuses xy
quarter-turns on non-square volumes (several `tune/` chunks). With 4 GPUs the 16 views are
split across ranks; this needs exactly one volume per launch. On 4x L40S a 100 M-voxel volume
takes ~9.5 min. Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (Hemi-brain chunks hit
OOM at `sw_batch_size` 64 without it).

Volumes thinner than the 32-section window (Ionsem has 30 sections) come back zero-padded to
32 sections (MONAI `SpatialPadd`, symmetric) and are not cropped back. Crop the prediction to
`[d // 2 : d // 2 + z]` along z, where d is the padding, before decoding; the reference results
did so (Ionsem: sections 1..30 of 32).

`evaluate_emneuron.py` decodes with ABISS over a grid (watershed high threshold, size
threshold, merge function, merge threshold), fills the voxels ABISS leaves unlabeled with the
nearest segment, and scores each result with SegNeuron's evaluator calls: skimage
`adapted_rand_error` and `variation_of_information` with ground-truth label 0 ignored (VI in
bits, split + merge). `--check-skimage` asserts that its contingency-table implementation
equals skimage. `--abiss-home` (or `PYTC_ABISS_HOME`) points at a build of
[ABISS](https://github.com/PytorchConnectomics/ABISS) with its `ws` binary; the single-volume
runner is the one `tutorials/basics/decoding_abiss.yaml` uses.

`label.tif` is the evaluation ground truth; `label_0.tif`, where it exists, is the same
labeling with most membrane voxels filled.

## What differs from `../banis+.yaml`

- **Affinity layout.** Channels x, y, z, with each edge (i-1, i) stored at voxel i
  (`affinity_mode: deepem`), the layout ABISS reads. Predictions go to ABISS with no channel
  reversal or shift. In test mode PyTC crops the first voxel of each axis from destination-
  stored affinities, so the test stage pads one voxel of reflect context and crops the
  trailing one to keep outputs on the input grid.
- **Patch.** EMNeuron mixes 6-9 nm xy with 8-50 nm z and many volumes hold 31-50 sections, so
  patches are 32x128x128 ZYX. Long-range offsets are 10 voxels in x/y and 4 sections in z,
  read as leading context.
- **Augmentation.** `aug_em_neuron_fast` (flips, xy quarter-turns, intensity, one section
  defect per sample) plus motion blur and xy-only scale jitter. `aug_banis` permutes z with
  x/y and rotates in 3D, which does not fit anisotropic sections.
- **Speed.** `model.compile` with the shift-sum depthwise convolution, fused AdamW with merged
  parameter groups, bf16, `cudnn.benchmark`, the numba affine resampler, and the frozen MedNeXt
  dummy tensor so DDP runs with `find_unused_parameters=false`.

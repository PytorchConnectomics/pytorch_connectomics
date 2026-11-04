# EMNeuron neuron segmentation (SegNeuron's split)

BANIS+ trained on the labeled part of EMNeuron, the multi-source EM database of
[SegNeuron](https://github.com/yanchaoz/SegNeuron) (Zhang et al., MICCAI 2024), and scored on
the paper's eight evaluation volumes with the paper's metrics.

| YAML | Deep learning | Decoding |
|---|---|---|
| `banis+.yaml` | BANIS+ (MedNeXt-L/k3, 6-channel ABISS-layout affinity, class-balanced BCE, EMA, erosion 2, 200k steps) with the MedNeXt speed knobs | ABISS ws |

## Results

Pending: the 200k-step run has not finished. The protocol is fixed in advance:
the final-step EMA model; ABISS settings and the TTA choice selected on the 10 held-out
`tune/` chunks only (lowest mean VI); the 8 evaluation volumes scored once with that
setting.

SegNeuron's numbers (VI / adapted Rand error against `label.tif`, lower is better). "Paper" is
Table 3 (3D); "reproduced" is the released checkpoint (`yanchaoz/SegNeuron`,
`SegNeuronModel.ckpt`) with SegNeuron's own inference and `FRMC_post.py` (beta 0.25), scored
with the same metric calls:

| volume | split | SegNeuron (paper) | SegNeuron (reproduced) | this tutorial |
|---|---|---|---|---|
| FAFB-1 | in-distribution | (bar chart) | 0.9337 / 0.1771 | pending |
| FAFB-2 | in-distribution | (bar chart) | 0.7232 / 0.2139 | pending |
| FIB25-1 | in-distribution | (bar chart) | 0.6031 / 0.0643 | pending |
| FIB25-2 | in-distribution | (bar chart) | 0.4082 / 0.0325 | pending |
| Kstr | in-distribution | (bar chart) | 1.0971 / 0.0618 | pending |
| Ionsem (vEM4) | out-of-distribution | 0.8655 / 0.1022 | 0.8789 / 0.1022 | pending |
| MicronsB (Basil) | out-of-distribution | 0.7719 / 0.1531 | 1.0240 / 0.1531 | pending |
| Harris | out-of-distribution | 1.0221 / 0.1170 | 1.2742 / 0.1205 | pending |

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
    --checkpoint <run>/checkpoints/last.ckpt \
    test.data.test.path=/path/to/emneuron_pytc/val   # or .../tune for decoder selection
python scripts/evaluate_emneuron.py --abiss-home /path/to/abiss --volume Harris \
    --pred <run>/test_last/Harris_img/raw_x1_ch0-1-2_weights-ema.h5 \
    --gt /path/to/EMNeuron/valid/OutofDistribution/Harris/label.tif --out scores.tsv
```

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

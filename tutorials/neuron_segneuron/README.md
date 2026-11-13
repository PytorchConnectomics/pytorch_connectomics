# EMNeuron neuron segmentation (SegNeuron's benchmark)

A pretrained 3D EM neuron segmentation model: BANIS+ (MedNeXt-L, affinities + ABISS
watershed) trained on [EMNeuron](https://huggingface.co/datasets/yanchaoz/EMNeuron), the
EM database of [SegNeuron](https://github.com/yanchaoz/SegNeuron) (Zhang et al.,
MICCAI 2024; 10 labeled sources). Use it zero-shot on a new EM volume or as the starting point for fine-tuning.

**Weights:** [`pytc/tutorial` / `neuron_segneuron/banis_emneuron_200k.ckpt`](https://huggingface.co/pytc/tutorial/tree/main/neuron_segneuron)
(EMA weights after 200k steps, 495 MB).

## Results: PyTC vs SegNeuron (current state of the art)

VI / adapted Rand error on SegNeuron's eight evaluation volumes; lower is better, **bold** is the
best in each row. SegNeuron: Table 3 of the paper (out-of-distribution volumes only). PyTC: one
model, TTA16, three decoders (`DECODER=` in `predict.sh`):

- `abiss`: ABISS watershed + single linkage at 0.6. The setting fixed before the evaluation
  volumes were scored.
- `agglo`: `abiss`, then greedy mean-affinity agglomeration of its segments down to 0.4.
- `basins`: the same agglomeration from the watershed basins, down to 0.35.

| volume | SegNeuron | PyTC `abiss` | PyTC `agglo` | PyTC `basins` |
|---|---|---|---|---|
| FAFB-1 | – | 0.7562 / 0.1793 | **0.6751** / 0.1746 | 0.6816 / 0.1746 |
| FAFB-2 | – | 0.5593 / 0.1827 | 0.5001 / **0.1794** | **0.4738** / 0.1866 |
| FIB25-1 | – | 0.3693 / 0.0095 | **0.2525** / **0.0085** | 0.3162 / 0.0434 |
| FIB25-2 | – | 0.2737 / 0.0159 | **0.2214** / **0.0149** | 0.2247 / 0.0182 |
| Kstr | – | 1.0341 / 0.0556 | 0.9449 / **0.0544** | **0.9344** / 0.0693 |
| Ionsem (vEM4) | **0.8655** / **0.1022** | 0.9929 / 0.1125 | 0.9482 / 0.1076 | 0.9431 / 0.1067 |
| MicronsB (Basil) | 0.7719 / 0.1531 | 0.6821 / 0.1724 | 0.6402 / 0.1684 | **0.6353** / **0.1170** |
| Harris | 1.0221 / 0.1170 | 1.0257 / 0.0987 | **0.8757** / **0.0923** | 1.1056 / 0.1081 |
| mean, out-of-distribution | 0.8865 / 0.1241 | 0.9002 / 0.1279 | **0.8214** / 0.1228 | 0.8947 / **0.1106** |
| mean, all 8 | – | 0.7117 / 0.1033 | **0.6323** / **0.1000** | 0.6643 / 0.1030 |

Which to use: `agglo` is the better default (lower VI than `abiss` on all eight volumes, lower
error on Harris than SegNeuron); `basins` when neurons are merged through thin contacts (it
removes the large merges on MicronsB, but adds merges on Harris and FIB25-1). Neither reaches
SegNeuron on Ionsem.

How the settings were chosen: all thresholds on held-out training chunks only. `agglo` and
`basins` were added after the evaluation volumes had been scored with `abiss`, so their
columns are post-hoc; `abiss` is the pre-specified result.

VI is in the paper's convention (scikit-image 0.17.2: the current scikit-image VI times the
foreground fraction of the ground truth, [SegNeuron issue #8](https://github.com/yanchaoz/SegNeuron/issues/8));
the adapted Rand error is the same in both. `scripts/evaluate_emneuron.py` writes both versions
(`vi_old`, `vi`); in the current scikit-image VI the `abiss` means are 1.0326 (out-of-distribution)
and 0.8694 (all 8).

## Use the pretrained model

```bash
pip install -U huggingface_hub
hf download pytc/tutorial neuron_segneuron/banis_emneuron_200k.ckpt --local-dir checkpoints
```

**Zero-shot.** Put each volume in a directory as `<name>_img.h5` (one 3D dataset, ZYX,
uint8 EM, cell membranes dark). The model was trained at 6-9 nm in xy and 8-50 nm in z;
resample volumes far outside that range first.

```bash
python scripts/main.py --config tutorials/neuron_segneuron/banis+.yaml --mode test \
    --checkpoint checkpoints/neuron_segneuron/banis_emneuron_200k.ckpt \
    test.data.test.path=/path/to/my_volumes
# -> checkpoints/test_banis_emneuron_200k/<name>_img/raw_x1_ch0-1-2_weights-ema.h5
#    (3 x Z x Y x X float16 affinities, x/y/z, on the input grid)
python -m connectomics.decoding.abiss_runner --abiss-home /path/to/abiss \
    --input checkpoints/test_banis_emneuron_200k/<name>_img/raw_x1_ch0-1-2_weights-ema.h5 \
    --output <name>_seg.h5 --ws-high-threshold 0.9 --ws-low-threshold 0.05 \
    --ws-size-threshold 1000000000 --ws-dust-threshold 0 --ws-merge-function max \
    --ws-merge-threshold 0.6
```

ABISS: build [PytorchConnectomics/ABISS](https://github.com/PytorchConnectomics/ABISS) (`ws`
binary). The merge threshold is the one setting that matters: raise it (0.65-0.7) if neurons
are merged, lower it if they are split. For the reported numbers, add TTA16 (below).
`tutorials/neuron_segneuron/predict.sh` runs both steps; `DECODER=agglo` or `DECODER=basins`
selects the agglomeration decoders of the table.

**Fine-tune** on your labels (`<name>_img.h5` + `<name>_seg.h5` instance labels): the weights
load into a fresh optimizer and schedule.

```bash
python scripts/main.py --config tutorials/neuron_segneuron/banis+.yaml \
    --checkpoint checkpoints/neuron_segneuron/banis_emneuron_200k.ckpt --external-prefix model.model. \
    train.data.train.path=/path/to/train train.data.val.path=/path/to/val \
    train.optimization.max_steps=20000 train.optimization.scheduler.params.t_max=20000
```

`max_steps` and the scheduler's `t_max` must be equal.

## Reproduce

**Data.** Download EMNeuron (two RAR archives; ~11 GB extracted) and convert:

```bash
hf download yanchaoz/EMNeuron --repo-type dataset --revision 62df92b933550cc2a5d3849fe8543191c356f2d4 \
    --local-dir /path/to/EMNeuron
cd /path/to/EMNeuron && unrar x labeled.rar && unrar x valid.rar
python scripts/prepare_emneuron.py --src /path/to/EMNeuron --out /path/to/emneuron_pytc
```

`manifest.json` should have sha256
`058211d8718f49a3210e99de160dcc0ba5861cafa190f966a0f8d2abf4401d6c`. `tune/` holds one chunk
per source, held out from training; it is the only data used to choose decoder settings.

**Train** (4 GPUs, global batch 16, ~16.4 h on 4x L40S):

```bash
python scripts/main.py --config tutorials/neuron_segneuron/banis+.yaml \
    train.data.train.path=/path/to/emneuron_pytc/train train.data.val.path=/path/to/emneuron_pytc/tune
```

Use `checkpoints/step=00200000.ckpt` (`last.ckpt` is the best-validation checkpoint).

**Predict with TTA16** (flips x xy quarter-turns, averaged; one volume per launch):

```bash
python scripts/main.py --config tutorials/neuron_segneuron/banis+.yaml --mode test \
    --checkpoint <ckpt> test.data.test.path=<dir with one volume> test.system.num_gpus=4 \
    default.inference.test_time_augmentation.enabled=true \
    default.inference.test_time_augmentation.flip_axes=all \
    'default.inference.test_time_augmentation.rotation90_axes=[[1,2]]' \
    'default.inference.test_time_augmentation.rotate90_k=[0,1]' \
    default.inference.test_time_augmentation.patch_first_local=false
```

**Score** (`label.tif`, VI and adapted Rand error with SegNeuron's skimage calls; also sweeps
the ABISS grid):

```bash
python scripts/evaluate_emneuron.py --abiss-home /path/to/abiss --volume Harris \
    --pred <pred>.h5 --gt /path/to/EMNeuron/valid/OutofDistribution/Harris/label.tif --out scores.tsv
# the table's three decoders only: abiss and agglo, then basins
python scripts/evaluate_emneuron.py ... --ws-high 0.9 --ws-size 1000000000 --ws-merge-function max \
    --ws-merge-thresholds 0.6 --agglomerate 0.4
python scripts/evaluate_emneuron.py ... --ws-high 0.9 --ws-size 0 --ws-merge-function max \
    --ws-merge-thresholds 0.6 --agglomerate 0.35
```

Volumes thinner than the 32-section window (Ionsem, 30 sections) are padded to 32 sections
and returned padded; crop `[d // 2 : d // 2 + z]` along z before decoding.

**Padding and these numbers.** The results above use zero padding, which is what this checkpoint
saw in training (its 31-section Mira-scn chunks) and what PyTC did before PR #225. Current PyTC
pads by reflection (#225), so rerunning the commands here reproduces Ionsem 1.0098 / 0.1143 and
a mean over all 8 of 0.8695 / 0.1036 with `abiss` (current scikit-image VI; 0.9944 and 0.7119
as the paper's VI; `agglo` 0.9459 / 0.1091, `basins` 0.9418 / 0.1102); the other seven volumes
are unchanged. For this checkpoint
zero padding is the better match: on the held-out 31-section Mira-scn chunk, the two sections next
to the padded face score VI 1.37 with zero padding vs 1.53 with reflection (equal elsewhere).
To reproduce the table exactly, use PyTC before #225 (commit 7c8f145).

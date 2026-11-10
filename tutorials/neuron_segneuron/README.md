# EMNeuron neuron segmentation (SegNeuron's benchmark)

A pretrained 3D EM neuron segmentation model: BANIS+ (MedNeXt-L, affinities + ABISS
watershed) trained on [EMNeuron](https://huggingface.co/datasets/yanchaoz/EMNeuron), the
16-source EM database of [SegNeuron](https://github.com/yanchaoz/SegNeuron) (Zhang et al.,
MICCAI 2024). Use it zero-shot on a new EM volume or as the starting point for fine-tuning.

**Weights:** [`pytc/tutorial` / `neuron_segneuron/banis_emneuron_200k.ckpt`](https://huggingface.co/pytc/tutorial/tree/main/neuron_segneuron)
(EMA weights after 200k steps, 495 MB).

## Results: ours vs SegNeuron (published)

Variation of information (VI) / adapted Rand error against `label.tif` on SegNeuron's eight
evaluation volumes; lower is better. SegNeuron's numbers are Table 3 of the paper, which gives
values only for the out-of-distribution volumes. Ours: TTA16 and the ABISS setting chosen on
held-out training chunks only; each evaluation volume scored once.

| volume | split | SegNeuron (published) | ours |
|---|---|---|---|
| FAFB-1 | in-distribution | – | 0.9965 / 0.1793 |
| FAFB-2 | in-distribution | – | 0.7403 / 0.1827 |
| FIB25-1 | in-distribution | – | 0.5006 / 0.0095 |
| FIB25-2 | in-distribution | – | 0.3734 / 0.0159 |
| Kstr | in-distribution | – | 1.2464 / 0.0556 |
| Ionsem (vEM4) | out-of-distribution | 0.8655 / 0.1022 | FILL |
| MicronsB (Basil) | out-of-distribution | 0.7719 / 0.1531 | 0.9050 / 0.1724 |
| Harris | out-of-distribution | 1.0221 / 0.1170 | 1.1844 / 0.0987 |
| **mean, out-of-distribution** | | **0.8865 / 0.1241** | **FILL** |

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
```

Volumes thinner than the 32-section window (Ionsem, 30 sections) are padded by reflection and
returned at 32 sections; crop `[d // 2 : d // 2 + z]` along z before decoding.

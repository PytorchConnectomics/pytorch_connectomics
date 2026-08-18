# SNEMI3D neuron segmentation

Instance segmentation of neurites in the SNEMI3D challenge volume (anisotropic
30 × 6 × 6 nm, 100 × 1024 × 1024). The recipe is a modernization of
[DeepEM](https://github.com/seung-lab/DeepEM) / [Lee et al.
2017](https://arxiv.org/abs/1706.00120): learn a 12-channel affinity target
(3 nearest-neighbor edges + 9 long-range auxiliaries), agglomerate the
nearest-neighbor channels with waterz, and score Adapted Rand.

| YAML | Deep learning | Schedule | Decoding |
|---|---|---|---|
| `neuron_snemi.yaml` | MedNeXt-S/k3, 12-ch affinity (`aff12`), deep supervision, EMA | 200 ep × 1000 steps, batch 12/GPU, `accumulate_grad_batches=4` | waterz `aff85_his256` @ 0.5 |
| `neuron_snemi_efficient.yaml` | same, deep supervision **off** | 100 ep × 200 steps, no accumulation | same |
| `neuron_snemi_v1.yaml` | same as `_efficient` + explicit AdamW `lr=5e-4`, stronger elastic / motion-blur augmentation, checkpoints on train loss | 100 ep × 200 steps | same, 50 tuning trials |
| `neuron_snemi_sdt.yaml` | MedNeXt-S, 9-ch affinity + skeleton-aware SDT (`aff9_sdt`, 10 out-channels) | 200 ep × 1000 steps, batch 4/GPU | waterz @ 0.4, TTA off |
| `neuron_snemi_sdt_multitask.yaml` | same target through 4 heads (`aff_r1`/`aff_r5`/`aff_r9`/`sdt`) with uncertainty loss balancing | 200 ep × 1000 steps, batch 4/GPU | waterz @ 0.4 |

All five share the DeepEM data split (top 80 z-slices train, bottom 20
validation), `[32, 160, 160]` patches with `[16, 80, 80]` context padding, and
50 %-overlap sliding-window inference. The rest of this page covers
`neuron_snemi.yaml`; the siblings take the same commands with a different
`--config`.

Adapted Rand scores for these configs are not published yet — no pretrained
SNEMI3D checkpoint ships with the tutorial, so step 2 is not optional.

## 1. Get the data

SNEMI3D is published on Zenodo as
[record 7142003](https://zenodo.org/records/7142003)
(DOI [10.5281/zenodo.7142003](https://doi.org/10.5281/zenodo.7142003),
CC-BY-4.0, `snemi.zip`, 185.6 MiB, md5
`3d25a7025f66698f33c7850ace885939`):

```bash
mkdir -p datasets/SNEMI
curl -L 'https://zenodo.org/records/7142003/files/snemi.zip?download=1' -o /tmp/snemi.zip
unzip -j /tmp/snemi.zip -d datasets/SNEMI     # -j flattens the image/ and seg/ folders
```

That archive holds the three volumes the challenge released:

```text
datasets/SNEMI/
├── train-input.tif      # 100 × 1024 × 1024, 30 × 6 × 6 nm
├── train-labels.tif     # dense neurite instance labels
└── test-input.tif       # held-out volume
```

The challenge never published the **test** labels, so the Zenodo archive alone
cannot score `--mode test` or `--mode tune`. The PyTC mirror repacks the same
three volumes (byte-identical, same md5s) together with a `test-labels.h5` for
offline evaluation, and lands them flat in the layout the configs expect:

```bash
just download snemi     # 190.0 MiB from huggingface.co/pytc/tutorial
```

```text
datasets/SNEMI/
├── train-input.tif
├── train-labels.tif
├── test-input.tif
└── test-labels.h5       # 100 × 1024 × 1024 uint16, 333 instances — not part of the public release
```

The configs read `datasets/SNEMI/` relative to the repository root; edit the
`train.data.train` / `test.data.test` / `tune.data.val` blocks if you stage the
data elsewhere.

## 2. Run training

```bash
conda activate pytc
python scripts/main.py --config tutorials/neuron_snemi/neuron_snemi.yaml
```

`system.profile: all-gpu-cpu` fans out across every visible GPU. Override the
GPU count or the per-GPU batch as needed:

```bash
python scripts/main.py --config tutorials/neuron_snemi/neuron_snemi.yaml \
    system.num_gpus=4 data.dataloader.batch_size=6
```

The schedule is 200 epochs × 1000 steps with `accumulate_grad_batches=4`
(batch 12 per GPU), warmup-cosine LR, bf16, and EMA (`decay=0.999`,
`validate_with_ema: true`). Checkpoints follow `val_loss_total` (top-3) on the
bottom-20 z-slice validation split, and land under
`outputs/neuron_snemi/<timestamp>/checkpoints/`.

```bash
just tensorboard neuron_snemi
```

## 3. Inference, decoding, evaluation

```bash
python scripts/main.py --config tutorials/neuron_snemi/neuron_snemi.yaml \
    --mode test \
    --checkpoint outputs/neuron_snemi/<timestamp>/checkpoints/last.ckpt
```

In order:

1. **Inference** — 32 × 160 × 160 sliding window, 50 % overlap, bump blending,
   and 16× TTA (8 flip variants × 90° xy rotations after deduplication,
   combined with `ensemble_mode: min`). The
   `crop_pad: [15, 16, 79, 80, 79, 80]` undoes the `[16, 80, 80]` input padding
   *and* the destination-index shift that `affinity_mode: deepem` introduces, so
   the saved 12-channel affinity sits back on the original image support.
2. **Decoding** — keeps the nearest-neighbor channels (`select_channel: [0, 1, 2]`;
   the long-range edges are training auxiliaries only) and runs waterz with
   `merge_function: aff85_his256`, `thresholds: 0.5`,
   `aff_threshold: [0.1, 0.999]`, `channel_order: xyz`, and dust merge
   (`size=800`, `affinity=0.3`, `remove_size=600`).
3. **Evaluation** — Adapted Rand against `datasets/SNEMI/test-labels.h5`.

Segmentation and metrics land in the checkpoint's run directory under
`test_<ckpt tag>/` — `outputs/neuron_snemi/<timestamp>/test_last/` for
`last.ckpt`, `test_step=00050000/` for a step checkpoint. TTA is the main cost
knob here; disable it for a fast pass:

```bash
python scripts/main.py --config tutorials/neuron_snemi/neuron_snemi.yaml \
    --mode test --checkpoint <ckpt> \
    inference.test_time_augmentation.enabled=false
```

## 4. Tune the decoder

The waterz threshold and merge function dominate Adapted Rand. `--mode tune`
runs an Optuna TPE search (25 trials, 300 s each, study `snemi_waterz_tuning`)
over `merge_function`, `thresholds ∈ [0.1, 0.9]`, and both `aff_threshold`
bounds:

```bash
python scripts/main.py --config tutorials/neuron_snemi/neuron_snemi.yaml \
    --mode tune \
    --checkpoint outputs/neuron_snemi/<timestamp>/checkpoints/last.ckpt
```

Only decode + evaluate re-run per trial — the affinity prediction is computed
once and reused, so trials are fast. Use `--mode tune-test` to chain the
selected parameters straight into a test decode.

**SNEMI3D has no separate validation volume**, so `tune.data.val` points at the
test volume: the searched parameters are selected on the same data they are
reported on. Swap in the commented-out `train-input.tif` / `train-labels.tif`
lines under `tune.data.val` if you need a clean split.

## Notes

- The two SDT configs read `datasets/SNEMI/train-labels_skeleton.h5`, which is
  **not** in either archive — it is precomputed automatically on the first
  training run (`label_aux_type: skeleton`, see
  `connectomics/training/lightning/data_factory.py`) and cached beside the
  labels.
- `erosion: 1` on the label transform widens instance borders before the
  affinity target is built; it is part of the recipe, not a preprocessing
  detail to drop.

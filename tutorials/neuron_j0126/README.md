# j0126: conservative segmentation, morphology-based reconnecting

Turns the j0126 EM volume into a neuron segmentation in four steps: train an affinity
model, predict affinity, decode conservatively with ABISS, reconnect high-confidence
branches. The decode deliberately under-merges — a split is cheap to repair, a merge
corrupts two neurons — and step 4 repairs the splits. Step 4 is prediction-only: it reads
the segmentation, affinity, predicted morphology and an external nucleus manifest, never
evaluation skeletons, their lookup table, or an FFN segmentation.

The volume is **9 × 9 × 20 nm (x, y, z)**, i.e. `[20, 9, 9]` in the ZYX order every config
uses. Nothing here is 10 nm isotropic: `im_align_10nm.zarr` is a misnomer for a store that
is byte-identical to the public mip 0.

## Run it

One driver runs all four steps. It checks each step's output artifact before running the
step and skips the step when it is already complete, so re-running resumes a partial
pipeline instead of recomputing it.

```bash
python scripts/run_j0126.py --check                      # what exists, what is missing
python scripts/run_j0126.py --checkpoint ckpt/aff.ckpt   # run every missing step
python scripts/run_j0126.py --steps abiss,ec --dry-run   # print the commands only
python scripts/run_j0126.py --force infer                # rerun a step that looks complete
```

On a cluster, `--launcher slurm` wraps each step in `sbatch --wrap`, chains the steps with
`afterok`, and submits step 2 as an array:

```bash
python scripts/run_j0126.py --launcher slurm --num-shards 80 \
  --checkpoint ckpt/aff.ckpt \
  --slurm-infer "-p gpu --gres=gpu:1 -c 8 --mem 64G -t 8:00:00" \
  --slurm-abiss "-c 64 --mem 250G -t 24:00:00" \
  --slurm-ec    "-c 8 --mem 64G -t 12:00:00"
```

| # | step | config | command | complete when |
|---|---|---|---|---|
| 1 | train (optional) | `1_affinity_supervised.yaml` | `scripts/main.py --mode train` | a `.ckpt` under `<save_path>/*/checkpoints/` |
| 2 | infer | `1_affinity_zeroshot.yaml` | `scripts/main.py --mode test --checkpoint …` | every chunk listed in `*.h5.index.json` is on disk |
| 3 | abiss | `2_abiss.yaml` | `scripts/run_abiss_chunk.py` | `abiss/precomputed/seg/info` exists |
| 4 | ec | `3_merge.yaml` | `scripts/run_error_correction.py --stage all --num-tasks 1` | `error_correction_v7/error_correction_manifest.json` exists |

Edit **only** [params.yaml](params.yaml): repository checkout, dataset root, writeable
output root. Every config inherits it, so paths are never duplicated. Keep the algorithmic
thresholds in the step YAMLs unchanged when reproducing the reference recipe.

Whole-volume planning figures are in [RESOURCE.md](RESOURCE.md); reclaiming disk while the
run is in flight is in [CLEANUP.md](CLEANUP.md). A bit-exact replay of the two published
ABISS rows, with fail-closed input checks, is [reproduction/](reproduction/README.md) — a
separate, frozen code path, not this one.

## Step 0 — data

**Training data** (395 MB, not needed if you use an existing model) — 33 densely labelled subvolumes:

```bash
wget https://huggingface.co/datasets/pytc/zebrafinch-j0126/resolve/main/j0126-train-33vol.zip
unzip j0126-train-33vol.zip -d <dataset_root>/train/
```

Gives `im_raw/` + `seg_gt/` and the padded pair `im_raw_4-32-32/` + `seg_gt_4-32-32/`.
Training reads the padded pair: the pad is real EM context on the image side and `-1` on
the label side, so the loss ignores the border and no mask volume is needed.

**Testing data** — the EM volume to segment, from the public FFN mirror (Januszewski et al. 2018), uint8:

```
gs://j0126-nature-methods-data/GgwKmcKgrcoNxJccKuGIzRnQqfit9hnfK1ctZzNbnuU/rawdata_realigned
```

For one or two chunks, point the config straight at the bucket. For a whole-volume run,
download mip 0 once (5700 × 10913 × 10664 uint8 ≈ 660 GB):

```bash
python scripts/download_precompute.py \
  gs://j0126-nature-methods-data/GgwKmcKgrcoNxJccKuGIzRnQqfit9hnfK1ctZzNbnuU/rawdata_realigned \
  --out <dataset_root>/j0126_em.zarr --mip 0 --tile-xy 2048 --slab 64 \
  --shard-id "$SLURM_ARRAY_TASK_ID" --num-shards 8

# or one 1008^3 crop for a smoke test (~1 GB)
python scripts/download_precompute.py <same source> \
  --out /tmp/j0126_crop.zarr --mip 0 --bbox 2016 3024 5040 6048 5040 6048
```

Point `params.data.raw_em` at the array, e.g. `<dataset_root>/j0126_em.zarr/main`. Take
mip 0 and do not resample: it is the grid FFN published, the grid the evaluation skeletons
index, and the grid the reference runs used.

**Tissue mask** — FFN's `tissue_classification` layer in the same bucket is a 6-channel
probability volume at 18 × 18 × 20 nm; thresholding it gives the binary keep-mask
`NOT(blood vessel | myelin | out-of-bounds)`, and combining that with the border ring
gives step 4's `keep_mask`:

```bash
python dev/zebrafinch/build_ffn_tissue_mask.py \
  --output <dataset_root>/ffn_tissue_mask_18-18-20.zarr \
  --shard-id "$SLURM_ARRAY_TASK_ID" --num-shards 16

python dev/zebrafinch/build_unclipped_mask_region.py \
  --out <experiment_root>/tissue_border_keep_mask_full.zarr \
  --bbox-xyz 0 0 0 10664 10912 5700 --shard "$SLURM_ARRAY_TASK_ID" --nshard 40
```

The mask is half resolution in XY and 1:1 in Z; it is nearest-upsampled 2× in XY at use
time. Note what it borrows: it comes from FFN's own CNN and removes 15.64% of the volume,
which makes a "we beat FFN" comparison weaker than it looks. If that matters,
`dev/zebrafinch/build_bv_border_mask.py` builds the same kind of mask from a blood-vessel
volume we own, masks no myelin at all, and removes 1.38% — the decode then has to cope
with myelin on its own.

## Step 1 — train the affinity model (optional)

`1_affinity_supervised.yaml` trains MedNeXt-L/k3 from scratch for 200k steps on the dense
GT cubes, 25 for training and 8 held out, at roughly four GPU-days on 4 GPUs:

```bash
python scripts/main.py --config tutorials/neuron_j0126/1_affinity_supervised.yaml --mode train
```

Skip this and pass `--checkpoint` instead if you already have one:

```bash
hf download pytc/j0126 affinity_scratch_48x96x96.ckpt --local-dir ckpt/
```

That checkpoint has seen labelled j0126 tissue, so a run starting from it is **not**
ground-truth-free; it is the supervised upper reference. It also predates the held-out
split — it trained on all 33 cubes and validated on 3 that were also in training, so its
validation curve was not a generalization estimate. Every number in the results table
comes from it. Retraining with the current YAML gives an honest held-out curve and a
slightly different model.

## Step 2 — predict affinity

```bash
python scripts/main.py --config tutorials/neuron_j0126/1_affinity_zeroshot.yaml \
  --mode test --checkpoint /path/to/affinity.ckpt
```

Output: chunked float16, three-channel affinity under `output_root/affinity`. The full
volume is 726 chunks of 1008³ with a 72-voxel halo; shard it with
`--shard-id`/`--num-shards`, one GPU per shard, no `torch.distributed`.

The config must match the window its checkpoint was trained at. MedNeXt normalizes without
running statistics, so the forward pass depends on the window extent and a mismatch
silently inverts the trained anisotropy. `1_affinity_zeroshot.yaml` runs `[144, 144, 144]`;
a checkpoint trained on the j0126 cubes needs `1_affinity_supervised.yaml` instead
(`[48, 96, 96]`, output `affinity_arm0_96/`, so step 3's `source_affinity_h5` must be
repointed there).

## Step 3 — ABISS decode

```bash
python scripts/run_abiss_chunk.py --config tutorials/neuron_j0126/2_abiss.yaml --prepare-only
python scripts/run_abiss_chunk.py --config tutorials/neuron_j0126/2_abiss.yaml
```

ABISS uses every CPU the scheduler grants the process, so submit **one shared-memory job**
(`--cpus-per-task=64` is a good start), never a job array — independent copies would race
on the same hierarchy layers. For the fastest whole-volume decode, shard each hierarchy
layer across nodes and wait for that layer before launching the next: the recorded 40-node
run took 3.75 h, and an 80/24/16/14/8/1-shard layout is projected at ~1.9 h.

## Step 4 — morphology error correction

Step 4 builds skeletons for large predicted segments, evaluates every sufficiently
confident contact, and accepts only hard-gated branch continuations. It protects external
nucleus identities and never joins two different ones.

`--stage all` runs the thirteen stages in order in one process, which is all a small volume
needs. It is serial, hence `--num-tasks 1`; add `--dry-run` to print the plan:

```bash
python scripts/run_error_correction.py \
  --config tutorials/neuron_j0126/3_merge.yaml --stage all --num-tasks 1
```

Only three stages are chunk-parallel. For the whole volume, submit `skeletonize`,
`contacts` and `postprocess` as Slurm arrays at the task count configured in
`3_merge.yaml` (80 in the reference run), and run the rest serially in between:

```bash
CFG=tutorials/neuron_j0126/3_merge.yaml
ARRAY="--task-id $SLURM_ARRAY_TASK_ID --num-tasks 80"

python scripts/run_error_correction.py --config "$CFG" --stage sizes
python scripts/run_error_correction.py --config "$CFG" --stage skeletonize $ARRAY   # array
python scripts/run_error_correction.py --config "$CFG" --stage skeletons
python scripts/run_error_correction.py --config "$CFG" --stage contacts $ARRAY      # array
python scripts/run_error_correction.py --config "$CFG" --stage contact_graph
python scripts/run_error_correction.py --config "$CFG" --stage candidates
python scripts/run_error_correction.py --config "$CFG" --stage junction_scope
python scripts/run_error_correction.py --config "$CFG" --stage junction_features
python scripts/run_error_correction.py --config "$CFG" --stage boundary
python scripts/run_error_correction.py --config "$CFG" --stage resolve
python scripts/run_error_correction.py --config "$CFG" --stage prepare_output
python scripts/run_error_correction.py --config "$CFG" --stage postprocess $ARRAY   # array
python scripts/run_error_correction.py --config "$CFG" --stage verify
```

That is the order `--stage all` executes, so the two forms agree.

Stages are restartable: completed chunk artifacts are reused. For a one-core smoke test,
append `--max-owned-chunks 1` to an array-stage command.

`3_merge.yaml`'s five input paths are pinned to the frozen reference run. Repoint all five
together when applying the method to your own decode; `run_j0126.py --check` reports which
of them are missing.

Use `erosion_radius_zyx: [0, 0, 0]` for morphology linking alone and `[1, 1, 1]` for the
strict-mt=0 cleanup. The scratch run freezes 749 branch unions. Evaluation is deliberately
outside the EC config and should be run only after the proposal has been frozen.

## Results

The table separates the affinity source, conservative decoder, and optional correction steps. The reported full-volume ablation uses the scratch affinity; the synthetic row is the zero-shot transfer of an NISB-trained checkpoint to this volume, evaluated on the same skeletons and metric.

| Affinity | Decoding | Error correction | NERL mt=0 ↑ | NERL mt=5 ↑ | VOI split ↓ | VOI merge ↓ | VOI ↓ |
|---|---|---|---:|---:|---:|---:|---:|
| **FFN reference** | — | — | **0.526** | 0.538 | **1.729** | 0.127 | **1.856** |
| scratch | ABISS, exclusion mask | — | 0.268 | 0.470 | 2.542 | 0.042 | 2.584 |
| scratch | + nucleus instance certificate | — | 0.287 | 0.482 | 2.543 | 0.019 | 2.562 |
| scratch | + nucleus instance certificate | morphology-guided branch linking | 0.301 | 0.539 | 2.355 | 0.019 | 2.374 |
| scratch | + nucleus instance certificate | + 3×3×3 inter-object erosion | 0.441 | 0.528 | 2.312 | 0.128 | 2.440 |
| synthetic | + nucleus instance certificate | — | 0.314 | 0.383 | 3.311 | 0.020 | 3.331 |

`mt=5` is the five-node merge-tolerance NERL. The 3×3×3 erosion is a strict-mt=0 cleanup, not the best operating point for mt=5 NERL or VOI sum.

On the synthetic affinity the nucleus certificate is **inert**: its scan finds 0 multi-nucleus watershed objects, so it publishes zero repairs and that row is also the exclusion-mask baseline. Whether the certificate has anything to correct is a property of the watershed, not of the nucleus mask — the scratch affinity fuses 8 soma pairs at the watershed stage and this one fuses none. An independent replay of the same affinity scores 0.383 at the same tolerance, so the no-op is confirmed rather than assumed. The synthetic row trails scratch on NERL and VOI split; it is a zero-shot transfer result, not a tuned one.

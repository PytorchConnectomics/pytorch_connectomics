# j0126: conservative segmentation, morphology-based reconnecting

This tutorial turns the j0126 EM volume (9 x 9 x 20 nm) into a neuron segmentation in three steps:

0. Download data
1. Predict voxel affinities.
2. Decode them conservatively with ABISS, preferring splits to false merges.
3. Reconnect only high-confidence, morphologically continuous branches.

For the frozen PyTC2 Zebrafinch replay, including fail-closed input checks and
non-interactive Slurm submission, follow the
[reproduction guide](reproduction/README.md).

The last step is prediction-only at runtime. It uses the segmentation, affinities, predicted morphology, and an external nucleus-instance manifest; it does not read evaluation skeletons, their lookup table, or an FFN segmentation.

## Before running

Edit **only** [params.yaml](params.yaml). It contains the repository checkout, dataset root, writeable output root, and the existing artifacts required to replay the frozen EC recipe. Every step inherits this file, so paths are not duplicated across the workflow YAMLs. Keep the algorithmic thresholds in the step YAMLs unchanged when reproducing the reference recipe.

The zero-shot affinity path needs a NISB-trained checkpoint. The supervised affinity YAML is included as a target-domain reference only: it uses j0126 dense labels, so it is not part of the zero-shot pipeline; its trained checkpoint can be downloaded instead of retrained (see [Step 1](#supervised-reference-affinity)).

Whole-volume planning figures live in [RESOURCE.md](RESOURCE.md), and the staged storage cleanup that keeps the peak down is in [CLEANUP.md](CLEANUP.md).


## Step 0 — get the data

**Training data** (395 MB) — 33 densely labelled subvolumes (No need if use existing models):

```bash
wget https://huggingface.co/datasets/pytc/zebrafinch-j0126/resolve/main/j0126-train-33vol.zip
unzip j0126-train-33vol.zip -d <dataset_root>/train/
```

That gives `im_raw/` + `seg_gt/` and the padded pair `im_raw_4-32-32/` +
`seg_gt_4-32-32/`, which is what [1_affinity_supervised.yaml](1_affinity_supervised.yaml)
reads: the padding is real EM context on the image side and `-1` on the label
side, so the loss ignores the border and no mask volume is needed.

**Testing data** — the public FFN mirror (Januszewski et al. 2018), uint8 at
9 × 9 × 20 nm (x, y, z), i.e. `[20, 9, 9]` in the ZYX order the configs use:

```
gs://j0126-nature-methods-data/GgwKmcKgrcoNxJccKuGIzRnQqfit9hnfK1ctZzNbnuU/rawdata_realigned
```

Running one or two chunks? Point the config straight at the bucket and skip the
copy. For a whole-volume run, or anything you will read more than a few times,
download mip 0 once into a local zarr:

```bash
# whole volume: 5700 x 10913 x 10664 uint8 = ~660 GB; shard it across an array job
# (the 5.3 TB uint64 FFN segmentation layer took ~2 h at 8 shards, so this is less)
python scripts/download_precompute.py \
  gs://j0126-nature-methods-data/GgwKmcKgrcoNxJccKuGIzRnQqfit9hnfK1ctZzNbnuU/rawdata_realigned \
  --out <dataset_root>/j0126_em.zarr --mip 0 --tile-xy 2048 --slab 64 \
  --shard-id "$SLURM_ARRAY_TASK_ID" --num-shards 8

# or one 1008^3 crop for a smoke test (~1 GB)
python scripts/download_precompute.py <same source> \
  --out /tmp/j0126_crop.zarr --mip 0 --bbox 2016 3024 5040 6048 5040 6048
```

Then point `params.data.raw_em` at the array, e.g. `<dataset_root>/j0126_em.zarr/main`.


## Step 1 — affinity prediction

- Option 1: Run zero-shot inference with an NISB checkpoint:

```bash
python scripts/main.py --config tutorials/neuron_j0126/1_affinity_zeroshot.yaml \
  --mode test --checkpoint /path/to/nisb_affinity.ckpt
```

For the full volume, run independent one-GPU shards:

```bash
python scripts/main.py --config tutorials/neuron_j0126/1_affinity_zeroshot.yaml \
  --mode test --checkpoint /path/to/nisb_affinity.ckpt \
  --shard-id "$SLURM_ARRAY_TASK_ID" --num-shards 80
```

The output is chunked float16, three-channel affinity under `output_root/affinity` after resolving `params.yaml`.

- Option 2: Supervised training

`1_affinity_supervised.yaml` is the target-domain reference: MedNeXt-L/k3 trained from scratch for 200k steps on the j0126 dense-GT cubes, 25 for training and 8 held out for validation. It has seen labelled j0126 tissue, so any run that starts here is not ground-truth-free.

Training it costs roughly four GPU-days. Download the reference checkpoint instead:

```bash
hf download pytc/j0126 affinity_scratch_48x96x96.ckpt --local-dir ckpt/

python scripts/main.py --config tutorials/neuron_j0126/1_affinity_supervised.yaml \
  --mode test --checkpoint ckpt/affinity_scratch_48x96x96.ckpt
```

The released checkpoint predates the held-out split: it trained on all 33 cubes and validated on a 3-cube subset that was also in training, so its validation curve was not a generalization estimate. Every number in the results table comes from that checkpoint. Retraining with the current YAML gives an honest held-out curve and a slightly different model — it does not reproduce the released weights bit for bit.

It must be inferred at its native `[48, 96, 96]` window, which the YAML already sets: MedNeXt normalizes without running statistics, so the forward pass depends on the window extent, and the zero-shot config's `[144, 144, 144]` would invert the trained Z-thin anisotropy. Its affinity lands under `output_root/affinity_arm0_96`, so step 2's `source_affinity_h5` has to be repointed there.

## Step 2 — conservative ABISS decode

First inspect the planned ABISS commands:

```bash
python scripts/run_abiss_chunk.py --config tutorials/neuron_j0126/2_abiss.yaml --prepare-only
```

Then run them:

```bash
python scripts/run_abiss_chunk.py --config tutorials/neuron_j0126/2_abiss.yaml
```

For a small or single-node run, use one shared-memory CPU job: ABISS uses the CPUs the scheduler grants the process. For Slurm, a good starting point is:

```bash
sbatch --cpus-per-task=64 --wrap='python scripts/run_abiss_chunk.py \
  --config tutorials/neuron_j0126/2_abiss.yaml'
```

For the fastest full-volume decode, shard each hierarchy layer across nodes, wait for that layer to finish, then launch the next layer. The recorded 40-node run took 3.75 h; a 80/24/16/14/8/1-shard layout is projected at ~1.9 h. Do not launch independent copies of the complete decoder: they would race on the same layers. The fixed watershed and agglomeration settings intentionally under-merge, leaving recoverable fragments for step 3 instead of welding uncertain neurons.

## Step 3 — morphology-based error correction

Step 3 builds skeletons for large predicted segments, evaluates every sufficiently confident contact, and accepts only hard-gated branch continuations. It protects external nucleus identities and never joins two different identities.

Preview the complete plan first:

```bash
python scripts/run_error_correction.py \
  --config tutorials/neuron_j0126/3_merge.yaml --stage all --num-tasks 1 --dry-run
```

For the full volume, run the array stages with the same task count configured in `3_merge.yaml` (80 in the reference run):

```bash
CFG=tutorials/neuron_j0126/3_merge.yaml

python scripts/run_error_correction.py --config "$CFG" --stage sizes
python scripts/run_error_correction.py --config "$CFG" --stage skeletonize \
  --task-id "$SLURM_ARRAY_TASK_ID" --num-tasks 80
python scripts/run_error_correction.py --config "$CFG" --stage contacts \
  --task-id "$SLURM_ARRAY_TASK_ID" --num-tasks 80

python scripts/run_error_correction.py --config "$CFG" --stage skeletons
python scripts/run_error_correction.py --config "$CFG" --stage contact_graph
python scripts/run_error_correction.py --config "$CFG" --stage candidates
python scripts/run_error_correction.py --config "$CFG" --stage junction_scope
python scripts/run_error_correction.py --config "$CFG" --stage junction_features
python scripts/run_error_correction.py --config "$CFG" --stage boundary
python scripts/run_error_correction.py --config "$CFG" --stage resolve
python scripts/run_error_correction.py --config "$CFG" --stage prepare_output

python scripts/run_error_correction.py --config "$CFG" --stage postprocess \
  --task-id "$SLURM_ARRAY_TASK_ID" --num-tasks 80
python scripts/run_error_correction.py --config "$CFG" --stage verify
```

Stages are restartable: completed chunk artifacts are reused. For a one-core smoke test, append `--max-owned-chunks 1` to an array-stage command.

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

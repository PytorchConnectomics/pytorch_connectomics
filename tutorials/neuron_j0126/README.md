# j0126: conservative segmentation, morphology-based reconnecting

Turns the j0126 EM volume into a neuron segmentation in four steps: train an affinity
model, predict affinity, decode conservatively with ABISS, reconnect high-confidence
branches. The decode deliberately under-merges — a split is cheap to repair, a merge
corrupts two neurons — and step 4 repairs the splits from the segmentation, affinity,
predicted morphology and an external nucleus manifest, never from ground truth.

The volume is **9 × 9 × 20 nm (x, y, z)** = `[20, 9, 9]` ZYX throughout — the native FFN
mip 0 grid. Nothing here is 10 nm isotropic, and nothing is resampled.

## 1. Download the data

**Training data** (395 MB, skip if you use an existing model) — 33 labelled subvolumes:

```bash
wget https://huggingface.co/datasets/pytc/zebrafinch-j0126/resolve/main/j0126-train-33vol.zip
unzip j0126-train-33vol.zip -d <dataset_root>/train/
```

Training reads the padded pair `im_raw_4-32-32/` + `seg_gt_4-32-32/`: the pad is real EM
context on the image and `-1` on the label, so the loss ignores the border.

**Testing data** — the EM volume, from the public FFN mirror (Januszewski et al. 2018):

```bash
SRC=gs://j0126-nature-methods-data/GgwKmcKgrcoNxJccKuGIzRnQqfit9hnfK1ctZzNbnuU/rawdata_realigned

# whole volume, ~660 GB uint8; or add --bbox z0 z1 y0 y1 x0 x1 for one crop
python scripts/download_precompute.py "$SRC" --out <dataset_root>/j0126_em.zarr \
  --mip 0 --tile-xy 2048 --slab 64 --shard-id "$SLURM_ARRAY_TASK_ID" --num-shards 8
```

Point `params.data.raw_em` at the array (`…/j0126_em.zarr/main`), or straight at the bucket
for a chunk or two. Take mip 0 and do not resample: it is the grid FFN published, the grid
the evaluation skeletons index, and the grid the reference runs used.

**Tissue mask** — FFN's `tissue_classification` layer thresholded to
`NOT(blood vessel | myelin | out-of-bounds)`, then combined with the border ring into
step 4's `keep_mask`:

```bash
python dev/zebrafinch/build_ffn_tissue_mask.py \
  --output <dataset_root>/ffn_tissue_mask_18-18-20.zarr \
  --shard-id "$SLURM_ARRAY_TASK_ID" --num-shards 16

python dev/zebrafinch/build_unclipped_mask_region.py \
  --out <dataset_root>/tissue_border_keep_mask_full.zarr \
  --bbox-xyz 0 0 0 10664 10912 5700 --shard "$SLURM_ARRAY_TASK_ID" --nshard 40
```

It comes from FFN's own CNN and removes 15.64% of the volume, which weakens a "we beat
FFN" comparison. `dev/zebrafinch/build_bv_border_mask.py` is the alternative built from a
vessel volume we own: no myelin masked, 1.38% removed.

## 2. Run the pipeline

Edit **only** [params.yaml](params.yaml) — repository, dataset root, writeable output
root. Every config inherits it. Then one driver runs all four steps.

On a single machine:

```bash
python scripts/run_j0126.py --checkpoint ckpt/aff.ckpt
```

On a Slurm cluster, where each step becomes an `sbatch --wrap` job chained with `afterok`
and step 2 becomes an array of `--num-shards` one-GPU jobs:

```bash
python scripts/run_j0126.py --launcher slurm --num-shards 80 \
  --checkpoint ckpt/aff.ckpt \
  --slurm-infer "-p gpu --gres=gpu:1 -c 8 --mem 64G -t 8:00:00" \
  --slurm-abiss "-c 64 --mem 250G -t 24:00:00" \
  --slurm-ec    "-c 8 --mem 64G -t 12:00:00"
```

| # | step | config | complete when |
|---|---|---|---|
| 1 | train (optional) | `1_train.yaml` | a `.ckpt` under `<save_path>/*/checkpoints/` |
| 2 | infer | `2_infer.yaml` | every chunk in `*.h5.index.json` is on disk |
| 3 | abiss | `3_abiss.yaml` | `abiss/precomputed/seg/info` exists |
| 4 | ec | `4_error_correction.yaml` | `error_correction_manifest.json` exists |

Keep the step YAMLs' thresholds unchanged to reproduce the reference recipe. Planning
figures: [RESOURCE.md](RESOURCE.md). Disk cleanup mid-run: [CLEANUP.md](CLEANUP.md).

## Resuming, inspecting, rerunning

Each step checks the artifact in the table above before it runs and skips the step when it
is already there, so the same command resumes a partial pipeline instead of recomputing
it. Step 2 resumes per chunk.

```bash
python scripts/run_j0126.py --check                      # what exists, what is missing
python scripts/run_j0126.py --steps abiss,ec --dry-run   # print the commands only
python scripts/run_j0126.py --steps infer                # run one step
python scripts/run_j0126.py --force infer                # rerun a step that looks complete
```

`--check` prints every input and output path with its status, including the affinity chunk
store it found for step 2 — which is the path `4_error_correction.yaml` needs.

## Step details

What the driver runs at each step, and the constraints that matter if you adapt it.

### Step 1 — train the affinity model (optional)

`1_train.yaml` trains MedNeXt-L/k3 from scratch for 200k steps on the dense GT cubes,
25 train / 8 held out, at roughly four GPU-days on 4 GPUs. Or download one and pass
`--checkpoint`:

```bash
hf download pytc/j0126 affinity_scratch_48x96x96.ckpt --local-dir ckpt/
```

That checkpoint has seen labelled j0126 tissue, so a run from it is **not**
ground-truth-free — it is the supervised upper reference. It also predates the held-out
split (all 33 cubes in training, 3 of them reused as validation), and every number in the
results table comes from it, so retraining gives an honest curve and a slightly different
model.

### Step 2 — predict affinity

Output is chunked float16, three-channel affinity under `output_root/affinity`; the full
volume is 726 chunks of 1008³ with a 72-voxel halo, one GPU per shard.

The config must match the window its checkpoint was trained at — MedNeXt normalizes
without running statistics, so the forward pass depends on the window extent and a
mismatch silently inverts the trained anisotropy. `2_infer.yaml` runs `[144, 144, 144]`;
a j0126-trained checkpoint needs `1_train.yaml` instead (`[48, 96, 96]`, output
`affinity_arm0_96/`, so step 3's `source_affinity_h5` must be repointed there).

### Step 3 — ABISS decode

ABISS is a separate C++ dependency, pinned to the commit the reference decode used:

```bash
git clone https://github.com/PytorchConnectomics/ABISS.git lib/abiss
git -C lib/abiss checkout 452efa5f87f9d3cb241891ee44010d966a33b316

cmake -S lib/abiss -B lib/abiss/build -DCMAKE_BUILD_TYPE=Release \
  -DBOOST_ROOT="$CONDA_PREFIX" -DEXTRACT_SIZE=ON -DBUILD_TESTING=ON
cmake --build lib/abiss/build --parallel 8
ctest --test-dir lib/abiss/build --output-on-failure
```

`EXTRACT_SIZE=ON` is not optional: without it `acme` writes empty supervoxel-size files and
`agg` fails on an incomplete RAG. Use Boost 1.82 and oneTBB from the same conda
environment — legacy `libtbb.so.2` or Boost 1.85 fail in mean-edge agglomeration.

ABISS uses every CPU granted to the process, so submit **one shared-memory job**
(`--cpus-per-task=64` is a good start), never a job array: independent copies race on the
same hierarchy layers. The recorded 40-node run took 3.75 h.

### Step 4 — morphology error correction

Builds skeletons for large segments, evaluates every sufficiently confident contact, and
accepts only hard-gated branch continuations. It protects external nucleus identities and
never joins two different ones. `--stage all` runs the thirteen stages in order:

```bash
python scripts/run_error_correction.py \
  --config tutorials/neuron_j0126/4_error_correction.yaml --stage all --num-tasks 1
```

Whole-volume runs shard `skeletonize`, `contacts` and `postprocess` as Slurm arrays at the
config's `task_count`, with the serial stages in between; pass `--stage <name>` with
`--task-id`/`--num-tasks` for those. Stages are restartable — completed chunk artifacts are
reused.

`4_error_correction.yaml`'s five input paths are pinned to the frozen reference run;
repoint all five together for your own decode. `erosion_radius_zyx` is `[0, 0, 0]` for
morphology linking alone and `[1, 1, 1]` for the strict-mt=0 cleanup.

## Results

The affinity source, conservative decoder, and optional correction steps are separated.
The scratch rows use the supervised affinity; the synthetic row is a zero-shot NISB
checkpoint on the same skeletons and metric.

| Affinity | Decoding | Error correction | NERL mt=0 ↑ | NERL mt=5 ↑ | VOI split ↓ | VOI merge ↓ | VOI ↓ |
|---|---|---|---:|---:|---:|---:|---:|
| **FFN reference** | — | — | **0.526** | 0.538 | **1.729** | 0.127 | **1.856** |
| scratch | ABISS, exclusion mask | — | 0.268 | 0.470 | 2.542 | 0.042 | 2.584 |
| scratch | + nucleus instance certificate | — | 0.287 | 0.482 | 2.543 | 0.019 | 2.562 |
| scratch | + nucleus instance certificate | morphology-guided branch linking | 0.301 | 0.539 | 2.355 | 0.019 | 2.374 |
| scratch | + nucleus instance certificate | + 3×3×3 inter-object erosion | 0.441 | 0.528 | 2.312 | 0.128 | 2.440 |
| synthetic | + nucleus instance certificate | — | 0.314 | 0.383 | 3.311 | 0.020 | 3.331 |

`mt=5` is the five-node merge-tolerance NERL. The 3×3×3 erosion is a strict-mt=0 cleanup,
not the best operating point for mt=5 NERL or VOI sum. On the synthetic affinity the
nucleus certificate is **inert** — its scan finds 0 multi-nucleus watershed objects, so
that row is also the exclusion-mask baseline; whether the certificate has anything to
correct is a property of the watershed, not of the nucleus mask.

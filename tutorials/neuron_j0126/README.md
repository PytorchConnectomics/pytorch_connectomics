# j0126: conservative segmentation, morphology-based reconnecting

Turns the j0126 EM volume into a neuron segmentation in four steps: train an affinity
model, predict affinity, decode conservatively with ABISS, reconnect high-confidence
branches. The decode deliberately under-merges — a split is cheap to repair, a merge
corrupts two neurons — and step 4 repairs the splits from the segmentation, affinity,
predicted morphology and an external nucleus manifest, never from ground truth.

The volume is **9 × 9 × 20 nm (x, y, z)** = `[20, 9, 9]` ZYX throughout — the native FFN
mip 0 grid.

## Run it

Edit **[params.yaml](params.yaml)** — three paths, `train: true` or `false`, and the Slurm
partition / CPUs / memory / walltime for each step. Then:

```bash
python scripts/run_j0126.py
```

That is the whole tutorial. It downloads the EM volume and the model, builds the exclusion
mask, trains if you asked it to, predicts affinity, decodes with ABISS, and runs error
correction. With `cluster.launcher: slurm` each step is one `sbatch`, chained with
`--dependency=afterok`, so the command queues the pipeline and returns; with `local` the
steps run in the foreground.

Before committing a cluster to 660 GB, set `download.em_bbox: [2900, 3908, 5000, 6008,
5000, 6008]` in params.yaml and run the same command: one 1008³ chunk, a few minutes, ~1 GB,
every step exercised.

```bash
python scripts/run_j0126.py --check              # what exists, what is missing
python scripts/run_j0126.py --dry-run            # print the commands only
python scripts/run_j0126.py --steps infer,abiss  # run part of it
python scripts/run_j0126.py --force abiss        # rerun a step that looks complete
```

Each step declares the artifact that proves it finished, checks it before running, and
skips the step when it is there, so the same command resumes a partial pipeline. Step 2
resumes per chunk; the download and mask steps resume per shard.

The one prerequisite `run_j0126.py` will not install for you is ABISS, a C++ dependency
pinned to the commit the reference decode used:

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

## What each step does

| # | step | config | complete when |
|---|---|---|---|
| 0 | download + exclusion mask | `params.yaml` | per-shard `.done.<id>` sentinels, progress sidecars |
| 1 | train (optional) | `1_train.yaml` | a `.ckpt` under `<save_path>/*/checkpoints/` |
| 2 | infer | `2_infer.yaml` | every chunk in `*.h5.index.json` is on disk |
| 3 | abiss | `3_abiss.yaml` | `abiss/precomputed/seg/info` exists |
| 4 | ec | `4_error_correction.yaml` | `error_correction_manifest.json` exists |

Keep the step YAMLs' thresholds unchanged to reproduce the reference recipe. Planning
figures: [RESOURCE.md](RESOURCE.md). Disk cleanup mid-run: [CLEANUP.md](CLEANUP.md).

### Step 0 — data and exclusion mask

The EM volume comes from the public FFN mirror (Januszewski et al. 2018) at native mip 0.
Do not resample: it is the grid FFN published, the grid the evaluation skeletons index, and
the grid the reference runs used.

The exclusion mask is FFN's own `tissue_classification` thresholded to
`NOT(blood vessel | myelin | out-of-bounds)`, combined with the 0/255 border ring of the
aligned EM. It removes 15.64% of the volume, and because it comes from FFN's CNN it weakens
a "we beat FFN" comparison; `dev/zebrafinch/build_bv_border_mask.py` is the alternative
built from a vessel volume we own (no myelin masked, 1.38% removed).

Training data is 33 densely labelled subvolumes, fetched only when `train.enabled` is true.
Training reads the padded pair `im_raw_4-32-32/` + `seg_gt_4-32-32/`: the pad is real EM
context on the image and `-1` on the label, so the loss ignores the border.

### Step 1 — train the affinity model (optional)

`train.enabled: false` downloads `pytc/vEM j0126/affinity_scratch_48x96x96.ckpt` instead.
That checkpoint has seen labelled j0126 tissue, so a run from it is **not**
ground-truth-free — it is the supervised upper reference. It also predates the held-out
split (all 33 cubes in training, 3 of them reused as validation), and every number in the
results table comes from it, so retraining gives an honest curve and a slightly different
model.

`train.enabled: true` trains MedNeXt-L/k3 from scratch for 200k steps on the dense GT
cubes, 25 train / 8 held out, at roughly four GPU-days on 4 GPUs, and infers from whatever
checkpoint that run produced.

### Step 2 — predict affinity

Output is chunked float16, three-channel affinity under `output_root/affinity`; the full
volume is 726 chunks of 1008³ with a 72-voxel halo, one GPU per shard.

`inference.window_size` must match the window its checkpoint was trained at — MedNeXt
normalizes without running statistics, so the forward pass depends on the window extent and
a mismatch silently inverts the trained anisotropy. `[48, 96, 96]` is the j0126 checkpoint
and anything `train.enabled: true` produces; `[144, 144, 144]` is a NISB-trained zero-shot
checkpoint. Any value must be a multiple of MedNeXt's 16-voxel stride and its half-overlap
stride must divide 1008.

### Step 3 — ABISS decode

Before ABISS runs, the driver writes `affinity/affinity.h5` as an HDF5 **virtual dataset**
over step 2's chunk store — no voxel is copied, which is the only option when the affinity
is ~4 TB, and it reads through h5py exactly like a stitched file. `affinity.h5.chunks` and
`affinity.h5.index.json` are symlinked next to it, so step 4 has a stable name for a store
whose real path is timestamped and checkpoint-named.

ABISS uses every CPU granted to the process, so this is **one shared-memory job**
(`abiss.cpus: 64` is a good start), never a job array: independent copies race on the same
hierarchy layers. The recorded 40-node run took 3.75 h.

`AGG_THRESHOLD 0.20` with `WS_HIGH 0.9` / `WS_LOW 0.1` reproduces the reference
segmentation. The pipeline deliberately under-agglomerates here and repairs the splits in
step 4, whose grow round is provably merge-safe: it only ever assigns a fragment to a host,
never welds two segments. Every ambiguous decision is better deferred than taken here.

Setting `data.nucleus_volume` turns on competitive nucleus growth and writes the identity
manifest step 4 uses as a firewall. Left empty — the default, because no nucleus volume
ships with this tutorial — the run completes without that protection, and the driver says
so. That is the difference between the second and third rows of the table below.

### Step 4 — morphology error correction

Builds skeletons for large segments, evaluates every sufficiently confident contact, and
accepts only hard-gated branch continuations. It protects external nucleus identities and
never joins two different ones. `erosion_radius_zyx` is `[0, 0, 0]` for morphology linking
alone and `[1, 1, 1]` for the strict-mt=0 cleanup.

The driver runs all thirteen stages in one task. Whole-volume runs can instead shard
`skeletonize`, `contacts` and `postprocess` as Slurm arrays at the config's `task_count`,
with the serial stages in between:

```bash
python scripts/run_error_correction.py --config tutorials/neuron_j0126/4_error_correction.yaml \
    --stage skeletonize --task-id $SLURM_ARRAY_TASK_ID --num-tasks 80
```

Stages are restartable — completed chunk artifacts are reused.

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

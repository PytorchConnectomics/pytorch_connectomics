# j0126: conservative segmentation, morphology-based reconnecting

Turns the j0126 EM volume into a neuron segmentation in four steps: train an affinity
model, predict affinity, decode conservatively with ABISS, reconnect high-confidence
branches. The decode deliberately under-merges — a split is cheap to repair, a merge
corrupts two neurons — and step 4 repairs the splits from the segmentation, affinity,
predicted morphology and an external nucleus manifest, never from ground truth.

The volume is **9 × 9 × 20 nm (x, y, z)** = `[20, 9, 9]` ZYX throughout — the native FFN
mip 0 grid.

## Run it

Read the [reproduction status](#reproduction-status) below before submitting. The current
driver includes several fixes from the reproduction report, but is not yet a validated
fresh-machine, crop-to-score workflow.

Edit **[params.yaml](params.yaml)** — three paths, `train.enabled: true` or `false`, and the Slurm
partition / CPUs / memory / walltime for each step. Then:

```bash
python scripts/run_j0126.py
```

The driver downloads the EM volume and the model, builds the exclusion
mask, trains if you asked it to, predicts affinity, decodes with ABISS, and runs error
correction, then evaluates the result. With `cluster.launcher: slurm` jobs are chained with
`--dependency=afterok`, so the command queues the pipeline and returns; with `local` the
steps run in the foreground.

Keep `download.em_bbox: []` for this workflow. The reported 1008³ crop
`[2900, 3908, 5000, 6008, 5000, 6008]` is not currently a supported end-to-end smoke test:
mask generation still uses the full-volume grid, and planning tries to crop a keep mask
before the queued mask job creates it. Even `--check` and `--dry-run` enter that crop
setup. A small download alone does not validate the remaining stages.

```bash
python scripts/run_j0126.py --check              # what exists, what is missing
python scripts/run_j0126.py --dry-run            # print the commands only
python scripts/run_j0126.py --steps infer,abiss  # run part of it
python scripts/run_j0126.py --force abiss        # rerun a step that looks complete
```

Each step checks output artifacts and skips when its completion check passes. These are
resume heuristics, not integrity checks; see the limitations below. Step 2
resumes per chunk; the download and mask steps resume per shard.

Install PyTC and its workflow dependencies before invoking the driver. ABISS also needs
GNU `parallel`, `tinybrain`, and `chunkiterator` (from `seung-lab/chunk_iterator`, not a
package named `chunkiterator` on PyPI). Its dependency recipe is in
[`lib/abiss/docker/Dockerfile`](../../lib/abiss/docker/Dockerfile) after cloning.
Use a separate environment for the C++ build dependencies to avoid changing the Python
runtime's NumPy stack; make the required runtime libraries and GNU `parallel` available
on compute nodes. The following commands build ABISS; they do not install dependencies.
The reference decode used this commit:

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
| 3 | abiss | `3_abiss.yaml` | expected storage-chunk count is present under the segmentation scale |
| 4 | ec | `4_error_correction.yaml` | `error_correction_manifest.json` exists |
| 5 | eval | `scripts/evaluate_j0126.py` | `output_root/eval/nerl.json` exists |

Keep the step YAMLs' thresholds unchanged to reproduce the reference recipe. Planning
figures: [RESOURCE.md](RESOURCE.md). Disk cleanup mid-run: [CLEANUP.md](CLEANUP.md).

### Step 0 — data and exclusion mask

The EM volume comes from the public FFN mirror (Januszewski et al. 2018) at native mip 0.
Do not resample: it is the grid FFN published, the grid the evaluation skeletons index, and
the grid the reference runs used.

The exclusion mask is FFN's own `tissue_classification` thresholded to
`NOT(blood vessel | myelin | out-of-bounds)`, combined with the 0/255 border ring of the
aligned EM. Because it comes from FFN's CNN it weakens a "we beat FFN" comparison;
`dev/zebrafinch/build_bv_border_mask.py` is the alternative built from a vessel volume we
own (no myelin masked, 1.38% removed).

The thresholds are FFN's published ones, per channel and not interchangeable: blood vessel
≥ 252, myelin ≥ 252, out-of-bounds ≥ 25. Verified against FFN's own `tissue_mask` layer —
voxel-exact agreement on a 12.6 M-voxel sample — and it excludes 14.61% of the volume.
**The numbers in the results table below predate that check**: they were measured with a
uniform threshold of 128 on all three channels, which over-excludes (17.53% of the
`tissue_classification` grid, 15.64% after the border ring), and that is also what the
prebuilt mask published below contains. Rebuilding with the corrected recipe gives the mask
FFN actually used and will shift the table slightly.

Building it streams the whole `tissue_classification` layer, so the mask the reference runs
used is published instead: `j0126-tissue-border-keep-mask.tar` on
[huggingface.co/datasets/pytc/zebrafinch-j0126](https://huggingface.co/datasets/pytc/zebrafinch-j0126/tree/main).
Untar it into `dataset_root` — it unpacks as `tissue_border_keep_mask_full.zarr`, the name
`data.keep_mask` already points at — and set `mask.enabled: false` to skip steps 0c and 0d.

Training data is 33 densely labelled subvolumes, fetched only when `train.enabled` is true.
Training reads the padded pair `im_raw_4-32-32/` + `seg_gt_4-32-32/`: the pad is real EM
context on the image and `-1` on the label, so the loss ignores the border.

### Step 1 — train the affinity model (optional)

`train.enabled: false` downloads `pytc/vEM j0126/affinity_scratch_48x96x96.ckpt` instead.
That checkpoint has seen labelled j0126 tissue, so a run from it is **not**
ground-truth-free — it is the supervised upper reference. It also predates the held-out
split (all 33 cubes in training, 3 of them reused as validation), and every number in the
results table comes from it, so retraining gives an honest curve and a slightly different
model. The reported from-scratch reproduction remains below the historical reference
score; applying the operational fixes does not establish numerical reproduction.

`train.enabled: true` trains MedNeXt-L/k3 from scratch for 200k steps on the dense GT
cubes, 25 train / 8 held out, at roughly four GPU-days on 4 GPUs, and infers from whatever
checkpoint has the newest modification time (`ls -t ... | head -1`). This is not a
best-validation policy and can select an overfit final checkpoint. For a controlled
comparison, run inference explicitly with `scripts/main.py --config
tutorials/neuron_j0126/2_infer.yaml --mode test --checkpoint /absolute/path/to/model.ckpt`
and verify its output location before resuming ABISS.

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
in the shipped driver. Do not launch independent copies against the same workdir:
they can race on hierarchy layers. A fleet requires disjoint atomic task lists and
barriers before composite layers; the reported fleet launcher is not shipped here.
The historical 3.75 h figure belongs to a 40-node deployment, not one 64-core job.

With `CHUNK_SIZE: [2048, 2048, 80]`, the reproduction measured about 7.8 GB per atomic
task: 64 concurrent tasks need roughly 500 GB before additional overhead, exceeding
the shipped 250G allocation. Lower `abiss.cpus` (8–16 is a starting point to benchmark)
or increase memory according to measured peak usage. On GPFS, the report measured
about 38 h for watershed alone; the default 24 h is not a whole-decode estimate.
Resubmit interrupted runs with the same geometry to reuse per-chunk DONE flags.
When changing `BBOX` or `CHUNK_SIZE`, use a new output root: existing chunk descriptors
are not regenerated, and `--force abiss` does not invalidate them.

`AGG_THRESHOLD 0.20` with `WS_HIGH 0.9` / `WS_LOW 0.1` is the historical reference
recipe, not a guarantee of its scores with a retrained model or corrected mask.
The pipeline deliberately under-agglomerates here and attempts to repair splits in
step 4 using morphology and, when supplied, nucleus identities.

Setting `data.nucleus_volume` turns on competitive nucleus growth and writes the identity
manifest step 4 uses as a firewall. The recipe names the reference volume as
`j0126-nucleus-instances-80nm.h5` on
[huggingface.co/datasets/pytc/zebrafinch-j0126](https://huggingface.co/datasets/pytc/zebrafinch-j0126/tree/main):
465 hand-proofread nucleus instances at 80 nm isotropic, which `nucleus_ratio: [4, 8, 8]`
upsamples onto the mip-0 grid. This filename is the reference asset named by the recipe;
its availability and registration must be verified before attempting those results.
The reproduction report could not find a published nucleus volume. Left empty — the
default — the current driver creates an empty qualified-label manifest accepted by
error correction, with a warning. This disables the nucleus firewall; it does not
reproduce any nucleus-protected result in the table.

### Step 4 — morphology error correction

Builds skeletons for large segments, evaluates every sufficiently confident contact, and
accepts only hard-gated branch continuations. It protects external nucleus identities and
never joins two different ones. `erosion_radius_zyx` is `[0, 0, 0]` for morphology linking
alone and `[1, 1, 1]` for the strict-mt=0 cleanup.

The driver chains thirteen stages. It shards `skeletonize`, `contacts` and `postprocess`
as Slurm arrays at the config's `task_count` (80), with serial reductions between them.
Array tasks request 8 CPUs and 64G each; reductions use `params.error_correction`.
There is no need to submit an isolated skeletonization stage manually. To inspect the
whole chain:

```bash
python scripts/run_j0126.py --steps ec --dry-run
```

Stages are restartable — completed chunk artifacts are reused.

### Step 5 — evaluation

The driver writes `output_root/eval/nerl.json` using the 50 test skeletons. The evaluator
computes edge lengths in **native voxels**, not nm; `--resolution-xyz` selects the
precomputed scale and does not rescale skeleton edges. Check the reference LUT separately:

```bash
python scripts/evaluate_j0126.py --node-lut /path/to/j0126/ffn_node_lut_test_50.h5 \
    --skeletons /path/to/j0126/test_50_skeletons.h5 --output /path/to/ffn-reference.json
```

The report reproduced FFN NERL 0.5258 / 0.5380 and VOI 1.856 with this convention.
The length convention for the historical PyTC rows still needs provenance confirmation.
For explicit ABISS-only scoring, use `--segmentation /absolute/path/to/abiss/precomputed/seg`
instead of `--node-lut`. The automatic evaluation chooses correction output by `info`
existence, so after interrupted or selectively rerun correction, verify completion and
score the intended layer explicitly.

## Reproduction status

Audited against the local report `.agent/issues/j0126/reproduction.md`
and the current shared driver in `connectomics/playbooks/cube_decode.py`.
Documentation corrections are not a new full-volume validation.

Already implemented: direct masked HDF5 affinity reads, channel indices `[0, 1, 2]`,
the per-channel tissue thresholds, Zarr-3-compatible mask creation, mask chunks that
tile 1008, the submitting Python executable, omission of empty `NUC_*` parameters,
specific `seg_size_*.data` matching, native output resolution, staged correction,
an empty nucleus-manifest fallback, and an evaluation step.

Remaining limitations and operational workarounds:

- Download progress lives beside the store as `<store>.progress*`. If replacing a
  partial store, remove its matching progress sidecars too; otherwise missing blocks
  can be skipped. Mask `.done.*` sidecars likewise need invalidating when rebuilding.
- Affinity completion still resolves index entries relative to the index directory;
  indexes containing run-relative paths can be reported incomplete. Inspect paths
  before relaunching inference. Single-file crop output is not covered by this check
  or the driver's unconditional VDS discovery command.
- Fetch completion checks the model or training directories, not the evaluation
  skeletons/LUT. Confirm both evaluation files exist before submitting scoring.
- Crop mask geometry, nucleus alignment, global skeleton coordinates, and shard
  counts are not consistently propagated. Do not use `em_bbox` as the advertised
  end-to-end smoke test until those contracts are fixed and tested.
- ABISS completion counts files rather than validating chunk contents; the correction
  manifest and evaluation report are also existence checks. Use a fresh output root
  for changed inputs or recipes, and verify results before deleting intermediates.
- The stock launcher does not implement the report's distributed atomic fleet or
  its cold-start and cross-node locking fixes. Its timings cannot be promised here.

[`run_everything.sh`](run_everything.sh) bootstraps dependencies, builds ABISS, rewrites
this checkout's `params.yaml`, and invokes the driver. It defaults to training; use
`--no-train` for the downloaded checkpoint. Its `--check` and `--dry-run` flags apply
only to the final driver invocation: installation and configuration writes still occur.
It shares the crop limitations above and installs build/runtime dependencies together.
The nucleus asset and a full crop-to-score regression remain requirements before
describing the bootstrap as a validated fresh-machine reproduction.

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

# j0126: conservative segmentation, morphology-based reconnecting

This tutorial turns a 10 nm j0126 EM volume into a neuron segmentation in three steps:

1. Predict voxel affinities.
2. Decode them conservatively with ABISS, preferring splits to false merges.
3. Reconnect only high-confidence, morphologically continuous branches.

For the frozen PyTC2 Zebrafinch replay, including fail-closed input checks and
non-interactive Slurm submission, follow the
[reproduction guide](reproduction/README.md).

The last step is prediction-only at runtime. It uses the segmentation, affinities, predicted morphology, and an external nucleus-instance manifest; it does not read evaluation skeletons, their lookup table, or an FFN segmentation.

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

## Resource budget

These are planning figures for the complete 5,700×10,913×10,664 voxel volume, not guarantees. They assume the configured 726 affinity chunks, a 48 GB-class GPU, a parallel filesystem, and no queue time. With the staged cleanup below, expect a **6–7 TiB peak** and reserve **8 TiB**. Keeping every intermediate or a second affinity arm can require 10 TiB. After final verification, retaining only the result and audit artifacts should take well below 1 TiB; retaining the source affinity raises that to roughly 3–4 TiB.

| Step | Compute specification | Estimated wall time | Storage while running |
|---|---|---:|---:|
| 1. Affinity | 1 GPU with at least 48 GB per shard; 726 independent shards | ~30 min/shard; ~6 h at 64 GPUs, ~5 h at 80 GPUs | 2.5–3 TiB for the float16 chunk store |
| 2. ABISS | Layer-aware CPU fleet; 16 CPU / 130 GB nodes for atomic chunks, then 1–8 CPU workers with 40–100 GB per composite chunk | 3.75 h measured on 40×16 CPU workers; ~1.9 h projected with the multi-node layout below | input affinity plus ~0.7 TiB resumable scratch, a precomputed affinity layer, and ~45 GB final segmentation; budget 4 TiB additional |
| 3. Error correction | 80 array tasks, 8 CPU workers/task and 64 GB/task; reductions run serially | 6–12 h estimate; this has not yet been benchmarked end-to-end | reuse steps 1–2 inputs; reserve 0.5–1 TiB for skeleton, contact, and output artifacts |

The step-1 timing is measured from the chunked Zarr input path. On a 40 GB GPU, lower `sw_batch_size` from 12 before running; that increases the per-chunk time. Reading tiled PNGs instead can take roughly 19 h **per chunk** and is not a usable production path. ABISS scratch and `work/` artifacts are only needed for resume; retain the final precomputed segmentation, parameter file, and manifests after recording the result, then reclaim scratch space.

### Staged storage cleanup

Cleanup trades storage for restartability. Run each block only after the named downstream artifact exists. Replace the first path with the resolved `params.paths.output_root`; the guards refuse an empty path, `/`, or a run without the expected output markers.

After EC `sizes` has written its aggregated inventory and ABISS has completed, remove the ABISS affinity copy, watershed, chunk map, resume scratch, and run workspace. This normally recovers several TiB while preserving the final ABISS segmentation and its parameter file:

```bash
J0126_OUTPUT_ROOT="/absolute/path/to/outputs/neuron_j0126"
ABISS_ROOT="$J0126_OUTPUT_ROOT/abiss"
EC_ROOT="$J0126_OUTPUT_ROOT/error_correction_v7"

case "$J0126_OUTPUT_ROOT" in ""|"/") exit 2;; esac
test -f "$ABISS_ROOT/precomputed/seg/info" || exit 2
test -s "$EC_ROOT/segment_sizes.data" || exit 2

rm -r -- \
  "$ABISS_ROOT/precomputed/affinity" \
  "$ABISS_ROOT/precomputed/ws" \
  "$ABISS_ROOT/chunkmap" \
  "$ABISS_ROOT/scratch" \
  "$ABISS_ROOT/run"
```

After `skeletons`, `contact_graph`, and `junction_features` complete, their per-chunk caches are redundant. Removing them preserves the aggregated morphology, contact graph, and junction features used by the resolver:

```bash
J0126_OUTPUT_ROOT="/absolute/path/to/outputs/neuron_j0126"
EC_ROOT="$J0126_OUTPUT_ROOT/error_correction_v7"

case "$J0126_OUTPUT_ROOT" in ""|"/") exit 2;; esac
test -s "$EC_ROOT/segment_skeleton_graph.h5" || exit 2
test -s "$EC_ROOT/contact_graph.npz" || exit 2
test -s "$EC_ROOT/junction_features_raw.npz" || exit 2

rm -r -- \
  "$EC_ROOT/skeleton_chunks" \
  "$EC_ROOT/contact_chunks" \
  "$EC_ROOT/skeleton_cache"
```

The original 2.5–3 TiB affinity store is needed through the EC `contacts` stage, but not after the final output verifies. At that point it can be archived or deleted; set the exact store explicitly rather than deleting the whole affinity output directory:

```bash
J0126_OUTPUT_ROOT="/absolute/path/to/outputs/neuron_j0126"
EC_ROOT="$J0126_OUTPUT_ROOT/error_correction_v7"
AFFINITY_STORE="/absolute/path/to/the/affinity.h5.chunks"

case "$J0126_OUTPUT_ROOT" in ""|"/") exit 2;; esac
case "$AFFINITY_STORE" in ""|"/") exit 2;; esac
test -f "$EC_ROOT/error_correction_manifest.json" || exit 2
find "$AFFINITY_STORE" -maxdepth 1 -name 'chunk_z*_y*_x*.h5' -print -quit \
  | grep -q . || exit 2

rm -r -- "$AFFINITY_STORE"
```

Keep `precomputed/seg`, `error_correction_manifest.json`, `segment_sizes.data`, the resolver reports, and `resolver/v7/frozen_junction_merges.{npz,json}`. Those are the compact result and provenance record. Also keep the source affinity if exact contact regeneration matters more than storage.

## Before running

Edit **only** [params.yaml](params.yaml). It contains the repository checkout, dataset root, writeable output root, and the existing artifacts required to replay the frozen EC recipe. Every step inherits this file, so paths are not duplicated across the workflow YAMLs. Keep the algorithmic thresholds in the step YAMLs unchanged when reproducing the reference recipe.

The zero-shot affinity path needs a NISB-trained checkpoint. The supervised affinity YAML is included as a target-domain reference only: it uses j0126 dense labels, so it is not part of the zero-shot pipeline; its trained checkpoint can be downloaded instead of retrained (see [Step 1](#supervised-reference-affinity)).

## Step 1 — affinity prediction

Run zero-shot inference with an NISB checkpoint:

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

### Supervised reference affinity

`1_affinity_supervised.yaml` is the target-domain reference: MedNeXt-L/k3 trained from scratch for 200k steps on the 33 j0126 dense-GT cubes. It has seen labelled j0126 tissue, so any run that starts here is not ground-truth-free.

Training it costs roughly four GPU-days. Download the reference checkpoint instead:

```bash
hf download pytc/j0126 affinity_scratch_48x96x96.ckpt --local-dir ckpt/

python scripts/main.py --config tutorials/neuron_j0126/1_affinity_supervised.yaml \
  --mode test --checkpoint ckpt/affinity_scratch_48x96x96.ckpt
```

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

## Frozen correction recipe

The scratch run freezes 749 branch unions. Evaluation is deliberately outside the EC config and should be run only after the proposal has been frozen. Use `erosion_radius_zyx: [0, 0, 0]` for morphology linking alone and `[1, 1, 1]` for the strict-mt=0 cleanup shown above.

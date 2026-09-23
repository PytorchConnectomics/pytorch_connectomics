# Task — S3 only: chunked ABISS decode, proven equal to whole-volume

Make the ABISS decode chunk-native and **prove it equals the whole-volume decode**
on a volume that already has one. This is stage S3 of card MSIDEPLOY-SCALE-001.

## Why this is its own run

A previous CCC run (`.agent/features/abiss_gcloud/`, canceled) bundled S3 with S4
(block-parallel execution across many VMs). Four plan rounds converged every
unresolved finding onto S4's distribution protocol while S3 carried almost none.
The bundling came from that run's task, not from a dependency.

**S3 is also the part that answers the 100 µm question.** A 100 µm cube is 60×
over ABISS's `uint32` watershed cap, so chunked decoding is the only route; until
chunked decoding is known to equal whole-volume decoding, throughput is
premature. S4 will be a separate run whose entire subject is the dispatcher.

## Explicitly out of scope

Distribution of any kind: no worker pool, no dispatcher, no queue, no Redis, no
`$PARALLEL_CMD` replacement. **Decode runs on ONE VM**, using ABISS's built-in
GNU-parallel-over-cores behaviour unchanged. Also out of scope: block-parallel
inference, multi-region fallback, acquisition, retraining, publication to
Neuroglancer, and which merge criterion is scientifically preferred.

## Verified facts, established by reading the vendored ABISS checkout

These are measurements, not assumptions; do not re-derive them.

- The chunked decode is an **octree**: `chunk_volume.py` → `generate_batches.py`
  → `run_layer.sh 0 atomic_chunk_<fn>` → `run_layer.sh i composite_chunk_<fn>`
  for `i` in `1..top_mip`. `remap_watershed` / `remap_agglomeration` apply the
  chunkmap so labels are globally consistent; `CHUNKMAP_OUTPUT` holds it.
  **Cross-chunk reconciliation already exists — do not design one.**
- Resume is **per task (per chunk)**, not per layer: `run_wrapper.sh` calls
  `check_task_flag.py` to skip completed work and `update_task_flag.py` to record
  it, keyed `${STATSD_PREFIX}_${STAGE}_${OP}_${CHUNK}`, stored via `CloudFiles`
  under `SCRATCH_PATH` or in Redis.
- Parallelism inside a layer is `PARALLEL_CMD="parallel --halt 2 -j ${ncpus}"`
  (`init.sh:107`) — **one machine, over cores. That is sufficient for S3.**
- Chunked agglomeration binaries are `acme` (mean-edge), `ac`/`agg` (`rlme`) and
  `accs`. **There is no `max` binary.** The whole-volume `ws` accepts
  `--ws-merge-function max`; the chunked path does not.
- `seg_chunk_size_xyz` (CloudVolume storage chunk) must align with every
  `param.CHUNK_SIZE` (ABISS logical chunk) boundary; the driver raises otherwise,
  "because ABISS uploads chunk outputs in parallel". Both are **XYZ**, while the
  rest of this project is ZYX.
- S1 (BC job `3031548`, ExPID82 val, 6 cubes) measured best VOI: `max` **0.5279**,
  `p75` 0.5628, `mean` **0.5827**. `max_compressed` and `max_uncompressed` agreed
  row for row at all five thresholds, so `max`'s monotone invariance is measured,
  not assumed. **All three optima sat at the bottom edge of the swept range**, so
  this establishes the ordering of criteria, not their best values.

## Requirements

1. **The merge criterion is a parameter, never a literal.** Configure it once and
   thread it identically into both decoders. Because the chunked path has no
   `max`, the implementation must fail at **config resolution** — before any VM —
   with a message naming the requested criterion and the available chunked stages,
   rather than failing mid-decode.
2. **The equivalence test holds the criterion fixed.** It asks whether *chunking*
   changes the answer, so both sides run the same criterion (`mean` is the only
   one available chunked). This requires producing a whole-volume `mean` reference
   for the test volume; the published `max` layer is not the reference and is not
   regenerated.
3. **Affinity representation must be stated and validated.** `mean` is not
   monotone-invariant, so it requires uncompressed (probability-space) affinity,
   where the stored value is `sigmoid(0.2·logit(p))`. Specify exactly which
   artifact ABISS consumes, where the inversion happens, the dtype, the clipping
   rule at the tails, and the path schema. Validate with S1's invariance check
   re-run on the real artifact: a whole-volume `max` decode of the probability
   layer must reproduce the compressed-artifact result.
4. **Chunk-alignment is asserted before allocation**, with both numbers named.
5. **Acceptance is an equivalence test with numeric tolerances**, on
   `ExPID108_32x_Cortex_L1_01` (0.26 Gvoxel, already decoded whole-volume):
   - VOI computed on **masked volumes**, not label subsets, so both sides are the
     same quantity on the same voxels;
   - boundary set `B` = whole-volume labels whose bounding box intersects a
     `CHUNK_SIZE` plane; interior set `I` = the rest;
   - `VOI_total(chunked, whole) ≤ 0.01` **and**
     `VOI_total(B) − VOI_total(I) ≤ 0.01` — 0.01 is this project's measured
     noise floor (two runs differing only in GPU count differ by 0.006);
   - **chunk-size invariance**: two alignment-valid `CHUNK_SIZE` values agree
     within the same tolerance. This is the strongest single signal, because a
     correct implementation cannot depend on chunk size;
   - a one-chunk run is a **plumbing test, not acceptance**.
6. **Seams fail quietly.** A mis-chunked decode returns a plausible segmentation,
   not an error. Every check above exists for that reason.

## Success criteria

Chunked decode of a volume with a known whole-volume answer reproduces it within
0.01 total VOI, seams included, at two chunk sizes — and the same code path runs
unmodified on a volume 8× larger without a whole-volume answer.

## Sizing context, for chunk-size choice only

100 µm at `[24,18,18]` nm ZYX is `(4167, 5556, 5556)` = 128.6 Gvoxel; XYZ
`(5556, 5556, 4167)`. Per-chunk RSS is **modelled** at the measured 71 GB/Gvoxel:

| `CHUNK_SIZE` XYZ | atomic chunks | Mvox/chunk | modelled RSS |
|---|---|---|---|
| `[1024, 1024, 128]` | 1,188 | 134 | 10 GB |
| `[2048, 2048, 80]` | 477 | 336 | 24 GB |
| `[2048, 2048, 256]` | 153 | 1074 | 76 GB |

`[2048, 2048, 80]` with `seg_chunk_size_xyz [512, 512, 80]` is the j0126 pairing
and is known to satisfy the alignment rule. Every option is far under the
2147 Mvox cap.

## Reuse, do not rebuild

`connectomics/runtime/abiss_chunk.py` (chunked driver), `scripts/run_abiss_chunk.py`,
`tutorials/neuron_j0126/3_abiss.yaml` (a worked chunked config for a
10664×10913×5700 volume), and `tutorials/neuron_liconn_moe/gcloud/` (the working
single-volume cloud path: GCS image archive, spot with resume,
`STAGES=all|gpu|cpu`, preflight checks).

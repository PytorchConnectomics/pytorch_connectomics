# Plan v2

## Summary

I read ABISS's vendored scripts on the cluster rather than reasoning about them,
and three of the v1 findings now have verified answers instead of assertions.

The decisive one: **ABISS's chunked pipeline cannot agglomerate with `max`.** It
ships `acme` (mean-edge), `ac`/`agg` (`rlme`) and `accs`; there is no max
binary and no `atomic_chunk_max.sh`. The whole-volume `ws` binary accepts
`--ws-merge-function max`; the octree pipeline does not. So the S1 outcome is not
a configuration branch — for one of its arms the chunked path **has no
implementation at all**, and that is now a checkable fact rather than a risk.

The second: v1's §2 was wrong in the way finding 4 said. ABISS's chunked decode
is an **octree** — `chunk_volume.py` → `generate_batches.py` → layer 0 atomic
chunks → layers 1..`top_mip` composite chunks — and parallelism inside a layer
comes from `$PARALLEL_CMD` in `run_layer.sh`, GNU parallel over CPU slots **on
one machine**. Layers are hard barriers. So decode across many small VMs is not
reachable by configuration; it needs `$PARALLEL_CMD` replaced by a cross-machine
dispatcher plus a per-layer barrier. That is a single, well-defined seam, and
this plan specifies it rather than either denying it (v1) or wrapping a second
queue around the stages (v0).

The third: finding 1's cost objection dissolves once the right artifact moves.
Relocating the **source** across regions is ~$2.64; relocating the affinity is
~$15.81 and the segmentation ~$10.57. Multi-region is therefore tractable if the
pipeline moves to the data's copy, not the data to the pipeline.

## Scope

Unchanged: S3 and S4 of card MSIDEPLOY-SCALE-001. Out of scope unchanged: S1's
criterion choice, S2, acquisition, retraining, a real 100 µm run, and — stated in
v1, retained — **publication**: S3/S4 deliver segmentation artifacts, not a
Neuroglancer volume, because `upload_seg_precomputed.py` reads whole
segmentations into RAM and needs its own card.

## Proposed Changes

### 1. Merge criterion — a verified fork, not a deferred one

*Finding 2.* Verified on the vendored checkout: chunked agglomeration variants are
`me` (`acme`, mean-edge), `rlme` (`ac` + `agg`) and `cs` (`accs`). **No max.**

Decision rule, executable without further design:

- **S1 selects `mean`** → configure `agglomerate_mean_edge`; `AGG_THRESHOLD`
  takes S1's threshold in probability space. No code change.
- **S1 selects `max`** → the chunked path cannot run it. Two options, and the
  plan commits to surfacing rather than silently choosing: (a) accept `mean` for
  the chunked path and treat S1's `max` result as the whole-volume reference only,
  which makes every 100 µm number a `mean` number and must be stated as such; or
  (b) evaluate `rlme` as the nearest available criterion, which is a new
  measurement, not a substitution. **Building a max agglomeration into ABISS is
  out of scope.**
- **Either way** the criterion and threshold stay unset configured inputs,
  threaded identically into `run_abiss_volume.py` (`--ws-merge-function`,
  `--ws-merge-thresholds`) and the chunked config (stage name, `AGG_THRESHOLD`).

Implementation asserts the selected criterion exists in the chunked stage list and
fails at config resolution, before any VM, if it does not.

### 2. The chunk contract, documented from the implementation

*Finding 3.* Recorded in the config and in `gcloud/README.md`, from the vendored
scripts:

- `chunk_volume.py` partitions `BBOX` by `param.CHUNK_SIZE` (**XYZ**);
  `generate_batches.py` emits one task list per octree layer.
- `run_layer.sh 0 atomic_chunk_<fn>` runs per-chunk watershed/agglomeration;
  `run_layer.sh i composite_chunk_<fn>` for `i` in `1..top_mip` merges upward.
- `remap_watershed` / `remap_agglomeration` apply the chunkmap so labels are
  globally consistent; `CHUNKMAP_OUTPUT` is where that mapping lives. **This
  change adds no halo, ownership or reconciliation logic** — it configures the
  existing octree.
- Resume granularity is the **layer**, via `check_task_flag.py` /
  `update_task_flag.py` on `TASK_KEY`. Not per chunk. A preempted decode resumes
  at the layer boundary, and that is the unit to size retries against.
- `seg_chunk_size_xyz` (CloudVolume storage) must align with every
  `param.CHUNK_SIZE` boundary; the driver raises otherwise, "because ABISS uploads
  chunk outputs in parallel". Our resolver re-checks it with the chosen numbers.

### 3. Distributed decode — replace `$PARALLEL_CMD`, keep the octree

*Finding 4, accepted.* `run_layer.sh` pipes a layer's task list into
`$PARALLEL_CMD`. That variable is the entire seam.

Specified:

- a dispatcher honouring the same contract — read task lines on stdin, run each
  as a command, exit non-zero if any task fails — but farming them to N workers;
- **queue semantics**: one task = one line = one atomic/composite chunk; claim by
  atomic create of a `claimed/<task>` object in the run prefix; lease by mtime
  with a configurable timeout; completion by `done/<task>`; retry when a lease
  expires; a task is idempotent because a chunk's output is overwritten by its own
  recomputation;
- **barrier**: the dispatcher returns only when every task in the layer has a
  `done/` marker, preserving the layer dependency;
- **missing-work detection**: task count from `generate_batches.py` compared to
  `done/` count before the layer is marked DONE.

Fallback: with one worker the dispatcher is equivalent to GNU parallel, so the
single-VM path remains available and is what the equivalence test in §6 uses.

### 4. Multi-region — move the pipeline to the source, never the affinity

*Finding 1, accepted.* Measured costs at $0.02/GiB one way:

| artifact | size | cross-region cost |
|---|---|---|
| source at model grid, uint8 | 129 GB | **$2.64** |
| affinity, 3-ch float16 | 772 GB | $15.81 |
| segmentation, uint32 | 516 GB | $10.57 |

Policy: alternate **zones** in `us-east1` → smaller GPU shapes → **fallback
region**, staging only the source and the image archive there and running
inference *and* decode in that region → on-demand, only when
`ALLOW_ON_DEMAND=1` → fail with the measured capacity error. The affinity and
scratch are created and consumed in whichever region the compute runs, and never
cross. The segmentation returns once.

Region selection is an ordered list (`REGIONS="us-east1 us-central1"`), tried in
order; resume state lives in the run prefix, which is small and is read
cross-region without material cost.

### 5. Concrete sizing

*Finding 5.* For the 100 µm cube, `(5556, 5556, 4167)` XYZ = 128.6 Gvoxel:

| `CHUNK_SIZE` XYZ | atomic chunks | Mvox/chunk | modelled RSS/chunk | `top_mip` |
|---|---|---|---|---|
| `[1024, 1024, 128]` | 1,188 | 134 | 10 GB | 6 |
| `[1024, 1024, 256]` | 612 | 268 | 19 GB | 5 |
| **`[2048, 2048, 80]`** | **477** | **336** | **24 GB** | **6** |
| `[2048, 2048, 256]` | 153 | 1074 | 76 GB | 5 |

**Proposed default `[2048, 2048, 80]`** with `seg_chunk_size_xyz [512, 512, 80]`
— the j0126 pairing, which is known to satisfy the alignment rule. 24 GB
modelled RSS per chunk fits any `n2-standard-8`-class worker, so decode workers
are small and interchangeable, which is the point of §3. Every option is far
under the 2147 Mvox cap.

RSS is **modelled** as 71 GB/Gvoxel — the only figure this project has measured,
and measured on whole-volume decodes. The first chunked run tests whether it
holds per chunk; `size_report.py` prints it labelled as a model. Retry overhead
is reported as a separate line, not folded in, and is **not** given a multiplier
until a preemption rate is measured (finding 9 accepted).

### 6. Equivalence test — executable metric

*Finding 6.* On `ExPID108_32x_Cortex_L1_01`, both decoders at the same criterion.

- **Object sets.** Let `B` = ground-truth-free set of labels in the *whole-volume*
  segmentation whose bounding box intersects any plane `k·CHUNK_SIZE[axis]`;
  `I` = all other labels.
- **Metric.** VOI is computed on a **masked volume**, not on a subset of labels:
  for `B`, mask to voxels whose whole-volume label ∈ `B`, and compute
  `VOI(chunked|mask, whole|mask)` as total = split + merge, reporting the three
  numbers. Same for `I`. This makes both sides the same quantity on the same
  voxels, which is what "difference" requires.
- **Acceptance.** `VOI_total(chunked, whole) ≤ 0.01` on the full volume, **and**
  `VOI_total(B) − VOI_total(I) ≤ 0.01`. Both are differences of two numbers
  computed identically; neither is a sum or a max over chunks.
- **Chunk-size invariance.** Two `CHUNK_SIZE` values, both alignment-valid, agree
  within 0.01.
- One-chunk case is a plumbing test, not acceptance (v1, retained).
- Worker-count correctness: 1 vs N dispatcher workers, plus a killed-and-retried
  worker, `VOI_total ≤ 0.01`.

### 7. Affinity layout

*Finding 7.* Precomputed on `gs://`: `float16`, 3 channels, repo channel order
`ch0=Z, ch1=Y, ch2=X` with `AFF_CHANNELS [2,1,0]` for ABISS, storage chunk equal
to `seg_chunk_size_xyz`, `voxel_offset` from `BBOX[:3]`, `volume_size` from
`BBOX[3:] - BBOX[:3]`. One inference block writes one disjoint region; a
post-inference validator asserts every expected block exists, is non-empty, and
that the layer's `info` matches the prepared volume's shape and resolution.

## Files and Areas

| Path | Change |
|---|---|
| `connectomics/runtime/abiss_chunk.py` | read only |
| `lib/abiss/scripts/init.sh` (`$PARALLEL_CMD`) | seam for §3; dispatcher supplied by env, not a fork |
| `tutorials/neuron_liconn_moe/gcloud/chunked_abiss.yaml` | new config |
| `tutorials/neuron_liconn_moe/gcloud/dispatch_tasks.py` | new: §3 queue/lease/barrier |
| `tutorials/neuron_liconn_moe/gcloud/run_volume.sh` | chunked decode stage, criterion assertion |
| `tutorials/neuron_liconn_moe/gcloud/launch.sh` | worker fan-out, zone→shape→region→on-demand |
| `tutorials/neuron_liconn_moe/gcloud/equivalence_test.py` | new: §6 |
| `tutorials/neuron_liconn_moe/gcloud/size_report.py` | new: §5 |
| inference output path | precomputed affinity writer + block validator |

## Verification Plan

1. `bash -n` on shell changes; chunked config resolves on the driver's dry path.
2. Criterion assertion fails, at config resolution, for a criterion absent from
   the chunked stage list (`max`).
3. Alignment assertion rejects a mis-aligned `CHUNK_SIZE`/`seg_chunk_size_xyz`
   pair, naming both.
4. Plumbing: one-chunk chunked decode reproduces whole-volume at the same
   criterion, `VOI_total = 0`.
5. **S3 acceptance:** multi-chunk equivalence, `VOI_total ≤ 0.01` and
   `VOI_total(B) − VOI_total(I) ≤ 0.01`, at two alignment-valid chunk sizes.
6. Dispatcher: 1 vs N workers `VOI_total ≤ 0.01`; a killed worker's task is
   re-leased and completes; layer count check fails a deliberately deleted
   `done/` marker.
7. Affinity validator fails a deliberately missing inference block.
8. Fallback: unavailable shape → alternate reported; only-cross-region → source
   staged, affinity confirmed created in the compute region, cost printed.
9. `size_report.py` 100 µm totals match the task table to two significant figures.

## Risks and Questions

- **If S1 selects `max`, there is no chunked implementation** (§1). Verified, not
  suspected. This is the single highest-consequence open item and the decision
  between accepting `mean` and evaluating `rlme` belongs to the card and the
  human, not to this change.
- **Layer-granularity resume** means a preemption late in a layer re-runs that
  layer's unfinished tasks only — acceptable — but a preemption of the
  *dispatcher* re-runs the layer's bookkeeping. Sized, not eliminated.
- **Per-chunk RSS is a model**, extrapolated from whole-volume measurements.
- **Modifying `$PARALLEL_CMD` behaviour touches vendored ABISS.** The plan
  supplies the dispatcher through the existing environment variable rather than
  forking the scripts, but the contract is inferred from one call site and should
  be re-read against the pinned ABISS revision before implementation.
- **Publication at 100 µm remains unsolved** and out of scope.

## Changes Since Previous Plan Version

Addresses all 6 major and 3 minor findings in `plan_v1_review.md`, with three
answered by reading the vendored ABISS scripts rather than by argument.

- **Finding 1 (multi-region) — accepted, and the v1 refusal withdrawn.** §4 moves
  the pipeline to the source rather than the source to the pipeline: staging the
  129 GB source costs $2.64 against $15.81 for the affinity, so the objection
  that made v1 refuse does not apply to the artifact that actually has to move.
  Region order, staging, cost handling and resume are specified.
- **Finding 2 (criterion) — resolved by verification.** ABISS chunked ships
  `acme`/`ac`/`accs` and **no max**. §1 gives the decision rule for each S1
  outcome and a config-resolution assertion.
- **Finding 3 (chunkmap contract) — documented** in §2 from `run_batch.sh`,
  `run_layer.sh`, `atomic_chunk_*.sh` and the task-flag scripts, including the
  layer-granularity resume that was previously unstated.
- **Finding 4 (distributed decode) — accepted; the v1 rejection was wrong.** §3
  places the queue at `$PARALLEL_CMD`, the actual seam, with claim/lease/retry/
  barrier/missing-work semantics, and keeps ABISS's octree rather than wrapping
  it.
- **Finding 5 (concrete numbers) — supplied** in §5: four chunk options with
  chunk counts, per-chunk voxels, modelled RSS and `top_mip`, a proposed default,
  and the storage-chunk pairing.
- **Finding 6 (equivalence metric) — made executable** in §6 by defining VOI on
  *masked volumes* rather than label subsets, so both sides are the same quantity
  and the difference is well posed.
- **Finding 7 (affinity layout) — completed** in §7 with metadata, shapes,
  disjoint-write discipline and a validator.
- **Finding 8 (fallback matrix) — ordered and named** in §4, with
  `ALLOW_ON_DEMAND` as the enabling interface.
- **Finding 9 (×1.3 allowance) — removed.** Retry overhead is now reported
  separately with no multiplier until a preemption rate is measured.

# Plan v1

## Summary

The v0 review was right that v0 deferred its hardest decisions. Acting on it
produced a finding that **shrinks the plan rather than growing it**: I read
`connectomics/runtime/abiss_chunk.py`, and cross-chunk reconciliation is not
something this change has to design.

ABISS already owns it. The chunked pipeline runs
`watershed → remap_watershed → agglomerate_mean_edge → remap_agglomeration`, the
`remap_*` stages exist precisely to make per-chunk labels globally consistent via
`CHUNKMAP_OUTPUT`, and the driver's own guard states that **"ABISS uploads chunk
outputs in parallel"** — it already distributes work internally and enforces an
alignment precondition between the CloudVolume storage chunk
(`seg_chunk_size_xyz`) and ABISS's logical chunk (`param.CHUNK_SIZE`), raising
rather than corrupting when they disagree.

So v0's finding-3 and finding-4 were asking me to design a distributed decoder
that exists. **v1 replaces "design it" with "configure it, state its
preconditions, and verify it"** — and the verification is the part that carries
the risk, because a mis-aligned or mis-configured chunked decode returns a
plausible segmentation rather than an error.

The remaining genuinely open items are narrower and are now decided in-plan: the
affinity layout, GCS-vs-local scratch, region policy, and criterion
pass-through.

## Scope

Unchanged from v0 in extent — S3 and S4 of card MSIDEPLOY-SCALE-001 — but
re-apportioned: less new machinery, more configuration, preconditions and tests.

**In scope:** chunked-decode configuration wired into the cloud driver; a
chunked affinity layout; an equivalence test with numeric tolerances; worker
fan-out for *inference*; zone/shape/region fallback policy; a sizing report.

**Out of scope, unchanged:** the merge criterion itself (S1, job `3031548`); the
GPU/CPU split (S2, `7d819bdb`); acquisition; retraining; a real 100 µm run.

**Newly stated boundary (review finding 13): S3/S4 deliver decoded segmentation
artifacts, not a published Neuroglancer volume.** `upload_seg_precomputed.py`
loads the whole segmentation into RAM (`_read_seg`) and has its own 60× problem.
That is a separate card, and this change does not claim the deploy spec's
"a link comes back" deliverable at 100 µm.

## Proposed Changes

### 1. Chunk decomposition and reconciliation — configure and constrain, do not design

*Addresses findings 3 and 4.*

Document in the config, as comments, the semantics ABISS already provides:
per-chunk watershed, then `remap_watershed` and `remap_agglomeration` producing
globally consistent labels through `CHUNKMAP_OUTPUT`. **No halo, ownership rule
or reconciliation logic is added by this change**, because adding one would
duplicate and likely contradict the existing mechanism.

What this change *does* own is the precondition the driver already checks, made
explicit and tested:

- `param.CHUNK_SIZE` (ABISS logical chunk, **XYZ**) and `seg_chunk_size_xyz`
  (CloudVolume storage chunk, **XYZ**) must be chosen so that every logical
  boundary lands on a storage boundary. The driver raises
  `"ABISS logical chunk uploads would require non-aligned CloudVolume writes"`
  otherwise, and notes this is unsafe *because uploads are parallel*.
- Add a **pre-run assertion in our own config resolver** that recomputes this
  alignment and fails with the chosen numbers, so a misconfiguration is caught
  before a VM is allocated rather than mid-decode.

### 2. Worker fan-out applies to inference, not to the ABISS stages

*Addresses finding 4, corrected.* ABISS parallelises its own chunks inside each
stage, and `run_abiss_chunk` executes stages **in order** (`for plan in plans`).
A second queue wrapped around that would fight it.

Therefore:

- **Decode** runs as one job per volume on a high-memory CPU VM, sized by
  *per-chunk* peak RSS rather than whole-volume RSS. Parallelism inside is
  ABISS's.
- **Inference** is where fan-out belongs, using the existing `-ji/-jn` block
  chunking. For that path only, specify: a block inventory derived from the
  volume shape and block size; each worker writes only its own blocks; a
  completion marker per block; and a **coverage check that every expected block
  exists and is non-empty before decode starts** (finding 11).
- Retry/idempotency for inference workers: blocks are content-addressed by index,
  a re-run overwrites its own block, and a missing block fails the coverage check
  rather than silently shortening the volume.

### 3. Affinity layout — decided, not deferred

*Addresses finding 5.* Decision: **write affinity as a CloudVolume/precomputed
layer on `gs://`**, not an h5 chunkstore, because `abiss_chunk.py` consumes a
precomputed `AFF_PATH` directly and the h5-chunkstore branch is the one that
carries the keep-mask special-casing.

Specified: `float16`, channel order as this repo writes it (`ch0=Z, ch1=Y,
ch2=X`) with `AFF_CHANNELS` reversed to `[2,1,0]` for ABISS, storage chunk equal
to `seg_chunk_size_xyz` so §1's alignment holds for the input too, voxel offset
from `BBOX`, and one writer per block with no overlapping writes.

First implementation step is still a check — whether a chunked-affinity writer
already exists — but the *target* is now fixed, so the branch is "wire up or
write", not "decide what to build".

### 4. Scratch — local SSD, with GCS for inputs and outputs only

*Addresses finding 6.* Decision rather than an open question: `SCRATCH_PATH` and
`CHUNKMAP_OUTPUT` go on **local SSD**; `AFF_PATH`, `WS_PATH`, `SEG_PATH` on
`gs://`. Rationale: ABISS scratch is small-object, high-frequency and rewritten,
which is the worst shape for object storage on both latency and Class A
operations, and the j0126 precedent uses `file://` throughout. GCS scratch is not
ruled out permanently, but proving it is not on the path to 100 µm and would be
its own experiment.

Consequence to size for: scratch must fit on the VM's local disk, which becomes
an input to the sizing report.

### 5. Region policy — argue for in-region, fail loudly

*Addresses finding 2.* The task asked for multi-region fallback; this plan
proposes **not** doing it, and says why, rather than silently omitting it.

Both buckets are US-EAST1 regional. A 100 µm affinity is 772 GB, so one
cross-region copy is ~$15 in transfer alone and adds hours. Fallback order:
**alternate zones in us-east1 → smaller GPU shapes → on-demand only if explicitly
enabled → fail with the measured capacity error.** Cross-region is implemented as
a *refusal with a clear message* naming the transfer cost, not as an automatic
path. If cross-region is genuinely wanted, the right answer is bucket
replication decided deliberately, which is a storage decision outside this
change.

### 6. Merge criterion — one parameter, no per-outcome branching

*Addresses finding 1.* S1 has four arms, and enumerating code changes per arm
would be disproportionate. Instead: criterion and threshold are **configured
inputs with no defaults**, threaded identically into both decoders —
`--ws-merge-function` / `--ws-merge-thresholds` for `run_abiss_volume.py`, and
`AGG_THRESHOLD` (plus the criterion stage name) for the chunked path.

The one real asymmetry is recorded as a precondition: the whole-volume decoder
accepts several criteria, while the chunked stage list offers
`agglomerate_mean_edge`. **If S1 selects anything other than `mean`, the chunked
path requires an ABISS-side stage that may not exist**, and that is a blocker to
surface immediately rather than a config change. The equivalence test in §7 is
what detects it, because it runs both decoders at the configured criterion and
cannot be set up at all if the chunked side lacks it.

### 7. Equivalence test — numeric

*Addresses findings 7, 8, 9, 10.*

Volume: `ExPID108_32x_Cortex_L1_01`, 0.26 Gvoxel, already decoded whole-volume.
Both decoders run at the **same configured criterion** so chunking is the only
variable.

Definitions, previously ambiguous:

- **metric**: total VOI (split + merge), with split and merge also reported
  separately; comparison is label-permutation invariant by construction.
- **boundary-crossing object**: a ground-truth-free selection — any object in the
  whole-volume segmentation whose bounding box intersects a plane at a multiple
  of `param.CHUNK_SIZE` on any axis. **Interior object**: all others.
- **tolerance**: total VOI between chunked and whole-volume ≤ **0.01**, the noise
  floor this project has measured (two runs differing only in GPU count differ by
  0.006). Formally: `VOI(chunked, whole) ≤ 0.01`, and
  `VOI_boundary − VOI_interior ≤ 0.01`.
- **chunk-size invariance**: two chunk sizes, both satisfying §1 alignment, agree
  within the same 0.01.
- **one-chunk case**: retained but **labelled a plumbing test, not an S3
  acceptance criterion** (finding 9).
- **parallel correctness**: for inference fan-out, 1 worker vs 4 workers, plus one
  run with a worker killed and retried; compared by total VOI ≤ 0.01, not by
  label identity (finding 10).
- **non-degeneracy bounds** (finding 8): segmentation max id > 0; affinity
  mid-plane std > 0.01; affinity max < 0.95 (the `scale_sigmoid` range never
  reaches 0.88 in practice, so 0.95 is a ceiling check, not a fit).

### 8. Sizing report — specified inputs and outputs

*Addresses finding 12.* Inputs: volume shape, target grid, `CHUNK_SIZE`,
`seg_chunk_size_xyz`, worker count. Outputs: prepared voxels; affinity GB;
chunk count; **per-chunk peak RSS estimated as 71 GB/Gvoxel applied to one
logical chunk**, which is the only RSS model this project has measured, stated as
such; scratch GB; ABISS cap headroom per chunk; and GPU-hours at 35 min/Gvoxel
**excluding** retries and preemption, with a separately reported ×1.3 allowance
so the two are not conflated. Committed output for the 100 µm cube must match the
task table's totals to two significant figures.

## Files and Areas

| Path | Change |
|---|---|
| `connectomics/runtime/abiss_chunk.py` | read only; document semantics, do not modify |
| `tutorials/neuron_liconn_moe/gcloud/chunked_abiss.yaml` | new: config, GCS in/out, local scratch |
| `tutorials/neuron_liconn_moe/gcloud/run_volume.sh` | chunked decode stage; alignment assertion |
| `tutorials/neuron_liconn_moe/gcloud/launch.sh` | worker fan-out for inference; zone/shape fallback; cross-region refusal |
| `tutorials/neuron_liconn_moe/gcloud/vm_startup.sh` | worker role, block coverage check |
| `tutorials/neuron_liconn_moe/gcloud/equivalence_test.py` | new: the §7 metrics |
| `tutorials/neuron_liconn_moe/gcloud/size_report.py` | new: §8 |
| inference output path | chunked affinity writer, if absent |

## Verification Plan

1. `bash -n` on all shell changes; the chunked config resolves via the driver's
   dry path with no filesystem or subprocess I/O.
2. **Alignment assertion** rejects a deliberately mis-aligned
   `CHUNK_SIZE`/`seg_chunk_size_xyz` pair before allocating anything, naming both.
3. Plumbing test: one-chunk chunked decode reproduces the whole-volume decode at
   the same criterion (VOI 0; label-permutation invariant). *Not* S3 acceptance.
4. **S3 acceptance:** multi-chunk equivalence, `VOI ≤ 0.01` total and
   `VOI_boundary − VOI_interior ≤ 0.01`, at two chunk sizes.
5. Block coverage check fails a deliberately missing inference block.
6. Inference fan-out: 1 vs 4 workers, and a killed-and-retried worker, `VOI ≤ 0.01`.
7. Fallback: with an unavailable shape requested, the launcher reports which
   alternate it used; with only cross-region available, it refuses and names the
   transfer cost.
8. Sizing report runs; 100 µm totals match the task table to two significant
   figures.

## Risks and Questions

- **If S1 does not select `mean`, the chunked path may have no matching stage**
  (§6). This is the highest-consequence open item and it is outside this change's
  control; §7 surfaces it at setup rather than mid-run.
- **Alignment is a silent-failure class** the driver already guards, but only for
  the cases it checks; the added assertion is deliberately redundant with it.
- **Per-chunk RSS is modelled, not measured.** 71 GB/Gvoxel was measured on
  whole-volume decodes; whether it holds per chunk is an assumption the first
  real chunked run tests. Sizing states it as a model.
- **Scratch sizing on local SSD** is unmeasured; it may bound chunk size more
  tightly than RAM does.
- **Meshing and publication remain unsolved at 100 µm** (§Scope). Flagged, not
  fixed, and named as its own card.
- **Not closed:** whether a chunked-affinity writer exists. The target layout is
  now fixed, so this is effort, not design risk.

## Changes Since Previous Plan Version

Addresses all 6 major and 7 minor findings in `plan_v0_review.md`.

- **Findings 3, 4 — rejected as stated, and the plan changed in the opposite
  direction.** I read `abiss_chunk.py`: ABISS already performs cross-chunk
  reconciliation (`remap_watershed`, `remap_agglomeration`, `CHUNKMAP_OUTPUT`)
  and already parallelises chunk uploads. Designing halo/ownership/queue
  semantics would duplicate a working mechanism. §1 documents what exists and
  adds the alignment precondition instead; §2 moves fan-out to inference, where
  it genuinely applies, and specifies the queue properties there.
- **Finding 1 — answered without per-outcome branching** (§6): criterion and
  threshold become unset configured inputs threaded into both decoders, with the
  one real asymmetry recorded as a blocker condition.
- **Finding 2 — answered by arguing the opposite** (§5): in-region fallback with
  an explicit, costed refusal for cross-region, rather than silent omission.
- **Finding 5 — decided** (§3): precomputed on `gs://`, with dtype, channel
  order, chunk alignment and writer discipline specified.
- **Finding 6 — decided** (§4): local-SSD scratch, GCS for inputs/outputs, with
  the reasoning and the sizing consequence.
- **Findings 7, 8, 10 — made numeric** (§7): total VOI, explicit
  boundary/interior object selection, 0.01 tolerance tied to the measured noise
  floor, explicit non-degeneracy bounds.
- **Finding 9 — accepted** (§7): the one-chunk test is relabelled a plumbing
  test.
- **Finding 11 — added** (§2): block inventory, per-block completion markers and
  a pre-decode coverage check.
- **Finding 12 — specified** (§8): inputs, outputs, the RSS model named as a
  model, and retry overhead reported separately.
- **Finding 13 — accepted and scoped** (§Scope): S3/S4 deliver segmentation
  artifacts, not a published volume; publication is named as a separate card.

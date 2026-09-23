# Plan v0

## Summary

Make the decode path chunk-native and the compute path shape-agnostic, then
prove both on a volume that already has a whole-volume answer before anything at
100 µm is attempted.

The plan rests on one reframing. A 100 µm cube is not "the current pipeline, but
bigger": at 128.6 Gvoxel it is 60× over ABISS's `uint32` watershed cap and its
affinity alone is 772 GB, so the single-HDF5, single-VM, whole-volume decode has
no scaled version. But the repository already contains a chunked ABISS driver
(`connectomics/runtime/abiss_chunk.py`) and a worked configuration for a
10664×10913×5700 volume (`tutorials/neuron_j0126/3_abiss.yaml`). **The work is
therefore mostly wiring an existing chunked path into the cloud driver and
proving it equals the whole-volume answer — not building a decoder.**

The one genuinely new risk is seams, and they fail quietly: a chunked decode that
mis-handles chunk borders returns a plausible segmentation, not an error. So the
central deliverable is an **equivalence test with a stated tolerance**, designed
so that chunking is the only variable.

## Scope

**In scope — S3 and S4 of card MSIDEPLOY-SCALE-001:**

- a chunked-decode configuration and driver stage for the moe cloud pipeline,
  writing to GCS-backed precomputed volumes;
- affinity written in a chunked form rather than one HDF5 file;
- an equivalence test of chunked vs whole-volume decode, with seam checks;
- block-parallel inference and decode across many small, interchangeable spot
  VMs, with zone and machine-shape fallback and resume after preemption;
- a sizing report for a 100 µm cube in GB and GPU-hours, produced before any such
  run.

**Explicitly out of scope:** choosing the merge criterion (S1, BC job `3031548`,
an input this plan does not get to pick); the GPU/CPU stage split (S2, already
committed as `7d819bdb`); acquiring a 100 µm cube; retraining; the 36 nm-Z model.

**Not attempted in this change:** an actual 100 µm run. This plan ends at "the
path is proven at a size we can check, and sized for 100 µm".

## Proposed Changes

### 1. Affinity stops being a single HDF5 file

At 772 GB the current `raw_x1_ch0-1-2.h5` has no scaled form, and it is also the
handoff between the GPU and CPU stages added in S2.

Determine first whether `scripts/main.py` can already write a chunked affinity —
`abiss_chunk.py` recognises an "h5 chunkstore: a directory of
`chunk_z*_y*_x*.h5` written by chunked inference", which implies a producer
exists somewhere in the repo. **If it does, use it; if it does not, add a writer
rather than a post-hoc converter**, because converting 772 GB doubles both the
storage and the wall clock.

Target form: precomputed/CloudVolume on `gs://`, which `abiss_chunk.py` consumes
directly as `AFF_PATH` and which the block-parallel workers can write to
concurrently.

### 2. A chunked decode stage in the cloud driver

Add `STAGES=cpu-chunked` (or extend the existing `cpu` stage with a
`DECODE=whole|chunked` switch) that runs `scripts/run_abiss_chunk.py` against a
config modelled on `tutorials/neuron_j0126/3_abiss.yaml`:

- `AFF_PATH`, `WS_PATH`, `SEG_PATH`, `SCRATCH_PATH`, `CHUNKMAP_OUTPUT` on
  `gs://donglai/...` rather than `file://`, so workers share state;
- `BBOX` and `CHUNK_SIZE` in **XYZ** — note this differs from the ZYX convention
  used everywhere else in this project and is a live source of error;
- `AGG_THRESHOLD` left as a required input, **not defaulted**, because it lives
  in a different space from every threshold this project holds (see Risks).

### 3. The equivalence test — the acceptance gate for S3

Decode `ExPID108_32x_Cortex_L1_01` (0.26 Gvoxel, already decoded whole-volume and
published) **both ways and compare**.

The design point that makes this a real test: the chunked pipeline agglomerates
with `mean` while the published answer used `max`, so a naive comparison would
measure the criterion, not the chunking. **Run the whole-volume decoder with
`mean` as well** — `run_abiss_volume.py` accepts `--ws-merge-function` — so
chunked and whole-volume differ *only* in chunking.

Then:

- force **multiple chunks** on this small volume by setting `CHUNK_SIZE` well
  below its extent, so seams are actually exercised rather than trivially absent;
- report VOI between the two segmentations, and separately restrict the
  comparison to objects whose bounding box **crosses a chunk boundary**, which is
  where a seam defect would hide;
- repeat at two chunk sizes; a correct implementation is invariant to chunk size,
  and that invariance is a stronger signal than either single number.

### 4. Block-parallel execution across interchangeable small VMs

- **Inference:** use the existing `-ji/-jn` block chunking
  (`tools/pytc-deploy.md`) so N workers each take a block index and write into the
  shared chunked affinity.
- **Decode:** ABISS's chunked stages are parallel across chunks by construction;
  the driver currently runs them in sequence. Drive them from a work queue so
  workers are interchangeable and a lost worker is retried, not fatal.
- **Launcher:** extend `launch.sh` to start N workers of a given stage, and add
  **zone and machine-shape fallback** — measured 2026-09-22, spot
  `g2-standard-48` was capacity-exhausted in all three us-east1 zones, and
  `g2-standard-16` became exhausted in us-east1-c within hours, with quota free.
  Fallback order should try alternate zones in-region first (same-region GCS is
  free), then smaller shapes, then on-demand only if explicitly allowed.

### 5. Sizing report

A small script that, given a volume shape and target grid, prints prepared
voxels, affinity GB, chunk count, per-chunk peak RSS, ABISS cap headroom, and
GPU-hours at the measured 35 min/Gvoxel. Run it for the 100 µm cube and commit
the output alongside the plan, so criterion 3 is satisfied by construction.

## Files and Areas

| Path | Change |
|---|---|
| `connectomics/runtime/abiss_chunk.py` | read; extend only if the GCS/queue path needs it |
| `scripts/run_abiss_chunk.py` | entry point, unchanged if possible |
| `tutorials/neuron_j0126/3_abiss.yaml` | template for the new config |
| `tutorials/neuron_liconn_moe/gcloud/run_volume.sh` | add the chunked decode stage |
| `tutorials/neuron_liconn_moe/gcloud/launch.sh` | N workers; zone/shape fallback |
| `tutorials/neuron_liconn_moe/gcloud/vm_startup.sh` | worker role, queue claim, resume |
| `tutorials/neuron_liconn_moe/gcloud/` (new) | `chunked_abiss.yaml`, `equivalence_test.py`, `size_report.py` |
| `scripts/main.py` / inference output | chunked affinity writer, if absent |

## Verification Plan

1. **Config resolves and dry-runs.** `run_abiss_chunk.py` with `--dry-run`-equivalent
   resolves every path and stage without executing; `bash -n` on all shell changes.
2. **Smoke: one small volume, one chunk.** Chunked decode of ExPID108_01 with
   `CHUNK_SIZE` ≥ extent must reproduce the whole-volume `mean` decode **exactly**
   (identical label count and VOI 0 up to relabelling). This isolates the driver
   from the chunking.
3. **Equivalence: same volume, many chunks.** VOI between chunked and
   whole-volume `mean` decode **within 0.01**, and the boundary-crossing subset no
   worse than the whole-volume-interior subset by more than that tolerance.
4. **Chunk-size invariance.** Two chunk sizes agree within the same tolerance.
5. **Non-degeneracy, reused from the existing pipeline.** Segmentation not all
   zero; label count in the expected order; affinity mid-plane std > 0.01 and max
   inside the `scale_sigmoid` range.
6. **Parallel correctness.** The same volume decoded with 1 worker and with 4
   workers gives the same segmentation.
7. **Fallback behaviour.** With a deliberately unavailable shape requested, the
   launcher tries the configured alternates and reports which it used, rather
   than failing the run.
8. **Sizing report** runs and its 100 µm numbers match the task's table.

## Risks and Questions

- **The merge criterion is an input, not a choice (blocking).** The chunked
  pipeline agglomerates with `mean`, which is not monotone-invariant, so none of
  this project's operating points — the percentile rule, IST `mt = 0.47`, the
  thirteen published moe layers — transfer. S1 (job `3031548`) is settling this.
  **This plan assumes only that a criterion and a threshold will be supplied**;
  `AGG_THRESHOLD` is deliberately left unset. If S1 returns that `mean`
  underperforms `max` materially, S3 still stands — the chunked path is the only
  one that runs at scale — but the *published quality* expectation must be
  restated, and that is a decision for the card, not for this change. Note the
  j0126 config uses `AGG_THRESHOLD: 0.20` against this project's `max`-space
  0.47–0.60, which is a reminder that the numbers are not comparable, not a
  suggested default.
- **Seams fail quietly.** The whole point of §3; called out again because a
  passing smoke test with one chunk proves nothing about seams.
- **Axis order.** `BBOX`/`CHUNK_SIZE`/`resolution_xyz` are **XYZ**, while
  `volumes.py`, the prepared volumes and every deviation figure in this project
  are **ZYX**. A transposition here produces a valid-looking wrong answer.
- **GCS as ABISS scratch is unproven here.** The j0126 config uses `file://`.
  Latency, consistency and Class A operation cost at 100 µm chunk counts are not
  measured. A local-SSD scratch with GCS only for inputs/outputs may be required.
- **Open question:** does a chunked-affinity *writer* already exist, or only the
  reader? This determines whether §1 is configuration or implementation, and it
  is the largest uncertainty in the estimate.
- **Open question:** what does `CHUNKED_AGG_OUTPUT: true` do? The j0126 config
  sets it false; at 100 µm the agglomeration output may itself need chunking.
- **Cost/placement.** Buckets are US-EAST1 regional; same-region transfer is free
  and cross-region is $0.02/GiB. Fallback must prefer in-region zones, or a
  100 µm run pays cross-region on 772 GB of affinity.
- **Not addressed:** meshing and Neuroglancer publication at 100 µm. The existing
  uploader loads the whole segmentation into RAM (`_read_seg`), which is another
  60× problem. Out of scope here, but it will block the deliverable that the
  deploy spec actually promises — a link — and should become its own card.

## Changes Since Previous Plan Version

Initial plan.

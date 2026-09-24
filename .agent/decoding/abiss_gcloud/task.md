# Task

Develop what is needed to decode a **100 µm cube** with ABISS on Google Cloud.

Verbatim request: *"use the context above to develop what's needed for 100um cube"*.

## Where this sits

Card **MSIDEPLOY-SCALE-001** (in the `dw-research` repo,
`projects/msi_liconn_deploy/cards/2026-09-22_100um-cube-scale-path.md`) stages the
work S1–S5. **S1 and S2 are already submitted/built and are not this task.** This
task is **S3 and S4**:

- **S3** — chunked ABISS decode, end to end, validated as an **equivalence test**
  against a volume that already has a whole-volume answer.
- **S4** — block-parallel inference and decode across many small spot VMs.

## Sizing (measured, not estimated)

A 100 µm cube at the checkpoint's `[24, 18, 18]` nm ZYX grid is
`(4167, 5556, 5556)` = **128.6 Gvoxel**:

| quantity | value |
|---|---|
| raw uint8 at model grid | 129 GB |
| affinity, 3-channel float16 | **772 GB** |
| whole-volume ABISS RAM at the measured 71 GB/Gvoxel | **9.1 TB** |
| ABISS `uint32` watershed cap | 2.147 Gvoxel — **the cube is 60× over** |
| inference at the measured 35 min/Gvoxel | **75 GPU-hours** |

The acquisition at 32× native `[12.5, 5.08, 5.08]` nm is `(8000, 19692, 19692)`
= 3.1 Tvoxel, ~3.1 TB uint8, before any resample.

Largest volume decoded so far in this project: 0.26 Gvoxel — **605× smaller**.

## The constraint that is not about compute

Whole-volume ABISS cannot run 60× over its cap, so the decode **must** be chunked.
`connectomics/runtime/abiss_chunk.py` and `scripts/run_abiss_chunk.py` exist and
drive ABISS's CloudVolume worker stack via `agglomerate_mean_edge`.

**That is `mean`, and `mean` is not monotone-invariant.** Every operating point
this project holds — the percentile-matched rule, the IST `mt = 0.47` optimum,
all thirteen published moe layers — was established with `max` on
`scale_sigmoid`-compressed affinity, where `max`'s monotone invariance is exactly
what makes a threshold swept in compressed space meaningful. On compressed
affinity, `mean` scores a different and arbitrary criterion; uncompressing is a
**precondition** for using it.

S1 (BC job `3031548`, four arms on ExPID82 val, which has ground truth) is
settling this and is **not** part of this task. **This plan must treat the merge
criterion as an input it does not get to choose**, and must state what it assumes
and what changes if S1 returns a different answer.

## Hardware reality, measured 2026-09-22

- spot `g2-standard-48` (4× L4): `ZONE_RESOURCE_POOL_EXHAUSTED` in **all three**
  us-east1 zones;
- spot `g2-standard-16` (1× L4): available earlier the same day, then **also
  exhausted** in us-east1-c a few hours later, with quota free (2/16 preemptible
  L4 in use).

So capacity, not quota, is the binding constraint, and it moves hour to hour. A
100 µm design **must not** assume any particular GPU shape or zone is available.
Multi-zone and multi-region fallback is a requirement, not a nicety.

Buckets `gs://donglai` and `gs://donglai_public` are **US-EAST1 regional**;
same-region GCS↔GCE transfer is free and cross-region is $0.02/GiB, so compute
placement is constrained by where the data is, or the data has to move.

## What already exists and should be reused, not rebuilt

- `tutorials/neuron_liconn_moe/gcloud/` — the working single-volume cloud path:
  image as a GCS archive (Artifact Registry is not usable in this project), spot
  by default with checkpoint/resume through the run prefix, `STAGES=all|gpu|cpu`
  splitting inference from the decode, preflight that verifies region match,
  quota, IAM by role, and volume registration.
- `connectomics/runtime/abiss_chunk.py` — chunked ABISS driver: watershed, remap,
  `agglomerate_mean_edge`, CloudVolume and h5-chunkstore backends.
- `tutorials/neuron_liconn_moe/volumes.py` — `choose_axis`/`auto_recipe`, the
  per-axis resample rule and `grid_deviation`.
- `tools/pytc-deploy.md` — `python main.py -t <task>` with `-ji/-jn` block
  chunking; `tools/seung-lab.md` — igneous and task-queue.

## Success criteria

1. A chunked decode reproduces a whole-volume decode on a volume that has one,
   within the ~0.01 VOI noise floor, **seams included**. Cross-chunk seams fail
   quietly — the natural failure is a slightly different segmentation, not an
   error — so this must be an explicit equivalence test with a stated tolerance.
2. Inference and decode are block-parallel across many small, interchangeable
   spot VMs, with zone/shape fallback and resume after preemption.
3. Sizing for a 100 µm cube is **stated in GB and GPU-hours before** such a run is
   launched, not discovered during it.
4. Nothing claims that an existing `max`-based operating point transfers to the
   chunked pipeline.

## Out of scope

Acquiring a 100 µm cube (3.1 TB at 32× is a microscopy question), retraining any
model, and the 36 nm-Z model in card MSIDEPLOY-MODEL-002.

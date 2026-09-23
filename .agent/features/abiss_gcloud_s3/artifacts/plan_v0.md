# Plan v0

## Summary

Configure ABISS's existing chunked decode for the moe cloud pipeline and prove it
equals the whole-volume decode. No new decoder, no reconciliation logic, no
distribution.

The whole risk is that **seams fail quietly**: a mis-chunked decode returns a
plausible segmentation rather than an error. So the deliverable is not "a chunked
decode runs" but "a chunked decode is *demonstrably* the same answer", and every
element below exists to make that demonstrable or to fail before spending money.

Three decisions are taken here rather than deferred, because the predecessor run
failed by deferring them: where the sigmoid inversion happens, what the criterion
parameter is and when it is validated, and what "complete affinity" means as a
check rather than a hope.

## Scope

**In:** a chunked-decode config for `ExPID108_32x_Cortex_L1_01`; a
probability-space affinity converter; criterion parameterisation with a
config-resolution guard; a chunk-alignment guard; a whole-volume `mean` reference;
the equivalence test and its metrics; a decode stage in the cloud driver that runs
on one VM.

**Out:** distribution of every kind (no dispatcher, queue, worker pool, Redis, or
`$PARALLEL_CMD` change); block-parallel inference; multi-region; publication;
which criterion is scientifically preferred.

## Proposed Changes

### 1. Criterion as a parameter, validated at config resolution

`chunked_abiss.yaml` takes `merge_function` and `merge_threshold` with **no
defaults**. Both are threaded to:

- whole-volume: `run_abiss_volume.py --ws-merge-function <fn> --ws-merge-thresholds <t>`
- chunked: the stage name `agglomerate_<fn>_edge` and `param.AGG_THRESHOLD: <t>`

A guard in the config resolver maps `merge_function` to the chunked stage list
(`mean` → `agglomerate_mean_edge`; `rlme`, `cs` as ABISS names them) and **raises
before any VM is created** if the requested function has no chunked stage, with a
message naming the request and the available stages. `max` is the case this
catches: the whole-volume `ws` accepts it, the chunked path ships no `max` binary.

The equivalence test in §5 therefore runs `mean` on **both** sides. That is a
statement about the test — it holds the criterion fixed so chunking is the only
variable — not a claim that `mean` is the right criterion for a product.

### 2. Probability-space affinity: a converter, and why not the writer

`mean` is not monotone-invariant, so it requires uncompressed affinity. Decision:
for S3, a standalone converter, **not** a change to the inference writer.

- **Source artifact:** the existing compressed affinity written by the current
  pipeline, `<test_out>/<volume>/raw_x1_ch0-1-2.h5`, dataset `main`, `(3,Z,Y,X)`
  float16. For `ExPID108_32x_Cortex_L1_01` this already exists from the
  2026-09-22 run.
- **Transform:** `p = sigmoid(logit(v)/0.2)`, i.e. the inverse of
  `channel_activations: scale_sigmoid`. **Reuse
  `tutorials/neuron_liconn_ist/merge_fn_sweep.py::uncompress` by importing it**
  rather than reimplementing, so the tail-clipping constant `EPS` has exactly one
  definition — S1's verified result depends on that clipping and a second copy
  could drift from it.
- **Output:** a CloudVolume precomputed layer, `float32`, 3 channels, storage
  chunk equal to `seg_chunk_size_xyz`, `voxel_offset` = `BBOX[:3]`, `volume_size`
  = `BBOX[3:] − BBOX[:3]`, `resolution` = the prepared volume's effective spacing
  in **XYZ**. Path `<run_prefix>/aff_prob/`. `AFF_CHANNELS: [2,1,0]` for ABISS,
  because this repo stores channel `c` as the edge along array axis `c` (ZYX)
  while `ws` reads XYZC with channel 0 = X.
- **float32, not float16:** after inversion the probabilities span roughly
  10⁻³–1, and `mean` weights the low tail that float16 resolves poorly. At S3
  scale this costs ~2.5 GB.
- **Why a converter and not the inference writer:** at S3 the affinity already
  exists, so converting is minutes and changes nothing upstream. At 100 µm the
  same choice would cost a 1.5 TB second copy and the inversion belongs in the
  writer — that is S4's problem, and this plan states the boundary rather than
  pretending the converter scales.

### 3. Chunk alignment, asserted before allocation

`param.CHUNK_SIZE` and `seg_chunk_size_xyz` are both **XYZ**, unlike the rest of
this project. The driver already raises on misalignment; this change recomputes it
in our resolver so it fails on the workstation, not on a running VM, naming both
values and the offending axis.

Defaults: `CHUNK_SIZE [2048, 2048, 80]`, `seg_chunk_size_xyz [512, 512, 80]` —
the j0126 pairing, known to satisfy the rule.

### 4. Affinity completeness — a chunk-count check, not a spot check

"Readable and non-empty at a bounding box" can pass on a partially written layer.
Instead, compare **actual to expected chunk objects**, the same technique that
caught a 24%-complete ingest on 2026-09-22:

```
expected = prod(ceil(volume_size[i] / storage_chunk[i]) for i in 0..2)
```

counted against the layer's existing keys. The decode refuses to start unless they
match, and the check is part of the config resolver so it also runs on the
workstation. It additionally asserts the layer's `volume_size` and `resolution`
equal the prepared volume's.

### 5. Equivalence test — the acceptance gate

Volume `ExPID108_32x_Cortex_L1_01`, 0.26 Gvoxel. A **new whole-volume `mean`
reference** is produced (`run_abiss_volume.py --ws-merge-function mean`, one CPU
decode, minutes). The published `max` layer is not the reference and is not
touched.

- **Metric:** VOI on **masked volumes**, not label subsets. For a label set `S`,
  mask to voxels whose whole-volume label ∈ `S` and compute
  `VOI(chunked|mask, whole|mask)` as total = split + merge, reporting all three.
  Both sides are then the same quantity on the same voxels, which is what a
  difference requires.
- **Sets:** `B` = whole-volume labels whose bounding box intersects any plane at a
  multiple of `CHUNK_SIZE` on any axis; `I` = all others.
- **Acceptance:** `VOI_total(chunked, whole) ≤ 0.01` over the full volume **and**
  `VOI_total(B) − VOI_total(I) ≤ 0.01`.
- **Chunk-size invariance:** two alignment-valid `CHUNK_SIZE` values agree within
  0.01. A correct implementation cannot depend on chunk size, so this is the
  strongest single signal.
- **One-chunk run:** a plumbing test, explicitly not acceptance.
- 0.01 is this project's measured noise floor: two 18 nm runs differing only in
  GPU count differ by 0.006 whole-val VOI.

### 6. Decode stage in the cloud driver

`STAGES=cpu` gains `DECODE=whole|chunked` (default `whole`, preserving today's
behaviour). `chunked` runs `scripts/run_abiss_chunk.py` with §1–§4's config on a
single high-memory CPU VM, using ABISS's built-in `parallel -j ncpus` across cores
unchanged. `SCRATCH_PATH` and `CHUNKMAP_OUTPUT` on the VM's local disk; `AFF_PATH`,
`WS_PATH`, `SEG_PATH` on `gs://`.

## Files and Areas

| Path | Change |
|---|---|
| `connectomics/runtime/abiss_chunk.py` | read only |
| `tutorials/neuron_liconn_moe/gcloud/chunked_abiss.yaml` | new: config + the §1/§3 guards' inputs |
| `tutorials/neuron_liconn_moe/gcloud/make_prob_affinity.py` | new: §2 converter |
| `tutorials/neuron_liconn_moe/gcloud/resolve_chunked.py` | new: criterion, alignment and completeness guards |
| `tutorials/neuron_liconn_moe/gcloud/equivalence_test.py` | new: §5 |
| `tutorials/neuron_liconn_moe/gcloud/run_volume.sh` | `DECODE=whole\|chunked` |
| `tutorials/neuron_liconn_ist/merge_fn_sweep.py` | imported for `uncompress`; not modified |

## Verification Plan

1. `bash -n` on shell changes; `chunked_abiss.yaml` resolves on the driver's dry
   path with no filesystem or subprocess I/O.
2. **Criterion guard:** `merge_function: max` raises at resolution, naming `max`
   and the available chunked stages. `merge_function: mean` resolves.
3. **Alignment guard:** `CHUNK_SIZE [2048,2048,80]` with
   `seg_chunk_size_xyz [512,512,96]` raises, naming both and the Z axis.
4. **Completeness guard:** deleting one chunk object from the `aff_prob` layer
   makes resolution fail with expected-vs-actual counts.
5. **§2 integrity:** a whole-volume `max` decode of `aff_prob` at the mapped
   threshold reproduces the compressed-artifact `max` result within VOI 0.001.
   This re-runs S1's measured invariance on the real artifact.
6. **Plumbing:** one-chunk chunked decode reproduces whole-volume `mean`,
   `VOI_total = 0`.
7. **Acceptance:** multi-chunk equivalence at `[2048,2048,80]` and a second
   alignment-valid size — `VOI_total ≤ 0.01` and
   `VOI_total(B) − VOI_total(I) ≤ 0.01` for both.
8. Non-degeneracy: chunked segmentation max id > 0; `aff_prob` min > 0 and
   max < 1.

## Risks and Questions

- **Per-chunk RSS is a model**, 71 GB/Gvoxel extrapolated from whole-volume
  measurements. The first chunked run tests it; at `[2048,2048,80]` the modelled
  24 GB/chunk leaves wide margin on any high-memory VM.
- **`AGG_THRESHOLD` has no established value.** S1 measured the *ordering* of
  criteria, but all its optima sat at the bottom edge of the swept range, so its
  `mean` threshold is not an optimum. For S3 this does not matter — both sides use
  the same value and the test is an equivalence, not a quality claim — but the
  number must not be quoted as tuned.
- **The converter does not scale** and is scoped to S3 deliberately (§2).
- **`rlme` and `cs` are unmapped.** §1's guard lists them as available chunked
  stages, but this plan does not establish what they compute; only `mean` is
  exercised.
- **Publication and distribution remain out of scope**, each needing its own run.

## Changes Since Previous Plan Version

Initial plan.

# Plan v2

## Summary

Two findings remained, both concrete inputs rather than design. This version
supplies them: the whole-volume `mean` reference is produced by the **same
`DECODE=whole` path already in the driver** rather than a test-only route, and
every placeholder is replaced by a real, verified object path.

The affinity source deserves a note. The VM that produced the 2026-09-22 ExPID108
affinity is deleted, so the container path `plan_v1` cited is provenance, not an
input. The durable artifact is the published copy in GCS, **verified present**:
`gs://donglai_public/liconn/moe/expid108/affinity/ExPID108_32x_Cortex_L1_01_affinity_x1_ch0-1-2.h5`,
996,454,624 bytes, written 2026-09-21T20:22:33Z. That object is the single input
to the converter.

## Scope

Unchanged. S3 only, single VM. Distribution, block-parallel inference,
multi-region, publication and criterion preference remain out of scope.

## Proposed Changes

### 1. The whole-volume `mean` reference — one decode path, run twice

*Finding 1.* No test-only script. The reference is produced by the driver's
existing whole-volume decode, invoked with the same configured criterion and
threshold the chunked run uses:

```
DECODE=whole  merge_function=mean  merge_threshold=0.1394546
  -> scripts/run_abiss_volume.py
       --input  <aff_prob as h5, see note>   --output <ref_out>
       --ws-merge-function mean --ws-merge-thresholds 0.1394546
       --ws-high-threshold 94% --ws-low-threshold 20%
       --ws-size-threshold 10000000 --ws-dust-threshold 200
       --channels 2,1,0 --edge-storage source
```

**Ordering, in the implementation contract:**
`resolve` → `preflight` → converter (§3) → whole-volume `mean` reference →
chunked decode(s) → `equivalence_test.py`. The equivalence test takes the
reference path as a required argument and fails if absent; it never decodes
anything itself.

**Note on the input.** `run_abiss_volume.py` reads an HDF5 affinity, while ABISS's
chunked path reads a precomputed layer. To keep both sides on *identical* data,
the converter (§3) writes **both** representations from one pass: the precomputed
layer for the chunked decode and an HDF5 copy for the whole-volume decode. They
are byte-identical in value by construction, and a check asserts equal shape,
dtype and a matching checksum over a fixed sample of voxels.

Output: `<run_prefix>/ref_mean/seg.h5`, dataset `main`, uint32.

### 2. Concrete paths — no placeholders

*Finding 2.* Fixed values for the S3 test, all verified to exist except those this
change creates:

| role | value |
|---|---|
| volume | `ExPID108_32x_Cortex_L1_01` |
| prepared shape ZYX | `(503, 650, 650)` at `[23.9811, 18.0, 18.0]` nm |
| **`BBOX` (XYZ, inclusive-exclusive)** | `[0, 0, 0, 650, 650, 503]` |
| `resolution_xyz` (nm) | `[18.0, 18.0, 23.9811]` |
| **affinity source (verified, 996,454,624 B)** | `gs://donglai_public/liconn/moe/expid108/affinity/ExPID108_32x_Cortex_L1_01_affinity_x1_ch0-1-2.h5` |
| run prefix | `gs://donglai/liconn/moe/runs/${RUN_ID}` with `RUN_ID` from `launch.sh`, as today |
| `aff_prob` (precomputed, created) | `${RUN_PREFIX}/aff_prob/` |
| `aff_prob` (h5 twin, created) | `${RUN_PREFIX}/aff_prob/aff_prob.h5` |
| whole-volume `mean` reference (created) | `${RUN_PREFIX}/ref_mean/seg.h5` |
| chunked outputs (created) | `${RUN_PREFIX}/chunked_<tag>/{ws,seg,chunkmap}` |
| scratch | VM local disk, `/work/abiss_scratch` |
| published `max` layer — **provenance only, never read** | `gs://donglai_public/liconn/moe/expid108/seg/ExPID108_32x_Cortex_L1_01_seg_abiss_mt0550.h5` |

`BBOX` is derived from the prepared volume's shape reversed to XYZ, and `resolve`
asserts it equals the affinity's own dimensions rather than trusting the table.

### 3. Converter — unchanged contract, two outputs

As `plan_v1` §6, with the §1 addition that it emits both the precomputed layer and
the HDF5 twin in one pass. Source is the GCS object in §2; clipping
`EPS = 1e-7`, then `p = sigmoid(logit(v)/0.2)` via the imported
`merge_fn_sweep.py::uncompress`; float32; `AFF_CHANNELS [2,1,0]`; storage chunk =
`seg_chunk_size_xyz`; `voxel_offset`/`volume_size` from `BBOX`.

### 4. Guards run before any allocation

*Finding 4.* Stated as an implementation contract, not an aspiration: the decode
entry point calls `resolve` and then `preflight` **as its first two actions**, and
returns non-zero on either failure **before** `_prepare_segmentation_output_layers`
or any CloudVolume `create_new_info` call. A test asserts that a misaligned config
produces no `ws`/`seg` layer on disk.

### 5. Error message content

*Finding 3.* The unsupported-criterion error contains, on separate lines: the
requested value; the accepted set `{mean}`; the chunked stage that backs it,
`agglomerate_mean_edge`; and, when the request is `max`, that ABISS's chunked path
ships no `max` binary while the whole-volume `ws` accepts one.

### 6. Everything else unchanged from `plan_v1`

Criterion set `{mean}` (§1 of v1); `AGG_THRESHOLD = 0.1394546`, fixed not tuned;
test chunk sizes A `[256,256,128]`/`[128,128,128]` and B `[192,192,96]`/`[64,64,96]`
with the ≥2-chunks-per-axis assertion; `resolve` pure / `preflight` I/O; exact
key-set comparison in both directions; the equivalence metric on masked volumes
with half-open boxes, interior planes only and background `0` excluded;
acceptance `VOI_total ≤ 0.01` and `VOI_total(B) − VOI_total(I) ≤ 0.01` for each
configuration, compared against the reference.

## Files and Areas

| Path | Change |
|---|---|
| `connectomics/runtime/abiss_chunk.py` | read only |
| `tutorials/neuron_liconn_moe/gcloud/chunked_abiss.yaml` | new: §2 paths, §6 values |
| `tutorials/neuron_liconn_moe/gcloud/make_prob_affinity.py` | new: §3, two outputs |
| `tutorials/neuron_liconn_moe/gcloud/resolve_chunked.py` | new: `resolve` + `preflight`, §4 ordering, §5 messages |
| `tutorials/neuron_liconn_moe/gcloud/equivalence_test.py` | new: takes the reference path as a required argument |
| `tutorials/neuron_liconn_moe/gcloud/run_volume.sh` | `DECODE=whole\|chunked`; §1 ordering |
| `tutorials/neuron_liconn_ist/merge_fn_sweep.py` | imported for `uncompress`/`EPS`; not modified |

## Verification Plan

1. `bash -n`; `resolve` runs with no filesystem or network I/O.
2. Criterion guard: `max`, `rlme`, `cs` raise with all of §5's lines; `mean`
   resolves.
3. Alignment guard raises on `[256,256,128]` with `[128,128,96]`, naming the Z axis.
4. ≥2-chunks-per-axis guard raises on `[2048,2048,80]`, reporting `[1,1,7]`.
5. **§4 ordering:** after a guard failure, no `ws`/`seg` layer exists.
6. Key-set guard: a deleted chunk fails naming the missing key; a stray key fails
   naming the extra.
7. `EPS == 1e-7` pin test.
8. **Twin check:** the precomputed layer and the HDF5 twin agree in shape, dtype
   and a fixed-sample checksum.
9. **Integrity:** whole-volume `max` on `aff_prob` at `0.35417863` reproduces the
   compressed-source `max` at `0.47` within `VOI_total ≤ 0.001`.
10. **Reference exists:** `equivalence_test.py` fails cleanly when
    `${RUN_PREFIX}/ref_mean/seg.h5` is absent.
11. Plumbing: one-chunk decode reproduces the reference, `VOI_total = 0`.
12. **Acceptance:** tests A and B each give `VOI_total ≤ 0.01` and
    `VOI_total(B) − VOI_total(I) ≤ 0.01`; the two agree within 0.01.
13. Non-degeneracy: chunked seg max id > 0; `aff_prob` min > 0, max < 1.

## Risks and Questions

- **The HDF5 twin doubles converter output** (~2.5 GB here). Accepted at S3 scale
  to keep both decoders on identical data; at 100 µm the whole-volume side does
  not exist, so the twin is not needed and must not be inherited.
- **`AGG_THRESHOLD = 0.1394546` is fixed, not tuned.** Valid for equivalence only.
- **Per-chunk RSS is a model**; test A's 8.4 Mvox chunks cannot validate it at
  100 µm scale.
- **Only `mean` is exercised**; `rlme`/`cs` remain unestablished.
- **Test-scale seams are not 100 µm seams.** 36 and 96 chunks exercise the
  mechanism; passing is necessary, not sufficient.

## Changes Since Previous Plan Version

Closes both major and both minor findings in `plan_v1_review.md`.

- **Finding 1 — the reference is produced by the existing `DECODE=whole` path**
  (§1), run once at `mean`/`0.1394546`, with the full command, the output artifact
  `${RUN_PREFIX}/ref_mean/seg.h5`, and the pipeline ordering stated. The converter
  now emits an HDF5 twin so both decoders read identical values, with a check.
- **Finding 2 — every placeholder replaced** (§2), including the verified GCS
  affinity object and its byte count, `BBOX [0,0,0,650,650,503]`, the resolution,
  and the published `max` layer explicitly marked provenance-only.
- **Finding 3 — error contents enumerated** (§5), including
  `agglomerate_mean_edge`.
- **Finding 4 — ordering made an implementation contract** (§4), with a test that
  no layer is created after a guard failure.

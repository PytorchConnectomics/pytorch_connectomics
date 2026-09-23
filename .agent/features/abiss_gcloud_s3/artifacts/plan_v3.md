# Plan v3

## Summary

`plan_v2_review` left two blocking findings, and each came down to an artifact
that `plan_v2` mentioned but never defined. This version defines both.

Closing finding 2 turned up a real defect in `plan_v2`, not just a missing
specification. The two decoders **do not interpret the same stored tensor the
same way**:

- The whole-volume path (`scripts/run_abiss_volume.py`) reads CZYX, reorders
  channels with `--channels 2,1,0`, and moves edges from source to destination
  storage with `--edge-storage source`
  (`_shift_to_destination_storage`, `run_abiss_volume.py:148`).
- The chunked path copies the source HDF5 into precomputed **unchanged**
  (`abiss_chunk.py` ~L680–694, `np.transpose(block, (3,2,1,0))` only). ABISS's
  `cut_chunk_common.cut_data` then takes channels `0..2` as they are, with no
  shift. `_aff_channels` (`abiss_chunk.py:49`) also **rejects** any non-prefix
  `AFF_CHANNELS`, so the `AFF_CHANNELS [2,1,0]` that `plan_v1`/`plan_v2`
  specified would have failed.

Under `plan_v2`, then, the two decoders would have read the same bytes with
different axis and edge meanings. The equivalence test would have measured that
mismatch, not chunking.

The fix is to have the converter **bake** the reordering, the edge shift and the
probability inversion into **one canonical HDF5**. Both decoders consume that
file with no-op flags. The precomputed "twin" and its checksum check go away,
because the chunked driver already builds its precomputed layer from an HDF5
source (`source_affinity_h5`). Finding 1 closes by adding a compressed `max`
decode that this run produces itself. That comparison also becomes the
end-to-end proof that the canonical tensor has the right semantics.

## Scope

Unchanged. S3 only, on a single VM. Out of scope: distribution, block-parallel
inference, multi-region, publication, and choosing which criterion is preferred.

## Proposed Changes

### 1. One canonical affinity artifact, `aff_canon.h5` (finding 2)

`make_prob_affinity.py` reads the verified GCS source
`gs://donglai_public/liconn/moe/expid108/affinity/ExPID108_32x_Cortex_L1_01_affinity_x1_ch0-1-2.h5`
(dataset discovered as its single dataset; CZYX, 3 channels, compressed
`sigmoid(0.2·logit(p))`). It writes `${RUN_PREFIX}/aff_canon/aff_canon.h5` with
this schema:

| property | value |
|---|---|
| dataset | `main` (the only dataset in the file) |
| shape | `(3, Z, Y, X)` = `(3, 503, 650, 650)` |
| dtype | `float32` |
| channel `k` | edge along **ABISS spatial axis `k`**, where `k=0` is X, `1` is Y and `2` is Z. This equals the source channel `2-k`, i.e. the `--channels 2,1,0` reversal applied once, at conversion |
| edge storage | **destination**: edge `(i-1, i)` stored at `i`. This is `_shift_to_destination_storage` applied once, on the **whole volume**, with the exposed face at index 0 zeroed |
| values | probability `p = sigmoid(logit(clip(v, EPS, 1-EPS)) / 0.2)`, `EPS = 1e-7`, via the imported `merge_fn_sweep.py::uncompress`. The zeroed face stays exactly `0` |

Order of operations: read, then reverse channels, then shift to destination
(both reused **by import** from `run_abiss_volume.py`, not reimplemented), then
invert every voxel except the zeroed face, which stays exactly `0`. The
inversion is elementwise, so this equals inverting first and then shifting, which
is what the whole-volume path does today.

**Why the shift is global.** Shifting before chunking gives every chunk-seam
voxel its true neighbour edge. A per-chunk shift would zero an artificial face at
every seam, which is precisely the quiet seam defect this run exists to catch.

**Both decoders consume it with no transform:**

- whole-volume: `run_abiss_volume.py --input aff_canon.h5 --input-dataset main
  --channels 0,1,2 --edge-storage destination` (identity selection, no shift);
- chunked: `abiss.source_affinity_h5: aff_canon.h5`, `source_dataset: main`,
  `AFF_CHANNELS: [0,1,2]`. The driver's existing HDF5→precomputed copy makes the
  layer at `${RUN_PREFIX}/chunked_<tag>/aff/`, with storage chunk
  `seg_chunk_size_xyz`, and `cut_data` passes channels `0..2` through unchanged.

A **layout unit test** pins the equivalence directly: on a random compressed
CZYX array `a`, `run_abiss_volume._to_abiss_affinity(uncompress(a),
channels=[2,1,0], edge_storage="source")` must equal
`_to_abiss_affinity(canon(a), channels=[0,1,2], edge_storage="destination")`
bitwise. So the whole-volume path gives the same ABISS mmap either way, and the
chunked path reads that same `(x,y,z,c)` tensor.

### 2. Integrity check on artifacts this run creates (finding 1)

The same `DECODE=whole` path runs three times. Every input and output below is
created by this run, and the published layer is never read.

| tag | input | flags | criterion / threshold | output |
|---|---|---|---|---|
| `ref_max_compressed` | GCS source (compressed, raw layout) | `--channels 2,1,0 --edge-storage source` | `max` / `0.47` | `${RUN_PREFIX}/ref_max_compressed/seg.h5` |
| `ref_max_canon` | `aff_canon.h5` | `--channels 0,1,2 --edge-storage destination` | `max` / `0.35417863` | `${RUN_PREFIX}/ref_max_canon/seg.h5` |
| `ref_mean` | `aff_canon.h5` | `--channels 0,1,2 --edge-storage destination` | `mean` / `0.1394546` | `${RUN_PREFIX}/ref_mean/seg.h5` |

The three runs share `--ws-high-threshold 94% --ws-low-threshold 20%
--ws-size-threshold 10000000 --ws-dust-threshold 200`. Here
`0.35417863 = sigmoid(logit(0.47)/0.2)`, recomputed as a pinned constant in the
test.

**Integrity check:** `VOI_total(ref_max_canon, ref_max_compressed) ≤ 0.001`.
Because `max` is monotone-invariant and the percentile watershed thresholds are
too, this passes only if the inversion is correct **and** the baked reordering
and shift match what the decoder flags do. It is S1's measured invariance, re-run
on the real artifact, and it end-to-end checks §1's schema. A failure here stops
the pipeline before any chunked decode.

The published `max` layer stays **provenance only** and is not an input to any
check.

### 3. Guards: `resolve` pure, `preflight` does I/O (minor finding 3)

- `resolve` (no I/O at all) checks: the criterion is in `{mean}`, with §5's
  error text; `CHUNK_SIZE` is a multiple of `seg_chunk_size_xyz` on every axis,
  naming both numbers and the axis; there are ≥2 chunks per axis given `BBOX`;
  and `AFF_CHANNELS == [0,1,2]`.
- `preflight` (reads artifact headers) checks: the `aff_canon.h5` dataset `main`
  is `(3, Z, Y, X)` float32 with `(X, Y, Z)` equal to `BBOX[3:]`, and the source
  object exists. **The `BBOX`-vs-artifact assertion moves here** from `resolve`.

Ordering contract, unchanged from `plan_v2` §4: the decode entry point calls
`resolve`, then `preflight`, as its first two actions. It exits non-zero before
any `_prepare_segmentation_output_layers` or CloudVolume `create_new_info` call.

### 4. Boundary and interior masks, defined exactly (minor finding 4)

Let `W` be `ref_mean` and `C` be a chunked output, both ZYX `uint32` over the
same `BBOX`, with `C` read back from its precomputed layer into ZYX.

- **Interior planes** along an axis of length `D` with chunk size `S` are
  `P = {k·S : k ≥ 1, k·S < D}`. A plane `p` separates voxels `p-1` and `p`.
  The volume faces `0` and `D` are not planes.
- For each nonzero label `ℓ` in `W`, its half-open box on each axis is
  `[lo, hi)`. `ℓ` **crosses** plane `p` iff `lo < p < hi`, i.e. it has voxels on
  both sides. `B` = labels that cross at least one interior plane on any axis;
  `I` = all other nonzero labels of `W`.
- **Masks:** `M_B = isin(W, B)`, `M_I = isin(W, I)`. `W == 0` is excluded from
  both. `VOI_total(X) = VOI(W[M_X], C[M_X])`, split plus merge, natural log,
  computed on the same voxel set for both volumes. A `0` in `C` inside a mask is
  counted as its own label, not dropped, so any unassigned voxels from a seam
  defect still register.
- `VOI_total` over everything is `VOI(W[W≠0], C[W≠0])`.

Acceptance, per configuration: `VOI_total ≤ 0.01` and
`VOI_total(B) − VOI_total(I) ≤ 0.01`. The two configurations must also agree with
each other within 0.01 (`VOI_total(C_A, C_B) ≤ 0.01`, same masking).

### 5. Error message content — unchanged from `plan_v2` §5

It gives the requested value, the accepted set `{mean}`, the backing stage
`agglomerate_mean_edge`, and, for `max`, a note that the chunked path ships no
`max` binary while whole-volume `ws` accepts one.

### 6. Unchanged from `plan_v2`

Criterion set `{mean}`; `AGG_THRESHOLD = 0.1394546`, fixed rather than tuned;
test chunk sizes A `[256,256,128]`/`[128,128,128]` and B
`[192,192,96]`/`[64,64,96]`; the exact key-set comparison of task flags in both
directions; paths and `BBOX [0,0,0,650,650,503]`, resolution
`[18.0,18.0,23.9811]`, run prefix and scratch as in `plan_v2` §2, **except** that
`aff_prob/` and `aff_prob/aff_prob.h5` are replaced by
`${RUN_PREFIX}/aff_canon/aff_canon.h5` and each chunked run's own
`${RUN_PREFIX}/chunked_<tag>/aff/`.

Pipeline order: `resolve` → `preflight` → converter → `ref_max_compressed`,
`ref_max_canon` → **integrity gate** → `ref_mean` → chunked A, B →
`equivalence_test.py` (which takes the reference path as a required argument and
never decodes).

## Files and Areas

| Path | Change |
|---|---|
| `connectomics/runtime/abiss_chunk.py` | read only (`source_affinity_h5` copy, `_aff_channels`) |
| `scripts/run_abiss_volume.py` | read only; `_select_affinity_channels`, `_shift_to_destination_storage`, `_to_abiss_affinity` imported |
| `tutorials/neuron_liconn_moe/gcloud/chunked_abiss.yaml` | new: §1 chunked keys, §6 values |
| `tutorials/neuron_liconn_moe/gcloud/make_prob_affinity.py` | new: §1, one output |
| `tutorials/neuron_liconn_moe/gcloud/resolve_chunked.py` | new: §3 `resolve`/`preflight`, §5 messages |
| `tutorials/neuron_liconn_moe/gcloud/equivalence_test.py` | new: §4 masks and acceptance; integrity metric of §2 |
| `tutorials/neuron_liconn_moe/gcloud/run_volume.sh` | `DECODE=whole\|chunked`; §2 three references and gate; §6 order |
| `tests/unit/test_abiss_s3_chunked.py` | new: offline tests 2–8 below |
| `tutorials/neuron_liconn_ist/merge_fn_sweep.py` | imported for `uncompress`/`EPS`; not modified |

## Verification Plan

Offline (unit tests, no cloud):

1. `bash -n run_volume.sh`; `resolve` runs with filesystem and network access
   patched to raise.
2. Criterion guard: `max`, `rlme` and `cs` raise with every line of §5; `mean`
   resolves.
3. Alignment guard raises on `[256,256,128]` with `[128,128,96]`, naming Z and
   both numbers. The ≥2-chunks guard raises on `[2048,2048,80]` and reports
   `[1,1,7]`. `AFF_CHANNELS [2,1,0]` is rejected.
4. **Layout equivalence (§1):** the bitwise test on random data of shape
   `(3, 7, 9, 11)`, plus a test that `aff_canon`'s index-0 faces are exactly 0.
5. **Global vs per-chunk shift:** splitting a random volume at an interior plane,
   shifting each half separately and concatenating differs from the global shift
   exactly on that plane's face. This documents why §1 is global.
6. **Masks (§4):** on a synthetic 1-D-separable case, a label spanning a plane
   lands in `B` and one ending exactly at `p` lands in `I`. A deliberately split
   seam label raises `VOI_total(B)` and leaves `VOI_total(I) = 0`.
7. Ordering: after a guard failure, no `ws`/`seg`/`aff` layer exists under the
   run prefix (local `file://` prefix). The key-set guard names missing and extra
   keys.
8. Pins: `EPS == 1e-7`; `sigmoid(logit(0.47)/0.2) == 0.35417863` to 1e-8.

On the VM (ExPID108):

9. `preflight` passes on `aff_canon.h5`; `min > 0` off the zero face, `max < 1`.
10. **Integrity gate:** `VOI_total(ref_max_canon, ref_max_compressed) ≤ 0.001`.
11. `equivalence_test.py` fails cleanly when `ref_mean/seg.h5` is absent.
12. Plumbing: a one-chunk run reproduces `ref_mean` with `VOI_total = 0`. This is
    not acceptance.
13. **Acceptance:** A and B each meet `VOI_total ≤ 0.01` and
    `VOI_total(B) − VOI_total(I) ≤ 0.01`, and `VOI_total(C_A, C_B) ≤ 0.01`.
14. Non-degeneracy: every segmentation has max id > 0, and `|B| > 0` for both A
    and B (otherwise the seam test is vacuous).

## Risks and Questions

- **ABISS's chunked `ws` assumes destination storage** and the same channel-axis
  mapping as whole-volume `ws`. Both invoke the same `ws` binary on an
  `aff.raw` mmap, and `cut_data` passes channels through unchanged, so this
  should hold. It is still *inferred from code, not measured*. Test 12 (one chunk
  must reproduce `ref_mean` exactly) is the measurement, and it runs before
  acceptance.
- **Zeroed face at index 0.** Both decoders see the same zeros, so equivalence is
  unaffected. The percentile watershed thresholds include these voxels on both
  sides.
- `aff_canon.h5` is ~2.5 GB, and at 100 µm it would be the only affinity artifact
  (the compressed source and the whole-volume references do not exist there). The
  converter must therefore stream by Z-slab. The global shift needs a one-slab
  halo along Z only, and streaming is required at 100 µm either way.
- Unchanged from `plan_v2`: `AGG_THRESHOLD` is fixed; per-chunk RSS is a model;
  only `mean` is exercised; test-scale seams are not 100 µm seams.

## Changes Since Previous Plan Version

This version closes both major and both minor findings of `plan_v2_review.md`,
and fixes one defect found while closing finding 2.

- **Finding 1 (major), no comparison artifact.** §2 adds `ref_max_compressed`, a
  `DECODE=whole` `max`/`0.47` decode of the compressed source that this run
  produces. The integrity gate compares it with `ref_max_canon`. No check reads
  the published layer.
- **Finding 2 (major), twin schema underspecified.** §1 replaces the HDF5 twin
  and precomputed twin with a single `aff_canon.h5`. Its full schema is stated:
  dataset `main`, `(3,Z,Y,X)`, float32, channel `k` = ABISS axis `k`, destination
  storage, probability values. Both decoders read it with no-op flags, with a
  bitwise layout unit test and the §2 end-to-end gate.
- **Defect found while closing finding 2.** `plan_v1`/`plan_v2`'s
  `AFF_CHANNELS [2,1,0]` is rejected by `_aff_channels`. The chunked copy also
  applies neither the reversal nor the destination shift, so the two decoders
  would have read the same data with different semantics. Baking both transforms
  into `aff_canon.h5`, globally, fixes this.
- **Finding 3 (minor).** The `BBOX`-vs-artifact assertion moves from `resolve` to
  `preflight` (§3), so `resolve` is pure.
- **Finding 4 (minor).** §4 defines interior planes, the crossing rule for
  half-open label boxes, masks `M_B`/`M_I`, the handling of `0` in each volume,
  and the VOI formula exactly.

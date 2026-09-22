# Plan v1

## Summary

Every `plan_v0` finding was a missing value or definition rather than a missing
design, so this version supplies them. It also fixes a defect the review did not
catch and that would have voided acceptance: **the proposed chunk size produces a
single chunk in X and Y on the test volume**, so the "multi-chunk" test would have
exercised only Z seams.

That defect is worth naming plainly, because it is the plan reproducing the exact
failure mode it exists to catch — a seam problem passing silently. Test chunk
sizes are now chosen against the *test volume* and required to be multi-chunk on
all three axes; 100 µm chunk sizes are a separate table and are not used for the
test.

## Scope

Unchanged from `plan_v0`. S3 only, single VM. Distribution, block-parallel
inference, multi-region, publication and criterion *preference* remain out.

## Proposed Changes

### 1. Criterion: accepted set is exactly `{mean}`

*Finding 7.* `rlme` and `cs` exist as ABISS chunked stages but their threshold
semantics are unestablished, and this change exercises neither. Accepting them
would advertise support that is untested.

Canonical mapping in the resolver:

| `merge_function` | chunked stage | whole-volume flag |
|---|---|---|
| `mean` | `agglomerate_mean_edge` | `--ws-merge-function mean` |

Everything else — including `max`, `rlme`, `cs` — raises at resolution with:
the requested value, the accepted set `{mean}`, and, when the request is `max`,
the explanatory line that ABISS's chunked path ships no `max` binary while the
whole-volume `ws` accepts one. Adding `rlme`/`cs` later is a table entry plus a
threshold-semantics measurement.

### 2. `AGG_THRESHOLD` is fixed at 0.1394546

*Finding 1.* The value is S1's `mean` arm best over six ExPID82 cubes
(0.13945465, which is `sigmoid(logit(0.41)/0.2)`).

**It is fixed for equivalence, not tuned.** S1's optima all sat at the bottom edge
of the swept range, so this is not an optimum and must never be quoted as one. It
is legitimate here because both sides of the comparison use the identical value
and the test asks whether *chunking* changes the answer.

### 3. Test chunk sizes, chosen against the test volume

*Finding 2, and the defect above.* `ExPID108_32x_Cortex_L1_01` prepares to
`(503, 650, 650)` ZYX = **`(650, 650, 503)` XYZ**.

| role | `CHUNK_SIZE` XYZ | `seg_chunk_size_xyz` | chunks XYZ | total | aligned |
|---|---|---|---|---|---|
| **test A** | `[256, 256, 128]` | `[128, 128, 128]` | 3 × 3 × 4 | 36 | yes |
| **test B** | `[192, 192, 96]` | `[64, 64, 96]` | 4 × 4 × 6 | 96 | yes |
| plumbing | `[1024, 1024, 512]` | `[128, 128, 128]` | 1 × 1 × 1 | 1 | yes |
| *100 µm (not used here)* | `[2048, 2048, 80]` | `[512, 512, 80]` | — | 477 | yes |

**Requirement, asserted in code:** an acceptance configuration must yield ≥ 2
chunks on **every** axis. The resolver computes `ceil(extent/chunk)` per axis and
raises otherwise. `[2048, 2048, 80]` on this volume gives `[1, 1, 7]` and is
therefore rejected for acceptance — which is the check that would have caught the
original error.

### 4. Resolve versus preflight — a stated boundary

*Finding 3.* Two phases, two commands, in order:

- **`resolve`** — pure. Parses the config, applies §1's criterion mapping, §3's
  alignment and ≥2-chunks-per-axis checks, and computes the expected affinity key
  set. **No filesystem or network I/O.** Unit-testable, runs on the workstation.
- **`preflight`** — does I/O. Lists the `aff_prob` layer's actual keys, compares
  against the set `resolve` produced, checks the layer's `volume_size`/`resolution`
  against the prepared volume, and verifies write permission on the output paths.
  Runs on the workstation before `launch.sh --run`, and again on the VM before the
  decode starts.

The verification plan's "no I/O" claim applies to `resolve` only.

### 5. Completeness is an exact key-set comparison

*Finding 4.* Not counts. `resolve` emits the expected key set as precomputed chunk
filenames — `"{x0}-{x1}_{y0}-{y1}_{z0}-{z1}"` under the mip-0 key prefix, with
edge chunks clipped to `volume_size` rather than padded, matching CloudVolume's
own naming. `preflight` lists actual keys and reports **both** directions:
missing-expected and unexpected-extra, failing on either. Equal-but-wrong sets
therefore cannot pass.

### 6. Affinity contract, fully specified

*Findings 5 and 6.*

- **Source:** `<test_out>/<volume>/raw_x1_ch0-1-2.h5`, dataset `main`, `(3,Z,Y,X)`
  float16, as written by the existing pipeline.
- **Clipping then transform, in this order:** `v ← clip(v.astype(float64), EPS,
  1 − EPS)` with **`EPS = 1e-7`**, then
  `p = 1 / (1 + exp(−(log(v/(1−v)) / 0.2)))`, then cast to float32. This is
  `merge_fn_sweep.py::uncompress` with `scale = 0.2`; the implementation
  **imports** it so the constant has one definition, and a unit test asserts the
  imported `EPS` still equals `1e-7` so a future change to it fails loudly here.
- **Output:** precomputed layer at `<run_prefix>/aff_prob/`, float32, 3 channels,
  storage chunk = `seg_chunk_size_xyz`, `voxel_offset = BBOX[:3]`,
  `volume_size = BBOX[3:] − BBOX[:3]`, `resolution` = the prepared volume's
  effective spacing in XYZ. ABISS reads it with `AFF_CHANNELS: [2,1,0]`.
- **Mapped threshold for the integrity check:** compressed `c` maps to
  `p = sigmoid(logit(c)/0.2)`. The check decodes whole-volume `max` twice — on the
  compressed artifact at **`c = 0.47`** and on `aff_prob` at
  **`p = 0.35417863`** — and requires `VOI_total ≤ 0.001` between the results.
  Both values are stated in the config so neither is recomputed at run time.

### 7. Equivalence test, with edge conventions

*Findings 8 and 9, plus `plan_v0` §5 retained.*

- **Metric:** VOI on masked volumes. For a label set `S`, mask to voxels whose
  *whole-volume* label ∈ `S`; compute `VOI(chunked|mask, whole|mask)` as
  total = split + merge, reporting all three.
- **Sets:** `B` = whole-volume labels whose bounding box, treated as **half-open**
  `[lo, hi)`, intersects an **interior** chunk plane — multiples of `CHUNK_SIZE`
  strictly inside the volume; planes at 0 and at the volume extent are **not**
  seams and are excluded. `I` = all other labels. **Background label `0` is
  excluded from both**, as it is from the VOI computation itself.
- **Acceptance:** `VOI_total(chunked, whole) ≤ 0.01` over the full volume **and**
  `VOI_total(B) − VOI_total(I) ≤ 0.01`, for **each** of test A and test B.
- **Invariance:** compared **against the whole-volume reference**, not
  chunked-vs-chunked — so each configuration is judged by the same standard and a
  shared systematic error cannot cancel. `|VOI_total(A, whole) −
  VOI_total(B, whole)| ≤ 0.01` is additionally reported.
- 0.01 is this project's measured noise floor (0.006 between two runs differing
  only in GPU count).

### 8. The 8× claim, made concrete or dropped

*Finding 10.* `plan_v0` claimed the same path runs on a volume 8× larger. That is
now a **named optional smoke step**, not a success criterion: run the chunked
decode on `ExPID107_14.5x_04` (1.72 Gvoxel prepared, whole-volume answer not
required) with `CHUNK_SIZE [256,256,128]`, and assert only that it completes and
produces a non-degenerate segmentation. Its affinity must come from the existing
pipeline; the §6 converter is single-volume and is not claimed to scale.

## Files and Areas

| Path | Change |
|---|---|
| `connectomics/runtime/abiss_chunk.py` | read only |
| `tutorials/neuron_liconn_moe/gcloud/chunked_abiss.yaml` | new: config incl. §2 threshold, §3 sizes |
| `tutorials/neuron_liconn_moe/gcloud/make_prob_affinity.py` | new: §6 converter |
| `tutorials/neuron_liconn_moe/gcloud/resolve_chunked.py` | new: `resolve` (pure) and `preflight` (I/O) |
| `tutorials/neuron_liconn_moe/gcloud/equivalence_test.py` | new: §7 |
| `tutorials/neuron_liconn_moe/gcloud/run_volume.sh` | `DECODE=whole\|chunked` |
| `tutorials/neuron_liconn_ist/merge_fn_sweep.py` | imported for `uncompress`/`EPS`; not modified |

## Verification Plan

1. `bash -n` on shell changes. `resolve` runs with no filesystem or network I/O
   (asserted by running it against a config whose paths do not exist).
2. **Criterion guard:** `max`, `rlme`, `cs` each raise, naming the request and the
   accepted set `{mean}`; `max` additionally reports the missing-binary reason.
   `mean` resolves.
3. **Alignment guard:** `CHUNK_SIZE [256,256,128]` with `seg_chunk_size_xyz
   [128,128,96]` raises, naming both and the Z axis.
4. **≥2-chunks-per-axis guard:** `[2048,2048,80]` on the test volume raises,
   reporting `[1,1,7]`.
5. **Key-set guard:** deleting one `aff_prob` chunk fails `preflight` naming the
   missing key; adding a stray key fails naming the extra.
6. **EPS pin:** unit test asserts the imported `EPS == 1e-7`.
7. **§6 integrity:** whole-volume `max` on `aff_prob` at `0.35417863` reproduces
   the compressed-artifact `max` at `0.47` within `VOI_total ≤ 0.001`.
8. **Plumbing:** one-chunk decode reproduces whole-volume `mean`, `VOI_total = 0`.
9. **Acceptance:** test A and test B each satisfy `VOI_total ≤ 0.01` and
   `VOI_total(B) − VOI_total(I) ≤ 0.01`; the two agree within 0.01.
10. Non-degeneracy: chunked segmentation max id > 0; `aff_prob` min > 0, max < 1.
11. *Optional:* §8 smoke on `ExPID107_14.5x_04` completes with a non-degenerate
    segmentation.

## Risks and Questions

- **`AGG_THRESHOLD = 0.1394546` is fixed, not tuned** (§2). Valid for equivalence;
  invalid as a quality claim. A product threshold needs S1's sweep extended below
  its current bottom edge.
- **Per-chunk RSS is a model** (71 GB/Gvoxel). At test A's 8.4 Mvox/chunk the
  modelled peak is under 1 GB, so the test cannot validate the model at 100 µm
  scale — only that the path is correct. Stated so the two are not conflated.
- **Only `mean` is exercised**, so nothing here establishes `rlme` or `cs`.
- **The converter is single-volume** and §8's smoke depends on an affinity the
  existing pipeline already produced.
- **Test-scale seams may be easier than 100 µm seams:** 36 and 96 chunks exercise
  the mechanism, not the scale. A passing equivalence test is necessary, not
  sufficient, for 100 µm.

## Changes Since Previous Plan Version

Closes all 7 major and 3 minor findings in `plan_v0_review.md`, plus one defect
the review did not raise.

- **Self-caught defect:** `plan_v0`'s `[2048,2048,80]` gives `[1,1,7]` chunks on
  the test volume, so acceptance would have tested Z seams only. §3 chooses test
  sizes against the test volume and adds a ≥2-chunks-per-axis assertion.
- **Finding 1 — value supplied** (§2): `0.1394546`, with its provenance and the
  explicit warning that it is fixed rather than tuned.
- **Finding 2 — sizes named** (§3): test A `[256,256,128]`/`[128,128,128]`, test B
  `[192,192,96]`/`[64,64,96]`, plumbing and 100 µm rows separated.
- **Finding 3 — boundary stated** (§4): `resolve` pure, `preflight` does I/O, with
  invocation order and where each runs.
- **Finding 4 — exact key sets** (§5), reporting missing and extra in both
  directions.
- **Finding 5 — threshold formula and both values** (§6): `c = 0.47` ↔
  `p = 0.35417863`, tolerance `VOI ≤ 0.001`.
- **Finding 6 — clipping fully specified** (§6): `EPS = 1e-7`, clip-then-logit
  order, import rather than reimplement, with a pin test.
- **Finding 7 — accepted set narrowed to `{mean}`** (§1) rather than advertising
  untested criteria, with the mapping table and the raise messages.
- **Finding 8 — edge conventions** (§7): half-open boxes, interior planes only,
  background `0` excluded.
- **Finding 9 — invariance defined** (§7) as each configuration against the
  whole-volume reference.
- **Finding 10 — demoted** (§8) from a claim to an optional named smoke step.

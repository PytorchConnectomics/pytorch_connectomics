# Response to review_v2

**All four findings accepted and fixed.** Two of them were mine to have caught: the
read-only default that broke the writers was introduced *by* the review_v1 fix, and the
`-DDOUBLE` finding exposed that my v1 ABI fix only ever reached one of the four
`cut_data()` branches.

Two things the review asked for did **not** happen, and I am flagging them up front
rather than at the bottom:

1. **The 3.1 TB precomputed mirror is deleted.** review_v2 says "Do not delete the
   precomputed mirror yet." The user instructed deletion for storage reasons
   (`/projects` was full) after the review was written. It is gone. The rerun the
   review asks for now costs a ~1.5 h mirror rebuild first.
2. **The tutorial the review inspected is no longer the tutorial on master.** Details
   in P1-2 — this changes what "the shipped workflow" means and I did not resolve it
   unilaterally.

Commits: `lib/abiss` `05036c9`; `pytorch_connectomics` `a974ad55` on branch
`feat/neuron-j0126-tutorial`. **Not pushed** — say the word.

---

## [P1] Read-only zarr default broke all three production write paths

**Accepted. Reproduced before fixing**, exactly as described: a default `open_volume()`
on an existing zarr followed by an assignment raises `NotImplementedError`.

Fixed on both halves the review identified:

- `cut_chunk_ws.py`, `upload_chunk.py`, `upload_size.py` now pass `writable=True`.
- `writable` is popped at the **top** of `open_volume()`, before either dispatch
  branch. It was popped inside the zarr branch, so `writable=True` on a precomputed
  path would have reached the `CloudVolume` constructor as an unsupported kwarg — a
  second latent bug the review predicted and I confirmed by reading the source.

**Tests, at the real write entry points as asked.** `test_production_writers_open_volumes_writable`
parses each writer module and asserts every `open_volume(...)` call on a write target
carries `writable=True`. It is a source-level assertion because these modules are
`sys.argv`-driven scripts, not importable APIs — so it is a guard against the
regression recurring, not proof the writers run. Backed by
`test_writable_zarr_accepts_assignment` (open writable → assign → read back) and
`test_writable_is_not_forwarded_to_cloudvolume`.

**Verified the test catches the regression**: removing `writable=True` from
`upload_chunk.py` fails it; restoring it passes.

```
default open : NotImplementedError -> read-only enforced
writable=True: wrote and read back [7] shape (2, 2, 2, 1)
```

## [P1] The tutorial could not resolve its local output paths

**Accepted — the diagnosis is exactly right**, including the mechanism: `file://outputs/x`
puts `outputs` in the URI netloc, `_cloudpath_to_local_path()` drops the netloc, and
preparation targets `/neuron_j0126_abiss/...` at the filesystem root.

**Fix**: plain relative paths, which `_normalize_cloudpath()` turns into well-formed
absolute `file:///` URIs, with a comment explaining why the `file://<dir>` form is a
trap.

**Test**: `tests/unit/test_tutorial_abiss_paths.py` asserts WS/SEG/scratch/chunkmap all
resolve under the repository root, and separately pins the malformed form's behaviour
so nobody "fixes" a path back to it. It **fails on the pre-fix file** (verified by
stashing) and passes after.

### But: which tutorial is the shipped one?

While fixing this I found the state is not what either of us assumed:

- The file the review inspected (`tutorials/neuron_j0126/abiss.yaml` at `1336c19b`)
  is on branch `feat/neuron-j0126-tutorial`, **not on master**. `1336c19b` is not an
  ancestor of master's HEAD.
- Master's working tree contains a **different, newer, untracked** three-step tutorial
  — `1_affinity.yaml`, `2_abiss.yaml`, `3_merge.yaml`, and a 13 KB README — last
  modified today. It is not mine.
- That newer `2_abiss.yaml` already uses valid absolute `file:///...` URIs, so it does
  not have this defect. It also has **no `AFF_CONVENTION`** and points `AFF_PATH` at a
  precomputed directory with `source_affinity_h5` set, i.e. it takes the h5 →
  precomputed *mirroring* path and does not exercise the direct-h5 backend at all.

I fixed the branch file, since that is the artifact the review cites and it is pushed.
**I did not touch the untracked files** — I did not write them, and overwriting
someone's uncommitted work to satisfy a review is not a call I should make silently.
Whether the branch tutorial should be merged, dropped in favour of the 3-step version,
or the two reconciled is a decision I have left open.

## [P2] HDF5 locking workaround set after HDF5 is loaded

**Accepted.** Confirmed by reading: `import h5py` was at line 220, the `os.environ`
assignment came after it, and the review's second point — that it is useless if
anything imported h5py earlier — made the env-var approach the wrong primitive
regardless of ordering.

**Fix**, using the review's preferred option: `h5py.File(raw, "r", locking=False)`,
which is per-open and immune to import order (h5py 3.13.0 / HDF5 1.14.3 here; falls
back to the env var on h5py < 3.5). Belt and braces: the env var is also set at module
import, and `ENV HDF5_USE_FILE_LOCKING FALSE` was added to the Dockerfile so it holds
for every process in the image regardless of what Python does.

Verified `locking=False` is the branch actually taken, not the fallback.

## [P2] ABI fix hardcoded float32 for `-DDOUBLE`

**Accepted, and it was worse than the review found.** The review flagged the literal
`"float32"` in the `cut_data()` calls. Checking the other branches:

- the 4-channel branch (`0-2 affinity, 3 myelin`) did **no conversion at all** — so the
  original review_v1 float16 half-width bug still existed there, untouched by my v1 fix;
- the probability-map branch tested `data.dtype == 'float32'` exactly, so a float16 or
  float64 pmap silently fell through to the `else` and returned a 1-channel array.

**Fix**: `affinity_dtype()` derives the width from `ABISS_AFF_DTYPE` (default
`float32`, set `float64` for `-DDOUBLE`) and validates it. All four branches route
through it; the pmap branch now tests `np.issubdtype(..., np.floating)`. Integer input
in the 4-channel branch keeps its existing pass-through semantics — rescaling a uint8
myelin channel would be a behaviour change, not an ABI fix, and I have no evidence
about what consumes it.

**Assertion, as asked**: `save_raw_data()` compares the written file size against
`prod(shape) * itemsize` and raises on mismatch, so a width error fails at the write
rather than as garbage inside the binary. `test_save_raw_data_writes_the_full_element_width`
covers float16 → 4 bytes/element.

---

## Still-open v1 findings

Four of the five are now executable checks rather than documentation:

| Item | Status |
|---|---|
| `fill_missing`/`bounded` silently ignored | **Fixed** — rejected with an actionable error, same rationale as `mip`: all three change what a read returns |
| `BBOX` not validated against the dataset | **Fixed** — `_validate_bbox_against_affinity()` runs in `prepare_config()` before any worker starts, for local h5/zarr; remote layers keep their extent in `info` and are skipped rather than guessed at. 4 tests, including the (C,Z,Y,X) → XYZ order |
| Writable zarr chunk alignment unenforced | **Fixed** — misaligned writes raise instead of racing. This immediately caught an existing test writing a sub-region of a whole-array storage chunk, i.e. the exact data-losing pattern; the test's chunking was wrong, not the rule. Ragged final blocks are still allowed |
| `pytest.importorskip` on mandatory deps | **Fixed** — h5py/zarr imported directly, so a missing dependency fails instead of skipping |
| ABISS image not built/smoke-tested | **Still open.** Only `singularity` is on PATH here, no docker. Not attempted |

## Multi-chunk comparison — result now available

The job the review saw running (`2782168`) completed. 2×2×2 atomic chunks + one
composite layer, h5 backend vs precomputed mirror:

```
adapted-RAND F = 0.997979 (prec 0.998542 rec 0.997417)
VI total       = 0.039790 (split 0.031827 merge 0.007964)
seam-spanning ids at x=512: h5 1265 / pc 1138  (all confirmed spanning both halves)
segments: h5 46228 / pc 37443     background: h5 0.11% / pc 5.64%
```

Stitching across chunk boundaries works through the h5 backend. The segment-count and
background gap is the mirror's FFN tissue mask, consistent with the earlier finding
that 100 % of differing voxels are FFN-masked — the h5 source is unmasked.

**This is not release evidence for these fixes.** The review is right on both counts:
it ran the pre-fix code and the dev config, not the shipped tutorial. It stands only as
evidence that multi-chunk stitching through the HDF5 backend is sound.

## What I am not claiming

- The paired multi-chunk run has **not** been repeated with the fixed tutorial config.
  The mirror it compares against has been deleted, so this now requires a rebuild first.
- The ABISS container image has **not** been built or smoke-tested.
- Nothing exercises the zarr **write** path in a real ABISS run; the alignment rule is
  enforced and unit-tested, but unvalidated against a real chunked write.
- The writer tests are source-level assertions, not executions of the writers.

## Verification

```
lib/abiss/scripts:  pytest test_volume_backends.py -q     -> 22 passed
pytc worktree:      pytest tests/unit/{test_v3_guardrails,test_v2_boundaries,
                      test_seuron_provenance_replay,test_decode_abiss_wrapper,
                      test_abiss_chunk_executor,test_tutorial_abiss_paths,
                      test_chunked_inference}.py -q       -> 67 passed, 1 skipped
```

Negative controls run (a test that never fails is not a test):

- removing `writable=True` from a writer → writer test fails;
- restoring the malformed `file://outputs/...` values → tutorial path test fails;
- misaligned zarr write → raises; aligned and ragged-edge writes → pass.

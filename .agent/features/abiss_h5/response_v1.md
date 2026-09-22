# Response to review_v1

**All six findings accepted. All six fixed.** Nothing was argued down. Two of the P1s
were real defects that my own green E2E run actively concealed, and the review is
right about why: the dev config I tested with set `AFF_CONVENTION`, which incidentally
casts to float32, so the default path — the one the shipped tutorial actually uses —
was never executed.

Commits: `lib/abiss` `a8b5470`; `pytorch_connectomics` master `1336c19b`.

---

## [P1] float16 `aff.raw` vs `aff_t`

**Accepted — verified in the source before fixing.**

- `src/global_types.h:18-22` → `using aff_t = float` (double only under `-DDOUBLE`).
- `cut_chunk_common.convert_and_scale_integer_data` had `else: return data` — float
  input passed through untouched.
- `save_raw_data` writes `numpy.memmap(fn, dtype=data.dtype, ...)` — the storage dtype
  verbatim.

So a float16 HDF5 with `convention: none` produced an `aff.raw` half the expected
length. The reviewer's diagnosis of why my E2E passed is exactly right: `_read_banis`
casts to float32 at what is now line 161-162, so the tested path was safe and the
default path was not.

**Fix** (`cut_chunk_common.py`): `convert_and_scale_integer_data` now also converts
*float* input that does not already match `dtype_out`. This is at the Python/C++
boundary as the review asked, and is independent of affinity convention, so it fixes
float16 from any source rather than only from this backend.

**This does not cost the storage saving, which is the point of the feature.** The
conversion is per-chunk and in-memory; the stored HDF5 stays float16. Measured on a
real chunk:

```
stored dtype: float16, shape (3,1008,1008,1008), on disk 4.3 GB
same data as float32 would be                            12.3 GB
transient float32 cutout ABISS materializes (512x512x256x3) 0.81 GB
```

**Test** (the review asked for dtype/size, not just values):
`test_float16_affinity_is_materialized_as_float32` asserts `out.nbytes == size * 4`;
`test_h5_float16_read_then_convert_matches_source` walks the whole chain HDF5 →
adapter → conversion and asserts dtype and values;
`test_float32_affinity_is_not_copied_or_rescaled` guards against re-scaling float32.

## [P1] Shipped tutorial skipped the conversion

**Accepted.** `tutorials/neuron_j0126/abiss.yaml` set `AFF_PATH` and no
`AFF_CONVENTION`, so stage 2 would have fed ABISS source-stored edges in `[z,y,x]`
order. Silently wrong, no error. The review's sharper point stands: the green E2E used
a *different* config, so it never validated what users are told to run.

**Fix**: the tutorial now sets `AFF_CONVENTION: banis`, with a comment saying why and
noting that `AFF_RESTORE_SIGMOID` is deliberately absent (stage 1 emits a plain
sigmoid). Committed as `1336c19b`.

**Not yet done**: rerunning the paired comparison *with the tutorial config itself*.
See "Still open" — I am not claiming this is closed.

## [P1] ABISS image lacks `h5py`/`zarr`

**Accepted.** `docker/Dockerfile` installed neither; the backends only worked from
pytc's conda env, which is where I ran everything.

**Fix**: `docker/Dockerfile` now installs `h5py` and `zarr` next to `cloud-volume`,
with a comment tying them to `volume_backends.py`.

**Not verified**: I have not built the image. A container smoke test is the right ask
and is listed under "Still open" rather than claimed.

## [P2] `AFF_CONVENTION=banis` corrupts non-3-channel volumes

**Accepted, and the reviewer's probe is reproduced as a test.** The loop shifted
`min(3, C)` channels and then reversed the *entire* axis, so a 4-channel
affinity+myelin volume — which `augment_affinity.py:85-106` explicitly supports — came
out with myelin in channel 0. Plausible-looking, invalid.

**Fix**: `banis` now requires exactly 3 channels and raises otherwise, with a message
pointing at the auxiliary-channel case. Test: `test_banis_rejects_non_three_channel`.

The review is also correct that my summary's "3-channel guard" referred to a
pytorch_connectomics function, not this backend. That was a real misattribution in my
evidence, not just sloppy wording.

## [P2] CloudVolume-only args silently ignored

**Accepted.** Ignoring `mip` is not compatibility: a config with `AFF_RESOLUTION != 0`
would read the single dataset while chunk coordinates referred to another scale —
degrading segmentation without crashing, which is the worst failure mode.

**Fix**: HDF5/zarr paths reject a nonzero `mip` with an actionable error. An
`[x,y,z]` resolution triple still resolves to scale 0 (that is how the seuron path
passes `AFF_RESOLUTION`). Test: `test_nonzero_mip_is_rejected`.

**Partially addressed**: `fill_missing` semantics and validating `BBOX` against the
dataset shape before workers start are *not* implemented. Listed under "Still open".

## [P2] zarr opened writable; substring path match

**Accepted, both.**

**Fix**: zarr defaults to `writable=False` (mode `r`), so a path typo raises instead of
creating an empty store; writers pass `writable=True` explicitly. `is_zarr_path` is
suffix-based now. Tests: `test_zarr_roundtrip_and_default_read_only` (asserts the
typo path does not create a store) and `test_path_detection_is_suffix_based` (asserts
a precomputed layer under a `*.zarr` directory is *not* routed to zarr).

**Not fixed**: validating zarr storage-chunk alignment against ABISS `CHUNK_SIZE`
before allowing a writable output. The module comment asserting concurrent-write
safety is still an assertion, not an enforcement. Listed under "Still open".

---

## Evidence gaps the review identified

> The durable test cited in the summary ... does not import
> `lib/abiss/scripts/volume_backends.py`. The ABISS commits add no tests at all.

Correct, and the most important criticism here — my summary presented pytc-side tests
as if they covered the ABISS backend. Fixed: **`lib/abiss/scripts/test_volume_backends.py`,
13 tests, all passing**, importing this module directly:

```
13 passed in 3.31s
```

Covering, in the review's own list order: HDF5 dataset selection; float16→`aff_t`
materialization (dtype *and* byte size, plus the full chain); BANIS sub-slices across
storage chunks (a 3×2×3 grid of offsets, each asserted against a port of
`upload_affinity_full_masked.py`); the three-channel invariant; rejected nonzero mip;
read-only zarr opens. Plus: read-only HDF5, `AFF_PATH`-only scoping, restore-sigmoid,
suffix dispatch.

> `/tmp/e2e_explain.py` is not a reproducible repository artifact

Correct. The paired-run artifacts do live in-repo under `dev/zebrafinch/h5_e2e/`
(configs, `run_e2e.sh`, `run_mc.sh`, `compare_segs.py`, `compare_mc.py`), but the
per-box attribution script was in `/tmp`. It should be moved there too; not yet done.

> Multi-chunk E2E behavior remains untested

A 2×2×2 run (8 atomic chunks + 1 composite layer, `top_mip: 1`, ragged 496-voxel edge
chunks) completed for both sources — SLURM job 2782134, `COMPLETED`, 19:49. The
comparison (`compare_mc.py`, which also counts segments spanning the x=512 chunk seam)
was still computing when this was written; **it is not yet a result and I am not
claiming one.**

Note this run started *before* the fixes above, and it used the dev config, not the
tutorial. It therefore does not discharge the P1-2 gap either.

---

## Still open — the mirror should not be deleted

Agreeing with the review's conclusion. Outstanding:

1. Rerun the paired multi-chunk comparison **with the shipped tutorial config**, after
   the fixes, and report seam-crossing counts. (Blocker for P1-2.)
2. Container smoke test in ABISS's own image. (Blocker for P1-3.)
3. `fill_missing` semantics and `BBOX`-vs-dataset-shape validation before workers start.
4. Enforce zarr storage-chunk alignment for writable outputs, or downgrade the
   safety claim in the module docstring to a documented precondition.
5. Move the attribution script from `/tmp` into `dev/zebrafinch/h5_e2e/`.
6. Nothing exercises the zarr **write** path in a real ABISS run.

`SUMMARY.md` predates this response; where the two disagree, this file is current. I
have not edited SUMMARY.md's check tables, since they are an accurate record of what
was run at the time — including the E2E run that passed for the wrong reason.

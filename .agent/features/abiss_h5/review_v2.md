# Review v2: response to `review_v1`

## Verdict

**Request changes.** The main HDF5 read path is substantially better: float16
affinity is materialized as float32 at the standard-build ABI boundary, the BANIS
three-channel invariant is enforced, mip misuse now fails, the tutorial enables the
required convention, and backend tests now exist and pass.

The response's opening claim that all six findings are fixed is still too strong.
Changing zarr to read-only by default broke every production zarr write call site,
the shipped tutorial still contains malformed local file URIs, and two safety
findings from v1 are explicitly left open.

## Findings

### [P1] The read-only zarr default broke all three production write paths

`open_volume()` now passes `writable=False` to zarr unless a caller opts in
(`scripts/volume_backends.py:317-323`). That is the right default for reads, but none
of the actual writers was updated:

- `scripts/cut_chunk_ws.py:35-37` opens `ADJUSTED_AFF_PATH` without
  `writable=True`.
- `scripts/upload_chunk.py:22-23` opens the WS/SEG target without
  `writable=True`.
- `scripts/upload_size.py:16-17` opens the size-map target without
  `writable=True`.

For a pre-existing zarr array, each call now constructs a read-only `ZarrVolume`;
the following assignment raises `NotImplementedError`. A direct probe reproduced
that failure. The new zarr test misses the regression because it calls
`open_volume(..., writable=True)` directly instead of exercising a production
writer.

Update the write sites and consume the `writable` option before the CloudVolume
fallback so it is not forwarded as an unsupported CloudVolume constructor
argument. Add tests around the real write entry points (or a shared writable-volume
helper), not only the adapter.

### [P1] The corrected tutorial still cannot resolve its local output paths

At `pytorch_connectomics` revision `1336c19b`,
`tutorials/neuron_j0126/abiss.yaml:60-63` uses paths such as:

```yaml
WS_PATH: file://outputs/neuron_j0126_abiss/precomputed/ws
```

That is not a relative file URI. URI parsing treats `outputs` as the network
location and `/neuron_j0126_abiss/...` as the path. The runtime preserves any value
containing `://` (`connectomics/runtime/abiss_chunk.py:167-174`) and then
`_cloudpath_to_local_path()` discards the URI netloc
(`connectomics/runtime/abiss_chunk.py:181-188`). Preparation consequently targets
`/neuron_j0126_abiss/...` at the filesystem root rather than
`<repo>/outputs/neuron_j0126_abiss/...`, normally failing before ABISS starts.

Use plain relative paths and let `_normalize_cloudpath()` create an absolute
`file:///...` URI, or provide valid absolute file URIs. Add a tutorial-resolution
test asserting the local WS, SEG, scratch, and chunkmap paths all remain under the
repository/output root. This is another reason the dev-config E2E does not validate
the shipped workflow.

### [P2] The advertised HDF5 locking workaround is set after HDF5 is loaded

`H5Volume.__init__()` imports `h5py` and only then sets
`HDF5_USE_FILE_LOCKING=FALSE` (`scripts/volume_backends.py:219-228`). Modern HDF5
reads this environment setting during library initialization; importing h5py loads
that library first. The assignment is therefore too late in the exact networked,
multi-reader case the comment says it protects. It is also ineffective whenever
h5py was imported elsewhere before this constructor.

Set the environment before importing h5py, or use a supported explicit
`h5py.File(..., locking=False)` option and verify it in the ABISS image. The current
cluster run succeeding does not prove the workaround is active.

### [P2] The ABI fix still hardcodes float32 for `-DDOUBLE` builds

The new conversion correctly fixes the standard build, but the comment says it
matches "`aff_t` (float, or double under `-DDOUBLE`)" while every `cut_data()` call
still passes the literal `"float32"` (`scripts/cut_chunk_common.py:46-70`).
`src/global_types.h` makes `aff_t` float64 under `-DDOUBLE`, so those binaries would
still mmap a file with the wrong element width.

Either derive/export the compiled affinity dtype and materialize that dtype, or
explicitly reject/document `-DDOUBLE` for this pipeline. In both modes, assert the
expected `aff.raw` byte length before mmap rather than relying on a Python comment.

## Still-open v1 findings

These are accurately acknowledged by `response_v1.md`, but they remain blockers to
calling the backend generally safe:

- HDF5/zarr still ignore `fill_missing`/`bounded`, and the runtime does not validate
  `BBOX` against the selected dataset before workers start.
- Writable zarr still has no check that storage chunks divide/align with ABISS
  logical chunks, so disjoint logical writes may race on one storage chunk.
- The ABISS image has not been built or smoke-tested. Moreover, the new test module
  uses `pytest.importorskip` for both newly mandatory packages, so a dependency
  omission can turn the whole backend suite into a skip instead of a failure.
- The multi-chunk comparison job is still running and used the pre-fix dev config,
  not the shipped tutorial. It is not release evidence for these fixes.

## Verification

- `pytest -q scripts/test_volume_backends.py`: **13 passed**.
- `git diff --check 91e182f..a8b5470`: passed.
- Python compilation of all changed backend scripts and the new test: passed.
- Direct zarr production-mode probe: default open followed by assignment raises
  `NotImplementedError`, confirming the P1 regression above.
- SLURM job `2782168` (`mc_cmp`) was still running when reviewed; no comparison
  result was available.

Do not delete the precomputed mirror yet. First fix the two P1s, make the remaining
safety preconditions executable checks, build/smoke-test the ABISS image, and rerun
the paired multi-chunk comparison through the corrected tutorial-resolved config.

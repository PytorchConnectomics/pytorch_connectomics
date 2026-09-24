# Review v1: ABISS HDF5/zarr backend

## Verdict

**Request changes.** The HDF5 adapter's zero-based, mip-0, three-channel read and
BANIS conversion appear correct for the exact `dev/zebrafinch/h5_e2e/abiss_h5.yaml`
case. The commit does route ABISS affinity reads through HDF5 and can therefore avoid
the precomputed affinity mirror in that narrow configuration.

It is not yet a correct replacement for the advertised float16 HDF5 workflow,
however. The successful E2E configuration hides a dtype ABI problem, and the shipped
tutorial does not enable the conversion that made the E2E run work. The official
ABISS image also lacks the new runtime dependencies.

## Findings

### [P1] Float16 HDF5 is written to `aff.raw` as float16, but ABISS always reads it as `aff_t`

The main reason for this feature is to read pytc's float16 HDF5 without making a
float32 precomputed copy. In the default adapter path,
`volume_backends._ArrayVolume.__getitem__` preserves the HDF5 dtype
(`scripts/volume_backends.py:133-144`). `cut_data()` only converts integer affinity;
floating-point data is returned unchanged (`scripts/cut_chunk_common.py:38-44,56-57`),
and `save_raw_data()` writes that dtype verbatim
(`scripts/cut_chunk_common.py:24-27`).

The C++ ABI is not dtype-generic: a normal build defines `aff_t = float`
(`src/global_types.h:18-22`) and watershed mmaps `aff.raw` as a
`float[x,y,z,3]` array (`src/ws/atomic_chunk.cpp:138-140`). Mean-edge extraction does
the same. A float16 `aff.raw` is half the expected byte length and will be
misinterpreted or fail while mapping.

The reported E2E run did not cover this default path. Its dev config sets
`AFF_CONVENTION: banis` and `AFF_RESTORE_SIGMOID: 0.2`; `_read_banis()` happens to
cast the block to float32 at `scripts/volume_backends.py:161-162`, so that run wrote a
compatible `aff.raw`.

Fix this at the Python/C++ boundary, independently of affinity convention: convert
all affinity cutouts to the build's `aff_t` before writing and assert the resulting
file byte size before launching a binary. For the standard build this means
float32. Add a test starting from a float16 HDF5 with `convention=none`; checking
only array values is insufficient—the test must check `aff.raw` dtype/size or run
the watershed binary.

### [P1] The shipped no-precompute tutorial skips the required BANIS-to-ABISS conversion

At target revision `f912fb2a`, `tutorials/neuron_j0126/infer_affinity.yaml:24-30`
stores float16 model channels `[z,y,x]` with plain sigmoid. Plain sigmoid removes
only the need for `AFF_RESTORE_SIGMOID`; it does not remove the edge-location shift
or channel reversal.

Nevertheless, `tutorials/neuron_j0126/abiss.yaml:45-76` sets `AFF_PATH` but no
`AFF_CONVENTION: banis`. The backend therefore feeds ABISS source-stored edges in
ZYX channel order, while ABISS expects destination-stored edges in XYZ order. It
also takes the broken float16 path from the previous finding.

The green E2E job used a different dev config which does set both conversion keys,
so it does not validate the tutorial users are instructed to run. Add
`AFF_CONVENTION: banis` to the tutorial, omit `AFF_RESTORE_SIGMOID` for its plain
sigmoid output, and run the actual tutorial config in the paired comparison.

### [P1] The official ABISS runtime does not install either backend dependency

`H5Volume` imports `h5py` at `scripts/volume_backends.py:209-210`, and `ZarrVolume`
imports `zarr` at lines 242-243. Neither package is installed or declared anywhere
in `lib/abiss`; in particular, `docker/Dockerfile:43-51` installs neither one.

The E2E run used the pytc conda environment, where both packages happen to exist,
so it does not prove that the ABISS commit works in ABISS's own image. An HDF5 path
in that image will raise `ModuleNotFoundError`.

Add explicit pinned/compatible dependencies to the ABISS runtime and a container
smoke test that imports both adapters and reads a tiny dataset. If this feature is
intentionally supported only when ABISS is launched from pytc's environment, state
and enforce that contract rather than advertising it as an ABISS backend.

### [P2] `AFF_CONVENTION=banis` silently corrupts non-three-channel volumes

The converter loops over `min(3, block.shape[0])` and then reverses the entire
channel axis (`scripts/volume_backends.py:169-180`). This is not a three-channel
guard. For a four-channel affinity+myelin array, channel 3 remains zero in
`shifted`, then becomes output channel 0; the three shifted affinity channels move
to output channels 1-3. ABISS explicitly supports a four-channel affinity+myelin
path (`scripts/augment_affinity.py:85-106`), so this produces plausible-looking but
scientifically invalid input rather than a clean failure.

A direct probe produced channel maxima `[0.0, 0.3, 0.2, 0.1]` from input maxima
`[0.1, 0.2, 0.3, 0.9]`. Require exactly three channels for the BANIS convention, or
define and test how the auxiliary channel is preserved while only the affinity
channels are reordered. The summary's claimed “3-channel guard” tests a different
pytc function, not this backend.

### [P2] CloudVolume-only arguments are silently ignored even when they change coordinates

`open_volume()` documents that `mip`, `fill_missing`, and `bounded` are accepted and
ignored for HDF5/zarr (`scripts/volume_backends.py:280-300`). Ignoring `mip` is not
behavioral compatibility: an existing ABISS config with `AFF_RESOLUTION != 0` will
silently read the only HDF5 dataset while the chunk coordinates refer to another
scale. Likewise, NumPy/HDF5 slices clamp at the array boundary instead of honoring
CloudVolume `fill_missing` behavior.

For a single-scale backend, reject every nonzero `mip` and unsupported coordinate
mode with an actionable error. Also validate the configured `BBOX` against the
dataset shape before workers start. Silent fallback is especially risky here
because wrong affinity geometry generally degrades segmentation without crashing.

### [P2] Zarr reads are opened writable and concurrent-write safety is asserted but not enforced

Every zarr open defaults to `writable=True`, so even read-only `load_data()` calls
use `zarr.open(..., mode="a")` (`scripts/volume_backends.py:242-250,294-300`). This
requires write permission and can create a new store after a path typo. Reads
should use mode `r`; write call sites should opt into writing explicitly.

The module also says concurrent writes are safe because regions are disjoint and
chunk-aligned, but the adapter never checks the zarr storage chunks against ABISS
`CHUNK_SIZE`. Disjoint logical regions can still share a storage chunk and race.
Validate alignment before allowing a writable zarr output, and make the path
detector suffix-based—`is_zarr_path()` currently misroutes any path merely
containing `.zarr` (`scripts/volume_backends.py:46-48`).

## Verification and evidence gaps

- `git diff --check d7f0721..91e182f` and Python compilation of the five changed
  scripts pass.
- A direct synthetic HDF5 slice matches the expected CZYX-to-XYZC transpose for
  the supported zero-based case.
- The durable test cited in the summary,
  `tests/unit/test_precomputed_affinity_output.py`, imports and tests
  `connectomics.inference.chunked._to_abiss_affinity_convention`; it does not import
  `lib/abiss/scripts/volume_backends.py`. The ABISS commits add no tests at all.
- `/tmp/e2e_explain.py` is not a reproducible repository artifact, and the only
  E2E comparison uses a conversion-enabled dev config rather than the shipped
  tutorial.
- Multi-chunk E2E behavior remains untested, as the summary acknowledges.

Add backend tests in `lib/abiss` for HDF5 dataset selection, float16-to-`aff_t`
materialization, BANIS sub-slices across storage chunks, the three-channel
invariant, rejected nonzero mip, read-only zarr opens, and output chunk alignment.
Then rerun a multi-chunk job using the exact tutorial config. Until those blockers
are fixed and that run agrees with the reference, the 3.1 TB mirror should not be
deleted.

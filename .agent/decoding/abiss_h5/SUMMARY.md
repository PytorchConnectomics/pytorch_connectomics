# ABISS reads affinity HDF5/zarr directly (no precomputed mirror)

Review target for Codex. Everything below was run; numbers are copied from actual
output, not estimated. Where something is unverified it says so explicitly.

## Problem

ABISS reached volumes only through CloudVolume, so affinity produced by pytc
inference (HDF5, `(C,Z,Y,X)`, float16) had to be converted into a precomputed layer
before ABISS could read it. For zebrafinch that conversion is a **3.1 TB second copy**
of the volume, and precomputed pins a dtype so float16 became float32 (~2x).

## What changed

### A. `lib/abiss` (repo `PytorchConnectomics/abiss`, branch `feature/configurable-edge-score`)

| commit | content |
|---|---|
| `add70b9` | `scripts/volume_backends.py` + dispatch |
| `91e182f` | BANIS→ABISS affinity convention on read |

- **`scripts/volume_backends.py`** (new). `open_volume(path)` dispatches on the path:
  `*.h5`/`*.hdf5` → `H5Volume`, `*.zarr` → `ZarrVolume`, anything else → CloudVolume
  unchanged. Adapters present CloudVolume's `(X,Y,Z,C)` view over arrays stored
  `(C,Z,Y,X)` or `(Z,Y,X)`, supporting the full surface ABISS uses: construct,
  `.shape`, `.dtype`, `__getitem__`, `__setitem__`.
- **HDF5 is read-only by design.** HDF5 has no multi-process writer support (SWMR is
  single-writer) and ABISS writes chunk outputs from many workers concurrently, so a
  write raises rather than silently corrupting. zarr is read+write (ABISS writes whole
  disjoint chunk-aligned regions, which zarr supports).
- **Affinity convention**, opt-in per volume via the ABISS param, applied only to the
  volume named by `AFF_PATH` (so WS/SEG volumes through the same backend are untouched):
  ```json
  "AFF_CONVENTION": "banis",        // edge v->v-1, channels [z,y,x]->[x,y,z], clip [0,1]
  "AFF_RESTORE_SIGMOID": 0.2        // optional; only for scale_sigmoid output
  ```
  This reproduces `dev/zebrafinch/upload_affinity_full_masked.py`, which previously did
  it as a separate whole-volume pass. The edge shift needs voxel `v-1`, which for a
  chunked read is outside the requested box, so the adapter reads **one extra voxel on
  each low face** and crops after shifting; only a true volume edge is zero-filled.
- Call sites touched: `cut_chunk_common.load_data`/`load_gt_data` (all reads),
  `upload_chunk.py`, `upload_size.py`, `cut_chunk_ws.py` (`ADJUSTED_AFF_PATH` write).

**Deliberately NOT reproduced:** the FFN tissue/border keep-mask from the upload script.
It was measured ~inert for reconstruction quality (NERL 0.701 masked vs 0.697 unmasked
on a 14%-masked region) and needs volumes (tissue mask, raw EM) ABISS does not have.
**This is the sole source of the residual disagreement in check 6 below.**

### B. `pytorch_connectomics` (master, through `f912fb2a`)

- `inference.chunking.precomputed*` — write chunks straight into a precomputed layer
  (kept, tested, but the j0126 tutorial no longer uses it: float32 is too big).
- `abiss_chunk`: `source_affinity_h5` is now optional. When omitted, AFF_PATH is used
  as-is and the h5→precomputed copy is skipped; `param.BBOX` becomes required.
- `tutorials/neuron_j0126/` — `infer_affinity.yaml` (single self-contained config),
  `abiss.yaml`, `README.md`.

## Check results

All commands were run in `pytc` env on the BC cluster.

**1. Backend vs CloudVolume ground truth** (synthetic, same data in both stores)

| check | result |
|---|---|
| h5 full read == precomputed | PASS (exact) |
| h5 sub-volume slice == precomputed | PASS (exact) |
| h5 write | correctly raises `NotImplementedError` |
| zarr read / slice == precomputed | PASS (exact) |
| zarr write round-trip | PASS |
| 3D single-channel (seg-shaped) read+write | PASS |
| precomputed path still returns a CloudVolume | PASS |

**2. Affinity convention vs a port of `upload_affinity_full_masked.py`** (synthetic)

| check | result |
|---|---|
| full read == reference | PASS |
| **every chunked sub-read across a 3x3x3 grid of offsets == reference** | PASS |
| non-`AFF_PATH` volume left unconverted | PASS |
| `AFF_RESTORE_SIGMOID` path == reference | PASS |

The chunked-sub-read row is the important one: it proves the 1-voxel low margin makes
chunk boundaries pull the true neighbour instead of a zero plane.

**3. Convention transform unit tests** — `tests/unit/test_precomputed_affinity_output.py`,
8 passed. Includes an independent port of the reference implementation, channel
semantics (`ch0` must be x-affinity), a halo-correctness test asserting the
chunk-local answer **differs** from the correct one, the 3-channel guard, and the
storage-chunk alignment guard.

**4. Backend read vs the real 3.1 TB mirror** (real data: `chunk_z2_y3_x3.h5`,
float16 `(3,1008,1008,1008)`, vs `dev/zebrafinch/seuron_gate/aff_mirror`)

| box (global z,y,x) | mirror range / std | max abs diff |
|---|---|---|
| 2200,3200,3200 | 0.000–1.000, σ 0.406 | **0.0** |
| 2600,3700,3600 | 0.999–1.000, σ 0.000 | **0.0** |
| 2900,3900,3900 | 0.000–1.000, σ 0.371 | **0.0** |
| 2100,3100,3800 | 0.000–1.000, σ 0.472 | 1.0 |
| 2800,3300,3100 | 0.000–1.000, σ 0.427 | 2.5e-3 |

Attribution of the two non-zero rows:

```
box z2100: 6077 of 331776 voxels differ >2e-3; of those, mirror==0 (masked): 100.0%
           agreement where mirror>0: max|diff| 0.00e+00
box z2800:   49 of 331776 voxels differ >2e-3; of those, mirror==0 (masked): 100.0%
           agreement where mirror>0: max|diff| 0.00e+00
```

**Wherever the mirror holds data, the h5 read is bit-identical.** 100% of differing
voxels are where the FFN mask zeroed the mirror. High-variance boxes were used
deliberately (a first attempt landed on a near-constant 0.997–1.0 region where a
transpose error could hide), plus a channel-swap control confirming the comparison is
sensitive to channel order.

**5. End-to-end ABISS run reading HDF5** — SLURM job 2782099, `COMPLETED`, 3:58.
All four stages (watershed → remap → mean-edge agglomeration → remap) ran with
`AFF_PATH` pointing at the affinity HDF5. Produced
`dev/zebrafinch/h5_e2e/h5/precomputed/seg`.

**6. Same-voxel segmentation comparison: HDF5-sourced vs mirror-sourced**

Identical ABISS parameters on the same 512x512x256 region; only the affinity source
differs (`dev/zebrafinch/h5_e2e/{abiss_h5,abiss_pc}.yaml`, compared by
`compare_segs.py`).

```
segments:         h5 = 6555        precomputed = 6061
background frac:  h5 = 0.0019      precomputed = 0.0194
adapted-RAND F = 0.998487   (precision 0.999768  recall 0.997208)
VI total       = 0.010871   (split 0.010045  merge 0.000826)
```

Interpretation: the 10x background difference (1.94% vs 0.19%) is exactly the FFN mask
the backend does not reproduce, and it shows up as **splits** (VI split 0.0100 vs merge
0.0008) plus the extra ~500 segments — masked regions are background in the mirror run
and segmented in the h5 run. Merge disagreement is ~0.
**Caveat on that script's last line:** it also prints "identical-label voxels after
relabel", which is meaningless (`relabel_sequential` assigns ids by order of
appearance, so it does not align labels across two segmentations). Ignore it; the
adapted-RAND/VI numbers are the real comparison.

**7. Regression suites** — `test_abiss_chunk_executor.py` 10 passed;
`test_v3_guardrails.py` 12 passed; combined sweep with
`test_precomputed_affinity_output.py` + `test_decode_abiss_wrapper.py`
33 passed / 1 skipped (skip = real-ABISS wrapper test needing `lib/abiss/build`).
`scripts/validate_tutorial_configs.py` exit 0.

## Bugs found and fixed while doing this

1. **abiss built without `-DEXTRACT_SIZE=ON`** — `acme` never writes `ns.data` /
   `ongoing_seg_size.data`, so agglomeration builds an empty supervoxel set and aborts
   `Should not happen, rg element does not exist`. Watershed succeeds either way, so it
   only surfaces halfway through. Fix: `cmake -DEXTRACT_SIZE=ON .. && make -j`.
2. **`libtbb.so.2` missing** — binaries link the old TBB soname. Expose only that lib on
   `LD_LIBRARY_PATH`; adding the whole legacy toolchain dir shadows `libstdc++`
   (`GLIBCXX_3.4.29` missing).
3. **Output-layer chunk misalignment** — WS/SEG storage chunk must divide ABISS'
   `CHUNK_SIZE`; the affinity's own chunk size does not.
4. **`config.sh` race** — `init.sh` creates it "if not exists"; sharded workers raced and
   read a truncated file. Generate once up front.
5. **`OVERLAP`/`META` unset** — the executor injects them; `init.sh` sets `META` but never
   `OVERLAP`, and `*_chunk_me.sh` run under `set -u`, so the whole agglomeration half
   died instantly while watershed was unaffected.
6. **Channel order** — first version of the precomputed writer passed channels through, so
   ABISS would have read z-affinity as x-affinity. Caught by reading
   `dev/zebrafinch/precompute_bndaff.py` (`pz = aff[0]`, `py = aff[1]`, `px = aff[2]`).
   The inherited base config comments this as "XYZ order", which is wrong.
7. **Tutorial inherited the tiled-PNG base** — ~19 h/chunk at 0% GPU vs ~30 min/chunk on
   the zarr.
8. **Stage-1 decoding left enabled** — after chunked inference the test pipeline reads the
   stitched whole-volume prediction with a single `np.array`, a **9.4 TiB** allocation.
   Found by the smoke run (job 2781622), fixed with `decoding.enabled: false`.
9. **My own `source_affinity_h5` guard rejected h5 AFF_PATH** — it assumed "omitted" meant
   "AFF_PATH is a precomputed layer" and demanded an `info` file, blocking exactly the
   case the backend enables. Found by the first e2e run (job 2782097, failed in 3 s).

## Known gaps / what a reviewer should push on

- The FFN keep-mask is not reproduced (deliberate, see above). If a run needs it, the
  affinity must be masked before ABISS reads it.
- Only a **single 512x512x256 chunk** was compared end-to-end. Multi-chunk behaviour
  through the h5 backend (cross-chunk stitching at composite layers) is untested.
- The zarr **write** path is unit-tested but has never been exercised by a real ABISS run.
- `is_zarr_path` matches any path containing `.zarr`, and `is_h5_path` matches the
  `.h5`/`.hdf5` suffix before `::`. A precomputed layer living under a directory named
  `*.zarr` would be misrouted.
- The 3.1 TB mirror at `dev/zebrafinch/seuron_gate/aff_mirror` is **still present**; it was
  not deleted, since these checks are the evidence for whether deleting it is safe.

## Reproducing

```bash
# backend vs CloudVolume + convention vs reference (synthetic, seconds)
python -m pytest tests/unit/test_precomputed_affinity_output.py -q

# real-data read comparison vs the mirror
python /tmp/e2e_explain.py   # h5-read vs mirror, per-box attribution

# end-to-end paired ABISS runs (SLURM, ~4 min)
sbatch dev/zebrafinch/h5_e2e/run_e2e.sh
python dev/zebrafinch/h5_e2e/compare_segs.py
```

Artifacts (all under gitignored `dev/`): `dev/zebrafinch/h5_e2e/` holds the two configs,
the runner, the comparison script, and both output layers.

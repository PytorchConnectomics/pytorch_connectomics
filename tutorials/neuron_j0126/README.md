# j0126 (zebra finch) — EM image → affinity → ABISS segmentation

Two stages, no conversion step in between:

| stage | config | output |
|---|---|---|
| 1. affinity | `infer_affinity.yaml` | float16 HDF5 under `outputs/neuron_j0126_affinity/` |
| 2. segmentation | `abiss.yaml` | precomputed layer `outputs/neuron_j0126_abiss/precomputed/seg` |

`infer_affinity.yaml` is self-contained: it inherits only `../banis+.yaml` — the shared,
dataset-free banis+ model recipe — and states all j0126-specific data/window/chunking
settings itself, so it does not break when experiment configs elsewhere are moved or
deleted.

```bash
# 1) EM image -> 3-channel affinity, straight into a precomputed layer
python scripts/main.py --config tutorials/neuron_j0126/infer_affinity.yaml \
    --mode test --checkpoint <affinity_model.ckpt>

# 2) affinity -> segmentation (watershed -> remap -> agglomeration -> remap)
python scripts/run_abiss_chunk.py --config tutorials/neuron_j0126/abiss.yaml
```

## Why the affinity stays HDF5

Stage 1 writes the same float16 HDF5 this pipeline has always written, and ABISS
reads it directly through its HDF5 backend (`lib/abiss/scripts/volume_backends.py`).
No conversion step, no second copy.

Converting to a precomputed layer instead would force **float32** — for this volume
~3.1 TB versus ~1.5 TB — purely to satisfy a format. Since ABISS can read the HDF5,
that cost buys nothing.

Zarr is supported by the same backend and is the right choice when output needs
concurrent writers: HDF5 has no multi-process writer support, so it is **read-only**
here. That is fine for affinity, which is written once by inference and only read by
ABISS.

## Why the ROI matters

The source zarr is rounded out to `5700 × 12288 × 12288` while the true EM extent is
`5700 × 10913 × 10664`. Without `inference.chunking.roi` the last ~2 chunks in Y and X
are pure zero padding: **1014 chunks instead of 726**, i.e. ~28% of the compute spent
predicting on zeros that are discarded downstream. The ROI clips the chunk grid to the
real geometry; `param.BBOX` in `abiss.yaml` covers the same box.

## Activation convention

The inherited base emits `scale_sigmoid` = `sigmoid(0.2 · x)` (BANIS' temperature).
ABISS consumes plain `sigmoid(x)`, which the historical upload script recovered
afterwards with a `logit/0.2` restore pass. This tutorial emits `sigmoid` directly, so
that restore step is unnecessary.

## Prerequisites: the ABISS build

Two build/runtime settings are required, and both fail loudly but confusingly if missed:

- **`-DEXTRACT_SIZE=ON`.** Without it `acme` never writes `ns.data` /
  `ongoing_seg_size.data` (the supervoxel sizes), so agglomeration builds an empty
  supervoxel set and aborts with `Should not happen, rg element does not exist`.
  Watershed succeeds either way, so the failure only appears halfway through.
  ```bash
  cd lib/abiss/build && cmake -DEXTRACT_SIZE=ON .. && make -j
  ```
- **`libtbb.so.2`.** The binaries link the old TBB soname; if the runtime only ships
  oneTBB (`libtbb.so.12`), expose just that one library on `LD_LIBRARY_PATH` — adding a
  whole legacy toolchain directory instead can shadow `libstdc++` and break the build
  with a missing `GLIBCXX_3.4.29`.

## Affinity convention (`precomputed_affinity_convention: abiss`)

The model's affinity is not in the layout ABISS reads. Two independent changes are
needed, and both are silent if missed — the segmentation just comes out worse:

1. **Edge shift `v → v-1`.** The model stores an edge on its *source* voxel
   (`v → v+1`); ABISS reads it on the *destination*. This is applied to the **haloed**
   prediction before the core is cropped, so every core face pulls the true neighbouring
   voxel. Doing it chunk-locally after cropping would zero-fill each chunk's low faces
   and corrupt every internal chunk boundary.
2. **Channel reversal `[z,y,x] → [x,y,z]`.** ABISS expects channel 0 = x-affinity; the
   model emits channel 0 = z-affinity (see `dev/zebrafinch/precompute_bndaff.py`, which
   reads `pz = aff[0]`, `py = aff[1]`, `px = aff[2]`). Note the inherited base config
   comments this as "XYZ order", which is misleading.

`dev/zebrafinch/upload_affinity_full_masked.py` applied both as a separate pass over the
saved HDF5; here they happen at write time, so the layer is directly consumable. The
conversion is verified against that reference implementation in the unit tests.

The FFN tissue mask that script also applied is **not** replicated: it was measured to be
~inert for reconstruction quality (NERL 0.701 masked vs 0.697 unmasked on a 14%-masked
region).

## Scoring

ERL/NERL against the j0126 test skeletons is covered by `lib/em_erl/examples/README.md`,
including the two conventions that change the number (resolution and merge threshold).

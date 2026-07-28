# j0126 (zebra finch) — EM image → affinity → ABISS segmentation

Two stages, no conversion step in between:

| stage | config | output |
|---|---|---|
| 1. affinity | `infer_affinity.yaml` | precomputed layer `outputs/neuron_j0126_affinity/affinity` |
| 2. segmentation | `abiss.yaml` | precomputed layer `outputs/neuron_j0126_abiss/precomputed/seg` |

```bash
# 1) EM image -> 3-channel affinity, straight into a precomputed layer
python scripts/main.py --config tutorials/neuron_j0126/infer_affinity.yaml \
    --mode test --checkpoint <affinity_model.ckpt>

# 2) affinity -> segmentation (watershed -> remap -> agglomeration -> remap)
python scripts/run_abiss_large.py --config tutorials/neuron_j0126/abiss.yaml
```

## Why the affinity is written as precomputed

ABISS reads `AFF_PATH` through CloudVolume, so writing inference output as a
**precomputed layer** means ABISS consumes it directly. `inference.chunking.precomputed`
writes each inference chunk into the layer instead of per-chunk HDF5 + a stitch pass,
which removes an entire second copy of the affinity (3.1 TB for this volume).

Chunks are disjoint and storage-chunk aligned, so ranks write the shared layer
concurrently without locking. `precomputed_chunk_size` **must divide the inference
`chunk_size`** on every axis (here 1008 = 7·144 = 14·72); a non-dividing value is
rejected at startup rather than silently racing two ranks on one storage chunk.

Zarr is *not* used for the affinity: cloud-volume can read a 3D zarr, but for a 4D
multi-channel array it exposes only one channel, and it cannot write zarr at all —
so a 3-channel affinity zarr is not readable by ABISS.

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

## Known delta vs. the reference segmentation

The affinity here is written in the model's native edge convention (edge `v → v+1`).
The reference zebrafinch layer was additionally shifted to the destination-stored
convention (`v → v-1`) by `dev/zebrafinch/upload_affinity_full_masked.py` before ABISS
consumed it. That one-voxel convention shift is **not** applied by this tutorial, so
results are not bit-comparable to that reference. The FFN tissue mask that script also
applied was measured to be ~inert for reconstruction quality (NERL 0.701 masked vs 0.697
unmasked on a 14%-masked region), so it is omitted here.

## Scoring

ERL/NERL against the j0126 test skeletons is covered by `lib/em_erl/examples/README.md`,
including the two conventions that change the number (resolution and merge threshold).

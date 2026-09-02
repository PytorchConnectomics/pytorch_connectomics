# j0126 staged storage cleanup

How to reclaim space while the whole-volume run is in flight. See
[RESOURCE.md](RESOURCE.md) for the totals these steps bring down, and
[README.md](README.md) for the workflow.

Cleanup trades storage for restartability. Run each block only after the named downstream artifact exists. Replace the first path with the resolved `params.paths.output_root`; the guards refuse an empty path, `/`, or a run without the expected output markers.

After EC `sizes` has written its aggregated inventory and ABISS has completed, remove the ABISS affinity copy, watershed, chunk map, resume scratch, and run workspace. This normally recovers several TiB while preserving the final ABISS segmentation and its parameter file:

```bash
J0126_OUTPUT_ROOT="/absolute/path/to/outputs/neuron_j0126"
ABISS_ROOT="$J0126_OUTPUT_ROOT/abiss"
EC_ROOT="$J0126_OUTPUT_ROOT/error_correction_v7"

case "$J0126_OUTPUT_ROOT" in ""|"/") exit 2;; esac
test -f "$ABISS_ROOT/precomputed/seg/info" || exit 2
test -s "$EC_ROOT/segment_sizes.data" || exit 2

rm -r -- \
  "$ABISS_ROOT/precomputed/affinity" \
  "$ABISS_ROOT/precomputed/ws" \
  "$ABISS_ROOT/chunkmap" \
  "$ABISS_ROOT/scratch" \
  "$ABISS_ROOT/run"
```

After `skeletons`, `contact_graph`, and `junction_features` complete, their per-chunk caches are redundant. Removing them preserves the aggregated morphology, contact graph, and junction features used by the resolver:

```bash
J0126_OUTPUT_ROOT="/absolute/path/to/outputs/neuron_j0126"
EC_ROOT="$J0126_OUTPUT_ROOT/error_correction_v7"

case "$J0126_OUTPUT_ROOT" in ""|"/") exit 2;; esac
test -s "$EC_ROOT/segment_skeleton_graph.h5" || exit 2
test -s "$EC_ROOT/contact_graph.npz" || exit 2
test -s "$EC_ROOT/junction_features_raw.npz" || exit 2

rm -r -- \
  "$EC_ROOT/skeleton_chunks" \
  "$EC_ROOT/contact_chunks" \
  "$EC_ROOT/skeleton_cache"
```

The original 2.5–3 TiB affinity store is needed through the EC `contacts` stage, but not after the final output verifies. At that point it can be archived or deleted; set the exact store explicitly rather than deleting the whole affinity output directory:

```bash
J0126_OUTPUT_ROOT="/absolute/path/to/outputs/neuron_j0126"
EC_ROOT="$J0126_OUTPUT_ROOT/error_correction_v7"
AFFINITY_STORE="/absolute/path/to/the/affinity.h5.chunks"

case "$J0126_OUTPUT_ROOT" in ""|"/") exit 2;; esac
case "$AFFINITY_STORE" in ""|"/") exit 2;; esac
test -f "$EC_ROOT/error_correction_manifest.json" || exit 2
find "$AFFINITY_STORE" -maxdepth 1 -name 'chunk_z*_y*_x*.h5' -print -quit \
  | grep -q . || exit 2

rm -r -- "$AFFINITY_STORE"
```

Keep `precomputed/seg`, `error_correction_manifest.json`, `segment_sizes.data`, the resolver reports, and `resolver/v7/frozen_junction_merges.{npz,json}`. Those are the compact result and provenance record. Also keep the source affinity if exact contact regeneration matters more than storage.

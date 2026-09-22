# Plan v1

## Summary

Replace the four-stage, tile-download j0126 ERL workflow with a single
CloudVolume-backed command. The new `scripts/j0126_workflow.py` takes one required
argument — the ground-truth skeleton HDF5 path — builds the ERL graph in memory,
samples FFN segment ids directly from the public `ffn_segmentation` CloudVolume at
each skeleton vertex, and prints the ERL. No downloaded tiles, no intermediate files,
no subcommands.

The rewrite is faithful to the existing metric: it reuses `skel_to_erlgraph` (no
resolution -> voxel-unit edge lengths, exactly as the current `prepare-gt`),
`ERLGraph.get_nodes_position(None)` for the canonical voxel node positions, and
`compute_erl_score`, so numbers stay comparable to the old pipeline. The only real new
code is a chunk-binned, thread-parallel CloudVolume point sampler.

This revision addresses the plan_v0 review: (1) sample at the canonical
`get_nodes_position(None)` grid, not an ad-hoc `round()`; (2) add explicit registration
sentinel checks to verification; (3) drop the out-of-scope `-r/--resolution` flag;
(4) pin CloudVolume URL normalization in a small testable helper.

## Scope

In scope:
- Full rewrite of `lib/em_erl/scripts/j0126_workflow.py`.
- Update the J0126 section of `lib/em_erl/scripts/README.md` to the new one-command
  usage (drop tile-download / sharding instructions; keep the data reference links).
- One offline unit test covering the chunk-binned sampler and the URL normalizer
  (fake volume, no network).

Out of scope:
- Changing `em_erl` library modules (`erl.py`, `eval.py`, `sampling.py`, `io.py`).
  The sampler lives in the script to keep the change surgical; extraction into
  `em_erl` is noted as a future option, not done here.
- Any physical-unit ERL option (`-r/--resolution`): removed. The script preserves the
  current voxel-unit ERL convention only, so results stay comparable and unambiguous.
- The other scripts (`volume_eval.py`, `seg_to_graph.py`, `skel_to_graph.py`).

## Proposed Changes

### 1. New `scripts/j0126_workflow.py`

CLI (single command, no subparsers):

```
python scripts/j0126_workflow.py \
    -g /projects/weilab/dataset/zebrafinch/test_50_skeletons.h5 \
    [--seg-url gs://j0126-nature-methods-data/.../ffn_segmentation] \
    [-mt/--merge-threshold 50] \
    [-w/--num-workers 16] \
    [-o/--output-path erl_score.pkl]
```

- `-g/--gt-skeleton` (required): path to the GT skeleton HDF5.
- `--seg-url`: default = the exact `gs://.../ffn_segmentation` path from the task.
- `-mt/--merge-threshold`: default `50` (matches the current `score` default).
- `-w/--num-workers`: thread-pool size for CloudVolume sampling, default `16`.
- `-o/--output-path`: optional; if set, pickle the `ERLScore` (mirrors
  `volume_eval.py`). Default `""` (print only).
- (No `-r/--resolution`: removed per review; voxel-unit ERL only.)

Program flow (one `run_j0126_eval(...)` function + thin `main`):

1. **Load skeletons.** Open the HDF5, iterate groups sorted by the same key order the
   current `prepare-gt` uses (`int(key)` with string fallback), read `vertices`
   (zyx voxel int) and `edges` into a `{skel_id: SimpleNamespace(vertices, edges)}`
   dict.
2. **Build ERL graph.** `graph = skel_to_erlgraph(skel_dict)` (no resolution). This
   fixes the canonical node ordering and voxel-unit edge lengths.
3. **Canonical node positions.** `node_zyx = graph.get_nodes_position(None)` — the same
   API `volume_eval.py` uses (returns `node_coords_zyx.astype(int)`, i.e. integer voxel
   zyx in graph order). This is the single source of truth for both the graph node
   order and the sampling grid; the sampler is fed exactly this array, so
   `len(node_segment_lut) == graph.num_nodes` and order is identical by construction.
   (Coords are < 2^24, so the float32 store is exact; truncation == the original ints.)
4. **Sample segment ids from CloudVolume** (new helper, see #2) ->
   `node_segment_lut` (uint64), out-of-bounds -> 0.
5. **Report.** `print_skeleton_assignment_zero_stats(node_segment_lut)`, then
   `compute_erl_score(graph, node_segment_lut, None, merge_threshold)`,
   `score.compute_erl()`, `score.print_erl()`. Optionally pickle to `-o`.

### 2. Chunk-binned CloudVolume point sampler (in the script)

`normalize_seg_url(seg_url) -> str` (small testable helper):
- Strip a leading `precomputed://` if present, then require the remainder to start with
  `gs://` or `https://` (raise a clear `ValueError` otherwise), and return
  `"precomputed://" + remainder`. This pins the exact CloudVolume path form and is
  unit-tested without network.

`open_seg_cloudvolume(seg_url, mip=0)`:
- `from cloudvolume import CloudVolume` (lazy import; clear error if missing) and
  return `CloudVolume(normalize_seg_url(seg_url), mip=mip, use_https=True,
  fill_missing=True, bounded=False, progress=False)`. `use_https=True` with the
  `gs://` source is the form already confirmed working against this public bucket in
  research.

`sample_cloudvolume_lut(cv, node_zyx, num_workers) -> np.ndarray[uint64]`:
- Accept an already-opened `cv` (so tests can inject a fake). Read `volume_size`
  (xyz) and `chunk_size` (xyz) from `cv`.
- Convert node zyx -> xyz. Mark out-of-bounds (`xyz < 0` or `xyz >= volume_size`) ->
  left as 0, excluded from fetches.
- Bin in-bounds points by native chunk grid `xyz // chunk_size` via
  `np.unique(chunk_ids, axis=0, return_inverse=True)`; group point indices per unique
  chunk.
- For each occupied chunk, fetch the chunk-aligned box `cv[x0:x1, y0:y1, z0:z1]`
  (clamped to `volume_size`), squeeze the channel axis, and index the box at local
  point offsets to fill `lut[idx]`.
- Parallelize chunk fetches with `concurrent.futures.ThreadPoolExecutor`
  (`max_workers=num_workers`); network I/O releases the GIL. Share the one `cv`
  instance (thread-safe for reads). Print a progress line every K chunks for a
  liveness signal.
- Return `lut` (length == len(node_zyx)); out-of-bounds and any unwritten -> 0.

Rationale for chunk-binning over `em_erl.sampling` z-slabs: a single mip0 z-slab is
~924 MB (10880*10624*8 B); slab-scanning the whole volume is infeasible. Points lie on
1-D skeleton curves, so only 82,845 of the volume's chunks are occupied.

### 3. `scripts/README.md`

Rewrite the J0126 subsection: one command, note it streams from the public bucket
(no download), keep the GT skeleton / data links, note `pip install -e ".[cloud,h5]"`.
Remove the 4-step `prepare-gt/map-lut/reduce-lut/score` block and the tile path
layout.

## Files and Areas

| File | Change |
|---|---|
| `lib/em_erl/scripts/j0126_workflow.py` | Full rewrite: single command + CloudVolume sampler + URL normalizer |
| `lib/em_erl/scripts/README.md` | Rewrite J0126 usage section |
| `lib/em_erl/tests/test_j0126_workflow.py` (new) | Offline unit tests: `sample_cloudvolume_lut` with a fake CloudVolume + `normalize_seg_url` |

## Verification Plan

1. **Import / CLI smoke.** `python scripts/j0126_workflow.py -h` lists the single
   command and args (no `-r`). No syntax/import errors.
2. **Offline unit tests (no network).** New `tests/test_j0126_workflow.py`:
   - **Sampler:** build a small known 3-D labeled array (e.g. 8x8x8). Wrap it in a
     fake object whose `[x0:x1,y0:y1,z0:z1]` returns the xyz sub-box with a trailing
     channel axis, exposing `volume_size` + `chunk_size` (chunk smaller than the array
     to exercise multi-chunk binning). Inject it into `sample_cloudvolume_lut`. Assert
     sampled ids equal direct array lookups for zyx points across different chunks,
     including an out-of-bounds point -> 0. Run single- and multi-worker.
   - **URL normalizer:** assert `normalize_seg_url` maps `gs://x`, `precomputed://gs://x`
     -> `precomputed://gs://x`, and raises on a bare local path.
   - `python -m pytest tests/test_j0126_workflow.py -q`.
3. **End-to-end real run with explicit registration checks (network; the key proof,
   weidf verifies by output).**
   `python scripts/j0126_workflow.py -g /projects/weilab/dataset/zebrafinch/test_50_skeletons.h5 -w 16`
   Before trusting the ERL, the run (or a short companion snippet captured in the code
   artifact) must confirm the registration anchors from research:
   - **Sentinel:** every sampled vertex of skeleton `"0"` maps to segment `1465128`
     (consistent single-neuron assignment); a coordinate swap/offset/scale error would
     break this even when the zero-ratio stays low.
   - **OOB count ≈ 37** of 500,845 (matches the verified mip0 bounds).
   - **Assignment-zero ratio small** (well under the ~0.06 background level from related
     zebrafinch decode work) — necessary but not sufficient, used alongside the sentinel.
   - Prints finite `ERL` / `gt ERL` with `#skel = 50`; completes in a few minutes.
   Capture this stdout into the code artifact. If the sentinel fails, STOP — do not ship
   a silently mis-registered evaluator.
4. **Alignment invariant.** Assert `len(node_segment_lut) == graph.num_nodes`; the LUT
   is built from `graph.get_nodes_position(None)` (graph node order), so re-ordering
   bugs are impossible by construction.
5. **Regression guard.** `python -m pytest tests/ -q` still passes (rewrite only touches
   the script + adds one test; library APIs unchanged).

## Risks and Questions

- **Coordinate convention (highest risk, mitigated).** Correctness depends on skeleton
  zyx voxel == mip0 xyz voxel. Verified empirically (consistent nonzero segment ids;
  far-x points in-bounds rule out a 9-vs-10 nm grid mismatch). Now also guarded at
  runtime by the skeleton-`"0"` sentinel + OOB-count check in step 3.
- **ERL units.** Preserved as voxel-unit (no resolution), matching the current
  `prepare-gt`, so results are directly comparable to the old pipeline. Physical-unit
  option intentionally removed. Open question for the maintainer (does not block this
  task): should the reference ERL eventually move to physical `20,9,9` nm? If so, that
  is a separate, explicit change to both this script and `volume_eval.py` callers.
- **Runtime.** ~82k chunk fetches; with 16 threads expect ~5-15 min depending on GCS
  latency. Far cheaper than downloading two multi-GB zips. Progress printing gives a
  liveness signal so a hang is visible.
- **Segment id dtype.** FFN ids are uint64; keep `node_segment_lut` uint64.
  `compute_erl_score` pair-packs via int64 — FFN ids are well under 2^63, so no
  overflow.
- **Dependency.** Needs `cloud-volume` (already the `[cloud]` extra). Imported lazily
  with a clear error if missing.
- **Out-of-bounds nodes (37).** Assigned 0 => counted as omitted, not merges — the
  correct conservative handling, matching how the tile pipeline drops points outside
  every tile.

## Changes Since Previous Plan Version

- **[major] Canonical node source:** replaced `graph.node_coords_zyx.round().astype(int64)`
  with `graph.get_nodes_position(None)` as the single sampling grid (same API as
  `volume_eval.py`); removed the ad-hoc rounding rule.
- **[major] Registration proof:** verification step 3 now requires explicit sentinel
  checks (skeleton `"0"` -> segment `1465128`, OOB count ≈ 37) in addition to the
  assignment-zero ratio, with a hard stop if the sentinel fails.
- **[minor] Scope tightening:** removed the `-r/--resolution` flag; the script is
  voxel-unit only.
- **[minor] CloudVolume opener pinned:** added a small, unit-tested `normalize_seg_url`
  helper and an explicit `open_seg_cloudvolume` form (`use_https=True`, `gs://` source),
  and a URL-normalizer test case.

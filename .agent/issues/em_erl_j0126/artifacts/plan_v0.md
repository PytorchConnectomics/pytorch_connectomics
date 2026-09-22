# Plan v0

## Summary

Replace the four-stage, tile-download j0126 ERL workflow with a single
CloudVolume-backed command. The new `scripts/j0126_workflow.py` takes one required
argument — the ground-truth skeleton HDF5 path — builds the ERL graph in memory,
samples FFN segment ids directly from the public `ffn_segmentation` CloudVolume at
each skeleton vertex, and prints the ERL. No downloaded tiles, no intermediate files,
no subcommands.

The rewrite is faithful to the existing metric: it reuses `skel_to_erlgraph` (no
resolution -> voxel-unit edge lengths, exactly as the current `prepare-gt`) and
`compute_erl_score`, so numbers stay comparable to the old pipeline. The only real new
code is a chunk-binned, thread-parallel CloudVolume point sampler.

## Scope

In scope:
- Full rewrite of `lib/em_erl/scripts/j0126_workflow.py`.
- Update the J0126 section of `lib/em_erl/scripts/README.md` to the new one-command
  usage (drop tile-download / sharding instructions; keep the data reference links).
- One offline unit test for the chunk-binned sampler (fake volume, no network).

Out of scope:
- Changing `em_erl` library modules (`erl.py`, `eval.py`, `sampling.py`, `io.py`).
  The sampler lives in the script to keep the change surgical; extraction into
  `em_erl` is noted as a future option, not done here.
- Changing the ERL metric convention (voxel-unit vs physical). Preserve current
  behavior; expose physical units only as an optional, default-off flag (see below).
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
- `-r/--resolution` (optional, default off): if provided as `z,y,x` nm, pass to
  `skel_to_erlgraph` so ERL is in physical units. Default keeps current voxel-unit
  behavior. (Include only if it does not complicate the core path; it is a 2-line
  passthrough. Reviewer may cut it.)

Program flow (one `run_j0126_eval(...)` function + thin `main`):

1. **Load skeletons.** Open the HDF5, iterate groups sorted by the same key order the
   current `prepare-gt` uses (`int(key)` with string fallback), read `vertices`
   (zyx voxel) and `edges` into a `{skel_id: SimpleNamespace(vertices, edges)}` dict.
2. **Build ERL graph.** `graph = skel_to_erlgraph(skel_dict)` (or with `resolution`
   scale if `-r` given). This fixes the canonical node ordering.
3. **Node positions for sampling.** Take integer voxel zyx positions aligned 1:1 with
   `graph`'s node array:
   - Default (no `-r`): `node_zyx = graph.node_coords_zyx.round().astype(int64)`
     (equals the raw vertices in graph order).
   - With `-r`: `node_zyx = graph.get_nodes_position(resolution)` (physical/res ->
     voxel), matching `volume_eval.py`.
   Node order is identical to `node_segment_lut` order required by
   `compute_erl_score`.
4. **Sample segment ids from CloudVolume** (new helper, see #2) ->
   `node_segment_lut` (uint64), out-of-bounds -> 0.
5. **Report.** `print_skeleton_assignment_zero_stats(node_segment_lut)`, then
   `compute_erl_score(graph, node_segment_lut, None, merge_threshold)`,
   `score.compute_erl()`, `score.print_erl()`. Optionally pickle to `-o`.

### 2. Chunk-binned CloudVolume point sampler (in the script)

`sample_cloudvolume_lut(seg_url, node_zyx, num_workers, mip=0) -> np.ndarray[uint64]`:

- Open `CloudVolume("precomputed://"+seg_url, mip=mip, use_https=True,
  fill_missing=True, bounded=False, progress=False)` (strip an existing
  `precomputed://` prefix if present; accept `gs://` or `precomputed://`).
- Convert node zyx -> xyz. Compute `volume_size` (xyz). Mark out-of-bounds
  (`xyz < 0` or `xyz >= volume_size`) -> assigned 0, excluded from fetches.
- Bin in-bounds points by native chunk grid `xyz // chunk_size`; group point indices
  per unique chunk (use a dict keyed by the integer chunk tuple, or `np.unique(...,
  axis=0, return_inverse=True)`).
- For each occupied chunk, fetch the chunk-aligned box
  `cv[x0:x1, y0:y1, z0:z1]` (clamped to `volume_size`), squeeze the channel, and
  index the box at local point offsets to fill `lut[idx]`.
- Parallelize chunk fetches with `concurrent.futures.ThreadPoolExecutor`
  (`max_workers=num_workers`); network I/O releases the GIL. Share one `CloudVolume`
  instance (thread-safe for reads). Optional lightweight progress print every K
  chunks.
- Return `lut` (length == len(node_zyx)); out-of-bounds and any unwritten -> 0.

Rationale for chunk-binning over `em_erl.sampling` z-slabs: a single mip0 z-slab is
~924 MB (10880*10624*8 B); slab scanning the whole volume is infeasible. Points lie on
1-D skeleton curves, so only 82,845 of the volume's chunks are occupied.

### 3. `scripts/README.md`

Rewrite the J0126 subsection: one command, note it streams from the public bucket
(no download), keep the GT skeleton / data links, note `pip install -e ".[cloud,h5]"`.
Remove the 4-step `prepare-gt/map-lut/reduce-lut/score` block and the tile path
layout.

## Files and Areas

| File | Change |
|---|---|
| `lib/em_erl/scripts/j0126_workflow.py` | Full rewrite: single command + CloudVolume sampler |
| `lib/em_erl/scripts/README.md` | Rewrite J0126 usage section |
| `lib/em_erl/tests/test_j0126_workflow.py` (new) | Offline unit test for `sample_cloudvolume_lut` with a fake CloudVolume-like object |

## Verification Plan

1. **Import / CLI smoke.** `python scripts/j0126_workflow.py -h` lists the single
   command and args. No syntax/import errors.
2. **Offline unit test (no network).** New `tests/test_j0126_workflow.py`:
   - Build a small known 3-D labeled array (e.g. 8x8x8). Wrap it in a fake object
     whose `[x0:x1,y0:y1,z0:z1]` returns the xyz sub-box with a trailing channel axis
     and that exposes `volume_size` + `chunk_size` (chunk smaller than the array to
     exercise multi-chunk binning). Inject it into `sample_cloudvolume_lut` (via a
     `cv=` param or a monkeypatched opener).
   - Assert sampled ids equal direct array lookups for a set of zyx points, including
     points in different chunks and an out-of-bounds point -> 0.
   - Run `python -m pytest tests/test_j0126_workflow.py -q`.
3. **End-to-end real run (network; the key proof, weidf verifies by output).**
   `python scripts/j0126_workflow.py -g /projects/weilab/dataset/zebrafinch/test_50_skeletons.h5 -w 16`
   - Expect: completes in a few minutes; prints an assignment-zero ratio that is
     small (well under the ~0.06 background level seen in related zebrafinch decode
     work — confirms correct registration, not a coordinate offset); prints a finite
     `ERL` / `gt ERL` with `#skel = 50`.
   - Capture stdout into the code artifact as the observed result.
4. **Alignment sanity.** Assert `len(node_segment_lut) == graph.num_nodes` and that
   the LUT is built in graph node order (the sampler is fed `graph.node_coords_zyx`,
   not a separately re-read vertex stack), so no re-ordering bug is possible.
5. **Regression guard for existing tests.** `python -m pytest tests/ -q` still passes
   (the rewrite only touches the script + adds one test; library APIs unchanged).

## Risks and Questions

- **Coordinate convention (highest risk, mitigated).** Correctness depends on skeleton
  zyx voxel == mip0 xyz voxel. Verified empirically (consistent nonzero segment ids;
  far-x points in-bounds rule out a 9-vs-10 nm grid mismatch). The unit test cannot
  cover the real grid; step 3's low assignment-zero ratio is the live check. If that
  ratio is unexpectedly high, stop — do not ship a silently-misregistered evaluator.
- **ERL units.** Current `prepare-gt` uses no resolution => voxel-unit ERL with
  anisotropic (20/9/9 nm) voxels, which is not physically exact. The rewrite preserves
  this to stay comparable; `-r` is offered as opt-in. Flagging for maintainer: is
  voxel-unit the intended reference, or should physical `20,9,9` be the default? Plan
  keeps current behavior unless told otherwise.
- **Runtime.** ~82k chunk fetches; with 16 threads expect ~5-15 min depending on GCS
  latency. Acceptable and far better than downloading two multi-GB zips. Progress
  printing keeps a liveness signal so a hang is visible.
- **Segment id dtype.** FFN ids are uint64; keep `node_segment_lut` uint64.
  `compute_erl_score` pair-packs via int64 — FFN ids are well under 2^63, so no
  overflow. Note but do not special-case.
- **Dependency.** Needs `cloud-volume` (already the `[cloud]` extra). Script imports
  it lazily inside the sampler with a clear error if missing.
- **Out-of-bounds nodes (37).** Assigned 0 => counted as omitted edges, not merges.
  This is the correct, conservative handling and matches how the tile pipeline would
  drop points outside every tile.

## Changes Since Previous Plan Version

Initial plan.

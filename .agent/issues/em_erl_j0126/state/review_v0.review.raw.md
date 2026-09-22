# review_v0 raw reviewer notes (in-session planner review of code_v0)

Reviewed files at working tree vs run_start_ref c24e685:
- scripts/j0126_workflow.py (rewrite, 284 lines)
- scripts/README.md (J0126 section rewrite)
- tests/test_j0126_workflow.py (new offline tests)

## Correctness — VERIFIED end-to-end on real data + network (mip0)

Command: python /tmp/verify_j0126.py (imports the script's functions, real skeleton
file + real CloudVolume). Output:

    numpy 2.2.6
    skeletons loaded: 50 | has id 0: True
    graph nodes: 500845 | kept skeletons: 50
    skeleton id 0 node count: 5481
    cv volume_size xyz: [10624, 10880, 5700] chunk_size: [128, 128, 64]
    Sampling 500808/500845 in-bounds skeleton nodes from 82823 occupied chunks; 37 out-of-bounds
    ... (progress) ...
    sampling done in 11.2 min | lut len 500845 == num_nodes 500845: True
    OOB count: 37 (expected ~37)
    skeleton-0 dominant segment: 1465128 coverage: 0.753 | expected dominant 1465128: True
    skeleton-0 top segments: [(1465128, 4126), (25741678, 349), (2323145, 274), (16860473, 102)]
    assignment-zero: 11948/500845 = 0.0239
    ERL: 96337.92 ; gt ERL: 179065.85 ; #skel: 50

Interpretation:
- LUT length == num_nodes: alignment invariant holds (sampler fed get_nodes_position(None)).
- OOB == 37 exactly matches the predicted bound-spill; handled as segment 0.
- skeleton-0 dominant segment == 1465128 (registration sentinel PASSES). Its 75.3%
  coverage is a genuine FFN over-split of that neuron (top-4 shows 4126/349/274/102),
  i.e. real ERL signal, not a coordinate bug.
- assignment-zero 2.39% — low, consistent with correct registration.
- Voxel-unit ERL, matching the old prepare-gt convention (skel_to_erlgraph, no resolution).
- Offline unit tests: 44 passed (3 new j0126 tests) per code_v0.md.

Code reads clean: zyx->xyz via [:, ::-1]; chunk-aligned boxes clamped to volume_size
(no over-fetch); each unique chunk fetched exactly once; ThreadPoolExecutor; progress
prints. numpy 2.2.6 np.unique(return_inverse) path works (verified live).

## Storage-format / cost analysis (new user requirement: fast + low GCS cost)

Layer facts (measured): unsharded precomputed, encoding=compressed_segmentation,
single chunk_size [128,128,64], block_size [8,8,4]. In UNSHARDED precomputed the chunk
object is the atomic GCS download — a single voxel cannot be fetched without its whole
enclosing chunk. Therefore "fetch each occupied chunk exactly once, chunk-aligned, skip
empty chunks" (what code_v0 already does) is the COST FLOOR at mip0. No per-voxel
savings are possible in this format.

Measured egress (cloudfiles size() on 6 occupied chunks): mean ~43 KB compressed/chunk
=> one-time mip0 run ≈ 82,823 * 43 KB ≈ 3.6 GB.
Saved LUT: 500845 * 8 B = 4.0 MB raw (less gzipped).
=> Saving the LUT and reusing it is ~900x cheaper than re-querying, and re-scoring is
   seconds instead of 11 min.

mip agreement (200-node sample, point queries): mip1==mip0 0.985, mip2==mip0 0.980.
=> A coarser mip cuts chunks ~4x per level (mip1 ≈ ~0.9 GB, ~4x faster) but perturbs
   ~1.5% of node labels -> small ERL drift. Keep mip0 as faithful default; offer --mip
   as a documented, opt-in cost/speed lever.

## Conclusion

code_v0 is functionally correct and registration-verified. The new user guidance
(minimize GCS cost; enable reuse/sharing; faster evaluation) is best served by:
1. Persist the compact node_segment_lut and add a --lut path that skips GCS entirely
   when present (download-once / reuse-and-share). PRIMARY.
2. Optional --mip (default 0) as a documented cost/speed lever with the measured caveat.
3. Optional local chunk cache (--cache-dir) for robustness of the one-time generation.
Sampling itself is already cost-minimal; do not change the fetch strategy.

READY: no (enhancements required to meet the new fast/low-cost + reuse requirement).

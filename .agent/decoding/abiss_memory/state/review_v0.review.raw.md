# Raw review notes (planner = claude, in-session), review_v0

Inputs examined: artifacts/code_v0.md; `git diff` of scripts/run_abiss_volume.py (Python worktree);
`git -C lib/abiss diff` of CMakeLists.txt, src/ws/{atomic_chunk.cpp, basic_watershed.hpp,
region_graph.hpp, agglomeration.hpp, utils.hpp}; lib/abiss/tests/ws_gate.py (lines 225-265,
520-548, 755-800) and tests/ws_benchmark.sbatch; stock build/CMakeFiles/*/flags.make; local gate
logs under build_mem/gate_scratch/logs/real_64x512x512/; benchmark job 3039644 log.

Coordinator action: submitted the step-5 benchmark as instructed (job 3039644, cb003), then
CANCELLED it after ~1 min on finding F1. It produced no results.

F1 [major] Step-5 benchmark driver computes ws thresholds with np.percentile on the raw float16
crop (ws_gate.py:779-780). numpy computes the virtual index in float16, (n-1)*q overflows to inf
for n > 65504, and the result is NaN. Reproduced on a 3x128x1024x1024 crop: np.percentile(c,94) ->
nan, np.percentile(c.astype(np.float32),94) -> 0.7231. With low=high=NaN, `m > low` is false for
every voxel and the watershed does no real work, so any memory number would be meaningless. Fix:
cast to float32 (or use a float32 strided subsample) before percentile, and require finite
thresholds with low < high before any run. The local correctness gate is NOT affected: it uses fixed
0.95/0.05 (logs show 751,241 supervoxels, 4,323,327 edges on the real crop).

F2 [major] Stdout contract regression. atomic_chunk.cpp dropped "num of sv:<c>" / "size of rg:<d>"
(single threshold) and the multi-threshold "(<thr>) ... in <secs> seconds" / "finished
agglomeration/writing in" lines. tutorials/neuron_liconn_ist/slurm/ws_sizing.py:93-94 parses
`num of sv:(\d+)` and `size of rg:(\d+)` from ws stdout, so it silently gets None. Restore the
single-threshold lines verbatim and keep the per-threshold multi line including the threshold value;
restore the timing lines.

F3 [minor] write_volume throws std::runtime_error / ios failure on I/O errors; nothing catches it in
main, so a full disk becomes std::terminate (SIGABRT, exit 134) rather than a clear message and a
defined exit code. Catch in main, print, return a documented code (e.g. 4).

F4 [minor] The internal_seg_t WHY comment (flag bits packed in the id, high_bit ceiling, ws vs ws64
cost) was reduced to one line. Keep the explanation; only the stale sentence about
relabel_segments deducing the on-disk type needs updating.

Checked and OK: C2 cursor reset + clear() keeps traversal order, neighbour arithmetic signed; C3
LUT == remaps[find_set(id)], graph built from same values, raw seg immutable, fresh counts copy per
threshold; C4 in_rg removal; C6 MAX `!count || value < a` == std::max_element first-max; MEAN order;
face positions (front: o=0,i=1; back: o=n-1,i=n-2) and naming; uint32 bound uses compact
counts.size()-1 with overflow-safe comparison; Python default argv unchanged, token only for
uint32, auto bound offset+interior (ids are seeded from interior voxels only), holder pop inside
callee verified by weakref test.

Plan correction (not a code finding): code_v0 is right that stock build/ws keeps the chunk-size
assert. CMakeLists sets CMAKE_CXX_FLAGS_RELEASE "-O3 -fopenmp", which overrides the cached
"-O3 -DNDEBUG"; flags.make for ws and ws64 shows "-O3 -fopenmp" and `strings build/ws` contains the
assert text. plan_v1/v2's "assert compiled out" claim was wrong. C1 is still worth keeping
(defined exit code instead of SIGABRT, also covers product overflow).

Step 5 (memory/runtime acceptance) remains unmeasured; it must run after F1 is fixed.

READY: no

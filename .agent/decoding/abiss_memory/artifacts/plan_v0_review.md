# Plan v0 Review

## Summary
Codex (coder) reviewed plan_v0 with `codex exec --sandbox read-only`; raw transcript at
`state/plan_v0_review.review.raw.md`. Reviewer answered `READY: no` with 7 major and 2 minor
findings. All are fixable within the plan; none is an external blocker. Coordinator verified
one factual point independently: both stock builds are `CMAKE_BUILD_TYPE=Release` with
`CMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG`, so the 2^31 `assert` in `atomic_chunk.cpp` is compiled
out (supports finding 5 and the question about the 3.82 Gvox measurement).

## Findings
1. [major] C5: passing `--seg-dtype=uint64` by default breaks the stock binaries (token parsed as a threshold via `std::stof`). Omit the token for uint64; test default command construction against the old CLI.
2. [major] C4: a uint32 face count can overflow on large ws64 volumes; use `size_t`. MAX must preserve `std::max_element` semantics (first-value init, NaN/signed-zero behaviour); MEAN equality also depends on compiler FP settings; real-crop comparison is evidence, not proof.
3. [major] Memory model omits the owning cropped copy in `write_volume` (8 B/interior voxel) plus the full-size `mapped_file_sink` output mapping in `write_multi_array`, and omits union-find/remap/counts/graph/sort storage. The 24 B/voxel attribution is unsupported; C2 likely saves more than stated. Separate padded vs interior counts, anonymous vs mapped vs cgroup page cache.
4. [major] C0 logging points miss the in-phase peaks (BFS queue, edge containers, sort workspace); VmHWM is cumulative. Instrument inside phases / sample with phase markers; measure the Python parent and job/cgroup peak separately (driver retains `predictions_czyx` during the subprocess); add a concrete acceptance criterion and report runtime.
5. [major] C1 relies on an `assert` that Release builds (`-DNDEBUG`) do not enforce; the 3.82 Gvox `ws` measurement exceeds the ID limit and must be explained. Require checked multiplication and runtime validation; test dispatch around the 2^32 cutoff and both queue implementations on identical inputs.
6. [major] Verification gaps: flag sets never exercise front seg faces or any `aff_i_*`; need deterministic synthetic cases (background-only, isolated comps, dust, ties, repeated edges, multiple thresholds); overflow fixture at `2**32-10` is not deterministic — test exact-fit, one-past, overflow-safe check; test dtype-token placement/invalid values, both widths, batch callbacks, both reader layouts.
7. [major] Byte-identity of `dend_*` is not a sound contract: `write_vector` dumps `std::tuple<F,ID,ID>` object bytes, which may contain uninitialized padding. Check stock-vs-stock determinism first and state the contract explicitly.
8. [minor] C3 LUT can approach voxel count; specify lifetime and free union-find/remap storage early; remove the by-value copy in `relabel_region_graph`.
9. [minor] Further exact reductions to evaluate: reset BFS cursors per search (queue bounded by largest search); `in_rg` appears redundant given the MST different-component test; Python reader full copy from memmap; releasing the affinity mapping after graph construction.

## Questions
- Why is the 3.82 Gvox measurement attributed to `ws` despite its ID limit? (Coordinator: `-DNDEBUG` means it likely overflowed silently; treat as invalid baseline.)
- Which compiler flags produced the stock binaries? (Coordinator: conda `x86_64-conda-linux-gnu-c++`, `-O3 -DNDEBUG`, conda CXXFLAGS incl. `-march=nocona -mtune=haswell -O2`, system allocator, no absl.)
- Acceptance metric: binary RSS, job/cgroup memory, or both?
- If stock graph-file padding is nondeterministic, what byte-exact contract replaces the promise?
- Batch overflow: may earlier thresholds' outputs remain?

## Verdict
VERDICT: NEEDS_CHANGES

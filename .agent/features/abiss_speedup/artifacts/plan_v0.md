# Plan v0

## Summary

Four opportunities, ranked by measured payoff per unit of risk. The headline finding
**corrects a claim in `dev/zebrafinch/lesson_efficiency.md` (L117)**: ABISS's upper-layer
binaries are *not* single-threaded. `src/agg/mean_aggl.cpp` and five other files use C++17
parallel algorithms (`std::execution::par`, 58 call sites), which libstdc++ backs with TBB —
and `ldd` confirms `agg`, `meme`, and `ws2` link `libtbb.so.12`. The earlier "single-threaded"
conclusion came from grepping `tbb::parallel` / `#pragma omp` / `std::thread`, which does not
match this idiom. We then scheduled those exact binaries with 1–2 CPUs.

That reframes the whole problem. The upper composite layers are 64% of wall-clock and were
believed to be an Amdahl floor; they are instead parallel-capable stages that were starved.

Second finding: the build is misconfigured. It inherited conda's `CXXFLAGS` and compiled
with **`-march=nocona`** — a 2004 Pentium 4 target (SSE2/SSE3, no AVX/AVX2/FMA) — on Xeon
Gold 5222 (Cascade Lake). Neither `USE_ABSL_HASHMAP` nor `USE_MIMALLOC` was enabled, so the
region graphs run on node-based `std::unordered_map` with glibc malloc.

## Scope

In scope: build configuration and allocator/container selection in `lib/abiss`; measurement
to confirm where parallelism is actually realized; a documented verdict on fp16 affinity.

Out of scope: any change to segmentation semantics. This pipeline reproduces a Seuron
provenance record; `CHUNK_SIZE`, `AGG_THRESHOLD`, and the WS thresholds are fixed by it.
Every change below must produce bit-identical segmentation, and P1/P2 are gated on proving it.

Cross-repo: opportunity 1 is a scheduling change in pytorch_connectomics
`dev/zebrafinch/submit_wholevol_sharded.sh`, not in lib/abiss. It is listed first because it
is the largest measured payoff and requires no code change; it should be executed as a
separate change, with lib/abiss work confined to this repo.

## Proposed Changes

### P0-A — Confirm and exploit TBB parallelism in composite stages (no code change)

`cfg_for()` currently gives layer 4 one CPU and layer 5 one CPU. `me_L5` (26.3 min) and
`ws_L5` (10.4 min) run `agg`/`meme`/`ws2`, all TBB-linked.

Step 1 is a **measurement, not an edit**: run one L5 composite chunk at
`--cpus-per-task` 1, 4, and 16 and record wall-clock. This is the load-bearing experiment —
if TBB does not actually parallelize (thread pool not initialized, sorts too small, or the
runtime serializes), the entire premise collapses and we must say so.

If it scales, raise cpus for L3/L4/L5 in the launcher. Note the memory interaction:
per-chunk RSS is ~39 GB at L4 and ~98 GB at L5, so cpus and concurrency must be raised
together with memory headroom, not independently.

### P0-B — Fix the build target: `-march=nocona` → native

Set `CMAKE_CXX_FLAGS_RELEASE` so the project's own `-march` wins over the conda-injected
one (currently both appear; the last `-march` on the command line wins, which is
`-mtune=haswell`/`-march=nocona` from conda, since the project appends only `-O3 -fopenmp`).

Use `-march=native -ffp-contract=off`. The `-ffp-contract=off` is not optional: enabling
FMA changes floating-point rounding, and agglomeration compares accumulated affinity sums
against thresholds, so contraction could change merge decisions and therefore the
segmentation. GCC does not reorder FP reductions without `-ffast-math`, which we will not
enable.

### P1-A — Enable `absl::flat_hash_map` (gated on an order-dependence audit)

The dominant memory structures are region-graph maps:
`ChunkedRGExtractor::m_edges` (key `SegPair<uint64>`, value `array<pair<float,size_t>,3>`)
and, worse, `mean_aggl.cpp:158` `std::vector<MapContainer<seg_t, handle_wrapper<...>>> incident`
— one hash map **per segment**, i.e. millions of small node-based maps, each with its own
bucket array plus per-node allocations. This is close to the worst case for
`std::unordered_map` and the likely cause of the 39–98 GB upper-layer footprints.

`CMakeLists.txt` already supports this via `find_package(absl)`; abseil is simply not
installed. Adding it is a dependency + rebuild, not a code change.

**Gate:** `flat_hash_map` has different (and deliberately non-deterministic) iteration order.
Before enabling, audit whether any agglomeration decision depends on map iteration order —
tie-breaking on equal affinity is the specific hazard. If it does, this is not safe for a
fidelity replay and must be rejected or made deterministic.

### P1-B — Enable mimalloc or jemalloc

`CMakeLists.txt` already probes both; neither was found. Given the allocation profile in
P1-A (millions of small nodes), an arena allocator typically wins on both time and
fragmentation. No semantic change — allocator choice cannot alter results.

### P2 — fp16 affinity: analyze and document, do not implement

Recommendation: **do not implement.** Recorded here so it is not revisited without the
evidence. `aff_t` is simultaneously the storage and arithmetic type (accumulators in
`src/agg/edges.h`), so narrowing it wholesale changes numerics. A storage/compute split is
feasible (`AffinityExtractor` is already templated on the chunk type) but the payoff is
small: `aff.raw` is **mmap'd** (`bio::mapped_file_source`), so it is page cache rather than
RSS; only three layer-0 files read it; and at 805 MB per 512×512×256 chunk against ~8 GB
per-chunk RSS it is ~5% of the wrong layer. The CPU has `f16c` (conversion) but not
`avx512_fp16` (arithmetic), so there is no compute win — only added conversions.

## Files and Areas

| Area | Change |
|---|---|
| `CMakeLists.txt` | `-march`/`-ffp-contract` in `CMAKE_CXX_FLAGS_RELEASE`; ensure project flags win over conda's |
| build environment | install abseil and mimalloc (or jemalloc) so the existing `find_package`/`pkg_check_modules` succeed |
| `src/global_types.h` | no edit; it already branches on `USE_ABSL_HASHMAP` / `USE_MIMALLOC` |
| `src/agg/mean_aggl.cpp` | read-only audit for map-iteration-order dependence (P1-A gate) |
| (cross-repo) `dev/zebrafinch/submit_wholevol_sharded.sh` | `cfg_for()` cpus for layers 3–5, after P0-A measurement |

## Verification Plan

Fidelity is the primary gate; speed is secondary. For every change:

1. **Bit-identical segmentation on a known chunk.** Re-run chunk `z2_y3_x3` (bbox
   `[3024,3024,2016]`–`[4032,4032,3024]`, the chunk that reproduced the reference at
   adapted-RAND F 0.999998) and compare the output segmentation against the pre-change run.
   Require exact equality of labels after relabel-sequential, not a similarity score.
2. **Timing, same chunk, same node type**, reported as before/after wall-clock.
3. **P0-A specifically:** one L5 composite chunk at cpus = 1 / 4 / 16, wall-clock each.
   Report the scaling curve. A flat curve refutes the premise and must be reported as such.
4. **P0-B specifically:** confirm the emitted flags actually changed
   (`grep CXX_FLAGS build/CMakeFiles/*/flags.make`) — the current bug is precisely that
   intended flags were silently overridden.
5. **P1-A specifically:** the order-dependence audit result must be stated explicitly before
   any timing is reported.
6. Peak RSS via `/usr/bin/time -v` or `sacct MaxRSS` for the memory claims.

No change is accepted on timing evidence alone.

## Risks and Questions

- **[major] The L117 "single-threaded" claim is wrong and is already written into
  `lesson_efficiency.md` and `lesson_abiss.md` L117.** Both need correcting once P0-A
  measurement settles what is actually true. Do not propagate the current text.
- **[major] absl iteration order.** If agglomeration tie-breaks via map order, P1-A changes
  the segmentation. This is the single biggest fidelity risk in the plan and gates P1-A.
- **[major] `-march=native` and FP contraction.** Mitigated by `-ffp-contract=off`, but
  must be *verified* by bit-identical output, not assumed.
- **[minor] `-march=native` on a heterogeneous cluster.** If build and compute nodes differ,
  native may emit unsupported instructions. Prefer an explicit `-march=cascadelake` if the
  partitions are mixed; confirm the node types first.
- **Question:** is `agg` the actual time sink inside `me_L5`, or is it `meme`/`assort`/
  `split_remap`? `composite_chunk_me.sh` runs seven binaries; only `agg` and `meme` link TBB.
  Per-binary timing inside one L5 chunk should come before any scheduling change.
- **Question:** was the 300G→200G memory cut the cause of `ws_L5` regressing 10.4→16.4 min?
  If the binary is TBB-parallel, cpus may matter more than the memory change.

## Changes Since Previous Plan Version

Initial plan.

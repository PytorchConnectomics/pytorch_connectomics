# Plan v1

## Summary

Plan v0 was rejected as not executable or fidelity-safe. All ten findings are accepted; none
is argued down. Three were factual errors in v0 (mmap/RSS accounting, L4 memory conflation,
citation), and the rest were under-specification. This version resolves the reviewer's three
open questions with measurements taken since, and restructures the work into a **profiling
phase that gates everything else**.

Resolved since v0:

- **Nodes are heterogeneous.** Build/login node is Xeon Gold 5222 (**Cascade Lake**);
  compute node `c134` is Xeon Platinum 8352Y (**Ice Lake**). Partitions also mix 48c/64c/96c
  nodes with different feature sets. `-march=native` is therefore **unsafe** and is dropped.
- **Abseil is genuinely absent**, not merely undetected: no `absl/` headers under
  `/usr/include`, `/usr/local/include`, or the pytc conda env, and no `libabsl*.so`.
  Likewise zero mimalloc/jemalloc libraries in the env. So `find_package(absl)` and
  `pkg_check_modules` failing is a true-negative; these are installs, not config bugs.
- **Which binary dominates `me_L5` is still unknown** and is now Phase 0 work, not an
  assumption. Only `agg`, `meme`, `ws2` link TBB; `acme`, `assort`, `reduce_chunk`, `ws`
  do not. `composite_chunk_me.sh` runs seven binaries.

The core claim that survives from v0 — and the reason this run exists — is that the
composite stages use `std::execution::par` and were scheduled with 1–2 CPUs. That is now
stated as a **hypothesis to be measured**, not a ranked payoff.

## Scope

In scope: (a) a profiling phase that measures where composite-stage time goes and whether
TBB parallelism is realized; (b) build-configuration and dependency changes in `lib/abiss`
justified by that profiling; (c) a corrected written verdict on fp16.

Out of scope: any change to segmentation semantics. Every candidate must produce
**bit-identical** segmentation, defined below.

Cross-repo: scheduling lives in pytorch_connectomics `dev/zebrafinch/submit_wholevol_sharded.sh`.
It is explicitly *not* changed in this run; Phase 0 only produces the evidence that would
justify a separate scheduling change.

## Proposed Changes

### Phase 0 — Profiling (gates everything; no code change)

No opportunity is ranked or implemented until Phase 0 reports. Deliverable is a table, not a
recommendation.

**0.1 Per-binary time inside one composite chunk.** Instrument `composite_chunk_me.sh` and
`composite_chunk_ws.sh` invocations with `/usr/bin/time -v` per binary (wrapper, no logic
change) on one L4 chunk and the single L5 chunk. Report wall-clock and MaxRSS per binary.
This answers whether `agg`/`meme` (TBB-linked) or the non-TBB binaries dominate.

**0.2 Does TBB actually parallelize?** For the dominant TBB-linked binary from 0.1, run the
same chunk at `--cpus-per-task` ∈ {1, 4, 16}, and for each record: wall-clock, `MaxRSS`, and
**CPU efficiency** from `/usr/bin/time -v` "Percent of CPU this job got" (>100% is the proof
of realized parallelism; `--cpus-per-task` alone proves nothing). Also confirm CPU binding
(`taskset -pc`) since `atomic_chunk_ws.sh` already uses `taskset -c $cpuid`, which may pin
these processes to a single core and defeat TBB regardless of allocation.

**0.3 Resource matrix.** For the same stage, sweep (CPUs-per-chunk × concurrent-chunks) over
at least {1,4,16} × {1,2,4} recording latency, throughput, CPU efficiency, and peak RSS.
CPUs-per-process and chunk concurrency are separate knobs; the matrix is what distinguishes
them.

**0.4 Per-stage memory, WS and ME modeled separately.** Corrected from v0's conflation:
`me_L4` ≈ 78.2/2 ≈ **39 GB**/concurrent chunk, `ws_L4` ≈ 160/2 ≈ **80 GB**/concurrent chunk.
Any later scheduling advice must use the per-stage figure.

**0.5 Determinism baseline.** Run the same chunk twice at 16 CPUs and compare outputs
bit-for-bit. If parallel execution is already nondeterministic at the current settings, the
entire fidelity premise of the pipeline is in question and that must be reported before any
optimization work proceeds.

### Phase 1 — Build configuration (after Phase 0)

**1.A Explicit, conservative `-march`.** `-march=native` is dropped: heterogeneous nodes,
and building on Cascade Lake while running on Ice Lake (or older) risks illegal
instructions. Instead make the architecture an explicit cache variable with a conservative
default, e.g. `-DABISS_ARCH=x86-64-v3` (AVX2/BMI2, safe on Cascade Lake, Ice Lake, and any
Haswell-or-newer node), overridable per site.

The current effective flags end with conda's `-march=nocona -mtune=haswell ... -O2` followed
by the project's `-O3 -fopenmp`; because conda's `-march` appears in `CMAKE_CXX_FLAGS` and
the project only appends, **nocona (2004, SSE2/SSE3) wins**. The fix must append the
project's `-march` *after* the inherited flags and verify it did.

Always with **`-ffp-contract=off`**. FMA contraction changes rounding, and agglomeration
compares accumulated affinity sums against thresholds, so contraction can change merge
decisions. No `-ffast-math`.

**Payoff estimate:** the affected work is integer/hash/compare-heavy rather than FP-dense,
so this is a broad but modest win. Conservative **0–5%**, best case **10–20%** on the
vectorizable scan/sort portions of L0 stages (`ws`, `acme`), which are 82 min of the 225 min
total. Expected absolute saving: 0–15 min. To be measured, not assumed.

### Phase 2 — Containers and allocator (each gated independently)

**2.A Abseil `flat_hash_map`.** Requires *installing* abseil (confirmed absent). The target
structures are `ChunkedRGExtractor::m_edges`, `ContactSurfaceExtractor::m_surfaces`,
`AffinityExtractorME::m_edges`, and `mean_aggl.cpp:158`
`std::vector<MapContainer<seg_t, handle_wrapper<...>>> incident` — one hash map per segment,
i.e. millions of small node-based maps, the worst case for `std::unordered_map`.

**Payoff estimate:** flat vs node-based typically saves 30–50% of map memory and 1.5–2.5× on
lookup-heavy loops. Applied to the 39–98 GB composite footprints, conservative **15%**
memory, best case **40%**; time 0–30%. Both to be measured per stage.

**Gate (blocking):** audit **every** `MapContainer`/`SetContainer` use, not just
`mean_aggl.cpp`, for iteration-order dependence. Pass criterion: no agglomeration decision,
tie-break, or output ordering depends on map traversal order — verified by reading each
iteration site and by the bit-identity gate below. `flat_hash_map` iteration order is
deliberately unstable across runs and salted; if any decision depends on it, **reject**.

**2.B Allocator — mimalloc and jemalloc benchmarked separately.** Both are absent and must be
installed. Enablement differs: mimalloc via `-DUSE_MIMALLOC` plus
`<mimalloc-new-delete.h>` (override at link), jemalloc via `pkg_check_modules` link flags
only (transparent `malloc` interposition). They are mutually exclusive in `CMakeLists.txt`'s
`if/elseif`; benchmark each against the same baseline in separate builds.

Correcting v0: "allocator choice cannot alter results" was too absolute. Different layout can
expose address-dependent behavior or latent UB. The full fidelity gate applies.

**Payoff estimate:** conservative **5%**, best case **25%** time on allocation-heavy
composite stages; memory effect can be negative (arena overhead) or positive (less
fragmentation) — must be measured, not assumed.

### Phase 3 — fp16 affinity: rejected, with corrected accounting

**Recommendation: do not implement.** v0's justification contained a factual error, corrected
here: touched file-backed mmap pages **do** count toward process RSS; they are evictable and
not heap-allocated, but they are resident. Corrected numbers: `aff.raw` is 805 MB per
512×512×256 chunk ≈ **10%** of the ~8 GB per-chunk L0 footprint; fp16 would save ~402 MB
≈ **5%**.

The rejection stands on: (1) it applies only to L0, which is 36% of wall-clock and not the
memory-constrained layer; (2) `aff_t` is both storage and arithmetic type
(`src/agg/edges.h` accumulators), so a storage/compute split is required — feasible via the
existing `AffinityExtractor<Ts,Ta,Chunk>` template but non-trivial; (3) the CPU has `f16c`
(conversion) but not `avx512_fp16` (arithmetic), so there is no compute win, only added
conversions. Acknowledged upside not in v0: halved affinity **I/O and page-cache pressure**,
which matters only if Phase 0 shows L0 is I/O-bound.

## Files and Areas

| Area | Change |
|---|---|
| `scripts/composite_chunk_{me,ws}.sh` | Phase 0 only: `/usr/bin/time -v` wrappers for per-binary profiling (revert or keep behind a flag) |
| `CMakeLists.txt` | `ABISS_ARCH` cache variable; append project `-march` after inherited flags; `-ffp-contract=off`; ensure absl/mimalloc branches are reachable |
| build environment | install abseil; install mimalloc and jemalloc (separately) |
| `src/global_types.h` | no edit — already branches on `USE_ABSL_HASHMAP` / `USE_MIMALLOC` |
| all `MapContainer`/`SetContainer` sites | read-only iteration-order audit (2.A gate) |
| (not changed here) `dev/zebrafinch/submit_wholevol_sharded.sh` | Phase 0 produces evidence only |

## Verification Plan

**Fidelity gate (blocking, applies to every change).** Strengthened per review:

1. Compare the **unrelabelled** segmentation: identical `shape`, `dtype`, and exact
   element-wise equality of raw label values — not partition equivalence, not
   `relabel_sequential`, not a similarity score.
2. Coverage is the **complete affected output**, not one chunk: per-chunk SHA-256 of every
   output block across the run, plus an exhaustive array comparison. A single interior chunk
   is explicitly insufficient — the same mistake produced a 0.999998 pass while the whole
   volume was wrong (`reports/wholevol_h5_unmasked_0340.md`).
3. **Repeat runs** at the parallel settings under test (≥2 identical runs) to catch
   scheduling-dependent nondeterminism. Any difference between repeats fails the gate and is
   reported as a finding about the pipeline, not about the change.
4. Baseline for comparison is the pre-change build at `run_start_ref` on the same chunk set.

**Per-change verification:**

- **1.A:** dump effective flags for **every** target
  (`grep CXX_FLAGS build/CMakeFiles/*/flags.make`) and assert exactly one `-march`, that it
  is the intended value, that `-ffp-contract=off` is present, and that no `-ffast-math`/
  `-Ofast`/`-funsafe-math-optimizations` appears. The v0 bug was silently-overridden flags,
  so verifying the emitted command line is mandatory, not optional.
- **2.A:** the order-dependence audit result is stated explicitly (site list + verdict)
  *before* any timing number is reported.
- **2.B:** two independent builds (mimalloc, jemalloc) each vs the same baseline; report
  time and peak RSS separately per allocator.
- **All:** wall-clock and peak RSS via `/usr/bin/time -v`, same node type, stated explicitly.

No change is accepted on timing evidence alone, and no ranking is published before Phase 0.

## Risks and Questions

- **[major, carried] `lesson_abiss.md` L117 and `lesson_efficiency.md` assert the binaries
  are single-threaded.** That is contradicted by 58 `std::execution::par` matches and TBB
  linkage in `agg`/`meme`/`ws2`. Both documents must be corrected once Phase 0 measures what
  is actually realized. Correction should not be written before the measurement, to avoid
  replacing one unverified claim with another.
- **[major] `taskset -c $cpuid` may pin composite binaries to one core.** Seen in
  `atomic_chunk_ws.sh`. If the composite scripts do the same, TBB cannot use extra CPUs no
  matter what SLURM grants, and the whole Phase-0 premise may resolve negative. This is the
  first thing 0.2 should check.
- **[major] Abseil iteration order** remains the largest fidelity risk; 2.A is rejected
  outright if determinism cannot be established.
- **[minor, fixed] Citations.** L117 is in `lesson_abiss.md`. "58" is a count of matching
  lines from `grep -rn ... | wc -l`, not verified-distinct call sites.
- **Question:** are 48c/96c partition nodes an even older microarchitecture than Cascade
  Lake? If so `x86-64-v3` may still be too aggressive and the default should drop to
  `x86-64-v2`. Resolve by sampling `/proc/cpuinfo` on one node per feature class before 1.A.

## Changes Since Previous Plan Version

Every finding from `plan_v0_review.md` is addressed:

1. **Fidelity test insufficient** → rewritten: unrelabelled exact value equality, complete
   affected output with per-chunk hashes, repeat runs for nondeterminism, explicit baseline.
2. **P0-A ranked before its experiment** → removed the ranking. All profiling moved into
   Phase 0, which explicitly "gates everything else" and whose deliverable is a table, not a
   recommendation. No opportunity carries a rank until Phase 0 reports.
3. **P0-A resource model under-specified/unsafe** → added CPU-efficiency measurement
   (>100% as proof), CPU-binding check, and an explicit CPUs × concurrency matrix; separated
   CPUs-per-process from chunk concurrency.
4. **L4 memory premise inconsistent** → corrected and separated: `me_L4` ≈ 39 GB,
   `ws_L4` ≈ 80 GB per concurrent chunk.
5. **Missing payoff estimates** → conservative/best-case ranges added for 1.A, 2.A, 2.B, each
   tied to a named measured stage; fp16 quantified. Reranking deferred to post-Phase-0 by
   design.
6. **P0-B mis-sequenced for cluster safety** → node homogeneity resolved (heterogeneous:
   Cascade Lake build node vs Ice Lake compute node); `-march=native` **dropped** in favour
   of an explicit `ABISS_ARCH` cache variable defaulting to `x86-64-v3`; per-target flag
   verification added.
7. **P1-A not executable** → confirmed Abseil is genuinely absent (headers and libs checked
   in three prefixes) rather than undetected; audit scope widened from `mean_aggl.cpp` to
   every `MapContainer`/`SetContainer` site; explicit pass/fail and an outright-reject path.
8. **P1-B conflates allocators** → split into separate builds with distinct integration
   paths (mimalloc header override vs jemalloc link interposition), benchmarked
   independently; withdrew the claim that allocator choice cannot alter results.
9. **fp16 factual error** → corrected: touched file-backed mmap pages do count toward RSS.
   Restated as ~10% of chunk footprint, ~5% saving, and added the I/O/page-cache upside.
10. **Citation/counting nits** → L117 attributed to `lesson_abiss.md`; "58" described as
    matching lines, not distinct call sites.

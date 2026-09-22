# Plan v2

## Summary

Final plan version under `p2`. All ten findings from `plan_v1_review.md` are accepted; none
is argued down. Two were verified independently by the coordinator and were errors in v1:

- **v1's architecture argument was invalid.** It inferred ISA heterogeneity from *model*
  heterogeneity. Cascade Lake (build node) and Ice Lake (`c134`) **both support x86-64-v3** —
  the build node reports `avx2 bmi2 fma f16c` — and `48c`/`64c`/`96c` are core counts, not
  ISA classes. v1's "decisively unsafe" conclusion does not follow from its evidence.
  Consequence in v2: **the architecture default becomes unset**, preserving current
  behaviour, with the ISA inventory as an explicit prerequisite before any non-default value.
- **`AGENTS.md:104`: "Do not add runtime dependencies unless explicitly approved."** Verified
  at the repository root. Every dependency-requiring item in v2 is now gated behind an
  explicit approval checkpoint and cannot proceed without it.

The plan is now structured as: **Phase 0 profiling → Phase A synthesis and ranking (with
thresholds) → gated implementation**. The candidate that requires no new dependency and no
semantic change (build flags) is separated from the two that require approval (containers,
allocator).

**This plan reaches a human decision point before implementation** — see `## Risks and
Questions`. Two items cannot be resolved by the planner: the dependency approval, and the
node-ISA inventory.

## Scope

In scope: profiling to locate composite-stage cost; a ranking step with explicit accept/
reject thresholds; build-configuration changes in `lib/abiss`; approval-gated dependency
experiments; a corrected written verdict on fp16.

Out of scope: any change to segmentation semantics; scheduling changes in
pytorch_connectomics (`dev/zebrafinch/submit_wholevol_sharded.sh`) — Phase 0 produces
evidence only.

## Proposed Changes

### Phase 0 — Profiling (no persistent code change; gates everything)

**0.0 Baseline manifest (new, per finding 9).** `run_start_ref` alone cannot baseline a
build-environment experiment. Before any measurement, record and store in `state/`: compiler
identity and version, the full effective `CXX_FLAGS`/`CXX_DEFINES` per target, `CMakeCache.txt`,
`ldd` output for every binary, `conda list -n pytc`, and the SHA-256 of each built binary.
Preserve the baseline segmentation outputs immutably (copy, not regenerate). Every later
comparison names the manifest it was measured against.

**0.1 Per-binary cost.** Profile via an **external non-persistent wrapper** — a
`ABISS_PROFILE_PREFIX` environment variable consumed by a copy of the composite scripts under
`state/profiling/`, leaving `scripts/` untouched (resolves the v1 self-contradiction; no file
in `scripts/` is modified). Measure wall-clock and MaxRSS per binary for all seven binaries
in `composite_chunk_me.sh` and for `composite_chunk_ws.sh`.

Sampling, per finding 2: **at least 5 representative L4 chunks** spanning the observed cost
spread (chunk cost within a layer is known non-uniform — `ws_L4` gained only 1.23× at 4×
concurrency), plus the single L5 chunk. L5 is measured **latency-only**; it is one chunk and
concurrency on it is not a production workload.

**0.2 Is TBB parallelism realized?** For the dominant TBB-linked binary from 0.1:
- Record `Cpus_allowed_list` from `/proc/<pid>/status` of the running binary (concrete, per
  the minor finding — not `taskset -pc` without a PID).
- Check whether the composite scripts pin with `taskset -c $cpuid` as
  `atomic_chunk_ws.sh` does. **If they pin to one core, TBB cannot use extra CPUs regardless
  of the SLURM allocation and the entire parallel-scheduling premise resolves negative.**
  This is the first check.
- Run at `--cpus-per-task` ∈ {1, 4, 16}; record wall-clock, MaxRSS, and **"Percent of CPU
  this job got"** from `/usr/bin/time -v`. >100% is the only proof of realized parallelism.

**0.3 Resource matrix, admission-checked.** Sweep CPUs-per-chunk × concurrent-chunks, but
**omit any point whose projected memory exceeds node capacity**: with `me_L5` ≈ 98 GB and
`ws_L4` ≈ 80 GB per concurrent chunk, 4 concurrent L5 processes ≈ 392 GB and is inadmissible.
Concurrency sweeps apply to L4 only, and each point is admitted only if
`concurrency × per-chunk RSS ≤ 0.8 × node memory`.

**0.4 Memory attribution (new, per finding 6).** Payoff for containers/allocator cannot be
estimated from whole-process RSS. Attribute heap usage with `valgrind --tool=massif` (or
`heaptrack`) on one L4 ME chunk and report the **fraction of peak heap held by
`MapContainer`/`SetContainer` allocations**. Without this number, 2.A and 2.B carry no
estimate — only a conditional formula.

**0.5 Determinism baseline.** Two identical runs at the highest admitted CPU setting,
compared bit-for-bit. A difference means the pipeline is already nondeterministic under
parallel execution and must be reported before any optimization proceeds.

### Phase A — Synthesis and ranking (new, per finding 1)

Phase 0 ends with an explicit ranking, not a table. For each candidate compute:

```
stage_saving   = measured_stage_minutes x expected_fraction
run_saving     = stage_saving / 225 min baseline
```

**Accept/reject thresholds**, declared in advance:

- **Accept** if projected whole-run saving ≥ 5% (≥ 11 min) **and** the fidelity gate passes.
- **Reject** if < 2% whole-run saving, or if the fidelity gate fails, or if determinism
  cannot be established.
- **Defer** between 2% and 5%, ranked below any accepted candidate.

Parallel scheduling carries a **conditional** estimate: if 0.2 shows CPU efficiency scaling
to *k*× on the dominant composite binary, projected saving is
`(me_L4 + me_L5 + ws_L3..L5 minutes) x (1 - 1/k)`, capped by the non-TBB binaries' share from
0.1. With the measured 143 min of composite time, k=4 on a binary holding 70% of that time
projects ≈ 75 min; k=1 projects zero. The measurement decides which.

### Phase B — Build flags (no new dependency; the only ungated candidate)

**B.1 Exactly one `-march`, and unset by default.** Add `ABISS_ARCH` as a CMake cache
variable, **default empty = current behaviour**. Non-empty values are only permitted after
the ISA inventory (prerequisite below). When set, `CMakeLists.txt` must *strip* any inherited
`-march=`/`-mtune=` from `CMAKE_CXX_FLAGS` before appending its own — appending alone yields
two `-march` options and contradicts the verification criterion (finding 4). Equivalent
accepted alternative: configure with sanitized `CXXFLAGS`.

**B.2 `-ffp-contract=off` unconditionally.** FMA contraction changes rounding, and
agglomeration compares accumulated affinity sums against thresholds. Never `-ffast-math`,
`-Ofast`, or `-funsafe-math-optimizations`.

**Prerequisite for any non-empty `ABISS_ARCH`:** inventory the effective ISA of every
eligible node class (one `/proc/cpuinfo` sample per class in `short`/`medium`), and choose
the **demonstrated common baseline**, not the build node's. Until that inventory exists, the
default stays empty and B.1 is a no-op refactor.

**Payoff:** the composite work is hash/compare/pointer-heavy rather than FP-dense, so this is
modest. Conditional on 0.1: applies to whichever stages 0.1 shows are compute-bound.
Conservative 0–5%, best case 10–20% of *those stages only*. Reranked in Phase A.

### Phase C — Containers and allocator (APPROVAL-GATED, per `AGENTS.md:104`)

**Neither C.1 nor C.2 may begin without explicit user approval to add build dependencies.**

**C.1 Abseil `flat_hash_map`.** First, confirm absence **through the configured toolchain**,
not by prefix inspection (finding 5): run `cmake --find-package` / a configure probe that
reports what `find_package(absl)` actually resolves, covering `CMAKE_PREFIX_PATH`, compiler
include paths, CMake package registries, and static libs. v1's three-prefix check was
suggestive, not conclusive.

Enablement path, stated explicitly: install abseil → `find_package(absl)` succeeds →
`CMakeLists.txt` adds `-DUSE_ABSL_HASHMAP` and links `absl::flat_hash_map`/`absl::flat_hash_set`
→ `src/global_types.h` switches `MapContainer`/`SetContainer`/`HashFunction` to the absl
types. No source edit required; the branch already exists.

**Blocking gate:** audit **every** `MapContainer`/`SetContainer` site for iteration-order
dependence — at minimum `ChunkedRGExtractor`, `ContactSurfaceExtractor`, `AffinityExtractorME`,
`mean_aggl.cpp`. Pass criterion: no agglomeration decision, tie-break, or output ordering
depends on traversal order. `flat_hash_map` order is unstable and salted across runs; if any
decision depends on it, **reject outright**.

**Payoff:** conditional on 0.4. If `MapContainer` allocations are fraction *f* of peak heap,
projected memory saving is `f × (30–50%)`. With no *f* measured, no number is asserted.

**C.2 Allocator, isolated (finding 7).** `CMakeLists.txt` uses `if(MIMALLOC_FOUND)
... elseif(JEMALLOC_FOUND)`, so a jemalloc build in an environment containing both would
silently select mimalloc. Therefore: **separate isolated prefixes** (one env per allocator),
plus explicit `-DABISS_ALLOCATOR=mimalloc|jemalloc|system` to force the branch. Verify at
runtime which allocator is actually loaded (`ldd` on the binary plus
`/proc/<pid>/maps` for the mapped allocator library) — build-time config is not proof.

Withdrawn from v1: "allocator choice cannot alter results." Changed layout can expose
address-dependent behaviour or latent UB, so the full fidelity gate applies to C.2 as well.

### Phase D — fp16 affinity: rejected

Unchanged from v1 and retained for the record. Corrected accounting: touched file-backed
mmap pages **do** count toward RSS; `aff.raw` is 805 MB ≈ 10% of the ~8 GB L0 per-chunk
footprint, so fp16 saves ~402 MB ≈ 5% — of layer 0, which is 36% of wall-clock and not the
memory-constrained layer. `aff_t` is both storage and arithmetic type, so a storage/compute
split is required. CPU has `f16c` but not `avx512_fp16`, so no compute win. Acknowledged
upside: halved affinity I/O and page-cache pressure — relevant only if 0.1 shows L0 is
I/O-bound.

## Files and Areas

| Area | Change |
|---|---|
| `state/profiling/` (new, not `scripts/`) | copies of the composite scripts with an `ABISS_PROFILE_PREFIX` hook; `scripts/` is not modified |
| `state/baseline_manifest.*` | compiler, flags per target, CMakeCache, ldd, conda list, binary SHA-256 |
| `CMakeLists.txt` | `ABISS_ARCH` (default empty) with inherited-`-march` stripping; `-ffp-contract=off`; `ABISS_ALLOCATOR` to force the allocator branch |
| build environments | **approval-gated**: isolated prefixes for abseil, mimalloc, jemalloc |
| `src/global_types.h` | no edit — branches already exist |
| all `MapContainer`/`SetContainer` sites | read-only iteration-order audit (C.1 gate) |
| (not changed) `dev/zebrafinch/submit_wholevol_sharded.sh` | Phase 0 yields evidence only |

## Verification Plan

**Fidelity gate (blocking, every change).** The reviewer assessed v1's gate as sufficient
for the stated constraint; it is carried forward unchanged:

1. Compare **unrelabelled** segmentation: identical `shape`, `dtype`, and exhaustive
   element-wise equality of raw label values. Not partition equivalence, not
   `relabel_sequential`, not a similarity score.
2. Coverage is the **complete affected output**: per-chunk SHA-256 of every output block plus
   an exhaustive array comparison. One interior chunk is explicitly insufficient — that
   mistake produced a 0.999998 pass while the whole volume was wrong
   (`reports/wholevol_h5_unmasked_0340.md`).
3. **≥2 repeat runs** at the parallel settings under test; any difference between repeats
   fails the gate and is reported as a pipeline finding.
4. Baseline is the preserved immutable output from 0.0, identified by manifest — not
   regenerated, and not identified by git ref alone.

Scope note carried from the review: this proves segmentation-array identity, not byte
identity of container metadata or compression. That is the stated constraint.

**Per-change:**

- **B:** dump `CXX_FLAGS` for **every** target in `build/CMakeFiles/*/flags.make`; assert
  exactly one `-march` with the intended value, `-ffp-contract=off` present, and no
  `-ffast-math`/`-Ofast`/`-funsafe-math-optimizations`. Verifying emitted command lines is
  mandatory: silently-overridden flags is the original bug.
- **C.1:** the audit result (site list + verdict) is stated **before** any timing number.
- **C.2:** two isolated builds; report time and peak RSS per allocator separately; prove the
  loaded allocator from `/proc/<pid>/maps`.
- **All:** wall-clock and peak RSS via `/usr/bin/time -v`, same node class, stated explicitly,
  against a named manifest.

No change is accepted on timing evidence alone; no ranking is published before Phase A.

## Risks and Questions

**Blocking decisions the planner cannot resolve — these are why this plan ends at a human
decision point:**

- **[major] Dependency approval.** `AGENTS.md:104` requires explicit approval to add runtime
  dependencies. Phase C (abseil, mimalloc, jemalloc) cannot start without it. Phase 0, Phase
  A, and Phase B require no new dependency and can proceed independently.
- **[major] Node-ISA inventory.** Required before any non-empty `ABISS_ARCH`. Needs cluster
  access to sample one node per class; two of three sampling attempts timed out under load.
  Until done, `ABISS_ARCH` stays empty and B.1 is a no-op.

**Other risks:**

- **[major] `taskset -c $cpuid` pinning may void the premise.** If composite scripts pin as
  `atomic_chunk_ws.sh` does, extra CPUs cannot be used and the parallel opportunity is zero.
  Checked first in 0.2.
- **[major] Abseil iteration order** remains the largest fidelity risk; C.1 is rejected
  outright if determinism cannot be established.
- **[major] `lesson_abiss.md` L117 / `lesson_efficiency.md` assert the binaries are
  single-threaded**, contradicted by 58 `std::execution::par` matching lines and TBB linkage
  in `agg`/`meme`/`ws2`. Correct **after** 0.2 measures what is realized, so one unverified
  claim is not swapped for another.
- **[minor] Phase 0 cost.** ≥5 L4 chunks × 3 CPU settings × 2 repeats is a non-trivial
  cluster spend competing with the running whole-volume job. Sequence after that completes.

## Changes Since Previous Plan Version

All ten findings from `plan_v1_review.md`:

1. **Ranked deliverable missing** → added **Phase A**: explicit synthesis with
   `stage_saving`/`run_saving` formulas and declared accept (≥5%) / defer (2–5%) / reject
   (<2%) thresholds, plus a conditional formula for the scheduling candidate.
2. **Phase 0 unrepresentative / resource-unsafe** → ≥5 representative L4 chunks; L5 is
   latency-only; every matrix point admission-checked against
   `concurrency × per-chunk RSS ≤ 0.8 × node memory`; the ~392 GB point is explicitly excluded.
3. **Architecture prerequisite unresolved** → conceded that v1's inference was invalid
   (Cascade Lake and Ice Lake both support v3; `48c/64c/96c` are core counts). `ABISS_ARCH`
   now **defaults empty**; ISA inventory is a stated prerequisite for any other value.
4. **Flag mechanism contradicts verification** → `CMakeLists.txt` strips inherited
   `-march`/`-mtune` before appending (or configure with sanitized `CXXFLAGS`), so exactly
   one `-march` survives; verification covers every target, not one glob.
5. **Abseil absence overstated** → absence must be confirmed **through the configured
   toolchain**, covering `CMAKE_PREFIX_PATH`, compiler include paths, package registries and
   static libs; full enablement path (option → target → define → typedef) now stated.
6. **Payoff not tied to measurements** → added **0.4 heap attribution** (massif/heaptrack) to
   measure the `MapContainer` fraction of peak heap; C.1/C.2 now carry conditional formulas
   and assert no number until *f* is measured.
7. **Allocator experiments not isolated** → isolated per-allocator prefixes plus explicit
   `ABISS_ALLOCATOR` to force the `if/elseif` branch, with runtime proof from
   `/proc/<pid>/maps`.
8. **Dependency approval missing** → Phase C is approval-gated on `AGENTS.md:104`; Phases 0,
   A, B require no new dependency and are separated so work can proceed without it.
9. **`run_start_ref` insufficient baseline** → added **0.0 baseline manifest** (compiler,
   per-target flags, CMakeCache, ldd, conda list, binary hashes) and immutable preserved
   baseline outputs; every comparison names its manifest.
10. **Phase 0 self-contradiction / imprecise command** → profiling moved to non-persistent
    copies under `state/profiling/` driven by `ABISS_PROFILE_PREFIX`, so `scripts/` is not
    modified and "no code change" is now true; CPU affinity read from
    `/proc/<pid>/status` `Cpus_allowed_list` instead of a PID-less `taskset -pc`.

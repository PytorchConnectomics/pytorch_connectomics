# Plan v1 Review

## Summary

Reviewer: codex (`codex exec --sandbox read-only`). Raw transcript:
`state/plan_v1_review.review.raw.md` (5867 bytes, 9 major + 1 minor, `READY: no`).

The reviewer credits v1 with materially improving the fidelity gate, correctly separating
WS/ME memory, and fixing the fp16 accounting, and explicitly assesses the strengthened gate
as sufficient: exhaustive unrelabelled element equality "prove[s] exact segmentation-array
identity for the compared runs ... sufficient for the stated segmentation constraint."

It still returns `READY: no`. Two findings were independently verified by the coordinator
and are correct:

1. **The architecture claim in v1 is a logical error.** v1 asserted node heterogeneity makes
   `-march=native` unsafe. Cascade Lake (build node) and Ice Lake (`c134`) **both support
   x86-64-v3** — confirmed on the build node, which reports `avx2 bmi2 fma f16c`. The
   `sinfo` labels `48c`/`64c`/`96c` are core counts and say nothing about ISA. Model
   heterogeneity was conflated with ISA heterogeneity; the sampled pair does not establish
   the conclusion v1 drew from it.
2. **`AGENTS.md:104` says "Do not add runtime dependencies unless explicitly approved."**
   Verified present at the repository root. v1 proposed installing abseil, mimalloc, and
   jemalloc with no approval checkpoint.

Also correct and accepted without dispute: v1's flag mechanism (append `-march` after
conda's) contradicts its own verification criterion of "exactly one `-march`"; the Phase 0
matrix is not resource-admissible (4 concurrent L5 processes at ~98 GB each ≈ 392 GB);
the container/allocator payoff percentages were applied to whole-process RSS without
knowing what fraction of RSS those maps occupy; and installing both allocators into one
environment can make the "jemalloc" build silently select mimalloc via the `if/elseif`.

No finding is disputed. All ten are carried into plan_v2.

## Findings

- **[major] Ranked deliverable still missing.** Phase 0 produces "a table, not a
  recommendation" but defines no post-profiling synthesis, ranking, or accept/reject
  thresholds; the parallel-scheduling opportunity has no conditional payoff estimate.
- **[major] Phase 0 not representative or resource-safe.** One L4 chunk cannot characterize
  a layer with non-uniform chunk costs; concurrent execution of the single L5 chunk is not a
  production workload; the `{1,4,16}×{1,2,4}` sweep could demand ~392 GB with no node-memory
  admission check. Use multiple representative L4 chunks, treat L5 as latency-only, omit
  matrix points exceeding node capacity.
- **[major] Architecture prerequisite unresolved.** Model heterogeneity ≠ ISA heterogeneity;
  the sampled Cascade Lake / Ice Lake pair both support v3, so that pair does not make
  `-march=native` unsafe. Inventory effective ISA per eligible node class before choosing a
  default; until then leave the default unset or use a demonstrated common baseline.
- **[major] Flag mechanism contradicts its verification criterion.** Appending
  `-march=x86-64-v3` after inherited `-march=nocona` yields two `-march` options while
  verification demands exactly one. Specify how inherited flags are removed or how
  configuration happens with sanitized `CXXFLAGS`; verify actual compile commands for every
  affected target, not one glob.
- **[major] Abseil-absence finding overstated, integration under-specified.** Three include
  prefixes plus shared libs do not exclude `CMAKE_PREFIX_PATH`, compiler include paths, CMake
  package registries, or static libs. The exact CMake option, package target, compile
  definition, and link requirement that activate `MapContainer = absl::flat_hash_map` are
  still unstated. Confirm absence through the configured toolchain.
- **[major] Payoff estimates not soundly tied to measurements.** Applying generic "30–50% map
  memory" to 39–98 GB whole-process RSS assumes an unknown fraction belongs to those maps.
  The allocator range lacks an allocation profile, an absolute saving against the 143-minute
  composite total, and any memory range. Add heap/allocation attribution or present explicit
  conditional formulas.
- **[major] Allocator experiments not isolated.** With `if/elseif`, installing both into one
  discoverable environment can make the jemalloc build select mimalloc; separate build
  directories do not prevent this. Use isolated prefixes or explicit mutually exclusive
  controls, and verify at runtime which allocator is loaded.
- **[major] Dependency installation lacks the approval required by `AGENTS.md`.** Add an
  approval checkpoint and use isolated, reproducible environments so installing one candidate
  cannot silently alter the baseline or another candidate build.
- **[major] `run_start_ref` is not a sufficient baseline for build-environment experiments.**
  Git does not capture compiler, inherited flags, CMake cache, dependency resolution, or
  loaded allocator. Preserve immutable baseline outputs and record a full build/runtime
  manifest.
- **[minor] Phase 0 self-contradiction and an imprecise command.** It claims "no code change"
  while the file table modifies two scripts and leaves "revert or keep behind a flag"
  undecided; `taskset -pc` requires a PID or an equivalent recorded `Cpus_allowed_list`.

## Questions

- Which node classes are actually eligible for these jobs, and what is the effective ISA of
  each? This is now the blocking prerequisite for any `-march` change.
- What fraction of composite-stage RSS is attributable to `MapContainer` allocations? Without
  this, container/allocator payoff cannot be estimated rather than guessed.
- Will the user approve adding abseil / mimalloc / jemalloc as build dependencies, per
  `AGENTS.md:104`?

## Verdict

VERDICT: NEEDS_CHANGES

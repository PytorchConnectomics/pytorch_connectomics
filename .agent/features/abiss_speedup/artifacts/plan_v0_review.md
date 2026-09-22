# Plan v0 Review

## Summary

Reviewer: codex (`codex exec --sandbox read-only`). Raw transcript:
`state/plan_v0_review.review.raw.md` (4338 bytes, 9 major + 1 minor findings, `READY: no`).

The reviewer accepts the leads as promising but rejects the plan as not executable or
fidelity-safe as written. Three criticisms are substantive and correct on inspection:

1. The fidelity gate was too weak. Equality after `relabel_sequential` proves *partition*
   equivalence, not bit-identical output, and one chunk cannot establish whole-volume
   fidelity for compiler/container/allocator/parallelism changes.
2. The mmap memory accounting was wrong. Touched file-backed pages **do** count toward
   process RSS; they are simply evictable and not heap. The plan's "page cache rather than
   RSS" phrasing is incorrect and the fp16 rejection must be re-argued on corrected numbers
   (~805 MB ≈ 10% of an 8 GB chunk; fp16 saves ~402 MB ≈ 5%).
3. The L4 memory premise conflated two different stages: `me_L4` is ~39 GB per concurrent
   chunk (78.2/2) but `ws_L4` is ~80 GB (160/2). Scheduling recommendations built on a
   single 39 GB figure would be unsafe.

The reviewer is also right that P0-A was ranked "largest measured payoff" while
simultaneously declaring its experiment load-bearing — an internal contradiction — and that
the deliverable's requirement of a payoff estimate per opportunity was not met for P0-B,
P1-A, or P1-B.

No finding is disputed. All ten are carried into plan_v1.

## Findings

Verbatim severity and substance from the raw transcript:

- **[major] Fidelity test insufficient.** Relabel-sequential equality proves partition
  equivalence, not bit-identity. One chunk cannot establish whole-volume fidelity. Require
  direct shape/dtype/value equality of unrelabelled segmentation over the complete affected
  output (chunkwise hashes + exhaustive comparison). Repeat P0-A runs to detect
  scheduling-dependent nondeterminism.
- **[major] P0-A ranked before its load-bearing experiment.** The 64% upper-layer share is
  not an estimate of *parallelizable* time; ME composite runs seven binaries and TBB linkage
  is established only for `agg` and `meme`. Per-binary timing and CPU-utilization/thread
  measurements must precede the payoff claim.
- **[major] P0-A resource model under-specified and partly unsafe.** `--cpus-per-task` does
  not demonstrate TBB observes the allocation; verify CPU binding and effective parallel
  utilization. The plan conflates CPUs-per-process with chunk concurrency; raising both
  increases aggregate memory pressure. Benchmark a CPUs-per-chunk × concurrent-chunks matrix
  over latency, throughput, CPU efficiency, and RSS.
- **[major] L4 memory premise inconsistent with the table.** `me_L4` ≈ 39 GB/chunk,
  `ws_L4` ≈ 80 GB/chunk. Model WS and ME separately.
- **[major] Missing payoff estimates.** P0-B, P1-A, P1-B give no numerical time/memory range
  tied to measurements; P0-A gives an unmeasured upside premise. Add conservative/best-case
  estimates, state which measured stage each applies to, and rerank after profiling.
- **[major] P0-B mis-sequenced for cluster safety.** Node homogeneity must be a prerequisite
  to `-march=native`; architecture should be configurable, not hard-coded. Verification must
  establish effective flags for every affected target: exactly one intended `-march`,
  `-ffp-contract=off`, and no unsafe FP flags.
- **[major] P1-A not executable.** Absence of `USE_ABSL_HASHMAP` does not prove Abseil is
  uninstalled vs disabled/incompletely detected. Identify the exact switch/package target,
  audit every affected `MapContainer` use (not only `mean_aggl.cpp`), define pass/fail for
  ordering and ties, and reject if determinism cannot be established.
- **[major] P1-B conflates two allocators.** Specify integration path and enablement for
  mimalloc and jemalloc separately and isolate their effects. "Allocator choice cannot alter
  results" is too absolute: changed layout can expose address-dependent behavior or latent
  UB. Benchmark each independently; retain the full fidelity gate.
- **[major] fp16 rationale contains a factual error.** Touched file-backed mmap pages do
  contribute to RSS. Corrected accounting: 805 MB ≈ 10% of an 8 GB chunk footprint; fp16
  saves ~402 MB ≈ 5%. The rejection may stand but must use correct numbers and mention
  possible I/O/page-cache savings.
- **[minor] Citation and counting nits.** L117 is in `lesson_abiss.md`, not
  `lesson_efficiency.md`. `grep | wc -l` counts matching lines, not distinct call sites.

## Questions

Raised by the review and to be resolved in plan_v1:

- Are the build node and all compute nodes the same microarchitecture? This gates `-march`.
- Is Abseil genuinely absent from the build environment, or present but undetected by
  `find_package(absl)`?
- Which of the seven binaries in `composite_chunk_me.sh` actually dominates `me_L5` wall
  time, and does any of them show >100% CPU under a multi-CPU allocation?

## Verdict

VERDICT: NEEDS_CHANGES

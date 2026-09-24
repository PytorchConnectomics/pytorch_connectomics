# Review v0

## Summary

Reviewer: claude (`claude --print --tools ""`, prompt-only). Raw transcript:
`state/review_v0.review.raw.md` (7476 bytes, 2 major + 6 minor, `READY: no`).

The implementation does what the plan asked and the isolation held: all work landed in
`work/abiss`, the live `lib/abiss` was untouched, no commits were created, and `HEAD` still
equals `run_start_ref`. The mutation guard (pre/post `git diff` and `git diff --cached`)
shows no change across the review command.

The coordinator independently verified the central build claims rather than trusting
`code_v0.md`: **17/17 targets** carry exactly one `-march`, `-ffp-contract=off`,
`EXTRACT_SIZE`, and zero unsafe FP flags; `build-system/agg` links no allocator while
`build-mimalloc/agg` links `libmimalloc.so.2`.

**One major finding is resolved by coordinator evidence.** The reviewer flagged that the
default allocator changed from auto-detect to `system`, and that this is only safe if the
production binaries had not silently picked one up. Checked against the live build:

```
live build CXX_DEFINES: ... -DEXTRACT_SIZE -DFINAL      (no USE_MIMALLOC, no USE_ABSL_HASHMAP)
ldd live agg / meme / ws2 / acme:  0 allocator libs each
```

The production binaries used the system allocator, so `ABISS_ALLOCATOR=system` as the new
default is configuration-equivalent to the baseline. The concern was legitimate; the answer
is favourable.

**The second major finding stands and blocks approval.** `-ffp-contract=off` is added
unconditionally, including in the default `ABISS_ARCH=""` path that the plan described as
"today's behaviour, a no-op". It is argued safe (no FMA on `-march=nocona`) but not shown
safe. The reviewer names the closing test: compile-and-compare object files with and
without the flag at the default configuration. Until that is done, the default build is not
demonstrably bit-identical to the one that produced the measured baseline.

Also confirmed by the coordinator: the `USE_ABSL_HASHMAP` evidence the reviewer said was
asserted only in prose does hold — 0/17 system-build targets define it (and 0/17 define
`USE_MIMALLOC`, while 17/17 of the mimalloc build do).

## Diff Baseline

run_start_ref: 3c4f56219488441edb4c51ee7ca4f07cdef2cd8f

Diff taken inside the isolated copy `work/abiss` (clone of `lib/abiss`, detached at
`run_start_ref`). `git status` there shows only ` M CMakeLists.txt` plus untracked build
trees; the live checkout is unmodified.

## Findings

- **[major] `-ffp-contract=off` is argued safe, not shown safe.** Added unconditionally, so
  the `ABISS_ARCH=""` default is not the pre-change configuration. Pre-change emitted two
  `-march=nocona` and no `-ffp-contract=off`; post-change emits one and the flag. The
  duplicate-`march` collapse is inert (same value), but the FP directive is a real codegen
  input. Close it by compiling a representative TU with and without the flag at the default
  configuration and comparing object files, or by the full fidelity gate. **Open.**
- **[major → RESOLVED] Default allocator changed from auto-detect to `system`.** Legitimate
  concern; resolved by coordinator evidence above — the production build defined no
  `USE_MIMALLOC` and its binaries link no allocator, so `system` reproduces the baseline
  configuration. No `auto` mode is required. Retained here because the reasoning must be
  recorded, not because work remains.
- **[minor] Six of nine "Files Changed" rows are unreviewable from the diff.** Only
  `CMakeLists.txt` appears; the `state/profiling/*.sh` scripts and README are outside it, so
  "the run-local composite copies cannot default to the live checkout" could not be
  confirmed. `bash -n` is a syntax check, not coverage evidence. Given the incident that
  motivated the isolation, this specific property deserves positive proof.
- **[minor] Phase C executed without citing the recorded approval.** `build-mimalloc` was
  built against the run-local deps prefix. The approval exists (`run.md` `## Approvals`,
  user direction 2026-07-29) but `code_v0.md` does not cite it. Procedural only; nothing
  entered the default build.
- **[minor] `-DCMAKE_DISABLE_FIND_PACKAGE_absl=TRUE` is weak evidence on its own.** The real
  evidence is the absence of `USE_ABSL_HASHMAP` from `flags.make`, which the audit table has
  no column for. Coordinator confirms 0/17 — add the column rather than asserting in prose.
  (Note: `CMakeLists.txt` does contain one `find_package(absl)`, so the flag is not inert.)
- **[minor] The `-march` strip mutates directory scope, not the cache.** A
  `-DCMAKE_CXX_FLAGS="-march=native"` would still show un-stripped in `CMakeCache.txt` while
  `flags.make` shows the sanitized value. Since Phase 0.0 captures `CMakeCache.txt`, a future
  audit reading the cache would be misled. Strip also covers only `-march`/`-mtune`, not
  `-mfma`/`-mavx*`.
- **[minor] `add_compile_options(-ffp-contract=off)` is language-agnostic.** Fine for an
  all-C++ tree; guard with `$<COMPILE_LANGUAGE:CXX>` if C/CUDA is ever added. Related:
  `find_package(PkgConfig REQUIRED)` now runs only inside non-system branches.
- **[minor] Build trees live inside the checkout and are NOT ignored.** Coordinator
  confirmed `git check-ignore build-system` fails, so `build-system/`, `build-mimalloc/`,
  `build-arch-sanitize/` could be swept in by a later `git add`. Add to `.gitignore` or
  build out-of-tree.

## Tests to Add

- Object-file equivalence test for `-ffp-contract=off` at the default configuration
  (closes the open major without a cluster run).
- A positive check that `state/profiling/*.sh` resolve ABISS binaries from the run-local
  copy and cannot fall back to `lib/abiss` — e.g. assert the resolved path is under the run
  folder, and a negative test that the scripts fail loudly if pointed at the live checkout.
- A `USE_ABSL_HASHMAP` / `USE_MIMALLOC` column in the per-target flag audit.
- The blocking fidelity gate (unrelabelled exhaustive equality over the complete output)
  remains unrun and is the release gate, not a unit test.

## Questions

- Should the default build carry `-ffp-contract=off` at all while `ABISS_ARCH` is empty?
  Making the flag conditional on a non-empty `ABISS_ARCH` would make the default a true
  no-op by construction and remove the need to prove it.
- Are the three build trees intended to persist in the work copy, or should they be
  out-of-tree?

## Verdict

VERDICT: NEEDS_CHANGES

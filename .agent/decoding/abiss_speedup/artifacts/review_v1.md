# Review v1

## Summary

Reviewer: claude (`claude --print --tools ""`, prompt-only). Raw transcript:
`state/review_v1.review.raw.md` (8793 bytes, `READY: yes`, minor findings only — the single
`[major]` token appears in a status table marking the v0 finding **Fixed**, not as a new
finding). Mutation guard clean: pre/post `git diff` and `git diff --cached` identical.

**Both `review_v0` majors are closed, and the first one the right way — by construction, not
by argument.** `-ffp-contract=off` now sits inside the `if(ABISS_ARCH)` branch with a
`$<COMPILE_LANGUAGE:CXX>` guard, so the default `ABISS_ARCH=""` configuration cannot emit an
FP directive at all. The coordinator's object-file comparison independently confirms the
default build's codegen matches the build that produced the measured baseline:

```
live build agg.dir/src/agg/mean_aggl.cpp.o  vs  code_v1 default build (same TU)
  objdump -d : 60737 lines each, only the embedded file-path header differs
  readelf -S : section layout identical
  size       : 1177728 vs 1177760 (delta == the longer path string)
```

That also retires the residual v0 concern about the `CMAKE_CXX_FLAGS_RELEASE` restructuring
and the `-march` dedup, which v0 had accepted only as "inert" — it is now measured.

The second major (allocator default `system`) was resolved by coordinator evidence at v0 and
is cited rather than re-litigated, which is the correct handling.

Everything still open is minor and is evidence presentation, not code. The reviewer found no
new problem in the CMake diff, having specifically cleared: `add_compile_options` placement
reaching all 17 targets; the `-march`/`-mtune` regexes handling the duplicated-token case
live actually had; `find_package(PkgConfig REQUIRED)` no longer running in the `system`
branch; and the move to `PkgConfig::MIMALLOC` narrowing include propagation.

**Segmentation bit-identity remains UNVERIFIED** and is correctly labelled so in `code_v1.md`.
That gate — unrelabelled exhaustive equality over the complete output — is the release
blocker and is untouched by this stage.

## Diff Baseline

run_start_ref: 3c4f56219488441edb4c51ee7ca4f07cdef2cd8f

Diff taken in the isolated copy `work/abiss`; `git status` shows only ` M CMakeLists.txt`
and `?? .gitignore`. No commits. Live `lib/abiss` untouched.

## Findings

Status of each v0 finding, per the reviewer's table:

| v0 finding | Status |
|---|---|
| [major] `-ffp-contract=off` unconditional | **Fixed** + proven inert by object-file comparison |
| [major→RESOLVED] allocator default `system` | **Cited**, not re-litigated |
| [minor] six Files Changed rows unreviewable | **Worked, not verified** (finding 1 below) |
| [minor] Phase C approval not cited | **Fixed** — `run.md ## Approvals` quoted |
| [minor] absl evidence prose-only | **Fixed** — `USE_ABSL_HASHMAP`/`USE_MIMALLOC` columns, 34 rows |
| [minor] `-march` strip scope vs cache | **Documented, not fixed** (finding 5) |
| [minor] `add_compile_options` language-agnostic | **Fixed** — generator expression in the diff |
| [minor] build trees not ignored | **Fixed** — `.gitignore` added; coordinator confirms |

New minor findings, carried forward:

1. **[minor] Isolation-harness changes remain outside the reviewable diff.** Six of eleven
   Files Changed rows are `state/profiling/*`; the supplied diff is only `work/abiss`. Not
   blocking — the property they protect (live checkout untouched) was verified directly by
   the coordinator — but the finding is now two rounds old.
2. **[minor] The pasted isolation transcript is narrower than the coverage claimed.** Text
   asserts 26 positive checks; transcript shows eight `resolved_binary:` lines plus one
   `resolved_build_dir:`. Four negative cases claimed, three diagnostics shown. Either the
   output was elided without saying so, or assertions run silently. Emit a summary count
   line (`positive=26 negative=4`) that can be pasted whole.
3. **[minor] Symlink rejection is argued, not shown.** No symlink appears in the test cases
   or transcript. The reviewer notes this is "the same shape as the v0 major it just fixed."
   One case closes it: symlink to the live checkout, expect the same refusal.
4. **[minor] One enforced boundary is looser than described.** `ABISS_REPO` is refused
   outside `<RUN>`, but `ABISS_BIN_PATH` outside `<RUN>/work/abiss`. A repo path under
   `<RUN>` but outside `work/abiss` passes the repo check. Impact small (build-dir and binary
   checks are pinned), but tighten or state the two-tier boundary deliberately.
5. **[minor] Cache/scope divergence documented only in the stage artifact.** `CMakeCache.txt`
   retains `-march=nocona -mtune=haswell` while `flags.make` is sanitized. A future audit
   reads the cache, not `code_v1.md` — put the comment beside the regex rewrite in
   `CMakeLists.txt`.
6. **[minor] The object-file proof is single-TU and executable-sections only.** Right test,
   passes, and generalizing is defensible since the transformation is directory-scope
   uniform — but hashing every `.o` across the two trees is cheap and would make it
   exhaustive. Also `objdump -d` does not compare `.rodata`, precisely where FP-contraction
   would deposit folded constants; `objdump -s -j .rodata` closes that. Not blocking, since
   the flag is absent by construction at the default.
7. **[minor] `.gitignore` is a new file in the vendored repo; v0's Question 2 unanswered.**
   Decide explicitly whether it ships back to `lib/abiss` with the CMake change or the build
   trees move out-of-tree and the ignore file is dropped.

## Tests to Add

- Symlink case for the isolation guard (finding 3) — closes an argued-not-shown claim.
- Summary count line in the isolation test so pasted evidence matches claimed coverage.
- Exhaustive `.o` hashing across both build trees, plus `.rodata` comparison, to upgrade the
  codegen proof from representative to exhaustive.
- **Blocking for release, not for this stage:** the fidelity gate — unrelabelled exhaustive
  element equality over the complete affected output, ≥2 repeat runs. Still unrun.

## Questions

- Does `.gitignore` ship back to `lib/abiss`, or do build trees move out-of-tree?
- Should the `ABISS_REPO` boundary be tightened to `<RUN>/work/abiss` to match the described
  contract?

## Verdict

VERDICT: APPROVE_WITH_MINOR_COMMENTS

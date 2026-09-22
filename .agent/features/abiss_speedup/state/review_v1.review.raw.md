# Review v1

## Summary

Both `review_v0` majors are closed, and the first one is closed the right way — by construction rather than by argument. `-ffp-contract=off` now lives inside the `if(ABISS_ARCH)` branch and carries a `$<COMPILE_LANGUAGE:CXX>` guard, so the default `ABISS_ARCH=""` configuration cannot emit an FP directive at all; the coordinator's object-file comparison (live production `agg.dir/src/agg/mean_aggl.cpp.o` vs the code_v1 default build: identical disassembly and section layout, differing only in the embedded path string) independently confirms the default build's codegen matches the build that produced the measured baseline. That also retires the residual concern about the v0 `CMAKE_CXX_FLAGS_RELEASE` restructuring and the `-march` dedup, which review_v0 had accepted only as "inert."

The second major was already resolved by coordinator evidence and is now cited rather than asserted.

Everything still open is minor, and most of it is evidence presentation rather than code. Nothing in the CMake diff introduces a new problem that I can see.

## Status of each review_v0 finding

| v0 finding | Status |
|---|---|
| [major] `-ffp-contract=off` unconditional | **Fixed** — moved inside the non-empty-`ABISS_ARCH` branch, and separately proven inert by object-file comparison |
| [major→RESOLVED] allocator default `system` | **Cited**, not re-litigated. Correct handling |
| [minor] six Files Changed rows unreviewable | **Worked, not verified** — see finding 1 |
| [minor] Phase C approval not cited | **Fixed** — `run.md ## Approvals`, 2026-07-29, quoted |
| [minor] absl evidence in prose only | **Fixed** — `USE_ABSL_HASHMAP` / `USE_MIMALLOC` columns added, 34 rows |
| [minor] `-march` strip mutates scope not cache | **Documented, not fixed** — see finding 5 |
| [minor] `add_compile_options` language-agnostic | **Fixed** — generator expression present in the diff |
| [minor] build trees not ignored | **Fixed** — `.gitignore` added, `git check-ignore -v` shown, coordinator confirms they no longer appear untracked |

v0's Question 1 was answered by taking the preferred option. Question 2 (build trees in-tree vs out-of-tree) was answered implicitly by choosing `.gitignore`; see finding 7.

## Findings

1. **[minor] The isolation-harness changes are still outside the reviewable diff.** Six of eleven Files Changed rows are `state/profiling/*`, and the diff supplied here is again only `work/abiss`. The coordinator's independent checks cover CMakeLists, git state, the live checkout, and the ignore rules — not the profiling scripts. So the physical-path guards and the new test exist only as pasted stdout in this artifact. Not blocking: the property they protect (the live checkout is not touched) was verified directly by the coordinator this round. But the finding is worked, not verified, and it is now two rounds old.

2. **[minor] The pasted isolation transcript is narrower than the coverage claimed for it.** The text asserts seven ME resolutions, one WS resolution, *all 17 manifest binary resolutions*, and the profiling wrapper's `agg` resolution — 26 positive checks. The transcript shows eight `resolved_binary:` lines plus one `resolved_build_dir:`. On the negative side it claims nonzero exits for four cases (`ABISS_REPO`, `ABISS_BIN_PATH`, manifest capture, profiling override) and shows three diagnostics. Either the output was elided without saying so, or those assertions run silently. State which, or have the test emit a summary count line (`positive=26 negative=4`) that can be pasted whole.

3. **[minor] Symlink rejection is argued, not shown.** "Symlinks resolving to the live checkout therefore fail the physical-path checks" is a derivation from using physical-path resolution; the Review Focus asks me to confirm it, but neither the enumerated test cases nor the transcript contains a symlink. This is the same shape as the v0 major it just fixed. One case closes it: `ln -s <PROJECT>/lib/abiss $TMPDIR/fake_repo` and then `ABISS_REPO=$TMPDIR/fake_repo`, expecting the same refusal.

4. **[minor] One enforced boundary is looser than the described one.** The transcript shows `Refusing ABISS_REPO outside <RUN>` but `Refusing ABISS_BIN_PATH outside <RUN>/work/abiss`. The prose contract is "outside the CCC run **or isolated repo**"; as diagnosed, a repository path under `<RUN>` but outside `work/abiss` passes the repo check. Impact is small because the build-dir and binary checks are pinned to `<RUN>/work/abiss`, so the executed artifacts are still the audited ones. Either tighten the repo check to `<RUN>/work/abiss` or state the two-tier boundary deliberately.

5. **[minor] The cache/scope divergence is documented only in the stage artifact.** The warning is correct and well-demonstrated (`CMakeCache.txt` retains `-march=nocona -mtune=haswell`, `flags.make` shows the sanitized set), but a future audit reads `CMakeCache.txt`, not `code_v1.md`. One comment beside the regex rewrite in `CMakeLists.txt` — that the rewrite is directory-scope, that the cache is intentionally not updated, and that only `-march`/`-mtune` are stripped — puts the warning where the misled reader actually is.

6. **[minor] The object-file proof is single-TU, single-target, and text-section only.** It is the right test and it passes, but the flag transformation it validates (`CMAKE_CXX_FLAGS_RELEASE` no longer re-embedding `CMAKE_CXX_FLAGS`, and the resulting `-march` dedup) applies to all 17 targets. The transformation is uniform across the directory scope, so generalizing from one TU is defensible — but hashing every `.o` in the two trees is cheap and would make it exhaustive rather than representative. Separately, `objdump -d` compares executable sections; `readelf -S` bounds section *sizes* but does not compare `.rodata` contents, which is precisely where an FP-contraction difference would deposit folded constants. `objdump -s -j .rodata` on the same pair would close that. Not blocking given the flag is now absent by construction at the default.

7. **[minor] `.gitignore` is a new file in the vendored ABISS repo, and v0's Question 2 is still unanswered.** The coordinator sees `?? .gitignore` in `work/abiss`. review_v0 sanctioned "add to `.gitignore` **or** build out-of-tree," so this is a licensed choice, but decide explicitly whether a `.gitignore` for CCC-run build trees is intended to ship back to `lib/abiss` with the CMake change, or whether the build trees should move out-of-tree and the ignore file be dropped from the deliverable.

## New problems introduced

None found in the CMake diff. Specific things I checked and cleared: `add_compile_options` sits above the `add_subdirectory` calls so it reaches all 17 targets (corroborated by the 17/17 `build-arch-sanitize` count); the `-march`/`-mtune` regexes handle the duplicated-token case that live actually had (`build-arch-sanitize` reports one `-march`, zero `-mtune`); `find_package(PkgConfig REQUIRED)` no longer running in the `system` branch is safe since that build configured and linked 17/17; moving from `${MIMALLOC_LDFLAGS}` + global `include_directories` to `PkgConfig::MIMALLOC` narrows include propagation to linking targets only, which is fine because all 17 link it. The `jemalloc` branch is unexercised, but it mirrors pre-existing capability and production used neither allocator.

## Verification claims — accuracy

Accurate and, where incomplete, honestly labelled: "No cluster benchmark, full-volume replay, or segmentation comparison was run," and Risks correctly keeps segmentation bit-identity at **UNVERIFIED** with the fidelity gate named as blocking. "Addressed all findings" is fair, since the cache/scope item is described as *recorded* rather than fixed.

The one place stated coverage exceeds shown evidence is the isolation test (findings 2 and 3) — the transcript does not contain the 17 manifest resolutions, the profiling-wrapper `agg` resolution, or any symlink case, yet all are claimed or invited for confirmation. That is an evidence-presentation overclaim, not a false result.

Minor transcription note: the new `USE_ABSL_HASHMAP`/`USE_MIMALLOC` columns restate counts the coordinator confirmed at v0 (0/17, 0/17, 17/17), and v1 did not touch allocator logic, so the residual risk there is transcription only.

## Verdict

Both blocking majors are closed, one of them with independent measurement stronger than what was asked for. The remainder is presentation and durability of evidence, none of it affecting the build that will produce the segmentation. Land it, and fold findings 2–5 into the next artifact rather than a new code round. The unrelabelled exhaustive segmentation equality gate remains the release blocker and is untouched by this stage.

READY: yes

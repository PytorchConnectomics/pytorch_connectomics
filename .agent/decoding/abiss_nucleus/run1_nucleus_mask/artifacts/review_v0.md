# Review v0

## Summary

Reviewer: claude (planner, in-session). Raw notes: `state/review_v0.review.raw.md`.

The implementation is substantively correct. I read the full tracked diff against
`run_start_ref` file by file plus the untracked `src/seg/NucExtractor.hpp`, and independently
re-ran the load-bearing verification rather than accepting `code_v0.md`:

* **V2, the Seuron bit-identity gate — reproduced.** Built baseline binaries at `3c4f5621` in a
  throwaway `git worktree` with the same flags, ran the same fixture under both builds:
  **35 identical, 0 differing, 0 missing**, and exactly the three modified-only sidecars
  (`ongoing_nuc.data`, `done_nuc.data`, `nuc_cuts.data`) present and size 0. Matches the coder's
  reported numbers exactly.
* **V5/V6 — recompiled and re-ran both unit binaries.** `test_nuc_algebra: PASS` including the
  checked-overflow abort; `test_nuc_extractor: PASS` with `conflict_sv 1 / minority_sv 1 /
  subfloor_sv 1 / subfloor_voxels 10`.
* **V3/V4 — re-ran all five nucleus modes.** `none`/`same`/`one` merge; `different` and `max` are
  vetoed with `nuc_cuts` naming `(100, 200)`. The fixture's affinity is ~0.97, so the veto is
  demonstrably **not** affinity-gated, and `0xFFFFFFFF` behaves as an ordinary id.

The six review rounds paid off in the code: `nuc_is_dominant` uses exact `__uint128_t` arithmetic
(K1a), `nuc_add` is checked (K1b), `nuc_can_merge` lives in `Types.h` where the unit binary can see
it (K4), the sub-floor branch leaves `total == 0` (J1), the frozen-edge condition at
`mean_aggl.cpp:686-687` was correctly left alone, and the README carries Invariant D's three
clauses, the `99/1` counterexample, and Bound C's corrected scope.

Two majors and four minors below. **None is a correctness defect in the nucleus feature itself** —
the majors are a new default-path failure mode and a verification-reproducibility gap.

## Diff Baseline

run_start_ref: 3c4f56219488441edb4c51ee7ca4f07cdef2cd8f

Reviewed: `git diff 3c4f5621` (13 tracked files, 484 insertions, 4 deletions) plus the untracked
`src/seg/NucExtractor.hpp`. `HEAD` verified equal to `run_start_ref`; no commits were created.

## Findings

**F1 [major] — nucleus env vars are parsed on the default path.**
`src/seg/atomic_chunk_ME.cpp:104-107` evaluates `nuc_ratio_from_env()` and
`nuc_min_tagged_from_env()` when constructing the extractor, *outside* the
`std::filesystem::exists("nuc.raw")` guard. Both abort on a malformed value. So a stale or
mistyped `ABISS_NUC_DOMINANCE` left in the environment now kills `acme` on a run that has **no
nucleus input at all** — a failure mode that did not exist before this change. V2 cannot catch it
because V2 runs with a clean environment. Move both calls inside the `exists` branch, or default
them silently when no nucleus source is present.

**F2 [major] — no re-runnable V2 harness.**
`work/test/` contains `make_fixture.py`, `run_atomic_fixture.sh`, and the per-feature tests, but
nothing that builds the baseline at `run_start_ref` and performs the comparison. V2 is the gate
protecting Seuron provenance bit-identity; I could only confirm it by reconstructing the baseline
worktree, the build, and the file-by-file comparison by hand. That is precisely the step a future
change will skip. Add `work/test/run_v2_invariance.sh` that does the whole thing and prints the
identical/differing/missing counts.

**F3 [minor] — collision counters over-count.**
`load_nuc` (`mean_aggl.cpp`), `reduce_nuc` (`reduce_chunk.cpp`), and `process_nucs`
(`match_chunks.cpp`) each increment whenever the *result* of `nuc_join` is CONFLICT, so a third
record joining an already-CONFLICT accumulator increments again. These counters are the designated
observable for detecting a veto failure (plan_v6 B6 and Q2); inflating them makes that signal
harder to read. Increment only on the transition into CONFLICT.

**F4 [minor] — default-path log noise.**
`NucExtractor::output` unconditionally prints four `nuc:` counter lines, and `load_nuc` prints
`nuc: no nucleus labels`, on every chunk even with no nucleus input — roughly 600 extra lines on a
143-shard whole-volume run. Gate the prints on the extractor having a source.

**F5 [minor] — `load_nuc` aborts on a nucleus sid missing from `seg_indices`.**
This mirrors `load_sem` (`mean_aggl.cpp:255-266`), so it is consistent with existing behavior and
not a regression. But it turns a data/config mismatch into a hard abort mid-run. Worth confirming
that the ongoing/done partition keeps the two sets aligned in the **composite** path, where
`seg_indices` is built from `ongoing_supervoxel_counts.data` rather than `ns.data`.

**F6 [minor] — mixed-binary pipelines break at the merge.**
`merge_chunks_me.py:60` now merges `ongoing_nuclei_labels` unconditionally, so a run whose earlier
chunks were produced by a pre-change binary has no such child files and the merge fails. Out of
scope to fix; it belongs in the README as an upgrade note.

## Tests to Add

1. **`work/test/run_v2_invariance.sh`** (addresses F2) — baseline worktree at `run_start_ref`,
   matching cmake flags, fixture under both builds, comparison over exactly the baseline-produced
   file set, plus the empty-sidecar assertion. This is the single most valuable missing artifact.
2. **A default-path environment test** (addresses F1) — run `acme` with no `nuc.raw` and
   `ABISS_NUC_DOMINANCE=garbage` set; it must succeed, not abort.
3. **A counter-transition test** (addresses F3) — three records for one sid, two of which already
   force CONFLICT, asserting the collision counter reports 1 rather than 2.
4. **A composite-path `load_nuc` test** (addresses F5) — exercise `agg` at the composite level with
   `ongoing_supervoxel_counts.data` as the ns file and a nucleus payload, confirming no abort.

## Questions

1. F1 and F4 are both "the feature changes behavior when it is switched off". Is there any other
   place the nucleus path is reachable with no `NUC_PATH` configured? The three readers all use
   `filesize() == 0` and degrade correctly, which is right, but `acme`'s env parsing shows the
   pattern is not applied uniformly.
2. F5's composite-path question is the one thing I could not settle by reading. Does the
   ongoing/done partition guarantee every sid in the merged `ongoing_nuclei_labels.data` appears in
   the merged `ongoing_supervoxel_counts.data`?

## Verdict

VERDICT: NEEDS_CHANGES

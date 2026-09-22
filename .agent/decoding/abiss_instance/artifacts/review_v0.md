# Review v0

## Summary

`code_v0` implements `plan_v3` competently and reports on itself honestly. Every mechanism the
plan required is present in the code, the 18 unit tests pass (independently reproduced, not taken
on trust), `HEAD` never moved, and Codex explicitly declined to claim the efficiency gate it could
not run. Its `## Risks and Unknowns` volunteers five real limitations including two unresolved
ship gates. That is the behaviour you want from an implementation stage.

It is nevertheless **NEEDS_CHANGES**, for one structural reason rather than any defect in the code
itself: **the change invalidates its own regression oracle.** `plan_v3` gates 2 and 3 make
correctness depend on comparing against two completed reference runs, but those runs' manifests are
schema **1.2** (native96) and **1.0** (win144), and the new overlay rejects pre-2.0 manifests by
design. So the plan's primary correctness gate has no executable procedure today, and reproducing
the "reference" side is not a simple checkout because `nucleus_overlay.py` was modified in place
and `lib/abiss` also carries unrelated 2026-08-16 modifications.

A second item needs an explicit decision rather than silence: `connectomics/runtime/abiss_chunk.py`
was modified, and `plan_v3` scoped that repair as a **written recommendation**, with watershed
reuse listed out of scope. The two-line change is exactly what §D.3 recommended and Codex disclosed
it in both `## Files Changed` and `## Risks and Unknowns` — but it sits outside the declared review
surface, which is precisely the case the scoped-diff decision was meant to surface rather than
absorb.

Scale for context: `nucleus_competition.py` grew 747 → 1441 lines and the test file 442 → 796. The
only correctness evidence available today is those 18 unit tests, because gates 2–8 all require
cluster runs.

## Diff Baseline

run_start_ref: c705458ae5b907bb9c32a75c63c85c6aad7edec7

Note on the surface: `lib/abiss` is a nested git repository, `dev/` is gitignored
(`.gitignore:159`), and two of the touched files are untracked in the parent repo, so only two of
the declared files produce a git diff. Attribution was therefore established by mtime and recorded
in `state/review_v0.review.raw.md` and in `run.md` § Diff-Surface Correction. Codex touched only
the two declared files inside `lib/abiss`; the nine other modifications there date to 2026-08-16
and belong to the earlier codex thread, not this run.

## Findings

- **[major] F1 — The regression oracle is invalidated by the schema change.** Gates 2 and 3 name
  `wholevol_arm0_native96_nuc_matchguard` and `wholevol_arm096_nuc_competitive_v2` as the
  comparison references. Their manifests are schema 1.2 and 1.0; the new overlay requires a
  completion marker and `plan_digest` that neither has. `code_v1` must supply an executable
  procedure — either a documented read-only compatibility path for pre-2.0 manifests used solely
  by the comparison harness, or a written-out method for reconstructing the pre-change overlay
  (noting that `git checkout` alone will not do it, because `lib/abiss` carries unrelated
  2026-08-16 edits). Without one, the change cannot be shown to preserve behaviour.
- **[major] F2 — Undeclared production change to `connectomics/runtime/abiss_chunk.py`.**
  `plan_v3` §D.3 scoped the `CHUNKMAP_INPUT` repair as a written recommendation and §Scope put
  watershed reuse out of scope; `code_v0` implemented it at lines 643/660. The change is correct
  and matches the recommendation exactly, and it was disclosed. The decision needed is explicit:
  either accept it and amend the scope record, or revert it to `code_v1` and keep the increment's
  boundary honest. This review does not silently absorb it.
- **[major] F3 — Correctness rests on 18 unit tests for a 694-line growth in the core file.** Not a
  defect, but a constraint on what the run may claim. Gates 2–8 are all cluster-bound and unrun, so
  "behaviour-preserving" is currently an intention, not a measurement. The ship verdict in
  `docs/nucleus_competition_review.md` must not read as though the refactor has been validated.
- **[minor] F4 — Two undeclared launcher scripts**, `sbatch_nuccomp_scan.sh` and
  `sbatch_nuccomp_merge.sh`. A natural consequence of the declared scan/flood/merge split, and
  disclosed; record them in the file list rather than treat them as a deviation.
- **[minor] F5 — `ccc-validate.sh` was not run by the coder** because the coordinator's prompt did
  not include the run folder. That is the coordinator's omission, not the coder's; the coordinator
  validated instead and the run is green.
- **[minor] F6 — `NUC_MAX_UNITS = 64` is unconfirmed policy**, ~7x the observed 8–9 units. Codex
  asked for a ruling; it needs one before the array runs in production.

Verified positively, and worth recording so `code_v1` does not redo them: all claimed mechanisms
are present (`internal_territory_id`, `emitted_id`, `plan_digest`, `NUC_MAX_UNITS`,
`stage_report.partial`, separated `scan_geometry`/`map_to_watershed`, `separation_claim`,
`zero_repairs`, `os.replace` for atomic publication); the overlay validates completion and plan
digest and consumes the explicit translation rather than re-deriving "largest"; and the
`CHUNKMAP_INPUT` fix is exactly the recommended one.

## Tests to Add

1. **A miniature gate-2 oracle that is not cluster-bound.** Build a fixture from a real completed
   run's territories and assert the emitted label array matches a stored expectation. Today the
   only end-to-end evidence lives behind a 7–11 h cluster run, which means a regression could sit
   undetected for a day.
2. **An explicit pre-2.0 manifest test.** Whatever F1's resolution, pin it: either assert the
   overlay rejects schema 1.0/1.2 with a clear, actionable error naming the required re-publication,
   or assert the read-only compatibility path reproduces the legacy emitted labels.
3. **A capacity test for `NUC_MAX_UNITS`** — assert scan fails closed above capacity rather than
   truncating, which `plan_v3` §B.3 requires and which becomes reachable only under the array.

## Questions

- **Q-1.** F2: keep the `abiss_chunk.py` change and amend the scope record, or revert it in
  `code_v1`? The coordinator's recommendation is to keep it — it is two lines, correct, disclosed,
  and removes a documented 126 MB workaround — but the scope record must then say so plainly.
- **Q-2.** F1: which resolution — a read-only legacy compatibility path used only by the comparison
  harness, or a documented reconstruction of the pre-change overlay? The first is more work but
  leaves a reusable regression oracle; the second is cheaper and one-shot.
- **Q-3.** F6: confirm `NUC_MAX_UNITS = 64`, or set a policy tied to the observed unit count.

## Verdict

VERDICT: NEEDS_CHANGES

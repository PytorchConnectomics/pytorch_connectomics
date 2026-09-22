# Review v0

## Summary
code_v0 is a strong, faithful implementation of plan_v3. The constrained union-find asserts both
axon SEPARATION and per-piece/terminal CONNECTIVITY (the central invariant), eligibility is
correctly upstream-only, background-0 is excluded, modes/slice-bounds are validated, and
`force_split` handles every stop cause with atomic ownership. `--self-test` PASSES (re-run) and the
pilot ran clean (47 fusions, 9 eligible, 87 sections split). One MATERIAL issue: a hard gate that
will likely abort the full-volume verification. Raw notes: `state/review_v0.review.raw.md`.

## Diff Baseline
run_start_ref: e8844b3da0f0a992431c901e7e2034486e7a678b

## Findings
- [major] `decode_v2.py` line 501-503 hard-asserts `substrate oracle-merge NERL >= 0.78`
  (`SUBSTRATE_ORACLE_FLOOR`). That 0.78 was measured for force-split-ALL (534 runs / 9421 sections
  split — splits every fusion incl. harmless same-axon over-splits). decode_v2 splits ELIGIBLE
  fusions ONLY (~9 runs / 87 sections on z0:96 → ~75 runs full-volume, far fewer), so its unlinked
  substrate splits much less and its oracle-merge NERL sits near the unsplit-sections baseline
  0.760 — very likely `< 0.78`. The full run (the primary verification) would then ABORT at
  line 501 before emitting NERL. Root cause: plan_v3 pinned the floor to the wrong (more
  aggressive) baseline's value. FIX (one line): set the hard floor to `>= 0.760` (the unsplit-
  sections baseline: the eligible substrate must not regress versus no split) and print the exact
  substrate value; keep 0.78 only as a reported target, not a hard abort.
- [minor] On z0:96 variants a/b/c are identical (F1 0.9133) because the block has few merges; the
  constrained relink's value only appears full-volume. Not a defect — note it so reviewers don't
  read the pilot equality as "constraint has no effect".
- [minor] The `selected` output uses GT-based argmax(A,C) — correctly flagged EVAL-ONLY in the
  code; acceptable for this project but not deployable (a GT-free selector remains future work).

## Tests to Add
- None critical: `--self-test` already covers UF direct/indirect/cross, background-0, upstream
  eligibility (incl. downstream-convergence non-effect), terminal bijection + terminal-as-run-head
  conflict, stale-seed conflict, and end-to-end connectivity. The remaining validation is the
  full-volume run itself (which the ≥0.78 gate currently blocks); it is the real pass/fail and
  should run after the fix.

## Questions
- Confirm the substrate floor should be `>= 0.760` (no-regression vs unsplit sections) rather than
  an absolute 0.78; the exact eligible-substrate value is only known after the full run, so a hard
  0.78 is unsafe regardless of whether it happens to pass.

## Verdict
VERDICT: NEEDS_CHANGES

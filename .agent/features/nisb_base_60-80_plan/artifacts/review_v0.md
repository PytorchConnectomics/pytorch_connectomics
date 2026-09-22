# Review v0

## Summary

Coordinator (claude/planner) in-session review of `code_v0`. The implementation is faithful
to `plan_v3` + the addendum, the artifact contract is satisfied, and the blocking sanities are
exact (cc0.66=0.601431470, cc0.75=0.545357139, BASE_NERL=0.835517, transform 100%). G0 (the
ratified 1-voxel oracle) is correct — edges-only union-find, foreign-corridor reject, 0 cross-GT
/ 0 false merges on the smoke. **One [major] correctness bug** in G1's Hermite curvature gate
must be fixed before the full G1 run, because it collapses generator recall to a false negative
(and is the true cause of the smoke's 0/9 curvature passes). Mutation guard clean: HEAD unchanged,
no tracked-file changes. Raw notes: `state/review_v0.review.raw.md`.

## Diff Baseline

run_start_ref: e8844b3da0f0a992431c901e7e2034486e7a678b

Review surface = the untracked deliverables under `dev/nisb/scripts/mesa/` + this run folder
(read in-session); no tracked content changed.

## Findings

- [major] **G1 Hermite curvature end-derivative sign is wrong** (`g1_banis_feasibility.py`,
  `hermite_min_curvature_radius`, ~line 405/420–421). `handle_v = chord * tangent_v` uses the
  OUTWARD tip tangent, but a smooth pass-through arc arrives at `v` travelling into `v`'s body,
  i.e. along `−tangent_v` (the agreement term already uses `−dir` for `v` for this exact reason).
  With `+tangent_v`, even ideal collinear continuations get a hairpin at `s=1`, so `ρ_min` is tiny
  and the gate `ρ_min ≥ 1.25·r_max` rejects nearly all candidates → generator recall ≈ 0 → G1 would
  falsely fail at the generator. This is the cause of the 256³ smoke's 9-evaluated/0-accepted, not a
  crop-size artifact. **Fix:** `handle_v = -chord * tangent_v` (or negate `tangent_v` before the
  Hermite); keep `handle_u = +chord * tangent_u`. (Root: the coordinator addendum A said "m_v =
  L·t_v" — that wording is also wrong; Codex implemented it faithfully. The addendum is corrected to
  `m_v = −L·t_v`.)
- [minor] GT-SDT ero1 uses `(1,1,0)` in-plane erosion (Z untouched) for the anisotropic `(X,Y,Z)`
  layout — plausible but unverified against the `x2diag/gt_sdt_region_ero1.h5` convention; the real
  check is C0 grow-NERL ≥0.990 on the full run.

## Tests to Add

- After the fix, re-run the 256³ G1 smoke and confirm a nonzero fraction of the curvature-evaluated
  candidates now pass (well-aligned continuations → near-straight Hermite → large `ρ_min`). If a
  clearly-collinear tip pair still fails, escalate.
- (Full run, coordinator) C0 GT-SDT grow local NERL ≥0.990 validates the ero1 `(1,1,0)` choice
  end-to-end.

## Questions

- None blocking. The fix is a 1-line sign correction the coder should apply and smoke-verify.

## Verdict

VERDICT: NEEDS_CHANGES

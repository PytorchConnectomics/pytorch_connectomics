# Plan v1 Review

## Summary

Reviewer: Codex (gpt-5.6-sol, ultra), read-only. Attested summary of
`state/plan_v1_review.review.raw.md`. Codex confirms the plan_v0 majors are
genuinely resolved (substrate reuse, whole-volume acceptance/crop,
endpoint-oriented candidate generation, transitive marker-set logic, streamed
eval) and that Step 1 is a reasonable bounded increment. It returns `READY: no`
on six remaining execution-contract gaps (all about making code_v0 deterministic
and its NERL number trustworthy). These are addressable in plan_v2 (the final
plan round).

## Findings

- [major] **Global-ID namespace underspecified.** The linker sees only
  face-crossing labels; the evaluator sees every foreground label. Define a
  *total*, collision-free `(chunk_key, local_label) → global_id` mapping
  (including untouched labels), with ID allocation and dtype, and prove the remap
  is applied exactly once before both linked base NERL and `branch_merge`.
- [major] **No-edge remap control missing.** Running the remap path with zero
  committed unions must reproduce the existing no-remap base/oracle/missing/
  n_skel/GT-length within roundoff — this validates the global namespacing and the
  modified scorer independently of any link (stronger than the 2–3 chunk check).
- [major] **Marker contract not executable.** Fix one source definitively (stop
  reopening `_neuron`); give the exact coordinate mapping (axis permutation,
  origin, coarse-voxel convention, boundary, marker→segment assignment); establish
  the values are distinct marker IDs, not binary seeds; and put marker sampling in
  one concrete stage (not split between extractor and linker `--markers`).
- [major] **Quarantine contradicts the assertion.** A pre-existing multi-marker
  segment stays multi-marker, so "no global_id holds ≥2 markers" cannot pass.
  Re-state the invariant: quarantined roots accept no edges and are unchanged, and
  no *accepted union* creates or absorbs a multi-marker component.
- [major] **Operating point not reproducible.** Weights, normalization, ambiguity,
  `tau_score`, and the selection protocol are undefined while tuning is deferred to
  Step 5 — code_v0 would invent parameters. Predeclare formulas + constants or a
  bounded calibration protocol. `tau_iou_min` must be exactly 0 or removed
  ("approximately zero" reinstates an overlap gate).
- [major] **Oracle gate internally inconsistent.** `|Δ| ≤ 0.002` permits a
  negative change that "any drop ⇒ fail" rejects, and 0.002 can hide a real merge.
  Use one justified numerical-roundoff tolerance: substantive negative fails; a
  positive change (impossible for a pure join) also fails pending investigation.
- [minor] Oracle-flat detects only merges among distinct sampled test-50 owners;
  markerless/off-GT merges are invisible — do not call it global merge safety.
- [minor] Validate all 600 symlink targets and the regenerated crossing schema;
  record *current root* marker sets (not just original endpoint sets) in
  certificates so transitive firewall decisions stay auditable.

Codex explicitly affirms the polarity-aware tangent rule and the dynamic per-root
marker-set union (including `A→unmarked→B` rejection) are sound.

## Questions

- Marker source resolved to `yl_cb_80nm.h5` (verified: labeled uint16, distinct
  soma IDs 1..827, 518 in-volume) with the coordinate mapping reused verbatim from
  `soma_recon_wholevol.py` (`POOL=[8,16,16]`, `CBSCALE=[4,8,8]`,
  `caff=(g*CBSCALE)//POOL`) — plan_v2 will pin this.

## Verdict

VERDICT: NEEDS_CHANGES

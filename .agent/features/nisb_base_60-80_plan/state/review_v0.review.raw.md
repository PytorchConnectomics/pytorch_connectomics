# review_v0 — coordinator (claude/planner) in-session review notes

## Method
- Mutation guard: HEAD == run_start_ref e8844b3d; tracked unstaged diff byte-identical to
  run_start (22104 B); staged 0; no tracked-file changes since run start. Deliverables untracked. OK.
- Verified sanities by OUTPUT (from code_v0.md, re-reading the numbers): Sanity A cc0.66
  0.601431470 (==0.601431), cc0.75 0.545357139 (==0.545); Sanity B BASE_NERL 0.835517 (0.836±0.003);
  transform 100% in-seg. All pass exactly.
- Read the code: g0 build_offset_graph/_union_roots, g1 tips/generate_candidates/
  hermite_min_curvature_radius/cap_candidates.

## G0 — clean
- `build_offset_graph`: edge = `voxel_skeleton_id[source]==voxel_skeleton_id[target]` (same-instance
  gated); identity used ONLY to construct edges; union-find over edges (`_union_roots`); cross-GT
  union assertion (0); foreign-corridor rejection applied; markers = components painted at skeleton
  voxels; grow via watershed on GT-SDT elsewhere. This is the ratified 1-voxel ancestor oracle,
  correct. Smoke: 0 cross-GT unions, 0 false merges, retained 99.8%, banks 13/16/24/62. Good.

## G1 — one MAJOR correctness bug (Hermite handle sign) + notes
- `radius_estimate`, tangent-agreement `0.5(⟨t_u,dir⟩+⟨t_v,−dir⟩)≥0.65`, distance limit
  `min(750,max(8r_max,60,108))`, radius-ratio ≤3, top-8-at-either-endpoint cap, degree-1 matching,
  break-cluster events, ΔNERL=2LaLb/D + validation, realizer control/repaired — all faithful to
  plan_v3 + addendum.
- **[MAJOR] Hermite curvature end-derivative sign is wrong.** `hermite_min_curvature_radius`
  (g1:405) sets `handle_v = chord * tangent_v` with `tangent_v` = the OUTWARD tip tangent (from
  `extract_tips`: `tangent = tip − nearest.mean`, points out of the fragment body toward the gap).
  For accepted candidates the agreement term forces `t_v ≈ −dir` (v's outward points back toward u).
  A smooth pass-through arc must ARRIVE at v travelling into v's body = `+dir ≈ −t_v`; so the Hermite
  end-derivative should be `handle_v = −chord * tangent_v`, not `+chord * tangent_v`. With the current
  `+t_v`, even an ideal collinear continuation gets a hairpin/cusp (derivative at s=1 points backward),
  so `ρ_min` is tiny and the gate `ρ_min ≥ 1.25·r_max` rejects essentially every candidate. This is
  the cause of the 256³ smoke's "9 curvature-evaluated, 0 accepted" — NOT a crop-size artifact. On the
  full chunk it would collapse generator recall to ~0 and make G1 a FALSE-NEGATIVE at the generator
  (violating the loop's branch-killing-negatives caution). Note: the exactly-degenerate collinear case
  stays 1-D (cross≈0 → ρ=inf) and passes, which is why it doesn't crash; realistic noisy tips do not.
  Fix: `handle_v = -chord * tangent_v` (equivalently negate `tangent_v` before the Hermite), consistent
  with the agreement formula's `−dir` convention for v. `handle_u = +chord * tangent_u` stays.
  NOTE: this originates in the coordinator's addendum A ("m_v = L·t_v"); the addendum wording is also
  wrong and should read `m_v = −L·t_v`. Codex implemented the addendum faithfully.
- Verify-after-fix: re-run the 256³ G1 smoke; some of the 9 curvature-evaluated candidates must now
  pass (well-aligned continuations → near-straight Hermite → ρ_min large). If still 0/9 with clearly
  collinear pairs present, escalate.

## Other (non-blocking)
- GT-SDT ero1 uses `(1,1,0)` erosion (in-plane only, Z untouched) for the (X,Y,Z) anisotropic layout
  — plausible; confirm grow-NERL ≥0.990 on the full C0 run (the actual gate).
- fragments/GT smoke 2.92 and NERL 0.489 are coverage-limited 256³ diagnostics, not thresholds — OK.

## Verdict
NEEDS_CHANGES — fix the Hermite handle sign (1-line) + re-verify the smoke; G0 and all sanities are
otherwise correct and the artifact contract is satisfied.

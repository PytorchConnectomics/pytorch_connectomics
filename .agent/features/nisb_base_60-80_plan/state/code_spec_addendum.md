# Code-stage spec addendum — pinned resolutions of the 4 plan_v3_review items

Authoritative alongside `artifacts/plan_v3.md`. These close the 4 remaining [major]
findings in `plan_v3_review.md`; the coder MUST implement them and the code reviewer MUST
verify them.

## A. G1 §1 — proposal geometry (exact equations)

- **Physical units / anisotropy.** All distances, EDTs, tangents, curvatures in physical nm
  with `sampling=[9,9,20]` (X,Y,Z). Boundary EDT of a fragment mask =
  `scipy.ndimage.distance_transform_edt(mask, sampling=[9,9,20])`.
- **Radius estimator.** `r_est(tip)` = median boundary-EDT over the K=12 fragment voxels
  nearest (physical) to the tip. `r_max = max(r_est_u, r_est_v)`.
- **Tangent agreement.** Tips carry unit outward tangents `t_u, t_v` (physical, from
  `_extract_tips_fast`). Let `d = (p_v − p_u)/‖p_v − p_u‖` (physical). Agreement score
  `A = 0.5·(⟨t_u, d⟩ + ⟨t_v, −d⟩)` ∈ [−1,1] (both tips point toward each other). Gate:
  proposal `A ≥ 0.65`; acceptance `A ≥ 0.80`.
- **Cubic Hermite + curvature.** Chord `L = ‖p_v − p_u‖`; handles `m_u = +L·t_u`,
  `m_v = -L·t_v` (v's outward tangent is negated to the travel/into-body direction,
  consistent with the agreement term's `-dir` convention for v — CORRECTED in review_v0);
  standard cubic Hermite `H(s), s∈[0,1]`. Sample N=32 points; at each,
  finite-difference `H', H''`; curvature radius `ρ(s) = ‖H'‖³ / ‖H' × H''‖` (∞ where the
  cross product ≈0). `ρ_min = min_s ρ(s)`. Gate: proposal `ρ_min ≥ 1.25·r_max`; acceptance
  `ρ_min ≥ 1.5·r_max`.
- **Radius ratio** `max(r_est_u,r_est_v)/min(...)`: proposal ≤3.0; acceptance ≤2.5.

## B. G1 §4 — realizer must test REPAIRED realization

Run the gated realizer from the **capped-oracle repaired** seed partition (fragments merged
per capped-covered same-GT events), not only the per-fragment control.
- **label-0 into the graph.** cc3d the unlabeled (`BASE_SEG==0`) mask into pseudo-fragments;
  add them as adjacency-graph nodes with no identity; they are the orphans to adopt.
- **Grow / cost / margin.** Fragment-adjacency graph; edge cost `-log(mean face aff_r1 + 1e-6)`
  (channel = the shared-face axis; reverse step = same undirected face). Assign each orphan to
  the min-cost identity; **abstain (label 0/unknown)** unless the runner-up margin ≥ `log 9`.
- **Gates (repaired run):** dense-GT **adoption precision ≥ 0.95** (adopted orphan voxels whose
  GT id matches the assigned identity's GT id, over adopted voxels with nonzero GT);
  **coverage** = adopted / eligible-orphan voxels, must be **> 0** and reported (target ≥0.5);
  **abstention rate** reported; **0 cross-GT** identities; grown NERL ≥ (pre-grow repaired
  relabel NERL − 0.005).
- **Control (per-fragment, no repair)** still run: grown NERL within 0.005 of `BASE_NERL`,
  0 cross-GT, 0 multi-ID — proves grow does not by itself corrupt identity.

## C. G1 §2 — matching is degree-≤1 per fragment (non-cascading)

The geometric matching (plan_v3 G1 §2) enforces **degree ≤ 1 per `BASE_SEG` fragment** (a
fragment accepts at most one incident accepted link), not per tip — so a 2-tip fragment cannot
form a transitive chain. Single round only; no accepted fragment is re-proposed. Deterministic
mutual-best ranking: candidate score = `A` (tangent agreement) desc, then distance asc, then
`(min(f_a,f_b),max(f_a,f_b))` asc; accept a link iff it is the mutual top choice of both its
fragments under the acceptance gates and both fragments are still free.

## D. DESIGN.md — pin the channel GT targets

- **`d_b` sign convention:** `d_b(v) = +log(1 + d_in(v)/s_xy)` for `v ∈ G` (interior positive),
  `−log(1 + d_out(v)/s_xy)` for `v ∉ G`; `s_xy = 9 nm`; `d_in/d_out` = physical EDT to `∂G`.
- **Medialness cutoff:** exact — `m*(v) = 0` where `‖x_v − x_π(v)‖ > 0.6·r_i(π_i(v))` (not "~").
- **Tangent masking:** the tangent loss is masked wherever `q_branch = 1` OR `q_multi = 1`.
- **Veto neighborhoods:** `q_multi`, `q_foreign`, `q_branch` are computed in a physical ball of
  radius `R_v = max(3·s_xy, 1.5·r_i)` about each voxel (definitions per plan_v3 table).
- **Sparse skip target:** SAME/MUTEX/DEFER by the same local-path rule as short edges but over
  the `|o|∞=2` half-shell, **eligible only where no accepted short-edge path already connects
  the endpoints**; DEFER where the true skeleton path leaves the crop or is ambiguous.

## Execution-model boundary (coordinator directive)

The code stage (Codex) **writes** `common.py`, `g0_phase0_oracle.py`, `g1_banis_feasibility.py`,
`DESIGN.md` and **runs only the fast 256³ smoke** (target < 5 min) to prove they execute and the
harness sanities (A/B/transform) pass. It must **NOT** run the full center-chunk gates inside
`codex exec` (450 M-voxel watersheds risk timeouts). The full center-chunk G0/G1 run is executed
by the coordinator in the background after code review, with results appended to
`results_g0.md`/`results_g1.md` and folded into the go/no-go. `code_v0.md` documents the smoke
numbers + the exact full-run commands.

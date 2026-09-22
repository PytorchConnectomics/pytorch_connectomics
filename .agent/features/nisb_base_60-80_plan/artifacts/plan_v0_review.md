# Plan v0 Review

## Summary

Codex (coder) reviewed `plan_v0.md` for executability and faithfulness. Verdict:
**READY: no** — 11 [major] + 2 [minor] findings. The design intent, scope, and gate
anchors are judged correct, but the plan leaves too many *reproducibility* choices to
implementation time for a cheap gate whose whole value is a trustworthy pass/fail, and
it contains three concrete defects: a missing scorer-sanity calibration (`0.75→0.545`),
a stray predicted-SDT `0.985` kill criterion in a G0 that only runs GT-SDT, and a G1
fallback (GT-derived fragments) that is not a valid substitute for the banis+ feasibility
test. Full reviewer output: `state/plan_v0_review.review.raw.md`.

## Findings

Faithful summary of the raw transcript (no material finding softened):

- [major] Step 0 sanity is incomplete: add `cc3d@0.75 → 0.545`; give tolerances for the
  `0.601431` / `0.833` checks; treat `0.772069` as supplemental unless tied to a named cache.
- [major] G0 has unresolved implementation choices (erlgraph vs kimimaro skeleton;
  watershed vs geodesic grow; medial-band construction; edge ignore/unknown states;
  corridor rasterization; coordinate/anisotropy order; exact cumulative offset banks with
  half-space + dedup rules). Pick one canonical pipeline and enumerate offsets.
- [major] G0 skip rasterization can silently unite different components where paths cross;
  union-find components must stay authoritative, rasterized voxels inherit component IDs,
  collisions rejected/unknown; MUTEX checked during union; spatial CC may not invent unions.
- [major] G0 perturbation protocol (95%/90% gap-closure) is not reproducible: define
  eligible events, trial counts, RNG seed, boundary exclusion, per-object vs global section
  deletion, crumb-distribution source, closure criterion, denominators, and per-condition pass.
- [major] G0 verification carries a `0.985` predicted-SDT kill threshold, but predicted SDT
  is out of scope — remove it from this gate (or add + report the predicted-SDT run).
- [major] G1 must bind ONE exact starting label map + LUT: resolve cc3d@0.66 fragments vs
  the 0.627 intersection-cut firewall before building bridge events/endpoints/gains/realizer.
- [major] G1 bridge-set source conflict: task names `ec_endpoint_bridge.py` (local endpoint/
  bridge events), plan derives from `banis+_oracle_merge.py` (identities + exact node-LUT
  gains). Specify each one's role, endpoint neighborhoods, boundary exclusion, pair/hyperedge
  handling, dedup matching, and how each marginal ΔNERL is computed.
- [major] G1 proposal caps: hard-capping at 8 makes the ≤8 success trivial and the >12 kill
  unreachable. Report post-pruning counts BEFORE capping, then recall after a deterministic
  ranked top-8 cap (define ranking + tie-break).
- [major] G1 realizer ("aff_r1 threshold", "reproduces base within the harness") is not an
  executable gate and can pass trivially (singletons / unchanged labels). Specify channels,
  threshold sweep/freeze, supervoxel algorithm, geodesic cost, seeds, competition/abstention,
  min coverage; run a base-preservation control AND candidate-oracle-repaired seeds; require
  grown NERL within 0.005 of seed NERL, 0 seed-ID unions, 0 multi-ID assignments.
- [major] G1 missing-affinity risk: GT-derived fragments are NOT a valid banis+-feasibility
  substitute. Resolve an exact cached affinity path or an inference config/checkpoint (crop
  axes, halo, key, channel order). If neither exists, G1 is blocked/no-go, not a surrogate pass.
- [major] Runtime guard/verdict: smoke ≠ completed gate; a backgrounded run must be monitored
  to successful exit + validated output before writing final results. Define the gray-zone
  outcome (train only if every success criterion passes; gray zone = no-go + cheap refinement)
  and always emit an overall result location even if G0 failure blocks `results_g1.md`.
- [major] DESIGN.md: the "43 dense channels" claim must be an indexed accounting that sums to
  43 (every target, edge class/ignore mask, veto target, sparse-skip representation defined).
- [minor] "Extend" existing scripts conflicts with their read-only status — state that new
  `mesa/` scripts import/adapt their logic without modifying tracked files; specify banis+ load/invoke.
- [minor] Do not add a second harder chunk in this run (task scopes to the center chunk);
  recommend it as a later validation gate only.

## Questions

- Resolved by the reviewer: the second-chunk open question in plan_v0 should be answered "no,
  not this run" — keep execution on the standard center chunk, recommend a harder chunk later.
- For G1: is a cached banis+ affinity crop for the center chunk actually present (path), or must
  a small inference be run? plan_v1 must resolve this to a concrete path or config, not a surrogate.

## Verdict

VERDICT: NEEDS_CHANGES

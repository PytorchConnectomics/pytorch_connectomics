# Plan v1 Review

## Summary

Reviewer: codex (coder role), read-only sandbox, `codex exec --sandbox read-only`.
Raw transcript: `state/plan_v1_review.review.raw.md`.

The reviewer returned `READY: no` with seven `[major]` findings. It confirmed that plan v1
genuinely resolves v0 findings 1, 2, 5 and 7 — the GT-informed candidate scope is gone, the
full-population pass replaces it, Phase 3 is properly out of scope, and the Stage 2 firewall
ambiguity is removed — and that the uncertainty-calculation part of v0 finding 6 is resolved. It
judged v0 findings 3, 4 and the corridor part of 6 only partially resolved, and raised new
objections about execution ordering, multi-hop feasibility, and evaluator ceiling composition.

Two findings identify concrete errors rather than gaps, and the coordinator confirms both: the v1
caliber formula `voxels · 1620 nm³ / centerline_nm` has units of `nm²`, so the `caliber_nm > 1500`
quarantine rule was dimensionally invalid; and `endpoint_frac <= 0.5` was tautological given v1's
own declared `0–0.5` range.

The coordinator accepts all seven findings. None is softened or discarded.

## Findings

Verbatim severity tags preserved from the raw transcript.

1. `[major]` **Execution order still violates the GT firewall.** `stage0_reproduce.py` reads the
   test skeletons before Stage 2 freezes assignments, and verification step 7 explicitly runs it
   "before any policy is resolved", while the task permits evaluator access only after the
   manifests are immutable. Separately, Stage 3 refers to an assignment SHA in a "pre-evaluation
   report", but the declared sibling JSON records only `proposal_sha256` — the artifact that
   freezes and attests the assignment bytes is undefined.
2. `[major]` **Selector behaviour still requires invented rules.** `contact_area` has no acceptance
   gate or clear-winner definition; floor-only policies do not say how exact ties satisfy the
   clear-winner requirement; singletons have no defined runner-up or margin; mixed
   clean/quarantined candidate sets do not say whether quarantined anchors participate in ranking;
   the `anchor_id` sort direction is unstated; and the `3e5` anchor-floor variant is not encoded in
   output names, so its artifacts would collide with the primary variant.
3. `[major]` **Morphology and quarantine numbers are internally inconsistent.**
   `voxels × 1620 nm³ / centerline_nm` has units of `nm²`, so `caliber_nm > 1500` is dimensionally
   invalid and could quarantine nearly every anchor. `endpoint_frac <= 0.5` is tautological.
   Principal-axis construction is unspecified, and `tangent_cos >= 0` is not invariant to the
   arbitrary sign of an unoriented principal axis.
4. `[major]` **The verification truth table is still insufficiently source-indexed.** "Shifted by
   `-e`" does not say whether output index `v` reads input `v-e` or `v+e`, nor state the valid
   slicing domain and boundary treatment. The plan also does not test agreement with the required
   canonical affinity helper.
5. `[major]` **Multi-hop is neither executable nor firewall-safe.** Stage 1 records only
   fragment-anchor edges while complete-graph depth-2 resolution needs fragment-fragment adjacency
   and its evidence; Stage 0 retains only RAG degree. The "precision gate" has no numeric
   definition, and triggering a new Stage 2 run from Stage 3's GT-measured precision would feed
   test evaluation back into selection.
6. `[major]` **Residual and escalation contracts remain incomplete.** An edge table containing only
   actual fragment-anchor edges cannot enumerate zero-contact fragments while Stage 2 reads "the
   candidate table only". With missing evidence first in precedence, `no_anchor_contact` may be
   unreachable. Stage 4 depends on endpoints no prior stage constructs, gives no tangent/endpoint
   geometry gate, and does not define what fraction of the material no-contact bucket must pass the
   corridor test, so `2000 nm` and `0.3` relabel rather than resolve the reproducibility issue.
7. `[major]` **Evaluator-only ceilings lack composition rules.** For oracle backbones the plan does
   not say whether frozen assignments are applied before or after oracle joins, or how joined
   anchor aliases affect ownership. For oracle-clean anchors, filtering already-frozen assignments
   versus removing contaminated anchors and re-ranking clean candidates produces different
   ceilings. Both contracts would have to be invented.

## Questions

Raised by the coordinator for `plan_v2`, arising from the findings:

- Finding 1's first half reflects a genuine tension inside `task.md`: its evaluator-only section
  forbids reading GT before assignments are frozen, while Phase 0 and the first decision gate
  require reproducing the baseline NERL and L126 inventory before any policy is tested. The task
  cannot be satisfied both ways, so v2 resolves it structurally — the reproduction is
  evaluator-side, emits only a gate plus constants already published in `lessons.md` L126, and no
  `gt_free/` script may read anything under `evaluation_gt/`.
- Finding 5 is best settled by dropping multi-hop from this task rather than specifying it, since
  `task.md` makes it optional and the reviewer's leak argument is correct.
- Finding 7's oracle-clean-anchor variant has no neutral answer; v2 chooses filtering without
  re-ranking and labels the row a lower bound, because re-ranking onto clean runners-up would be a
  different, GT-informed policy.

## Verdict

VERDICT: NEEDS_CHANGES

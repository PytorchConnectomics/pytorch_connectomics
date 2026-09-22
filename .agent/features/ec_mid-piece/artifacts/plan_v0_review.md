# Plan v0 Review

## Summary

Reviewer: codex (coder role), read-only sandbox, `codex exec --sandbox read-only`.
Raw transcript: `state/plan_v0_review.review.raw.md`.

The reviewer returned `READY: no` with five `[major]` and two `[minor]` findings. The central
objection is that the plan's compute compromise — deriving the candidate-generation scope from
the GT LUT — is a ground-truth leak that the task's firewall forbids, and that calling it a
budget-only restriction does not repair it. The reviewer additionally judged that several
load-bearing policy contracts (bands, quarantine thresholds, morphology fusion, residual
precedence) are underspecified to the point where the coder would have to invent scientifically
consequential rules, and that the affinity verification as described would not reliably catch a
channel swap, sign flip, or one-voxel anchoring error.

The coordinator accepts all seven findings. No finding is softened or discarded.

## Findings

Verbatim severity tags preserved from the raw transcript.

1. `[major]` **GT-informed scope selector violates the firewall.** Passing
   `evaluation_gt/scope_segment_ids.txt`, derived from the LUT, into candidate generation uses GT
   to choose candidates, and contradicts the task's explicit "build candidates over the full
   segment population" requirement.
2. `[major]` **The scope-invariance assertion proves the wrong thing.** It shows only that
   assignments to fragments without sampled nodes do not directly change the current LUT. Omitted
   segments can still be anchors, competitors, contacts, relays, or contributors to RAG degree,
   quarantine, winner margins and component conflicts — so both one-hop ranking and the promised
   complete-graph multi-hop resolution can change under the restricted scope.
3. `[major]` **Essential policy contracts are undefined:** numerical fragment/anchor bands, anchor
   eligibility, "extreme" quarantine thresholds, the morphology score and fusion formula,
   reciprocal-best, residual-reason precedence, and the naming/freezing of individual grid-point
   assignments.
4. `[major]` **The affinity verification convention is internally unclear.** The plan declares
   channel `c` as array axis `c` anchored `v→v+1`, then describes an ABISS cross-check returning
   channel `2-c` "at an unspecified shifted position", without a source-indexed coordinate/channel
   truth table. The proposed real-crop boundary statistic would not reliably detect an axis swap,
   sign flip, or one-voxel anchoring error.
5. `[major]` **Phase 3's gate is insufficient for `r10_minpool`.** The task forbids reusing
   historically buggy r10 TTA/min output unless it passes the referenced affinity-TTA contract;
   substituting a newly implemented, incompletely specified geometry validation would be out of
   scope or violate the artifact restriction.
6. `[minor]` **Reporting rules are non-reproducible:** the required uncertainty calculation, the
   "material share" escalation threshold, and the quantitative corridor-evidence gate are not
   specified.
7. `[minor]` **"Pure LUT-level rescore" in Stage 2 reads as a firewall conflict** unless it means
   candidate-table-only scoring; any LUT access before freezing would violate stage separation.

## Questions

Raised by the coordinator for `plan_v1`, arising from the findings rather than from the reviewer:

- Finding 1/2 are fixable only by removing the scope selector and running the candidate pass over
  the full segment population via the chunked local-pass architecture the task itself prescribes.
  Plan v0's premise that this is unaffordable is the thing to re-derive, with an explicit cost
  estimate from the real chunk-store size (726 chunks, 2.6 TB) rather than from the raw voxel
  count.
- Finding 5 is most cleanly settled by declaring Phase 3 not run in this task and recording the
  gate decision, since `.agent/features/affinity_tta/task.md` is explicitly out of scope.
- Plan v0's own open question about the mechanism-ceiling rung (L123 277 joins vs the stronger
  L126 `>=50-owner-node` rung) was not addressed by the reviewer and remains the planner's call.

## Verdict

VERDICT: NEEDS_CHANGES

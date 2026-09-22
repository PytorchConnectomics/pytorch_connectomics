# Review v1

## Summary

Coordinator (claude/planner) in-session review of `code_v1`. The single [major] finding from
`review_v0` — the G1 Hermite curvature end-derivative sign — is fixed exactly
(`handle_v = -chord * tangent_v`, `handle_u` unchanged) and verified: the 256³ G1 smoke curvature
gate now passes **7/9** candidates (was 0/9), confirming the sign was the cause. Mutation guard
clean (HEAD unchanged, no tracked/staged changes), scope limited to `g1_banis_feasibility.py`,
artifact contract satisfied. **Approve.** Raw notes: `state/review_v1.review.raw.md`.

## Diff Baseline

run_start_ref: e8844b3da0f0a992431c901e7e2034486e7a678b

## Findings

- No new issues. The fix is minimal and correct; G0, `common.py`, `DESIGN.md`, and all sanities
  (cc0.66=0.601431, cc0.75=0.545357, BASE_NERL=0.8355, transform 100%) are unchanged from `code_v0`.
- The smoke's covered-events 0/2 is a 256³ artifact (only 2 oracle events exist in the tiny crop and
  the 7 accepted candidates did not happen to bridge those specific fragment pairs); it is not a
  defect. Full-chunk candidate recall is the real generator gate and is measured next by the
  coordinator.

## Tests to Add

- (Coordinator, full run) the actual G1 generator-recall gate (count ≥80%, ΔNERL-weighted ≥90%) and
  the G0 C0 grow-NERL ≥0.990 gate on the center chunk — these are the pass/fail numbers, run next.

## Questions

- None.

## Verdict

VERDICT: APPROVE

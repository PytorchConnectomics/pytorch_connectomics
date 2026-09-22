# Review v1

## Summary
code_v1 fix verified by the full-volume run (job 2710061, COMPLETED, 10 min). The `>=0.78` gate
issue from review_v0 is resolved (floor now 0.76) AND empirically vindicated: the eligible-substrate
oracle-merge NERL is 0.7696 — it would have aborted under the old 0.78 gate. The pipeline runs
end-to-end and IMPROVES over tube-bb; no-regression holds. APPROVE. Raw evidence:
`state/review_v1.review.raw.md`.

## Diff Baseline
run_start_ref: e8844b3da0f0a992431c901e7e2034486e7a678b

## Findings
- Real NERL: a=tube-bb 0.5933; b=plain 0.6048; c=constrained 0.6048. SELECTED=c 0.6048 >=
  max(recomputed a 0.5933, 0.593) — no-regression guaranteed and decode_v2 BEATS tube-bb (+0.0115),
  fewer merges (270 vs 278 gt), fewer splits (61 vs 62), fewer missed (256 vs 257).
- Substrate oracle-merge NERL 0.7696 (>= 0.76 floor; -0.0104 vs the 0.78 report target). Confirms
  the review_v0 finding: an eligible-only substrate splits far fewer sections (94 eligible / 73
  committed / 782 split) than force-split-ALL (534/9421 -> 0.780), so 0.78 was the wrong hard gate.
- [minor] The constrained union-find rejected 0 candidate links (c == b on this data): the
  eligibility gate + mandatory same-side links already keep axons separate, so the constraint is an
  UNFIRED safety net here. Correct and cheap; note that its value is unexercised on this dataset and
  would only manifest where the natural relink tries to re-merge a split.
- [minor] The GT-based argmax(A,C) selection is EVAL-ONLY (correctly flagged); c==a-or-better here so
  moot, but a GT-free selector remains future work.
- [infra, coordinator-fixed] `sbatch_decode_v2.sh` ran `set -u` before `source activate pytc`;
  conda's binutils activation references an unbound `ADDR2LINE`, aborting the job in 5s. Fixed by
  moving strict flags after activation (`set -eo pipefail`). Not a decode_v2.py defect.

## Tests to Add
- None. `--self-test` passes; the full-volume run is the real verification and completed with a
  measured improvement. Optional future: a fixture that forces a constraint rejection (c != b) to
  exercise the unfired safety net.

## Questions
- None blocking.

## Verdict
VERDICT: APPROVE

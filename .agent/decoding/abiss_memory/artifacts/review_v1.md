# Review v1

## Summary
Planner (claude, in-session) reviewed code_v1 and benchmark job 3039688; raw notes in `state/review_v1.review.raw.md`. All 4 review_v0 findings fixed; 0 major, 2 minor new findings.
Step 5 passes: 48/48 runs exit 0, 16/16 acceptance checks; 5-threshold ws64 on 1.21 Gvox 72.2 -> 39.4 GB (ratio 0.546, target <= 0.70), wall <= 0.81x stock everywhere.

## Diff Baseline
run_start_ref: 3cbdf39c06bf42c4b7cf739e1208751403a18bc7
current_head: 3cbdf39c06bf42c4b7cf739e1208751403a18bc7
abiss_start_ref: 92abc91f496304ecfe3d5c593e7713e8abd98011

## Findings
- review_v0:1 fixed — float32 percentile + finite `low < high` guard; the benchmark ran with valid thresholds.
- review_v0:2 fixed — parser-visible stdout lines restored and regex-checked by the gate; matches `ws_sizing.py:93-94`.
- review_v0:3 fixed — `ws runtime error:` + exit 4, tested on both widths.
- review_v0:4 fixed — rationale comment restored (drops the concrete LICONN ratios; not material).
- m1 [minor] `scripts/run_abiss_volume.py` — the Python driver is now the job peak: 41.9 GB RSS on every 1.21 Gvox case, vs 31.9 (ws) / 39.4 (ws64) GB for the binary, so about 10 GB of Python arrays stay resident. It is within acceptance; follow-up is to free or stream them around the ws subprocess.
- m2 [minor] benchmark — ws 5-threshold Python uint32 cgroup peak 63.8 GB > uint64 62.8 GB, the only case where uint32 is not lower; uncontrolled page-cache writeback. Report only.

## Tests to Add
none

## Questions
none

## Verdict
VERDICT: APPROVE_WITH_MINOR_COMMENTS

# Review v0

## Summary
Planner (claude, in-session) reviewed code_v0 against the approved plan_v2 and the real diffs in
both repositories; raw notes in `state/review_v0.review.raw.md`. The core C++ changes (C1-C7) and
the Python driver match the plan, and the exactness arguments hold in the code as written. Two major
issues block approval: the step-5 benchmark driver produces NaN thresholds (the job was submitted,
found broken and cancelled, 0 results), and the binary no longer prints stdout lines that an
existing script parses. Step 5 acceptance is still unmeasured. The plan's "assert compiled out"
claim was wrong (code_v0 is right: stock flags are `-O3 -fopenmp`, the assert is active).

## Diff Baseline
run_start_ref: 3cbdf39c06bf42c4b7cf739e1208751403a18bc7
current_head: 3cbdf39c06bf42c4b7cf739e1208751403a18bc7
abiss_start_ref: 92abc91f496304ecfe3d5c593e7713e8abd98011

## Findings
1. [major] `lib/abiss/tests/ws_gate.py:779-780` runs `np.percentile` on the float16 crop, which returns NaN for n > 65504 (reproduced: nan vs 0.7231 in float32). NaN thresholds make the watershed trivial, so the benchmark is invalid. Cast to float32 first and require finite `low < high` before running. Local correctness gate unaffected (fixed 0.95/0.05).
2. [major] `src/ws/atomic_chunk.cpp` dropped the `num of sv:` / `size of rg:` stdout lines, the per-threshold `(<thr>) ... in <secs> seconds` line, and the timing lines. `tutorials/neuron_liconn_ist/slurm/ws_sizing.py:93-94` regex-parses the first two. Restore them verbatim.
3. [minor] I/O exceptions from `write_volume` are uncaught in `main`, so they abort (exit 134). Catch them, print a message, and return a documented exit code.
4. [minor] The `internal_seg_t` rationale comment was cut to one line. Keep the high_bit / width-cost explanation; update only the stale sentence about on-disk type.

## Tests to Add
- Gate/benchmark: assert thresholds are finite with `low < high` (would have caught F1).
- A check that ws stdout still contains `num of sv:` and `size of rg:` in single-threshold mode, e.g. in `ws_gate.py`'s local mode.

## Questions
None for the coder. Step 5 will be run by the coordinator after code_v1.

## Verdict
VERDICT: NEEDS_CHANGES

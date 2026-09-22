# Review v0

## Summary

Reviewer: claude (planner), performed in-session. Raw transcript:
`state/review_v0.review.raw.md`.

The implementation is good work and the contested plan points landed correctly: the directed
interface statistic, the two separate mask contracts with their different pad values, and the
corrected R1/R3/R5 resolver are all implemented as `plan_v2` specified. The coder's claims were
checked rather than trusted — 21 tests pass, the legacy pretrain reproduction still passes, and
the CC3D store is complete and readable at 726/726.

Four major and four minor findings. One major finding is serious: the tier10 fidelity gate is
anchored on a sixteen-digit constant that nothing in the repository reproduces, which would turn
a task-declared blocker gate into either a spurious failure or a widened tolerance. The other
three majors are scale defects that the passing test suite cannot see because it exercises
single-digit candidate counts; they block the data stages rather than the code's correctness.

## Diff Baseline

run_start_ref: c705458ae5b907bb9c32a75c63c85c6aad7edec7

`.gitignore:159` ignores `dev/`, so the entire change surface is untracked and produces no git
diff. The review therefore read the files directly, as recorded in `run.md`. `git diff` and
`git diff --cached` were captured before and after the review and are unchanged.

## Findings

1. [major] **The tier10 gate constant has no verifiable provenance.** `common.py:62` declares
   `EXPECTED_TIER10_NERL = 0.7758321820491062`. Aggregating the ten archived
   `reports/*_HUMANGT_t10_arm0_win144_res20-9-9.csv` rows at threshold 0.70 gives
   0.694207 (mean `base_nerl_funlib`), 0.749818 (mean `base_nerl_emerl`), 0.764275 (mean
   oracle funlib), 0.800745 (mean oracle emerl), 0.682068 (median), 0.910325 (Σerl ratio), or
   0.794245 (erl-weighted) — none is 0.775832, and the string appears nowhere in the repository
   outside `task.md` and this experiment's own files. The `arm0_native` variant measures
   0.837572 / 0.852206 against task.md's 0.8544, so that row does not reproduce either. The
   coder reported B2 was never run, so the constant was never exercised; as written the gate
   fails. The danger is the obvious repair — widening the tolerance — which would convert
   task.md's Phase-0 blocker into a rubber stamp. Derive the constant from a named reproducible
   aggregation recorded beside it, or declare the gate unsatisfiable and escalate. The task's
   qualitative claim is unaffected: native still beats win144, by a wider margin than task.md's
   table implies.

2. [major] **Stage D's resolver is O(fragments × candidates).** `stage_d_freeze.py:156-170`
   rescans the whole candidate array per fragment via
   `np.flatnonzero(selected_band & (candidates["fragment"] == fragment))`, and calls `_edge_gate`
   once per edge in Python, for each of 16 policies. At full population this is on the order of
   1e11 comparisons and will not finish. Invisible to the tests, which use single-digit candidate
   counts. Group once with `np.unique(..., return_inverse=True)` and vectorise the gate.

3. [major] **Stage C5 persistence is O(candidates × chunk volume).** `stage_c5_affinity.py:102`
   allocates a full 1008³ boolean (~1.0 GB) and runs `np.argwhere` on it once per candidate,
   only to recover a bounding box Stage C3 already computed per component. Carry the bbox
   forward instead.

4. [major] **Stage C5 peak memory likely exceeds its allocation.** It holds the full affinity
   (~6.1 GB) plus `read_segmentation_zyx`'s **uint64** label crop (~8.1 GB) plus the mask
   concurrently, before finding 3's per-candidate temporaries — ~15 GB resident at minimum.
   ABISS ids fit in uint32, which halves the largest array. The coder flagged C3 memory itself
   but not C5.

5. [minor] **C5 rewrites a frozen-chain artifact in place.** `--output` defaults to
   `--candidates`, overwriting `gt_free/overlap_candidates.npz`. Stage D digests it afterwards so
   the hash stays self-consistent, but provenance then depends on execution order. Write a
   separate features file.

6. [minor] **The freeze digest covers evaluator code.** `stage_d_freeze.py:241` digests every
   `*.py` except `stage_b_fidelity.py`, so `stage_e/f/g` are inside the freeze hash; editing
   evaluator code later invalidates it and will read as a firewall breach during audit.

7. [minor] **`summarize_interface` returns inconsistent key sets** between its empty and
   non-empty branches (`interface_axis_count`). Latent — nothing reads it yet.

8. [minor] **Abstention reason codes are alphabetical, not causal.** `stage_d_freeze.py:169`
   uses `sorted(failure_reasons)[0]`, so `low_anchor_overlap` masks `low_support`. This distorts
   exactly the lost-opportunity attribution task.md requires the funnel to report.

## Tests to Add

* A scale guard for Stage D: synthesise ~1e4 fragments × ~1e5 candidate edges and assert the
  resolver completes within a fixed time budget. This is the test whose absence hid finding 2.
* A Stage C5 test asserting the persistence bbox comes from the C3 record, not from a dense
  rescan, and a peak-memory assertion for the chunk-level path.
* A provenance test for every declared constant in `common.py`: each must be reproducible from a
  named input by a named aggregation, or explicitly marked underived.
* A reason-code test asserting the recorded abstention equals the first failing gate in ladder
  order, over a candidate that fails several gates at once.

## Questions

None requiring an external decision. Finding 1 may, however, require re-deriving a number quoted
in `task.md` itself; if that number turns out to be unreproducible from archived artifacts, the
Phase-0 blocker gate needs redefining by the task author rather than by the implementation.

## Verdict

VERDICT: NEEDS_CHANGES

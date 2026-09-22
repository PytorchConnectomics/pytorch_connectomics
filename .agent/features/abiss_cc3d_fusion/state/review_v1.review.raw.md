# review_v1 raw transcript

Reviewer: claude (planner), performed in-session. Structured artifact: `artifacts/review_v1.md`.

## Method

Re-ran every claim in `code_v1.md` rather than trusting it, and wrote an independent
differential test for the one change that could silently alter results.

## Independently reproduced

- `pytest -q dev/zebrafinch/abiss_cc3d_fusion/tests` -> **28 passed** in 20.63s. Matches the
  coder's claim.
- `pytest -q dev/zebrafinch/cc3d/tests/test_compare_cc3d_abiss.py` -> **5 passed**. Legacy
  pretrain reproduction intact (required test 12).
- `git diff --stat dev/zebrafinch/cc3d/compare_cc3d_abiss.py` -> empty. The shared helper module
  is byte-identical; the coder imported it without editing it, as required.
- `git log c705458a..HEAD` -> 0 commits. `git rev-parse HEAD` == run_start_ref. No git state
  was altered.
- CC3D store re-verified during review_v0: 726/726 keys, 39 GB, zero unreadable files. This
  closes the coder's own caveat that "full readability was not rechecked".
- `python stage_b_fidelity.py` raises
  `RuntimeError: evaluator firewall: Stage D freeze_manifest.json is required`. The tier10
  blocker is therefore behind the freeze-chronology guard, which is the correct ordering from
  plan_v2 F1 — B2 must not run before Stage D.

## Differential verification of the vectorized resolver (finding 2's fix)

The rewrite from a per-fragment Python loop to `lexsort`/`unique`/`bincount`/`maximum.at`
reductions is the riskiest change in this version: it rewrites safety-critical selection logic,
and the coder's own tests are written against the same understanding that produced the code.

I wrote a reference resolver directly from the plan_v2 R1/R2/R3 specification — per-fragment
loop, per-anchor max aggregation on `(support, anchor_voxels, -component_uid)`, ranking on
`(-support, -anchor_voxels, anchor)`, `multi_host` abstention when `margin_min is None`, margin
test otherwise — and compared fragment->anchor assignments on randomised inputs.

    300 random trials x 5 policies (P2, P3, P4, P5, P6) = 1500 comparisons
    randomised: support, anchor overlap, anchor voxels, size ratio, interface mean/p10,
                direct contact count (including zeros), persistence, and all four veto flags
    mismatches = 0  -> EQUIVALENT

The randomisation exercises the veto path, the no-direct-contact path, the multi-host abstention
path, and the margin adjudication path. I did not compare reason codes, because finding 8
intentionally changed them from alphabetical to causal order.

## Finding disposition

1. **Resolved, honestly.** `common.py` retains `EXPECTED_TIER10_NERL = 0.7758321820491062` and
   adds `EXPECTED_TIER10_NERL_PROVENANCE` with `status: "unsatisfiable_from_archived_artifacts"`.
   `stage_b_fidelity.py` reproduces seven named archived aggregations and raises before doing
   replay work. The seven values the coder reports (0.694207, 0.7498184, 0.7642754, 0.800745,
   0.682068, 0.9103251973, 0.7942451740) match my own independent measurements exactly. The
   tolerance was not widened and the constant was not silently altered — which was the specific
   failure mode I was guarding against.
2. **Resolved and verified by differential test.** 10,000 fragments / 100,000 edges resolve in
   0.11 s against a 5 s budget.
3. **Resolved.** C3 component bboxes are carried through C4 into C5; `np.argwhere` bbox discovery
   is gone.
4. **Resolved.** Bounded crops plus overflow-checked `uint32` segmentation slabs; modeled peak
   15.262 GiB vs 19.077 GiB for the rejected `uint64` path, with a 16 GiB ceiling enforced.
5. **Resolved.** C5 writes `overlap_candidate_features.npz`; in-place overwrite is rejected.
6. **Resolved.** Explicit GT-free implementation list; B/E/F/G excluded from the freeze digest.
7. **Resolved.** `interface_axis_count` present on both branches and persisted.
8. **Resolved.** Causal ladder ordering with a multi-edge lost-opportunity test.

## Residual observations

- [minor] The scale guard's synthetic population gives every fragment ten edges to a **single**
  anchor, so the R3 multi-host adjudication and `np.maximum.at` loss-rank reduction are exercised
  at scale only in the degenerate single-host case. My differential test covers multi-host
  correctness but only at small n. A scale case with contested hosts would close this.
- [minor] Formatting verification is weaker than the artifact implies: an earlier Black run
  exited 124 (timeout) and the final checks used `--fast` because the `pytc` env is Python 3.11
  against a 3.12 target. Cosmetic only; mypy, flake8, and isort passed normally.
- Not a finding: no data stage has run beyond Stage A's manifest. That is a consequence of the
  B2 blocker and of cluster cost, not of code quality, and the coder stated it plainly.

## Verdict rationale

All four majors and all four minors from review_v0 are resolved; the riskiest fix is verified
equivalent by an independent implementation; the honest-blocker response to finding 1 is exactly
what was asked for. The two residual items are minor and do not affect correctness, safety, or
any reported number. Approving with those comments recorded.

VERDICT: APPROVE_WITH_MINOR_COMMENTS

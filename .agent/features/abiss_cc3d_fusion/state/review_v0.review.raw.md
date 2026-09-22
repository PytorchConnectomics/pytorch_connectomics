# review_v0 raw transcript

Reviewer: claude (planner), performed in-session per the CCC stage-ownership rule. This file is
the unabridged reviewer output; `artifacts/review_v0.md` is the structured artifact.

## Method

- Verified `git rev-parse HEAD` == `run_start_ref` (c705458a) after the user-approved re-baseline.
- `.gitignore:159` ignores `dev/`, so the change surface is untracked; reviewed the actual files
  rather than a diff, as recorded in run.md.
- Re-ran the coder's claims rather than trusting `code_v0.md`.
- Read in full: `common.py`, `stage_d_freeze.py`, `stage_c5_affinity.py`. Read partially:
  `stage_c0_maskdiff.py`, `stage_c3_evidence.py`, `stage_c4_candidates.py`, tests.

## Independently reproduced

- `pytest -q dev/zebrafinch/abiss_cc3d_fusion/tests` -> 21 passed in 6.73s. Coder's claim of 21
  passing is accurate.
- `pytest -q dev/zebrafinch/cc3d/tests/test_compare_cc3d_abiss.py` -> 5 passed. The legacy
  pretrain reproduction path is intact, satisfying required test 12.
- CC3D store: 726/726 keys present against the affinity index, 39 GB, zero unreadable or
  malformed files, border chunk `z2_y10_x7` correctly shaped (1008, 833, 1008).

## Confirmed correct against the contested plan points

- Interface orientation (plan F7): `directed_interface_values` builds 6-adjacent pairs per axis
  where one side is the fragment and the other the anchor, takes `affinity[axis]` at the
  lower-index voxel, and records the opposite convention as `interface_opposite_mean`. This is
  what plan_v2 specified. NaN from an empty interface cannot leak into a threshold comparison
  because `_edge_gate` returns `no_direct_contact` before comparing.
- Mask contracts (plan F2): `materialize_tier10_mask` pads KEEP=1 and requires even XY starts;
  `materialize_abiss_mask` pads 0. Both are separate functions with the correct pad values.
- Resolver R1/R3/R5 (plan F5): multi-anchor support components raise rather than emit;
  `margin_min is None` yields `multi_host` abstention for P2-P4; R5 invariants are asserted as
  post-hoc `RuntimeError`s, not used as the selection mechanism. This matches the corrected
  design.
- `classify_band` closes every case: `mid_hi == anchor_voxels` in all three bands, so the
  trailing `AssertionError` is unreachable and no value is left unclassified.
- Determinism: `deterministic_npz_bytes` fixes the zip timestamp, mode, and key order; all
  writes go through tmp + `os.replace`.

## Finding 1 [major] - the tier10 gate constant has no verifiable provenance

`common.py:62` declares `EXPECTED_TIER10_NERL = 0.7758321820491062`. Sixteen significant digits
assert a measured float. Nothing in the repository produces it.

Aggregating the ten archived `reports/*_thr_sweep_HUMANGT_t10_arm0_win144_res20-9-9.csv` rows at
threshold 0.70:

    mean base_nerl_funlib      0.694207
    mean base_nerl_emerl       0.749818
    mean oracle_nerl_funlib    0.764275
    mean oracle_nerl_emerl     0.800745
    median base_nerl_funlib    0.682068
    sum(berl)/sum(oerl)        0.910325
    erl-weighted base          0.794245

None is 0.775832. The string `0.775832` does not occur anywhere in the repository outside
`task.md` and this experiment's own files. The `arm0_native` variant measures 0.837572 (funlib)
/ 0.852206 (emerl) against task.md's 0.8544, so that row does not reproduce either.

The coder reported that gate B2 was not run, so this constant was never exercised. As written B2
will fail. The risk is not the failure - it is that the natural response to a failing gate is to
widen the tolerance, which would convert the task's Phase-0 blocker into a rubber stamp.

Required: derive the constant from a named, reproducible aggregation of named inputs and record
that definition next to it, or mark the gate unsatisfiable and escalate, because task.md makes
reproducing this row a blocker.

Note the task's qualitative claim survives: native (0.8376 / 0.8522) still beats win144
(0.6942 / 0.7498), and by a wider margin than task.md's table implies. The substrate resolution
to `arm0_win144` is unaffected.

## Finding 2 [major] - Stage D resolver is O(fragments x candidates)

`stage_d_freeze.py:156-170`:

    for fragment in fragments:
        all_indices = np.flatnonzero(selected_band & (candidates["fragment"] == fragment))

Every fragment rescans the entire candidate array, and `_edge_gate` is a Python-level call per
edge. arm0_96 has ~11.5k GT-visible pieces but the full-population mid band over the whole volume
is plausibly 1e5-1e6 ABISS labels with a comparable or larger candidate count, giving ~1e11
elementwise comparisons plus ~1e6 Python calls, repeated for each of the 16 policies. Stage D
will not finish.

The tests use single-digit candidate counts, so this is invisible to the suite. Group once with
`np.unique(fragment, return_inverse=True)` and an `argsort` bucket, and vectorise `_edge_gate`
over the candidate arrays.

## Finding 3 [major] - Stage C5 persistence is O(candidates x chunk volume)

`stage_c5_affinity.py:102-107`, called once per candidate:

    relevant = (labels == fragment) | (labels == anchor)
    coordinates = np.argwhere(relevant)

Each call allocates a full 1008^3 boolean (~1.0 GB) and `argwhere` materialises every set
coordinate, purely to recover a bounding box that Stage C3 already computed per component. With
hundreds of candidates per chunk this dominates the stage. Carry the bbox forward from C3 instead
of recomputing it from the dense array.

## Finding 4 [major] - Stage C5 peak memory likely exceeds the allocation

`main` holds the full affinity (3 x 1008^3 float16, ~6.1 GB), plus
`read_segmentation_zyx` which returns **uint64** labels (1008^3 x 8 B = ~8.1 GB), plus the mask,
before any per-candidate temporary. That is ~15 GB resident before Finding 3's ~1 GB per-call
allocations. ABISS label ids fit comfortably in uint32; casting there halves the largest array.
The coder independently flagged C3 memory in its own Risks section, but not C5.

## Finding 5 [minor] - C5 rewrites a frozen-chain artifact in place

`--output` defaults to the same path as `--candidates`, so C5 overwrites
`gt_free/overlap_candidates.npz`. Stage D digests that file afterwards so the recorded hash is
still self-consistent, but an in-place mutation of a GT-free artifact makes the provenance chain
depend on execution order. Write `overlap_candidates_features.npz` instead.

## Finding 6 [minor] - freeze digest covers evaluator code

`stage_d_freeze.py:241` builds `implementation_sha256` over every `*.py` except
`stage_b_fidelity.py`, so `stage_e_evaluate.py`, `stage_f_diagnostics.py`, and
`stage_g_report.py` are inside the freeze digest. Editing evaluator code after the freeze then
invalidates the recorded hash even though no GT-free logic changed, which will read as a
firewall breach during audit. Digest the GT-free modules only.

## Finding 7 [minor] - inconsistent feature keys

`summarize_interface` returns `interface_axis_count` only on the non-empty branch. Nothing
downstream reads it today, so this is latent rather than live.

## Finding 8 [minor] - misleading abstention reason codes

`stage_d_freeze.py:169` sets the reason to `sorted(failure_reasons)[0]`, which is alphabetical,
not the first failing gate. `low_anchor_overlap` will mask `low_support`, distorting the funnel's
lost-opportunity attribution that task.md requires be attributed correctly.

## Verdict rationale

Findings 2-4 are scale defects invisible to the passing test suite; they block the data stages
rather than the code's correctness. Finding 1 is the serious one, because it silently converts a
blocker gate into either a spurious failure or a widened tolerance. Another code version is
allowed under c2, so the run continues to code_v1.

VERDICT: NEEDS_CHANGES

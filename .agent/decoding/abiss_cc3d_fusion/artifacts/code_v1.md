# Code v1

## Overview

Revised the arm0_96 ABISS/CC3D fusion implementation to address all eight review findings. The tier10 gate now fails honestly, and the Stage D/C5 scale defects are guarded by meaningful high-cardinality and memory tests.

## What Changed

- Preserved the tier10 constant and marked it explicitly unsatisfiable from archived artifacts.
- Replaced Stage D’s fragment-by-candidate scans with vectorized gates and grouped sorts.
- Carried C3 component bounding boxes into C4/C5.
- Made C5 process bounded crops using slabbed `uint32` segmentation reads.
- Separated C4 proposals from C5 feature output.
- Restricted the freeze digest to GT-free implementation files.
- Made interface summaries schema-consistent.
- Recorded abstentions in causal gate order.

## Implementation Details

- B2 now reproduces seven named archived aggregations and raises before launching any replay because none equals `0.7758321820491062` at the unchanged `1e-6` tolerance.
- Stage D uses `np.unique(..., return_inverse=True)`, vectorized gate masks, and globally sorted fragment/anchor reductions. The 100,000-edge regression completes in 0.11 seconds.
- C4 stores the supporting component’s global C3 bbox with each assignment candidate.
- C5 expands that bbox by one voxel, loads only the bounded affinity/segmentation/mask crop, reads segmentation as checked `uint32` slabs, and computes persistence once per supporting component.
- C5 writes `overlap_candidate_features.npz`; attempting to overwrite `overlap_candidates.npz` is rejected.
- The modeled full-1008³ peak is 15.262 GiB with `uint32`, versus 19.077 GiB for the rejected `uint64` path.
- Stage D hashes an explicit A/C0–C5/D implementation list. B/E/F/G are excluded.
- Fragment loss attribution records the first ladder stage after which no candidate edge survives.

## Files Changed

| File | Purpose |
|---|---|
| `dev/zebrafinch/abiss_cc3d_fusion/README.md` | Document honest B2 blocker, completed tier10 store, separate C5 artifact, and digest scope |
| `dev/zebrafinch/abiss_cc3d_fusion/common.py` | Declare constant provenance and add slabbed, checked segmentation reads |
| `dev/zebrafinch/abiss_cc3d_fusion/schemas.py` | Add bbox/interface fields and the separate C5 feature artifact schema |
| `dev/zebrafinch/abiss_cc3d_fusion/stage_b_fidelity.py` | Reproduce named archived aggregations and fail the underived tier10 gate loudly |
| `dev/zebrafinch/abiss_cc3d_fusion/stage_c4_candidates.py` | Carry C3 component bboxes into assignment candidates |
| `dev/zebrafinch/abiss_cc3d_fusion/stage_c5_affinity.py` | Use bounded crops, `uint32` slabs, shared persistence decoding, and separate output |
| `dev/zebrafinch/abiss_cc3d_fusion/stage_d_freeze.py` | Vectorize resolution, correct reason ordering, and narrow the implementation digest |
| `dev/zebrafinch/abiss_cc3d_fusion/stage_e_evaluate.py` | Consume the C5 feature artifact |
| `dev/zebrafinch/abiss_cc3d_fusion/stage_f_diagnostics.py` | Consume the C5 feature artifact |
| `dev/zebrafinch/abiss_cc3d_fusion/tests/test_constant_provenance.py` | Test declared provenance and the fail-closed tier10 gate |
| `dev/zebrafinch/abiss_cc3d_fusion/tests/test_evidence_candidates.py` | Cover bbox propagation into candidates |
| `dev/zebrafinch/abiss_cc3d_fusion/tests/test_interface_firewall.py` | Test bbox persistence, interface keys, and modeled peak memory |
| `dev/zebrafinch/abiss_cc3d_fusion/tests/test_resolver.py` | Add causal-reason, digest-scope, and 100,000-edge scale guards |

## Git Baseline

run_start_ref: c705458ae5b907bb9c32a75c63c85c6aad7edec7
current_head: c705458ae5b907bb9c32a75c63c85c6aad7edec7

## Verification

- `pytest -q dev/zebrafinch/abiss_cc3d_fusion/tests`: **28 passed, 3 external SWIG deprecation warnings in 21.14s**.
- Scale guard: **10,000 fragments / 100,000 edges resolved in 0.11s**, under the fixed 5-second budget.
- Tier10 provenance check raised the intended `RuntimeError: tier10 fidelity gate is explicitly unsatisfiable-from-archived-artifacts`.
- Reproduced archived values: `0.694207`, `0.7498184`, `0.7642754`, `0.800745`, `0.682068`, `0.9103251973`, and `0.7942451740`; none matches the retained constant.
- Black final checks, isort, flake8, mypy, `compileall`, CLI `--help`, and `git diff --check`: passed.
- An earlier Black formatting invocation exited `124` after formatting all five flagged files; subsequent clean checks passed using a fresh `/tmp` cache.
- Confirmed the tier10 CC3D directory contains 726 chunk files. Full readability was not rechecked.
- Confirmed `dev/zebrafinch/cc3d/compare_cc3d_abiss.py` has no diff and its regression test passes.
- Stages **A, C0–C5, D, B1, full B2 replay, and E–G were not run**. No SLURM job, git commit, staging operation, or push was performed.

## Review Focus

- Confirm that B2’s explicit blocker is the desired response pending a task-author-provided aggregation.
- Review the one-voxel halo around the carried C3 component bbox.
- Review the vectorized R1–R5 reductions against the previous resolver semantics.
- Confirm the separate C5 feature artifact and narrowed freeze digest preserve the intended chronology.

## Risks and Unknowns

- B2 is intentionally blocked until the tier10 constant receives reproducible provenance.
- C5’s real full-population runtime and memory remain unmeasured; it now uses bounded crops and enforces a modeled 16 GiB ceiling.
- No scientific fusion conclusion exists because the data and evaluator stages were not run.
- The 726-file store count was confirmed, but all files were not reopened during this revision.
- `dev/` remains ignored by `.gitignore`; unrelated pre-existing worktree changes remain untouched.
- Black’s normal safety check is incompatible with the Python 3.11 environment versus the repository’s Python 3.12 target, so final checks used `--fast`.

## Changes Since Previous Code Version

1. **Finding 1:** Retained `0.7758321820491062`, recorded explicit unsatisfiable provenance beside it, reproduced all seven named archived checks, and made B2 fail loudly before replay work.
2. **Finding 2:** Replaced the O(fragments × candidates) resolver with vectorized gates and grouped global sorts; added the 10k-fragment/100k-edge time guard.
3. **Finding 3:** Carried C3 component bboxes through C4 and removed `np.argwhere` bbox discovery from persistence.
4. **Finding 4:** Replaced dense `uint64` chunk reads with slabbed, overflow-checked `uint32` bounded reads; added a peak-memory regression and runtime budget check.
5. **Finding 5:** Changed C5’s default output to `overlap_candidate_features.npz` and explicitly rejects in-place candidate overwrite.
6. **Finding 6:** Replaced wildcard Python hashing with an explicit GT-free implementation list excluding B/E/F/G.
7. **Finding 7:** Added `interface_axis_count` to both empty and non-empty summaries and persisted it in the feature schema.
8. **Finding 8:** Replaced alphabetical reason selection with causal ladder ordering, including a multi-edge lost-opportunity test.
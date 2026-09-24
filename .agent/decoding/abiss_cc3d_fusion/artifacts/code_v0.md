# Code v0

## Overview

Implemented the approved arm0_96 ABISS/CC3D@0.70 fusion experiment under `dev/zebrafinch/abiss_cc3d_fusion/`.

The implementation preserves ABISS global identity, restricts fusion to frozen one-hop fragment-to-anchor assignments, enforces the GT firewall, and leaves cluster-dependent stages runnable without fabricating results.

## What Changed

- Added stage-specific CLIs for identity, mask comparison, conditional decoding, evidence extraction, candidate generation, policy freezing, evaluation, diagnostics, and reporting.
- Added deterministic artifacts, schemas, hashes, atomic writes, resumable shards, and global namespace handling.
- Implemented all three GT-free candidate families: assignment, face continuation, and bounded cut proposals.
- Implemented P1–P6, persistence, robustness-neighbor, and band policies with corrected R1–R5 resolution.
- Added CPU tests covering the required safety, connectivity, alignment, remapping, firewall, and determinism contracts.
- Generated the real 726-chunk GT-free input manifest.
- Added SLURM batch files but did not submit them.
- Did not modify existing Zebrafinch scripts or canonical inputs.

## Implementation Details

- The selected substrate is explicitly recorded as `arm0_win144`; `arm0_native` remains the `_win48x96x96` sibling.
- C0 compares both masks voxelwise. Any `n_extra > 0` forces an ABISS-mask re-decode; `n_missing > 0` is fatal.
- CC3D identities use `(chunk_ordinal << 32) | local_label`.
- Multi-anchor support components never emit assignment edges.
- P2–P4 abstain on multi-host fragments; P5+ applies the frozen winner margin.
- Interface affinity uses the declared directed axis channel at the lower-index voxel and records the opposite convention diagnostically.
- Stage D freezes proposals and implementation hashes before evaluator stages.
- Evaluator outputs separate frozen policies, negative controls, and GT-only ceilings.
- Composition requires exact ABISS-info and LUT-shard digests and rejects multi-anchor invariant violations.

## Files Changed

| File | Purpose |
|---|---|
| `dev/zebrafinch/abiss_cc3d_fusion/README.md` | Commands, firewall, artifacts, and actual stage status |
| `dev/zebrafinch/abiss_cc3d_fusion/__init__.py` | Experiment package marker |
| `dev/zebrafinch/abiss_cc3d_fusion/common.py` | Paths, masks, CC3D semantics, namespacing, deterministic IO, remapping |
| `dev/zebrafinch/abiss_cc3d_fusion/schemas.py` | Closed GT-free JSON/NPZ schemas and leakage audit |
| `dev/zebrafinch/abiss_cc3d_fusion/stage_a_identity.py` | 726-chunk identity manifest and tag-resolution evidence |
| `dev/zebrafinch/abiss_cc3d_fusion/stage_c0_maskdiff.py` | Sharded voxelwise mask comparison and finalization |
| `dev/zebrafinch/abiss_cc3d_fusion/stage_c1_redecode.py` | Exact reuse or ABISS-mask CC3D re-decode |
| `dev/zebrafinch/abiss_cc3d_fusion/stage_c2_storegate.py` | Fresh-versus-stored partition fidelity gate |
| `dev/zebrafinch/abiss_cc3d_fusion/stage_c3_evidence.py` | Per-chunk component, overlap, nucleus, mask, and face evidence |
| `dev/zebrafinch/abiss_cc3d_fusion/stage_c3_reduce.py` | Completeness checks and deterministic global reduction |
| `dev/zebrafinch/abiss_cc3d_fusion/stage_c4_candidates.py` | Assignment, reciprocal-face, and cut candidates |
| `dev/zebrafinch/abiss_cc3d_fusion/stage_c5_affinity.py` | Directed interface and persistence features |
| `dev/zebrafinch/abiss_cc3d_fusion/stage_d_freeze.py` | Policy ladder, resolver, abstentions, and freeze hashes |
| `dev/zebrafinch/abiss_cc3d_fusion/stage_b_fidelity.py` | Baseline and ten-chunk evaluator fidelity gates |
| `dev/zebrafinch/abiss_cc3d_fusion/stage_e_evaluate.py` | Partition audit, oracles, frozen policy metrics, Gate A, funnel |
| `dev/zebrafinch/abiss_cc3d_fusion/stage_f_diagnostics.py` | Visual probes, split diagnostics, false-merge cross-tab, composition |
| `dev/zebrafinch/abiss_cc3d_fusion/stage_g_report.py` | Final ordered research report |
| `dev/zebrafinch/abiss_cc3d_fusion/sbatch_stage_c0.sh` | C0 16-shard batch script |
| `dev/zebrafinch/abiss_cc3d_fusion/sbatch_stage_c1.sh` | C1 16-shard batch script |
| `dev/zebrafinch/abiss_cc3d_fusion/sbatch_stage_c3.sh` | C3 16-shard batch script |
| `dev/zebrafinch/abiss_cc3d_fusion/gt_free/input_manifest.json` | Real manifest for all 726 affinity chunks |
| `dev/zebrafinch/abiss_cc3d_fusion/tests/conftest.py` | Test import setup |
| `dev/zebrafinch/abiss_cc3d_fusion/tests/test_cc3d_masks.py` | Foreground, connectivity, masks, namespacing, store fidelity |
| `dev/zebrafinch/abiss_cc3d_fusion/tests/test_evidence_candidates.py` | Overlaps, candidates, faces, and band boundaries |
| `dev/zebrafinch/abiss_cc3d_fusion/tests/test_resolver.py` | R1/R3/R5, relay prevention, and permutation invariance |
| `dev/zebrafinch/abiss_cc3d_fusion/tests/test_interface_firewall.py` | Interface orientation, schemas, GT rejection, LUT remapping |
| `dev/zebrafinch/abiss_cc3d_fusion/tests/test_alignment_and_regression.py` | Canonical CloudVolume alignment and legacy helper regression |

## Git Baseline

run_start_ref: c705458ae5b907bb9c32a75c63c85c6aad7edec7
current_head: c705458ae5b907bb9c32a75c63c85c6aad7edec7

Coordinator correction (baseline fields updated after the run was re-baselined; the values
recorded when this artifact was produced were run_start_ref
6ad67866c3ca3e1cf1d52437a0cb300b0340c1bd and current_head
c705458ae5b907bb9c32a75c63c85c6aad7edec7): the coder was instructed to write `current_head` equal to
`run_start_ref`, which was true when the code prompt was assembled. HEAD moved during the
stage and the field now records the measured value. The mover was a concurrent session,
not this coder — commit `c705458a` "Harden and document nucleus ABISS replays" touches no
file under `dev/zebrafinch/abiss_cc3d_fusion/`. The coder independently reported the
divergence in `## Risks and Unknowns` and altered no git state.

## Verification

- `pytest -q dev/zebrafinch/abiss_cc3d_fusion/tests`: **21 passed in 3.95s**.
- `mypy --config-file .github/mypy_changed.ini ...`: **Success, no issues in 16 source files**.
- Black, isort, flake8, and `git diff --check`: passed.
- Every stage’s `--help` and all three batch files via `bash -n`: passed. Evaluator imports emitted only a temporary Matplotlib-cache warning.
- Stage A completed: **726 affinity files** and **9 direct HDF5 probes** recorded; selected tag is `arm0_win144`.
- GT-free structural audit passed for the generated manifest.
- Canonical arm0_96 baseline assertion against `0.4443760423975249` passed at `1e-10` tolerance.
- Synthetic clipped-border mask/decode smoke passed.
- Real `z2_y10_x7` C0 border smoke was attempted but interrupted after approximately three minutes without completing; it emitted no artifact or Python error.
- Ten-chunk tier10 reproduction, full C0, C1–C5, Stage D freeze, Phase E–G evaluation, and 726-chunk processing were not run because they require cluster-scale data work and/or the still-running CC3D store.
- No SLURM job, commit, staging operation, or push was performed.

## Review Focus

- Verify C0/C1’s exact mask-difference and reuse/re-decode contract.
- Review R1 multi-anchor refusal and R3 multi-host handling across P2–P6.
- Confirm the directed affinity channel/index convention in Stage C5.
- Review C3 memory requirements before launching the 16-shard array.
- Confirm Stage D’s schema audit and freeze chronology.
- Review cut-oracle interpretation and composition invariant rejection.

## Risks and Unknowns

- No scientific fusion conclusion exists yet because the full-population candidates have not been frozen or evaluated.
- The real border-chunk C0 smoke did not finish interactively; cluster execution remains unverified.
- C3 intentionally processes one dense chunk per worker and may approach the requested 96 GB allocation.
- The live repository `HEAD` observed during final inventory was `c705458ae5b907bb9c32a75c63c85c6aad7edec7`, which differs from the baseline lines mandated for this artifact. Git state was not altered.
- `.gitignore` ignores `dev/`, so these new files do not appear in normal `git status`; the prohibition on `git add` was honored.
- Numerous unrelated pre-existing worktree changes remain untouched.

## Changes Since Previous Code Version

Initial implementation.
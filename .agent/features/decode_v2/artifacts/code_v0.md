# Code v0

## Overview

Implemented the PLAN v3 LiCONN decode pipeline: tube-bb baseline, exactly-two fusion detection,
upstream-only eligibility, exclusive seeded-watershed propagation, plain and constrained relinking,
instance/full-volume metrics, and evaluation-only no-regression selection. The full 800-slice SLURM
run was intentionally not executed.

## What Changed

- Added `decode_v2.py` with strict mutually exclusive full/slice/self-test modes, validated slice
  bounds and shapes, cached-section loading, fusion/run reporting, A/B/C evaluation, suffixed slice
  output, full-volume NERL gates, and selected-output writing.
- Extracted one reusable `force_split(seg2d, affxy, incoming, eligible_runheads, area_tol,
  next_overlap)` implementation and made the existing `force_split_decode.py` CLI call it.
- Added exclusive label ownership, explicit propagation stop causes, max-overlap successor choice,
  total-overlap 2x2 terminal assignment, per-side overlap gates, and mandatory seed chains.
- Added a constrained union-find that applies mandatory links first, orders optional links by
  `(-IoU, min-id, max-id)`, and prevents components from acquiring both sides of one split run.
- Added the requested three-hour, 120G short-partition SLURM launcher.

N>2 incoming fusions are deliberately skipped and counted. Splitting higher-multiplicity fusions is
the accepted scope reduction in PLAN v3; neither propagation nor relinking assigns a still-fused
N>=2 successor to one side.

## Implementation Details

Positive waterz section labels are asserted to occur on exactly one z-slice; background label 0 is
explicitly excluded. Detection uses raw consecutive `segs_to_iou` rows with an independent
`--fuse-iou` threshold. Eligibility traverses only decreasing-z conservative-spine predecessors,
requires disjoint reachable sets, and measures distinct upstream z-depth.

Committed force-split runs own their head, seed anchors, fresh pieces, and matched terminals. A
candidate run with any owned head/seed is conflict-skipped. Propagation chooses the largest union
overlap with smallest-label tie-break, stops before any multi-incoming or owned successor, and
claims a natural-separation pair only after comparing both bijections and validating both sides.

The constrained union-find builds one annotation per owned side label, conflict-checks forced and
optional unions, and asserts that every piece and terminal remains connected to its seed anchor.
Slice mode reports instance metrics only and saves C as the candidate. Full mode saves A/B/C plus
the unlinked substrate, computes real and oracle-merge NERL, gates the substrate oracle at 0.78,
and writes the better real-NERL result of A/C subject to the 0.593 no-regression floor. This
GT-based A/C selection is evaluation-only.

## Files Changed

| File | Purpose |
|---|---|
| `dev/mit_liconn/decode_v2.py` | Full decode v2 pipeline, constrained union-find, metrics, CLI, and self-test. |
| `dev/mit_liconn/force_split_decode.py` | Reusable exclusive force-split implementation; legacy main delegates to it. |
| `dev/mit_liconn/sbatch_decode_v2.sh` | Requested SLURM full-run launcher and output-path reporting. |
| `.agent/features/decode_v2/artifacts/code_v0.md` | CCC implementation and verification handoff. |

## Git Baseline

run_start_ref: e8844b3da0f0a992431c901e7e2034486e7a678b
current_head: e8844b3da0f0a992431c901e7e2034486e7a678b

## Verification

- `source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc && python
  dev/mit_liconn/decode_v2.py --self-test` — exit 0; `decode_v2 self-test: PASS`.
- `source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc && python
  dev/mit_liconn/decode_v2.py --zslice 0:96` — exit 0. Fusion summary:
  `exactly2=47`, `area_matched=47`, `N>2_skipped=0`, `eligible=9`, `committed=9`,
  `split_sections=87`, `terminals=4`; stops were 4 degenerate splits, 1 no-next, 2 natural
  separations, and 2 volume ends. Constrained relink applied 178 forced and 27,807 candidate links
  with 0 rejected in this crop.
- Pilot instance results: A/B/C voxel P/R/F1 were all `0.9334 / 0.8941 / 0.9133`; A had 549
  predictions and B/C had 545; all had 11 false-merge predictions affecting 22 GT, 6 false-split
  GT, and 39 missed GT. VI sum was 0.0487 for A and 0.0480 for B/C.
- Pilot artifact `outputs/mit_liconn/DL288B_crop1/decode_v2_z0-96.h5` was written as dataset
  `main`, shape `(96, 1024, 1024)`, `uint32`, gzip level 1.
- `python -m py_compile ...`, `black --check --fast ...`, `isort --check-only ...`, and
  `flake8 --max-line-length=100 ...` passed for both changed Python files. `bash -n
  dev/mit_liconn/sbatch_decode_v2.sh` passed.
- The full 800-slice SLURM job was not run, as requested; full real/oracle NERL and hard-gate
  results remain for the reviewer/coordinator.

## Review Focus

Review the propagation branch precedence and ownership accounting, especially ambiguous/owned
successors and terminal matching; verify mandatory-chain connectivity and conflict rejection on a
full run; then confirm the unlinked-substrate oracle floor and selected A/C no-regression gate from
the SLURM output.

## Risks and Unknowns

- Full-volume NERL, oracle-merge NERL, memory use, and wall time remain unverified until the SLURM
  job runs. `split_merges.md` therefore has not been updated with fabricated or partial results.
- `segs_to_iou` is the established best-successor-per-source raw table rather than a complete
  all-pairs overlap table; propagation uses direct pixel overlap to enumerate next sections.
- GT-based selected-output choice is evaluation-only and is not suitable for deployment without a
  future GT-free selector.

## Changes Since Previous Code Version

Initial implementation.

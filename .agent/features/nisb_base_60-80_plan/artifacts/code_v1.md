# Code v1

## Overview

Applied the single G1 correctness fix requested by `review_v0`: the cubic-Hermite end derivative at
tip `v` now points along the travel direction into `v`'s body. No tracked framework file, G0/shared
code, design document, threshold, or other behavior was changed.

Re-running only the fixed 256^3 G1 smoke at origin `[475, 658, 31]` confirmed that the curvature
gate no longer rejects every evaluated continuation: 7 of 9 evaluated candidates passed.

## What Changed

- Changed `handle_v` in `hermite_min_curvature_radius` from `chord * tangent_v` to
  `-chord * tangent_v`.
- Kept `handle_u = chord * tangent_u` unchanged.
- Re-ran only `g1_banis_feasibility.py --smoke` in the required `pytc` environment.

## Implementation Details

The stored tip tangent at `v` is outward-facing, while the cubic-Hermite derivative at the end of
the curve represents travel arriving at `v`. The end handle therefore uses `-tangent_v`, consistent
with the existing tangent-agreement term `dot(tip_v.tangent, -direction)`. The 32-sample curvature
calculation and the proposal gate `rho_min >= 1.25 * r_max` are otherwise unchanged.

## Files Changed

| File | Purpose |
|---|---|
| `dev/nisb/scripts/mesa/g1_banis_feasibility.py` | Correct the cubic-Hermite end-handle sign at tip `v` |
| `.agent/features/nisb_base_60-80_plan/artifacts/code_v1.md` | Record the scoped fix and smoke verification |

Both files are untracked/ignored CCC research deliverables. No tracked file was edited, and nothing
was staged or committed.

## Git Baseline

run_start_ref: e8844b3da0f0a992431c901e7e2034486e7a678b
current_head: e8844b3da0f0a992431c901e7e2034486e7a678b

`HEAD` remains at the run-start reference. The visible tracked and staged state is unchanged from
the pre-fix check.

## Verification

Only the requested fast G1 smoke was run from the repository root:

```bash
source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc
export NUMBA_CACHE_DIR=/tmp MPLCONFIGDIR=/tmp PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.
python dev/nisb/scripts/mesa/g1_banis_feasibility.py --smoke
```

The command exited 0 and emitted:

- Origin/shape: `[475, 658, 31]`, `256^3`.
- Curvature gate: 9 candidates evaluated; 7 passed `rho_min >= 1.25 * r_max` (`77.8%`).
- Candidate counts: 7 raw, 7 capped, 2 geometrically matched.
- Covered events: 0 of 2.
- Supporting smoke counts: 976 queried tip pairs, 914 cross-fragment pairs, 38 within the distance
  limit, 9 tangent-agreement passes, 9 radius-ratio passes, and raw max 1 candidate per endpoint.
- Unchanged smoke context: base NERL `0.7714994297166`, 346 instances, and 224 tips.
- Script-reported runtime: `183.00129318237305` seconds.

Sanity A/B, transform checks, G0, the full G1 gate, and all full-center gates were not re-run.

## Review Focus

- Confirm `handle_v = -chord * tangent_v` and `handle_u = chord * tangent_u` in
  `hermite_min_curvature_radius`.
- Confirm no other G1 behavior or CCC research deliverable changed.
- Confirm the smoke's 7 accepted candidates equal the raw proposal count because curvature is the
  final generator gate.

## Risks and Unknowns

- The sign fix clears the specific false-negative failure: 7 of 9 curvature-evaluated candidates
  now pass. The smoke still covers 0 of its 2 events, so it does not establish full G1 recall.
- The full G1 and full-center gates remain deliberately unrun for the coordinator.
- The smoke reported 183.0 seconds in this run, slower than the prior warm-run reference; no broader
  performance investigation was performed because it is outside this fix.

## Changes Since Previous Code Version

Corrected the Hermite end derivative at `v` from `+chord * tangent_v` to
`-chord * tangent_v`; no other implementation behavior changed.

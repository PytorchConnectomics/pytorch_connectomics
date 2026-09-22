# Code v0

## Overview

Implemented the ratified MESA-EM cheap-gate package without modifying tracked framework code or
launching training. G0 is the approved 1-voxel-skeleton oracle, not a medial-band Phase-0 redesign.
G1 keeps generator recall, fragment-degree geometry matching, the GT-confirmed upper bound, and
control/repaired fragment-graph realization as separate measurements. The four code-stage addendum
resolutions are implemented explicitly.

Only the blocking harness sanities and 256^3 smoke paths were run. The full center-chunk G0 offset
sweep and G1 pipeline remain for the coordinator after code review.

## What Changed

- Added shared bound-asset loaders, scorer adapters, skeleton coordinate/paint helpers, local crop
  graph construction, GT-SDT construction, cached-LUT sanity, and closed-form DeltaNERL validation.
- Added G0 sparse offset-edge construction, identity-blind union-find, GT-SDT watershed grow,
  skeleton-identity audits, physical retained-length metrics, and C1/C2/C3 graph perturbations.
- Added G1 break-cluster events, exact physical proposal geometry, top-8 capping, fragment-degree-1
  mutual matching, capped/full oracle relabels, and control/repaired fragment-adjacency realization.
- Added a concrete MESA-EM architecture/experiment design and an identical CCC artifact mirror.
- Preserved all pre-existing tracked and untracked user changes; nothing was staged or committed.

## Implementation Details

The shared harness binds `(X,Y,Z)` and `[9,9,20]` nm throughout. Sanity A scores only cached
whole-volume node LUTs. Sanity B crops the bound three-channel affinity, calls
`decode_affinity_cc(..., backend="numba", edge_offset=0)`, and uses the local center skeleton with
the canonical NERL scorer.

G0 paints `node_skeleton_index+1`, constructs SAME-edge targets sparsely, then passes only node
occupancy and accepted endpoint pairs to union-find. The cumulative banks are asserted at
13/16/24/62. Long edges use a digital-line foreign-corridor audit against raw `seg.h5` IDs. Dense
GT values are separately split into 6-connected local instances for endpoint consistency, SDT
targets, and foreground masking. The ero1 SDT uses explicit `(1,1,0)` erosion for the actual
`(X,Y,Z)` layout. False merges are gated in the scorer-visible universe: a grown label may not span
multiple local skeleton identities.

G1 implements addendum A verbatim: fragment-mask EDT at `[9,9,20]`, K=12 tip radius, the pinned
tangent agreement, 32-sample finite-difference cubic-Hermite curvature, and distinct proposal vs.
acceptance thresholds. Matching collapses duplicate fragment pairs and performs one immutable
mutual-top round with degree at most one per baseline fragment. Event deltas use `2*L_a*L_b/D`, are
validated on 20 deterministic events, and fall back to canonical per-event scoring if needed.

The addendum-B realizer adds 6-connected label-0 pseudo-fragments to the face RAG, uses the matching
affinity channel for each undirected face, retains best and runner-up distinct-identity geodesic
costs, and applies the `log(9)` abstention margin. It runs both per-fragment control and
capped-oracle repaired seeds, audits grown skeleton identities after adoption, measures dense-GT
adoption precision/coverage/abstention, and never splits an atomic fragment. Full-center tip
extraction is a memory-bounded, per-label exact adaptation of `_extract_tips_fast`.

The design pins the 43 dense channels, interior-positive `d_b`, exact medialness cutoff, tangent
masking, physical veto ball, and conditional sparse-skip SAME/MUTEX/DEFER rule from addendum D. It
also states the G0-oracle/Phase-2-learnability boundary, phase ladder, forecast off 0.627, falsified
directions, merge audit, and `tile_0_0_2` recommendation.

## Files Changed

| File | Purpose |
|---|---|
| `dev/nisb/scripts/mesa/common.py` | Bound assets, graph/crop/paint helpers, GT-SDT, both scorers, Sanity A/B/transform, and DeltaNERL validator |
| `dev/nisb/scripts/mesa/g0_phase0_oracle.py` | Ratified G0 union-find oracle, grow/audit, offset banks, and C1-C3 perturbations |
| `dev/nisb/scripts/mesa/g1_banis_feasibility.py` | G1 events, proposal recall, fragment matching, oracle upper bound, and repaired/control realizer |
| `dev/nisb/scripts/mesa/DESIGN.md` | Codebase-grounded MESA-EM model, target, decode, experiment, gate, and risk contract |
| `.agent/features/nisb_base_60-80_plan/artifacts/DESIGN.md` | Byte-identical design mirror for CCC review |
| `.agent/features/nisb_base_60-80_plan/artifacts/code_v0.md` | This implementation and verification handoff |

## Git Baseline

run_start_ref: e8844b3da0f0a992431c901e7e2034486e7a678b
current_head: e8844b3da0f0a992431c901e7e2034486e7a678b

The tracked unstaged diff, staged diff, and visible `git status --short` are byte-identical to the
run-start snapshots. The five implementation/design deliverables plus this artifact are untracked or
ignored research files. No commit or index operation was performed.

## Verification

Environment for every Python check:

```bash
source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc
export PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR=/tmp MPLCONFIGDIR=/tmp
cd /projects/weilab/weidf/lib/pytorch_connectomics
```

Actual fast-check results:

- Sanity A cached LUT: cc0.66 `0.601431470`; cc0.75 `0.545357139` (both pass).
- Sanity B center decode: `BASE_NERL=0.835517282`, `4197` instances, `142.9 s` including all
  harness work (passes `0.836 +/- 0.003`).
- Transform: `491,235` painted voxels; `100.000000%` inside `seg.h5>0` (passes >=99%).
- 256^3 G0 smoke, origin `[475,658,31]`, B0 only: `17,159` nodes, `18,113` SAME edges,
  `149` union-find fragments, NERL `0.489470369`, fragments/GT `2.921568627`, retained physical
  length `99.799575%`, false merges `0`, cross-GT unions `0`, foreign-corridor rejects `0`,
  runtime `83.2 s`.
- 256^3 G1 smoke on the same crop: local base NERL `0.771499430`, `346` instances, `2` break
  events, `224` tips. The generator examined `976` <=750-nm tip pairs (`914` cross-fragment),
  `38` passed the distance limit, and `9` passed agreement/radius and exercised the Hermite
  curvature calculation. The pinned curvature gate accepted `0`; therefore raw/capped/matched
  proposal counts are `0/0/0`, covered events `0`, and raw max candidates/endpoint `0`. Runtime
  `26.9 s`. This is an honest smoke finding, not a full-G1 verdict.
- Contract unit smoke: bank cardinalities 13/16/24/62, runner-up graph margin across a seed
  articulation, and multi-anchor C2 visitation all passed.
- Black formatter equivalence (`black.format_file_contents`, target py311), `isort
  --check-only`, and focused `flake8 --max-line-length=100` passed on the three Python files.
- `cmp` confirms the two `DESIGN.md` files are identical. No smoke run created a cache, result,
  `__pycache__`, or any other repository file.

The coordinator should launch the complete gates after approval with this exact background command:

```bash
cd /projects/weilab/weidf/lib/pytorch_connectomics
nohup bash -lc '
set -euo pipefail
source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc
cd /projects/weilab/weidf/lib/pytorch_connectomics
export PYTHONPATH=.
export PYTHONDONTWRITEBYTECODE=1
export NUMBA_CACHE_DIR=/tmp
export MPLCONFIGDIR=/tmp
python dev/nisb/scripts/mesa/common.py --sanity
python dev/nisb/scripts/mesa/g0_phase0_oracle.py \
  --full \
  --sdt-workers 1 \
  --sdt-cache dev/nisb/scripts/mesa/gt_sdt_center.h5 \
  --results dev/nisb/scripts/mesa/results_g0.md
python dev/nisb/scripts/mesa/g1_banis_feasibility.py \
  --full \
  --results dev/nisb/scripts/mesa/results_g1.md
test -s dev/nisb/scripts/mesa/results_g0.md
test -s dev/nisb/scripts/mesa/results_g1.md
' > /tmp/mesa_center_full.log 2>&1 &
echo $!
```

Monitor without starting another run:

```bash
tail -f /tmp/mesa_center_full.log
```

The `set -e` chain enforces the blocking order: failed harness sanity stops G0; failed G0 stops G1.
The coordinator must validate the two non-empty result files and process exit before folding their
full numbers into the go/no-go.

## Review Focus

- Confirm G0 decode consumes only the GT-derived edge set, never instance IDs, and that the
  scorer-visible false-merge audit is the correct 626-skeleton universe.
- Check C1 degree/tip/face eligibility, all-strand C2 anchors with independent visitation, C3 arc
  sampling, and physical retained-length denominator.
- Verify addendum-A equations and thresholds, break-cluster coverage, top-8-at-either-endpoint, and
  fragment-level degree-1 matching.
- Verify alternate identities can traverse the complete RAG to supply a true runner-up cost while
  final seed identities remain fixed, and that repaired/control post-grow audits gate the outputs.
- Confirm the exact addendum-D target definitions and the explicit G0/learnability boundary in both
  identical design files.

## Risks and Unknowns

- The complete center G0/G1 gates were deliberately not run. `gt_sdt_center.h5` and
  `results_g{0,1}.md` do not exist yet; all final pass/kill/go conclusions remain pending.
- The selected smoke is a clipped coverage-limited graph, so its NERL and fragments/GT are execution
  diagnostics, not Phase-0 thresholds.
- No smoke candidate passed the strict Hermite curvature gate. Nine candidates reached that exact
  calculation, proving the path executes, but the full generator may still fail its recall gate. Do
  not relax the pinned equation or infer the full verdict from one crop.
- Full GT-SDT and fragment-face aggregation are memory-intensive. The command uses one SDT worker to
  avoid swallowed per-instance OOMs; output postconditions reject empty/corrupt SDT before caching.
- The center chunk may have less than 0.02 full-oracle split headroom. In that case recovered
  fraction is reported but not gated, and `tile_0_0_2` is required before trusting G1.
- Geometry-only matching is expected to have a precision wall; that number is diagnostic and must
  not be mistaken for the GT-confirmed upper bound or repaired realizer.

## Changes Since Previous Code Version

Initial implementation.

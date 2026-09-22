# Plan v0

## Summary

Operationalize the MESA-EM deep-research design (reply in
`deepresearch_skeleton_seglink_0714.md`) for the NISB `base_banis+` 0.60→0.80 NERL
problem in two deliverables, both cheap and both required BEFORE any head training:

1. **A codebase-grounded design doc** (`DESIGN.md`) mapping MESA-EM's signed medial
   edge-graph → link → grow → fill pipeline onto the real `dev/nisb/` workspace,
   naming exact channels, GT-derivable targets, decode, and which existing scripts
   are reused vs. new.
2. **Two implemented + actually-run cheap gates on the standard center chunk**, with
   real numbers reported against the reply's own kill criteria:
   - **G0 — Phase-0 oracle plumbing (CPU, GT channels, no training):** does the
     representation+decode (medial nodes + SAME/MUTEX/skip edges → edge-graph
     union-find → grow on GT-SDT) reconstruct instances merge-safely *given correct
     edges*? This is the go/no-go for the whole representation.
   - **G1 — banis+ small-chunk feasibility (no retrain):** using the *existing*
     banis+ affinity (no new heads), measure geometric candidate-recall of the
     branch-merge-oracle bridge set and the merge-safe supervoxel + geodesic-grow
     realizer. Does the linker's generator + realizer have enough headroom to be
     worth training the heads?

The design is built on lesson 20: perfect SDT + GT connected per-instance seeds →
grow ≈0.994, but thresholding a scalar shatters (no identity). G0 tests exactly the
missing object — a connected merge-safe centerline **graph** — in isolation from any
learned prediction. G1 tests whether banis+ already supplies enough for the generator
and realizer. **No GPU training and no full-volume run is launched in this CCC run.**

## Scope

**In scope (this run):**
- `DESIGN.md`: MESA-EM mapped to `dev/nisb/` (channels, GT targets, reused scripts,
  forward experiment ladder Phases 0–3, honest gain forecast off 0.627).
- G0: Phase-0 GT-channel plumbing decode + audit on the center chunk, under the
  reply's 4 node conditions (intact / 1 voxel deleted / 1 Z-section deleted / 2–5-node
  crumb deleted) and offset-set sweep (O1 / O1+ / +one-section / full |o|∞=2 shell).
- G1: geometric candidate-recall + realizer feasibility on banis+ affinity, center chunk.
- `results_g0.md` / `results_g1.md`: measured numbers vs. reply Phase-0/Phase-1 gates
  + a single go/no-go recommendation.

**Out of scope (explicitly deferred — the "whole experiment"):**
- Training any new head (medialness / edges / veto / PairNet / BiLOQ). No
  `main.py` training, no `sbatch`, no GPU fit.
- Full-volume seed101 decode.
- Learned validators (PairNet, BiLOQ), empirical-risk threshold calibration.
- Fill Phase 3 beyond the realizer smoke inside G0/G1 (design-only here).

A gate failure ends the branch with a written finding (loop.md cheap-gate law); do
**not** escalate to a full run to "make sure."

## Proposed Changes

All new files are untracked research code under `dev/nisb/scripts/mesa/` (same
convention as sibling `dev/nisb/scripts/*.py`); nothing tracked is edited.

### Step 0 — Harness sanity (mandatory first, reused by G0+G1)

Reuse the node-LUT scorer, do not re-derive NERL:
`from connectomics.metrics.nerl import import_em_erl` →
`ERLGraph, compute_erl_score = import_em_erl()`; score via
`compute_erl_score(graph, lut.astype(uint64), mask_segment_id=None, merge_threshold=1)`,
`NERL = pred_erl/gt_erl`. Prefer extending `ec_merge/ec_common.py`
(`UnionFind`, `load_lut_cache`, `score_lut`, `load_roi_seg/affinities`) and
`scripts/banis+_oracle_merge.py` (`build_oracle_branch_merge_mapping`,
`compute_pred_bboxes`, intersection scan) rather than re-rolling.

- **Sanity A (whole-vol invariant):** reproduce cc066 = **0.601431** and branch-merge
  oracle = **0.772069** via the existing LUT cache. If either drifts, stop — the
  harness is mis-wired.
- **Sanity B (center-chunk local):** score cc3d@0.66 on the center chunk against its
  **local** skeleton `data/center_chunk/seg.erlgraph.npz` (NOT the whole-vol LUT — this
  is the lesson_testcrop_preflight gotcha) and confirm local NERL ≈ **0.833**.
  All G0/G1 center-chunk numbers use this local skeleton and are RELATIVE (a crop
  scores higher than whole-vol; do not compare crop numbers to 0.627/0.752 directly).

### G0 — Phase-0 oracle plumbing (CPU, GT channels)

Extend the existing `scripts/centerline_graph_decode_gt.py` (121 L — the direct
ancestor of this design). Pipeline:
1. Load center-chunk GT `seg.h5`; obtain per-instance GT skeleton (from the erlgraph
   node coords / or kimimaro on cc3d(GT)); build GT-SDT (reuse
   `scripts/make_gt_sdt_region.py`; reuse cached `x2diag/gt_sdt_region_ero{0,1,2}.h5`
   for the small smoke region to avoid recomputing EDT).
2. Build **medial-band nodes** (graded medialness core) and **SAME/MUTEX short edges**
   (13 half-space offsets) + **skip edges** (offset sweep) using the reply's §1
   local-path label rule (SAME iff both endpoints in one instance's medial band, nearest
   skeleton points joined by a short local arc, corridor crosses no foreign medial band;
   MUTEX iff different instances or corridor pierces a foreign band). **Strip instance
   IDs before decode** — identity must come only from edges.
3. **Union-find over the edge graph** (attractive SAME, forbidden MUTEX), rasterize
   accepted skip paths, CC → seed IDs.
4. **Grow** seeds on GT-SDT (seeded watershed / geodesic) → instance seg.
5. **Audit vs GT after decode:** fragments/GT, count of graph components spanning >1 GT
   id (false merges), retained skeleton length, gap-closure rate, and local NERL via the
   harness.

Run under the 4 node conditions and the offset-set sweep; pick the smallest offset set
meeting the gate. Smoke on a small region first (reuse an `x2diag` crop), then the full
center chunk.

### G1 — banis+ small-chunk feasibility (no retrain)

New `scripts/mesa/g1_banis_feasibility.py`:
1. **Locate banis+ affinity for the center chunk** (crop of the whole-vol prediction, or
   the path used by `scripts/center_chunk_decode_sweep.py`). If genuinely absent, run a
   single small banis+ **inference** (not training) on the center chunk — this is the
   user's explicit "feasibility check with banis+ model on smaller chunks." Prefer the
   cached crop; document whichever is used.
2. Build cc3d@0.66 base fragments (reproduce Sanity B ≈0.833).
3. Compute the **branch-merge-oracle bridge set** on the center chunk (reuse
   `banis+_oracle_merge.py`): the same-GT fragment pairs the oracle would join = the
   heal prize set, with each bridge's marginal ΔNERL.
4. Detect **endpoints/tips** on cc3d fragments (reuse `scripts/ec_endpoint_bridge.py`
   tip detection).
5. **Geometric candidate generation** with the reply §2.2 proposal gates
   (radius-aware distance, tangent from local fragment skeleton ≥0.65, radius ratio,
   Hermite curvature, ≤8 candidates/endpoint, ≤3 protected-trunk candidates). Learned
   tangent/medialness are NOT required for the *generator* recall test.
6. **Measure:** count recall and ΔNERL-weighted recall of the oracle bridge set;
   candidates/endpoint and candidates/recovered-bridge; and a realizer smoke —
   merge-safe supervoxel (aff_r1 threshold with 0 cross-GT SV on GT) + geodesic grow
   (reply §3 cost) reproducing the base merge-safely.

### DESIGN.md

Codex writes `scripts/mesa/DESIGN.md` from the reply, grounded to real paths:
channel table (43 dense + sparse skip on MedNeXt-L warm start), GT-derived targets
from `seg` + cached skeleton, the link/grow/fill decode, the Phase 0–3 ladder with the
reply's gate numbers, and the honest gain forecast (central 0.792, stretch 0.812; all
deltas off 0.627). Mirror it into this run folder's `artifacts/`.

## Files and Areas

| File | Purpose |
|---|---|
| `dev/nisb/scripts/mesa/DESIGN.md` | MESA-EM mapped to `dev/nisb/`; experiment ladder; forecast |
| `dev/nisb/scripts/mesa/common.py` | shared: center-chunk load, GT-SDT/medial/edge builders, harness scoring wrapper |
| `dev/nisb/scripts/mesa/g0_phase0_oracle.py` | G0 GT-channel plumbing decode + audit (extends `centerline_graph_decode_gt.py`) |
| `dev/nisb/scripts/mesa/g1_banis_feasibility.py` | G1 candidate-recall + realizer feasibility on banis+ affinity |
| `dev/nisb/scripts/mesa/results_g0.md` | G0 measured numbers + verdict vs Phase-0 gate |
| `dev/nisb/scripts/mesa/results_g1.md` | G1 measured numbers + verdict vs Phase-1 gate + overall go/no-go |
| `.agent/features/nisb_base_60-80_plan/artifacts/DESIGN.md` | mirror of the design doc |

Read-only reuse (do not modify): `ec_merge/ec_common.py`, `scripts/banis+_oracle_merge.py`,
`scripts/banis+_oracle_fill.py`, `scripts/centerline_graph_decode_gt.py`,
`scripts/make_gt_sdt_region.py`, `scripts/ec_endpoint_bridge.py`,
`scripts/center_chunk_decode_sweep.py`, `data/center_chunk/*`, `x2diag/gt_sdt_region_ero*.h5`.

## Verification Plan

Every number is verified by OUTPUT, not code reading. Report each in the results md.

1. **Harness sanity (blocking):** cc066 = 0.601431 and branch-merge oracle = 0.772069
   (whole-vol); center-chunk cc3d@0.66 local NERL ≈ 0.833. Fail ⇒ stop and fix harness.
2. **G0 pass (reply §12 Phase 0), on the center chunk:**
   - 0 graph components spanning >1 GT id (0 false merges);
   - 0 accepted paths through a foreign GT medial band;
   - fragments/GT ≤ 1.02; retained GT skeleton length ≥ 99.5%;
   - ≥95% of one-voxel gaps closed; ≥90% of one-section gaps closed;
   - GT-SDT grow local NERL ≥ 0.990 (target ≈0.994).
   Report the smallest offset set meeting this and the per-condition breakdown.
   **G0 kill:** any cross-GT union, decode needs instance IDs (not edges), full offset
   bank can't reach ≥0.985, or fragments/GT stays >1.05.
3. **G1 pass (reply §12 Phase 1 analog, center chunk, recall is the scale-invariant gate):**
   - branch-bridge **count recall ≥ 80%**; **ΔNERL-weighted recall ≥ 90%**;
   - ≤8 proposals/endpoint; ≤6 proposals/recovered bridge;
   - candidate-oracle matching introduces **0 false merges**;
   - realizer smoke: merge-safe SV + geodesic grow reproduces base local NERL within
     the harness (no cross-GT SV on GT).
   Report local candidate-oracle NERL as secondary context (NOT compared to the
   whole-vol 0.752). **G1 kill:** weighted recall <80%, >12 candidates/endpoint after
   pruning, or recall needs global BFS/whole-volume tracing.
4. **Go/no-go:** a one-paragraph recommendation in `results_g1.md`: proceed to train
   the MESA-EM heads (and which channels are highest-value), or which sub-part
   (representation vs. generator vs. realizer) is the wall to fix first.

Runtime guard (user's "surface silent failure loudly"): G0/G1 first run the small-region
smoke; if the full center-chunk pass is heavy (EDT/watershed on 450 M voxels), the coder
runs it in the background with a stated result path + liveness log and reports the smoke
numbers immediately, rather than blocking or silently hanging. No step waits on GPU
training.

## Risks and Questions

- **banis+ affinity on the center chunk may not be pre-cropped.** Mitigation: G1 step 1
  locates the cached crop first (`center_chunk_decode_sweep.py` path); only if absent,
  a single small inference. If neither is quickly available, G1 reports the generator
  recall on cc3d(GT-derived) fragments as a lower-bound surrogate and flags the gap.
- **Center-chunk compute is non-trivial** (450 M voxels). Mitigation: smoke on the
  `x2diag` GT-SDT crop first; reuse cached SDT; background the full run with liveness.
- **Local-vs-whole-vol scale trap** (lesson_testcrop_preflight): crop NERL runs higher;
  the plan uses recall (scale-invariant) as the G1 gate and forbids comparing crop NERL
  to 0.627/0.752. Sanity B guards this.
- **G0 could pass with GT-derived edges yet be infeasible to LEARN** — G0 only proves
  the decode; it does not prove the heads are learnable. This is stated in `DESIGN.md`
  as the boundary between G0 (decode feasibility) and the deferred Phase-2 (learnability).
- **Cross-repo/untracked review surface:** deliverables are untracked; code review
  embeds file contents + the `git status` delta rather than a tracked diff (noted in
  run.md). Coder must not edit tracked framework code or commit.
- **Open question for the reviewer:** is the center chunk (783 GT, ceiling ~0.90) a
  representative enough gate for G0/G1, or should a second harder chunk (e.g.
  split-dominated `tile_0_0_2`) be added before trusting the go/no-go? Plan currently
  uses the standard center chunk per lesson_standard_smallscale; a second chunk is a
  cheap add if the reviewer deems the single-chunk gate too weak.

## Changes Since Previous Plan Version

Initial plan.

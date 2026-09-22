# Task — NISB base 0.60→0.80 NERL: MESA-EM design + oracle/feasibility gate

## One-line

Turn the GPT deep-research reply (MESA-EM design) into a concrete, codebase-grounded
model + experiment plan for the NISB `base_banis+` run, and **implement + run the
cheap oracle plumbing and a banis+ small-chunk feasibility check BEFORE any full
training experiment is launched.**

## Source design (read first, verbatim)

`/projects/weilab/pytc-agent/projects/2026_nisb_base/research_plan/deepresearch_skeleton_seglink_0714.md`

- Lines 1–265 = the prompt we sent (problem framing, numbers, invariants I1–I5,
  falsified-list, partial positives).
- Lines 266+ = the reply. The design is **MESA-EM** (Mutexed Edge-Skeleton Assembly
  with Bidirectional Local Object Queries): a signed medial **edge graph** whose
  connected components ARE the instance IDs (short SAME/MUTEX edges + sparse skip
  edges + medialness + radius + tangent + 3 veto channels on the warm-started
  MedNeXt-L), then conservative trunk union → sparse candidate repair → PairNet +
  BiLOQ validators → degree-constrained non-cascading local transactions →
  supervoxel grow → bounded abstaining geodesic fill.
- The reply's own **§12 Phased build and kill gates** (Phase 0 CPU plumbing on GT
  channels; Phase 1 candidate-recall gate; Phase 2 predicted heads + linker
  ablations; Phase 3 grow+fill) is the experiment skeleton to operationalize.

## What this CCC run must produce

1. **A design doc** that maps MESA-EM onto the *actual* NISB codebase and data —
   concrete channels, GT-derivable targets, decode, and which existing scripts to
   reuse vs. write. This is "design the model with experiments."
2. **Implemented + RUN cheap gates**, in priority order (the user's explicit ask:
   "try oracle and feasibility check with banis+ model on smaller chunks before
   kicking off the whole experiment"):
   - **G0 — Phase-0 oracle plumbing (CPU, GT channels, no training).** On the
     standard small-scale test chunk, build medial-band nodes + SAME/MUTEX (+skip)
     edge labels from the GT skeleton, union-find over the EDGE graph (instance IDs
     stripped), grow on GT-SDT, and score. This tests whether the *representation +
     decode* can reconstruct instances merge-safely GIVEN correct edges. Reply's
     Phase-0 success target: fragments/GT ≤1.02, ≥95% one-voxel gaps and ≥90%
     one-section gaps closed, GT-SDT grow NERL ≥0.990. This is the go/no-go for the
     whole representation.
   - **G1 — banis+ small-chunk feasibility (no retrain).** Using the EXISTING
     banis+ prediction/affinity on the same chunk (no new heads trained), test how
     far the pieces MESA-EM depends on already exist or can be surrogate-measured:
     e.g. candidate-recall of the branch-merge-oracle bridge set from banis+
     endpoints (reply Phase-1 gate: ≥80% count / ≥90% ΔNERL-weighted recall), and
     whether banis+ affinity supports the merge-safe supervoxel + geodesic-grow
     realizer. Report real numbers vs. the reply's gates.
3. **A go/no-go readout** with the actual measured numbers against the reply's
   Phase-0 and Phase-1 kill criteria, plus the recommended next step (proceed to
   train the new heads, or which sub-part to fix first). Do NOT launch any GPU
   training run in this CCC run.

## Anchors, invariants, and the falsified list (do not violate — from the reply/state.yaml)

- Baseline (realized SOTA): ch0-1-2 affinity + cc3d@0.66 = **NERL 0.601431**;
  realized GT-free SOTA **0.627**; heal oracle **0.772**; heal+fill oracle **0.921**.
  0.80 needs BOTH heal (→0.772) and fill (stacked). Reply's central forecast 0.792,
  stretch 0.812; deltas quoted **off 0.627**, never fractions of a ceiling.
- **I1** no realizer over a frozen label map beats 0.601 (new links must be
  image-conditioned/learned). **I2** NERL is length-weighted (score by recovered
  backbone length; weight candidates by ~2·l_A·l_B). **I3** evidence absent not
  under-sampled (object-context, not finer input). **I4** VOI/ARAND ≠ NERL
  (no waterz/global multicut/agglomeration decides identity). **I5** thin-split is
  object-context-bound (a new scalar field does NOT clear it; needs relational
  identity or changed FOV).
- Falsified — do NOT re-propose as the fix: MALIS; SDT/clDice/cbDice/soft-skeleton
  as the deciding objective; 2×-XY; banis2 occupancy targets; waterz/ABISS/
  low-threshold salvage; naive EDT fill; global multicut/BFS-flood; whole-volume
  root tracers; methods needing perfect instance seeds.
- Diagnostic the design is built on (lesson 20): perfect SDT + GT connected
  per-instance seeds → grow ≈0.994; thresholding a scalar shatters (no identity).
  The missing object is a connected merge-safe centerline GRAPH, not a better field.

## Grounded paths (verified this session)

- **Operative repo:** `/projects/weilab/weidf/lib/pytorch_connectomics` (this is
  where `.agent/features/` and the runnable `dev/nisb/` workspace both live).
- **Runnable NISB workspace + harness + data:** `<repo>/dev/nisb/`
  - standard small chunk data: `dev/nisb/data/center_chunk/{img.h5, seg.h5, seg.erlgraph.npz}`
    (tile_1_1_1, origin [1000,1000,450], 1000×1000×450, 783 GT; cc3d@0.66 local NERL
    ~0.833, coverage-limited ceiling ~0.90 — see lesson_standard_smallscale).
  - prior art to reuse: `dev/nisb/scripts/centerline_graph_decode_gt.py` (the exact
    Phase-0 GT node+edge union-find decode this design descends from),
    `make_gt_sdt_region.py`, `ec_endpoint_bridge.py` (tip detection + oracle bridge
    set), `extract_center_chunk.py`; GT-SDT crops `dev/nisb/x2diag/gt_sdt_region_ero{0,1,2}.h5`.
  - node-LUT / NERL harness: catalogued in `dev/nisb/../` `scripts.md §0` (score must
    reproduce 0.66→0.601431 and 0.75→0.545 exactly). Locate the exact scorer as
    step 0 of implementation and reuse it — do not re-derive NERL.
- **Knowledge base (docs, read-only context):** `/projects/weilab/pytc-agent/projects/2026_nisb_base/`
  (`state.yaml`, `lessons/lessons.md`, `loop.md`, `scripts.md`).
- **New code location:** `<repo>/dev/nisb/scripts/mesa/` (untracked research scripts,
  same convention as the sibling `dev/nisb/scripts/*.py`). Design doc mirrored into
  this run folder's `artifacts/`.

## Constraints / discipline (loop.md)

- **Cheap-gate law:** no GPU/training earns time until a <1% surrogate passes. This
  run IS that surrogate (G0/G1). If a gate fails, write the finding and stop; do not
  "run the full version to make sure."
- Reuse the existing node-LUT scorer and center-chunk skeleton; a crop NERL needs a
  LOCAL skeleton (already provided as `seg.erlgraph.npz`) — do not slice the whole-
  volume skeleton.
- Verify by OUTPUT (measured fragments/GT, NERL, recall, merge count), not by code
  reading. Surface any silent failure/hang loudly with the result path.
- Do NOT commit; do NOT edit tracked framework code. All deliverables are new
  untracked files under `dev/nisb/` + this run folder.

## Success criterion for the CCC run

The design doc is concrete and codebase-grounded; G0 and G1 are implemented and
**actually run on the center chunk with real numbers reported** against the reply's
Phase-0/Phase-1 gates; and there is a clear, evidence-backed go/no-go recommendation
for whether (and how) to proceed to training the MESA-EM heads. No full-volume or
GPU training run is launched.

# Plan v2

## Summary

Unchanged goal and scope from v1 — a codebase-grounded MESA-EM design doc plus two
cheap gates (G0 = CPU Phase-0 oracle plumbing on GT channels; G1 = banis+ no-retrain
feasibility) run on the standard center chunk with real numbers vs. the reply's kill
criteria; no GPU training, no full-volume run. v2 pins every remaining ambiguity to a
verified API/asset so the coder can implement without a judgment call and every gate
yields a trustworthy pass/fail. Design rationale is lesson 20 (connected per-instance
seeds make the field near-perfect; the missing object is a merge-safe centerline
**graph**). G0 tests the decode given GT-correct edges; G1 tests whether banis+ already
supplies enough for the candidate **generator** and body **realizer** to justify
training the heads. **G1 is a feasibility surrogate, not a Phase-1 certification**
(that needs trained heads and is deferred).

## Scope

**In:** `DESIGN.md`; G0 (Phase-0 GT-channel decode + graph-level perturbation audit,
center chunk); G1 (geometric candidate-recall + realizer feasibility on the cached
banis+ affinity crop, center chunk); `results_g0.md` / `results_g1.md` with measured
numbers + one go/no-go. **Out (deferred "whole experiment"):** training any head
(medialness/edges/veto/PairNet/BiLOQ); `main.py` train / `sbatch` / GPU fit;
full-volume decode; learned validators + empirical-risk calibration; predicted-SDT grow
(G0 uses GT-SDT only); fill Phase 3 beyond the realizer control; a second harder chunk
(recommended in `DESIGN.md` for later, not run now).

## Verified assets (bound; used verbatim)

- **Affinity (banis+ seed42):** `outputs/nisb_base_banis_v3_erosion2/20260508_224029/test_step=00200000/seed101/raw_x1_ch0-1-2.h5`,
  HDF5 key `main`, shape `(3, 3000, 3000, 1350)`, dtype float16, axes `(C, X, Y, Z)`.
  **Center-chunk crop** = `main[:, 1000:2000, 1000:2000, 450:900]` → `(3, 1000, 1000, 450)`,
  cast float32. Channel `c` = affinity to the +1 neighbor along array spatial axis `c`
  (0=X@9nm, 1=Y@9nm, 2=Z@20nm). Decode: `decode_affinity_cc(aff, threshold=0.66,
  backend="numba", edge_offset=0)`.
- **GT seg:** `dev/nisb/data/center_chunk/seg.h5` (center-chunk GT, array order `(X,Y,Z)`
  = `(1000,1000,450)`; assert against the affinity crop shape).
- **Local skeleton graph:** `dev/nisb/data/center_chunk/seg.erlgraph.npz` — 626 skeletons,
  491,235 nodes; `node_coords_zyx` are **physical nm** (max ≈ [8991,8991,8980]),
  `node_skeleton_index` ∈ [0,625], skeleton graph edges in `edge_u/edge_v/edge_ptr`,
  `edge_len` physical.
- **Canonical nm→voxel mapping (reuse; do NOT hand-roll):** load via
  `connectomics.metrics.nerl.load_nerl_graph(str(SKEL), None, resolution=[9,9,20])` →
  `erl_graph.get_nodes_position([9,9,20])` returns per-node voxel indices in the seg's
  array order. This is exactly what `compute_nerl_score_details` uses, so a skeleton
  volume painted from it agrees with the scorer by construction.
- **Scorer (center-chunk):** `compute_nerl_score_details(seg, str(SKEL),
  resolution=[9,9,20], merge_threshold=1, num_workers=16)` → `.nerl`.
- **Recorded anchor:** cc3d@0.66 on the affinity crop = **0.836** (2346 inst),
  `exp/center_chunk_decode_sweep.md`.

## Proposed Changes

New untracked files under `dev/nisb/scripts/mesa/`; they import/adapt read-only prior
art, modify no tracked file, and commit nothing.

### Step 0 — Harness sanity (blocking; shared)

- **A (whole-vol):** cc3d@0.66 = 0.601431 ± 0.0005; cc3d@0.75 = 0.545 ± 0.001
  (via the whole-vol node-LUT harness). 0.772069 supplemental, non-blocking.
- **B (center-chunk local):** cc3d@0.66 on the affinity crop = 0.836 ± 0.003
  (bind proves the affinity crop, channel-axis convention, and scorer end-to-end).
- **Transform sanity:** paint `skel` from `get_nodes_position` and assert ≥99% of
  painted skeleton voxels fall inside `seg.h5 > 0`. <99% ⇒ coordinate/axis error ⇒ STOP.
Any A/B/transform failure ⇒ stop with a written finding; do not run G0/G1.

### G0 — Phase-0 oracle plumbing (CPU, GT channels), fully specified

Adapts `centerline_graph_decode_gt.py` (`components()`, `UF`, `undirected_offsets`) from
the retired crumb to the center chunk. **Identity comes only from same-instance edges;
instance IDs are never read during decode.**

1. **Nodes = skeleton voxels** (no medial band for the GT oracle — the band is a
   learned-head concept, deferred to DESIGN.md/Phase-2). Paint `skel` (int32,
   `(1000,1000,450)`): for each node, `skel[get_nodes_position(node)] =
   node_skeleton_index + 1`.
2. **GT-SDT** for grow: build a signed distance from `seg.h5` reusing
   `make_gt_sdt_region.py`'s ero1 convention; cache `mesa/gt_sdt_center.h5`; assert its
   shape == the affinity crop.
3. **Edges = same-instance offset union-find** (`edge(v,v+o)=1 iff skel[v]>0 &
   skel[v+o]>0 & skel[v]==skel[v+o]`). A cross-instance union is impossible by the `==`
   test (built-in MUTEX); assert 0 cross-GT unions. **Foreign-corridor rule (adds the
   reply's SAME-with-clear-corridor semantics, esp. for skip edges):** before uniting an
   edge whose Chebyshev length > 1, walk the straight voxel corridor `u→u+o`; if any
   interior corridor voxel has `seg>0` with a GT id ≠ `seg[u]`, **reject the edge** (do
   not union) and increment a `foreign_corridor_rejected` counter. Union-find roots stay
   authoritative; no spatial connected-components ever create a union.
4. **Offset banks** (array order `(X,Y,Z)`, Z = anisotropic axis 2; half-space dedup via
   `undirected_offsets`; report each bank's exact deduped tuple list + count):
   - `B0` = 26-neighborhood half-space (**13**);
   - `B1` = B0 + axis-2 skips {(2,0,0),(0,2,0),(0,0,2)} (**16**);
   - `B2` = B1 + one-Z-section skips {(dx,dy,2): dx,dy∈{-1,0,1}} deduped (**24**);
   - `B3` = full `|o|∞=2` half-shell (**49**).
   Radius-3/4 axis skips may be printed as diagnostics only. Choose the smallest bank
   meeting the gate.
5. **Grow:** `watershed(-gt_sdt, markers=components, mask=(gt_sdt > -0.3))` over the fg
   mask only (float32). Markers ARE the union-find components; watershed only fills.
6. **Audit** on the local skeleton: fragments/GT and retained length computed with
   physical `edge_len` (I2-respecting); count of grown labels spanning >1 GT id (false
   merges, must be 0); `foreign_corridor_rejected`; local NERL via
   `compute_nerl_score_details`.

**Perturbation protocol (graph-level, reproducible, cheap).** Closure is measured on the
UNION-FIND GRAPH (do the two sides share a component?), NOT by re-growing the volume — so
perturbations cost O(nodes·offsets), and a full watershed+NERL runs only for `C0` (per
bank) and the final chosen bank. `numpy.random.default_rng(101)`, **reset per condition**:
- **C0 intact** — full skeleton; the primary NERL/fragments gate.
- **C1 one interior voxel deleted** — eligible skeleton: ≥5 nodes, has a non-endpoint
  interior node ≥2 voxels from any tip and ≥3 voxels from the crop face. Delete that node;
  the two **side anchors** = its two skeleton-graph neighbors (`edge_u/edge_v`). Closed =
  both anchors in one component AND no new cross-GT union. Sample min(N_elig,300); report
  denominator. Gate: ≥95%.
- **C2 one interior Z-section deleted** — eligible skeleton spans ≥3 Z-sections, not
  touching Z faces; deterministically delete all its nodes in the **median interior
  Z-section** of its z-range. Anchors = the nearest surviving node on each z-side. Closed
  as in C1. Gate: ≥90%.
- **C3 2–5-node crumb deleted** — delete a contiguous arc of length ~U{2,3,4,5}
  (`rng(101)`); anchors = flanking surviving nodes. Reported as a diagnostic (no reply
  threshold), with the closure curve vs arc length.

**G0 pass (reply §12 Phase 0):** 0 grown labels spanning >1 GT id; 0 cross-GT unions
(asserted); fragments/GT ≤ 1.02; retained skeleton length ≥ 99.5%; C1 ≥95%; C2 ≥90%;
C0 GT-SDT grow local NERL ≥ 0.990 (target ≈0.994). Report the smallest passing bank +
per-condition table + `foreign_corridor_rejected`.
**G0 kill:** any cross-GT union; decode needs instance IDs; best bank C0 NERL < 0.985
(GT-SDT grow); or fragments/GT > 1.05.

### G1 — banis+ feasibility (no retrain); one bound map; explicit surrogate

`BASE_SEG = decode_affinity_cc(aff_crop, 0.66, backend="numba", edge_offset=0)` (the
Sanity-B seg; local NERL 0.836, 2346 inst). The center chunk has ≈0 baseline false
merges, so no intersection-cut firewall is applied on the crop (stated).

1. **Local oracle events + ΔNERL** (adapts `ec_endpoint_bridge.py` local logic; the
   whole-vol `banis+_oracle_merge.py` is NOT used on the crop). Map each erlgraph node →
   its `BASE_SEG` fragment (via `get_nodes_position`). An **oracle event** = an unordered
   fragment pair `(f_a,f_b)` such that some skeleton edge (`edge_u,edge_v`) has its two
   endpoints in `f_a` and `f_b` (same GT skeleton, skeleton-adjacent). The event's **break
   location** = the midpoint voxel of that crossing edge. Exclude events whose break is <2
   voxels from a crop face (denominator exclusion). **ΔNERL(event)** = incremental
   node-LUT gain from merging `f_a,f_b` computed closed-form on the affected skeleton's
   node partition (length-weighted `Σ pred·gt/Σ gt²`; do NOT full-re-score per event).
   Full-oracle = merge all same-skeleton fragments.
2. **Endpoints/tips** = `ec_endpoint_bridge._extract_tips_fast(BASE_SEG)` (2 tips +
   physical outward tangents; seconds; no kimimaro).
3. **Geometric candidate generation** (reply §2.2 proposal gates, physical units with
   res [9,9,20]): distance ≤ radius-aware limit; tangent agreement ≥0.65; radius ratio
   ≤3.0; Hermite curvature radius ≥1.25·r_max. Produce ALL proposals (no cap).
4. **Recall** — a candidate **covers** an event iff it links `f_a,f_b` AND each of its two
   tips is within `R_end = 45 nm` (≈5 XY voxels) of the break location. Report:
   - raw post-pruning candidates/endpoint (mean, p95, **max**);
   - **count recall** and **ΔNERL-weighted recall**, computed on the **capped** set;
   - cap rule: rank candidates per endpoint by (tangent agreement desc, distance asc,
     partner-id asc); a proposal is retained if it is in the top-8 at **either** endpoint;
   - proposals/recovered-bridge on the capped set.
5. **Realizer feasibility (non-trivial; sparse seeds + coverage gate).** Seeds =
   **sparse centerlines** (the skeleton voxels per identity), NOT full fragment bodies.
   Grow assigns **every fg voxel** (`gt_sdt`-fg proxy or aff-fg) to the nearest identity by
   geodesic cost `c(p,q) = -log(aff_r1(p,q)+1e-6)` (aff_r1 = the axis-matched channel of
   the p→q step), competition margin `log 9`; require ≥95% of fg assigned (abstaining
   everywhere fails). Three identity sets, each realized and scored:
   - **full-oracle** (merge all same-skeleton fragments);
   - **capped-candidate-recovered** (merge only pairs the capped candidate set covers) —
     the feasibility number;
   - **control** = BASE_SEG fragment identities, no repair.
   Control gate: grown NERL within **0.005** of BASE_SEG (0.836), **0 cross-GT**, **0
   multi-ID supervoxel assignments** (SV = the merge-safe `BASE_SEG` fragments; assert 0
   cross-GT SV on GT first).

**G1 pass (feasibility surrogate; scale-appropriate, predeclared):**
- ΔNERL-weighted recall (capped) ≥ 90%; count recall ≥ 80%;
- raw max candidates/endpoint ≤ 8 (9–12 = gray, >12 = kill);
- ≤6 proposals/recovered bridge; candidate matching adds 0 false merges;
- realizer control passes its three conditions;
- **recovered fraction of local oracle gain** =
  `(NERL(capped-recovered-realized) − 0.836) / (NERL(full-oracle-realized) − 0.836) ≥ 0.60`.
Absolute whole-vol gates (0.752/0.742) are explicitly NOT applied on the crop; G1 does
not certify Phase-1 (deferred to post-training).
**G1 kill:** ΔNERL-weighted recall < 80%; raw max candidates/endpoint > 12; recall needs
global BFS / whole-volume tracing; or realizer control fails (moves NERL >0.005 or any
cross-GT).

### DESIGN.md (indexed channel table pinned here)

`scripts/mesa/DESIGN.md` (mirrored to `artifacts/DESIGN.md`) expands each channel below
with its target formula from `seg` + cached skeleton, the SAME/MUTEX/DEFER edge encoding,
and the three veto targets. **Dense index map (0–42, sums to 43):**

| idx | channel | target |
|---|---|---|
| 0–8 | affinity (existing 9-ch banis+, r1+r10, ero2) | current targets (merge-safe anchor) |
| 9 | graded medialness | radius-normalized Gaussian to skeleton (reply §1) |
| 10 | signed log boundary distance | `sign(∈G)·log(1+d(∂G)/s_xy)` |
| 11–13 | tangent (3) | local skeleton principal direction, sign-invariant |
| 14–39 | short SAME/MUTEX edges (13 offsets × 2 logits) | per offset: SAME (even), MUTEX (odd), else DEFER/ignore |
| 40–42 | vetoes (3) | multiplicity, foreign-neurite, true-branch |

Plus a **sparse skip-edge head** (2 logits per queried edge over the `|o|∞=2` half-shell;
not in the dense 43). Also: the link→grow→fill decode mapped to `dev/nisb/`, the Phase 0–3
ladder with the reply's gate numbers, the honest forecast (central 0.792, stretch 0.812;
deltas off 0.627), and the `tile_0_0_2` second-chunk recommendation.

## Files and Areas

| File | Purpose |
|---|---|
| `dev/nisb/scripts/mesa/DESIGN.md` | MESA-EM mapped to `dev/nisb/`; indexed 43-ch table; Phase 0–3 ladder; forecast |
| `dev/nisb/scripts/mesa/common.py` | assets/loaders: affinity crop, GT seg, `load_nerl_graph`+`get_nodes_position` skeleton paint, GT-SDT, offset banks, both scorers, incremental-ΔNERL |
| `dev/nisb/scripts/mesa/g0_phase0_oracle.py` | G0 decode + graph-level perturbation audit |
| `dev/nisb/scripts/mesa/g1_banis_feasibility.py` | G1 recall + realizer feasibility |
| `dev/nisb/scripts/mesa/gt_sdt_center.h5` | cached center-chunk GT-SDT |
| `dev/nisb/scripts/mesa/results_g0.md` / `results_g1.md` | measured numbers + verdicts + go/no-go |
| `.agent/features/nisb_base_60-80_plan/artifacts/DESIGN.md` | mirror |

Read-only (import/adapt): `dev/nisb/waterz/per_chunk_nerl.py`, `connectomics/metrics/nerl.py`
(`load_nerl_graph`, `compute_nerl_score_details`), `connectomics/decoding/decoders/segmentation.py`
(`decode_affinity_cc`), `scripts/centerline_graph_decode_gt.py`, `scripts/make_gt_sdt_region.py`,
`scripts/ec_endpoint_bridge.py`, `scripts/center_chunk_decode_sweep.py`, `ec_merge/ec_common.py`,
`data/center_chunk/*`, `x2diag/gt_sdt_region_ero*.h5`, the affinity `raw_x1_ch0-1-2.h5`.

## Verification Plan

Every number verified by output, written to the results md.
1. **Sanity (blocking):** A (0.601431±0.0005, 0.545±0.001), B (0.836±0.003), transform
   (≥99% skel-in-seg). Fail ⇒ stop + finding.
2. **G0:** offset-bank sweep × C0–C3; pass = the six §"G0 pass" thresholds; report smallest
   bank + per-condition table + `foreign_corridor_rejected` + the 0-cross-GT assertion.
3. **G1:** recall (raw max + capped, count + ΔNERL-weighted), realizer control + full-oracle
   + capped-recovered; pass = the §"G1 pass" thresholds incl. recovered-fraction ≥0.60.
4. **Go/no-go** in `results_g1.md`, three outcomes: **go** only if EVERY G0+G1 success
   criterion passes → train the heads, naming highest-value channels; **gray zone** (some
   pass, none killed) → no-go, name the cheap refinement; **kill** → stop the branch on the
   failing gate. Always emit a result-location line even if a G0 failure blocks
   `results_g1.md` (write the reason into `results_g0.md`).

**Resource strategy (cheap-gate discipline):** paint `skel` once; perturbations are
graph-level (no re-grow); full `watershed`+`compute_nerl_score_details` runs only for C0
per bank + the chosen bank and the ≤5 G1 realized segs (BASE, full-oracle,
capped-recovered, control). Per-event ΔNERL uses the closed-form incremental node-LUT gain,
never a full re-score. Watershed/geodesic operate on the fg mask only; affinity kept
float16→float32 per-op; `(1000,1000,450)` int32/uint32 labels (~1.8 GB) fit in RAM. Both
gates first run a 256³ sub-crop smoke and print numbers immediately; the full center-chunk
pass, if backgrounded, is monitored to a successful exit + validated non-empty output
before results are finalized (smoke never substitutes for a completed gate; surface any
hang/crash with the result path).

## Risks and Questions

- **G0 proves the DECODE, not learnability** (GT-derived edges); Phase-2 learnability is
  deferred — stated as the G0↔Phase-2 boundary in `DESIGN.md`.
- **GT-SDT grow is an oracle input;** predicted-SDT grow deferred (no predicted-SDT claim).
- **Channel-axis convention** of the affinity is validated end-to-end by reproducing
  0.836 (Sanity B) and by the geodesic realizer control; a wrong convention fails Sanity B.
- **Local ≠ whole-vol scale** (crop peaks @0.55, base @0.66=0.836); G1 uses recall +
  recovered-fraction (scale-appropriate), never the whole-vol 0.752/0.627, and is labeled a
  feasibility surrogate, not a Phase-1 certification.
- **Untracked review surface:** deliverables untracked; code review embeds file contents +
  `git status` delta (run.md). Coder edits no tracked code and commits nothing.

## Changes Since Previous Plan Version

Addresses every plan_v1 review finding (raw: `state/plan_v1_review.review.raw.md`):

- **[major] #2 G0 representation exact →** nodes = skeleton voxels painted via the scorer's
  own `erl_graph.get_nodes_position([9,9,20])` (agrees with the scorer by construction; no
  hand-rolled transform); coords are physical nm (bound); ≥99%-in-seg transform assertion;
  no medial band for the GT oracle (deferred to DESIGN.md).
- **[major] #3 foreign-corridor / rasterization →** kept union-find roots authoritative;
  added a foreign-corridor **rejection** rule + `foreign_corridor_rejected` audit for
  Cheb-length>1 edges (the reply's SAME-with-clear-corridor semantics), so the "0 paths
  through a foreign band" audit exists without unsafe spatial-CC unions.
- **[major] #4 perturbation →** graph-level closure (cheap); C1/C2/C3 with exact side
  anchors (skeleton-graph neighbors), deterministic C2 median-section selection, C3
  U{2..5} arc, per-condition `rng(101)` reset, min(N,300) sampling, denominators, and a
  per-condition gate table; fragments/GT + retained length use physical `edge_len` (I2).
- **[major] #7 G1 oracle events →** defined as skeleton-adjacent same-skeleton fragment
  pairs (via erlgraph edges), break location = crossing-edge midpoint, hit rule = link +
  both tips within 45 nm of the break, boundary-margin exclusion, per-event ΔNERL via
  incremental node-LUT gain.
- **[major] #8 cap gate →** raw max candidates/endpoint banded ≤8 pass / 9–12 gray / >12
  kill; retention = top-8 at either endpoint; proposals/recovered-bridge on the capped set.
- **[major] #9 realizer trivially passable →** sparse centerline seeds (not full bodies) +
  ≥95% coverage requirement + control-vs-capped-recovered; realize from the **capped**
  recovered set (not full oracle), so the generator is not bypassed.
- **[major] #10 exact affinity asset →** full path/key/shape/dtype/axes/crop/channel-axis
  mapping bound; validated by Sanity B (0.836).
- **[major] #12 DESIGN 43-ch table →** indexed table (0–42) pinned in this plan with the
  per-block targets; DESIGN.md expands each formula + sparse skip head.
- **[major] absolute Phase-1 gate on a crop →** replaced with predeclared scale-appropriate
  gates (ΔNERL-weighted recall ≥90%, realizer control, recovered-fraction ≥0.60); G1
  explicitly labeled a feasibility surrogate, not Phase-1 certification (answers the
  reviewer's open question).
- **[major] resource/runtime strategy →** added: paint-once, graph-level perturbations,
  incremental ΔNERL, masked watershed, dtype/memory budget, 256³ smoke + monitored full run.
- **[major] G0 offset banks →** replaced the crumb's radius-3/4 sweep with the reply's
  Phase-0 banks B0/B1/B2/B3 (13/16/24/49), enumerated with counts + half-space dedup; Z is
  the anisotropic axis-2.
- Prior resolved findings (#1,#5,#6,#11,#13,#14) retained.

# Plan v1

## Summary

Same two deliverables as v0 — a codebase-grounded MESA-EM design doc plus two cheap
gates (G0 = CPU Phase-0 oracle plumbing on GT channels; G1 = banis+ no-retrain
feasibility) actually run on the standard center chunk with real numbers vs. the
reply's kill criteria — but every implementation choice the reviewer flagged is now
pinned to a verified script/API/path so the gate is reproducible. No GPU training and
no full-volume run is launched. A gate failure ends the branch with a written finding.

The design rests on lesson 20 (perfect SDT + GT connected per-instance seeds → grow
≈0.994; a thresholded scalar shatters, carrying no identity). G0 tests exactly the
missing object — a connected merge-safe centerline **graph** built from same-instance
edges — in isolation from any learned prediction. G1 tests whether banis+ already
supplies enough for the candidate **generator** and the body **realizer** to be worth
training the heads.

## Scope

**In scope:** `DESIGN.md`; G0 (Phase-0 GT-channel decode + audit, center chunk); G1
(geometric candidate-recall + realizer feasibility on cached banis+ affinity, center
chunk); `results_g0.md` / `results_g1.md` with measured numbers + one go/no-go readout.

**Out of scope (deferred "whole experiment"):** training any head (medialness / edges /
veto / PairNet / BiLOQ); `main.py` train / `sbatch` / GPU fit; full-volume seed101
decode; learned validators + empirical-risk calibration; predicted-SDT grow (G0 uses
GT-SDT only); fill Phase 3 beyond the realizer control inside G0/G1. Per the reviewer,
**no second harder chunk is added this run** — the center chunk is the gate; a
split-dominated chunk (`tile_0_0_2`) is only *recommended* in `DESIGN.md` as a later
validation gate.

## Proposed Changes

New untracked files under `dev/nisb/scripts/mesa/`. New scripts **import/adapt** logic
from the read-only prior-art scripts; they do **not** modify any tracked file.

### Step 0 — Harness sanity (blocking; shared)

Reuse the node-LUT scorer, do not re-derive NERL. Two scoring entry points, used
consistently:
- **Whole-vol node-LUT** (`per_chunk_nerl.load_cache/nerl/subgraph` +
  `connectomics.metrics.nerl.import_em_erl`) for the whole-vol invariants.
- **Center-chunk local** (`connectomics.metrics.nerl.compute_nerl_score_details(seg,
  str(SKEL), resolution=[9,9,20], merge_threshold=1)` with
  `SKEL=dev/nisb/data/center_chunk/seg.erlgraph.npz`) for ALL center-chunk numbers —
  the same path `center_chunk_decode_sweep.py` uses. Never score a crop against the
  whole-vol LUT (lesson_testcrop_preflight).

Blocking sanity (with tolerances):
- **A (whole-vol):** cc3d@0.66 = **0.601431 ± 0.0005**; cc3d@0.75 = **0.545 ± 0.001**.
  (Branch-merge oracle 0.772069 is *supplemental* — report if the cache is loaded, do
  not block on it.)
- **B (center-chunk local):** cc3d@0.66 on the cached baseline affinity =
  **0.836 ± 0.003** (recorded value, `exp/center_chunk_decode_sweep.md`;
  `decode_affinity_cc(aff3, threshold=0.66, backend="numba", edge_offset=0)` on
  `raw_x1_ch0-1-2.h5[:,1000:2000,1000:2000,450:900]`). If A or B drifts beyond
  tolerance, STOP — the harness is mis-wired; write the finding, do not run G0/G1.

### G0 — Phase-0 oracle plumbing (CPU, GT channels), canonical pipeline

Port the proven `centerline_graph_decode_gt.py` mechanism from the retired crumb to the
center chunk. **Identity comes only from same-instance edges; instance IDs are never
read during decode.** One canonical pipeline (no unresolved either/ors):

1. **Skeleton source = the center-chunk erlgraph** (consistency with the scorer):
   paint a skeleton-label volume `skel` (shape 1000×1000×450, array order matching the
   affinity crop) from `seg.erlgraph.npz` `node_coords_zyx` + per-node `skeleton_id`.
   This is the same skeleton NERL scores against — no separate kimimaro run.
2. **GT-SDT** for grow: reuse `make_gt_sdt_region.py` to compute a signed distance from
   the center-chunk GT `seg.h5` (radius-aware, matching the ero1 convention already in
   `x2diag/gt_sdt_region_ero1.h5`). Cache to `dev/nisb/scripts/mesa/gt_sdt_center.h5`.
3. **Edge graph = same-instance union-find over offset banks** (exactly
   `components()`/`UF` from the ancestor): `edge(v,v+o)=1 iff skel[v]>0 & skel[v+o]>0 &
   skel[v]==skel[v+o]`. Because the SAME test already requires equal instance id, a
   cross-instance union is impossible by construction (this is the built-in MUTEX);
   record and assert this holds. **Skip edges are additional offsets in the same
   union-find, NOT rasterized spatial paths** — so spatial CC can never invent an
   identity union (resolves the v0 rasterization risk). Grow (step 5) handles the
   spatial fill.
4. **Offset banks** (cumulative, half-space dedup by `undirected_offsets`, one of each
   ± pair): enumerate and report each bank's exact offset count:
   - `O1` = 13 (26-neighborhood half-space);
   - `O1+2` = O1 + {(±2,0,0),(0,±2,0),(0,0,±2)} deduped;
   - `O1+2,3`; `O1+2,3,4` (the ancestor's `[],[2],[2,3],[2,3,4]` sweep);
   - plus one **anisotropy/one-section** bank: add ΔZ=2 with Δx,Δy∈{-1,0,1}.
   Pick the smallest bank meeting the gate.
5. **Grow:** `watershed(-sdt, markers=components, mask=(sdt > -0.3))` (ancestor's
   realizer). Markers ARE the union-find components; watershed only fills.
6. **Audit vs GT after decode** on the center-chunk local skeleton: fragments/GT;
   maxGTspan (# components a GT skeleton is split across) and its inverse — count of
   grown labels spanning ≥2 GT ids (false merges, must be 0); retained skeleton length;
   local NERL.

**Perturbation protocol (reproducible; fixes the 95%/90% gate):** 4 node conditions,
all with `numpy.random.default_rng(101)`:
- **C0 intact** — the full skeleton graph.
- **C1 one interior voxel deleted** — per eligible skeleton (skeleton with ≥5 nodes and
  a non-endpoint interior node ≥2 voxels from any tip and ≥3 voxels from the crop
  boundary), delete exactly one interior node; sample min(N,300) eligible skeletons;
  fixed seed. "Gap closed" = the two sides land in the SAME grown label AND no new
  cross-GT merge. Closure rate = closed / eligible-sampled. Report denominator.
- **C2 one full Z-section deleted** — per eligible skeleton (spans ≥3 Z sections, not
  touching the crop's Z faces), delete all its nodes in one interior Z section (per
  object, not global); same sampling/seed/criterion.
- **C3 2–5-node crumb deleted** — delete a random contiguous 2–5-node arc drawn from the
  observed fragment-length distribution; same sampling/seed/criterion.
Each condition scored independently; **all four must pass** their thresholds.

**G0 pass (reply §12 Phase 0):** 0 grown labels spanning >1 GT id (0 false merges); 0
cross-GT graph unions (asserted by construction); fragments/GT ≤ 1.02; retained GT
skeleton length ≥ 99.5%; ≥95% one-voxel gaps closed (C1); ≥90% one-section gaps closed
(C2); GT-SDT grow local NERL ≥ 0.990 (target ≈0.994). Report the smallest passing bank +
per-condition breakdown.

**G0 kill:** any cross-GT union; decode needs instance IDs (not edges); best bank NERL
< 0.985 (GT-SDT grow — the predicted-SDT `0.985` criterion from v0 is **removed**, this
is the GT-SDT value); or fragments/GT stays > 1.05.

### G1 — banis+ small-chunk feasibility (no retrain), one bound label map

New `scripts/mesa/g1_banis_feasibility.py`. **One starting map, bound throughout:**
`BASE_SEG = decode_affinity_cc(aff, 0.66, edge_offset=0)` on the cached banis+ affinity
`raw_x1_ch0-1-2.h5[:,1000:2000,1000:2000,450:900]` (the Sanity-B seg; local NERL 0.836,
2346 inst). The 0.627 intersection-cut firewall is a whole-vol construct; the center
chunk has ~0 baseline false merges (lesson), so G1 operates on the merge-safe cc3d@0.66
fragments directly and states this explicitly (no firewall on the crop).

1. **Local oracle bridge set (the prize) + per-bridge ΔNERL** — adapt the local logic of
   `ec_endpoint_bridge.py` (NOT the whole-vol `banis+_oracle_merge.py`): map each local
   erlgraph node → its `BASE_SEG` fragment; per GT skeleton, the set of distinct
   fragments it touches = its split set; each adjacent split pair is an oracle bridge.
   Each bridge's marginal ΔNERL = local NERL(merge that one pair) − NERL(BASE_SEG),
   via `compute_nerl_score_details`. The union of all bridges per skeleton is the
   frags/GT ceiling (the oracle).
2. **Endpoints/tips** = `ec_endpoint_bridge._extract_tips_fast` on `BASE_SEG` fragments
   (2 farthest-point tips + physical outward tangents; seconds, no kimimaro).
3. **Geometric candidate generation** with reply §2.2 proposal gates (physical distance
   radius-aware; tangent agreement ≥0.65; radius ratio ≤3.0; Hermite curvature; endpoint
   neighborhoods defined; crop-boundary tips excluded from *recall denominator* since
   their partner may be outside). Produce ALL proposals first (no cap).
4. **Recall, reported both ways (fixes the cap-makes-it-trivial finding):**
   - raw post-pruning candidates/endpoint distribution (mean, p95, max) BEFORE capping;
   - a bridge is a "hit" if a proposed candidate connects its two fragment ids;
   - recall AFTER a deterministic ranked **top-8** cap per endpoint (rank: tangent
     agreement desc, then distance asc; tie-break: partner fragment id asc). Report
     count recall and ΔNERL-weighted recall for both raw and capped.
5. **Realizer feasibility (executable, with a control):** supervoxels = the merge-safe
   `BASE_SEG` cc3d@0.66 fragments (0 cross-GT on the center chunk — assert). Grow orphan/
   background voxels by geodesic cost `c(p,q) = -log(aff_r1(p,q)+eps)` (reply §3.3,
   aff_r1 = ch0-1-2 face edges) to the nearest centerline identity, competition margin
   `log 9`, min body-prob gate from `sdt>-0.3`. Two runs:
   - **(a) base-preservation control:** seeds = BASE_SEG fragments, no repairs → grown
     local NERL within **0.005** of BASE_SEG NERL, **0 seed-ID unions**, **0 multi-ID
     supervoxel assignments** (guards against trivial pass);
   - **(b) candidate-oracle-repaired seeds:** merge fragments joined by the oracle bridge
     set, then grow → grown NERL should approach the local oracle, still 0 cross-GT.

**G1 pass (reply §12 Phase 1 analog; recall is the scale-invariant gate):** count recall
≥ 80%; ΔNERL-weighted recall ≥ 90% (capped top-8); ≤8 candidates/endpoint after cap AND
≤12 before cap (else kill); ≤6 proposals/recovered bridge; candidate-oracle matching adds
0 false merges; realizer control (a) passes its three conditions. Local candidate-oracle
NERL is reported as **secondary context only**, never compared to whole-vol 0.752/0.627.

**G1 kill:** ΔNERL-weighted recall < 80%; >12 candidates/endpoint after geometric pruning;
recall needs global BFS / whole-volume tracing; or realizer control (a) fails (grow moves
NERL >0.005 or creates any cross-GT).

### DESIGN.md (indexed, verifiable)

`scripts/mesa/DESIGN.md`, mirrored to this run's `artifacts/DESIGN.md`. Must include an
**indexed dense-channel accounting summing to 43**: aff 9 + graded medialness 1 + signed
log boundary distance 1 + tangent 3 + short SAME/MUTEX 26 (13 half-space offsets × 2
logits) + vetoes 3 = **43**; plus the sparse skip head (2 logits per queried edge, not in
the 43). For every channel: target formula from `seg` + cached skeleton, edge SAME/MUTEX/
DEFER label rule, veto target definition. Then the link→grow→fill decode mapped to
`dev/nisb/` scripts, the Phase 0–3 experiment ladder with the reply's gate numbers, the
honest forecast (central 0.792, stretch 0.812; all deltas off 0.627), and the
`tile_0_0_2` second-chunk recommendation for later validation.

## Files and Areas

| File | Purpose |
|---|---|
| `dev/nisb/scripts/mesa/DESIGN.md` | MESA-EM mapped to `dev/nisb/`; 43-channel accounting; Phase 0–3 ladder; forecast |
| `dev/nisb/scripts/mesa/common.py` | shared: center-chunk load (affinity crop, GT seg, erlgraph→skel volume), GT-SDT build, both scorers, offset banks |
| `dev/nisb/scripts/mesa/g0_phase0_oracle.py` | G0 decode + perturbation audit (adapts `centerline_graph_decode_gt.py`) |
| `dev/nisb/scripts/mesa/g1_banis_feasibility.py` | G1 recall + realizer feasibility (adapts `ec_endpoint_bridge.py`) |
| `dev/nisb/scripts/mesa/gt_sdt_center.h5` | cached center-chunk GT-SDT (build once) |
| `dev/nisb/scripts/mesa/results_g0.md` | G0 numbers + verdict vs Phase-0 gate |
| `dev/nisb/scripts/mesa/results_g1.md` | G1 numbers + verdict vs Phase-1 gate + overall go/no-go |
| `.agent/features/nisb_base_60-80_plan/artifacts/DESIGN.md` | mirror of the design doc |

Read-only (import/adapt, never modify): `dev/nisb/waterz/per_chunk_nerl.py`,
`connectomics/metrics/nerl.py`, `connectomics/decoding/decoders/segmentation.py`
(`decode_affinity_cc`), `scripts/centerline_graph_decode_gt.py`,
`scripts/make_gt_sdt_region.py`, `scripts/ec_endpoint_bridge.py`,
`scripts/center_chunk_decode_sweep.py`, `ec_merge/ec_common.py`,
`data/center_chunk/*`, `x2diag/gt_sdt_region_ero*.h5`,
`outputs/.../seed101/raw_x1_ch0-1-2.h5`.

## Verification Plan

Every number verified by OUTPUT, written into the results md; no claim from code reading.

1. **Harness sanity (blocking):** A: 0.601431±0.0005 and 0.545±0.001; B: 0.836±0.003.
   Fail ⇒ stop + finding.
2. **G0:** run the offset-bank sweep × 4 perturbation conditions on the center chunk;
   pass = the six §"G0 pass" thresholds; report smallest passing bank + per-condition
   table + the 0-false-merge assertion. Kill per §"G0 kill".
3. **G1:** run recall (raw + top-8 capped, count + ΔNERL-weighted) and realizer (a)+(b);
   pass = the §"G1 pass" thresholds; report candidate-count distribution, per-bridge
   recall, realizer control deltas. Kill per §"G1 kill".
4. **Go/no-go:** one paragraph in `results_g1.md` — three outcomes explicitly:
   (i) **go** only if EVERY G0 and G1 success criterion passes → recommend training the
   heads, naming the highest-value channels; (ii) **gray zone** (some pass, none killed)
   → no-go for training, name the cheap refinement to try first; (iii) **kill** → stop the
   branch with the failing gate. Always emit a result-location line even if a G0 failure
   prevents `results_g1.md` (write the reason into `results_g0.md`).

**Runtime guard** (user's "surface silent failure loudly"): G0/G1 first run a small-region
smoke (reuse an `x2diag` GT-SDT crop / a 256³ sub-crop) and print smoke numbers
immediately. The full center-chunk pass (EDT/watershed on 450 M voxels) runs to
completion — if backgrounded, it is monitored to a **successful process exit + validated
non-empty output** before the results md is finalized; smoke numbers never substitute for
a completed gate. No step waits on GPU training.

## Risks and Questions

- **G0 proves the DECODE, not learnability.** Passing G0 with GT-derived edges does not
  prove the heads are trainable; that is the deferred Phase 2. Stated as the G0↔Phase-2
  boundary in `DESIGN.md`.
- **Center-chunk local optimum ≠ whole-vol.** Base peaks @0.55 (0.856) locally vs @0.66
  whole-vol; G1 fixes @0.66 for consistency and reports the local optimum as context.
  All crop numbers are relative; recall (scale-invariant) is the gate.
- **GT-SDT for grow is an oracle input** (real pipeline would use predicted SDT). G0 is
  explicitly the *decode* oracle; predicted-SDT grow is deferred (no `0.985`
  predicted-SDT claim in this run).
- **`make_gt_sdt_region.py` convention** must match the erlgraph/scorer array order and
  the ero1 radius convention; `common.py` asserts shape/orientation against `seg.h5`
  before use.
- **Cross-repo/untracked review surface:** deliverables are untracked; code review embeds
  file contents + the `git status` delta (run.md). Coder must not edit tracked code/commit.
- **Resolved (was v0 open question):** no second chunk this run — center chunk is the
  gate; `tile_0_0_2` recommended in `DESIGN.md` for later.
- **Resolved:** banis+ center-chunk affinity is a cached crop (`raw_x1_ch0-1-2.h5`), so G1
  needs no inference; the whole-vol → crop indices are `[:,1000:2000,1000:2000,450:900]`.

## Changes Since Previous Plan Version

Addresses every plan_v0 review finding (raw: `state/plan_v0_review.review.raw.md`):

- **[major] Step 0 sanity incomplete →** added cc3d@0.75=0.545 with tolerances on all
  three sanities (0.601431±0.0005, 0.545±0.001, center 0.836±0.003); 0.772069 demoted to
  supplemental/non-blocking.
- **[major] G0 unresolved choices →** pinned one canonical pipeline: erlgraph-painted
  skeleton (not kimimaro), watershed-on-GT-SDT grow, same-instance union-find, enumerated
  offset banks (13/O1+2/…+one-section) with half-space dedup, array order asserted vs seg.h5.
- **[major] G0 skip rasterization can invent unions →** skip edges are extra union-find
  offsets (same-instance-gated), NOT rasterized paths; spatial CC never creates identity;
  grow handled by watershed over the union-find markers. Cross-GT union impossible by the
  SAME test; asserted.
- **[major] G0 perturbation not reproducible →** defined C0–C3 with `rng(101)`, eligibility
  rules, per-object (not global) section deletion, crumb-length source, min(N,300) sampling,
  closure criterion, denominators, and independent per-condition pass.
- **[major] stray 0.985 predicted-SDT kill →** removed; G0 runs GT-SDT only; predicted-SDT
  explicitly deferred; the 0.985 kill now refers to the GT-SDT grow value.
- **[major] G1 unbound starting map →** bound to `BASE_SEG = cc3d@0.66` on the cached
  affinity crop; firewall stated as inapplicable on the crop (≈0 baseline merges).
- **[major] G1 bridge-source conflict →** local `ec_endpoint_bridge.py` tip + oracle-pair
  logic on the center chunk is canonical; `banis+_oracle_merge.py` (whole-vol LUT) is NOT
  used for the crop. Endpoint neighborhoods, boundary exclusion, and per-bridge ΔNERL via
  `compute_nerl_score_details` specified.
- **[major] G1 caps make success trivial →** report raw post-pruning counts before any cap,
  then recall after a deterministic ranked top-8 cap; added the ≤12-before-cap kill so >12
  is reachable.
- **[major] G1 realizer not executable →** specified aff_r1 channels, geodesic cost,
  supervoxels = merge-safe cc3d fragments, competition margin, min coverage, and BOTH a
  base-preservation control (NERL within 0.005, 0 unions, 0 multi-ID) and oracle-repaired
  seeds.
- **[major] GT-fragment surrogate invalid →** removed; banis+ affinity crop path resolved
  (`raw_x1_ch0-1-2.h5`), so G1 uses real banis+ predictions; if it were absent G1 would be
  no-go, not a surrogate pass.
- **[major] smoke ≠ completed gate / gray zone / result location →** runtime guard now
  requires a monitored successful exit + validated output before finalizing results; the
  go/no-go defines go / gray-zone / kill; a result-location line is always emitted.
- **[major] DESIGN.md 43-channel claim →** DESIGN.md must carry the indexed accounting that
  sums to 43 (9+1+1+3+26+3) plus the sparse skip head, each channel's target defined.
- **[minor] "extend" vs read-only →** new `mesa/` scripts import/adapt logic without
  modifying tracked files; banis+ affinity load path specified.
- **[minor] second chunk →** not added this run; recommended in DESIGN.md for later.

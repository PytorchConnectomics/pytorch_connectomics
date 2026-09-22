# Plan v3

## Summary

Final, fully-specified plan for the two cheap gates (G0 = CPU Phase-0 oracle plumbing on
GT channels; G1 = banis+ no-retrain feasibility) on the standard center chunk, plus the
MESA-EM `DESIGN.md`. No GPU training, no full-volume decode. v3 (a) ratifies the
human decision that **G0 is the proven-ancestor 1-voxel-skeleton oracle** (`centerline_graph_decode_gt.py`
mechanism), which the reviewer must accept as scoped rather than re-litigate against the
reply's learned medial-band Phase-0; and (b) folds in every real correctness/scope fix from
`plan_v2_review.md` and pins all constants, formulas, and metric definitions so each gate
yields a trustworthy pass/fail. G1 is an explicit **feasibility surrogate** (generator +
geometric-matching + realizer), not a Phase-1 certification (that needs trained heads).

## Scope

Unchanged from v2. **In:** `DESIGN.md`; G0 (center chunk); G1 (center chunk); results md +
one go/no-go. **Out:** training any head; `main.py`/`sbatch`/GPU fit; full-volume decode;
learned validators + risk calibration; predicted-SDT grow; fill Phase 3 beyond the realizer
control; a second chunk (recommended in `DESIGN.md`, not run).

**Ratified design decision (human, 2026-07-15):** G0 uses 1-voxel skeleton nodes +
same-instance offset union-find + watershed-on-GT-SDT (the ancestor). Medial-band nodes,
signed SAME/MUTEX/DEFER edge *logits*, and rasterized skip-path painting are learned-pipeline
(Phase-2) constructs that carry no extra information when identity is GT-known; they are
specified in `DESIGN.md` for the training phase, not implemented in the G0 oracle.

## Verified assets (bound; used verbatim)

- **Affinity (banis+ seed42):** `outputs/nisb_base_banis_v3_erosion2/20260508_224029/test_step=00200000/seed101/raw_x1_ch0-1-2.h5`,
  key `main`, shape `(3,3000,3000,1350)`, float16, axes `(C,X,Y,Z)`; crop
  `main[:,1000:2000,1000:2000,450:900]`→`(3,1000,1000,450)` float32; channel `c` = +1
  neighbor affinity along array axis `c` (0=X@9nm,1=Y@9nm,2=Z@20nm); decode
  `decode_affinity_cc(aff,0.66,backend="numba",edge_offset=0)`.
- **GT seg:** `dev/nisb/data/center_chunk/seg.h5`, order `(X,Y,Z)=(1000,1000,450)`.
- **Local skeleton graph:** `dev/nisb/data/center_chunk/seg.erlgraph.npz` — 626 skeletons,
  491,235 nodes; `node_coords_zyx` physical nm; `node_skeleton_index`∈[0,625];
  `edge_u/edge_v/edge_ptr/edge_len` (physical) define the skeleton graph.
- **nm→voxel (reuse, do not hand-roll):** `connectomics.metrics.nerl.load_nerl_graph(str(SKEL),
  None, resolution=[9,9,20])` → `erl_graph.get_nodes_position([9,9,20])` (seg array order;
  same mapping the scorer uses).
- **Scorers:** center-chunk `compute_nerl_score_details(seg, str(SKEL), resolution=[9,9,20],
  merge_threshold=1, num_workers=16).nerl`; whole-vol cached-LUT
  `per_chunk_nerl.load_cache` + `import_em_erl` + `compute_erl_score`.
- **Cached whole-vol LUT (for Sanity A, seconds — NOT a decode):**
  `outputs/.../seed101/seg_fusion/oracle_lut/ch0-1-2_cc0.66_node_luts.npz` (and the cc0.75
  LUT if present).
- **Recorded anchor:** cc3d@0.66 on the crop = 0.836 (2346 inst), `exp/center_chunk_decode_sweep.md`.

## Proposed Changes

New untracked files under `dev/nisb/scripts/mesa/`; import/adapt read-only prior art;
modify no tracked file; commit nothing.

### Step 0 — Harness sanity (blocking; shared)

- **A (whole-vol, cached LUT, seconds):** `load_cache` the cc0.66 node-LUT and score via
  `compute_erl_score(graph, pred_lut.astype(uint64), merge_threshold=1)` → **0.601431 ±
  0.0005**. If the cc0.75 LUT is present, likewise → **0.545 ± 0.001**; else record cc0.75
  as an unavailable provenance note (non-blocking). **Do NOT decode the 12.15 B-voxel
  volume.**
- **B (center-chunk, measured):** `decode_affinity_cc(crop,0.66,...)` → score →
  `BASE_NERL` (expect 0.836 ± 0.003). **Bind `BASE_NERL` as the measured value** and use it
  (not the rounded 0.836) in all G1 deltas.
- **Transform sanity:** paint skeleton via `get_nodes_position`; assert ≥99% of painted
  voxels lie in `seg.h5>0`. Assert `seg.h5` shape == crop shape.
Any failure ⇒ stop with a written finding; do not run G0/G1.

### G0 — Phase-0 oracle plumbing (CPU, GT channels; ancestor mechanism)

Adapt `centerline_graph_decode_gt.py` (`components()`, `UF`, `undirected_offsets`) to the
center chunk. **Identity comes only from same-instance edges; instance IDs never read at
decode.**

1. **Nodes = skeleton voxels.** Paint `skel` int32 `(1000,1000,450)`:
   `skel[get_nodes_position(n)] = node_skeleton_index[n]+1`.
2. **GT-SDT** via `make_gt_sdt_region.py` ero1 convention on `seg.h5`; cache
   `mesa/gt_sdt_center.h5`; assert shape==crop.
3. **Edges = same-instance offset union-find** (`skel[v]>0 & skel[v+o]>0 & skel[v]==skel[v+o]`);
   cross-instance union impossible by the `==` test — assert 0 cross-GT unions.
   **Foreign-corridor rule (Cheb-length>1 offsets only):** before uniting, walk the straight
   voxel corridor `u→u+o`; if any interior corridor voxel has `seg>0` with GT id ≠ `seg[u]`,
   reject the edge and increment `foreign_corridor_rejected`. Union-find roots stay
   authoritative; no spatial CC creates unions. (This is the audit substitute for the
   reply's foreign-band-traversal check, appropriate to the 1-voxel oracle.)
4. **Cumulative offset banks** (array order `(X,Y,Z)`; half-space dedup via
   `undirected_offsets`; the script prints each bank's exact deduped tuple list + count):
   - `B0` = 26-nbr half-space = **13**;
   - `B1` = B0 ∪ {(2,0,0),(0,2,0),(0,0,2)} = **16**;
   - `B2` = B1 ∪ {(dx,dy,2):dx,dy∈{-1,0,1}} (one-Z-section) = **24**;
   - `B3` = B0 ∪ (full `|o|∞=2` half-shell, 49 offsets) = **62** (the shell excludes the
     `|o|∞=1` nearest neighbors, so it is *added to* B0, not a replacement — corrects the
     v2 "49"). Choose the smallest bank meeting the gate.
5. **Grow:** `watershed(-gt_sdt, markers=components, mask=(gt_sdt>-0.3))` over fg only.
6. **Audit** (local scorer): local NERL; `false_merge` = # grown labels spanning >1 GT id
   (must be 0); `foreign_corridor_rejected`; **fragments/GT** = mean over the 626 scored
   skeletons of (# distinct grown labels covering that skeleton's nodes); **retained skeleton
   length** = `Σ edge_len over skeleton edges whose two endpoint voxels share one nonzero
   grown label / Σ all edge_len` (physical, I2-respecting).

**Perturbation protocol (graph-level, reproducible, cheap).** Closure measured on the
union-find graph (no re-grow); full watershed+NERL only for C0 (per bank) + the chosen bank.
`np.random.default_rng(101)`, **reset per condition**:
- **C0 intact** — the primary NERL/fragments/retained gate.
- **C1 one interior voxel deleted** — eligible node = **erlgraph degree-2**, ≥2 vox from any
  tip, ≥3 vox from the crop face. Delete its voxel; anchors = its two erlgraph neighbors'
  voxels. Closed = anchors in one component AND no new cross-GT. Sample `min(N_elig,300)`;
  report denominator. **Gate ≥95%.**
- **C2 one interior occupied Z-section deleted** — eligible skeleton has ≥3 occupied Z
  sections (a section = a Z index holding ≥1 of its nodes), none on the Z faces. Delete all
  its nodes in the **median occupied interior section**. That section may cut several strands;
  **for every cut strand** the anchor pair = the nearest surviving node on each Z-side;
  closed = **all** strands' anchor pairs reconnect with no cross-GT. **Gate ≥90%.**
- **C3 2–5-node crumb deleted** — delete a contiguous arc of length `U{2,3,4,5}`; anchors =
  flanking survivors; closed as above. **Diagnostic** (no reply threshold); report closure vs
  arc length.

**G0 pass (reply §12 Phase 0):** false_merge=0; 0 cross-GT unions (asserted); fragments/GT
≤1.02; retained length ≥99.5%; C1 ≥95%; C2 ≥90%; C0 GT-SDT grow local NERL ≥0.990 (target
≈0.994). Report smallest passing bank + per-condition table + `foreign_corridor_rejected`.
**G0 kill:** any cross-GT union; decode needs instance IDs; best-bank C0 NERL <0.985;
fragments/GT >1.05.

### G1 — banis+ feasibility (no retrain); four separated measurements

`BASE_SEG = decode_affinity_cc(crop,0.66,edge_offset=0)` (NERL `BASE_NERL`≈0.836, 2346 inst).
No firewall on the crop (≈0 baseline merges; assert 0 cross-GT `BASE_SEG` fragments on GT).

**Oracle events (break clusters) + ΔNERL.** Map each erlgraph node → `BASE_SEG` fragment via
`get_nodes_position`. A skeleton edge whose endpoints fall in different fragments is a *cut
edge*. Group all cut edges between the same unordered pair `(f_a,f_b)` of one GT skeleton
into one **event** carrying the **set** of break midpoints (handles multiple crossings).
Exclude events whose every break is <2 vox from a crop face. `full_oracle` = merge all
same-skeleton fragments. **ΔNERL(event)** closed form = `2·L_a·L_b / D`, where `L_a,L_b` =
physical skeleton length of that GT skeleton inside `f_a,f_b` (Σ `edge_len` of intra-fragment
edges; cut edges split half/half), `D` = the global `Σ gt²` normalizer. **Validate:** on a
deterministic sample of 20 events compare closed-form ΔNERL to (full-scorer NERL after
merging that one pair − `BASE_NERL`); require |Δ|≤1e-3, else fall back to per-event
full-scorer on the (bounded) event set and note the cost. If **`full_oracle` relabel gain ≤
0** (or <0.02), flag the center chunk as low-split-headroom, report it, and recommend adding
`tile_0_0_2` before trusting G1 recovered-fraction (honest handling of the crop's small
+0.06 headroom).

1. **Generator recall.** Tips = `ec_endpoint_bridge._extract_tips_fast(BASE_SEG)` (2
   farthest-point tips + physical outward tangents). **Radius estimator** `r_est(tip)` =
   median boundary-EDT of the tip's fragment over its K=12 nearest fragment voxels (×res).
   `r_max=max(r_est_u,r_est_v)`. **Proposal gates** (physical): distance ≤ `D_lim =
   min(750nm, max(8·r_max, 3·20nm, 12·9nm))`; outward-tangent agreement ≥0.65; radius ratio
   ≤3.0; cubic-Hermite (tips + tip tangents) min curvature radius ≥1.25·r_max. Degeneracy
   (fragment <12 vox, or undefined tangent) → skip with a logged reason. Produce ALL
   proposals (no cap). A candidate **covers** an event iff it links `f_a,f_b` AND ≥1 tip is
   within `R_end=45nm` of ANY break in the cluster. Report raw candidates/endpoint (mean, p95,
   **max**); then a deterministic **top-8** cap per endpoint (rank: tangent desc, distance
   asc, partner-id asc; retained if top-8 at either endpoint). **Count recall** and
   **ΔNERL-weighted recall** on the capped set; proposals/recovered-bridge (capped).
2. **Geometric-matching merge-safety (the realistic linker test).** Run non-transitive greedy
   best-buddy matching on capped candidates using the tighter **acceptance** gates (tangent
   ≥0.80, radius ratio ≤2.5, curvature radius ≥1.5·r_max), degree-1 per endpoint, mutual
   top-1, no transitive chains. Relabel `BASE_SEG` by accepted links. Report **precision**
   (accepted links joining the same GT / all accepted), **# false merges** (accepted links
   joining different GT skeletons), and NERL(matched). This is expected to WALL (geometry-only
   ≈0.67 precision, lesson 17) and quantifies the gap the learned PairNet/BiLOQ must close —
   **diagnostic, reported, not a go/no-go gate** (a nonzero geometry-only merge count is
   expected).
3. **Identity-gain upper bound (oracle-confirmed).** Relabel `BASE_SEG` by the capped-covered
   events that are GT-same → NERL. **Recovered fraction** =
   `(NERL_capped_oracle − BASE_NERL)/(NERL_full_oracle − BASE_NERL)`. This is trivially
   0-merge (GT-confirmed) and is explicitly the **recall upper bound**, distinct from (2).
4. **Realizer fidelity (fragment-adjacency graph; bounded).** Assign every `BASE_SEG`
   fragment to one centerline identity; **grow orphan/background** (fragments with no skeleton
   node, and unlabeled voxels) over the **fragment-adjacency graph** (~2346 nodes, O(frags),
   not a 450 M-voxel search). Edge `(f_i,f_j)` cost = `-log(mean aff_r1 over their shared
   face + 1e-6)`, aff_r1 read from the channel matching the face orientation (reverse step =
   same undirected face). Assign an orphan to the min-cost identity; **runner-up margin**
   `≥log 9` else label **abstain=0 (unknown)**. Atomic-SV policy: a fragment is never split
   (merge-safe SV); multi-identity fragments cannot occur because `BASE_SEG` fragments are
   asserted 0-cross-GT. **Control** = per-fragment identity (no repair): realized == relabel
   of `BASE_SEG`, so grown NERL must be within **0.005 of `BASE_NERL`**, **0 cross-GT**, **0
   multi-ID assignments**; report coverage gained (orphan voxels adopted). Peak resources:
   label vols `(1000,1000,450)` uint32 ≈1.8 GB + float32 aff crop ≈5.4 GB (or stream per
   axis); adjacency graph negligible.

**G1 pass (feasibility surrogate; predeclared, scale-appropriate):** (i) count recall ≥80%
AND ΔNERL-weighted recall ≥90% (capped); (ii) raw max candidates/endpoint ≤8 (9–12 gray, >12
kill); (iii) realizer control within 0.005 of `BASE_NERL`, 0 cross-GT, 0 multi-ID; (iv)
recovered-fraction (upper bound) ≥0.60 **provided** `full_oracle` gain ≥0.02 (else the chunk
is flagged low-headroom and (iv) is reported, not gated). The geometric-matching precision/
merges (2) are reported as the realistic number and the motivation for the learned scorer;
whole-vol absolute gates (0.752/0.742) are NOT applied on the crop. **G1 kill:** ΔNERL-weighted
recall <80%; raw max candidates/endpoint >12; recall needs global BFS/whole-vol tracing;
realizer control fails.

### DESIGN.md (channel formulas pinned here; resolve #12)

`scripts/mesa/DESIGN.md` (mirrored to `artifacts/DESIGN.md`). **Dense channel map (0–42, =43):**

| idx | channel | formula / target |
|---|---|---|
| 0–8 | affinity (existing 9-ch) | current banis+ r1+r10 ero2 targets (merge-safe anchor) |
| 9 | graded medialness `m*` | `exp(-‖x_v−x_π(v)‖²/2σ²)`, `σ=clip(0.35·r_i(π), s_xy, 2.5·s_xy)`, truncated 0 beyond ~0.6·r_i |
| 10 | signed log boundary dist `d_b` | `sign(v∈G)·log(1+d(v,∂G)/s_xy)` |
| 11–13 | tangent (3) | local skeleton principal direction (PCA over window `R_t=max(4·s_xy,2·r_i)`), sign-invariant loss `1−‖t̂·t*‖`; masked at branch/high-multiplicity |
| 14–39 | short SAME/MUTEX (13 offsets×2) | per offset: SAME=1 if both in one instance's medial band, nearest-skel points joined by a short local arc (`d_S(π_u,π_v) ≤ 1.35‖x_u−x_v‖+2·s_xy`), corridor crosses no foreign band (tube radius `0.35·min(r_u,r_v)`); MUTEX=1 if different instances or corridor pierces a foreign band; else DEFER/ignore |
| 40–42 | vetoes | `q_multi` (≥2 orientation modes >35° apart), `q_foreign` (≥2 instance medial bands local), `q_branch` (≥3 arms of one skeleton leave the nbhd) |

Plus a **sparse skip-edge head** (2 logits/queried edge over the `|o|∞=2` half-shell; not in
the 43). DESIGN.md also carries: the link→grow→fill decode mapped to `dev/nisb/`; the Phase
0–3 ladder with the reply's gates; the honest forecast (central 0.792, stretch 0.812; deltas
off 0.627); the G0(oracle)↔Phase-2(learnability) boundary; and the `tile_0_0_2`
second-chunk recommendation.

## Files and Areas

| File | Purpose |
|---|---|
| `dev/nisb/scripts/mesa/DESIGN.md` | MESA-EM mapped to `dev/nisb/`; channel formulas; Phase 0–3 ladder; forecast |
| `dev/nisb/scripts/mesa/common.py` | assets/loaders; `load_nerl_graph`+`get_nodes_position` paint; GT-SDT; offset banks; both scorers; closed-form ΔNERL + its validator |
| `dev/nisb/scripts/mesa/g0_phase0_oracle.py` | G0 decode + graph-level perturbation audit |
| `dev/nisb/scripts/mesa/g1_banis_feasibility.py` | G1: generator recall, geometric matching, oracle upper bound, realizer fidelity |
| `dev/nisb/scripts/mesa/gt_sdt_center.h5` | cached center-chunk GT-SDT |
| `dev/nisb/scripts/mesa/results_g0.md` / `results_g1.md` | numbers + verdicts + go/no-go |
| `.agent/features/nisb_base_60-80_plan/artifacts/DESIGN.md` | mirror |

Read-only (import/adapt): `dev/nisb/waterz/per_chunk_nerl.py`, `connectomics/metrics/nerl.py`
(`load_nerl_graph`, `compute_nerl_score_details`, `import_em_erl`),
`connectomics/decoding/decoders/segmentation.py` (`decode_affinity_cc`),
`scripts/centerline_graph_decode_gt.py`, `scripts/make_gt_sdt_region.py`,
`scripts/ec_endpoint_bridge.py`, `scripts/center_chunk_decode_sweep.py`, `ec_merge/ec_common.py`,
the cached LUT + affinity, `data/center_chunk/*`, `x2diag/gt_sdt_region_ero*.h5`.

## Verification Plan

Every number verified by output, written to the results md.
1. **Sanity (blocking):** A cached-LUT 0.601431±0.0005 (+0.545±0.001 if present); B measured
   `BASE_NERL`≈0.836±0.003; transform ≥99% in-seg. Fail ⇒ stop + finding.
2. **G0:** offset-bank sweep × C0–C3; pass = the §"G0 pass" thresholds; report smallest bank,
   per-condition table, `foreign_corridor_rejected`, the 0-cross-GT assertion.
3. **G1:** (1) recall raw-max + capped (count + ΔNERL-weighted) with the ΔNERL closed-form
   validated on 20 events; (2) geometric-matching precision/merges/NERL; (3) oracle
   recovered-fraction; (4) realizer control (0.005 / 0-cross-GT / 0-multi-ID / coverage).
   Pass = §"G1 pass".
4. **Go/no-go** in `results_g1.md`, three outcomes: **go** iff EVERY G0+G1 gate passes → train
   the heads, naming highest-value channels (and reporting the geometry-only precision gap the
   learned scorer must close); **gray zone** → no-go + named cheap refinement (e.g. add
   `tile_0_0_2` if low-headroom, or learned scorer if geometry-only walls); **kill** → stop on
   the failing gate. Always emit a result-location line even if G0 failure blocks
   `results_g1.md` (reason into `results_g0.md`).

**Resource strategy:** paint once; perturbations graph-level (no re-grow); full watershed +
`compute_nerl_score_details` only for C0 per bank + the chosen bank + the ≤6 G1 relabels
(`BASE_SEG`, full_oracle, capped_oracle, matched, realizer control, +grown); per-event ΔNERL
closed-form (validated), never a full re-score per event; realizer on the fragment-adjacency
graph (O(frags)); masked watershed/geodesic; dtypes uint32/float16→float32 per-op; both gates
run a 256³ smoke first and print numbers immediately; a backgrounded full run is monitored to
a successful exit + validated non-empty output before results are finalized; any hang/crash
surfaced with the result path.

## Risks and Questions

- **G0 proves the DECODE, not learnability** (GT-derived identity); Phase-2 learnability
  deferred — the G0↔Phase-2 boundary is stated in `DESIGN.md`. (Ratified: G0 stays the 1-voxel
  oracle.)
- **Center-chunk split headroom is small** (base 0.836, ceiling ~0.90, oracle gain ~+0.06);
  if `full_oracle` gain <0.02 the recovered-fraction gate is unreliable → G1 flags it and
  recommends `tile_0_0_2`. Recall (scale-invariant) remains the primary G1 signal.
- **Geometry-only matching is expected to wall** (~0.67 precision); G1 reports this as the
  motivation for the learned scorer rather than a failure.
- **ΔNERL closed-form** is validated against the canonical scorer on 20 events before being
  trusted; fallback to per-event full-score if it disagrees.
- **Channel-axis convention** validated end-to-end by Sanity B and the realizer control.
- **Untracked review surface:** deliverables untracked; code review embeds file contents +
  `git status` delta. Coder edits no tracked code and commits nothing.

## Changes Since Previous Plan Version

Addresses `plan_v2_review.md` (raw: `state/plan_v2_review.review.raw.md`):

- **[ratified, not a change] G0 medial-band/full-Phase-0 vs 1-voxel oracle →** human decided
  the 1-voxel proven-ancestor oracle; `DESIGN.md` holds the learned medial-band spec for
  Phase-2. Reviewer to assess G0 under this constraint, not re-litigate it.
- **[major] Sanity A scope →** now scores the **cached node-LUT** (seconds), never decodes the
  12.15 B-voxel volume; cc0.75 nonblocking if its LUT is absent.
- **[major] B3 offset count →** corrected to **cumulative 62** (B0's 13 ∪ the 49-offset shell),
  with all four banks enumerated (13/16/24/62).
- **[major] realizer conflation →** split into four separated measurements: (1) generator
  recall, (2) **real geometric-matching** merge-safety (precision + merges, diagnostic), (3)
  oracle-confirmed **upper-bound** recovered-fraction, (4) **realizer fidelity** on the
  fragment-adjacency graph (bounded, reverse-step affinity, runner-up margin log9, abstain
  label, atomic-SV). Control uses measured `BASE_NERL`, not rounded 0.836. Failure defined when
  `full_oracle` gain ≤0.
- **[major] 450 M-voxel geodesic →** replaced by the O(frags) fragment-adjacency realizer with
  a peak-resource estimate.
- **[major] G1 oracle-event multiplicity →** events are **break clusters** (all cut edges per
  fragment pair, a break *set*); hit = link + a tip within 45 nm of any break in the cluster.
- **[major] ΔNERL normalization →** pinned `2·L_a·L_b/D` with `L`, `D` defined and **validated**
  against the canonical scorer on a deterministic 20-event sample.
- **[major] proposal-gate constants →** radius estimator (K=12 boundary-EDT), `D_lim`
  equation, tangent/radius/Hermite thresholds (proposal vs acceptance), degeneracy handling.
- **[major] C1/C2 + metric denominators →** C1 restricted to erlgraph **degree-2** nodes; C2
  defines occupied sections + **all** cut-strand anchor pairs; fragments/GT denominator = the
  **626 scored skeletons**; retained-length numerator = physical `edge_len` fraction with
  shared grown label.
- **[major] channel binding →** the full formula table is pinned **in this plan**; `DESIGN.md`
  elaborates but no longer holds undefined promises.
- Prior resolved items (affinity/scorer binding, `get_nodes_position`, top-8 cap, surrogate
  framing, runtime/verdict, read-only prior art, second-chunk) retained.

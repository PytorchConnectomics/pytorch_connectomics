# MESA-EM cheap-gate results + go/no-go (coordinator gate verification)

Full numbers: `dev/nisb/scripts/mesa/results_g0.md`, `results_g1.md`. Bench = standard center
chunk (tile_1_1_1, 1000×1000×450, 626 local skeletons). Harness sanity exact throughout
(whole-vol cc0.66=0.601431470, cc0.75=0.545357139; center BASE_NERL=0.835517; transform 100%).

## G0 — Phase-0 oracle plumbing (GT-correct edges): representation VALIDATED, Z-skip is the wall

Given GT-correct same-instance edges, the 1-voxel centerline-graph decode + GT-SDT grow
reconstructs the chunk at **NERL 0.9936, fragments/GT 1.00, 0 false merges, 99.7% retained** —
≈ the 0.994 lesson-20 ceiling. This confirms the core thesis: connected per-instance seeds make
the field near-perfect, so the whole 0.60→0.80 gap lives in producing correct connectivity, not
in the representation or the realizer.

Gap-closure by offset bank (perturbation audit):

| bank | offsets | 1-voxel gap (C1) | 1-Z-section gap (C2) |
|---|---:|---:|---:|
| B0 | 13 | 6.7% | 0.0% |
| B1 | 16 | 16.3% | 2.7% |
| B2 | 24 | 19.0% | 9.9% |
| B3 | 62 | **100.0%** | **45.0%** |

**Verdict: gray-zone.** No bank meets C2≥90%. Single-voxel breaks are fully bridgeable; a full
missing Z-section (anisotropic Z@20nm, with lateral drift) is out of reach even for the `|o|∞=2`
shell. **Design correction:** the learned sparse skip-edge head needs wider, Z-aware reach
(larger Δz + lateral tolerance), not just the local shell. Z-section linking is the hard part.

## G1 — banis+ feasibility (no retrain): KILL, but BENCH-INVALID + 2 generator findings

- **The center chunk is not a valid split-healing bench.** Full split-oracle gain = **+0.0366**
  (0.8355→0.8721), only **7 oracle events**. cc3d@0.66 has near-split-solved this easy interior
  chunk; residual loss is coverage/merges, not thin splits. The plan's low-headroom guard (<0.02)
  was set too low — 0.037 is still far too little to test a split-healer. (Consistent with
  lesson_standard_smallscale: center chunk ceiling ~0.90 is coverage-limited.)
- **Generator misses sub-12-voxel crumbs.** 2814/4197 fragments (67%) are <12 voxels → no tips →
  the tip-based generator can't propose links for exactly the thin shatter the project targets.
  Generator recall 0/7. This is a real, transferable limitation independent of the bench.
- **Geometry-only matching walls:** 34 accepted links, **precision 79.4%, 7 false merges**, NERL
  barely moves (0.8355→0.8355). Confirms geometry alone is not merge-safe (lesson 17) → the learned
  PairNet/BiLOQ is necessary, not optional. (ΔNERL closed-form validated to 1.5e-6.)
- Realizer adopted 0 orphans (100% abstention) — a no-op on this near-solved chunk; re-verify on a
  real bench (possible margin/eligibility issue).
- Data realities surfaced (both required coordinator hotfixes to g1_banis_feasibility.py; the 256³
  smoke used a clean sub-crop that hid them): **13 baseline cross-GT fragments** (cc3d@0.66 is not
  merge-free locally) and **2 fragments spanning multiple local skeletons** (CC-split artifact).
  Fixes: incremental merge-safety gates; dominant local-skeleton owner.

**Verdict: KILL on this bench, but not a refutation** — the bench lacks split headroom, so the
linker/realizer can't be fairly evaluated here.

## Combined go/no-go

**NO-GO for training the full MESA-EM head stack right now — but the direction is de-risked and the
next steps are specific:**

1. **The representation is validated** (G0 decode ≈0.994 with correct edges). Keep the
   skeleton-graph → link → grow framing.
2. **Fix the skip-edge design for Z-section drops** before training — the fixed local-offset shell
   caps at 45% on C2. This is the concrete model change G0 demands.
3. **Fix candidate generation for sub-12-voxel crumbs** — tip-based proposal misses 67% of
   fragments; the thin shatter needs a crumb-aware generator (e.g. medialness-node seeds, not just
   farthest-point tips).
4. **Re-bench G1 on a split-dominated chunk (tile_0_0_2)** before trusting any feasibility verdict.
   The center chunk is fine for G0 (decode plumbing) but wrong for G1 (split-healing feasibility).
5. Geometry alone is not merge-safe (79% precision / 7 merges) → the learned PairNet/BiLOQ is
   required; plan accordingly.

Cost note: G0 ran ~4 h (62-offset graph over 491k nodes + four 450 M-voxel watersheds) — heavier
than a cheap gate should be; optimize (or cache) before reuse. G1 ran ~5 min.

---

## Addendum — G1 re-benched on split-dominated tile_0_0_2 (origin 0,0,900)

Harness parameterized via env (`MESA_CHUNK_ORIGIN/SEG/SKEL`, `MESA_BASE_NERL_EXPECT=none`);
data `dev/nisb/data/tile_0_0_2/` (extracted + local erlgraph built: 90 GT → **271 local
skeletons**, 419k nodes). Full numbers: `dev/nisb/scripts/mesa/results_g1_tile002.md`.
BASE_NERL=0.3244 (matches recorded 0.325; parameterization verified correct).

**Decisive finding: crop-level split-healing feasibility is confounded by the local-NERL
CC-split artifact — it can't be validly cheap-gated on a single crop.**
- The split-merge oracle gain is only **+0.017** even here (vs +0.037 on the easy center
  chunk), because cc3d(GT[crop]) shatters 90 GT neurons into 271 local skeletons; a neuron
  broken into N fragments spans multiple LOCAL skeletons, so the same-local-skeleton oracle
  can't merge them and local-NERL penalizes it (the [[zebrafinch_local_nerl_ccsplit_artifact]]).
  Only **5 events** on a 2660-fragment chunk — the link headroom is an artifact floor, not the
  real +0.171 whole-volume heal prize.
- Geometry-only matching precision **improved** on the harder chunk: **90.5% / 2 merges** (vs
  79% / 7 on center) — the matcher is sounder than the center read implied, but still not
  merge-safe and recovers no length (crumbs).
- Confirmed on BOTH benches (bench-independent, real bugs): tip generator misses **71%** of
  sub-12-voxel crumbs (no tips → 0 recall on the thin shatter); realizer **abstains 100%**
  (0 orphans adopted) — a likely realizer defect.

**Revised go/no-go — still NO-GO to train, with a sharper path:**
1. **Do NOT cheap-gate split-healing feasibility on a single crop.** Measure it WHOLE-VOLUME
   (where the real 0.601→0.772 heal headroom lives, un-artifacted), OR build events/oracle on
   DENSE-GT identity, OR apply the crop_split `--break-threshold-nm 1000` fix
   ([[zebrafinch_local_nerl_ccsplit_artifact]]) before trusting any crop recovered-fraction.
2. Fix the crumb-aware generator (medialness-node seeds, not farthest-point tips) — 71% of
   fragments get no candidates.
3. Fix / diagnose the realizer 100%-abstention (log9 margin or orphan-eligibility bug).
4. Geometry alone still walls (2 merges even at 90% precision) → learned PairNet/BiLOQ required.
5. G0 remains the solid positive: representation validated (0.994 decode); make the skip head Z-aware.

---

## Addendum 2 — de-artifacted re-measurement on tile_0_0_2 (the recommended fork (a))

New tooling (both reusable for any crop bench):
- `dev/nisb/scripts/mesa/deartifact_erlgraph.py` — rewrites an existing crop
  `.erlgraph.npz` by unioning local skeletons that share a dense GT label and come
  within `--break-threshold-nm` (kd-tree nearest node), carrying vertices/edges over
  verbatim. No re-skeletonisation (38 s, not a kimimaro rerun) and total centerline
  length is preserved bit-exactly. tile_0_0_2: **271 → 166 (1000 nm) → 132 (1600 nm)
  → 90 (inf = one skeleton per GT label)**, all 271 skeletons resolved to a GT label.
- `dev/nisb/scripts/mesa/artifact_sweep.py` — one cc@0.66 decode, scored against every
  graph variant. Full table: `results_artifact_tile002.md`.
- `g1_banis_feasibility.py` gained `MESA_DENSE_IDENTITY=cc|label` (default `cc`,
  unchanged). Merge-safety accounting had the SAME confound as the scorer: it judged
  against `cc3d(GT)`, so healing a weave-out split was itself counted as a false merge
  (which also deflated the 90.5 % matching precision). `label` pairs with a `_bt*.npz`.

| graph | skeletons | base NERL | same-skeleton oracle | gain | events |
|---|---:|---:|---:|---:|---:|
| `seg.erlgraph.npz` (as gated) | 271 | 0.324435 | 0.341853 | +0.017418 | 5 |
| `seg.erlgraph_bt1000.npz` | 166 | 0.316766 | 0.352839 | +0.036074 | 5 |
| `seg.erlgraph_bt1600.npz` | 132 | 0.313016 | 0.356786 | +0.043770 | 5 |
| `seg.erlgraph_btinf.npz` | 90 | 0.311045 | 0.359072 | +0.048027 | 5 |

The threshold-0 row reproduces the recorded G1 numbers exactly (0.324435 / 0.341853 /
5 events, 2660 instances) — harness check passes, nothing else moved.

**Finding 1 — the artifact is real but is NOT the whole story.** De-artifacting raises
the measured link headroom **2.8x** (+0.0174 → +0.0480), monotonic in the threshold. But
+0.048 is still nowhere near the +0.171 whole-volume heal prize. Base NERL *falls* as
arcs merge (0.3244 → 0.3110) because ERL is length^2-weighted: joining arcs lengthens the
skeleton faster than it lengthens any single correct run. That weighting is exactly why a
crop under-represents the linking prize **structurally**, artifact or no artifact — an
in-crop neuron is ~9 um, so the quadratic reward for joining its fragments is small.

**Finding 2 — G1's event set is the wrong target set.** The event count is **5 at every
threshold**. An `Event` is a *contact* split (one skeleton edge whose two endpoints land
in two different labelled fragments). With 29 044 / 419 148 background nodes (6.9 %),
essentially every real split here is a **gap** split — the centerline runs through
background between the fragments — which generates *omitted* edges, not events. So the
0/5 generator-recall gate was scored against 5 contact splits while the actual headroom
comes from merging fragments that are mostly not adjacent. The generator-recall KILL is
therefore not interpretable as stated, independently of the bench.

**Finding 3 — the loss decomposition says tile_0_0_2 is not split-limited.** LUT-only
counterfactuals on the cached cc@0.66 decode (`loss_decomposition.py`, full table
`results_loss_tile002.md`); `fill` = background skeleton nodes inherit the nearest
labelled fragment along the centerline (perfect growing), `link` = union-find merge of
every fragment on one skeleton, `safe_link` = each fragment relabelled to its
majority-overlap GT label (`connectomics.metrics.oracle.oracle_merge_segmentation`,
merge-safe: it does NOT chain two neurons through one impure fragment).

| graph | base | fill | link | fill+link | safe_link | safe_fill+link |
|---|---:|---:|---:|---:|---:|---:|
| `seg.erlgraph.npz` (271) | 0.324435 | 0.357678 | 0.341853 | 0.390504 | **0.066797** | 0.079232 |
| `seg.erlgraph_bt1000.npz` (166) | 0.316766 | 0.349223 | 0.352839 | 0.404328 | **0.163259** | 0.188378 |
| `seg.erlgraph_btinf.npz` (90) | 0.311045 | 0.342916 | 0.359072 | 0.413441 | **0.375419** | 0.432874 |

Edge-length classes are identical for every graph: **correct 92.1 % | contact-split ~0.0 % |
omitted 7.9 %**. `safe_link` walking 0.067 → 0.163 → 0.375 as the graph is de-artifacted is
the artifact proven end-to-end: on the CC-split graph one GT label owns several local
skeletons, so em_erl calls the *correct* identity assignment a merge and zeroes it.

**Finding 4 — the tile is MERGE-limited, by two segments.** `merge_damage.py` on the
correct (bt-inf) graph: under `safe_link`, **4 of 90 skeletons are zeroed and they carry
56.7 % of the sum-of-squares NERL mass**, from exactly two merging segments:

- `seg 45` <- skel 31 (95.3 um, 5631 nodes) + skel 87 (222.7 um, **3 nodes**)
- `seg 11` <- skel 58 (38.2 um, 119 nodes) + skel 63 (**1210.3 um**, 57 948 nodes)

Both are asymmetric knife-edges: at `merge_threshold=1` a **3-node** contact zeroes a
222.7 um neuron, and a 38 um neurite zeroes a 1.21 mm one. The chunk was selected as
"split-dominated" because its local NERL was low (0.325) -- but low local NERL != splits.
A split-healer has at most +0.064 to win here (0.311 -> 0.375); those two merges are
worth up to +0.567.

**Finding 5 — the metric is NOT a knife-edge whole-volume.** Re-scoring the cached
whole-volume node LUTs at several `merge_threshold` values (no decode):

| merge_threshold | 1 | 2 | 5 | 10 | 25 | 50 |
|---|---:|---:|---:|---:|---:|---:|
| cc0.66 (416 skeletons) | 0.601431 | 0.617232 | 0.622876 | 0.624366 | 0.628024 | 0.628024 |
| cc0.75 | 0.545357 | 0.546320 | 0.546320 | 0.546320 | 0.546320 | 0.546320 |

Whole-volume the strict rule costs only **+0.027** (4.4 % relative); on a 90-skeleton crop
it costs 56.7 %. The canonical 0.601431 baseline is robust -- the 0.60 -> 0.80 program is
NOT chasing a metric artifact. (The harness hardcodes `merge_threshold=1` throughout;
em_erl's own `volume_eval.py` defaults to 50.)

## Revised conclusion — crop-level cheap-gating is structurally invalid here

Three independent, now-quantified reasons, only the first of which is fixable:

1. **CC-split artifact** — `cc3d(GT[crop])` shatters 90 GT into 271 local skeletons; any
   GT-identity oracle collapses (merge-safe ceiling reads 0.067 instead of 0.375).
   FIXED by `deartifact_erlgraph.py` (use `_bt1000`/`_btinf` for any crop bench).
2. **Length^2 truncation** — an in-crop neuron is ~9 um, so ERL's quadratic reward for
   joining its fragments is small: link headroom reads +0.048 on the crop where the
   whole volume shows +0.171. Not fixable by any graph repair.
3. **Merge-mass concentration** — with only 90 skeletons, 2 neurons hold 57 % of the mass,
   so one 3-node contact dominates the score. Whole-volume the same rule costs +0.027.

**Do not re-run G1 on a crop.** Its `Event` = a *contact* split, and this data has ~0 %
contact-split length (7.9 % omitted, i.e. gap splits) — which is why the event count was 5
at every de-artifacting threshold and generator recall was 0/5 regardless. Re-running G1
with `MESA_DENSE_IDENTITY=label` + `_btinf` would still score recall against those same 5
contact events and produce another uninterpretable KILL. The next measurement has to be
**whole-volume** (fork (b)), where all three distortions vanish.

---

## Addendum 3 — the whole-volume measurement (fork (b)), and it costs 2 seconds

`whole_volume_decomposition.py`. **No decode is needed**: the cached whole-volume node
LUTs (`seg_fusion/oracle_lut/*_node_luts.npz`) already carry the full ERLGraph AND the
node->segment map, so every counterfactual is a LUT edit. 2 s per substrate (38 s for the
heavily over-segmented one) — versus ~4 h for G0 and ~5 min per crop G1 that we now know
cannot answer the question. Full table: `results_wholevol.md`.

416 skeletons, 791 035 nodes. `safe_*` = each fragment assigned to the skeleton owning
most of its nodes (merge-safe, the node-level analogue of `oracle_merge_segmentation`).

| substrate | base | fill | link | fill+link | safe_link | **safe_fill+link** | merge zeroing | oracle pairs |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| ch0-1-2 @0.66 | 0.601431 | 0.638634 | 0.729176 | 0.869656 | 0.766202 | **0.915004** | 26 skel / 7.0 % mass | 10 997 |
| ch0-1-2 @0.75 | 0.545357 | 0.641774 | 0.700862 | 0.981715 | 0.702427 | **0.984346** | 3 skel / 1.0 % | 12 926 |
| ch3-4-5 @0.75 | 0.325367 | 0.354444 | 0.683614 | 1.000000 | 0.683614 | **1.000000** | 1 skel / 0.0 % | 88 696 |

| substrate | correct len | split len | omitted len |
|---|---:|---:|---:|
| ch0-1-2 @0.66 | 91.1 % | 0.3 % | 8.5 % |
| ch0-1-2 @0.75 | 84.2 % | 0.3 % | 15.5 % |
| ch3-4-5 @0.75 | 72.7 % | 10.0 % | 17.3 % |

**Consistency check:** `base` reproduces the canonical 0.601431 / 0.545357 exactly, and
`safe_link` at cc0.66 = **0.766** independently reproduces the recorded 0.772 oracle-merge
ceiling (small gap = node-majority vs voxel-majority overlap).

**Finding 6 — the substrate, not the linker, sets the ceiling.** The threshold tuned for
*base* NERL is the wrong starting point for a link+grow program. cc0.66 has the best base
(0.601) but 7.0 % of the mass is destroyed by merges that no linker or grower can undo,
capping it at 0.915. cc0.75 gives up 0.056 of base and buys a 0.984 ceiling. And
**ch3-4-5 @0.75 has NO false merges at all** — `safe_link` == chaining `link` to six
decimals, ceiling exactly **1.000000** — so its whole 0.325 -> 1.0 gap is addressable.
The cost is 88 696 link decisions vs 10 997 (8x the linking problem).
Realization fractions needed to hit 0.80: 63 % of headroom from cc0.66, 58 % from cc0.75,
70 % from ch3-4-5. Similar realization burden, wildly different ceilings and problem sizes.

**Finding 7 — growing is not optional.** `link` alone reaches 0.729 / 0.701 / 0.684, but
`fill+link` reaches 0.870 / 0.982 / 1.000. Coverage (8.5-17.3 % omitted centerline length)
has to be recovered for the linking ceiling to be realizable at all. Meanwhile contact
splits are 0.3 % of edge length on the ch0-1-2 substrates (10.0 % on ch3-4-5) — so at
every scale the errors are GAP splits, and a tip/contact-based proposer structurally
cannot see them (Finding 2).

**Caveat:** these are node-LUT analyses, so "no merge damage" means no fragment fuses two
GT *centerlines*. That is exactly what ERL scores, but it is weaker than a dense
voxel-majority purity check.

## What to do next

1. **Make the whole-volume decomposition the first gate for any substrate.** It is 2 s and
   it dominates every crop-level cheap gate we built. Any new decode should report
   `safe_fill+link` before anyone designs a head for it.
2. **Pick the substrate deliberately.** ch3-4-5 @0.75 (ceiling 1.0, zero merge damage) is
   the honest target for a link+grow program; ch0-1-2 @0.66 caps the whole program at 0.915.
3. **Retire the crop gates for split-healing** (keep the center chunk for G0 decode plumbing).
4. Generator must be **gap-bridging**, not contact/tip-based, and the realizer must grow as
   well as link.

---

## Addendum 4 — Addendum 3 re-run on the OFFICIAL metric (2026-08-19)

Addendum 3 read the cached node LUTs as stored, i.e. `networkx_skeleton_to_erl_graph(
resolution=None)` — edge lengths in raw voxels, so every 20 nm z-step was weighted as 9 nm
(lesson V / `nisb_erl_resolution_bug`). `whole_volume_decomposition.py` now takes
`--resolution 9,9,20`, which recomputes `edge_len`/`skeleton_len` from the stored (x, y, z)
index positions. The rescale is owned by `dev/nisb/scripts/erl_graph.py::rescale_fields`
(generalised from the hard-wired cc0.66 loader to any `*_node_luts.npz`); the mesa script
delegates to it rather than re-rolling a third copy.

**Harness validation:** rescaled `base` at cc0.66 = **0.604428**, funlib-exact to six decimals.
Without the flag it still reproduces **0.601431**. Graph load, LUT indexing and the ERL kernel
are therefore validated end-to-end against the official evaluator.

| substrate | base | fill | link | fill+link | safe_link | **safe_fill+link** | merge damage | oracle pairs |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| ch0-1-2 @0.66 | 0.604428 | 0.642291 | 0.729993 | 0.869781 | 0.767103 | **0.915280** | 26 skel / 6.9 % | 10 997 |
| ch0-1-2 @0.75 | 0.547620 | 0.646165 | 0.701184 | 0.981863 | 0.702701 | **0.984458** | 3 skel / 1.0 % | 12 926 |
| ch3-4-5 @0.75 | 0.331087 | 0.361759 | 0.683778 | 1.000000 | 0.683778 | **1.000000** | 1 skel / 0.0 % | 88 696 |

| substrate | correct len | split len | omitted len |
|---|---:|---:|---:|
| ch0-1-2 @0.66 | 91.2 % | 0.4 % | 8.5 % |
| ch0-1-2 @0.75 | 84.1 % | 0.3 % | 15.5 % |
| ch3-4-5 @0.75 | 72.9 % | 9.8 % | 17.3 % |

**Every Addendum-3 conclusion survives.** Absolutes shift +0.002…+0.006, merge damage moves
7.0 % → 6.9 %, and the realization fraction needed to reach 0.80 is unchanged at
62.9 % / 57.8 % / 70.1 %. Findings 6 (substrate sets the ceiling) and 7 (growing is not optional,
gap splits dominate) hold verbatim.

**Scope note:** only the *whole-volume* tables needed this. The crop harness (`mesa/common.py`)
already loaded its graphs at (9, 9, 20) nm — verified directly on `data/tile_0_0_2/seg.erlgraph*.npz`
(coords max 8991 nm = 999 × 9; longest skeleton 1 210 288 nm = the 1210.3 µm neuron of Addendum 2
Finding 4) — so Addenda 1–2 and the µm figures in them are already on the official metric.

Tables: `dev/nisb/scripts/mesa/results_wholevol_nm.md` (canonical), `results_wholevol.md` (voxel,
historical; both now carry an explicit units line). Lessons:
`projects/2026_nisb_base/lessons/lesson_mesa_gates.md` (#24) and
`lesson_wholevol_decomposition.md` (X).

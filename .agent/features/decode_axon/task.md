# Task: decode_axon.py — unified strong/weak, 2D+3D, split-then-link axon decoder

Design and implement `dev/mit_liconn/decode_axon.py`: one modular pipeline that decodes the LiCONN
DL288B crop1 model affinity `(3, 800, 1024, 1024)` into an axon instance segmentation, consolidating
every component validated this session. GT-free at inference; the GT is heavily over-split and is NOT
trusted for scoring (see metric below). Beat the current best `decode_p1p2`.

This is an ORCHESTRATION + a few NEW pieces, not a rewrite: reuse the validated helpers in
`dev/mit_liconn/` as imported functions; add only the strong/weak split, weak-region sectioning,
hysteresis link, and crumb cleanup. Match existing `dev/mit_liconn/` style (module-level REPO/paths,
`--full`/`--zslice`/`--self-test`, flush prints, small surgical functions).

## Architecture (user sketch, improved — implement this)

The user's "step N" numbers refer to the `decode_v2.py` pipeline stages:
- **steps 1-3** = build 2D sections: load affinity → `waterz_2d_spacefill(thr=0.3)` per z → `section_index`.
- **steps 4-6** = link sections into tubes: raw consecutive-slice IoU table → `conservative_pairs(0.2)`
  + `bb_pairs(0.3)` (best-buddy, NO z-contact dominance, NO EDT fill) → apply links.
- **steps 7-8** = force-split fused sections: `detect_fusions` (exactly-2, area-matched N→1) +
  `eligible_fusions` (two independent deep upstream components) → `force_split` (seeded watershed,
  propagated down the run) → `constrained_relabel` (re-link with per-split mutex constraints).

### a) Strong / weak foreground split
- **strong** = max-over-channel affinity `> 0.66` (this is today's section threshold `AFF_BG=0.66` in
  `decode_lib.waterz_2d_spacefill`; today's sections ARE the strong fg).
- **weak** = `0.3 < aff ≤ 0.66` (currently discarded as background).

### b) Two decode tiers
1. **2D-seg-based (decode_v2 steps 1-8), split-then-link order:** build 2D sections (1-3) → force-split
   fused sections (7-8) → link the clean sections into tubes (4-6). (In practice force-split needs a
   base link to compute upstream components, so the concrete order is base-link → detect+force-split
   eligible fusions → final constrained re-link, exactly `decode_v2._run_pipeline`'s a→split→c flow.)
2. **3D-seg-based for INCOMPLETE tubes (touch < 2 volume faces), with a PARALLEL-axon veto** — this is
   today's `decode_p1p2` (SPLIT first, then MERGE; merge-first was measured wrong):
   - **Prob-1 SPLIT** starting from each incomplete tube's first/last slice: through-intruder carve
     (`decode_v4_split.py`) + bidirectional orphan-endpoint extension (`tube_extend.py`) — both gated
     by the validated **bump-safe rule** (commit a carve only if it removes a host bump without adding
     one to host or orphan; measured 27 accept / 97 reject, strictly bump-safe).
   - **Prob-2 MERGE** for incomplete tubes: recompute a region graph on the 3D seg using the RAW
     affinity (BEFORE bg zero-out — kept high on purpose so the 2D pass over-splits), and merge a pair
     iff BOTH are incomplete AND boundary affinity high (≥0.5, contact ≥20 vox) AND area-matched AND
     their z-ranges do NOT overlap (`compute_bbox_all_3d`, `--max-zoverlap 0.5`: a real over-split is
     sequential in z, parallel axons are concurrent) (`decode_v3_merge.py`).

### c) Modular functions
Break the pipeline into small, independently testable functions (sectioning, linking, force-split,
prob1-split, prob2-merge, weak-recovery, crumb-cleanup, orchestrator). Add `--self-test` data-free
contract tests in the `decode_v2.py` style.

### d) Section building per region type
- **strong regions:** decode_v2 steps 1-3 (`waterz_2d_spacefill`, `thr=0.3`, `aff_bg=0.66`, `fill=0`).
- **weak regions:** per 2D slice, `fastmorph` multi-label opening + `cc3d` connected components on the
  `0.3 < aff ≤ 0.66` band (opening removes salt/bridges between weak blobs before CC).

### e) Order of application
Run the full pipeline on the STRONG region first (tiers b1 then b2), then the WEAK region.

### f) Strong↔weak hysteresis
Weak sections are used ONLY to LINK — i.e. bridge two STRONG tubes that are otherwise split by a weak
gap. Weak regions never seed new independent axons and are never linked weak-to-weak into a standalone
object (hysteresis: strong = seed, weak = grow only to reconnect split strong tubes). This is the guard
against weak-region bridging re-introducing merges among the packed parallel axons.

### g) Crumb cleanup
Final pass: remove/absorb small or orphan segments (dust below a size floor; orphan = touches < 2 faces
and below a volume floor), so the output isn't polluted by fragments.

## Evaluation (GT-free; primary)
`dev/mit_liconn/valid_tube_metric.py` — a correctly decoded axon crosses the crop (touches ≥ 2 of 6
faces) AND is a single unbranched tube (no per-slice-area bump, not a persistent side-by-side parallel
merge). Report: VALID (cross & clean & single) count + **volume %**, total bumps, PARALLEL-merge segs.
Secondary: `dev/mit_liconn/liconn_nerl.py` (NERL vs GT skeletons) — informational only, GT is over-split.

## Baseline to beat (GT-free valid-tube metric, corrected: faces≥2 + bump + parallel detector)
```
decode_v2:   VALID vol 53.5%  bumps 152  PARALLEL 9
decode_p1p2: VALID vol 62.7%  bumps 117  PARALLEL 8   <- current best (guarded split-then-merge)
```

## Deliverable
`dev/mit_liconn/decode_axon.py` with `--full` / `--zslice Z0:Z1` / `--self-test`, producing
`outputs/mit_liconn/DL288B_crop1/decode_axon.h5` and printing the valid-tube metric (+ NERL in `--full`).
Reuse: `decode_lib` (waterz/link primitives), `decode_v2.{force_split via force_split_decode, detect_fusions,
eligible_fusions, constrained_relabel}`, `decode_v4_split` (Prob-1), `decode_v3_merge` (Prob-2),
`tube_extend` (bump-safe gate), `valid_tube_metric`. Do NOT re-implement waterz, linking, or the metric.

## Success criteria
- Runs end-to-end on the full 800³ volume (inline ~5–8 min, or SLURM `-p short -c 8 --mem 90G`).
- Valid-tube **volume ≥ 62.7%** (goal: exceed via weak-region coverage) with **PARALLEL-merge ≤ 8** and
  **bumps ≤ 117** — i.e. gain coverage without adding parallel false merges or new contamination.
- Each new stage is behind a flag and independently ablatable; `--self-test` passes; report the metric
  for strong-only, strong+3D-cleanup, and strong+weak so the weak tier's contribution is isolated.
- If weak recovery can't beat 62.7% net (adds merges), it must be OFF by default and the strong+3D
  result must at least match `decode_p1p2` (no regression).

## Key facts / constraints (measured this session)
- thr-0.3 strong sections are GT-pure; merges are a LINKING problem, not a sectioning/model problem.
- Force-split ALONE regresses NERL (0.593→0.553) unless the re-link keeps split pieces separate — the
  per-split mutex (`decode_v2.constrained_relabel` / `ConstrainedUnionFind`) is LOAD-BEARING.
- Prob-2 merge WITHOUT the z-overlap parallel veto fused 584 parallel axons (guardless v3); the veto is
  mandatory.
- decode_p1p2 order is SPLIT then MERGE. Merge-first fused parallel axons (36620/26061 blobs).
- Weak-region recovery on packed axons risks new merges → strict hysteresis (bridge split strong only),
  and the bump-safe gate on any carve/extension.
- Bounds: even perfect linking leaves ~9.5% axons missed (thin-axon coverage floor, a model-recall
  limit, not decode); ~69% of missed axons are no-signal (weak-0.3 can't recover them).

## Environment
env `pytc` (`source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc`); affinity
`datasets/mit-liconn/raw_x1_head-aff_r1.h5` (`main`, CZYX, ch0=z ch1=y ch2=x); cached strong sections
`outputs/mit_liconn/DL288B_crop1/sections_thr0.3_z0-800.h5`; outputs in
`outputs/mit_liconn/DL288B_crop1/`. `fastmorph` and `cc3d` are installed. Node a002 has 57 GB free / 7
cores; the 3D post-passes run inline in 1–3 min each.

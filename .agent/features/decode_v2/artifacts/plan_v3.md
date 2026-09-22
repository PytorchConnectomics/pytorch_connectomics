# Plan v3

## Summary
`dev/mit_liconn/decode_v2.py` = tube-bb base → detect exactly-2-incoming area-matched fusions →
force-split ELIGIBLE fusions (component-based upstream independence) with propagated seeds,
one-to-one terminal matching, exclusive ownership → relink with a seed-anchored constrained
union-find guaranteeing separation AND per-piece/terminal connectivity → in-script metrics +
no-regression selection. This version pins the remaining plan_v2 edge-cases (eligibility
convergence, terminal handling, UF-init accumulation, detection reproducibility, slice-mode,
verification gate consistency, SLURM syntax).

## Scope
New `dev/mit_liconn/decode_v2.py` + `dev/mit_liconn/sbatch_decode_v2.sh`; minimal extraction of
`force_split(...)` in `force_split_decode.py`; append results to `dev/mit_liconn/split_merges.md`
after the full run (now listed, fix minor). Out of scope: weak-recovery, model/GT/affinity changes,
git commits, N>2 splitting (skipped+counted), other refactors.

## Proposed Changes
**Node identity.** `waterz_2d_spacefill` labels are volume-unique (per-slice offset). ASSERT at load
(each label in exactly one z; else abort). Use the integer label as the single global node id
everywhere; new pieces get ids `>= max+1`. `z_of[label]` kept for geometry only.

**Stage 1 — base + spine.** `segs_to_iou` → IoU edges; `conservative_pairs(0.2)` → spine graph;
`+bb_pairs(0.3)` → tube-bb ref seg `A`.

**Stage 2 — detection (fix #4).** Detection is on the RAW consecutive-slice overlaps from
`segs_to_iou`. `incoming[t] = {s : IoU(s,t) >= fuse_iou}`, `--fuse-iou` default 0.2 (a SEPARATE
param from the conservative-link 0.2; documented as independent). Candidate = `t` with EXACTLY 2
incoming and `|area(t)-(area(s1)+area(s2))| <= area_tol*area(t)`, `--area-tol` default 0.5. N>2
skipped+counted.

**Stage 3 — eligibility via upstream components (fix #1).** Remove section `t` and all links
incident to `t` from the spine graph; take connected COMPONENTS of the remainder. `t` is ELIGIBLE
iff `s1` and `s2` fall in DIFFERENT components (components are maximal ⇒ they never converge,
handling branching predecessors), and each of the two components spans `>= min_len` distinct
z-slices (AXIAL DEPTH, not node count), `--min-len` default 3.

**Stage 4 — force-split (fix #2,#3).** `force_split(seg2d, affxy, incoming, eligible_runheads,
area_tol, next_overlap) -> (split_seg, runs)`:
- Deterministic run-head order `(z_of[t], t)`. Exclusive global `owner: label->run_id`.
- START a run only if `t, s1, s2` are all UNOWNED; else deterministic conflict-SKIP (counted).
  Mark `t, s1, s2` owned on start (seed anchors owned — fix ownership consistency).
- Seeded watershed on `t` (markers = `s1,s2` footprints from z-1; ridge `1-mean(aff[1:],0)`) →
  pieces `l1,l2` (fresh ids), mark owned. `sideA={s1,l1}`, `sideB={s2,l2}`; `mand_links` seeds the
  chain `s1-l1`, `s2-l2`.
- PROPAGATE: at z+1, candidate next sections = those overlapping the current footprint `l1∪l2`;
  pick the single one with MAX overlap voxels whose overlap `>= next_overlap * area(l1∪l2)`
  (`--next-overlap` default 0.5), ties broken by smallest label id. If that next section `nt` is
  UNOWNED and area-matched to `(l1,l2)` and NOT itself a 2-incoming fusion: split `nt` → `l1',l2'`,
  mark owned, append to sides, extend `mand_links` (`l1-l1'`, `l2-l2'`), continue with `l1',l2'`.
- STOP causes, each explicit (fix #2 terminal):
  (a) next is a 2-incoming fusion → stop; current `l1,l2` are the terminals (owned, in sides).
  (b) axons separated — the two best next sections `n1,n2` (unowned) match `l1,l2` ONE-TO-ONE by
      max overlap (2×2 greedy, deterministic): assign each to its side, mark owned, add to sides +
      `mand_links` (`l1-n1`, `l2-n2`). Stop.
  (c) a single next section overlaps both but is NOT area-matched (ambiguous) → do NOT claim it
      (leave unowned, would lose one identity); stop; report. The constrained relink + IoU handles it.
  (d) no next / volume end → stop.
- Returns per run: `sideA,sideB` (owned labels incl. anchors, pieces, matched terminals),
  `mand_links` (must-union pairs threading each side incl. terminals).

**Stage 5 — constrained UF relink (fix #3).** Per-label annotation map `ann[label] = (run_id, side)`
built from side sets; ASSERT each label appears in ≤1 (run,side) (guaranteed by exclusive ownership;
abort otherwise). Component state = dict `run_id -> side`. Apply `mand_links` as FORCED unions first
(conflict-checked like any union; never conflict by construction, checked anyway). Then candidate
links `conservative_pairs(0.2)+bb_pairs(0.3)` on `split_seg`, ordered `(-IoU, min(u,v), max(u,v))`;
union `(u,v)` unless merging component dicts conflicts (same run→different side). Relabel → seg `C`.

**Stage 6 — metrics + selection (fix #5,#6).**
- In-script (import eval_inst, liconn_nerl) for a=tube-bb, b=force-split+PLAIN relink (no
  constraint), c=`C`: print `eval_inst.score` + (full mode) real NERL + oracle-merge NERL.
- Evaluate oracle-merge NERL on the UNLINKED `split_seg` substrate (fix #6); HARD-assert `>= 0.78`.
- Save `decode_v2_a_tubebb{tag}.h5`, `_b_plain{tag}.h5`, `_c_constrained{tag}.h5`.
- FULL mode: SELECTED `decode_v2{tag}.h5` = argmax(real NERL){a,c}; HARD guarantee selected NERL
  `>= max(recomputed a, 0.593)`; print candidate(c) + selected; banner if c<a. GT-based selection
  is EVAL-ONLY (a GT-free selector is future work).

## Files and Areas
| File | Change |
|---|---|
| `dev/mit_liconn/decode_v2.py` | NEW — stages 1-6, constrained UF, `--self-test`, in-script metrics, selection, CLI |
| `dev/mit_liconn/force_split_decode.py` | MINIMAL — add `force_split(...)` (single signature) with ownership/terminals/mand_links; keep `main` |
| `dev/mit_liconn/sbatch_decode_v2.sh` | NEW — `-p short -c8 --mem=120G` (fix #7), `source .../activate pytc`, `--full`, echo exact paths |
| `dev/mit_liconn/split_merges.md` | append the a/b/c + substrate results after the full run (fix minor) |
| reused unchanged | decode_lib, eval_inst, liconn_nerl, endpoint_link |

## Verification Plan
1. **`--self-test` (no data, <1s; non-zero exit on any failure):**
   - UF: direct A–B rejected; indirect A–X–B rejected; two runs cross-run-allowed/same-run-cross-
     side-rejected; same-side allowed; `(-IoU,min,max)` tie determinism.
   - split path (synthetic 3D graphs): (i) stale-seed — two interacting runs share a section →
     second conflict-skipped; (ii) consecutive/overlapping runs; (iii) terminal one-to-one match
     assigns terminals to correct sides; (iv) a terminal later encountered as a run-head →
     conflict-skip fires (fix #6 test); (v) repeated cross-slice label → assertion aborts; (vi)
     end-to-end split→relink CONNECTIVITY: every piece AND every terminal ends in a component
     containing its seed anchor, and no component holds both sides of a run (fix #2/#6).
2. **Pilot (instance-only):** `source .../activate pytc && python dev/mit_liconn/decode_v2.py
   --zslice 0:96`. `--zslice` crops GT to `[z0,z1)` with a shape assertion; reports INSTANCE
   metrics only for a/b/c (no NERL/selection in slice mode — no sub-volume skeleton graph); writes
   the CANDIDATE `decode_v2{tag}_z0-96.h5`. `--full` XOR `--zslice` required (error otherwise);
   `--tag` default "" composes before the z-suffix.
3. **Full (SLURM):** `sbatch dev/mit_liconn/sbatch_decode_v2.sh`. Report a/b/c real NERL, oracle-
   merge NERL, instances/FM-gt/FS-gt/missed, and unlinked-substrate oracle-merge NERL.
4. Append results to `split_merges.md`.

**HARD gates (fix #6 consistency — the ONLY pass/fail):** `--self-test` passes; full run completes;
SELECTED real NERL `>= max(recomputed a, 0.593)` (no-regression); unlinked-substrate oracle-merge
NERL `>= 0.78`. Everything else (c vs b, FM-gt(c) vs 278, instance counts) is REPORTING-ONLY.

## Risks and Questions
- Component-based eligibility can mis-classify a spine over-split of ONE axon as two components
  (false-eligible); the constrained UF keeps that split (minor real-decode split cost, none for
  oracle-merge). Conflict-skip may drop interacting merges (counted). Both acceptable.
- Ambiguous stop-cause (c) leaves a section unclaimed; the IoU relink + constraint resolves it,
  possibly imperfectly; reported.
- GT-based selection is eval-only; GT-free selector out of scope.

## Changes Since Previous Plan Version
Pins every plan_v2 review finding:
- #1 eligibility: connected-COMPONENTS after removing `t`+incident links, DIFFERENT components,
  each `>= min_len` AXIAL z-depth — resolves convergence and branching.
- #2 terminals/ownership: explicit stop causes (a-d); never assign a still-fused section to one
  side; one-to-one terminal matching on separation; terminals owned + in `mand_links`; seed
  anchors owned; every committed label owned exactly once.
- #3 UF init: per-label annotation with `<=1 (run,side)` assertion; forced unions conflict-checked.
- #4 detection reproducibility: `--fuse-iou` (0.2, separate param), `--area-tol` (0.5),
  `--next-overlap` (0.5) with max-overlap selection + smallest-id tie-break; clarified raw-overlap
  detection vs conservative-link threshold.
- #5 slice-mode: GT crop + shape assert; instance-only (no NERL/selection); candidate artifact
  name + z-suffix/tag composition; `--full` XOR `--zslice`; default `--tag ""`.
- #6 verification gate: single HARD gate set (self-test + no-regression selected + substrate
  `>=0.78`); `c>b` demoted to reporting-only; self-test adds terminal connectivity + terminal-as-
  run-head case.
- #7 SLURM: `--mem=120G` (valid syntax).
- minor: `split_merges.md` added to Scope + Files.

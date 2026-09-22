# Plan v0

## Summary
Add `dev/mit_liconn/decode_v2.py`: an integrated LiCONN decode that (1) builds tube-bb sections,
(2) detects fused merge cross-sections, (3) force-splits them with seeded watershed propagated
down each merge run, and (4) **relinks with a split-aware constraint** so the split pieces are
not re-merged by the linker. The load-bearing novelty over the existing `force_split_decode.py`
is step 4: force-split alone regresses real NERL (0.593→0.553) because plain conservative+bb
re-links the two axons back together. decode_v2 forbids cross-side links along each split run.
Everything else reuses existing helpers. Evaluate with NERL + oracle-merge NERL + instance
metrics; target real NERL ≥ tube-bb 0.593 (goal: exceed by retaining the force-split gain).

## Scope
In scope:
- New module `dev/mit_liconn/decode_v2.py` (one file) with CLI `--full|--zslice`, `--tag`,
  and knobs for the split/relink.
- A SLURM wrapper `dev/mit_liconn/sbatch_decode_v2.sh`.
- Reuse (import, do NOT re-implement): `decode_lib.py` (waterz_2d_spacefill, conservative_pairs,
  bb_pairs, getScoreFunc, params), `force_split_decode.py` split helpers (membrane/seeded
  watershed/propagation), `eval_inst.py` (score), `liconn_nerl.py` (evaluate/oracle_merge),
  `endpoint_link.py`/`detect_parallel.py` geometry as needed, waterz + em_util via existing paths.

Out of scope (do NOT do here):
- weak-0.4 / hysteresis coverage recovery (separate follow-up).
- Any change to the model, GT, or the affinity.
- Refactoring the existing dev/mit_liconn scripts beyond minimal extraction needed for reuse.
- Committing to git.

## Proposed Changes
`decode_v2.py` runs four stages on the thr-0.3 sections (load cached
`outputs/mit_liconn/DL288B_crop1/sections_thr0.3_z0-800.h5`, else build via
`waterz_2d_spacefill(aff, ones, thr=0.3, SMALL, AFF_LOW, rg_zero=RG_ZERO,
score=getScoreFunc('aff30_his256_ran255'), aff_bg=AFF_BG)`):

1. **tube-bb base (reference labels).** Not strictly needed for the pipeline but compute the
   plain spine+bb linking once for the metrics baseline row.

2. **Error detection (fused merge cross-sections).** From `segs_to_iou` over consecutive
   slices, build `incoming[t] = [s for s at z with IoU(s,t) >= 0.2]`. A fusion is any `t` with
   `len(incoming) >= 2` and `|area(t) - Σ area(s_i)| <= 0.5*area(t)` (area-matched, axons don't
   branch). Keep the two largest incoming `s1,s2` as the two sides. (This mirrors
   `detect_parallel.py`; 539 area-matched fusions, ~48 true 2-GT.) Do NOT membrane-gate —
   under the split-aware relink, over-splitting a same-axon fusion is recovered because both
   sides link back into the SAME tube (no forbidden pair fires across one axon).

3. **Force-split via propagated seeds.** For each fusion entry, seeded watershed on `t` with
   markers = `s1,s2` footprints projected from z-1 and ridge `1 - in-plane-affinity` (mean of
   aff channels 1,2), assign pieces `l1,l2`; propagate down the run (next single fused section
   whose overlap ≈ the two seeds, area-matched) carrying `l1,l2` as new seeds; stop at a
   2-incoming section (handled separately) or natural separation. Reuse the exact loop in
   `force_split_decode.py::main` (extract into a function `force_split(seg2d, affxy, ...)` that
   returns the split section image AND, per split run, the two **side sets** `A_ids`, `B_ids`
   (all piece labels on each side). This side bookkeeping is the new bit force_split lacks.

4. **Split-aware relink.** Build link candidates on the split sections with
   `conservative_pairs(min_iou=0.2)` + `bb_pairs(0.3)` (same as tube-bb). Then **drop any link
   (u,v) where u and v are opposite sides of the same split run** (u∈A_ids[k] and v∈B_ids[k] for
   some k). Union the surviving links with `merge_id`. This keeps each force-split axon separate
   while still de-fragmenting everything else. (v0 uses simple forbidden-edge deletion; a
   constrained union-find is a possible v1 if edges leak the merge back via a longer path.)

Output `outputs/mit_liconn/DL288B_crop1/decode_v2.h5`; print instance metrics via `score(seg,gt)`
and (in the SLURM wrapper) NERL via `liconn_nerl.py --eval` and `--oracle-merge`.

## Files and Areas
| File | Change |
|---|---|
| `dev/mit_liconn/decode_v2.py` | NEW — the 4-stage pipeline + CLI + save + `score()` print |
| `dev/mit_liconn/force_split_decode.py` | MINIMAL — extract the detect+split+propagate loop into `force_split(seg2d, affxy, iou..., return_sides=True)` returning split image + per-run A/B side id sets; keep the existing `main` working by calling it |
| `dev/mit_liconn/sbatch_decode_v2.sh` | NEW — SLURM (`-p short -c8 --mem120G`): run decode_v2 --full, then `liconn_nerl.py --eval` (base) and `--oracle-merge` |
| `.agent/features/decode_v2/` | CCC artifacts only |

Reused unchanged: `decode_lib.py`, `eval_inst.py`, `liconn_nerl.py`, `endpoint_link.py`.

## Verification Plan
1. **Pilot correctness (fast, login node):**
   `python dev/mit_liconn/decode_v2.py --zslice 0:96` — must run without error and print instance
   metrics. Sanity: labels > 0; false-merge not worse than tube-bb on the block; report #fusions
   detected and #sections split. (z0:96 has few full-volume merges, so mainly a smoke test.)
2. **Full volume (SLURM):** `sbatch dev/mit_liconn/sbatch_decode_v2.sh`. Collect:
   - real NERL (`liconn_nerl.py --eval decode_v2.h5`) vs tube-bb 0.593 (MUST NOT regress; goal >).
   - oracle-merge NERL (`--oracle-merge`) — expect ≥0.78 (the force-split ceiling), confirming the
     split substrate is intact.
   - instance metrics (`eval_inst.py --pred decode_v2.h5`): false-merge gt should be ≤ tube-bb 278
     and false-split not much worse than 62 (the relink constraint should curb the over-split
     regression seen in force-split-all: splits 92 → target closer to 62).
3. **A/B ablation printed by decode_v2:** report metrics for (a) tube-bb, (b) force-split + plain
   relink, (c) force-split + split-aware relink, so the constraint's effect is measured directly.
4. Numbers recorded in `dev/mit_liconn/split_merges.md` run log.

Success = full run completes, real NERL ≥ 0.593, oracle-merge NERL ≥ 0.78, and the split-aware
relink (c) beats plain relink (b) on real NERL. If (c) still < 0.593, that is an acceptable v0
outcome IF documented with the ablation showing the relink is the remaining knob.

## Risks and Questions
- **Re-merge via longer paths:** forbidden-edge deletion only blocks direct A–B links; the merge
  can reform if A and B reconnect through a third section. Mitigation/possible v1: constrained
  union-find (enemy sets) instead of edge deletion. Flag if (c) ≈ (b).
- **Watershed with no membrane:** merge points have weak membranes; the seeded watershed splits by
  geodesic distance from seeds — pieces may mis-assign boundary voxels. Acceptable because relink
  uses IoU on the pieces; note any instability.
- **Same-axon over-split noise:** ungated splitting makes many pieces (force-split-all made 9421);
  ensure the relink re-merges same-axon pieces (no forbidden pair fires within one axon) so
  fragmentation doesn't explode. Verify #instances stays ~tube-bb (2625), not ~force-split (3600+).
- **Runtime:** force-split-all took ~9 min full-volume; keep within SLURM time. Use cached sections.
- Question for reviewer: is simple forbidden-edge deletion sufficient for v0, or should v0 ship the
  constrained union-find directly?

## Changes Since Previous Plan Version
Initial plan.
